import { z } from "zod";
import { AnnotationSchema, BlockCandidateSchema, type Annotation, type BlockCandidate } from "@vb/core";
import { keywordMatches, type Categories, type Category } from "../categories.js";
import { parseWith } from "../ego4d/adapter.js";
import { bump, type IngestReport } from "../jsonl.js";
import { DEFAULT_MIN_EVENT_SEC } from "../time.js";

/** takes.json has no fps field; (timesync_end_idx - timesync_start_idx) / duration_sec is 30 for every take. */
export const EGOEXO_FPS = 30;

/** takes.json: only the fields we rely on; everything else passes through. */
export const EgoExoTakeSchema = z
  .object({
    take_uid: z.string(),
    take_name: z.string(),
    parent_task_name: z.string(),
    task_name: z.string(),
    duration_sec: z.number(),
    task_start_sec: z.number().nullable().optional(),
    task_end_sec: z.number().nullable().optional(),
    root_dir: z.string(),
    frame_aligned_videos: z.record(z.string(), z.unknown()),
  })
  .passthrough();
export type EgoExoTake = z.infer<typeof EgoExoTakeSchema>;

export const EgoExoTakesSchema = z.array(EgoExoTakeSchema);

const AriaStreamsSchema = z
  .object({ rgb: z.object({ relative_path: z.string() }).passthrough() })
  .passthrough();

export const EgoExoDescriptionSchema = z.object({ text: z.string(), timestamp: z.number() }).passthrough();
const EgoExoAtomicRunSchema = z
  .object({ rejected: z.boolean(), descriptions: z.array(EgoExoDescriptionSchema) })
  .passthrough();

/** atomic_descriptions_<split>.json: take_uid -> annotator runs. */
export const EgoExoAtomicFileSchema = z
  .object({ annotations: z.record(z.string(), z.array(EgoExoAtomicRunSchema)) })
  .passthrough();
export type EgoExoAtomic = z.infer<typeof EgoExoAtomicFileSchema>["annotations"];

export const EgoExoKeystepSchema = z
  .object({ start_time: z.number(), end_time: z.number(), step_name: z.string() })
  .passthrough();
export type EgoExoKeystep = z.infer<typeof EgoExoKeystepSchema>;

/** keystep_<split>.json: take_uid -> take entry with its segments. */
export const EgoExoKeystepFileSchema = z
  .object({
    annotations: z.record(z.string(), z.object({ segments: z.array(EgoExoKeystepSchema) }).passthrough()),
  })
  .passthrough();
export type EgoExoKeysteps = z.infer<typeof EgoExoKeystepFileSchema>["annotations"];

export const parseEgoExoTakes = (json: unknown, file: string): EgoExoTake[] => parseWith(EgoExoTakesSchema, json, file);
export const parseEgoExoAtomic = (json: unknown, file: string): EgoExoAtomic =>
  parseWith(EgoExoAtomicFileSchema, json, file).annotations;
export const parseEgoExoKeysteps = (json: unknown, file: string): EgoExoKeysteps =>
  parseWith(EgoExoKeystepFileSchema, json, file).annotations;

/** Path of the take's ego rgb video, relative to the Ego-Exo4D download root. */
function egoVideoPath(take: EgoExoTake): string | undefined {
  const cams = Object.keys(take.frame_aligned_videos).filter((k) => k.startsWith("aria")).sort();
  for (const cam of cams) {
    const streams = AriaStreamsSchema.safeParse(take.frame_aligned_videos[cam]);
    if (streams.success) return `${take.root_dir}/${streams.data.rgb.relative_path}`;
  }
  return undefined;
}

/** The task span clipped to the video, or the whole take when the span is missing or unusable. */
function takeSpan(take: EgoExoTake): { start: number; end: number; whole: boolean } {
  const { task_start_sec: start, task_end_sec: end, duration_sec: duration } = take;
  if (start == null || end == null || start < 0 || end <= start || start >= duration) {
    return { start: 0, end: duration, whole: true };
  }
  return { start, end: Math.min(end, duration), whole: false };
}

const SUMMARY_STEPS = 3;

/** takes.json has no summary text: many takes share a task name, so the steps (or take name) tell them apart. */
function takeSummary(take: EgoExoTake, segments: EgoExoKeystep[]): string {
  const ordered = [...segments].sort((a, b) => a.start_time - b.start_time);
  const steps = [...new Set(ordered.map((s) => s.step_name))].slice(0, SUMMARY_STEPS);
  return steps.length > 0 ? `${take.task_name}: ${steps.join(", ")}` : `${take.task_name} (${take.take_name})`;
}

/**
 * Build Ego-Exo4D event candidates and per-take annotations.
 * - One take is one candidate, in the FIRST egoexo4d category (config order) whose scenarios contain the
 *   take's parent_task_name and whose keywords are empty or appear in task_name (case-insensitive).
 * - Annotations are the first accepted atomic run (points; a second annotator would duplicate the
 *   timeline) and the keystep segments (intervals), copied verbatim; remap filters by span later.
 * Skip reasons: invalid_time (duration_sec not positive), too_short (span under minSec), no_ego_video,
 * no_atomic (no accepted run with descriptions), end_before_start (keystep segments).
 */
export function buildEgoExoCatalog(
  takes: EgoExoTake[],
  atomic: EgoExoAtomic,
  keysteps: EgoExoKeysteps,
  categories: Categories | Category[],
  report: IngestReport,
  minSec: number = DEFAULT_MIN_EVENT_SEC,
): { candidates: BlockCandidate[]; annotations: Map<string, Annotation[]> } {
  const cats = (Array.isArray(categories) ? categories : categories.categories).filter(
    (c) => c.dataset === "egoexo4d",
  );
  const candidates: BlockCandidate[] = [];
  const annotations = new Map<string, Annotation[]>();

  for (const take of [...takes].sort((a, b) => (a.take_uid < b.take_uid ? -1 : a.take_uid > b.take_uid ? 1 : 0))) {
    const uid = take.take_uid;
    const cat = cats.find((c) => c.scenarios.includes(take.parent_task_name) && keywordMatches(c, take.task_name));
    if (!cat) continue;

    if (take.duration_sec <= 0) {
      bump(report.skipped, "invalid_time");
      continue;
    }
    const { start, end, whole } = takeSpan(take);
    if (end - start < minSec) {
      bump(report.skipped, "too_short");
      continue;
    }
    const path = egoVideoPath(take);
    if (path === undefined) {
      bump(report.skipped, "no_ego_video");
      continue;
    }
    const run = atomic[uid]?.find((r) => !r.rejected && r.descriptions.length > 0);
    if (!run) {
      bump(report.skipped, "no_atomic");
      continue;
    }

    const anns: Annotation[] = run.descriptions.map((d, i) =>
      AnnotationSchema.parse({
        id: `egoexo4d:${uid}:a:${i}`,
        track: "captions",
        level: "atomic",
        time: { kind: "point", t: d.timestamp },
        text: d.text,
        original: d,
      }),
    );
    const segments: EgoExoKeystep[] = [];
    keysteps[uid]?.segments.forEach((s, i) => {
      if (s.end_time <= s.start_time) {
        bump(report.skipped, "end_before_start");
        return;
      }
      segments.push(s);
      anns.push(
        AnnotationSchema.parse({
          id: `egoexo4d:${uid}:k:${i}`,
          track: "captions",
          level: "keystep",
          time: { kind: "interval", start: s.start_time, end: s.end_time },
          text: s.step_name,
          original: s,
        }),
      );
    });

    candidates.push(
      BlockCandidateSchema.parse({
        id: `egoexo4d:${uid}:${start.toFixed(3)}-${end.toFixed(3)}`,
        role: "event",
        category: cat.id,
        source: { dataset: "egoexo4d", video_id: uid, media: { kind: "video", fps: EGOEXO_FPS, path } },
        src_start: start,
        src_end: end,
        summary: takeSummary(take, segments),
      }),
    );
    annotations.set(uid, anns);
    if (whole) bump(report.ingested, "egoexo4d:whole_take_span");
    bump(report.ingested, "egoexo4d:atomic", run.descriptions.length);
    bump(report.ingested, "egoexo4d:keysteps", segments.length);
  }

  bump(report.ingested, "egoexo4d:candidates", candidates.length);
  return { candidates, annotations };
}
