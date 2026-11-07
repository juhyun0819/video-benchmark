import { readFileSync, statSync } from "node:fs";
import { z } from "zod";
import { AnnotationSchema, BlockCandidateSchema, type Annotation, type BlockCandidate } from "@vb/core";
import { keywordMatches, type Categories, type Category } from "../categories.js";
import { bump, type IngestReport } from "../jsonl.js";

/** Files larger than this are refused before reading (exit 1, naming the file). */
export const MAX_JSON_BYTES = 512 * 1024 * 1024;

/** Used when ego4d.json carries no positive video_metadata.fps for a video. */
export const DEFAULT_FPS = 30;

/**
 * ego4d.json (https://ego4d-data.org/docs/data/metadata/): only the fields we rely on; everything else
 * (split_*, video_components, clips, ...) passes through.
 */
export const Ego4dMetaSchema = z
  .object({
    videos: z.array(
      z
        .object({
          video_uid: z.string(),
          scenarios: z.array(z.string()),
          duration_sec: z.number().optional(),
          video_metadata: z.object({ fps: z.number().optional() }).passthrough().optional(),
          redacted_intervals: z
            .array(z.object({ start_sec: z.number(), end_sec: z.number() }).passthrough())
            .nullable()
            .optional(),
        })
        .passthrough(),
    ),
  })
  .passthrough();
export type Ego4dMeta = z.infer<typeof Ego4dMetaSchema>;

export const Ego4dNarrationSchema = z.object({ timestamp_sec: z.number(), narration_text: z.string() }).passthrough();
export const Ego4dSummarySchema = z
  .object({ start_sec: z.number(), end_sec: z.number(), summary_text: z.string() })
  .passthrough();
export type Ego4dNarration = z.infer<typeof Ego4dNarrationSchema>;
export type Ego4dSummary = z.infer<typeof Ego4dSummarySchema>;

/** narration.json: video_uid -> annotation passes. Only narration_pass_1 is used. */
export const Ego4dNarrationsSchema = z.record(
  z.string(),
  z
    .object({
      narration_pass_1: z
        .object({
          narrations: z.array(Ego4dNarrationSchema),
          summaries: z.array(Ego4dSummarySchema),
        })
        .passthrough()
        .optional(),
    })
    .passthrough(),
);
export type Ego4dNarrations = z.infer<typeof Ego4dNarrationsSchema>;

export function parseWith<T>(schema: z.ZodType<T, z.ZodTypeDef, unknown>, json: unknown, file: string): T {
  const parsed = schema.safeParse(json);
  if (!parsed.success) {
    const issue = parsed.error.issues
      .map((iss) => `${iss.path.join(".") || "(root)"}: ${iss.message}`)
      .join("; ");
    throw new Error(`${file}: ${issue}`);
  }
  return parsed.data;
}

export const parseEgo4dMeta = (json: unknown, file: string): Ego4dMeta => parseWith(Ego4dMetaSchema, json, file);
export const parseEgo4dNarrations = (json: unknown, file: string): Ego4dNarrations =>
  parseWith(Ego4dNarrationsSchema, json, file);

/** Read and JSON.parse a file, refusing anything over `maxBytes`. Errors name the file. */
export function readJsonFile(path: string, maxBytes: number = MAX_JSON_BYTES): unknown {
  const size = statSync(path).size;
  if (size > maxBytes) {
    throw new Error(`${path}: file is too large (${size} bytes; the limit is 512 MB)`);
  }
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch (e) {
    throw new Error(`${path}: invalid JSON: ${(e as Error).message}`);
  }
}

const cmpStr = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);

/** Scenario -> number of videos carrying it, sorted by count desc then name. */
export function listScenarios(meta: Ego4dMeta): Array<{ scenario: string; videos: number }> {
  const counts = new Map<string, number>();
  for (const v of meta.videos) {
    for (const s of new Set(v.scenarios)) counts.set(s, (counts.get(s) ?? 0) + 1);
  }
  return [...counts]
    .map(([scenario, videos]) => ({ scenario, videos }))
    .sort((a, b) => b.videos - a.videos || cmpStr(a.scenario, b.scenario));
}

/**
 * Build Ego4D event candidates and per-video annotations.
 * - A video matches a category when its scenarios intersect the category's scenarios.
 * - Each narration_pass_1 summary of a matching video becomes one candidate in the FIRST category
 *   (config order) whose keywords are empty or appear in summary_text (case-insensitive).
 * - Annotations (all pass-1 narrations as points, all valid summaries as intervals) are produced only
 *   for videos that yield at least one candidate; remap filters by span later.
 * - Candidate media fps is the video's video_metadata.fps (DEFAULT_FPS when absent). A candidate end past
 *   duration_sec is clipped to it; a candidate overlapping a redacted interval is dropped.
 * Skip reasons: no_narration (a category-matching video without narration_pass_1), end_before_start,
 * beyond_duration (starts at or after duration_sec), redacted and duplicate_span (candidate summaries),
 * invalid_time (narrations of candidate videos).
 */
export function buildEgo4dCatalog(
  meta: Ego4dMeta,
  narrations: Ego4dNarrations,
  categories: Categories | Category[],
  report: IngestReport,
): { candidates: BlockCandidate[]; annotations: Map<string, Annotation[]> } {
  // Both datasets have a "Cooking" scenario, so a category only applies to its own dataset.
  const cats = (Array.isArray(categories) ? categories : categories.categories).filter((c) => c.dataset === "ego4d");
  const candidates: BlockCandidate[] = [];
  const annotations = new Map<string, Annotation[]>();

  const videos = new Map(meta.videos.map((v) => [v.video_uid, v] as const));
  for (const uid of [...videos.keys()].sort(cmpStr)) {
    const video = videos.get(uid);
    const scenarios = new Set(video?.scenarios ?? []);
    const matched = cats.filter((c) => c.scenarios.some((s) => scenarios.has(s)));
    if (matched.length === 0) continue;

    const rawFps = video?.video_metadata?.fps;
    const fps = rawFps !== undefined && Number.isFinite(rawFps) && rawFps > 0 ? rawFps : DEFAULT_FPS;
    const duration = video?.duration_sec;
    const hasDuration = duration !== undefined && Number.isFinite(duration) && duration > 0;
    const redacted = video?.redacted_intervals ?? [];

    const pass1 = narrations[uid]?.narration_pass_1;
    if (!pass1) {
      bump(report.skipped, "no_narration");
      continue;
    }

    const seen = new Set<string>();
    let built = 0;
    pass1.summaries.forEach((s) => {
      const cat = matched.find((c) => keywordMatches(c, s.summary_text));
      if (!cat) return;
      if (!Number.isFinite(s.start_sec) || !Number.isFinite(s.end_sec)) {
        bump(report.skipped, "invalid_time");
        return;
      }
      if (s.end_sec <= s.start_sec) {
        bump(report.skipped, "end_before_start");
        return;
      }
      if (hasDuration && s.start_sec >= duration) {
        bump(report.skipped, "beyond_duration");
        return;
      }
      const start = s.start_sec;
      const end = hasDuration ? Math.min(s.end_sec, duration) : s.end_sec;
      if (redacted.some((r) => start < r.end_sec && end > r.start_sec)) {
        bump(report.skipped, "redacted");
        return;
      }
      const id = `ego4d:${uid}:${start.toFixed(3)}-${end.toFixed(3)}`;
      if (seen.has(id)) {
        bump(report.skipped, "duplicate_span");
        return;
      }
      seen.add(id);
      candidates.push(
        BlockCandidateSchema.parse({
          id,
          role: "event",
          category: cat.id,
          source: { dataset: "ego4d", video_id: uid, media: { kind: "video", fps } },
          src_start: start,
          src_end: end,
          summary: s.summary_text,
        }),
      );
      built++;
    });
    if (built === 0) continue;

    const anns: Annotation[] = [];
    let nNarr = 0;
    pass1.narrations.forEach((n, i) => {
      if (!Number.isFinite(n.timestamp_sec)) {
        bump(report.skipped, "invalid_time");
        return;
      }
      anns.push(
        AnnotationSchema.parse({
          id: `ego4d:${uid}:n:${i}`,
          track: "captions",
          level: "narration",
          time: { kind: "point", t: n.timestamp_sec },
          text: n.narration_text,
          original: n,
        }),
      );
      nNarr++;
    });
    let nSum = 0;
    pass1.summaries.forEach((s, i) => {
      // Invalid spans of matching summaries were already counted above; unrelated ones are dropped silently.
      if (!Number.isFinite(s.start_sec) || !Number.isFinite(s.end_sec) || s.end_sec <= s.start_sec) return;
      anns.push(
        AnnotationSchema.parse({
          id: `ego4d:${uid}:s:${i}`,
          track: "captions",
          level: "summary",
          time: { kind: "interval", start: s.start_sec, end: s.end_sec },
          text: s.summary_text,
          original: s,
        }),
      );
      nSum++;
    });
    annotations.set(uid, anns);
    bump(report.ingested, "ego4d:narrations", nNarr);
    bump(report.ingested, "ego4d:summaries", nSum);
  }

  bump(report.ingested, "ego4d:candidates", candidates.length);
  return { candidates, annotations };
}
