import { parse } from "csv-parse/sync";
import { z } from "zod";
import { AnnotationSchema, BlockCandidateSchema, type Annotation, type BlockCandidate } from "@vb/core";
import { bump, type IngestReport } from "../jsonl.js";

/** Columns of high-level/activities/PXX_recipe_timestamps.csv (spec section 11, HD-EPIC findings). */
export const HIGH_LEVEL_COLUMNS = {
  video_id: "video_id",
  recipe_id: "recipe_id",
  label: "high_level_activity_label",
  start: "start_time",
  end: "end_time",
} as const;

export const ERRATUM_COLUMN = "erratum_narration_id";

/**
 * One row of a PXX_recipe_timestamps.csv. `end` stays the literal "end" when the CSV says so
 * (resolved in buildHdepicCatalog); unparseable numbers are NaN (skipped as invalid_time).
 * `recipe_id` is the raw cell: one id ("R01") or a comma list ("R08, R07"); absent when empty.
 */
export type HighLevelSegment = {
  video_id: string;
  start: number;
  end: number | "end";
  label: string;
  recipe_id?: string;
};

const NUMBER = /^-?\d+(\.\d+)?$/;
const parseSeconds = (s: string): number => (NUMBER.test(s.trim()) ? Number(s) : Number.NaN);

function readCsv(text: string, file: string, required: string[]): Record<string, string>[] {
  return parse(text, {
    columns: (header: string[]) => {
      for (const col of required) {
        if (!header.includes(col)) throw new Error(`${file}: missing column "${col}"`);
      }
      return header;
    },
    skip_empty_lines: true,
    bom: true,
  }) as Record<string, string>[];
}

export function parseHighLevel(csvText: string, file: string): HighLevelSegment[] {
  const C = HIGH_LEVEL_COLUMNS;
  return readCsv(csvText, file, Object.values(C)).map((r) => {
    const end = (r[C.end] ?? "").trim();
    const recipe = (r[C.recipe_id] ?? "").trim();
    const seg: HighLevelSegment = {
      video_id: (r[C.video_id] ?? "").trim(),
      start: parseSeconds(r[C.start] ?? ""),
      end: end === "end" ? "end" : parseSeconds(end),
      label: (r[C.label] ?? "").trim(),
    };
    if (recipe) seg.recipe_id = recipe;
    return seg;
  });
}

/** HD_EPIC_Narrations_erratum.csv -> the set of excluded unique_narration_ids. */
export function parseErratumCsv(csvText: string, file: string): Set<string> {
  const ids = readCsv(csvText, file, [ERRATUM_COLUMN]).map((r) => (r[ERRATUM_COLUMN] ?? "").trim());
  return new Set(ids.filter(Boolean));
}

/** Row of the converted HD_EPIC_Narrations.jsonl (scripts/convert_hdepic.py). Extra columns are kept in `original`. */
export const HdNarrationSchema = z
  .object({
    unique_narration_id: z.string(),
    video_id: z.string(),
    narration: z.string(),
    start_timestamp: z.number(),
    end_timestamp: z.number(),
  })
  .passthrough();
export type HdNarration = z.infer<typeof HdNarrationSchema>;

const participantOf = (videoId: string): string => videoId.split("-")[0] ?? videoId;
const byNumber = (a: number, b: number): number => a - b;
const cmpStr = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);

function recipeNames(seg: HighLevelSegment, recipes: Record<string, { name: string }>): string[] {
  if (!seg.recipe_id) return [];
  const participant = participantOf(seg.video_id);
  const names: string[] = [];
  for (const id of seg.recipe_id.split(",").map((s) => s.trim()).filter(Boolean)) {
    const name = recipes[`${participant}_${id}`]?.name;
    if (name) names.push(name);
  }
  return names;
}

/**
 * Build the HD-EPIC cooking candidates (one per high-level segment) and per-video narration annotations.
 * - `end: "end"` -> next segment's start (same video, sorted by start); the last one -> max narration
 *   end_timestamp of the video; no narrations -> skipped as open_end.
 * - Narrations listed in `erratum` are dropped (skipped: erratum).
 * - Narrations are attached by video_id regardless of the candidate span (remap filters later), but only
 *   for videos that yield at least one candidate.
 * - Segments resolving to the same video and span (e.g. one activity listed per recipe) become one candidate
 *   with the recipe names merged; the extra rows count as skipped: duplicate_span.
 * Skip reasons: invalid_time, open_end, end_before_start (segments and narrations), erratum, duplicate_span.
 */
export function buildHdepicCatalog(
  segments: HighLevelSegment[],
  narrations: HdNarration[],
  recipes: Record<string, { name: string }>,
  report: IngestReport,
  erratum: ReadonlySet<string> = new Set(),
): { candidates: BlockCandidate[]; annotations: Map<string, Annotation[]> } {
  // Narrations: drop erratum / bad ranges, group by video.
  const narrByVideo = new Map<string, HdNarration[]>();
  for (const n of narrations) {
    if (erratum.has(n.unique_narration_id)) {
      bump(report.skipped, "erratum");
      continue;
    }
    if (!Number.isFinite(n.start_timestamp) || !Number.isFinite(n.end_timestamp)) {
      bump(report.skipped, "invalid_time");
      continue;
    }
    if (n.end_timestamp < n.start_timestamp) {
      bump(report.skipped, "end_before_start");
      continue;
    }
    const list = narrByVideo.get(n.video_id) ?? [];
    list.push(n);
    narrByVideo.set(n.video_id, list);
  }

  // Segments: group by video, drop unparseable starts, sort by start (stable).
  const segsByVideo = new Map<string, HighLevelSegment[]>();
  for (const s of segments) {
    if (!Number.isFinite(s.start) || (s.end !== "end" && !Number.isFinite(s.end))) {
      bump(report.skipped, "invalid_time");
      continue;
    }
    const list = segsByVideo.get(s.video_id) ?? [];
    list.push(s);
    segsByVideo.set(s.video_id, list);
  }

  const candidates: BlockCandidate[] = [];
  const annotations = new Map<string, Annotation[]>();

  for (const videoId of [...segsByVideo.keys()].sort(cmpStr)) {
    const segs = (segsByVideo.get(videoId) ?? []).slice().sort((a, b) => byNumber(a.start, b.start));
    const videoNarrs = narrByVideo.get(videoId) ?? [];
    const maxNarrEnd = videoNarrs.length > 0 ? Math.max(...videoNarrs.map((n) => n.end_timestamp)) : undefined;

    const pending = new Map<string, { id: string; seg: HighLevelSegment; end: number; names: string[] }>();
    segs.forEach((seg, i) => {
      let end: number;
      if (seg.end === "end") {
        const next = segs[i + 1];
        if (next) end = next.start;
        else if (maxNarrEnd !== undefined) end = maxNarrEnd;
        else {
          bump(report.skipped, "open_end");
          return;
        }
      } else {
        end = seg.end;
      }
      if (seg.start >= end) {
        bump(report.skipped, "end_before_start");
        return;
      }
      const id = `hdepic:${videoId}:${seg.start.toFixed(3)}-${end.toFixed(3)}`;
      const names = recipeNames(seg, recipes);
      const prior = pending.get(id);
      if (prior) {
        // Same activity span listed once per recipe (or repeated verbatim): one candidate, names merged.
        for (const n of names) if (!prior.names.includes(n)) prior.names.push(n);
        bump(report.skipped, "duplicate_span");
        return;
      }
      pending.set(id, { id, seg, end, names: [...names] });
    });

    for (const { id, seg, end, names } of pending.values()) {
      const summary =
        `${participantOf(videoId)} · ${seg.label || "(unlabeled)"}` +
        (names.length > 0 ? ` (${names.join(", ")})` : "");
      candidates.push(
        BlockCandidateSchema.parse({
          id,
          role: "event",
          category: "cooking",
          source: { dataset: "hdepic", video_id: videoId, media: { kind: "video", fps: 30 } },
          src_start: seg.start,
          src_end: end,
          summary,
        }),
      );
    }
    const built = pending.size;

    if (built > 0 && videoNarrs.length > 0) {
      const anns = videoNarrs
        .slice()
        .sort((a, b) => byNumber(a.start_timestamp, b.start_timestamp) || cmpStr(a.unique_narration_id, b.unique_narration_id))
        .map((n) =>
          AnnotationSchema.parse({
            id: `hdepic:${n.unique_narration_id}`,
            track: "captions",
            level: "narration",
            time: { kind: "interval", start: n.start_timestamp, end: n.end_timestamp },
            text: n.narration,
            original: n,
          }),
        );
      annotations.set(videoId, anns);
      bump(report.ingested, "hdepic:narrations", anns.length);
    }
  }

  bump(report.ingested, "hdepic:candidates", candidates.length);
  return { candidates, annotations };
}
