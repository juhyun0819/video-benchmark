import type { Annotation, Clip } from "@vb/core";
import type { BlockCandidate } from "./blocks.js";
import type { Category } from "./categories.js";
import { bump, type IngestReport } from "./jsonl.js";

/** Convert an adapter candidate to a catalog clip using the category's `clip` mapping. */
export function candidateToClip(c: BlockCandidate, clip: NonNullable<Category["clip"]>): Clip {
  const group = `${c.source.dataset}:${c.source.video_id}`;
  return {
    clip_id: c.id,
    category: clip.category,
    setting: clip.setting,
    node: clip.node,
    activity: c.summary,
    time_of_day: "unknown",
    tags: [...clip.tags, c.category],
    source: c.source,
    source_start_s: c.src_start,
    source_end_s: c.src_end,
    day_group: group,
    person_source: group,
    fingerprint: null,
    extractor_version: `${c.source.dataset}-ingest.v1`,
  };
}

/**
 * Convert candidates to clips. A candidate whose category has no `clip` mapping is dropped and counted as
 * skipped.no_clip_mapping. Annotations are kept only for videos that still have at least one clip.
 */
export function candidatesToClips(
  candidates: BlockCandidate[],
  annotations: Map<string, Annotation[]>,
  categories: Category[],
  report: IngestReport,
): { clips: Clip[]; annotations: Map<string, Annotation[]> } {
  const byId = new Map(categories.map((c) => [c.id, c]));
  const clips: Clip[] = [];
  const kept = new Map<string, Annotation[]>();
  for (const c of candidates) {
    const mapping = byId.get(c.category)?.clip;
    if (!mapping) {
      bump(report.skipped, "no_clip_mapping");
      continue;
    }
    clips.push(candidateToClip(c, mapping));
    const anns = annotations.get(c.source.video_id);
    if (anns) kept.set(c.source.video_id, anns);
  }
  return { clips, annotations: kept };
}
