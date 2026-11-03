import type { Annotation, BlockCandidate, SourceRef } from "./schema.js";

const MIN_BLOCK_SECONDS = 60;
const UNLABELED = "(unlabeled)";

export interface TileInput {
  source: SourceRef;
  duration: number;
  l4: Annotation[];
}

/**
 * Split a recording into base blocks at L4 activity boundaries (spec 6.2).
 * Blocks shorter than 60 s merge into their predecessor (or successor, for
 * the first block); a lone block is never merged.
 */
export function tileRecording(input: TileInput): BlockCandidate[] {
  const { source, duration } = input;

  const intervals: { start: number; end: number; text: string }[] = [];
  for (const a of input.l4) {
    if (a.time.kind === "interval") {
      intervals.push({ start: a.time.start, end: a.time.end, text: a.text });
    }
  }

  const inner = intervals.map((i) => i.start).filter((s) => s > 0 && s < duration);
  const boundaries = [...new Set([0, duration, ...inner])].sort((a, b) => a - b);

  let spans: { start: number; end: number }[] = [];
  for (let i = 0; i + 1 < boundaries.length; i++) {
    const start = boundaries[i];
    const end = boundaries[i + 1];
    if (start !== undefined && end !== undefined) spans.push({ start, end });
  }
  if (spans.length === 0) spans = [{ start: 0, end: duration }];

  for (;;) {
    if (spans.length < 2) break;
    const idx = spans.findIndex((s) => s.end - s.start < MIN_BLOCK_SECONDS);
    if (idx === -1) break;
    const cur = spans[idx];
    if (!cur) break;
    if (idx > 0) {
      const prev = spans[idx - 1];
      if (!prev) break;
      spans.splice(idx - 1, 2, { start: prev.start, end: cur.end });
    } else {
      const next = spans[1];
      if (!next) break;
      spans.splice(0, 2, { start: cur.start, end: next.end });
    }
  }

  return spans.map(({ start, end }) => {
    const hit = intervals.find((i) => i.start <= start && start < i.end);
    return {
      id: `${source.dataset}:${source.video_id}:${start.toFixed(3)}-${end.toFixed(3)}`,
      role: "base",
      category: "base",
      source,
      src_start: start,
      src_end: end,
      summary: hit ? hit.text : UNLABELED,
    };
  });
}
