import type { Annotation, ManifestCaption, SourceRef, TimeSpan } from "./schema.js";

export type PlacedSpan = {
  block_id: string;
  source: SourceRef;
  src_start: number;
  src_end: number;
  composed_start: number;
};

/**
 * True when `time` falls in the half-open source span [start, end): a point (or zero-length interval)
 * counts when start <= t < end, any other interval when it overlaps the span.
 */
export function overlapsSpan(time: TimeSpan, start: number, end: number): boolean {
  if (time.kind === "point") return time.t >= start && time.t < end;
  if (time.start === time.end) return time.start >= start && time.start < end;
  return time.start < end && time.end > start;
}

/**
 * Maps source annotations into composed time for one placed span.
 * Values are unrounded; Task 6 rounds. Output follows annotation input order.
 */
export function remapAnnotations(block: PlacedSpan, annotations: Annotation[]): ManifestCaption[] {
  const out: ManifestCaption[] = [];
  const shift = (t: number) => block.composed_start + (t - block.src_start);

  for (const a of annotations) {
    let time: TimeSpan;
    let clipped: { start?: true; end?: true } | undefined;

    const isPoint = a.time.kind === "point" || a.time.start === a.time.end;
    if (a.time.kind === "point") {
      if (!overlapsSpan(a.time, block.src_start, block.src_end)) continue;
      time = { kind: "point", t: shift(a.time.t) };
    } else if (isPoint) {
      const t = a.time.start;
      if (!overlapsSpan(a.time, block.src_start, block.src_end)) continue;
      time = { kind: "interval", start: shift(t), end: shift(t) };
    } else {
      const { start, end } = a.time;
      if (!overlapsSpan(a.time, block.src_start, block.src_end)) continue;
      const cs = Math.max(start, block.src_start);
      const ce = Math.min(end, block.src_end);
      time = { kind: "interval", start: shift(cs), end: shift(ce) };
      if (start < block.src_start) clipped = { start: true };
      if (end > block.src_end) clipped = { ...clipped, end: true };
    }

    const orig_time_string: { start?: string; end?: string } = {};
    if (typeof a.original["start"] === "string") orig_time_string.start = a.original["start"];
    if (typeof a.original["end"] === "string") orig_time_string.end = a.original["end"];

    out.push({
      id: `${block.block_id}:${a.id}`,
      block_id: block.block_id,
      level: a.level,
      time,
      text: a.text,
      ...(clipped ? { clipped } : {}),
      provenance: {
        dataset: block.source.dataset,
        video_id: block.source.video_id,
        orig_id: a.id,
        src_time: a.time,
        ...(Object.keys(orig_time_string).length > 0 ? { orig_time_string } : {}),
        original: a.original,
      },
    });
  }
  return out;
}
