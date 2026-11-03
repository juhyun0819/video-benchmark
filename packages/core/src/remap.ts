import type { Annotation, ManifestCaption, TimeSpan } from "./schema.js";
import type { TimelineBlock } from "./timeline.js";

/**
 * Maps source annotations into composed time for one timeline block.
 * Values are unrounded; Task 6 rounds. Output follows annotation input order.
 */
export function remapAnnotations(block: TimelineBlock, annotations: Annotation[]): ManifestCaption[] {
  const out: ManifestCaption[] = [];
  const shift = (t: number) => block.composed_start + (t - block.src_start);

  for (const a of annotations) {
    let time: TimeSpan;
    let clipped: { start?: true; end?: true } | undefined;

    const isPoint = a.time.kind === "point" || a.time.start === a.time.end;
    if (a.time.kind === "point") {
      if (!(a.time.t >= block.src_start && a.time.t < block.src_end)) continue;
      time = { kind: "point", t: shift(a.time.t) };
    } else if (isPoint) {
      const t = a.time.start;
      if (!(t >= block.src_start && t < block.src_end)) continue;
      time = { kind: "interval", start: shift(t), end: shift(t) };
    } else {
      const { start, end } = a.time;
      if (!(start < block.src_end && end > block.src_start)) continue;
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
