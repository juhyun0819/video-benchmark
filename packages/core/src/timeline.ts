import type { Catalog } from "./catalog.js";
import { CompositionError } from "./errors.js";
import type { BlockCandidate, Composition, SourceRef } from "./schema.js";

export type TimelineBlock = {
  block_id: string;
  role: "base" | "event";
  category: string;
  summary: string;
  source: SourceRef;
  src_start: number;
  src_end: number;
  composed_start: number;
  composed_end: number;
  slot?: number;
  ref_id?: string;
  base_index?: number;
};

/**
 * Order: slot0 events, base0, slot1 events, base1, ... slotN events.
 * Composed times are contiguous starting at 0 (unrounded).
 */
export function resolveTimeline(comp: Composition, catalog: Catalog): TimelineBlock[] {
  const baseDay = catalog.getBaseDay(comp.base_day);
  if (!baseDay) {
    throw new CompositionError("BASE_DAY_NOT_FOUND", `Base day ${comp.base_day} not found`, {
      day: comp.base_day,
    });
  }

  // Resolve every event first so all missing ids are reported at once.
  const resolved = new Map<string, BlockCandidate>();
  const missing: string[] = [];
  for (const slot of comp.slots) {
    for (const ref of slot) {
      if (resolved.has(ref.candidate_id) || missing.includes(ref.candidate_id)) continue;
      const cand = catalog.getCandidate(ref.candidate_id);
      if (cand) resolved.set(ref.candidate_id, cand);
      else missing.push(ref.candidate_id);
    }
  }
  if (missing.length > 0) {
    throw new CompositionError(
      "MISSING_CANDIDATES",
      `Composition references ${missing.length} unknown candidate(s): ${missing.join(", ")}`,
      { ids: missing },
    );
  }

  const blocks: TimelineBlock[] = [];
  let cursor = 0;
  const push = (
    block_id: string,
    cand: BlockCandidate,
    extra: Pick<TimelineBlock, "slot" | "ref_id" | "base_index">,
  ) => {
    const duration = cand.src_end - cand.src_start;
    blocks.push({
      block_id,
      role: extra.base_index !== undefined ? "base" : "event",
      category: cand.category,
      summary: cand.summary,
      source: cand.source,
      src_start: cand.src_start,
      src_end: cand.src_end,
      composed_start: cursor,
      composed_end: cursor + duration,
      ...extra,
    });
    cursor += duration;
  };

  const emitSlot = (slotIndex: number) => {
    for (const ref of comp.slots[slotIndex] ?? []) {
      push(`evt:${ref.ref_id}`, resolved.get(ref.candidate_id)!, { slot: slotIndex, ref_id: ref.ref_id });
    }
  };

  for (let i = 0; i < baseDay.blocks.length; i++) {
    emitSlot(i);
    push(`base:${baseDay.day}:${i}`, baseDay.blocks[i]!, { base_index: i });
  }
  emitSlot(baseDay.blocks.length);
  return blocks;
}
