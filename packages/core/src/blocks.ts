import type { ClipCatalog } from "./clip.js";
import { parseClock } from "./clock.js";
import { TimelineError } from "./errors.js";
import type { PlacedClip } from "./layout.js";
import type { ScheduleDay, Slot } from "./schedule/types.js";
import type { Timeline } from "./timeline.js";

/**
 * One clip of a block plan (B2): `slot_id` is a slot id or a transition id `d<N>-t<k>`;
 * the source span is a part of the catalog clip's span.
 */
export type BlockPlanEntry = { slot_id: string; clip_id: string; source_start_s: number; source_end_s: number };

/**
 * A clip placed by layoutBlocks. `source_start_s`/`source_end_s` are the span actually used,
 * cut at the block end when `trimmed`.
 */
export type BlockPlacement = PlacedClip & { source_start_s: number; source_end_s: number; trimmed: boolean };

/** A schedule transition as a block: span in seconds since the day's 00:00. */
export type TransitionBlock = { transition_id: string; from_slot: string; to_slot: string; span: [number, number] };

export const transitionId = (dayIndex: number, k: number): string => `d${dayIndex}-t${k}`;

const round3 = (x: number) => Math.round(x * 1000) / 1000;

/**
 * Seconds of travel right before `next`, in the gap after `slot` (G23, CR6.3): the day's transition
 * from `slot` to `next`, cut to the gap between their windows; 0 when there is no gap or no record.
 */
export function travelBefore(day: ScheduleDay, slot: Slot, next: Slot | undefined): number {
  const we = parseClock(slot.window[1]) * 60;
  const gapEnd = next === undefined ? Infinity : parseClock(next.window[0]) * 60;
  if (next === undefined || gapEnd <= we) return 0;
  const moved = (day.transitions.find((x) => x.from_slot === slot.slot_id && x.to_slot === next.slot_id)?.minutes ?? 0) * 60;
  return Math.min(moved, gapEnd - we);
}

/** The day's slots in window-start order, as the manifest walks them. */
export const slotsByStart = (day: ScheduleDay): Slot[] =>
  [...day.slots].sort((a, b) => parseClock(a.window[0]) - parseClock(b.window[0]));

/**
 * Every transition of the day as a block `d<N>-t<k>` (k: 1-based index into `day.transitions`).
 * Its span is the manifest's "transition" no-video span: [to_slot start − travel, to_slot start).
 * A transition that is not the first record between two window-adjacent slots gets no travel time,
 * so its span is empty, as in the manifest.
 */
export function transitionBlocks(day: ScheduleDay): TransitionBlock[] {
  const sorted = slotsByStart(day);
  const out: TransitionBlock[] = [];
  day.transitions.forEach((tr, i) => {
    const j = sorted.findIndex((s) => s.slot_id === tr.to_slot);
    if (j < 0) return;
    const to = sorted[j]!;
    const prev = j > 0 ? sorted[j - 1] : undefined;
    const first = day.transitions.findIndex((x) => x.from_slot === tr.from_slot && x.to_slot === tr.to_slot) === i;
    const travel = prev !== undefined && prev.slot_id === tr.from_slot && first ? travelBefore(day, prev, to) : 0;
    const end = parseClock(to.window[0]) * 60;
    out.push({ transition_id: transitionId(day.day_index, i + 1), from_slot: tr.from_slot, to_slot: tr.to_slot, span: [end - travel, end] });
  });
  return out;
}

/** A slot window or transition span in seconds since its day's 00:00. */
export type BlockSpan = { id: string; day_index: number; span: [number, number] };

/** Slot windows and transition spans of every day, in seconds since that day's 00:00, by block id. */
export function blockSpans(t: Timeline): Map<string, BlockSpan> {
  const out = new Map<string, BlockSpan>();
  for (const day of t.schedule.days) {
    for (const s of day.slots) {
      out.set(s.slot_id, { id: s.slot_id, day_index: day.day_index, span: [parseClock(s.window[0]) * 60, parseClock(s.window[1]) * 60] });
    }
    for (const tr of transitionBlocks(day)) out.set(tr.transition_id, { id: tr.transition_id, day_index: day.day_index, span: tr.span });
  }
  return out;
}

/**
 * B2: places a block plan. Each block's clips go back to back from the block start in plan order;
 * the clip that would pass the block end is trimmed there and later clips of the block are dropped.
 * Output is ordered by day, then block start (ties by block end, then id), then plan order.
 */
export function layoutBlocks(t: Timeline, catalog: ClipCatalog, plan: readonly BlockPlanEntry[]): BlockPlacement[] {
  const blocks = blockSpans(t);
  const byBlock = new Map<string, BlockPlanEntry[]>();
  for (const e of plan) {
    if (!blocks.has(e.slot_id)) {
      throw new TimelineError("SLOT_NOT_FOUND", `block ${e.slot_id} is neither a slot nor a transition of the schedule`, { slot_id: e.slot_id });
    }
    const c = catalog.getClip(e.clip_id);
    if (!c) throw new TimelineError("CLIP_NOT_FOUND", `planned clip ${e.clip_id} is not in the catalog`, { clip_id: e.clip_id });
    if (!(e.source_start_s >= c.source_start_s && e.source_end_s <= c.source_end_s && e.source_end_s > e.source_start_s)) {
      throw new TimelineError(
        "BLOCK_PLAN_INVALID",
        `planned span ${e.source_start_s}-${e.source_end_s} of clip ${e.clip_id} is empty or outside ${c.source_start_s}-${c.source_end_s}`,
        { slot_id: e.slot_id, clip_id: e.clip_id },
      );
    }
    byBlock.set(e.slot_id, [...(byBlock.get(e.slot_id) ?? []), e]);
  }

  const order = [...byBlock.keys()]
    .map((id) => blocks.get(id)!)
    .sort((a, b) => a.day_index - b.day_index || a.span[0] - b.span[0] || a.span[1] - b.span[1] || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  const placed: BlockPlacement[] = [];
  for (const block of order) {
    const dayStart = (block.day_index - 1) * 86400;
    const [, end] = block.span;
    let at = block.span[0];
    for (const e of byBlock.get(block.id)!) {
      if (at >= end) break;
      const clip = catalog.getClip(e.clip_id)!;
      const len = e.source_end_s - e.source_start_s;
      const trimmed = at + len > end;
      const used = trimmed ? end - at : len;
      placed.push({
        slot_id: block.id,
        day_index: block.day_index,
        clip,
        source_start_s: e.source_start_s,
        source_end_s: trimmed ? round3(e.source_start_s + used) : e.source_end_s,
        synthetic_start_s: round3(dayStart + at),
        synthetic_end_s: round3(dayStart + at + used),
        trimmed,
      });
      at += used;
    }
  }
  return placed;
}
