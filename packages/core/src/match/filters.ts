import type { Clip, ClipCatalog } from "../clip.js";
import { bucketOverlaps, parseClock } from "../clock.js";
import { TimelineError } from "../errors.js";
import type { ScheduleDay, Slot } from "../schedule/types.js";
import type { Relaxation, Timeline } from "../timeline.js";
import { tripClipNode, tripPlaceOf, type World } from "../world.js";

export type HardFilter =
  | "category_setting_node"
  | "time_of_day"
  | "already_used"
  | "day_group_other_day"
  | "does_not_fit"
  | "tags"
  | "trip_sequence";

export type LocatedSlot = { day: ScheduleDay; slot: Slot };

export function findSlot(t: Timeline, slotId: string): LocatedSlot {
  for (const day of t.schedule.days) {
    const slot = day.slots.find((s) => s.slot_id === slotId);
    if (slot) return { day, slot };
  }
  throw new TimelineError("SLOT_NOT_FOUND", `slot ${slotId} is not in the schedule`, { slot_id: slotId });
}

export const slotWindow = (slot: Slot): [number, number] => [parseClock(slot.window[0]), parseClock(slot.window[1])];
export const clipSeconds = (c: Clip): number => c.source_end_s - c.source_start_s;

/**
 * True when every binding field present for the node matches; `kitchen` uses its default node's binding.
 * A trip place `<trip>:<place>` uses its trip's binding and takes only clips labeled `trip:<place>`
 * whose day_group is in the trip's day_group_sequence (N2).
 */
export function bindingAccepts(world: World, nodeName: string, clip: Clip): boolean {
  const trip = tripPlaceOf(world, nodeName);
  const def = world.nodes[nodeName];
  const name = trip !== undefined ? trip.trip : def?.kind === "kitchen" ? def.default : nodeName;
  const b = world.node_binding[name];
  if (!b) return false;
  if (b.dataset && !b.dataset.includes(clip.source.dataset)) return false;
  if (b.person_source !== undefined && b.person_source !== clip.person_source) return false;
  if (trip !== undefined && clip.node !== tripClipNode(trip.place)) return false;
  if (b.node !== undefined && b.node !== clip.node) return false;
  if (b.day_group_sequence && !b.day_group_sequence.includes(clip.day_group)) return false;
  if (b.tags_any && !b.tags_any.some((tag) => clip.tags.includes(tag))) return false;
  return true;
}

/** Hard filter 1: category, setting and bound node. */
export function matchesSlotNode(world: World, slot: Slot, clip: Clip): boolean {
  return clip.category === slot.spec.category && clip.setting === slot.spec.setting && bindingAccepts(world, slot.spec.node, clip);
}

const videoKey = (c: Clip) => `${c.source.dataset}:${c.source.video_id}`;

/** What the timeline already uses, collected once per query. */
export type Usage = {
  clipIds: Set<string>;
  groupDays: Map<string, Set<number>>;
  spans: Map<string, [number, number][]>;
  slotSeconds: Map<string, number>;
};

export function usageOf(t: Timeline, catalog: ClipCatalog): Usage {
  const u: Usage = { clipIds: new Set(), groupDays: new Map(), spans: new Map(), slotSeconds: new Map() };
  for (const day of t.schedule.days) {
    for (const slot of day.slots) {
      for (const id of t.slots[slot.slot_id]?.clip_ids ?? []) {
        const c = catalog.getClip(id);
        if (!c) throw new TimelineError("CLIP_NOT_FOUND", `assigned clip ${id} is not in the catalog`, { clip_id: id });
        u.clipIds.add(id);
        const days = u.groupDays.get(c.day_group) ?? new Set<number>();
        u.groupDays.set(c.day_group, days.add(day.day_index));
        const spans = u.spans.get(videoKey(c)) ?? [];
        u.spans.set(videoKey(c), [...spans, [c.source_start_s, c.source_end_s]]);
        u.slotSeconds.set(slot.slot_id, (u.slotSeconds.get(slot.slot_id) ?? 0) + clipSeconds(c));
      }
    }
  }
  return u;
}

/** Names of the hard filters the clip fails for the slot at the given relaxations (§6.2, D13, R32). */
export function failuresWith(world: World, at: LocatedSlot, clip: Clip, relax: readonly Relaxation[], u: Usage): HardFilter[] {
  const { day, slot } = at;
  const window = slotWindow(slot);
  const failed: HardFilter[] = [];
  if (!matchesSlotNode(world, slot, clip)) failed.push("category_setting_node");
  if (!bucketOverlaps(clip.time_of_day, window, relax.includes("widen_window_30m") ? 30 : 0)) failed.push("time_of_day");
  const overlapsUsedSpan = (u.spans.get(videoKey(clip)) ?? []).some(
    ([s, e]) => clip.source_start_s < e && s < clip.source_end_s,
  );
  if (u.clipIds.has(clip.clip_id) || overlapsUsedSpan) failed.push("already_used");
  if ([...(u.groupDays.get(clip.day_group) ?? [])].some((d) => d !== day.day_index)) failed.push("day_group_other_day");
  if ((u.slotSeconds.get(slot.slot_id) ?? 0) + clipSeconds(clip) > (window[1] - window[0]) * 60) failed.push("does_not_fit");
  if (!relax.includes("ignore_tags") && slot.spec.tags.length > 0 && !slot.spec.tags.some((tag) => clip.tags.includes(tag))) {
    failed.push("tags");
  }
  if (day.trip !== undefined && clip.day_group !== tripDayGroup(world, day.trip)) failed.push("trip_sequence");
  return failed;
}

/** N5: the source day_group a trip day must use, from its binding's day_group_sequence; never relaxed. */
export function tripDayGroup(world: World, trip: NonNullable<ScheduleDay["trip"]>): string | undefined {
  return world.node_binding[trip.node]?.day_group_sequence?.[trip.index];
}

export function hardFilterFailures(
  t: Timeline,
  catalog: ClipCatalog,
  slotId: string,
  clip: Clip,
  relax: readonly Relaxation[],
): HardFilter[] {
  return failuresWith(t.world, findSlot(t, slotId), clip, relax, usageOf(t, catalog));
}
