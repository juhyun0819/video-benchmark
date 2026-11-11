import { z } from "zod";
import type { ClipCatalog } from "./clip.js";
import { TimelineError } from "./errors.js";
import { findSlot, hardFilterFailures } from "./match/filters.js";
import { ScheduleSchema, type Schedule, type ScheduleDay } from "./schedule/types.js";
import { WorldSchema, type World } from "./world.js";

export const RelaxationSchema = z.enum(["ignore_tags", "widen_window_30m", "dropped_optional"]);
export type Relaxation = z.infer<typeof RelaxationSchema>;

export const SlotAssignmentSchema = z.object({
  clip_ids: z.array(z.string()),
  relaxations: z.array(RelaxationSchema),
});
export type SlotAssignment = z.infer<typeof SlotAssignmentSchema>;

/** A schedule plus the clips chosen for its slots; every slot of every day has an entry. */
export const TimelineSchema = z.object({
  schema_version: z.literal(1),
  id: z.string(),
  name: z.string().optional(),
  revision: z.number().int().min(0),
  world: WorldSchema,
  schedule: ScheduleSchema,
  slots: z.record(z.string(), SlotAssignmentSchema),
});
export type Timeline = z.infer<typeof TimelineSchema>;

const RELAXATION_ORDER = RelaxationSchema.options;

/** Empty entries for every slot of the schedule, in schedule order; `keep` supplies existing entries. */
export function slotEntries(
  days: readonly ScheduleDay[],
  keep: (slotId: string) => SlotAssignment | undefined = () => undefined,
): Record<string, SlotAssignment> {
  const out: Record<string, SlotAssignment> = {};
  for (const day of days) {
    for (const slot of day.slots) out[slot.slot_id] = keep(slot.slot_id) ?? { clip_ids: [], relaxations: [] };
  }
  return out;
}

export function createTimeline(world: World, schedule: Schedule, genId: () => string): Timeline {
  return { schema_version: 1, id: genId(), revision: 0, world, schedule, slots: slotEntries(schedule.days) };
}

export function assertRevision(t: Timeline, expected?: number): void {
  if (expected !== undefined && expected !== t.revision) {
    throw new TimelineError("REVISION_MISMATCH", `expected revision ${expected}, timeline is at ${t.revision}`, {
      expected,
      actual: t.revision,
    });
  }
}

function withSlot(t: Timeline, slotId: string, entry: SlotAssignment): Timeline {
  return { ...t, revision: t.revision + 1, slots: { ...t.slots, [slotId]: entry } };
}

/** Adds a clip to a slot after checking the hard filters at the given relaxations. */
export function assignClip(
  t: Timeline,
  catalog: ClipCatalog,
  a: { slotId: string; clipId: string; relaxations?: Relaxation[] },
): Timeline {
  findSlot(t, a.slotId);
  const clip = catalog.getClip(a.clipId);
  if (!clip) throw new TimelineError("CLIP_NOT_FOUND", `clip ${a.clipId} is not in the catalog`, { clip_id: a.clipId });
  const relax = a.relaxations ?? [];
  if (relax.includes("dropped_optional")) {
    // R36: only fillTimeline marks a slot dropped; an assigned slot is never dropped.
    throw new TimelineError("CLIP_INELIGIBLE", `"dropped_optional" is not a relaxation a clip can be assigned with`, {
      slot_id: a.slotId,
      clip_id: a.clipId,
      failed: ["invalid_relaxation"],
    });
  }
  const failed = hardFilterFailures(t, catalog, a.slotId, clip, relax);
  if (failed.length > 0) {
    throw new TimelineError("CLIP_INELIGIBLE", `clip ${a.clipId} does not fit slot ${a.slotId}: ${failed.join(", ")}`, {
      slot_id: a.slotId,
      clip_id: a.clipId,
      failed,
    });
  }
  const entry = t.slots[a.slotId] ?? { clip_ids: [], relaxations: [] };
  const used = new Set([...entry.relaxations.filter((r) => r !== "dropped_optional"), ...relax]);
  return withSlot(t, a.slotId, {
    clip_ids: [...entry.clip_ids, a.clipId],
    relaxations: RELAXATION_ORDER.filter((r) => used.has(r)),
  });
}

/** Removes one clip from a slot, or clears the slot when `clipId` is absent. */
export function unassignClip(t: Timeline, a: { slotId: string; clipId?: string }): Timeline {
  findSlot(t, a.slotId);
  const entry = t.slots[a.slotId] ?? { clip_ids: [], relaxations: [] };
  if (a.clipId === undefined) return withSlot(t, a.slotId, { clip_ids: [], relaxations: [] });
  if (!entry.clip_ids.includes(a.clipId)) {
    throw new TimelineError("CLIP_NOT_FOUND", `clip ${a.clipId} is not assigned to slot ${a.slotId}`, {
      slot_id: a.slotId,
      clip_id: a.clipId,
    });
  }
  const clip_ids = entry.clip_ids.filter((id) => id !== a.clipId);
  return withSlot(t, a.slotId, { clip_ids, relaxations: clip_ids.length === 0 ? [] : entry.relaxations });
}
