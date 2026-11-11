import type { Clip, ClipCatalog } from "./clip.js";
import { TIME_OF_DAY_RANGES } from "./clock.js";
import { TimelineError } from "./errors.js";
import { clipSeconds, slotWindow } from "./match/filters.js";
import { rngFor } from "./rng.js";
import type { Timeline } from "./timeline.js";

export type PlacedClip = {
  slot_id: string;
  day_index: number;
  clip: Clip;
  synthetic_start_s: number;
  synthetic_end_s: number;
};

const round3 = (x: number) => Math.round(x * 1000) / 1000;

/**
 * Window part the clips can sit in, in seconds since the day's 00:00. When every
 * clip has a known bucket, the window is cut to the span of their buckets
 * (their union taken as one interval), unless that leaves less than the clips need.
 */
function placementRange(window: [number, number], clips: readonly Clip[], need: number): [number, number] {
  const full: [number, number] = [window[0] * 60, window[1] * 60];
  const ranges = clips.map((c) => (c.time_of_day === "unknown" ? undefined : TIME_OF_DAY_RANGES[c.time_of_day]));
  if (ranges.some((r) => r === undefined)) return full;
  const lo = Math.max(window[0], Math.min(...ranges.map((r) => r![0]))) * 60;
  const hi = Math.min(window[1], Math.max(...ranges.map((r) => r![1]))) * 60;
  return hi - lo >= need ? [lo, hi] : full;
}

/** Synthetic times for every assigned clip, in schedule and assignment order (§6.5, R26). */
export function layoutTimeline(t: Timeline, catalog: ClipCatalog): PlacedClip[] {
  const placed: PlacedClip[] = [];
  for (const day of t.schedule.days) {
    const dayStart = (day.day_index - 1) * 86400;
    for (const slot of day.slots) {
      const clips = (t.slots[slot.slot_id]?.clip_ids ?? []).map((id) => {
        const c = catalog.getClip(id);
        if (!c) throw new TimelineError("CLIP_NOT_FOUND", `assigned clip ${id} is not in the catalog`, { clip_id: id });
        return c;
      });
      if (clips.length === 0) continue;
      const need = clips.reduce((n, c) => n + clipSeconds(c), 0);
      const [lo, hi] = placementRange(slotWindow(slot), clips, need);
      const free = Math.max(0, hi - lo - need);
      // k sorted uniforms split `free` into k+1 non-negative gaps.
      const rng = rngFor(t.world.seed, "place", slot.slot_id);
      const cuts = clips.map(() => rng.next()).sort((a, b) => a - b);
      let at = lo;
      let prevCut = 0;
      clips.forEach((clip, i) => {
        at += (cuts[i]! - prevCut) * free;
        prevCut = cuts[i]!;
        const end = at + clipSeconds(clip);
        placed.push({
          slot_id: slot.slot_id,
          day_index: day.day_index,
          clip,
          synthetic_start_s: round3(dayStart + at),
          synthetic_end_s: round3(dayStart + end),
        });
        at = end;
      });
    }
  }
  return placed;
}
