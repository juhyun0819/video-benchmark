import type { Clip, ClipCatalog } from "./clip.js";
import { TimelineError } from "./errors.js";
import {
  bindingAccepts,
  failuresWith,
  findSlot,
  usageOf,
  type LocatedSlot,
} from "./match/filters.js";
import type { ScheduleDay, Slot } from "./schedule/types.js";
import { slotEntries, type Relaxation, type Timeline } from "./timeline.js";

export { bindingAccepts, hardFilterFailures, type HardFilter } from "./match/filters.js";

export type RankedCandidate = { clip: Clip; score: number; reasons: string[] };
export type RecentPick = { day_index: number; role: string; clip_id: string };

/** Picks one clip id from the top candidates (an LLM through MCP, or `rankSelector`). */
export interface Selector {
  choose(slot: Slot, candidates: RankedCandidate[], recent: RecentPick[]): string;
}

/** Candidates handed to the selector (§6.3). */
export const SELECTOR_TOP_K = 10;
/** Days of history handed to the selector. */
export const RECENT_DAYS = 3;
/** Day regenerations after the first fill fails (§6.4 step 3). */
export const MAX_REGEN_ATTEMPTS = 5;
/** The clip's day_group already fills another slot of the same day at the same node. */
const SAME_GROUP_BONUS = 0.5;
/** The clip's day_group already fills a slot of the same day at any node (R43): one synthetic day should retire one source day. */
const SAME_DAY_GROUP_BONUS = 0.25;
/** §6.4 steps 0–2; filters 1, 3, 4 are never relaxed. */
const LADDER: readonly Relaxation[][] = [[], ["ignore_tags"], ["ignore_tags", "widen_window_30m"]];

/** Lowercase letter/digit tokens. */
export function words(text: string): Set<string> {
  return new Set(text.toLowerCase().split(/[^\p{L}\p{N}]+/u).filter((w) => w.length > 0));
}

/** |a ∩ b| / |a ∪ b|; two empty sets give 0. */
export function jaccard(a: ReadonlySet<string>, b: ReadonlySet<string>): number {
  let inter = 0;
  for (const w of a) if (b.has(w)) inter++;
  const union = a.size + b.size - inter;
  return union === 0 ? 0 : inter / union;
}

const textOf = (activity: string, tags: readonly string[]) => words([activity, ...tags].join(" "));

function clipsIn(t: Timeline, catalog: ClipCatalog, slots: readonly Slot[]): Clip[] {
  return slots.flatMap((s) => (t.slots[s.slot_id]?.clip_ids ?? []).map((id) => catalog.getClip(id)!));
}

function rank(t: Timeline, catalog: ClipCatalog, at: LocatedSlot, relax: readonly Relaxation[]): RankedCandidate[] {
  const { day, slot } = at;
  const usage = usageOf(t, catalog);
  const slotWords = textOf(slot.spec.activity, slot.spec.tags);
  const sameNode = day.slots.filter((s) => s.spec.node === slot.spec.node);
  const todayGroups = new Set(clipsIn(t, catalog, sameNode).map((c) => c.day_group));
  const todayAnyNode = new Set(clipsIn(t, catalog, day.slots).map((c) => c.day_group));
  const yesterday = t.schedule.days.find((d) => d.day_index === day.day_index - 1);
  const yesterdayWords = clipsIn(t, catalog, yesterday?.slots.filter((s) => s.role === slot.role) ?? []).map((c) =>
    textOf(c.activity, c.tags),
  );
  const routine = t.world.dials.routine_strength;

  const ranked: RankedCandidate[] = [];
  for (const clip of catalog.clips()) {
    if (failuresWith(t.world, at, clip, relax, usage).length > 0) continue;
    const clipWords = textOf(clip.activity, clip.tags);
    const reasons: string[] = [];
    let score = jaccard(slotWords, clipWords);
    if (score > 0) reasons.push(`activity/tags overlap ${score.toFixed(2)}`);
    if (todayGroups.has(clip.day_group)) {
      score += SAME_GROUP_BONUS;
      reasons.push(`day_group ${clip.day_group} already used today at ${slot.spec.node}`);
    }
    if (todayAnyNode.has(clip.day_group)) {
      score += SAME_DAY_GROUP_BONUS;
      reasons.push(`day_group ${clip.day_group} already used today`);
    }
    const similarity = Math.max(0, ...yesterdayWords.map((w) => jaccard(clipWords, w)));
    if (routine * similarity > 0) {
      score += routine * similarity;
      reasons.push(`similar to yesterday's ${slot.role} (${similarity.toFixed(2)})`);
    }
    ranked.push({ clip, score, reasons });
  }
  return ranked.sort(
    (a, b) => b.score - a.score || (a.clip.clip_id < b.clip.clip_id ? -1 : a.clip.clip_id > b.clip.clip_id ? 1 : 0),
  );
}

/** Clips passing every hard filter at `relax`, by score desc then clip_id (§6.2, §6.3). */
export function candidatesFor(t: Timeline, catalog: ClipCatalog, slotId: string, relax: Relaxation[]): RankedCandidate[] {
  return rank(t, catalog, findSlot(t, slotId), relax);
}

export const rankSelector: Selector = {
  choose(slot, candidates) {
    const top = candidates[0];
    if (!top) throw new Error(`rankSelector: no candidates for ${slot.slot_id}`);
    return top.clip.clip_id;
  },
};

/** Assignments of the RECENT_DAYS days before `dayIndex`, in day and slot order (the selector's `recent`). */
export function recentPicks(t: Timeline, dayIndex: number): RecentPick[] {
  return t.schedule.days
    .filter((d) => d.day_index >= dayIndex - RECENT_DAYS && d.day_index < dayIndex)
    .flatMap((d) =>
      d.slots.flatMap((s) =>
        (t.slots[s.slot_id]?.clip_ids ?? []).map((clip_id) => ({ day_index: d.day_index, role: s.role, clip_id })),
      ),
    );
}

/**
 * D7: each node a required slot uses needs at least one distinct eligible
 * day_group per day it appears on. Eligible = bound to the node and matching
 * the category and setting of one of its required slots.
 */
export function preflightCapacity(t: Timeline, catalog: ClipCatalog): void {
  const need = new Map<string, { days: Set<number>; specs: Set<string> }>();
  for (const day of t.schedule.days) {
    for (const slot of day.slots) {
      if (!slot.required) continue;
      const n = need.get(slot.spec.node) ?? { days: new Set<number>(), specs: new Set<string>() };
      n.days.add(day.day_index);
      n.specs.add(`${slot.spec.category}|${slot.spec.setting}`);
      need.set(slot.spec.node, n);
    }
  }
  const shortfalls: { node: string; needed_days: number; distinct_day_groups: number }[] = [];
  for (const [node, n] of need) {
    const groups = new Set(
      catalog
        .clips()
        .filter((c) => n.specs.has(`${c.category}|${c.setting}`) && bindingAccepts(t.world, node, c))
        .map((c) => c.day_group),
    );
    if (groups.size < n.days.size) shortfalls.push({ node, needed_days: n.days.size, distinct_day_groups: groups.size });
  }
  if (shortfalls.length > 0) {
    const text = shortfalls.map((s) => `${s.node} needs ${s.needed_days} days, has ${s.distinct_day_groups} day_groups`);
    throw new TimelineError("BANK_CAPACITY", `clip bank too small: ${text.join("; ")}`, { shortfalls });
  }
}

/** N6: whether `day` may leave a required `role` slot empty: fewer than max_days other days have used the allowance. */
function allowanceLeft(t: Timeline, day: ScheduleDay, role: string): boolean {
  const max = t.world.dials.missing_required_allowance.find((a) => a.role === role)?.max_days ?? 0;
  if (max === 0) return false;
  const used = t.schedule.days.filter(
    (d) =>
      d.day_index !== day.day_index &&
      d.slots.some((s) => s.role === role && t.slots[s.slot_id]?.relaxations.includes("dropped_required")),
  ).length;
  return used < max;
}

/** Fills the empty slots of one day in order; mutates `t.slots` (a private copy) and returns unfillable required slots. */
function fillDay(t: Timeline, catalog: ClipCatalog, selector: Selector, day: ScheduleDay): string[] {
  const failed: string[] = [];
  for (const slot of day.slots) {
    if ((t.slots[slot.slot_id]?.clip_ids.length ?? 0) > 0) continue;
    const at = { day, slot };
    let level: readonly Relaxation[] | undefined;
    let ranked: RankedCandidate[] = [];
    for (const relax of LADDER) {
      ranked = rank(t, catalog, at, relax);
      if (ranked.length > 0) {
        level = relax;
        break;
      }
    }
    if (level === undefined) {
      if (slot.required && allowanceLeft(t, day, slot.role)) t.slots[slot.slot_id] = { clip_ids: [], relaxations: ["dropped_required"] };
      else if (slot.required) failed.push(slot.slot_id);
      else t.slots[slot.slot_id] = { clip_ids: [], relaxations: ["dropped_optional"] };
      continue;
    }
    const top = ranked.slice(0, SELECTOR_TOP_K);
    const chosen = selector.choose(slot, top, recentPicks(t, day.day_index));
    if (!top.some((c) => c.clip.clip_id === chosen)) {
      throw new TimelineError("CLIP_INELIGIBLE", `selector chose ${chosen}, which is not a candidate for ${slot.slot_id}`, {
        slot_id: slot.slot_id,
        clip_id: chosen,
        failed: ["not_a_candidate"],
      });
    }
    t.slots[slot.slot_id] = { clip_ids: [chosen], relaxations: [...level] };
  }
  return failed;
}

/** Swaps in a regenerated day; every assignment of that day is cleared (R27). */
function replaceDay(t: Timeline, index: number, next: ScheduleDay): Timeline {
  const days = t.schedule.days.map((d, i) => (i === index ? next : d));
  const fresh = new Set(next.slots.map((s) => s.slot_id));
  return { ...t, schedule: { ...t.schedule, days }, slots: slotEntries(days, (id) => (fresh.has(id) ? undefined : t.slots[id])) };
}

const isInfeasible = (e: unknown) => e instanceof TimelineError && e.code === "SCHEDULE_INFEASIBLE";

/**
 * Fills every empty slot, day by day (§6.4, D10, D11, D13). A day whose required
 * slot finds no candidate at any ladder level is replaced by
 * `regen(day, a + 1 … a + 5)`, where `a` is the day's original attempt (R34), and
 * refilled; after the 5th failure nothing is returned and UNFILLABLE names the
 * slots. One revision per call.
 */
export function fillTimeline(
  t: Timeline,
  catalog: ClipCatalog,
  selector: Selector,
  regen: (dayIndex: number, attempt: number) => ScheduleDay,
): Timeline {
  let cur: Timeline = { ...t, slots: { ...t.slots } };
  for (let i = 0; i < cur.schedule.days.length; i++) {
    const { day_index: dayIndex, attempt: base } = cur.schedule.days[i]!;
    let failed = fillDay(cur, catalog, selector, cur.schedule.days[i]!);
    let lastFailed = failed;
    for (let k = 1; failed.length > 0 && k <= MAX_REGEN_ATTEMPTS; k++) {
      let next: ScheduleDay;
      try {
        next = regen(dayIndex, base + k);
      } catch (e) {
        if (isInfeasible(e)) continue; // R6: counts as a failed attempt.
        throw e;
      }
      cur = replaceDay(cur, i, next);
      failed = fillDay(cur, catalog, selector, next);
      if (failed.length > 0) lastFailed = failed;
    }
    if (failed.length > 0) {
      throw new TimelineError(
        "UNFILLABLE",
        `day ${dayIndex}: no clip for ${lastFailed.join(", ")} after ${MAX_REGEN_ATTEMPTS} regenerations`,
        { slots: lastFailed, day: dayIndex, attempts: MAX_REGEN_ATTEMPTS },
      );
    }
  }
  return { ...cur, revision: t.revision + 1 };
}
