import type { ClipSpec, Setting } from "../clip.js";
import { formatClock, timeOfDayAt } from "../clock.js";
import { TimelineError } from "../errors.js";
import { TRIP_PLACE_KIND, placeKindOf, type PlaceKind } from "../gaps.js";
import { tripPlaceOf, type EveningActivity, type NodeDef, type World } from "../world.js";
import type {
  AnchorName,
  DayAnchors,
  DayDecisions,
  DroppedSlot,
  ReduceStep,
  Slot,
  SlotTransition,
  TemplateSlot,
} from "./types.js";

/** Transition gap in minutes between two place kinds; `index` is the kept-slot index of the later slot. */
export type GapFn = (from: PlaceKind, to: PlaceKind, index: number) => number;

/** What a slot needs from its node: setting, gap place kind and, for a hobby node, its definition. */
type Place = { setting: Setting; kind: PlaceKind; hobby?: Extract<NodeDef, { kind: "hobby" }> };
type Kept = { slot: TemplateSlot; node: string; place: Place };

/** A world node or a trip's virtual place (N2); undefined for an unknown name or a bare trip node. */
function placeOf(world: World, node: string): Place | undefined {
  const trip = tripPlaceOf(world, node);
  if (trip !== undefined) {
    return { setting: trip.def[`${trip.place}_setting`], kind: TRIP_PLACE_KIND[trip.place] };
  }
  const def = world.nodes[node];
  switch (def?.kind) {
    case undefined:
    case "trip":
      return undefined;
    case "home":
    case "kitchen":
      return { setting: "home_indoor", kind: placeKindOf(def.kind) };
    case "workplace":
      return { setting: "venue_indoor", kind: placeKindOf(def.kind) };
    case "hobby":
      return { setting: def.setting, kind: placeKindOf(def.kind), hobby: def };
    case "frequent_place":
      return { setting: def.setting, kind: placeKindOf(def.kind) };
  }
}

const TRIP_NODE_PREFIX = "$trip:";

/** World node a template slot points at today; null for a skipped meal, undefined when it resolves to nothing. */
function resolveNode(slot: TemplateSlot, decisions: DayDecisions, world: World): string | null | undefined {
  if (slot.node === "$hobby") return decisions.hobby_node;
  if (slot.node.startsWith(TRIP_NODE_PREFIX)) {
    const trip = decisions.trip;
    return trip === undefined ? undefined : `${trip.node}:${slot.node.slice(TRIP_NODE_PREFIX.length)}`;
  }
  if (slot.node !== "$meal") return slot.node;
  const place = slot.meal === undefined ? undefined : decisions.meal_place[slot.meal];
  if (place === "skip") return null;
  if (place === undefined) return undefined;
  if (place === "home" || place === "workplace") return place;
  return world.meal_style.place_nodes[place];
}

type Layout = { start: number[]; end: number[]; gaps: number[] };
type Failure = { index: number; role: string; reason: string };

/**
 * Times for the kept slots: each starts at max(anchor, previous end + gap); it
 * ends at its end anchor, after its duration, or where the next slot's gap
 * begins; the last one at sleep. Returns the first slot that does not fit.
 */
function layout(kept: Kept[], anchors: DayAnchors, gap: GapFn): Layout | Failure {
  const anchor = (name: AnchorName): number => anchors[name]!;
  const n = kept.length;
  // gaps[i]: transition before kept[i] (0 when the node does not change).
  const gaps: number[] = kept.map((k, i) => {
    if (i === 0) return 0;
    const prev = kept[i - 1]!;
    return prev.node === k.node ? 0 : gap(prev.place.kind, k.place.kind, i);
  });

  const start: number[] = [];
  for (let i = 0; i < n; i++) {
    const { slot } = kept[i]!;
    if (i === 0) {
      start.push(anchors.wake);
      continue;
    }
    const prev = kept[i - 1]!.slot;
    const prevStart = start[i - 1]!;
    const prevEnd =
      prev.end_anchor !== undefined
        ? anchor(prev.end_anchor)
        : prev.duration_minutes !== undefined
          ? prevStart + prev.duration_minutes
          : prevStart + prev.min_minutes;
    const earliest = prevEnd + gaps[i]!;
    start.push(slot.anchor !== undefined ? Math.max(anchor(slot.anchor), earliest) : earliest);
  }
  const end: number[] = kept.map(({ slot }, i) => {
    if (slot.end_anchor !== undefined) return anchor(slot.end_anchor);
    if (slot.duration_minutes !== undefined) return start[i]! + slot.duration_minutes;
    if (i < n - 1) return start[i + 1]! - gaps[i + 1]!;
    return anchors.sleep;
  });

  for (let i = 0; i < n; i++) {
    const { slot } = kept[i]!;
    const s = start[i]!;
    const e = end[i]!;
    const fail = (reason: string): Failure => ({ index: i, role: slot.role, reason });
    if (e - s < slot.min_minutes) return fail(`${e - s} min is shorter than ${slot.min_minutes}`);
    if (i < n - 1 && e > start[i + 1]! - gaps[i + 1]!) return fail("overlaps the next slot");
    if (e > anchors.sleep) return fail(`ends after sleep ${formatClock(anchors.sleep)}`);
  }
  return { start, end, gaps };
}

/**
 * R24: the failing slot itself when optional, else the nearest optional slot
 * before it, else the last optional slot after it; -1 when none is optional.
 * B16: overridden (`exempt`) slots count as not optional.
 */
function slotToDrop(kept: Kept[], failing: number, exempt: ReadonlySet<string>): number {
  const optional = (i: number) => !kept[i]!.slot.required && !exempt.has(kept[i]!.slot.role);
  for (let i = failing; i >= 0; i--) if (optional(i)) return i;
  for (let i = kept.length - 1; i > failing; i--) if (optional(i)) return i;
  return -1;
}

/**
 * E4: applies the first unused reduce step, in order, whose optional slot is
 * still kept and can affect the failing slot (G10: a slot's layout depends only
 * on itself, the slots before it and the next one, so a drop may target up to
 * failing + 1; a shrink only up to failing). A shrink needs a duration above
 * min_minutes. Returns false when no step applies. B16: steps on an
 * overridden (`exempt`) role never apply.
 */
function applyReduceStep(
  kept: Kept[],
  steps: readonly ReduceStep[],
  used: boolean[],
  failing: number,
  dropped: DroppedSlot[],
  exempt: ReadonlySet<string>,
): boolean {
  for (let si = 0; si < steps.length; si++) {
    if (used[si]) continue;
    const step = steps[si]!;
    const i = exempt.has(step.role) ? -1 : kept.findIndex((k) => k.slot.role === step.role && !k.slot.required);
    if (i < 0) {
      used[si] = true;
      continue;
    }
    if (i > failing + (step.action === "drop" ? 1 : 0)) continue;
    used[si] = true;
    const k = kept[i]!;
    if (step.action === "drop") {
      kept.splice(i, 1);
      dropped.push({ role: step.role, reason: "no_time" });
      return true;
    }
    const { duration_minutes: duration, min_minutes: min } = k.slot;
    if (duration === undefined || duration <= min) continue;
    kept[i] = { ...k, slot: { ...k.slot, duration_minutes: min } };
    dropped.push({ role: step.role, reason: "shrunk" });
    return true;
  }
  return false;
}

/**
 * G20: a slot anchored at sport (the hobby_day hobby) goes in anchor order
 * among the anchored meal slots: right before the first meal whose anchor is
 * later than the sport anchor, else right after the last meal. Other slots
 * keep their template order. Templates without such a slot are unchanged.
 */
export function orderBySportAnchor(template: readonly TemplateSlot[], anchors: DayAnchors): TemplateSlot[] {
  const at = template.findIndex((s) => s.anchor === "sport");
  const sport = anchors.sport;
  if (at < 0 || sport === undefined) return [...template];
  const hobby = template[at]!;
  const rest = template.filter((_, i) => i !== at);
  const isMeal = (s: TemplateSlot) => s.meal !== undefined && s.anchor !== undefined && anchors[s.anchor] !== undefined;
  const later = rest.findIndex((s) => isMeal(s) && anchors[s.anchor!]! > sport);
  let insert = later;
  if (insert < 0) {
    insert = at;
    rest.forEach((s, i) => {
      if (isMeal(s)) insert = i + 1;
    });
  }
  return [...rest.slice(0, insert), hobby, ...rest.slice(insert)];
}

/** B8: the shortest an evening activity, and home_evening on its day, may be shrunk to. */
export const EVENING_MIN_MINUTES = 20;
/** B8: the template slot each position is inserted after. */
const EVENING_AFTER: Record<EveningActivity["position"], string> = { before_dinner: "work_pm", after_dinner: "dinner" };
/** B8: the slot that absorbs a push before the activity shrinks. */
const EVENING_FOLLOWER = "home_evening";

/**
 * B8: the template with each evening activity inserted right after work_pm
 * (before_dinner) or right after dinner (after_dinner), in dial order. The
 * slot is optional, lasts duration_min, may shrink to 20 min and takes its
 * place, setting, activity and tags from its hobby node (category hobby).
 * With any activity, home_evening may also shrink to 20 min. An empty list
 * returns the template unchanged.
 */
export function withEveningSlots(template: readonly TemplateSlot[], evening: readonly EveningActivity[]): TemplateSlot[] {
  if (evening.length === 0) return [...template];
  const roles = new Set(template.map((s) => s.role));
  for (const a of evening) {
    if (roles.has(a.role) || !roles.has(EVENING_AFTER[a.position])) {
      const why = roles.has(a.role) ? `role ${a.role} is already a template role` : `the template has no ${EVENING_AFTER[a.position]} slot`;
      throw new TimelineError("WORLD_INVALID", `dials.evening_activities: ${why}`, {
        issues: [{ path: "dials.evening_activities", message: why }],
      });
    }
  }
  const slotOf = (a: EveningActivity): TemplateSlot => ({
    role: a.role,
    category: "hobby",
    node: a.node,
    activity: "hobby",
    tags: [],
    required: false,
    duration_minutes: a.duration_min,
    min_minutes: Math.min(EVENING_MIN_MINUTES, a.duration_min),
  });
  const out: TemplateSlot[] = [];
  for (const slot of template) {
    out.push(slot.role === EVENING_FOLLOWER ? { ...slot, min_minutes: Math.min(slot.min_minutes, EVENING_MIN_MINUTES) } : slot);
    for (const a of evening) if (EVENING_AFTER[a.position] === slot.role) out.push(slotOf(a));
  }
  return out;
}

/**
 * B8: shrinks evening activities (roles in `elastic`) at or before the failing
 * slot, nearest first, minute by minute, until the failure is gone or moves to
 * a later slot. Nothing shrinks when even all of them at min_minutes would not
 * get past the failure. The first shrink of a role is recorded.
 */
function shrinkEvening(
  kept: Kept[],
  elastic: ReadonlySet<string>,
  failing: number,
  anchors: DayAnchors,
  gap: GapFn,
  dropped: DroppedSlot[],
): boolean {
  const candidates: number[] = [];
  for (let i = Math.min(failing, kept.length - 1); i >= 0; i--) {
    const { role, duration_minutes: d, min_minutes: min } = kept[i]!.slot;
    if (elastic.has(role) && d !== undefined && d > min) candidates.push(i);
  }
  if (candidates.length === 0) return false;
  const passes = (trial: Kept[]): boolean => {
    const r = layout(trial, anchors, gap);
    return !("role" in r) || r.index > failing;
  };
  const withDuration = (k: Kept, minutes: number): Kept => ({ ...k, slot: { ...k.slot, duration_minutes: minutes } });
  const allMin = kept.map((k, i) => (candidates.includes(i) ? withDuration(k, k.slot.min_minutes) : k));
  if (!passes(allMin)) return false;
  for (const i of candidates) {
    const k = kept[i]!;
    const role = k.slot.role;
    if (!dropped.some((x) => x.role === role && x.reason === "shrunk")) dropped.push({ role, reason: "shrunk" });
    for (let m = k.slot.duration_minutes! - 1; m >= k.slot.min_minutes; m--) {
      kept[i] = withDuration(k, m);
      if (passes(kept)) return true;
    }
  }
  return true;
}

/** B16: one day's slot override (dials.slot_overrides without the day). */
export type SlotDuration = { role: string; duration_min: number };
/** B16: home-place slots after an overridden slot may shrink to this (or their own lower min) before they are dropped. */
export const OVERRIDE_HOME_MIN_MINUTES = 20;
/** B16: the home-place slots that give way to an overridden slot before it. */
const OVERRIDE_HOME_ROLES: ReadonlySet<string> = new Set(["home_afternoon", "chores", "home_evening"]);

/**
 * B16 step 1: the overridden slot as the normal layout sees it. It lasts at
 * most its override (a template duration above it is cut to it) and its
 * min_minutes is at most the override; the slot is exempt from reductions.
 */
function overrideCapped(slot: TemplateSlot, minutes: number): TemplateSlot {
  const capped = { ...slot, min_minutes: Math.min(slot.min_minutes, minutes) };
  if (slot.duration_minutes !== undefined) capped.duration_minutes = Math.min(slot.duration_minutes, minutes);
  return capped;
}

/**
 * B16 step 2: the home-place slot after the first overridden slot that gives
 * way for a failure at `failing`. Any one that can still shrink (min_minutes
 * above 20, required or not) goes before any that can be dropped (optional);
 * within each, the failing slot itself, else the nearest one before it, else
 * the last one after it. Undefined when there is none.
 */
function homeToGiveWay(
  kept: Kept[],
  first: number,
  failing: number,
  exempt: ReadonlySet<string>,
): { index: number; action: "shrink" | "drop" } | undefined {
  const home = (i: number) => i > first && OVERRIDE_HOME_ROLES.has(kept[i]!.slot.role) && !exempt.has(kept[i]!.slot.role);
  const pick = (ok: (i: number) => boolean): number => {
    for (let i = Math.min(failing, kept.length - 1); i > first; i--) if (home(i) && ok(i)) return i;
    for (let i = kept.length - 1; i > failing; i--) if (home(i) && ok(i)) return i;
    return -1;
  };
  const shrink = pick((i) => kept[i]!.slot.min_minutes > OVERRIDE_HOME_MIN_MINUTES);
  if (shrink >= 0) return { index: shrink, action: "shrink" };
  const drop = pick((i) => !kept[i]!.slot.required);
  return drop >= 0 ? { index: drop, action: "drop" } : undefined;
}

/**
 * Lays one day's template out in time (§5.2 step 5). Slots are dropped for
 * "skip" meals, absent optional keys and unresolved optional nodes. While the
 * windows do not fit, the day type's reduce order is applied step by step (E4);
 * a failure no remaining step can affect drops an optional slot by R24 (R21).
 * Every change is recorded. A sport-anchored slot is first put in anchor
 * order among the meals (G20).
 * Required slots are never shrunk or dropped; when no optional slot is left it throws.
 * B8: `evening` activities are inserted (withEveningSlots); while the day does
 * not fit they shrink first (shrinkEvening), before the reduce order.
 * B16: with `overrides`, the day is first laid out as above with each
 * overridden slot capped at its override and exempt from every reduction
 * (overrideCapped). Then each overridden slot gets exactly its duration from
 * the start that layout gave it (an end anchor no longer applies) and the day
 * is laid out again: later slots shift; while it does not fit, the home-place
 * slots after the first overridden one shrink to 20 min (homeToGiveWay;
 * recorded "shrunk" when shorter than their template min), then optional ones
 * are dropped as no_time; nothing else gives way. A day that
 * still does not fit, or lacks an overridden slot, throws SCHEDULE_INFEASIBLE
 * with details.slot_override; an override of a role the day's template does
 * not have is WORLD_INVALID.
 */
export function resolveDayWindows(
  template: TemplateSlot[],
  anchors: DayAnchors,
  decisions: DayDecisions,
  world: World,
  gap: GapFn,
  reduceOrder: readonly ReduceStep[] = [],
  evening: readonly EveningActivity[] = [],
  overrides: readonly SlotDuration[] = [],
): { slots: Slot[]; transitions: SlotTransition[]; dropped: DroppedSlot[] } {
  const day = decisions.day_index;
  const infeasible = (role: string, reason: string): TimelineError =>
    new TimelineError("SCHEDULE_INFEASIBLE", `day ${day}: slot ${role} is infeasible (${reason})`, { day, role, reason });
  const fixed = new Map(overrides.map((o) => [o.role, o.duration_min]));
  const overrideInfeasible = (role: string, reason: string): TimelineError =>
    new TimelineError(
      "SCHEDULE_INFEASIBLE",
      `day ${day}: slot override ${role} (${fixed.get(role)} min) is infeasible (${reason})`,
      { day, role, reason, slot_override: true },
    );

  const daySlots = withEveningSlots(orderBySportAnchor(template, anchors), evening);
  for (const { role } of overrides) {
    if (daySlots.some((s) => s.role === role)) continue;
    const why = `day ${day} (${decisions.template}) has no ${role} slot`;
    throw new TimelineError("WORLD_INVALID", `dials.slot_overrides: ${why}`, {
      day,
      role,
      issues: [{ path: "dials.slot_overrides", message: why }],
    });
  }

  const kept: Kept[] = [];
  for (const entry of daySlots) {
    const minutes = fixed.get(entry.role);
    const slot = minutes === undefined ? entry : overrideCapped(entry, minutes);
    if (slot.optional_key !== undefined && decisions.optional[slot.optional_key] === false) continue;
    const node = resolveNode(slot, decisions, world);
    if (node === null) continue;
    const place = node === undefined ? undefined : placeOf(world, node);
    if (node === undefined || place === undefined) {
      if (slot.required) throw infeasible(slot.role, `node ${slot.node} resolves to nothing`);
      continue;
    }
    for (const name of [slot.anchor, slot.end_anchor]) {
      if (name !== undefined && anchors[name] === undefined) throw infeasible(slot.role, `anchor ${name} is missing`);
    }
    kept.push({ slot, node, place });
  }
  if (kept.length === 0) throw infeasible("-", "no slots");
  for (const { role } of overrides) {
    if (!kept.some((k) => k.slot.role === role)) throw overrideInfeasible(role, "the slot is not on this day");
  }

  const dropped: DroppedSlot[] = [];
  const used = reduceOrder.map(() => false);
  const exempt: ReadonlySet<string> = new Set(fixed.keys());
  const elastic = new Set(evening.map((a) => a.role).filter((role) => !exempt.has(role)));
  let result = layout(kept, anchors, gap);
  while ("role" in result) {
    if (shrinkEvening(kept, elastic, result.index, anchors, gap, dropped)) {
      // shrunk an evening activity
    } else if (!applyReduceStep(kept, reduceOrder, used, result.index, dropped, exempt)) {
      const victim = slotToDrop(kept, result.index, exempt);
      if (victim < 0) {
        throw fixed.has(result.role) ? overrideInfeasible(result.role, result.reason) : infeasible(result.role, result.reason);
      }
      dropped.push({ role: kept[victim]!.slot.role, reason: "no_time" });
      kept.splice(victim, 1);
    }
    result = layout(kept, anchors, gap);
  }

  if (fixed.size > 0) {
    // B16 step 2: exact durations from the starts step 1 gave; home slots after the first one give way.
    const first = kept.findIndex((k) => fixed.has(k.slot.role));
    const templateMin = new Map<string, number>();
    kept.forEach((k, i) => {
      const minutes = fixed.get(k.slot.role);
      if (minutes !== undefined) {
        const { end_anchor: _end, ...slot } = k.slot;
        kept[i] = { ...k, slot: { ...slot, duration_minutes: minutes, min_minutes: minutes } };
      } else if (i > first && OVERRIDE_HOME_ROLES.has(k.slot.role)) {
        templateMin.set(k.slot.role, k.slot.min_minutes);
      }
    });
    result = layout(kept, anchors, gap);
    while ("role" in result) {
      const give = homeToGiveWay(kept, first, result.index, exempt);
      if (give === undefined) {
        // Name the overridden slot the failure follows from: the last one at or before the failing slot.
        let at = first;
        for (let i = first; i <= Math.min(result.index, kept.length - 1); i++) if (exempt.has(kept[i]!.slot.role)) at = i;
        throw overrideInfeasible(kept[at]!.slot.role, `${result.role}: ${result.reason}`);
      }
      const k = kept[give.index]!;
      if (give.action === "shrink") {
        kept[give.index] = { ...k, slot: { ...k.slot, min_minutes: OVERRIDE_HOME_MIN_MINUTES } };
      } else {
        dropped.push({ role: k.slot.role, reason: "no_time" });
        kept.splice(give.index, 1);
      }
      result = layout(kept, anchors, gap);
    }
    const laid = result;
    kept.forEach((k, i) => {
      const min = templateMin.get(k.slot.role);
      const role = k.slot.role;
      if (min === undefined || laid.end[i]! - laid.start[i]! >= min) return;
      if (!dropped.some((x) => x.role === role && x.reason === "shrunk")) dropped.push({ role, reason: "shrunk" });
    });
  }
  const { start, end, gaps } = result;

  const slots: Slot[] = kept.map(({ slot, node, place }, i) => {
    const hobby = place.hobby;
    const spec: ClipSpec = {
      category: slot.category,
      setting: place.setting,
      node,
      activity: hobby ? hobby.activity : slot.activity,
      time_of_day: timeOfDayAt(start[i]!),
      tags: hobby ? [...new Set([...slot.tags, ...hobby.tags])] : [...slot.tags],
    };
    return {
      slot_id: `d${day}-s${i + 1}`,
      role: slot.role,
      required: slot.required,
      window: [formatClock(start[i]!), formatClock(end[i]!)],
      spec,
    };
  });

  const transitions: SlotTransition[] = [];
  for (let i = 1; i < kept.length; i++) {
    if (kept[i - 1]!.node === kept[i]!.node) continue;
    transitions.push({ from_slot: slots[i - 1]!.slot_id, to_slot: slots[i]!.slot_id, minutes: gaps[i]! });
  }
  return { slots, transitions, dropped };
}
