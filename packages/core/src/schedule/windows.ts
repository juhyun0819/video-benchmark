import type { ClipSpec, Setting } from "../clip.js";
import { formatClock, timeOfDayAt } from "../clock.js";
import { TimelineError } from "../errors.js";
import { placeKindOf, type PlaceKind } from "../gaps.js";
import type { NodeDef, World } from "../world.js";
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

type Kept = { slot: TemplateSlot; node: string; def: NodeDef };

function settingOf(def: NodeDef): Setting {
  switch (def.kind) {
    case "home":
    case "kitchen":
      return "home_indoor";
    case "workplace":
      return "venue_indoor";
    case "hobby":
    case "frequent_place":
      return def.setting;
  }
}

/** World node a template slot points at today; null for a skipped meal, undefined when it resolves to nothing. */
function resolveNode(slot: TemplateSlot, decisions: DayDecisions, world: World): string | null | undefined {
  if (slot.node === "$hobby") return decisions.hobby_node;
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
    return prev.node === k.node ? 0 : gap(placeKindOf(prev.def.kind), placeKindOf(k.def.kind), i);
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
 */
function slotToDrop(kept: Kept[], failing: number): number {
  for (let i = failing; i >= 0; i--) if (!kept[i]!.slot.required) return i;
  for (let i = kept.length - 1; i > failing; i--) if (!kept[i]!.slot.required) return i;
  return -1;
}

/**
 * E4: applies the first unused reduce step, in order, whose optional slot is
 * still kept and can affect the failing slot (G10: a slot's layout depends only
 * on itself, the slots before it and the next one, so a drop may target up to
 * failing + 1; a shrink only up to failing). A shrink needs a duration above
 * min_minutes. Returns false when no step applies.
 */
function applyReduceStep(
  kept: Kept[],
  steps: readonly ReduceStep[],
  used: boolean[],
  failing: number,
  dropped: DroppedSlot[],
): boolean {
  for (let si = 0; si < steps.length; si++) {
    if (used[si]) continue;
    const step = steps[si]!;
    const i = kept.findIndex((k) => k.slot.role === step.role && !k.slot.required);
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

/**
 * Lays one day's template out in time (§5.2 step 5). Slots are dropped for
 * "skip" meals, absent optional keys and unresolved optional nodes. While the
 * windows do not fit, the day type's reduce order is applied step by step (E4);
 * a failure no remaining step can affect drops an optional slot by R24 (R21).
 * Every change is recorded. A sport-anchored slot is first put in anchor
 * order among the meals (G20).
 * Required slots are never shrunk or dropped; when no optional slot is left it throws.
 */
export function resolveDayWindows(
  template: TemplateSlot[],
  anchors: DayAnchors,
  decisions: DayDecisions,
  world: World,
  gap: GapFn,
  reduceOrder: readonly ReduceStep[] = [],
): { slots: Slot[]; transitions: SlotTransition[]; dropped: DroppedSlot[] } {
  const day = decisions.day_index;
  const infeasible = (role: string, reason: string): TimelineError =>
    new TimelineError("SCHEDULE_INFEASIBLE", `day ${day}: slot ${role} is infeasible (${reason})`, { day, role, reason });

  const kept: Kept[] = [];
  for (const slot of orderBySportAnchor(template, anchors)) {
    if (slot.optional_key !== undefined && decisions.optional[slot.optional_key] === false) continue;
    const node = resolveNode(slot, decisions, world);
    if (node === null) continue;
    const def = node === undefined ? undefined : world.nodes[node];
    if (node === undefined || def === undefined) {
      if (slot.required) throw infeasible(slot.role, `node ${slot.node} resolves to nothing`);
      continue;
    }
    for (const name of [slot.anchor, slot.end_anchor]) {
      if (name !== undefined && anchors[name] === undefined) throw infeasible(slot.role, `anchor ${name} is missing`);
    }
    kept.push({ slot, node, def });
  }
  if (kept.length === 0) throw infeasible("-", "no slots");

  const dropped: DroppedSlot[] = [];
  const used = reduceOrder.map(() => false);
  let result = layout(kept, anchors, gap);
  while ("role" in result) {
    if (!applyReduceStep(kept, reduceOrder, used, result.index, dropped)) {
      const victim = slotToDrop(kept, result.index);
      if (victim < 0) throw infeasible(result.role, result.reason);
      dropped.push({ role: kept[victim]!.slot.role, reason: "no_time" });
      kept.splice(victim, 1);
    }
    result = layout(kept, anchors, gap);
  }
  const { start, end, gaps } = result;

  const slots: Slot[] = kept.map(({ slot, node, def }, i) => {
    const hobby = def.kind === "hobby" ? def : undefined;
    const spec: ClipSpec = {
      category: slot.category,
      setting: settingOf(def),
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
