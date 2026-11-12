import type { Clip, ClipCatalog } from "./clip.js";
import { parseClock } from "./clock.js";
import { bindingAccepts, hardFilterFailures, slotWindow, tripDayGroup, type HardFilter } from "./match/filters.js";
import { layoutTimeline } from "./layout.js";
import type { Exception, ScheduleDay } from "./schedule/types.js";
import { unassignClip, type Timeline } from "./timeline.js";
import { tripClipNode, tripPlaceOf } from "./world.js";

export const RULE_IDS = [
  "day_type",
  "slot_window",
  "wake_sleep",
  "no_overlap",
  "monotonic",
  "node_binding",
  "clip_unique",
  "day_group_single_day",
  "time_of_day",
  "exceptions",
  "trip",
] as const;
export type RuleId = (typeof RULE_IDS)[number];

export type Violation = { slot_id?: string; day_index?: number; detail: string };

export type VerifierReport = {
  pass: boolean;
  rules: Record<RuleId, { pass: boolean; violations: Violation[] }>;
  flags: { kind: "swap_point"; day_index: number; slot_ids: [string, string] }[];
  seams: { day_index: number; before_clip: string; after_clip: string; at_s: number }[];
  missing_clips: string[];
};

const REQUIRED_ROLES: Record<string, string[]> = {
  workday: ["work_am", "work_pm"],
  workday_late: ["work_am", "work_pm"],
  hobby_day: ["hobby"],
  home_day: ["chores"],
  leave_day: ["chores"],
  trip_day: ["lodging_morning", "outing"],
};

/** Hard-filter failures that map onto a rule; `tags` has no rule of its own. */
const FILTER_RULE: Partial<Record<HardFilter, RuleId>> = {
  category_setting_node: "node_binding",
  already_used: "clip_unique",
  day_group_other_day: "day_group_single_day",
  time_of_day: "time_of_day",
  does_not_fit: "slot_window",
  // trip_sequence is reported by the trip rule itself, with the expected date.
};

/** Rule-based checks of a filled timeline (§7.1, §7.3). Never throws on missing clips. */
export function verifyTimeline(t: Timeline, catalog: ClipCatalog): VerifierReport {
  const violations = Object.fromEntries(RULE_IDS.map((r) => [r, [] as Violation[]])) as Record<RuleId, Violation[]>;
  const add = (rule: RuleId, v: Violation) => violations[rule].push(v);

  const missing = new Set<string>();
  for (const day of t.schedule.days) {
    for (const slot of day.slots) {
      for (const id of t.slots[slot.slot_id]?.clip_ids ?? []) if (!catalog.getClip(id)) missing.add(id);
    }
  }
  const missing_clips = [...missing].sort();
  const needClips: RuleId[] = [
    "slot_window",
    "no_overlap",
    "monotonic",
    "node_binding",
    "clip_unique",
    "day_group_single_day",
    "time_of_day",
    "trip",
  ];
  if (missing_clips.length > 0) {
    for (const rule of needClips) add(rule, { detail: `clips missing from the catalog: ${missing_clips.join(", ")}` });
  }

  for (const day of t.schedule.days) {
    const d = day.day_index;
    const roles = new Set(day.slots.map((s) => s.role));
    for (const role of REQUIRED_ROLES[day.day_type] ?? []) {
      if (!roles.has(role)) add("day_type", { day_index: d, detail: `${day.day_type} has no ${role} slot` });
    }
    if ((day.day_type === "trip_day") !== (day.trip !== undefined)) {
      add("day_type", { day_index: d, detail: `${day.day_type} ${day.trip === undefined ? "has no" : "must not have a"} trip` });
    }
    if (day.day_type !== "workday" && day.day_type !== "workday_late") {
      for (const s of day.slots) {
        if (s.role.startsWith("work_")) add("day_type", { day_index: d, slot_id: s.slot_id, detail: `${day.day_type} has work slot ${s.role}` });
      }
    }

    const wake = parseClock(day.wake);
    const sleep = parseClock(day.sleep);
    for (const s of day.slots) {
      const [lo, hi] = slotWindow(s);
      if (lo < wake || hi > sleep) {
        add("wake_sleep", { day_index: d, slot_id: s.slot_id, detail: `window ${s.window[0]}-${s.window[1]} is outside ${day.wake}-${day.sleep}` });
      }
      if (s.required && (t.slots[s.slot_id]?.clip_ids.length ?? 0) === 0) {
        add("slot_window", { day_index: d, slot_id: s.slot_id, detail: `required slot ${s.role} has no clip` });
      }
    }

    checkExceptions(t, day, add);
    if (missing_clips.length === 0) checkTrip(t, catalog, day, add);
  }

  if (missing_clips.length === 0) {
    for (const day of t.schedule.days) {
      for (const slot of day.slots) {
        const entry = t.slots[slot.slot_id];
        for (const clipId of entry?.clip_ids ?? []) {
          const clip = catalog.getClip(clipId) as Clip;
          // R37: evaluate without the clip itself.
          const others = unassignClip(t, { slotId: slot.slot_id, clipId });
          const failed = hardFilterFailures(others, catalog, slot.slot_id, clip, entry?.relaxations ?? []);
          for (const f of failed) {
            const rule = FILTER_RULE[f];
            if (rule) add(rule, { day_index: day.day_index, slot_id: slot.slot_id, detail: `clip ${clipId}: ${f}` });
          }
        }
      }
    }
  }

  const flags: VerifierReport["flags"] = [];
  const seams: VerifierReport["seams"] = [];
  if (missing_clips.length === 0) {
    const placed = layoutTimeline(t, catalog);
    const windows = new Map(t.schedule.days.flatMap((day) => day.slots.map((s) => [s.slot_id, slotWindow(s)] as const)));
    const nodeOf = new Map(t.schedule.days.flatMap((day) => day.slots.map((s) => [s.slot_id, s.spec.node] as const)));
    let prev: (typeof placed)[number] | undefined;
    for (const p of placed) {
      const [lo, hi] = windows.get(p.slot_id)!;
      const base = (p.day_index - 1) * 86400;
      if (p.synthetic_start_s < base + lo * 60 - 1e-6 || p.synthetic_end_s > base + hi * 60 + 1e-6) {
        add("slot_window", { day_index: p.day_index, slot_id: p.slot_id, detail: `clip ${p.clip.clip_id} lies outside window` });
      }
      if (prev) {
        if (p.synthetic_start_s < prev.synthetic_start_s) {
          add("monotonic", { day_index: p.day_index, slot_id: p.slot_id, detail: `clip ${p.clip.clip_id} starts before ${prev.clip.clip_id}` });
        }
        if (p.synthetic_start_s < prev.synthetic_end_s - 1e-6 && prev.synthetic_start_s < p.synthetic_end_s - 1e-6) {
          add("no_overlap", { day_index: p.day_index, slot_id: p.slot_id, detail: `clip ${p.clip.clip_id} overlaps ${prev.clip.clip_id}` });
        }
        if (p.day_index === prev.day_index && p.clip.day_group !== prev.clip.day_group) {
          if (nodeOf.get(p.slot_id) === nodeOf.get(prev.slot_id)) {
            flags.push({ kind: "swap_point", day_index: p.day_index, slot_ids: [prev.slot_id, p.slot_id] });
          }
          seams.push({ day_index: p.day_index, before_clip: prev.clip.clip_id, after_clip: p.clip.clip_id, at_s: p.synthetic_start_s });
        }
      }
      prev = p;
    }

  }

  const rules = Object.fromEntries(
    RULE_IDS.map((r) => [r, { pass: violations[r].length === 0, violations: violations[r] }]),
  ) as VerifierReport["rules"];
  return { pass: RULE_IDS.every((r) => rules[r].pass), rules, flags, seams, missing_clips };
}

type Add = (rule: RuleId, v: Violation) => void;

const sameException = (a: Exception, b: Exception): boolean => JSON.stringify(a) === JSON.stringify(b);

/**
 * N5: on a trip day every slot is a place of the day's trip, and every clip carries
 * that place's trip label, comes from the sequence date at the day's index and is
 * no home or workplace clip.
 */
function checkTrip(t: Timeline, catalog: ClipCatalog, day: ScheduleDay, add: Add): void {
  if (day.trip === undefined) return;
  const d = day.day_index;
  const { node: trip, index } = day.trip;
  const want = tripDayGroup(t.world, day.trip);
  const fixed = Object.entries(t.world.nodes)
    .filter(([, def]) => def.kind === "home" || def.kind === "kitchen" || def.kind === "workplace")
    .map(([name, def]) => [name, def.kind === "workplace" ? "workplace" : "home"] as const);
  for (const slot of day.slots) {
    const v = (detail: string) => add("trip", { day_index: d, slot_id: slot.slot_id, detail });
    const place = tripPlaceOf(t.world, slot.spec.node);
    if (place?.trip !== trip) {
      const kind = fixed.find(([name]) => name === slot.spec.node)?.[1];
      v(`slot ${slot.role} uses ${kind !== undefined ? `${kind} ` : ""}node ${slot.spec.node}, not a place of trip ${trip}`);
    }
    for (const id of t.slots[slot.slot_id]?.clip_ids ?? []) {
      const clip = catalog.getClip(id) as Clip;
      const kinds = [...new Set(fixed.filter(([name]) => bindingAccepts(t.world, name, clip)).map(([, kind]) => kind))];
      if (kinds.length > 0) v(`clip ${id} is a ${kinds.join(" and ")} clip on a trip day`);
      else if (place !== undefined && clip.node !== tripClipNode(place.place)) {
        v(`clip ${id} has node ${clip.node}, expected ${tripClipNode(place.place)}`);
      }
      if (clip.day_group !== want) {
        v(`clip ${id} comes from ${clip.day_group}; trip day ${index + 1} of ${trip} uses ${want ?? "(no sequence date)"}`);
      }
    }
  }
}

function checkExceptions(t: Timeline, day: ScheduleDay, add: Add): void {
  const d = day.day_index;
  if (day.exceptions.length > 1) add("exceptions", { day_index: d, detail: `day has ${day.exceptions.length} exceptions` });
  const multi = day.exceptions.filter((e) => e.scope === "multi_day");
  if (day.trip !== undefined && multi.length === 0) add("exceptions", { day_index: d, detail: `trip day has no multi_day record` });
  for (const ex of multi) {
    const [s, e] = ex.day_range;
    const bad = (detail: string) => add("exceptions", { day_index: d, detail: `multi_day ${ex.type}: ${detail}` });
    if (ex.type !== "trip" || day.trip === undefined || ex.node !== day.trip.node) {
      bad(`record node ${ex.node ?? "(none)"} does not match the day's trip ${day.trip?.node ?? "(none)"}`);
      continue;
    }
    if (d < s || d > e || d - s !== day.trip.index) bad(`day_range ${s}-${e} does not put day ${d} at trip index ${day.trip.index}`);
    if (!Array.isArray(ex.replaces) || ex.replaces.length !== e - s + 1) bad(`replaces must list one day type per day of ${s}-${e}`);
    for (let k = s; k <= e; k++) {
      if (k === d) continue;
      const other = t.schedule.days.find((x) => x.day_index === k);
      if (!other?.exceptions.some((x) => sameException(x, ex))) bad(`day ${k} does not carry the same record`);
    }
  }
  for (const ex of day.exceptions) {
    if (ex.scope === "day" && (ex.day_range[0] !== d || ex.day_range[1] !== d)) {
      add("exceptions", { day_index: d, detail: `day exception ${ex.type} has day_range ${ex.day_range.join("-")}` });
    }
    if (ex.scope !== "slot") continue;
    if (ex.meal_place === undefined) {
      add("exceptions", { day_index: d, detail: `slot exception ${ex.type} has no meal_place` });
      continue;
    }
    const slot = day.slots.find((s) => s.role === ex.replaces);
    if (!slot) {
      add("exceptions", { day_index: d, detail: `exception ${ex.type} replaces ${ex.replaces}, which has no slot` });
      continue;
    }
    // Same resolution as scheduling: home and workplace are node names, the rest come from place_nodes.
    const place = ex.meal_place;
    const node = place === "home" || place === "workplace" ? place : t.world.meal_style.place_nodes[place];
    if (slot.spec.node !== node) {
      add("exceptions", {
        day_index: d,
        slot_id: slot.slot_id,
        detail: `exception ${ex.type} expects node ${node ?? `(none for ${place})`}, slot has ${slot.spec.node}`,
      });
    }
  }
}
