import type { Clip, ClipCatalog } from "./clip.js";
import { parseClock } from "./clock.js";
import { blockSpans, transitionBlocks, type BlockPlacement } from "./blocks.js";
import { bindingAccepts, hardFilterFailures, slotWindow, tripDayGroup, type HardFilter } from "./match/filters.js";
import { layoutTimeline } from "./layout.js";
import type { Exception, ScheduleDay, Slot } from "./schedule/types.js";
import { unassignClip, type Timeline } from "./timeline.js";
import { TRIP_PLACES, tripClipNode, tripPlaceOf, type World } from "./world.js";

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
  /** `note` is set only in block mode, on a rule that is skipped there. */
  rules: Record<RuleId, { pass: boolean; violations: Violation[]; note?: string }>;
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

/**
 * B4: verify block placements (layoutBlocks) instead of t.slots. The placements are the source of truth
 * for clips; t.slots clip ids are not read. time_of_day is skipped (reported as pass with a note);
 * slot_window checks only that each placement lies inside its slot window or transition span, so empty
 * blocks are allowed; clip_unique, day_group_single_day, node_binding (place only, see placeAccepts)
 * and trip are checked over the placements, a transition counting as its day. Other rules are unchanged.
 */
export type VerifyOptions = { mode: "block"; placements: readonly BlockPlacement[] };

const TIME_OF_DAY_SKIPPED = "skipped in block mode: real and synthetic clock times may differ";

/** Rule-based checks of a filled timeline (§7.1, §7.3). Never throws on missing clips. */
export function verifyTimeline(t: Timeline, catalog: ClipCatalog, opts?: VerifyOptions): VerifierReport {
  const violations = Object.fromEntries(RULE_IDS.map((r) => [r, [] as Violation[]])) as Record<RuleId, Violation[]>;
  const add = (rule: RuleId, v: Violation) => violations[rule].push(v);
  const block = opts?.mode === "block" ? opts.placements : undefined;

  const missing = new Set<string>();
  if (block !== undefined) {
    for (const p of block) if (!catalog.getClip(p.clip.clip_id)) missing.add(p.clip.clip_id);
  } else {
    for (const day of t.schedule.days) {
      for (const slot of day.slots) {
        for (const id of t.slots[slot.slot_id]?.clip_ids ?? []) if (!catalog.getClip(id)) missing.add(id);
      }
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

  // N6: a dropped required slot needs an allowance for its role, spent on at most max_days days.
  const droppedDays = new Map<string, number[]>();
  for (const day of block !== undefined ? [] : t.schedule.days) {
    for (const s of day.slots) {
      if (!t.slots[s.slot_id]?.relaxations.includes("dropped_required")) continue;
      const days = droppedDays.get(s.role) ?? [];
      if (!days.includes(day.day_index)) days.push(day.day_index);
      droppedDays.set(s.role, days);
    }
  }
  for (const [role, days] of droppedDays) {
    const max = t.world.dials.missing_required_allowance.find((a) => a.role === role)?.max_days ?? 0;
    if (days.length > max) {
      add("slot_window", { day_index: days[max], detail: `required ${role} is empty on ${days.length} days (days ${days.join(", ")}); allowance is ${max}` });
    }
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
      if (block === undefined && s.required && (t.slots[s.slot_id]?.clip_ids.length ?? 0) === 0 && !t.slots[s.slot_id]?.relaxations.includes("dropped_required")) {
        add("slot_window", { day_index: d, slot_id: s.slot_id, detail: `required slot ${s.role} has no clip` });
      }
    }

    checkExceptions(t, day, add);
    if (missing_clips.length === 0) checkTrip(t, catalog, day, add, block);
  }

  if (missing_clips.length === 0 && block !== undefined) {
    checkPlacements(t, block, add);
  } else if (missing_clips.length === 0) {
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
    const placed = block ?? layoutTimeline(t, catalog);
    // Seconds since the day's 00:00; in block mode transitions d<N>-t<k> are blocks too.
    const windows = block !== undefined
      ? new Map([...blockSpans(t)].map(([id, b]) => [id, { day_index: b.day_index, span: b.span }] as const))
      : new Map(t.schedule.days.flatMap((day) => day.slots.map((s) => {
          const [lo, hi] = slotWindow(s);
          return [s.slot_id, { day_index: day.day_index, span: [lo * 60, hi * 60] as [number, number] }] as const;
        })));
    const nodeOf = new Map(t.schedule.days.flatMap((day) => day.slots.map((s) => [s.slot_id, s.spec.node] as const)));
    let prev: (typeof placed)[number] | undefined;
    for (const p of placed) {
      const w = windows.get(p.slot_id);
      const base = (p.day_index - 1) * 86400;
      if (w === undefined || w.day_index !== p.day_index) {
        add("slot_window", { day_index: p.day_index, slot_id: p.slot_id, detail: `clip ${p.clip.clip_id} is placed in unknown block ${p.slot_id} of day ${p.day_index}` });
      } else if (p.synthetic_start_s < base + w.span[0] - 1e-6 || p.synthetic_end_s > base + w.span[1] + 1e-6) {
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
          if (nodeOf.get(p.slot_id) !== undefined && nodeOf.get(p.slot_id) === nodeOf.get(prev.slot_id)) {
            flags.push({ kind: "swap_point", day_index: p.day_index, slot_ids: [prev.slot_id, p.slot_id] });
          }
          seams.push({ day_index: p.day_index, before_clip: prev.clip.clip_id, after_clip: p.clip.clip_id, at_s: p.synthetic_start_s });
        }
      }
      prev = p;
    }

  }

  const rules = Object.fromEntries(
    RULE_IDS.map((r) => [
      r,
      block !== undefined && r === "time_of_day"
        ? { pass: true, violations: [], note: TIME_OF_DAY_SKIPPED }
        : { pass: violations[r].length === 0, violations: violations[r] },
    ]),
  ) as VerifierReport["rules"];
  return { pass: RULE_IDS.every((r) => rules[r].pass), rules, flags, seams, missing_clips };
}

type Add = (rule: RuleId, v: Violation) => void;

const sameException = (a: Exception, b: Exception): boolean => JSON.stringify(a) === JSON.stringify(b);

const videoKey = (c: Clip) => `${c.source.dataset}:${c.source.video_id}`;

const MEAL_ROLES = new Set(["breakfast", "lunch", "dinner"]);

/**
 * Person sources of the world's fixed (home and workplace) bindings: the bound NextMe wearer.
 * Empty when none of them names one.
 */
function boundPersons(world: World): Set<string> {
  const out = new Set<string>();
  for (const [name, def] of Object.entries(world.nodes)) {
    if (def.kind !== "home" && def.kind !== "workplace") continue;
    const ps = world.node_binding[name]?.person_source;
    if (ps !== undefined) out.add(ps);
  }
  return out;
}

/**
 * B4 node_binding in block mode checks the clip's place only, never category or setting:
 * - a meal slot (breakfast, lunch, dinner) takes a clip of any bound place of the world, or on a
 *   trip day of any place of the day's trip (meals use the real place);
 * - another slot takes a clip its node's binding accepts;
 * - a transition takes a transit clip: one the `transit` binding accepts when the world has one,
 *   otherwise a clip labeled node `transit` from the bound person (boundPersons).
 * Returns the violation detail, or undefined when the place is accepted.
 */
function placeAccepts(world: World, day: ScheduleDay, slot: Slot | undefined, clip: Clip): string | undefined {
  if (slot === undefined) {
    const persons = boundPersons(world);
    const ok = world.node_binding["transit"] !== undefined
      ? bindingAccepts(world, "transit", clip)
      : clip.node === "transit" && (persons.size === 0 || persons.has(clip.person_source));
    return ok ? undefined : `clip ${clip.clip_id}: not a transit clip of the bound person`;
  }
  if (MEAL_ROLES.has(slot.role)) {
    const places = day.trip !== undefined
      ? TRIP_PLACES.map((p) => `${day.trip!.node}:${p}`)
      : Object.entries(world.nodes).filter(([, def]) => def.kind !== "trip").map(([name]) => name);
    return places.some((name) => bindingAccepts(world, name, clip)) ? undefined : `clip ${clip.clip_id}: not accepted by any bound place`;
  }
  return bindingAccepts(world, slot.spec.node, clip) ? undefined : `clip ${clip.clip_id}: not accepted by the binding of node ${slot.spec.node}`;
}

/**
 * B4 block mode: node_binding (placeAccepts), clip_unique (a clip placed once, no footage of a video
 * placed twice) and day_group_single_day (a transition counts as its day) over the placements.
 */
function checkPlacements(t: Timeline, placements: readonly BlockPlacement[], add: Add): void {
  const days = new Map(t.schedule.days.map((day) => [day.day_index, day] as const));
  const slots = new Map(t.schedule.days.flatMap((day) => day.slots.map((s) => [s.slot_id, s] as const)));
  const transitions = new Set(t.schedule.days.flatMap((day) => transitionBlocks(day).map((x) => x.transition_id)));
  const groupDays = new Map<string, Set<number>>();
  for (const p of placements) groupDays.set(p.clip.day_group, (groupDays.get(p.clip.day_group) ?? new Set()).add(p.day_index));
  placements.forEach((p, i) => {
    const at = { day_index: p.day_index, slot_id: p.slot_id };
    const id = p.clip.clip_id;
    const slot = slots.get(p.slot_id);
    const day = days.get(p.day_index);
    // Unknown blocks are reported by slot_window.
    if (day !== undefined && (slot !== undefined || transitions.has(p.slot_id))) {
      const bad = placeAccepts(t.world, day, slot, p.clip);
      if (bad !== undefined) add("node_binding", { ...at, detail: bad });
    }
    const reused = placements.some(
      (q, j) =>
        j !== i &&
        (q.clip.clip_id === id ||
          (videoKey(q.clip) === videoKey(p.clip) && q.source_start_s < p.source_end_s && p.source_start_s < q.source_end_s)),
    );
    if (reused) add("clip_unique", { ...at, detail: `clip ${id}: already_used` });
    if ([...groupDays.get(p.clip.day_group)!].some((d) => d !== p.day_index)) {
      add("day_group_single_day", { ...at, detail: `clip ${id}: day_group_other_day` });
    }
  });
}

/**
 * N5: on a trip day every slot is a place of the day's trip, and every clip carries
 * that place's trip label, comes from the sequence date at the day's index and is
 * no home or workplace clip. In block mode the clips are the day's placements; a
 * transition's clips get the same checks except the place label.
 */
function checkTrip(t: Timeline, catalog: ClipCatalog, day: ScheduleDay, add: Add, placements?: readonly BlockPlacement[]): void {
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
    const clips = placements !== undefined
      ? placements.filter((p) => p.slot_id === slot.slot_id).map((p) => p.clip)
      : (t.slots[slot.slot_id]?.clip_ids ?? []).map((id) => catalog.getClip(id) as Clip);
    for (const clip of clips) checkClip(clip, place, v);
  }
  if (placements === undefined) return;
  const slotIds = new Set(day.slots.map((s) => s.slot_id));
  for (const p of placements) {
    if (p.day_index !== d || slotIds.has(p.slot_id)) continue;
    checkClip(p.clip, undefined, (detail) => add("trip", { day_index: d, slot_id: p.slot_id, detail }));
  }

  function checkClip(clip: Clip, place: ReturnType<typeof tripPlaceOf>, v: (detail: string) => void): void {
    const id = clip.clip_id;
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
