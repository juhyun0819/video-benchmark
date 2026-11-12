import { formatClock, parseClock } from "../clock.js";
import {
  clampToBounds,
  donorBoundViolations,
  selectDonors,
  type DayKind,
  type DayProfile,
  type DistributionSource,
  type DonorPick,
  type DonorAdjustment,
  type DonorViolation,
} from "../donor.js";
import { TimelineError } from "../errors.js";
import { gapQuantile, sampleGap, type GapTable } from "../gaps.js";
import { rngFor } from "../rng.js";
import type { World } from "../world.js";
import { decideDay, decideDays, effectiveMealStyle, mealPlaceUsable } from "./decisions.js";
import { realizedOf } from "./realized.js";
import type {
  AnchorName,
  DayAnchors,
  DayDecisions,
  Exception,
  ExceptionCatalog,
  Schedule,
  ScheduleDay,
  ScheduleDefaults,
  TemplateName,
  Templates,
} from "./types.js";
import { resolveDayWindows } from "./windows.js";

/** schedule.v2: rev3 (rest_day meals, donor/day acceptance, realized). Older schedules do not parse (G25). */
export const SCHEDULE_GENERATOR_VERSION = "schedule.v2";
/** Regeneration attempts after attempt 0 fails (D11). */
const MAX_RETRIES = 5;

export type ScheduleInput = {
  world: World;
  templates: Templates;
  gaps: GapTable;
  exceptions: ExceptionCatalog;
  source: DistributionSource;
  boundNodes: Set<string>;
  defaults: ScheduleDefaults;
};

type Donors = Schedule["donors"];

/** Day-level jitter σ in minutes: lerp from routine_strength 0 to 1 (§5.2 step 4, R5). */
function daySigma(input: ScheduleInput): number {
  const { routine_strength_0: s0, routine_strength_1: s1 } = input.defaults.day_level_sigma;
  return s0 + (s1 - s0) * input.world.dials.routine_strength;
}

/** Redraws per anchor after its first draw before it falls back to a clamp (E6, G13). */
const MAX_JITTER_REDRAWS = 5;
/** Shortest workday the E5 work_end bound allows, in minutes after work_start. */
const WORK_MIN_SPAN = 240;

/** Template durations the bounds depend on: the workday lunch (R22) and the weekend hobby at sport (R25). */
function boundMinutes(templates: Templates): { lunchMinutes: number; hobbyMinutes: number } {
  return {
    lunchMinutes: templates.day_types.workday.find((s) => s.anchor === "lunch")?.duration_minutes ?? 0,
    hobbyMinutes: templates.day_types.hobby_day.find((s) => s.anchor === "sport")?.duration_minutes ?? 0,
  };
}

type JitterItem = AnchorName | "sleep";

/**
 * Anchors in dependency order (G13): the primaries wake, sleep and work_start
 * first, then work_end | work_start, breakfast | wake, lunch | work window,
 * dinner | sleep and sport | sleep.
 */
const JITTER_ORDER: readonly JitterItem[] = ["wake", "sleep", "work_start", "work_end", "breakfast", "lunch", "dinner", "sport"];

/** Anchors whose bounds are checked per template (G12); the others are jittered once and left alone. */
const CHECKED: Record<TemplateName, ReadonlySet<JitterItem>> = {
  workday: new Set(["wake", "sleep", "work_start", "work_end", "lunch", "dinner"]),
  hobby_day: new Set(["wake", "sleep", "breakfast", "lunch", "dinner", "sport"]),
  home_day: new Set(["wake", "sleep", "breakfast", "lunch", "dinner"]),
};

function anchorsOf(p: DayProfile): DayAnchors {
  const a: DayAnchors = { wake: p.wake, sleep: p.sleep, breakfast: p.breakfast, lunch: p.lunch, dinner: p.dinner };
  if (p.work_start !== undefined) a.work_start = p.work_start;
  if (p.work_end !== undefined) a.work_end = p.work_end;
  if (p.sport !== undefined) a.sport = p.sport;
  return a;
}

/**
 * Donor times jittered per day and rounded (E6, G12, G13). In dependency
 * order, each anchor the template uses is redrawn (keys "jitter", item, k for
 * k = 1..5) while it violates its own E5 bound given the anchors already
 * accepted; after that it is clamped to the bound and the clamp is recorded.
 * On workdays work_end and dinner are also redrawn while the day's required
 * slots do not lay out with its own gap draws (G21); if they still do not,
 * work_end is pulled earlier to the latest integer minute (≥ work_start + 240)
 * at which they do, recorded like a clamp.
 */
function dayAnchors(
  input: ScheduleInput,
  profile: DayProfile,
  decisions: DayDecisions,
  attempt: number,
): { anchors: DayAnchors; adjustments: DonorAdjustment[]; redraws: number } {
  const { seed } = input.world;
  const d = decisions.day_index;
  const sigma = daySigma(input);
  const workday = decisions.template === "workday";
  const kind: DayKind = workday ? "weekday" : "weekend";
  const checked = CHECKED[decisions.template];
  const { lunchMinutes, hobbyMinutes } = boundMinutes(input.templates);
  const base: Partial<Record<JitterItem, number>> = {
    wake: profile.wake,
    sleep: profile.sleep,
    breakfast: profile.breakfast,
    lunch: profile.lunch,
    dinner: profile.dinner,
    sport: profile.sport ?? parseClock(input.defaults.weekend.sports_start.p50),
  };
  if (workday) {
    base.work_start = profile.work_start;
    base.work_end = profile.work_end;
  }
  const jitter = (item: JitterItem, value: number, k: number): number =>
    Math.round(value + rngFor(seed, "day", d, attempt, "jitter", item, k).normal(0, sigma));

  // Every item gets its first draw up front; later items do not affect earlier items' bounds.
  const p: DayProfile = { wake: 0, sleep: 0, breakfast: 0, lunch: 0, dinner: 0, meal_loc: {} };
  for (const item of JITTER_ORDER) {
    const value = base[item];
    if (value !== undefined) p[item] = jitter(item, value, 0);
  }
  const violation = (item: JitterItem) =>
    donorBoundViolations(kind, p, lunchMinutes, hobbyMinutes).find((v) => v.item === item);
  // G21: the workday's required slots must lay out with the day's own gap draws.
  const fits = (): boolean => !workday || layoutFits(input, anchorsOf(p), decisions, attempt);
  const fitChecked = (item: JitterItem) => workday && (item === "work_end" || item === "dinner");
  const rejected = (item: JitterItem) => violation(item) !== undefined || (fitChecked(item) && !fits());

  const adjustments: DonorAdjustment[] = [];
  let redraws = 0;
  for (const item of JITTER_ORDER) {
    const value = base[item];
    if (value === undefined || !checked.has(item)) continue;
    for (let k = 1; k <= MAX_JITTER_REDRAWS && rejected(item); k++) {
      p[item] = jitter(item, value, k);
      redraws += 1;
    }
    const v = violation(item);
    if (v) {
      p[item] = v.bound;
      adjustments.push({ item, from: v.value, to: v.bound });
    }
  }
  if (workday && p.work_start !== undefined && p.work_end !== undefined && !fits()) {
    const from = p.work_end;
    for (let to = from - 1; to >= p.work_start + WORK_MIN_SPAN; to--) {
      p.work_end = to;
      if (fits()) {
        adjustments.push({ item: "work_end", from, to });
        return { anchors: anchorsOf(p), adjustments, redraws };
      }
    }
    // Nothing fits: keep the drawn value and let the layout fail into an attempt retry.
    p.work_end = from;
  }
  return { anchors: anchorsOf(p), adjustments, redraws };
}

/** True when the day lays out (optional slots may be reduced); false when a required slot cannot fit. */
function layoutFits(input: ScheduleInput, anchors: DayAnchors, decisions: DayDecisions, attempt: number): boolean {
  try {
    layoutDay(input, anchors, decisions, attempt);
    return true;
  } catch (e) {
    if (isInfeasible(e)) return false;
    throw e;
  }
}

const isInfeasible = (e: unknown): e is TimelineError =>
  e instanceof TimelineError && e.code === "SCHEDULE_INFEASIBLE";

/**
 * E5 template check for a weekday donor: the workday's required slots (a
 * required $meal eaten at home) lay out with the donor's raw times, p90
 * transition gaps and sleep moved σ earlier, so daily jitter keeps slack (G11).
 */
function requiredWorkdaySlotsFit(input: ScheduleInput, profile: DayProfile): boolean {
  const decisions: DayDecisions = {
    day_index: 0,
    weekday: "Mon",
    day_type: "workday",
    template: "workday",
    meal_place: { breakfast: "home", lunch: "home", dinner: "home" },
    optional: {},
    exceptions: [],
  };
  const required = input.templates.day_types.workday.filter((s) => s.required);
  try {
    const anchors = { ...anchorsOf(profile), sleep: profile.sleep - daySigma(input) };
    resolveDayWindows(required, anchors, decisions, input.world, (from, to) => gapQuantile(input.gaps, from, to, "p90"));
    return true;
  } catch (e) {
    if (isInfeasible(e)) return false;
    throw e;
  }
}

/**
 * G9: a final weekday pick that still fails the required-slots check gets the
 * latest integer work_end (searched down to work_start + 240) at which the
 * required slots fit, recorded in its adjustments. Unchanged when none fits.
 */
function repairWorkEnd(input: ScheduleInput, pick: DonorPick): DonorPick {
  const { profile } = pick;
  if (profile.work_start === undefined || profile.work_end === undefined) return pick;
  if (requiredWorkdaySlotsFit(input, profile)) return pick;
  const from = profile.work_end;
  const floor = Math.ceil(profile.work_start + WORK_MIN_SPAN);
  for (let to = Math.ceil(from) - 1; to >= floor; to--) {
    const candidate = { ...profile, work_end: to };
    if (requiredWorkdaySlotsFit(input, candidate)) {
      return { ...pick, profile: candidate, adjustments: [...pick.adjustments, { item: "work_end", from, to }] };
    }
  }
  return pick;
}

type DayLayout = ReturnType<typeof resolveDayWindows>;

/**
 * The day's template laid out with its reduce order and the day's gap draws.
 * The base layout and every slot-exception re-layout go through here (G14).
 */
function layoutDay(input: ScheduleInput, anchors: DayAnchors, decisions: DayDecisions, attempt: number): DayLayout {
  const { world, gaps, templates } = input;
  const d = decisions.day_index;
  return resolveDayWindows(
    templates.day_types[decisions.template],
    anchors,
    decisions,
    world,
    (from, to, i) => sampleGap(gaps, from, to, rngFor(world.seed, "day", d, attempt, "gap", i)),
    templates.reduce_order?.[decisions.template],
  );
}

type SlotExceptionPlacement = { layout: DayLayout; exceptions: Exception[]; eligible: boolean; candidates: number };

/**
 * §5.2 step 8 (E1, E3, G14): after windows and reductions. A candidate is a
 * catalog slot exception for the day type whose meal slot is still kept and
 * whose place is usable (D4), and for which the day re-laid out with that
 * place keeps the replaced slot and every kept required slot. A day without a
 * day exception and with a candidate is eligible: Bernoulli(exception_rate.slot)
 * decides, the candidate is picked uniformly, and its re-layout becomes the day.
 */
function placeSlotException(
  input: ScheduleInput,
  anchors: DayAnchors,
  decisions: DayDecisions,
  attempt: number,
  base: DayLayout,
): SlotExceptionPlacement {
  const none: SlotExceptionPlacement = { layout: base, exceptions: decisions.exceptions, eligible: false, candidates: 0 };
  if (decisions.exceptions.length > 0) return none;
  const { world, boundNodes } = input;
  const kept = new Set(base.slots.map((s) => s.role));
  const required = base.slots.filter((s) => s.required).map((s) => s.role);

  const feasible: { entry: ExceptionCatalog["slot"][number]; layout: DayLayout }[] = [];
  for (const entry of input.exceptions.slot) {
    const role = entry.replaces_role;
    if (!entry.day_types.includes(decisions.day_type) || !kept.has(role)) continue;
    if (!mealPlaceUsable(world, entry.meal_place, boundNodes)) continue;
    const meal_place = { ...decisions.meal_place, [role]: entry.meal_place };
    let layout: DayLayout;
    try {
      layout = layoutDay(input, anchors, { ...decisions, meal_place }, attempt);
    } catch (e) {
      if (isInfeasible(e)) continue;
      throw e;
    }
    const roles = new Set(layout.slots.map((s) => s.role));
    if (roles.has(role) && required.every((r) => roles.has(r))) feasible.push({ entry, layout });
  }
  if (feasible.length === 0) return none;

  const d = decisions.day_index;
  const placed: SlotExceptionPlacement = { ...none, eligible: true, candidates: feasible.length };
  if (!rngFor(world.seed, "day", d, attempt, "slot_exception").bernoulli(world.dials.exception_rate.slot)) return placed;
  const { entry, layout } = feasible[rngFor(world.seed, "day", d, attempt, "slot_exception_pick").int(0, feasible.length)]!;
  const routine = decisions.meal_place[entry.replaces_role];
  const exception: Exception = {
    scope: "slot",
    type: entry.type,
    replaces: entry.replaces_role,
    day_range: [d, d],
    meal_place: entry.meal_place,
    // The replaced slot was kept, so its routine place is a place, never "skip".
    ...(routine !== undefined && routine !== "skip" ? { routine_place: routine } : {}),
  };
  return { ...placed, layout, exceptions: [exception] };
}

function buildDay(input: ScheduleInput, donors: Donors, decisions: DayDecisions, attempt: number): ScheduleDay {
  const d = decisions.day_index;
  const profile = decisions.template === "workday" ? donors.weekday.profile : donors.weekend.profile;
  const { anchors, adjustments, redraws } = dayAnchors(input, profile, decisions, attempt);
  const base = layoutDay(input, anchors, decisions, attempt);
  const { layout, exceptions, eligible, candidates } = placeSlotException(input, anchors, decisions, attempt, base);
  return {
    day_index: d,
    weekday: decisions.weekday,
    day_type: decisions.day_type,
    attempt,
    wake: formatClock(anchors.wake),
    sleep: formatClock(anchors.sleep),
    exceptions,
    slot_exception_eligible: eligible,
    slot_exception_candidates: candidates,
    slots: layout.slots,
    dropped_slots: layout.dropped,
    transitions: layout.transitions,
    anchor_adjustments: adjustments,
    jitter_redraws: redraws,
  };
}

/** One day at a given regeneration attempt; attempt 0 matches generateSchedule's first try. */
export function generateDay(input: ScheduleInput, donors: Donors, dayIndex: number, attempt: number): ScheduleDay {
  const opts = { boundNodes: input.boundNodes, defaults: input.defaults };
  const decisions = decideDay(input.world, input.exceptions, opts, dayIndex, attempt);
  return buildDay(input, donors, decisions, attempt);
}

/**
 * schedule.json for one world (§4.3, §5.2). A day that is infeasible is
 * regenerated with attempts 1..5; after the 5th the error names the day.
 */
export function generateSchedule(input: ScheduleInput): Schedule {
  const { world, templates, gaps, exceptions, source, boundNodes, defaults } = input;
  const requiredSlots: DonorViolation[] = [{ item: "required_slots", bound: 0, value: 1 }];
  // G7/G9: a weekday pick that still fails this check after its clamps has
  // work_end pulled earlier (repairWorkEnd); if nothing fits, its days rely on
  // attempt retries and finally SCHEDULE_INFEASIBLE.
  const picked = selectDonors(world.attributes, source, {
    seed: world.seed,
    allowCountryFallback: world.allow_country_fallback,
    ...boundMinutes(templates),
    accept: (kind, profile) => (kind === "weekday" && !requiredWorkdaySlotsFit(input, profile) ? requiredSlots : []),
  });
  const donors: Donors = {
    weekday: repairWorkEnd(input, picked.weekday),
    weekend: picked.weekend,
    country_fallback: picked.country_fallback,
    weekend_redraws: picked.weekend_redraws,
  };

  const days = decideDays(world, exceptions, { boundNodes, defaults }).map((decisions) => {
    let last: TimelineError;
    try {
      return buildDay(input, donors, decisions, 0);
    } catch (e) {
      if (!isInfeasible(e)) throw e;
      last = e;
    }
    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
      try {
        return generateDay(input, donors, decisions.day_index, attempt);
      } catch (e) {
        if (!isInfeasible(e)) throw e;
        last = e;
      }
    }
    const d = decisions.day_index;
    throw new TimelineError(
      "SCHEDULE_INFEASIBLE",
      `day ${d} is infeasible after ${MAX_RETRIES + 1} attempts; last: ${last.message}`,
      { ...last.details, day: d, attempts: MAX_RETRIES + 1 },
    );
  });

  return {
    person_id: world.person_id,
    generator_version: SCHEDULE_GENERATOR_VERSION,
    template_version: templates.version,
    gap_table_version: gaps.version,
    gap_table_reviewed: gaps.reviewed,
    exception_catalog_version: exceptions.version,
    distribution_source: source.id,
    donors,
    meal_style_effective: effectiveMealStyle(world, boundNodes),
    realized: realizedOf(days, world, templates),
    days,
  };
}
