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
import { medianGap, sampleGap, type GapTable } from "../gaps.js";
import { rngFor } from "../rng.js";
import type { World } from "../world.js";
import { decideDay, decideDays, effectiveMealStyle } from "./decisions.js";
import type {
  AnchorName,
  DayAnchors,
  DayDecisions,
  ExceptionCatalog,
  Schedule,
  ScheduleDay,
  ScheduleDefaults,
  Templates,
} from "./types.js";
import { resolveDayWindows } from "./windows.js";

export const SCHEDULE_GENERATOR_VERSION = "schedule.v1";
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

/** Jitter vector redraws after the first draw before the day falls back to clamping (E6). */
const MAX_JITTER_REDRAWS = 5;

/** Template durations the bounds depend on: the workday lunch (R22) and the weekend hobby at sport (R25). */
function boundMinutes(templates: Templates): { lunchMinutes: number; hobbyMinutes: number } {
  return {
    lunchMinutes: templates.day_types.workday.find((s) => s.anchor === "lunch")?.duration_minutes ?? 0,
    hobbyMinutes: templates.day_types.hobby_day.find((s) => s.anchor === "sport")?.duration_minutes ?? 0,
  };
}

function anchorsOf(p: DayProfile): DayAnchors {
  const a: DayAnchors = { wake: p.wake, sleep: p.sleep, breakfast: p.breakfast, lunch: p.lunch, dinner: p.dinner };
  if (p.work_start !== undefined) a.work_start = p.work_start;
  if (p.work_end !== undefined) a.work_end = p.work_end;
  if (p.sport !== undefined) a.sport = p.sport;
  return a;
}

/**
 * Donor times jittered per day and rounded (E6). The whole jitter vector is
 * redrawn (keys "jitter", k for k = 0..5) until the E5 bounds hold; after
 * that the last draw is clamped like a donor and each clamp is recorded.
 */
function dayAnchors(
  input: ScheduleInput,
  profile: DayProfile,
  decisions: DayDecisions,
  attempt: number,
): { anchors: DayAnchors; adjustments: DonorAdjustment[] } {
  const { seed } = input.world;
  const d = decisions.day_index;
  const sigma = daySigma(input);
  const workday = decisions.template === "workday";
  const kind: DayKind = workday ? "weekday" : "weekend";
  const { lunchMinutes, hobbyMinutes } = boundMinutes(input.templates);
  const sport = profile.sport ?? parseClock(input.defaults.weekend.sports_start.p50);

  const draw = (k: number): DayProfile => {
    const jitter = (name: AnchorName | "sleep", value: number): number =>
      Math.round(value + rngFor(seed, "day", d, attempt, "jitter", k, name).normal(0, sigma));
    const p: DayProfile = {
      wake: jitter("wake", profile.wake),
      sleep: jitter("sleep", profile.sleep),
      breakfast: jitter("breakfast", profile.breakfast),
      lunch: jitter("lunch", profile.lunch),
      dinner: jitter("dinner", profile.dinner),
      sport: jitter("sport", sport),
      meal_loc: {},
    };
    if (workday && profile.work_start !== undefined) p.work_start = jitter("work_start", profile.work_start);
    if (workday && profile.work_end !== undefined) p.work_end = jitter("work_end", profile.work_end);
    return p;
  };
  const violates = (p: DayProfile) => donorBoundViolations(kind, p, lunchMinutes, hobbyMinutes).length > 0;

  let p = draw(0);
  for (let k = 1; k <= MAX_JITTER_REDRAWS && violates(p); k++) p = draw(k);
  const adjustments = violates(p) ? clampToBounds(kind, p, lunchMinutes, hobbyMinutes) : [];
  return { anchors: anchorsOf(p), adjustments };
}

const isInfeasible = (e: unknown): e is TimelineError =>
  e instanceof TimelineError && e.code === "SCHEDULE_INFEASIBLE";

/**
 * E5 template check for a weekday donor: the workday's required slots (a
 * required $meal eaten at home) lay out with the donor's raw times and the
 * gap table's medians.
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
    resolveDayWindows(required, anchorsOf(profile), decisions, input.world, (from, to) =>
      medianGap(input.gaps, from, to),
    );
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
  const floor = Math.ceil(profile.work_start + 240);
  for (let to = Math.ceil(from) - 1; to >= floor; to--) {
    const candidate = { ...profile, work_end: to };
    if (requiredWorkdaySlotsFit(input, candidate)) {
      return { ...pick, profile: candidate, adjustments: [...pick.adjustments, { item: "work_end", from, to }] };
    }
  }
  return pick;
}

function buildDay(input: ScheduleInput, donors: Donors, decisions: DayDecisions, attempt: number): ScheduleDay {
  const { world, gaps, templates } = input;
  const d = decisions.day_index;
  const profile = decisions.template === "workday" ? donors.weekday.profile : donors.weekend.profile;
  const { anchors, adjustments } = dayAnchors(input, profile, decisions, attempt);
  const { slots, transitions, dropped } = resolveDayWindows(
    templates.day_types[decisions.template],
    anchors,
    decisions,
    world,
    (from, to, i) => sampleGap(gaps, from, to, rngFor(world.seed, "day", d, attempt, "gap", i)),
    templates.reduce_order?.[decisions.template],
  );
  return {
    day_index: d,
    weekday: decisions.weekday,
    day_type: decisions.day_type,
    attempt,
    wake: formatClock(anchors.wake),
    sleep: formatClock(anchors.sleep),
    // R45: a slot exception whose slot was dropped for time did not happen.
    exceptions: decisions.exceptions.filter(
      (ex) => ex.scope !== "slot" || !dropped.some((x) => x.role === ex.replaces && x.reason === "no_time"),
    ),
    slots,
    dropped_slots: dropped,
    transitions,
    anchor_adjustments: adjustments,
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
    days,
  };
}
