import { formatClock, parseClock } from "../clock.js";
import { selectDonors, type DayProfile, type DistributionSource } from "../donor.js";
import { TimelineError } from "../errors.js";
import { sampleGap, type GapTable } from "../gaps.js";
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

/** Donor times jittered per day, rounded, then bent into a plausible order (R14). */
function dayAnchors(input: ScheduleInput, profile: DayProfile, decisions: DayDecisions, attempt: number): DayAnchors {
  const { seed } = input.world;
  const d = decisions.day_index;
  const sigma = daySigma(input);
  const jitter = (name: AnchorName | "sleep", value: number): number =>
    Math.round(value + rngFor(seed, "day", d, attempt, "jitter", name).normal(0, sigma));

  const workday = decisions.template === "workday";
  const a: DayAnchors = {
    wake: jitter("wake", profile.wake),
    sleep: jitter("sleep", profile.sleep),
    breakfast: jitter("breakfast", profile.breakfast),
    lunch: jitter("lunch", profile.lunch),
    dinner: jitter("dinner", profile.dinner),
    sport: jitter("sport", profile.sport ?? parseClock(input.defaults.weekend.sports_start.p50)),
  };
  if (workday && profile.work_start !== undefined) a.work_start = jitter("work_start", profile.work_start);
  if (workday && profile.work_end !== undefined) a.work_end = jitter("work_end", profile.work_end);

  a.breakfast = Math.max(a.breakfast!, a.wake + 10);
  if (workday && a.work_start !== undefined && a.work_end !== undefined) {
    a.work_end = Math.max(a.work_end, a.work_start + 240);
    a.lunch = Math.min(Math.max(a.lunch!, a.work_start + 60), a.work_end - 60);
  }
  a.dinner = Math.min(a.dinner!, a.sleep - 60);
  return a;
}

function buildDay(input: ScheduleInput, donors: Donors, decisions: DayDecisions, attempt: number): ScheduleDay {
  const { world, gaps, templates } = input;
  const d = decisions.day_index;
  const profile = decisions.template === "workday" ? donors.weekday.profile : donors.weekend.profile;
  const anchors = dayAnchors(input, profile, decisions, attempt);
  const { slots, transitions } = resolveDayWindows(
    templates.day_types[decisions.template],
    anchors,
    decisions,
    world,
    (from, to, i) => sampleGap(gaps, from, to, rngFor(world.seed, "day", d, attempt, "gap", i)),
  );
  return {
    day_index: d,
    weekday: decisions.weekday,
    day_type: decisions.day_type,
    attempt,
    wake: formatClock(anchors.wake),
    sleep: formatClock(anchors.sleep),
    exceptions: decisions.exceptions,
    slots,
    transitions,
  };
}

/** One day at a given regeneration attempt; attempt 0 matches generateSchedule's first try. */
export function generateDay(input: ScheduleInput, donors: Donors, dayIndex: number, attempt: number): ScheduleDay {
  const decisions = decideDay(input.world, input.exceptions, { boundNodes: input.boundNodes }, dayIndex, attempt);
  return buildDay(input, donors, decisions, attempt);
}

const isInfeasible = (e: unknown): e is TimelineError =>
  e instanceof TimelineError && e.code === "SCHEDULE_INFEASIBLE";

/**
 * schedule.json for one world (§4.3, §5.2). A day that is infeasible is
 * regenerated with attempts 1..5; after the 5th the error names the day.
 */
export function generateSchedule(input: ScheduleInput): Schedule {
  const { world, templates, gaps, exceptions, source, boundNodes } = input;
  const picked = selectDonors(world.attributes, source, {
    seed: world.seed,
    allowCountryFallback: world.allow_country_fallback,
  });
  const donors: Donors = {
    weekday: picked.weekday,
    weekend: picked.weekend,
    country_fallback: picked.country_fallback,
    weekend_redraws: picked.weekend_redraws,
  };

  const days = decideDays(world, exceptions, { boundNodes }).map((decisions) => {
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
