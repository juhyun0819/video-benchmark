import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  ExceptionCatalogSchema,
  GapTableSchema,
  OCCUPATION_CODES,
  ScheduleDefaultsSchema,
  ScheduleSchema,
  TemplatesSchema,
  TimelineError,
  WorldSchema,
  atus2025Source,
  createTimeline,
  generateDay,
  generateSchedule,
  parseClock,
  parseWorld,
  verifyTimeline,
  type DonorRow,
  type GapTable,
  type Schedule,
  type ScheduleDay,
  type ScheduleInput,
  type Slot,
  type World,
} from "../src/index.js";
import { genId, memoryCatalog } from "./fixtures.js";

// Synthetic data only: B8 evening activities (a Tue/Thu gym before dinner, a counted hobby after dinner).

const readConfig = (rel: string): unknown =>
  JSON.parse(readFileSync(new URL(`../../../config/${rel}`, import.meta.url), "utf8"));

const GYM = { node: "gym_1", role: "exercise", position: "before_dinner", duration_min: 75, weekdays: ["Tue", "Thu"] };
const GUITAR = { node: "hobby_2", role: "hobby_2", position: "after_dinner", duration_min: 45, count: 6, max_per_week: 2 };
const EVENING_ROLES = ["exercise", "hobby_2"];

type Over = { seed?: number; day?: number; slot?: number; late?: number; evening?: unknown[]; tripStart?: number };

function worldJson(o: Over = {}): Record<string, unknown> {
  return {
    person_id: "p-evening",
    attributes: { country: "US", occupation_group: "professional", work_pattern: "day_shift" },
    distribution_source: "atus_2025",
    nodes: {
      home: { kind: "home" },
      kitchen: { kind: "kitchen", default: "home" },
      workplace: { kind: "workplace", job: "clerk" },
      walking: { kind: "hobby", activity: "walking", setting: "outdoor", weekly_freq: 1, tags: ["walking"] },
      gym_1: { kind: "hobby", activity: "gym workout", setting: "venue_indoor", weekly_freq: 0, tags: ["gym"] },
      hobby_2: { kind: "hobby", activity: "guitar", setting: "venue_indoor", weekly_freq: 0, tags: ["music_guitar"] },
      diner: { kind: "frequent_place", setting: "venue_indoor", optional: true },
      trip_1: { kind: "trip", lodging_setting: "venue_indoor", outing_setting: "outdoor", meal_setting: "venue_indoor" },
    },
    node_binding: {
      home: { dataset: ["nextme800"], person_source: "p1", node: "home" },
      workplace: { dataset: ["nextme800"], person_source: "p1", node: "work" },
      walking: { dataset: ["ego4d"], tags_any: ["walking"] },
      gym_1: { dataset: ["nextme800"], person_source: "p1", tags_any: ["gym"] },
      hobby_2: { dataset: ["egoexo4d"], tags_any: ["music_guitar"] },
      diner: { node: "d" },
      trip_1: { dataset: ["nextme800"], person_source: "p1", day_group_sequence: ["2026-05-07", "2026-05-08", "2026-05-09"] },
    },
    meal_style: {
      workday: { lunch: { workplace: 0.8, eat_out: 0.2 }, dinner: { home: 0.8, eat_out: 0.2 } },
      rest_day: { breakfast: { home: 1 }, lunch: { home: 0.7, eat_out: 0.3 }, dinner: { home: 0.8, eat_out: 0.2 } },
      breakfast_skip_prob: 0,
      place_nodes: { eat_out: "diner" },
    },
    dials: {
      length_days: 28,
      routine_strength: 0.6,
      exception_rate: { slot: o.slot ?? 0.15, day: o.day ?? 0.15 },
      wake_sleep: "default_v1",
      late_workdays_per_week: o.late ?? 1,
      // Day 16 is a Tuesday (start_weekday Mon): the trip covers Tue-Thu of week 3.
      trips: [{ node: "trip_1", length_days: 3, start_day: o.tripStart ?? 16 }],
      ...(o.evening !== undefined ? { evening_activities: o.evening } : {}),
    },
    seed: o.seed ?? 11,
  };
}

const makeWorld = (o: Over = {}): World => parseWorld(worldJson(o));

let counter = 0;
function row(over: Partial<DonorRow> = {}): DonorRow {
  counter += 1;
  return {
    id: `r${counter}`,
    daytype: "weekday",
    occ: OCCUPATION_CODES["professional"]!,
    w: 1 + (counter % 3),
    wake: 400 + (counter % 5) * 5,
    bed: 1370,
    work_start: 500 + (counter % 4) * 10,
    work_end: 1010,
    work_min: 480,
    worked: true,
    breakfast_time: 430,
    breakfast_loc: 1,
    lunch_time: 740,
    lunch_loc: 2,
    dinner_time: 1110,
    dinner_loc: 1,
    sport_has: false,
    sport_time: null,
    sport_outdoor: null,
    house_min: 30,
    social_out: false,
    ...over,
  };
}

function pool(): DonorRow[] {
  counter = 0;
  const weekday = Array.from({ length: 40 }, () => row());
  const weekend = Array.from({ length: 40 }, () =>
    row({ daytype: "weekend", worked: false, work_start: null, work_end: null, wake: 470, bed: 1390, breakfast_time: 500, lunch_time: 760, dinner_time: 1100 }),
  );
  const late = Array.from({ length: 40 }, () =>
    row({ work_start: 720, work_end: 1230, wake: 450, bed: 1400, breakfast_time: 480, lunch_time: 800, dinner_time: 1290 }),
  );
  return [...weekday, ...weekend, ...late];
}

function input(o: Over = {}): ScheduleInput {
  const world = makeWorld(o);
  return {
    world,
    templates: TemplatesSchema.parse(readConfig("timeline/templates.v3.json")),
    gaps: GapTableSchema.parse(readConfig("timeline/gaps.v1.json")) as GapTable,
    exceptions: ExceptionCatalogSchema.parse(readConfig("timeline/exceptions.v1.json")),
    source: atus2025Source(pool()),
    boundNodes: new Set(Object.keys(world.node_binding)),
    defaults: ScheduleDefaultsSchema.parse(readConfig("schedule_defaults_atus2025.json")),
  };
}

const json = (x: unknown) => JSON.stringify(x, null, 2);
const roles = (day: ScheduleDay): string[] => day.slots.map((s) => s.role);
const slotOf = (day: ScheduleDay, role: string): Slot | undefined => day.slots.find((s) => s.role === role);
const startOf = (s: Slot): number => parseClock(s.window[0]);
const endOf = (s: Slot): number => parseClock(s.window[1]);
const lengthOf = (s: Slot): number => endOf(s) - startOf(s);
/** Transition minutes into a slot (0 when the node does not change). */
const gapInto = (day: ScheduleDay, s: Slot): number => day.transitions.find((t) => t.to_slot === s.slot_id)?.minutes ?? 0;
/** ISO week (Mon-Sun) of a day; the fixtures start on a Monday. */
const weekOf = (d: number): number => Math.floor((d - 1) / 7);
/** A routine workday: template workday, not late, no exception of any kind. */
const routineWorkday = (day: ScheduleDay): boolean => day.day_type === "workday" && day.exceptions.length === 0;

function errorOf(fn: () => unknown): TimelineError {
  try {
    fn();
  } catch (e) {
    if (e instanceof TimelineError) return e;
    throw e;
  }
  throw new Error("expected a TimelineError");
}

describe("dials.evening_activities (B8): world schema", () => {
  it("is optional and absent when not given, so existing worlds parse unchanged", () => {
    const w = makeWorld();
    expect("evening_activities" in w.dials).toBe(false);
    expect(makeWorld({ evening: [GYM, GUITAR] }).dials.evening_activities).toEqual([GYM, GUITAR]);
  });

  it.each([
    ["an unknown node", [{ ...GYM, node: "nowhere" }], /not a hobby node/],
    ["a non-hobby node", [{ ...GYM, node: "workplace" }], /not a hobby node/],
    ["both weekdays and count", [{ ...GYM, count: 4 }], /exactly one of weekdays and count/],
    ["neither weekdays nor count", [{ node: "gym_1", role: "exercise", position: "before_dinner", duration_min: 75 }], /exactly one of weekdays and count/],
    ["max_per_week without count", [{ ...GYM, max_per_week: 1 }], /max_per_week needs count/],
    ["a role listed twice", [GYM, { ...GUITAR, role: "exercise" }], /role "exercise" is listed twice/],
    ["an unknown position", [{ ...GYM, position: "at_lunch" }], /position/],
  ])("rejects %s", (_, evening, message) => {
    const e = errorOf(() => makeWorld({ evening }));
    expect(e.code).toBe("WORLD_INVALID");
    expect(e.message).toMatch(message);
  });

  it("rejects a role that is already a template role when generating", () => {
    const e = errorOf(() => generateSchedule(input({ evening: [{ ...GYM, role: "dinner" }] })));
    expect(e.code).toBe("WORLD_INVALID");
    expect(e.message).toMatch(/dinner/);
  });
});

describe("evening activities in the schedule (B8)", () => {
  it("the golden path is unchanged: an empty dial generates the same schedule as no dial", () => {
    const none = json(generateSchedule(input()));
    expect(json(generateSchedule(input({ evening: [] })))).toBe(none);
  });

  it("Tue/Thu exercise goes right after work_pm and before dinner on routine Tue/Thu workdays only", () => {
    const s = generateSchedule(input({ evening: [GYM] }));
    expect(ScheduleSchema.parse(s)).toEqual(s);
    let placed = 0;
    for (const day of s.days) {
      const want = (day.weekday === "Tue" || day.weekday === "Thu") && routineWorkday(day);
      expect(roles(day).includes("exercise"), `day ${day.day_index}`).toBe(want);
      if (!want) continue;
      placed += 1;
      const r = roles(day);
      const i = r.indexOf("exercise");
      expect(r[i - 1]).toBe("work_pm");
      expect(r[i + 1]).toBe("dinner");
      const ex = day.slots[i]!;
      expect(ex.required).toBe(false);
      expect(ex.spec).toMatchObject({ category: "hobby", node: "gym_1", setting: "venue_indoor", activity: "gym workout" });
      expect(ex.spec.tags).toContain("gym");
      const workPm = day.slots[i - 1]!;
      const dinner = day.slots[i + 1]!;
      expect(gapInto(day, ex)).toBeGreaterThan(0);
      expect(startOf(ex)).toBe(endOf(workPm) + gapInto(day, ex));
      expect(lengthOf(ex)).toBe(75);
      // The dinner is pushed past the activity and the travel to it.
      expect(startOf(dinner)).toBeGreaterThanOrEqual(endOf(ex) + gapInto(day, dinner));
      for (const sl of day.slots) expect(endOf(sl)).toBeLessThanOrEqual(parseClock(day.sleep));
    }
    expect(placed).toBeGreaterThanOrEqual(4);
  });

  it("a before_dinner activity shifts dinner later than on the same day without it", () => {
    const without = generateSchedule(input());
    const withGym = generateSchedule(input({ evening: [GYM] }));
    let pushed = 0;
    for (const day of withGym.days) {
      if (!roles(day).includes("exercise")) {
        // Days without the activity are untouched.
        expect(json(day)).toBe(json(without.days[day.day_index - 1]));
        continue;
      }
      const base = without.days[day.day_index - 1]!;
      expect(day.wake).toBe(base.wake);
      expect(slotOf(day, "work_pm")!.window).toEqual(slotOf(base, "work_pm")!.window);
      if (startOf(slotOf(day, "dinner")!) > startOf(slotOf(base, "dinner")!)) pushed += 1;
    }
    expect(pushed).toBeGreaterThan(0);
  });

  it("count 6 with max 2 per week picks the routine workdays whose work_pm ends earliest", () => {
    const s = generateSchedule(input({ evening: [GUITAR] }));
    expect(s.days.every((d) => d.attempt === 0)).toBe(true);
    const eligible = s.days.filter(routineWorkday);
    const end = (day: ScheduleDay) => endOf(slotOf(day, "work_pm")!);
    expect(new Set(eligible.map(end)).size).toBeGreaterThan(3);
    const chosen = s.days.filter((d) => roles(d).includes("hobby_2"));
    expect(chosen).toHaveLength(6);
    for (const day of chosen) expect(routineWorkday(day)).toBe(true);
    const perWeek = new Map<number, ScheduleDay[]>();
    for (const day of chosen) perWeek.set(weekOf(day.day_index), [...(perWeek.get(weekOf(day.day_index)) ?? []), day]);
    for (const days of perWeek.values()) expect(days.length).toBeLessThanOrEqual(2);
    // Greedy by earliest end: a skipped day is no earlier than all 6 picks, or its week is full of no-later picks.
    const latest = Math.max(...chosen.map(end));
    for (const day of eligible) {
      if (chosen.includes(day)) continue;
      const week = perWeek.get(weekOf(day.day_index)) ?? [];
      const weekFull = week.length === 2 && week.every((c) => end(c) <= end(day));
      expect(end(day) >= latest || weekFull, `day ${day.day_index}`).toBe(true);
    }
  });

  it("an after_dinner activity goes right after dinner and pushes home_evening later, never past sleep", () => {
    const without = generateSchedule(input());
    const s = generateSchedule(input({ evening: [GUITAR] }));
    let pushed = 0;
    for (const day of s.days.filter((d) => roles(d).includes("hobby_2"))) {
      const r = roles(day);
      const i = r.indexOf("hobby_2");
      expect(r[i - 1]).toBe("dinner");
      const h = day.slots[i]!;
      expect(h.spec).toMatchObject({ category: "hobby", node: "hobby_2", setting: "venue_indoor", activity: "guitar" });
      expect(h.spec.tags).toContain("music_guitar");
      expect(startOf(h)).toBe(endOf(day.slots[i - 1]!) + gapInto(day, h));
      expect(lengthOf(h)).toBe(45);
      const home = slotOf(day, "home_evening");
      const baseHome = slotOf(without.days[day.day_index - 1]!, "home_evening");
      if (home !== undefined && baseHome !== undefined) {
        expect(startOf(home)).toBeGreaterThan(startOf(baseHome));
        expect(endOf(home)).toBe(parseClock(day.sleep));
        pushed += 1;
      }
      for (const sl of day.slots) expect(endOf(sl)).toBeLessThanOrEqual(parseClock(day.sleep));
    }
    expect(pushed).toBeGreaterThan(0);
  });

  it("when it would pass sleep, home_evening shrinks to 20 min first, then the activity (never below 20)", () => {
    const s = generateSchedule(input({ evening: [{ ...GUITAR, duration_min: 300 }] }));
    const chosen = s.days.filter((d) => roles(d).includes("hobby_2"));
    expect(chosen).toHaveLength(6);
    let homeKept = 0;
    for (const day of chosen) {
      const h = slotOf(day, "hobby_2")!;
      expect(lengthOf(h)).toBeLessThan(300);
      expect(lengthOf(h)).toBeGreaterThanOrEqual(20);
      expect(day.dropped_slots).toContainEqual({ role: "hobby_2", reason: "shrunk" });
      for (const sl of day.slots) expect(endOf(sl)).toBeLessThanOrEqual(parseClock(day.sleep));
      const home = slotOf(day, "home_evening");
      if (home !== undefined) {
        homeKept += 1;
        expect(lengthOf(home)).toBe(20);
        expect(endOf(home)).toBe(parseClock(day.sleep));
      }
    }
    expect(homeKept).toBeGreaterThan(0);
  });

  it("late, trip, leave and slot-exception days get no evening activity", () => {
    const everyWeekday = { ...GYM, weekdays: ["Mon", "Tue", "Wed", "Thu", "Fri"] };
    const s = generateSchedule(input({ evening: [everyWeekday, GUITAR], day: 0.25, slot: 0.3 }));
    const weekdays = s.days.filter((d) => !["Sat", "Sun"].includes(d.weekday));
    const has = (day: ScheduleDay) => roles(day).some((r) => EVENING_ROLES.includes(r));
    const excluded = weekdays.filter((d) => !routineWorkday(d));
    // The fixture must cover every excluded kind.
    for (const kind of ["workday_late", "trip_day", "leave_day"]) {
      expect(excluded.some((d) => d.day_type === kind), kind).toBe(true);
    }
    expect(excluded.some((d) => d.day_type === "workday" && d.exceptions.some((e) => e.scope === "slot"))).toBe(true);
    for (const day of excluded) expect(has(day), `day ${day.day_index} ${day.day_type}`).toBe(false);
    for (const day of weekdays.filter(routineWorkday)) expect(roles(day)).toContain("exercise");
    for (const day of s.days.filter((d) => d.weekday === "Sat" || d.weekday === "Sun")) expect(has(day)).toBe(false);
  });

  it("a day that is both a gym day and a hobby_2 day gets both, and the hobby_2 days do not depend on the gym", () => {
    const both = generateSchedule(input({ evening: [GYM, GUITAR] }));
    const guitarOnly = generateSchedule(input({ evening: [GUITAR] }));
    const days = (s: Schedule) => s.days.filter((d) => roles(d).includes("hobby_2")).map((d) => d.day_index);
    expect(days(both)).toEqual(days(guitarOnly));
    const shared = both.days.filter((d) => roles(d).includes("exercise") && roles(d).includes("hobby_2"));
    expect(shared.length).toBeGreaterThan(0);
    for (const day of shared) {
      const r = roles(day).filter((x) => ["work_pm", "exercise", "dinner", "hobby_2"].includes(x));
      expect(r).toEqual(["work_pm", "exercise", "dinner", "hobby_2"]);
    }
  });

  it("weekly_freq 0 activity nodes stay out of the routine hobby slots", () => {
    const s = generateSchedule(input({ evening: [GYM, GUITAR] }));
    const hobbyNodes = s.days.flatMap((d) => d.slots.filter((x) => x.role === "hobby").map((x) => x.spec.node));
    expect(hobbyNodes.length).toBeGreaterThan(0);
    expect(new Set(hobbyNodes)).toEqual(new Set(["walking"]));
  });

  it("is deterministic and generateDay reproduces every day, including the counted picks", () => {
    const inp = input({ evening: [GYM, GUITAR] });
    const s = generateSchedule(inp);
    expect(json(generateSchedule(input({ evening: [GYM, GUITAR] })))).toBe(json(s));
    for (const day of s.days) {
      expect(json(generateDay(inp, s.donors, day.day_index, day.attempt))).toBe(json(day));
    }
  });

  it("ties are broken by the seed", () => {
    // Different seeds may pick differently, but each seed is stable.
    for (const seed of [3, 5]) {
      const a = generateSchedule(input({ seed, evening: [GUITAR] }));
      const b = generateSchedule(input({ seed, evening: [GUITAR] }));
      expect(json(a)).toBe(json(b));
    }
  });
});

describe("verifier and evening activities (B8)", () => {
  const timelineWith = (evening: unknown[]) => {
    const inp = input({ evening });
    return createTimeline(inp.world, generateSchedule(inp), genId);
  };

  it("accepts the evening slots under the day_type rule", () => {
    const t = timelineWith([GYM, GUITAR]);
    expect(t.schedule.days.some((d) => roles(d).includes("exercise"))).toBe(true);
    const r = verifyTimeline(t, memoryCatalog([]));
    expect(r.rules.day_type).toEqual({ pass: true, violations: [] });
    expect(r.rules.wake_sleep.pass).toBe(true);
  });

  it("flags an evening activity on a non-workday or out of place", () => {
    const t = timelineWith([GYM, GUITAR]);
    const gymDay = t.schedule.days.find((d) => roles(d).includes("exercise"))!;
    const ex = slotOf(gymDay, "exercise")!;
    const rest = t.schedule.days.find((d) => d.day_type === "hobby_day")!;
    rest.slots.push({ ...ex, slot_id: `d${rest.day_index}-s99` });
    // Move the exercise after dinner on its own day.
    const i = gymDay.slots.indexOf(ex);
    gymDay.slots.splice(i, 1);
    gymDay.slots.splice(gymDay.slots.findIndex((s) => s.role === "dinner") + 1, 0, ex);
    const details = verifyTimeline(t, memoryCatalog([])).rules.day_type.violations.map((v) => v.detail);
    expect(details).toContainEqual(expect.stringMatching(/evening activity exercise on a hobby_day/));
    expect(details).toContainEqual(expect.stringMatching(/exercise \(before_dinner\) must come after work_pm and before dinner/));
  });
});
