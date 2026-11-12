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
  resolveDayWindows,
  verifyTimeline,
  type DayAnchors,
  type DayDecisions,
  type DonorRow,
  type GapTable,
  type ScheduleDay,
  type ScheduleInput,
  type Slot,
  type Templates,
  type World,
} from "../src/index.js";
import { genId, memoryCatalog } from "./fixtures.js";

// Synthetic data only: B16 per-day slot duration overrides (dials.slot_overrides).

const readConfig = (rel: string): unknown =>
  JSON.parse(readFileSync(new URL(`../../../config/${rel}`, import.meta.url), "utf8"));

const json = (x: unknown) => JSON.stringify(x, null, 2);
const roles = (day: ScheduleDay): string[] => day.slots.map((s) => s.role);
const slotOf = (day: ScheduleDay, role: string): Slot | undefined => day.slots.find((s) => s.role === role);
const startOf = (s: Slot): number => parseClock(s.window[0]);
const endOf = (s: Slot): number => parseClock(s.window[1]);
const lengthOf = (s: Slot): number => endOf(s) - startOf(s);

function errorOf(fn: () => unknown): TimelineError {
  try {
    fn();
  } catch (e) {
    if (e instanceof TimelineError) return e;
    throw e;
  }
  throw new Error("expected a TimelineError");
}

// ---------------------------------------------------------------------------
// Hand-computed layouts (resolveDayWindows, constant 20-min gaps).

function templates(): Templates {
  return TemplatesSchema.parse(readConfig("timeline/templates.v3.json"));
}

function unitWorld(): World {
  return WorldSchema.parse({
    person_id: "p-override-unit",
    attributes: { country: "US", occupation_group: "professional", work_pattern: "day_shift" },
    distribution_source: "atus_2025",
    nodes: {
      home: { kind: "home" },
      workplace: { kind: "workplace", job: "clerk" },
      hobby_a: { kind: "hobby", activity: "lego building", setting: "home_indoor", weekly_freq: 1, tags: ["hobby_lego"] },
    },
    node_binding: { home: { node: "h" }, workplace: { node: "w" }, hobby_a: { node: "a" } },
    meal_style: {
      workday: { lunch: { workplace: 1 }, dinner: { home: 1 } },
      rest_day: { breakfast: { home: 1 }, lunch: { home: 1 }, dinner: { home: 1 } },
      breakfast_skip_prob: 0,
      place_nodes: {},
    },
    dials: { length_days: 28, routine_strength: 1, exception_rate: { slot: 0, day: 0 }, wake_sleep: "default_v1" },
    seed: 1,
  });
}

const gap20 = () => 20;
const table = (slots: Slot[]) => slots.map((s) => `${s.role} ${s.window[0]}-${s.window[1]}`);

// Rest day: wake 07:00, breakfast 07:30, sport 10:00, lunch 13:00, dinner 18:30, sleep 23:00.
const REST: DayAnchors = { wake: 420, breakfast: 450, sport: 600, lunch: 780, dinner: 1110, sleep: 1380 };
const restDay: DayDecisions = {
  day_index: 6,
  weekday: "Sat",
  day_type: "hobby_day",
  template: "hobby_day",
  meal_place: { breakfast: "home", lunch: "home", dinner: "home" },
  optional: {},
  hobby_node: "hobby_a",
  exceptions: [],
};
function resolveRest(overrides: { role: string; duration_min: number }[]) {
  const t = templates();
  return resolveDayWindows(t.day_types.hobby_day, REST, restDay, unitWorld(), gap20, t.reduce_order?.hobby_day, [], overrides);
}

// Workday: wake 06:30, work 07:34-17:00, lunch 12:00, dinner 18:30, sleep 22:18; hobby on.
const WORK: DayAnchors = { wake: 390, work_start: 454, lunch: 720, work_end: 1020, dinner: 1110, sleep: 1338 };
const workday: DayDecisions = {
  day_index: 1,
  weekday: "Mon",
  day_type: "workday",
  template: "workday",
  meal_place: { lunch: "workplace", dinner: "home" },
  optional: { hobby: true, home_evening: true },
  hobby_node: "hobby_a",
  exceptions: [],
};
function resolveWork(anchors: DayAnchors, overrides: { role: string; duration_min: number }[]) {
  const t = templates();
  return resolveDayWindows(t.day_types.workday, anchors, workday, unitWorld(), gap20, t.reduce_order?.workday, [], overrides);
}

describe("resolveDayWindows with slot overrides (B16, hand-computed)", () => {
  it("without overrides the rest day is the template layout", () => {
    const { slots, dropped } = resolveRest([]);
    expect(table(slots)).toEqual([
      "home_morning 07:00-07:30",
      "breakfast 07:30-07:50",
      "hobby 10:00-12:00",
      "lunch 13:00-13:45",
      "home_afternoon 13:45-18:30",
      "dinner 18:30-19:30",
      "home_evening 19:30-23:00",
    ]);
    expect(dropped).toEqual([]);
  });

  it("a 171-min hobby keeps its start, lasts exactly 171 min and shifts lunch later", () => {
    const { slots, dropped } = resolveRest([{ role: "hobby", duration_min: 171 }]);
    expect(table(slots)).toEqual([
      "home_morning 07:00-07:30",
      "breakfast 07:30-07:50",
      "hobby 10:00-12:51",
      "lunch 13:11-13:56",
      "home_afternoon 13:56-18:30",
      "dinner 18:30-19:30",
      "home_evening 19:30-23:00",
    ]);
    expect(dropped).toEqual([]);
  });

  it("a shorter override shortens the slot to exactly that length", () => {
    const { slots } = resolveRest([{ role: "hobby", duration_min: 106 }]);
    expect(table(slots).slice(2, 4)).toEqual(["hobby 10:00-11:46", "lunch 13:00-13:45"]);
  });

  it("near sleep the home slots after it give way and are dropped as no_time; meals keep their length", () => {
    const { slots, dropped } = resolveRest([{ role: "hobby", duration_min: 650 }]);
    expect(table(slots)).toEqual([
      "home_morning 07:00-07:30",
      "breakfast 07:30-07:50",
      "hobby 10:00-20:50",
      "lunch 21:10-21:55",
      "dinner 21:55-22:55",
    ]);
    expect(dropped).toEqual([
      { role: "home_afternoon", reason: "no_time" },
      { role: "home_evening", reason: "no_time" },
    ]);
  });

  it("a home slot shrinks down to 20 min (recorded as shrunk) before it is dropped", () => {
    // Template home_evening has min 30; 105 min of hobby leaves it 23 min.
    const shrunk = resolveWork(WORK, [{ role: "hobby", duration_min: 105 }]);
    expect(table(shrunk.slots).slice(4)).toEqual(["dinner 18:30-19:30", "hobby 19:50-21:35", "home_evening 21:55-22:18"]);
    expect(shrunk.dropped).toEqual([{ role: "home_evening", reason: "shrunk" }]);
    // 120 min leaves it 8 min: below 20, so it goes.
    const gone = resolveWork(WORK, [{ role: "hobby", duration_min: 120 }]);
    expect(table(gone.slots).slice(4)).toEqual(["dinner 18:30-19:30", "hobby 19:50-21:50"]);
    expect(gone.dropped).toEqual([{ role: "home_evening", reason: "no_time" }]);
  });

  it("a required home slot (chores) shrinks to 20 min first and is never dropped", () => {
    const t = templates();
    const homeDay: DayDecisions = { ...restDay, day_type: "home_day", template: "home_day" };
    const run = (minutes: number) =>
      resolveDayWindows(t.day_types.home_day, REST, homeDay, unitWorld(), gap20, t.reduce_order?.home_day, [], [{ role: "breakfast", duration_min: minutes }]);
    // With room before sleep chores keeps its 60 min and pushes lunch later.
    const roomy = run(300);
    expect(table(roomy.slots).slice(1, 4)).toEqual(["breakfast 07:30-12:30", "chores 12:30-13:30", "lunch 13:30-14:15"]);
    expect(roomy.dropped).toEqual([]);
    const { slots, dropped } = run(760);
    expect(table(slots)).toEqual([
      "home_morning 07:00-07:30",
      "breakfast 07:30-20:10",
      "chores 20:10-20:30",
      "lunch 20:30-21:15",
      "home_afternoon 21:15-21:25",
      "dinner 21:25-22:25",
      "home_evening 22:25-23:00",
    ]);
    expect(dropped).toEqual([{ role: "chores", reason: "shrunk" }]);
    const e = errorOf(() => run(900));
    expect(e.code).toBe("SCHEDULE_INFEASIBLE");
    expect(e.details).toMatchObject({ day: 6, role: "breakfast", slot_override: true });
  });

  it("the overridden slot is exempt from the reduce order: never shrunk or dropped", () => {
    const tight = { ...WORK, sleep: 1260 };
    // Without the override the reduce order shrinks, then drops the hobby.
    expect(resolveWork(tight, []).dropped).toEqual([
      { role: "hobby", reason: "shrunk" },
      { role: "hobby", reason: "no_time" },
    ]);
    const { slots, dropped } = resolveWork(tight, [{ role: "hobby", duration_min: 50 }]);
    expect(table(slots).slice(4)).toEqual(["dinner 18:30-19:30", "hobby 19:50-20:40"]);
    expect(dropped).toEqual([{ role: "home_evening", reason: "no_time" }]);
  });

  it("an override that cannot fit before sleep throws SCHEDULE_INFEASIBLE naming the day and role", () => {
    const e = errorOf(() => resolveRest([{ role: "hobby", duration_min: 660 }]));
    expect(e.code).toBe("SCHEDULE_INFEASIBLE");
    expect(e.message).toMatch(/day 6/);
    expect(e.message).toMatch(/hobby/);
    expect(e.details).toMatchObject({ day: 6, role: "hobby", slot_override: true });
    const w = errorOf(() => resolveWork({ ...WORK, sleep: 1260 }, [{ role: "hobby", duration_min: 90 }]));
    expect(w.code).toBe("SCHEDULE_INFEASIBLE");
    expect(w.details).toMatchObject({ day: 1, role: "hobby", slot_override: true });
  });

  it("an overridden evening activity (B8) lasts exactly its override and is not shrunk", () => {
    const t = templates();
    const gym = { node: "hobby_a", role: "exercise", position: "before_dinner" as const, duration_min: 75, weekdays: ["Mon" as const] };
    const off = { ...workday, optional: { hobby: false, home_evening: true } };
    const run = (minutes: number) =>
      resolveDayWindows(t.day_types.workday, WORK, off, unitWorld(), gap20, t.reduce_order?.workday, [gym], [{ role: "exercise", duration_min: minutes }]);
    const { slots, dropped } = run(100);
    expect(table(slots).slice(3)).toEqual(["work_pm 12:45-17:00", "exercise 17:20-19:00", "dinner 19:20-20:20", "home_evening 20:20-22:18"]);
    expect(dropped).toEqual([]);
    const e = errorOf(() => run(250));
    expect(e.details).toMatchObject({ day: 1, role: "exercise", slot_override: true });
  });

  it("an override of a role the day's template lacks is WORLD_INVALID", () => {
    const e = errorOf(() => resolveRest([{ role: "chores", duration_min: 60 }]));
    expect(e.code).toBe("WORLD_INVALID");
    expect(e.message).toMatch(/day 6/);
    expect(e.message).toMatch(/chores/);
  });

  it("an override of a slot that is not on the day (optional off) is SCHEDULE_INFEASIBLE", () => {
    const t = templates();
    const off = { ...workday, optional: { hobby: false, home_evening: true } };
    const e = errorOf(() =>
      resolveDayWindows(t.day_types.workday, WORK, off, unitWorld(), gap20, t.reduce_order?.workday, [], [{ role: "hobby", duration_min: 60 }]),
    );
    expect(e.code).toBe("SCHEDULE_INFEASIBLE");
    expect(e.details).toMatchObject({ day: 1, role: "hobby", slot_override: true });
  });
});

// ---------------------------------------------------------------------------
// Whole schedules (same synthetic setup as the B8 evening tests).

type Over = { overrides?: unknown[]; meals?: unknown[]; seed?: number };

function worldJson(o: Over = {}): Record<string, unknown> {
  return {
    person_id: "p-override",
    attributes: { country: "US", occupation_group: "professional", work_pattern: "day_shift" },
    distribution_source: "atus_2025",
    nodes: {
      home: { kind: "home" },
      kitchen: { kind: "kitchen", default: "home" },
      workplace: { kind: "workplace", job: "clerk" },
      lego: { kind: "hobby", activity: "lego building", setting: "home_indoor", weekly_freq: 1, tags: ["hobby_lego"] },
      diner: { kind: "frequent_place", setting: "venue_indoor", optional: true },
    },
    node_binding: {
      home: { dataset: ["nextme800"], person_source: "p1", node: "home" },
      workplace: { dataset: ["nextme800"], person_source: "p1", node: "work" },
      lego: { dataset: ["youtube"], tags_any: ["hobby_lego"] },
      diner: { node: "d" },
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
      exception_rate: { slot: 0.15, day: 0.15 },
      wake_sleep: "default_v1",
      ...(o.overrides !== undefined ? { slot_overrides: o.overrides } : {}),
      ...(o.meals !== undefined ? { meal_place_overrides: o.meals } : {}),
    },
    seed: o.seed ?? 11,
  };
}

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
  return [...weekday, ...weekend];
}

function input(o: Over = {}): ScheduleInput {
  const world = parseWorld(worldJson(o));
  return {
    world,
    templates: templates(),
    gaps: GapTableSchema.parse(readConfig("timeline/gaps.v1.json")) as GapTable,
    exceptions: ExceptionCatalogSchema.parse(readConfig("timeline/exceptions.v1.json")),
    source: atus2025Source(pool()),
    boundNodes: new Set(Object.keys(world.node_binding)),
    defaults: ScheduleDefaultsSchema.parse(readConfig("schedule_defaults_atus2025.json")),
  };
}

const base = generateSchedule(input());
const restDays = base.days.filter((d) => d.day_type === "hobby_day" && roles(d).includes("hobby"));
const firstRest = restDays[0]!;

describe("dials.slot_overrides (B16): world validation", () => {
  it("is optional and absent when not given", () => {
    expect("slot_overrides" in parseWorld(worldJson()).dials).toBe(false);
    const o = [{ day_index: 6, role: "hobby", duration_min: 171 }];
    expect(parseWorld(worldJson({ overrides: o })).dials.slot_overrides).toEqual(o);
  });

  it.each([
    ["day_index 0", [{ day_index: 0, role: "hobby", duration_min: 60 }], /day_index/],
    ["a day past length_days", [{ day_index: 29, role: "hobby", duration_min: 60 }], /day_index 29 is past length_days 28/],
    ["a zero duration", [{ day_index: 6, role: "hobby", duration_min: 0 }], /duration_min/],
    [
      "the same day and role twice",
      [
        { day_index: 6, role: "hobby", duration_min: 60 },
        { day_index: 6, role: "hobby", duration_min: 90 },
      ],
      /day 6 role "hobby" is listed twice/,
    ],
  ])("rejects %s", (_, overrides, message) => {
    const e = errorOf(() => parseWorld(worldJson({ overrides })));
    expect(e.code).toBe("WORLD_INVALID");
    expect(e.message).toMatch(message);
  });

  it("rejects a role no template has before generating", () => {
    const e = errorOf(() => generateSchedule(input({ overrides: [{ day_index: 6, role: "nap", duration_min: 30 }] })));
    expect(e.code).toBe("WORLD_INVALID");
    expect(e.message).toMatch(/slot_overrides.*nap/);
    const inp = input({ overrides: [{ day_index: 6, role: "nap", duration_min: 30 }] });
    expect(errorOf(() => generateDay(inp, base.donors, 1, 0)).code).toBe("WORLD_INVALID");
  });
});

describe("slot overrides in the schedule (B16)", () => {
  it("worlds without the dial, or with an empty one, are byte-identical", () => {
    expect(json(generateSchedule(input()))).toBe(json(base));
    expect(json(generateSchedule(input({ overrides: [] })))).toBe(json(base));
  });

  it("a rest-day hobby lengthened to 171 min keeps its start and shifts later slots; other days are unchanged", () => {
    expect(restDays.length).toBeGreaterThan(2);
    const d = firstRest.day_index;
    const s = generateSchedule(input({ overrides: [{ day_index: d, role: "hobby", duration_min: 171 }] }));
    expect(ScheduleSchema.parse(s)).toEqual(s);
    for (const day of s.days) if (day.day_index !== d) expect(json(day)).toBe(json(base.days[day.day_index - 1]));
    const day = s.days[d - 1]!;
    const hobby = slotOf(day, "hobby")!;
    const was = slotOf(firstRest, "hobby")!;
    expect(lengthOf(hobby)).toBe(171);
    expect(lengthOf(was)).toBeLessThan(171);
    expect(startOf(hobby)).toBe(startOf(was));
    const after = day.slots.slice(day.slots.indexOf(hobby) + 1);
    expect(after.length).toBeGreaterThan(0);
    const next = after[0]!;
    expect(startOf(next)).toBeGreaterThan(startOf(slotOf(firstRest, next.role)!));
    for (const sl of after) {
      const old = slotOf(firstRest, sl.role);
      if (old !== undefined) expect(startOf(sl)).toBeGreaterThanOrEqual(startOf(old));
      if (sl.spec.category === "meal" && old !== undefined) expect(lengthOf(sl)).toBe(lengthOf(old));
    }
    for (const sl of day.slots) expect(endOf(sl)).toBeLessThanOrEqual(parseClock(day.sleep));
  });

  it("three rest days take 133, 106 and 171 min exactly, and generateDay reproduces every day", () => {
    const lengths = [133, 106, 171];
    const overrides = restDays.slice(0, 3).map((d, i) => ({ day_index: d.day_index, role: "hobby", duration_min: lengths[i]! }));
    const inp = input({ overrides });
    const s = generateSchedule(inp);
    overrides.forEach((o) => expect(lengthOf(slotOf(s.days[o.day_index - 1]!, "hobby")!)).toBe(o.duration_min));
    expect(json(generateSchedule(input({ overrides })))).toBe(json(s));
    for (const day of s.days) expect(json(generateDay(inp, s.donors, day.day_index, day.attempt))).toBe(json(day));
  });

  it("an override too long for the day fails at once with SCHEDULE_INFEASIBLE naming the day and role", () => {
    const d = firstRest.day_index;
    const e = errorOf(() => generateSchedule(input({ overrides: [{ day_index: d, role: "hobby", duration_min: 900 }] })));
    expect(e.code).toBe("SCHEDULE_INFEASIBLE");
    expect(e.message).toMatch(new RegExp(`day ${d}\\b`));
    expect(e.message).toMatch(/hobby/);
    expect(e.details).toMatchObject({ day: d, role: "hobby", slot_override: true });
    // Not retried with other attempts: the error is the override's own.
    expect(e.details.attempts).toBeUndefined();
  });

  it("an override on a day whose template lacks the role is WORLD_INVALID", () => {
    const workdayIndex = base.days.find((d) => d.day_type === "workday")!.day_index;
    const e = errorOf(() => generateSchedule(input({ overrides: [{ day_index: workdayIndex, role: "breakfast", duration_min: 30 }] })));
    expect(e.code).toBe("WORLD_INVALID");
    expect(e.message).toMatch(new RegExp(`day ${workdayIndex}\\b.*breakfast`));
  });
});

describe("verifier and slot overrides (B16)", () => {
  const timelineWith = (overrides: unknown[]) => {
    const inp = input({ overrides });
    return createTimeline(inp.world, generateSchedule(inp), genId);
  };

  it("accepts the overridden duration", () => {
    const t = timelineWith([{ day_index: firstRest.day_index, role: "hobby", duration_min: 171 }]);
    const r = verifyTimeline(t, memoryCatalog([]));
    expect(r.rules.wake_sleep).toEqual({ pass: true, violations: [] });
    expect(r.rules.day_type).toEqual({ pass: true, violations: [] });
    const details = r.rules.slot_window.violations.map((v) => v.detail);
    expect(details.filter((x) => /override/.test(x))).toEqual([]);
  });

  it("flags a slot whose length differs from its override", () => {
    const d = firstRest.day_index;
    const t = timelineWith([{ day_index: d, role: "hobby", duration_min: 171 }]);
    const hobby = slotOf(t.schedule.days[d - 1]!, "hobby")!;
    hobby.window = [hobby.window[0], "23:59"];
    const details = verifyTimeline(t, memoryCatalog([])).rules.slot_window.violations.map((v) => v.detail);
    expect(details).toContainEqual(expect.stringMatching(/hobby .*slot_overrides sets 171 min/));
    // A missing overridden slot is reported too.
    const day = t.schedule.days[d - 1]!;
    day.slots = day.slots.filter((s) => s.role !== "hobby");
    const again = verifyTimeline(t, memoryCatalog([])).rules.slot_window.violations.map((v) => v.detail);
    expect(again).toContain("slot override hobby (171 min) has no slot");
  });
});

// ---------------------------------------------------------------------------
// B21: dials.meal_place_overrides.

const nodeOf = (day: ScheduleDay, role: string): string | undefined => slotOf(day, role)?.spec.node;

describe("dials.meal_place_overrides (B21)", () => {
  const restLunch = base.days.find((d) => d.day_type === "hobby_day" && roles(d).includes("lunch"));
  const rest = restLunch!;

  it("is optional and absent when not given; worlds without it are byte-identical", () => {
    expect("meal_place_overrides" in parseWorld(worldJson()).dials).toBe(false);
    expect(json(generateSchedule(input({ meals: undefined })))).toBe(json(base));
  });

  it("an override changes only that day's lunch place and node; other days are identical", () => {
    expect(rest).toBeDefined();
    const d = rest.day_index;
    const s = generateSchedule(input({ meals: [{ day_index: d, meal: "lunch", place: "eat_out" }] }));
    const before = nodeOf(rest, "lunch");
    const after = s.days.find((x) => x.day_index === d)!;
    expect(nodeOf(after, "lunch")).toBe("diner");
    expect(before).not.toBe("diner");
    for (const day of s.days.filter((x) => x.day_index !== d)) {
      expect(json(day)).toBe(json(base.days.find((x) => x.day_index === day.day_index)));
    }
    // regeneration reproduces it
    const inp = input({ meals: [{ day_index: d, meal: "lunch", place: "eat_out" }] });
    expect(json(generateDay(inp, s.donors, d, after.attempt))).toBe(json(after));
  });

  it("a workday lunch drawn eat_out can be moved to the workplace", () => {
    const out = base.days.find((x) => x.day_type === "workday" && nodeOf(x, "lunch") === "diner");
    expect(out).toBeDefined();
    const d = out!.day_index;
    const s = generateSchedule(input({ meals: [{ day_index: d, meal: "lunch", place: "workplace" }] }));
    expect(nodeOf(s.days.find((x) => x.day_index === d)!, "lunch")).toBe("workplace");
  });

  it.each([
    ["a day past length_days", [{ day_index: 29, meal: "lunch", place: "home" }], /day_index 29 is past length_days 28/],
    [
      "a duplicate (day, meal)",
      [
        { day_index: 6, meal: "lunch", place: "home" },
        { day_index: 6, meal: "lunch", place: "eat_out" },
      ],
      /day 6 meal "lunch" is listed twice/,
    ],
    ["an unknown meal", [{ day_index: 6, meal: "snack", place: "home" }], /meal/],
  ])("world validation rejects %s", (_, meals, message) => {
    const e = errorOf(() => parseWorld(worldJson({ meals })));
    expect(e.code).toBe("WORLD_INVALID");
    expect(e.message).toMatch(message);
  });

  it("an override for a meal the day lacks (workday breakfast) is WORLD_INVALID naming day and meal", () => {
    const e = errorOf(() => generateSchedule(input({ meals: [{ day_index: 1, meal: "breakfast", place: "home" }] })));
    expect(e.code).toBe("WORLD_INVALID");
    expect(e.message).toMatch(/day 1 breakfast/);
  });

  it("a place that is not usable (unbound eat_out node) is WORLD_INVALID", () => {
    const inp = input({ meals: [{ day_index: 1, meal: "lunch", place: "eat_out" }] });
    inp.boundNodes = new Set([...inp.boundNodes].filter((n) => n !== "diner"));
    const e = errorOf(() => generateSchedule(inp));
    expect(e.code).toBe("WORLD_INVALID");
    expect(e.message).toMatch(/day 1 lunch/);
  });
});
