import { readFileSync } from "node:fs";
import { afterEach, describe, expect, it } from "vitest";
import {
  ExceptionCatalogSchema,
  GapTableSchema,
  OCCUPATION_CODES,
  ScheduleDaySchema,
  ScheduleDefaultsSchema,
  ScheduleSchema,
  TemplatesSchema,
  TimelineError,
  WorldSchema,
  atus2025Source,
  createTimeline,
  decideDay,
  generateDay,
  generateSchedule,
  hardFilterFailures,
  parseClock,
  verifyTimeline,
  type Clip,
  type ClipCatalog,
  type DonorRow,
  type GapTable,
  type ScheduleInput,
  type World,
} from "../src/index.js";

const readConfig = (rel: string): unknown =>
  JSON.parse(readFileSync(new URL(`../../../config/${rel}`, import.meta.url), "utf8"));

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

function rows(weekend: Partial<DonorRow> = {}, weekdayOver: Partial<DonorRow> = {}): DonorRow[] {
  const weekday = Array.from({ length: 40 }, () => row(weekdayOver));
  const we = Array.from({ length: 40 }, () =>
    row({
      daytype: "weekend",
      worked: false,
      work_start: null,
      work_end: null,
      wake: 470,
      bed: 1390,
      breakfast_time: 500,
      lunch_time: 760,
      dinner_time: 1100,
      ...weekend,
    }),
  );
  return [...weekday, ...we];
}

type Over = { placeNodes?: Record<string, string>; day?: number; slot?: number; routine?: number };

function makeWorld(o: Over = {}): World {
  return WorldSchema.parse({
    person_id: "p-test",
    attributes: { country: "US", occupation_group: "professional", work_pattern: "day_shift" },
    distribution_source: "atus_2025",
    nodes: {
      home: { kind: "home" },
      kitchen: { kind: "kitchen", default: "home" },
      workplace: { kind: "workplace", job: "clerk" },
      hobby_a: { kind: "hobby", activity: "climbing", setting: "venue_indoor", weekly_freq: 2, tags: ["sport_climbing"] },
      hobby_b: { kind: "hobby", activity: "cycling", setting: "outdoor", weekly_freq: 1 },
      diner: { kind: "frequent_place", setting: "venue_indoor", optional: true },
      friend_home: { kind: "frequent_place", setting: "home_indoor", optional: true },
    },
    node_binding: {
      home: { node: "h" },
      workplace: { node: "w" },
      hobby_a: { node: "a" },
      hobby_b: { node: "b" },
      diner: { node: "d" },
      friend_home: { node: "f" },
    },
    meal_style: {
      workday: { lunch: { workplace: 0.8, eat_out: 0.2 }, dinner: { home: 0.8, eat_out: 0.1, others_home: 0.1 } },
      weekend: {
        breakfast: { home: 0.9, eat_out: 0.1 },
        lunch: { home: 0.6, eat_out: 0.3, others_home: 0.1 },
        dinner: { home: 0.7, eat_out: 0.2, others_home: 0.1 },
      },
      breakfast_skip_prob: 0.3,
      place_nodes: o.placeNodes ?? { eat_out: "diner", others_home: "friend_home" },
    },
    dials: {
      length_days: 28,
      routine_strength: o.routine ?? 0.6,
      exception_rate: { slot: o.slot ?? 0.15, day: o.day ?? 0.1 },
      wake_sleep: "default_v1",
    },
    seed: 20261112,
  });
}

function input(o: Over = {}, weekend: Partial<DonorRow> = {}, weekday: Partial<DonorRow> = {}): ScheduleInput {
  counter = 0;
  const world = makeWorld(o);
  return {
    world,
    templates: TemplatesSchema.parse(readConfig("timeline/templates.v1.json")),
    gaps: GapTableSchema.parse(readConfig("timeline/gaps.v1.json")) as GapTable,
    exceptions: ExceptionCatalogSchema.parse(readConfig("timeline/exceptions.v1.json")),
    source: atus2025Source(rows(weekend, weekday)),
    boundNodes: new Set(Object.keys(world.node_binding)),
    defaults: ScheduleDefaultsSchema.parse(readConfig("schedule_defaults_atus2025.json")),
  };
}

const json = (x: unknown) => JSON.stringify(x, null, 2);
const ORIGINAL_TZ = process.env.TZ;
afterEach(() => {
  process.env.TZ = ORIGINAL_TZ;
});

describe("generateSchedule", () => {
  it("byte-identical across runs and time zones", () => {
    const a = json(generateSchedule(input()));
    const b = json(generateSchedule(input()));
    expect(a).toBe(b);
    process.env.TZ = "UTC";
    const utc = json(generateSchedule(input()));
    process.env.TZ = "America/New_York";
    const ny = json(generateSchedule(input()));
    expect(utc).toBe(a);
    expect(ny).toBe(a);
  });

  it("28 days, slot ids unique, windows inside [wake, sleep], transitions match node changes", () => {
    const schedule = generateSchedule(input());
    expect(ScheduleSchema.parse(schedule)).toEqual(schedule);
    expect(schedule.generator_version).toBe("schedule.v1");
    expect(schedule.template_version).toBe("templates.v1");
    expect(schedule.gap_table_version).toBe("gaps.v1");
    expect(schedule.exception_catalog_version).toBe("exceptions.v1");
    expect(schedule.days.map((d) => d.day_index)).toEqual(Array.from({ length: 28 }, (_, i) => i + 1));

    const ids = schedule.days.flatMap((d) => d.slots.map((s) => s.slot_id));
    expect(new Set(ids).size).toBe(ids.length);

    for (const day of schedule.days) {
      const wake = parseClock(day.wake);
      const sleep = parseClock(day.sleep);
      const win = day.slots.map((s) => s.window.map(parseClock) as [number, number]);
      expect(win[0]![0]).toBe(wake);
      for (const [start, end] of win) {
        expect(start).toBeGreaterThanOrEqual(wake);
        expect(end).toBeLessThanOrEqual(sleep);
        expect(end).toBeGreaterThan(start);
      }
      const expected: { from_slot: string; to_slot: string }[] = [];
      for (let i = 1; i < day.slots.length; i++) {
        const prev = day.slots[i - 1]!;
        const next = day.slots[i]!;
        expect(win[i]![0]).toBeGreaterThanOrEqual(win[i - 1]![1]);
        if (prev.spec.node !== next.spec.node) expected.push({ from_slot: prev.slot_id, to_slot: next.slot_id });
      }
      expect(day.transitions.map(({ from_slot, to_slot }) => ({ from_slot, to_slot }))).toEqual(expected);
      for (const t of day.transitions) {
        const i = day.slots.findIndex((s) => s.slot_id === t.to_slot);
        expect(win[i]![0] - win[i - 1]![1]).toBeGreaterThanOrEqual(t.minutes);
        expect(t.minutes).toBeGreaterThan(0);
      }
    }
  });

  it("generateDay at attempt 1 differs from attempt 0 and leaves the other days untouched", () => {
    const inp = input();
    const schedule = generateSchedule(inp);
    const d = schedule.days.find((day) => day.day_type === "workday" && day.attempt === 0)!.day_index;
    const retried = generateDay(inp, schedule.donors, d, 1);
    expect(retried.attempt).toBe(1);
    expect(retried.day_index).toBe(d);
    expect(json(retried)).not.toBe(json(schedule.days[d - 1]));
    for (const day of schedule.days) {
      expect(json(generateDay(inp, schedule.donors, day.day_index, day.attempt))).toBe(json(day));
    }
    expect(json(generateSchedule(inp))).toBe(json(schedule));
  });

  it("meal_style_effective drops eat_out without place_nodes.eat_out and sums to 1", () => {
    const schedule = generateSchedule(input({ placeNodes: { others_home: "friend_home" } }));
    const eff = schedule.meal_style_effective;
    const dists = [eff.workday.lunch, eff.workday.dinner, eff.weekend.breakfast, eff.weekend.lunch, eff.weekend.dinner];
    for (const dist of dists) {
      expect(dist).not.toHaveProperty("eat_out");
      expect(Object.values(dist).reduce((s, p) => s + p, 0)).toBeCloseTo(1, 12);
    }
    expect(eff.workday.lunch).toEqual({ workplace: 1 });
    for (const day of schedule.days) {
      for (const slot of day.slots) expect(slot.spec.node).not.toBe("diner");
    }
  });

  it("uses the weekend donor for leave days and records day exceptions", () => {
    // Weekend donors wake at 10:00, weekday donors around 06:45; routine 1 keeps σ at 20 min.
    const schedule = generateSchedule(input({ day: 1, routine: 1 }, { wake: 600, breakfast_time: 630 }));
    const leave = schedule.days.filter((d) => d.day_type === "leave_day");
    expect(leave.length).toBeGreaterThan(0);
    for (const day of leave) {
      expect(parseClock(day.wake)).toBeGreaterThan(540);
      expect(day.exceptions).toEqual([{ scope: "day", type: "leave_day", replaces: "workday", day_range: [day.day_index, day.day_index] }]);
      expect(day.slots.map((s) => s.role)).toContain("chores");
      expect(day.slots.some((s) => s.spec.category === "work")).toBe(false);
    }
  });

  it("workday hobby with no room before sleep is dropped as no_time; required slots stay (R21)", () => {
    // Weekday donors go to bed at 21:00, so a 90-min hobby after dinner never fits.
    const schedule = generateSchedule(input({ day: 0, slot: 0, routine: 1 }, {}, { bed: 1260 }));
    const workdays = schedule.days.filter((d) => d.day_type === "workday");
    const hobbyDropped = workdays.filter((d) => d.dropped_slots.some((x) => x.role === "hobby"));
    expect(hobbyDropped.length).toBeGreaterThan(0);
    for (const day of workdays) {
      for (const x of day.dropped_slots) expect(x.reason).toBe("no_time");
      const roles = day.slots.map((s) => s.role);
      for (const role of ["home_morning", "work_am", "work_pm"]) expect(roles).toContain(role);
      expect(day.slots.every((s) => parseClock(s.window[1]) <= parseClock(day.sleep))).toBe(true);
    }
    expect(schedule.days.filter((d) => d.day_type !== "workday").every((d) => d.dropped_slots.length === 0)).toBe(true);
  });

  it("a short work window keeps work_pm at least 60 min (R14, R22)", () => {
    // Donor work spans 100 min with a late lunch; sanitizing stretches work and pulls lunch in.
    const inp = input({ day: 0, slot: 0, routine: 1 }, {}, { work_start: 540, work_end: 640, lunch_time: 900 });
    const schedule = generateSchedule(inp);
    let workplaceLunches = 0;
    for (const day of schedule.days.filter((d) => d.day_type === "workday")) {
      const decided = decideDay(inp.world, inp.exceptions, { boundNodes: inp.boundNodes }, day.day_index, day.attempt);
      if (decided.meal_place.lunch === "workplace") {
        workplaceLunches += 1;
        expect(day.slots.map((x) => x.role)).toContain("lunch");
      }
      const pm = day.slots.find((s) => s.role === "work_pm")!;
      const [s, e] = pm.window.map(parseClock) as [number, number];
      expect(e - s).toBeGreaterThanOrEqual(60);
    }
    expect(workplaceLunches).toBeGreaterThan(0);
  });

  it("a weekend donor with sport at 22:00 and bed at 22:50 yields feasible hobby days (R25)", () => {
    const schedule = generateSchedule(
      input({ day: 0, slot: 0, routine: 1 }, { sport_has: true, sport_time: 1320, bed: 1370 }),
    );
    const hobbyDays = schedule.days.filter((d) => d.day_type === "hobby_day");
    expect(hobbyDays.length).toBe(8);
    for (const day of hobbyDays) {
      const hobby = day.slots.find((s) => s.role === "hobby")!;
      expect(parseClock(hobby.window[1])).toBeLessThanOrEqual(parseClock(day.sleep) - 60);
    }
  });

  it("a donor going to bed at 27:55 with σ 45 never sleeps past 27:59 and every slot window parses (R44)", () => {
    const inp = input({ day: 0, slot: 0, routine: 0 }, { bed: 1675 }, { bed: 1675 });
    const schedule = generateSchedule(inp);
    expect(ScheduleSchema.parse(schedule)).toEqual(schedule);
    expect(Math.max(...schedule.days.map((d) => parseClock(d.sleep)))).toBe(1679);
    // The matcher parses every slot window (slotWindow); "28:xx" used to throw there.
    const t = createTimeline(inp.world, schedule, () => "tl-late");
    const probe: Clip = {
      clip_id: "probe",
      category: "home",
      setting: "home_indoor",
      node: "h",
      activity: "probe",
      time_of_day: "unknown",
      tags: [],
      source: { dataset: "nextme800", video_id: "v", media: { kind: "frames", approx_fps: 1, tar: "t.tar" } },
      source_start_s: 0,
      source_end_s: 60,
      day_group: "g",
      person_source: "p",
      fingerprint: null,
      extractor_version: "fixture.v1",
    };
    const catalog: ClipCatalog = { clips: () => [probe], getClip: () => probe, getAnnotations: () => [] };
    for (const day of schedule.days) {
      for (const slot of day.slots) expect(() => hardFilterFailures(t, catalog, slot.slot_id, probe, [])).not.toThrow();
    }
  });

  it("a donor waking at 03:20 wakes at 04:00 or later (R44)", () => {
    const schedule = generateSchedule(input({ day: 0, slot: 0, routine: 1 }, {}, { wake: 200, breakfast_time: 220 }));
    const wakes = schedule.days.map((d) => parseClock(d.wake));
    expect(Math.min(...wakes)).toBe(240);
    for (const day of schedule.days) expect(day.slots[0]!.spec.time_of_day).not.toBe("night");
  });

  it("schedule clocks accept 00:00–27:59 and reject 28:00 and minute 60 (R44)", () => {
    const day = generateSchedule(input()).days[0]!;
    expect(ScheduleDaySchema.safeParse({ ...day, sleep: "27:59" }).success).toBe(true);
    expect(ScheduleDaySchema.safeParse({ ...day, sleep: "28:00" }).success).toBe(false);
    expect(ScheduleDaySchema.safeParse({ ...day, wake: "06:60" }).success).toBe(false);
    const slot = day.slots[0]!;
    expect(ScheduleDaySchema.safeParse({ ...day, slots: [{ ...slot, window: [slot.window[0], "28:10"] }] }).success).toBe(false);
  });

  it("a slot exception whose slot is dropped for time is removed from the day (R45)", () => {
    // Work spans 4 h with lunch pinned to its end: an eat-out lunch adds transitions and leaves work_pm < 60 min.
    const inp = input({ day: 0, slot: 1, routine: 1 }, {}, { work_start: 540, work_end: 640, lunch_time: 900 });
    const schedule = generateSchedule(inp);
    const lunchDropped = schedule.days.filter((d) => d.dropped_slots.some((x) => x.role === "lunch"));
    expect(lunchDropped.length).toBeGreaterThan(0);
    let slotExceptions = 0;
    for (const day of schedule.days) {
      for (const ex of day.exceptions.filter((e) => e.scope === "slot")) {
        slotExceptions += 1;
        expect(day.slots.map((s) => s.role), `day ${day.day_index} ${ex.type}`).toContain(ex.replaces);
      }
    }
    expect(slotExceptions).toBeGreaterThan(0);
    for (const day of lunchDropped) expect(day.exceptions.some((e) => e.replaces === "lunch")).toBe(false);
    const empty: ClipCatalog = { clips: () => [], getClip: () => undefined, getAnnotations: () => [] };
    const report = verifyTimeline(createTimeline(inp.world, schedule, () => "tl-exc"), empty);
    expect(report.rules.exceptions).toEqual({ pass: true, violations: [] });
  });

  it("rethrows SCHEDULE_INFEASIBLE naming the day after the 5th retry", () => {
    let err: unknown;
    try {
      generateSchedule(input({ day: 0, slot: 0 }, { bed: 480 }));
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(TimelineError);
    expect((err as TimelineError).code).toBe("SCHEDULE_INFEASIBLE");
    expect((err as TimelineError).details).toMatchObject({ day: 6, attempts: 6 });
    expect((err as TimelineError).message).toContain("day 6");
  });
});
