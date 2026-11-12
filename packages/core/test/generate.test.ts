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
  realizedOf,
  rngFor,
  selectDonors,
  verifyTimeline,
  type Clip,
  type ClipCatalog,
  type DonorRow,
  type GapTable,
  type ScheduleInput,
  type World,
} from "../src/index.js";
import { makeDay } from "./fixtures.js";

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

type Over = { placeNodes?: Record<string, string>; day?: number; slot?: number; routine?: number; seed?: number; length?: number };

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
      rest_day: {
        breakfast: { home: 0.9, eat_out: 0.1 },
        lunch: { home: 0.6, eat_out: 0.3, others_home: 0.1 },
        dinner: { home: 0.7, eat_out: 0.2, others_home: 0.1 },
      },
      breakfast_skip_prob: 0.3,
      place_nodes: o.placeNodes ?? { eat_out: "diner", others_home: "friend_home" },
    },
    dials: {
      length_days: o.length ?? 28,
      routine_strength: o.routine ?? 0.6,
      exception_rate: { slot: o.slot ?? 0.15, day: o.day ?? 0.1 },
      wake_sleep: "default_v1",
    },
    seed: o.seed ?? 20261112,
  });
}

function input(
  o: Over = {},
  weekend: Partial<DonorRow> = {},
  weekday: Partial<DonorRow> = {},
  pool?: DonorRow[],
): ScheduleInput {
  counter = 0;
  const world = makeWorld(o);
  return {
    world,
    templates: TemplatesSchema.parse(readConfig("timeline/templates.v2.json")),
    gaps: GapTableSchema.parse(readConfig("timeline/gaps.v1.json")) as GapTable,
    exceptions: ExceptionCatalogSchema.parse(readConfig("timeline/exceptions.v1.json")),
    source: atus2025Source(pool ?? rows(weekend, weekday)),
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
    expect(schedule.template_version).toBe("templates.v2");
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

  it("parses the rev3 defaults: weekday_evening_home and rest_day meal priors", () => {
    const d = ScheduleDefaultsSchema.parse(readConfig("schedule_defaults_atus2025.json"));
    expect(d.weekday_evening_home.slot_probability).toBe(0.91);
    expect(d.weekday_evening_home.minutes).toMatchObject({ p25: 105, p50: 187, p75: 270 });
    expect(d.meal_location_priors.rest_day).toHaveProperty("lunch");
  });

  it("meal_style_effective drops eat_out without place_nodes.eat_out and sums to 1", () => {
    const schedule = generateSchedule(input({ placeNodes: { others_home: "friend_home" } }));
    const eff = schedule.meal_style_effective;
    const dists = [eff.workday.lunch, eff.workday.dinner, eff.rest_day.breakfast, eff.rest_day.lunch, eff.rest_day.dinner];
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

  it("workday hobby with no room before sleep is shrunk or dropped; required slots stay (R21, E4)", () => {
    // Weekday donors go to bed at 21:00, so a 90-min hobby after dinner never fits.
    const schedule = generateSchedule(input({ day: 0, slot: 0, routine: 1 }, {}, { bed: 1260 }));
    const workdays = schedule.days.filter((d) => d.day_type === "workday");
    const hobbyDropped = workdays.filter((d) => d.dropped_slots.some((x) => x.role === "hobby"));
    expect(hobbyDropped.length).toBeGreaterThan(0);
    for (const day of workdays) {
      for (const x of day.dropped_slots) expect(["no_time", "shrunk"]).toContain(x.reason);
      const roles = day.slots.map((s) => s.role);
      for (const role of ["home_morning", "work_am", "work_pm", "dinner"]) expect(roles).toContain(role);
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
      const decided = decideDay(inp.world, inp.exceptions, { boundNodes: inp.boundNodes, defaults: inp.defaults }, day.day_index, day.attempt);
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
    // E6: late sleeps are redrawn, then clamped, so the bound holds without every day sitting on it.
    expect(Math.max(...schedule.days.map((d) => parseClock(d.sleep)))).toBeLessThanOrEqual(1679);
    expect(Math.max(...schedule.days.map((d) => parseClock(d.sleep)))).toBeGreaterThan(1600);
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
    expect(Math.min(...wakes)).toBeGreaterThanOrEqual(240);
    expect(Math.min(...wakes)).toBeLessThan(270);
    for (const day of schedule.days) expect(day.slots[0]!.spec.time_of_day).not.toBe("night");
  });

  it("every workday keeps its required dinner (CR2)", () => {
    for (const o of [{ routine: 0 }, { routine: 0.5 }, { routine: 1 }]) {
      const schedule = generateSchedule(input({ ...o, day: 0 }));
      const workdays = schedule.days.filter((d) => d.day_type === "workday");
      expect(workdays.length).toBeGreaterThan(0);
      for (const day of workdays) {
        const dinner = day.slots.find((s) => s.role === "dinner");
        expect(dinner, `day ${day.day_index}`).toBeDefined();
        expect(dinner!.required).toBe(true);
      }
    }
  });

  it("a jitter that pushes wake below 04:00 is redrawn, not clamped (E6, G13)", () => {
    // Weekday donors wake at 04:10; with σ 20 min the first jitter draw is below 04:00 on about a third of workdays.
    const inp = input({ day: 0, slot: 0, routine: 1 }, {}, { wake: 250, breakfast_time: 330 });
    const schedule = generateSchedule(inp);
    const sigma = inp.defaults.day_level_sigma.routine_strength_1;
    const firstDrawEarly = schedule.days.filter(
      (d) =>
        d.day_type === "workday" &&
        Math.round(250 + rngFor(inp.world.seed, "day", d.day_index, d.attempt, "jitter", "wake", 0).normal(0, sigma)) < 240,
    );
    expect(firstDrawEarly.length).toBeGreaterThan(0);
    for (const day of firstDrawEarly) expect(day.jitter_redraws).toBeGreaterThan(0);
    for (const day of schedule.days) {
      expect(day.anchor_adjustments).toEqual([]);
      expect(parseClock(day.wake)).toBeGreaterThanOrEqual(240);
    }
  });

  it("a workday does not check breakfast: a breakfast jitter before wake + 10 is neither redrawn nor adjusted (G12)", () => {
    // Weekday donors eat breakfast 20-40 min before waking (the donor clamp moves it to wake + 10);
    // σ 20 then puts the day's breakfast before wake + 10 on about half the workdays.
    const inp = input({ day: 0, slot: 0, routine: 1 }, {}, { breakfast_time: 380 });
    const schedule = generateSchedule(inp);
    const p = schedule.donors.weekday.profile;
    const sigma = inp.defaults.day_level_sigma.routine_strength_1;
    const first = (d: { day_index: number; attempt: number }, item: string, base: number) =>
      Math.round(base + rngFor(inp.world.seed, "day", d.day_index, d.attempt, "jitter", item, 0).normal(0, sigma));
    const workdays = schedule.days.filter((d) => d.day_type === "workday");
    const early = workdays.filter((d) => first(d, "breakfast", p.breakfast) < first(d, "wake", p.wake) + 10);
    expect(early.length).toBeGreaterThan(0);
    for (const day of workdays) {
      expect(day.jitter_redraws).toBe(0);
      expect(day.anchor_adjustments).toEqual([]);
    }
  });

  it("per-item redraw keeps the rest-day wake unbiased when only breakfast violates (G13)", () => {
    // Weekend donors wake 07:50 with breakfast 08:20; at σ 45 breakfast is before wake + 10 on about a
    // third of first draws. Whole-vector redraw pulled the accepted wake earlier; per-item redraw must not.
    const inp = input({ day: 0, slot: 0, routine: 0 });
    const { donors } = generateSchedule(inp);
    const p = donors.weekend.profile;
    expect(p.breakfast - p.wake).toBe(30);
    let n = 0;
    let shift = 0;
    let redrawn = 0;
    for (let d = 1; n < 2000; d++) {
      const day = generateDay(inp, donors, d, 0);
      if (day.day_type === "workday") continue;
      n += 1;
      shift += parseClock(day.wake) - p.wake;
      if (day.jitter_redraws > 0) redrawn += 1;
    }
    expect(redrawn / n).toBeGreaterThan(0.2);
    expect(Math.abs(shift / n)).toBeLessThanOrEqual(3);
  });

  it("a huge σ exhausts the redraws and records the clamps as anchor_adjustments (E6)", () => {
    const base = input({ day: 0, slot: 0, routine: 1 });
    const { donors } = generateSchedule(base);
    const wild: ScheduleInput = {
      ...base,
      defaults: { ...base.defaults, day_level_sigma: { routine_strength_0: 400, routine_strength_1: 400 } },
    };
    const adjusted = [];
    for (let d = 1; d <= 28; d++) {
      try {
        const day = generateDay(wild, donors, d, 0);
        if (day.anchor_adjustments.length > 0) adjusted.push(day);
      } catch (e) {
        if (!(e instanceof TimelineError && e.code === "SCHEDULE_INFEASIBLE")) throw e;
      }
    }
    expect(adjusted.length).toBeGreaterThan(0);
    for (const day of adjusted) {
      expect(parseClock(day.wake)).toBeGreaterThanOrEqual(240);
      expect(parseClock(day.sleep)).toBeLessThanOrEqual(1679);
      for (const a of day.anchor_adjustments) expect(a.from).not.toBe(a.to);
    }
  });

  it("a weekday donor whose required workday slots cannot lay out is redrawn (E5)", () => {
    // Bad rows end work at 21:40 with dinner at 21:30 and bed at 22:40: the bounds hold, but dinner
    // starts after work_end + 20 min and runs past bed. Good rows carry 4x the weight.
    counter = 0;
    const bad = Array.from({ length: 20 }, (_, i) =>
      row({ id: `bad${i}`, w: 1, work_end: 1300, dinner_time: 1290, bed: 1360 }),
    );
    const good = Array.from({ length: 20 }, (_, i) => row({ id: `good${i}`, w: 4 }));
    const weekend = Array.from({ length: 40 }, () =>
      row({ daytype: "weekend", worked: false, work_start: null, work_end: null, wake: 470, bed: 1390, breakfast_time: 500 }),
    );
    const pool = [...bad, ...good, ...weekend];
    const src = atus2025Source(pool);
    const attrs = makeWorld().attributes;
    let seed = 1;
    while (!selectDonors(attrs, src, { seed, allowCountryFallback: false }).weekday.case_id.startsWith("bad")) seed++;

    const schedule = generateSchedule(input({ day: 0, slot: 0, routine: 1, seed }, {}, {}, pool));
    expect(schedule.donors.weekday.redraws).toBeGreaterThan(0);
    expect(schedule.donors.weekday.case_id).toMatch(/^good/);
    expect(schedule.donors.weekday.adjustments).toEqual([]);
  });

  it("a weekday donor working past bedtime on every draw gets an earlier work_end, recorded (G9)", () => {
    // Every weekday row works until 23:00 and goes to bed at 23:18: the bounds hold, but the
    // required dinner (60 min after work_end + gap) never fits. G11: p90 workplace->home gap
    // 55 min and sleep - σ (20 at routine 1) give work_end <= 23:18 - 20 - 55 - 60 = 21:03.
    const inp = input({ day: 0, slot: 0, routine: 1 }, {}, { work_end: 1380, dinner_time: 1140, bed: 1398 });
    const schedule = generateSchedule(inp);
    const { weekday } = schedule.donors;
    expect(weekday.redraws).toBe(5);
    expect(weekday.profile.work_end).toBe(1263);
    expect(weekday.adjustments).toEqual([{ item: "work_end", from: 1380, to: 1263 }]);
    for (const day of schedule.days.filter((d) => d.day_type === "workday")) {
      expect(day.slots.map((s) => s.role)).toContain("dinner");
    }
  });

  it("the past-bedtime donor repaired with slack generates an 84-day world (G11)", () => {
    for (const routine of [0, 1]) {
      const inp = input({ day: 0, slot: 0, routine, length: 84 }, {}, { work_end: 1380, dinner_time: 1140, bed: 1398 });
      const schedule = generateSchedule(inp);
      expect(schedule.days).toHaveLength(84);
      expect(schedule.donors.weekday.adjustments.map((a) => a.item)).toEqual(["work_end"]);
    }
  });

  it("a weekday donor whose required slots fit keeps adjustments [] (G9)", () => {
    const { weekday } = generateSchedule(input({ day: 0, slot: 0, routine: 1 })).donors;
    expect(weekday.redraws).toBe(0);
    expect(weekday.adjustments).toEqual([]);
  });

  it("schedule clocks accept 00:00–27:59 and reject 28:00 and minute 60 (R44)", () => {
    const day = generateSchedule(input()).days[0]!;
    expect(ScheduleDaySchema.safeParse({ ...day, sleep: "27:59" }).success).toBe(true);
    expect(ScheduleDaySchema.safeParse({ ...day, sleep: "28:00" }).success).toBe(false);
    expect(ScheduleDaySchema.safeParse({ ...day, wake: "06:60" }).success).toBe(false);
    const slot = day.slots[0]!;
    expect(ScheduleDaySchema.safeParse({ ...day, slots: [{ ...slot, window: [slot.window[0], "28:10"] }] }).success).toBe(false);
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

const emptyClips: ClipCatalog = { clips: () => [], getClip: () => undefined, getAnnotations: () => [] };

/** Node a meal place resolves to, as the scheduler does. */
const placeNode = (world: World, place: string): string | undefined =>
  place === "home" || place === "workplace" ? place : world.meal_style.place_nodes[place as "eat_out" | "others_home"];

describe("slot exceptions after windows (CR1, E1, E3)", () => {
  it("exception_rate.slot 0.5 over 40 seeds: the mean realized slot_exception_rate is within 0.05 of 0.5", () => {
    const rates: number[] = [];
    for (let seed = 1; seed <= 40; seed++) {
      const { realized } = generateSchedule(input({ seed, slot: 0.5 }));
      expect(realized.slot_exception_eligible_days).toBeGreaterThan(0);
      rates.push(realized.slot_exception_rate!);
    }
    const mean = rates.reduce((s, r) => s + r, 0) / rates.length;
    expect(Math.abs(mean - 0.5)).toBeLessThanOrEqual(0.05);
  });

  it("every slot exception replaces a kept slot whose node is the place node, and verify passes", () => {
    let count = 0;
    for (let seed = 1; seed <= 10; seed++) {
      const inp = input({ seed, slot: 0.5 });
      const schedule = generateSchedule(inp);
      for (const day of schedule.days) {
        for (const ex of day.exceptions.filter((e) => e.scope === "slot")) {
          count += 1;
          expect(day.exceptions).toHaveLength(1);
          expect(day.slot_exception_eligible).toBe(true);
          expect(ex.day_range).toEqual([day.day_index, day.day_index]);
          const slot = day.slots.find((s) => s.role === ex.replaces);
          expect(slot, `day ${day.day_index} ${ex.type}`).toBeDefined();
          expect(slot!.spec.node).toBe(placeNode(inp.world, ex.meal_place!));
        }
      }
      const report = verifyTimeline(createTimeline(inp.world, schedule, () => `tl-exc-${seed}`), emptyClips);
      expect(report.rules.exceptions).toEqual({ pass: true, violations: [] });
    }
    expect(count).toBeGreaterThan(0);
  });

  it("exception_rate.slot 1 places one on every eligible day, never on a slot that does not fit (R45 removed)", () => {
    // Work spans 100 min with lunch pinned late: an eat-out lunch adds transitions and squeezes work_pm,
    // so lunch_out is infeasible on some workdays while dinner_out still fits.
    const inp = input({ day: 0, slot: 1, routine: 1 }, {}, { work_start: 540, work_end: 640, lunch_time: 900 });
    const schedule = generateSchedule(inp);
    expect(schedule.realized.slot_exception_rate).toBe(1);
    for (const day of schedule.days) {
      const slotExc = day.exceptions.filter((e) => e.scope === "slot");
      expect(slotExc).toHaveLength(day.slot_exception_eligible ? 1 : 0);
      for (const ex of slotExc) expect(day.slots.map((s) => s.role)).toContain(ex.replaces);
    }
    const workdays = schedule.days.filter((d) => d.day_type === "workday");
    expect(workdays.some((d) => d.slot_exception_candidates === 1)).toBe(true);
    expect(workdays.every((d) => d.slots.some((s) => s.role === "dinner"))).toBe(true);
  });

  it("no day has both a day and a slot exception; leave days are not eligible", () => {
    let leave = 0;
    for (let seed = 0; seed < 10; seed++) {
      const schedule = generateSchedule(input({ seed, day: 0.5, slot: 1 }));
      for (const day of schedule.days) {
        expect(day.exceptions.length).toBeLessThanOrEqual(1);
        if (day.day_type === "leave_day") {
          leave += 1;
          expect(day.exceptions[0]!.scope).toBe("day");
          expect(day.slot_exception_eligible).toBe(false);
          expect(day.slot_exception_candidates).toBe(0);
        }
      }
    }
    expect(leave).toBeGreaterThan(0);
  });

  it("without an eat_out node only rest-day lunch_at_friend remains; without any place node nothing is eligible", () => {
    const friendOnly = generateSchedule(input({ day: 0, slot: 1, placeNodes: { others_home: "friend_home" } }));
    for (const day of friendOnly.days) {
      const slotExc = day.exceptions.filter((e) => e.scope === "slot");
      if (day.day_type === "workday") expect(slotExc).toEqual([]);
      for (const ex of slotExc) expect(ex).toMatchObject({ type: "lunch_at_friend", replaces: "lunch", meal_place: "others_home" });
    }
    expect(friendOnly.days.some((d) => d.exceptions.length > 0)).toBe(true);

    const none = generateSchedule(input({ day: 0, slot: 1, placeNodes: {} }));
    expect(none.days.every((d) => d.exceptions.length === 0 && !d.slot_exception_eligible)).toBe(true);
    expect(none.realized.slot_exception_eligible_days).toBe(0);
    expect(none.realized.slot_exception_rate).toBeNull();
  });

  it("day exceptions do not depend on exception_rate.slot", () => {
    const a = generateSchedule(input({ slot: 0 }));
    const b = generateSchedule(input({ slot: 0.5 }));
    const dayExc = (s: typeof a) => s.days.map((d) => [d.day_type, d.exceptions.filter((e) => e.scope === "day")]);
    expect(dayExc(b)).toEqual(dayExc(a));
    expect(b.realized.day_exception_rate).toBe(a.realized.day_exception_rate);
    expect(a.days.some((d) => d.day_type === "leave_day")).toBe(true);
    expect(b.days.some((d) => d.exceptions.some((e) => e.scope === "slot"))).toBe(true);
  });
});

describe("realized (CR1, E1, E2)", () => {
  it("rates, presence and shrink counts equal counts recomputed from days; dials are echoed", () => {
    const inp = input({ seed: 7, slot: 0.3, day: 0.2, length: 56 }, {}, { bed: 1290 });
    const schedule = generateSchedule(inp);
    const { days, realized } = schedule;
    const slotDays = days.filter((d) => d.exceptions.some((e) => e.scope === "slot")).length;
    const eligible = days.filter((d) => d.slot_exception_eligible).length;
    const slotExc = days.flatMap((d) => d.exceptions.filter((e) => e.scope === "slot")).length;
    const candidates = days.reduce((s, d) => s + d.slot_exception_candidates, 0);
    expect(eligible).toBeGreaterThan(0);
    expect(realized.slot_exception_eligible_days).toBe(eligible);
    expect(realized.slot_exception_rate).toBe(slotDays / eligible);
    expect(realized.slot_exception_share).toBe(slotExc / candidates);

    const base = (d: (typeof days)[number]) => d.exceptions.find((e) => e.scope === "day")?.replaces ?? d.day_type;
    const leave = days.filter((d) => d.day_type === "leave_day").length;
    expect(leave).toBeGreaterThan(0);
    expect(realized.day_exception_rate).toBe(leave / days.filter((d) => base(d) === "workday").length);

    const templateOf = (d: (typeof days)[number]) => (d.day_type === "leave_day" ? "home_day" : d.day_type);
    const presence: Record<string, [number, number]> = {};
    for (const d of days) {
      for (const slot of inp.templates.day_types[templateOf(d)].filter((s) => !s.required)) {
        const c = (presence[slot.role] ??= [0, 0]);
        c[1] += 1;
        if (d.slots.some((s) => s.role === slot.role)) c[0] += 1;
      }
    }
    expect(Object.keys(realized.optional_presence).sort()).toEqual(Object.keys(presence).sort());
    for (const [role, [kept, total]] of Object.entries(presence)) expect(realized.optional_presence[role]).toBe(kept / total);
    expect(realized.optional_presence.home_evening).toBeLessThan(1);

    const shrunk = days.filter(
      (d) => d.dropped_slots.some((x) => x.role === "hobby" && x.reason === "shrunk") && d.slots.some((s) => s.role === "hobby"),
    ).length;
    expect(shrunk).toBeGreaterThan(0);
    expect(realized.shrunk).toEqual({ hobby: shrunk });

    expect(realized.dials).toEqual({ routine_strength: 0.6, exception_rate: { slot: 0.3, day: 0.2 } });
    expect(Object.keys(realized.composition_consistency)).toEqual(["workday", "hobby_day", "home_day"]);
  });

  it("composition_consistency on hand-built days: Jaccard, order similarity, start differences (E2)", () => {
    const inp = input();
    const a = makeDay(1);
    const b = makeDay(2);
    // Day 2 has no lunch and starts work_pm 10 min later.
    const noLunch = { ...b, slots: b.slots.filter((s) => s.role !== "lunch") };
    noLunch.slots = noLunch.slots.map((s) => (s.role === "work_pm" ? { ...s, window: ["13:10", "17:00"] as [string, string] } : s));
    const r = realizedOf([a, noLunch, makeDay(6)], inp.world, inp.templates);
    // Fixture workdays: optional lunch, dinner, home_evening; 6 roles vs 5.
    const cc = r.composition_consistency.workday!;
    expect(cc.pairs).toBe(1);
    expect(cc.optional_jaccard).toBeCloseTo(2 / 3, 12);
    expect(cc.order_similarity).toBeCloseTo(1 - 1 / 6, 12);
    expect(cc.start_diff_minutes).toBeCloseTo(10 / 5, 12);
    // A template used on fewer than 2 days is omitted.
    expect(r.composition_consistency).not.toHaveProperty("hobby_day");
    // Lunch is optional in both templates: kept on day 1 only of the two workdays and the hobby day.
    expect(r.optional_presence.lunch).toBe(1 / 3);
    // Dinner is required on workdays, so only the hobby day counts.
    expect(r.optional_presence.dinner).toBe(1);
  });

  it("composition_consistency rises with routine_strength on workdays (E2 monotonicity)", () => {
    const seeds = [1, 2, 3, 4, 5, 6, 7, 8];
    const avg = (routine: number) => {
      const sum = { optional_jaccard: 0, order_similarity: 0, start_diff_minutes: 0 };
      for (const seed of seeds) {
        const cc = generateSchedule(input({ seed, routine })).realized.composition_consistency.workday!;
        sum.optional_jaccard += cc.optional_jaccard / seeds.length;
        sum.order_similarity += cc.order_similarity / seeds.length;
        sum.start_diff_minutes += cc.start_diff_minutes! / seeds.length;
      }
      return sum;
    };
    const [r0, r5, r1] = [0, 0.5, 1].map(avg) as [ReturnType<typeof avg>, ReturnType<typeof avg>, ReturnType<typeof avg>];
    expect(r5.optional_jaccard).toBeGreaterThanOrEqual(r0.optional_jaccard);
    expect(r1.optional_jaccard).toBeGreaterThanOrEqual(r5.optional_jaccard);
    expect(r5.order_similarity).toBeGreaterThanOrEqual(r0.order_similarity);
    expect(r1.order_similarity).toBeGreaterThanOrEqual(r5.order_similarity);
    expect(r5.start_diff_minutes).toBeLessThanOrEqual(r0.start_diff_minutes);
    expect(r1.start_diff_minutes).toBeLessThanOrEqual(r5.start_diff_minutes);
  });
});
