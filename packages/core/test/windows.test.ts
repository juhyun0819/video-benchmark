import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  TemplatesSchema,
  TimelineError,
  WorldSchema,
  resolveDayWindows,
  type DayAnchors,
  type DayDecisions,
  type PlaceKind,
  type Templates,
  type World,
} from "../src/index.js";

function templates(): Templates {
  const url = new URL("../../../config/timeline/templates.v1.json", import.meta.url);
  return TemplatesSchema.parse(JSON.parse(readFileSync(url, "utf8")));
}

function world(): World {
  return WorldSchema.parse({
    person_id: "p-test",
    attributes: { country: "US", occupation_group: "professional", work_pattern: "day_shift" },
    distribution_source: "atus_2025",
    nodes: {
      home: { kind: "home" },
      workplace: { kind: "workplace", job: "clerk" },
      hobby_a: { kind: "hobby", activity: "rock climbing", setting: "venue_indoor", weekly_freq: 1, tags: ["sport_climbing"] },
      diner: { kind: "frequent_place", setting: "venue_indoor", optional: true },
    },
    node_binding: { home: { node: "h" }, workplace: { node: "w" }, hobby_a: { node: "a" }, diner: { node: "d" } },
    meal_style: {
      workday: { lunch: { workplace: 1 }, dinner: { home: 1 } },
      rest_day: { breakfast: { home: 1 }, lunch: { home: 1 }, dinner: { home: 1 } },
      breakfast_skip_prob: 0,
      place_nodes: { eat_out: "diner" },
    },
    dials: {
      length_days: 28,
      routine_strength: 1,
      exception_rate: { slot: 0, day: 0 },
      wake_sleep: "default_v1",
    },
    seed: 1,
  });
}

const ANCHORS: DayAnchors = { wake: 390, work_start: 454, lunch: 720, work_end: 1020, dinner: 1110, sleep: 1338 };

function workday(over: Partial<DayDecisions> = {}): DayDecisions {
  return {
    day_index: 1,
    weekday: "Mon",
    day_type: "workday",
    template: "workday",
    meal_place: { lunch: "workplace", dinner: "home" },
    optional: { hobby: false, home_evening: true },
    exceptions: [],
    ...over,
  };
}

const gap20 = () => 20;

function resolveWorkday(decisions: DayDecisions = workday(), anchors: DayAnchors = ANCHORS) {
  return resolveDayWindows(templates().day_types.workday, anchors, decisions, world(), gap20);
}

const table = (slots: { role: string; window: [string, string]; spec: { node: string } }[]) =>
  slots.map((s) => `${s.role} ${s.window[0]}-${s.window[1]} ${s.spec.node}`);

describe("resolveDayWindows (hand-computed workday)", () => {
  it("produces the expected windows", () => {
    const { slots } = resolveWorkday();
    expect(table(slots)).toEqual([
      "home_morning 06:30-07:14 home",
      "work_am 07:34-12:00 workplace",
      "lunch 12:00-12:45 workplace",
      "work_pm 12:45-17:00 workplace",
      "dinner 18:30-19:30 home",
      "home_evening 19:30-22:18 home",
    ]);
    expect(slots.map((s) => s.slot_id)).toEqual(["d1-s1", "d1-s2", "d1-s3", "d1-s4", "d1-s5", "d1-s6"]);
  });

  it("fills spec setting, category and time_of_day from the node and window start", () => {
    const { slots } = resolveWorkday();
    expect(slots[0]!.spec).toEqual({
      category: "home",
      setting: "home_indoor",
      node: "home",
      activity: "getting ready",
      time_of_day: "early_morning",
      tags: [],
    });
    expect(slots[2]!.spec).toMatchObject({ category: "meal", setting: "venue_indoor", time_of_day: "midday" });
    expect(slots[4]!.spec).toMatchObject({ category: "meal", setting: "home_indoor", time_of_day: "evening" });
    expect(slots.map((s) => s.required)).toEqual([true, true, false, true, false, false]);
  });

  it("records one transition per place change", () => {
    const { transitions } = resolveWorkday();
    expect(transitions).toEqual([
      { from_slot: "d1-s1", to_slot: "d1-s2", minutes: 20 },
      { from_slot: "d1-s4", to_slot: "d1-s5", minutes: 20 },
    ]);
  });

  it("asks the gap function for place kinds", () => {
    const calls: string[] = [];
    const gap = (from: PlaceKind, to: PlaceKind) => {
      calls.push(`${from}->${to}`);
      return 20;
    };
    resolveDayWindows(templates().day_types.workday, ANCHORS, workday(), world(), gap);
    expect(calls).toEqual(["home->workplace", "workplace->home"]);
  });

  it("dinner at eat_out adds gaps on both sides", () => {
    const { slots, transitions } = resolveWorkday(workday({ meal_place: { lunch: "workplace", dinner: "eat_out" } }));
    expect(table(slots).slice(4)).toEqual(["dinner 18:30-19:30 diner", "home_evening 19:50-22:18 home"]);
    expect(slots[4]!.spec.setting).toBe("venue_indoor");
    expect(transitions).toEqual([
      { from_slot: "d1-s1", to_slot: "d1-s2", minutes: 20 },
      { from_slot: "d1-s4", to_slot: "d1-s5", minutes: 20 },
      { from_slot: "d1-s5", to_slot: "d1-s6", minutes: 20 },
    ]);
  });

  it("work_end before lunch + 45 drops lunch, the nearest optional slot before work_pm (R24)", () => {
    const { slots, dropped } = resolveWorkday(workday(), { ...ANCHORS, work_end: 750 });
    expect(dropped).toEqual([{ role: "lunch", reason: "no_time" }]);
    expect(table(slots)).toEqual([
      "home_morning 06:30-07:14 home",
      "work_am 07:34-08:34 workplace",
      "work_pm 08:34-12:30 workplace",
      "dinner 18:30-19:30 home",
      "home_evening 19:30-22:18 home",
    ]);
  });

  it("work_end too early for work_am + work_pm is infeasible at work_pm once no optional slot is left", () => {
    let err: unknown;
    try {
      resolveWorkday(workday(), { ...ANCHORS, work_end: 500 });
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(TimelineError);
    expect((err as TimelineError).code).toBe("SCHEDULE_INFEASIBLE");
    expect((err as TimelineError).details).toMatchObject({ day: 1, role: "work_pm" });
  });

  it("a 90-min workday hobby that cannot end before sleep is the only slot dropped (R24)", () => {
    const decisions = workday({ optional: { hobby: true, home_evening: true }, hobby_node: "hobby_a" });
    const { slots, dropped } = resolveWorkday(decisions, { ...ANCHORS, sleep: 1260 });
    expect(dropped).toEqual([{ role: "hobby", reason: "no_time" }]);
    expect(table(slots).slice(3)).toEqual([
      "work_pm 12:45-17:00 workplace",
      "dinner 18:30-19:30 home",
      "home_evening 19:30-21:00 home",
    ]);
  });

  it("a hobby that fits before sleep is kept", () => {
    const decisions = workday({ optional: { hobby: true, home_evening: false }, hobby_node: "hobby_a" });
    const { slots, dropped } = resolveWorkday(decisions, { ...ANCHORS, sleep: 1300 });
    expect(dropped).toEqual([]);
    expect(slots.at(-1)!.window).toEqual(["19:50", "21:20"]);
  });

  it("nothing is dropped when the day fits", () => {
    expect(resolveWorkday().dropped).toEqual([]);
  });

  it("sleep after midnight ends the last window past 24:00", () => {
    const { slots } = resolveWorkday(workday(), { ...ANCHORS, sleep: 1470 });
    expect(slots.at(-1)!.window).toEqual(["19:30", "24:30"]);
  });

  it("drops skipped meals and absent optional slots, keeps a present hobby", () => {
    const { slots } = resolveWorkday(
      workday({ meal_place: { lunch: "skip", dinner: "home" }, optional: { hobby: true, home_evening: false }, hobby_node: "hobby_a" }),
    );
    expect(slots.map((s) => s.role)).toEqual(["home_morning", "work_am", "work_pm", "dinner", "hobby"]);
    expect(slots[4]!.spec).toMatchObject({
      category: "hobby",
      node: "hobby_a",
      setting: "venue_indoor",
      activity: "rock climbing",
      tags: ["sport_climbing"],
    });
    expect(slots[4]!.window).toEqual(["19:50", "21:20"]);
  });

  it("a required $hobby with no hobby_node is infeasible", () => {
    const decisions: DayDecisions = {
      day_index: 6,
      weekday: "Sat",
      day_type: "hobby_day",
      template: "hobby_day",
      meal_place: { breakfast: "home", lunch: "home", dinner: "home" },
      optional: {},
      exceptions: [],
    };
    const anchors: DayAnchors = { wake: 480, breakfast: 510, sport: 645, lunch: 780, dinner: 1100, sleep: 1380 };
    expect(() => resolveDayWindows(templates().day_types.hobby_day, anchors, decisions, world(), gap20)).toThrow(
      expect.objectContaining({ code: "SCHEDULE_INFEASIBLE", details: expect.objectContaining({ role: "hobby" }) }),
    );
  });
});
