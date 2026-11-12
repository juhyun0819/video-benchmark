import { describe, expect, it } from "vitest";
import { WorldSchema, parseWorld } from "../src/world.js";

function world(): Record<string, any> {
  return {
    person_id: "p01",
    attributes: {
      country: "US",
      occupation_group: "construction_extraction",
      sex: "male",
      age_band: "25-34",
      work_pattern: "day_shift",
    },
    distribution_source: "atus_2025",
    nodes: {
      home: { kind: "home" },
      kitchen: { kind: "kitchen", default: "home" },
      workplace: { kind: "workplace", job: "carpenter" },
      hobby_1: { kind: "hobby", activity: "running", setting: "outdoor", weekly_freq: 2 },
      frequent_1: { kind: "frequent_place", setting: "venue_indoor", optional: true },
    },
    node_binding: {
      home: { dataset: ["youtube"], person_source: "creator-1" },
      workplace: { tags_any: ["carpentry"] },
      hobby_1: { node: "running" },
    },
    meal_style: {
      workday: {
        lunch: { workplace: 0.84, home: 0.1, eat_out: 0.06 },
        dinner: { home: 0.83, eat_out: 0.09, workplace: 0.07, others_home: 0.01 },
      },
      rest_day: {
        breakfast: { home: 0.91, others_home: 0.04, eat_out: 0.05 },
        lunch: { home: 0.67, eat_out: 0.27, others_home: 0.06 },
        dinner: { home: 0.79, eat_out: 0.14, others_home: 0.07 },
      },
      breakfast_skip_prob: 0.3,
      place_nodes: { eat_out: "frequent_1" },
    },
    dials: {
      length_days: 28,
      routine_strength: 0.7,
      exception_rate: { slot: 0.1, day: 0.05 },
      wake_sleep: "default_v1",
    },
    seed: 1234,
  };
}

function worldInvalid(w: unknown): { message: string; issues: { path: string }[] } {
  try {
    parseWorld(w);
  } catch (e) {
    expect(e).toMatchObject({ name: "TimelineError", code: "WORLD_INVALID" });
    const err = e as { message: string; details: { issues: { path: string }[] } };
    return { message: err.message, issues: err.details.issues };
  }
  throw new Error("expected WORLD_INVALID");
}

function issuePaths(w: unknown): string[] {
  const r = WorldSchema.safeParse(w);
  return r.success ? [] : r.error.issues.map((i) => i.path.join("."));
}

describe("WorldSchema", () => {
  it("parses the spec example and applies defaults", () => {
    const w = WorldSchema.parse(world());
    expect(w.allow_country_fallback).toBe(false);
    expect(w.dials.start_weekday).toBe("Mon");
  });

  it("rejects zero and four hobbies", () => {
    const none = world();
    delete none.nodes.hobby_1;
    delete none.node_binding.hobby_1;
    expect(WorldSchema.safeParse(none).success).toBe(false);

    const four = world();
    for (const k of ["hobby_2", "hobby_3", "hobby_4"]) {
      four.nodes[k] = { kind: "hobby", activity: "x", setting: "outdoor", weekly_freq: 1 };
      four.node_binding[k] = { node: "x" };
    }
    expect(WorldSchema.safeParse(four).success).toBe(false);
  });

  it("rejects length_days below 28", () => {
    const w = world();
    w.dials.length_days = 27;
    expect(WorldSchema.safeParse(w).success).toBe(false);
  });

  it("rejects routine_strength outside [0,1]", () => {
    const w = world();
    w.dials.routine_strength = 1.1;
    expect(WorldSchema.safeParse(w).success).toBe(false);
  });

  it("reports a missing workplace binding at node_binding.workplace", () => {
    const w = world();
    delete w.node_binding.workplace;
    expect(issuePaths(w)).toContain("node_binding.workplace");
  });

  it("allows an optional frequent_place without a binding", () => {
    const w = world();
    expect(w.node_binding.frequent_1).toBeUndefined();
    expect(WorldSchema.safeParse(w).success).toBe(true);
  });

  it("requires place_nodes to name a frequent_place node", () => {
    const w = world();
    w.meal_style.place_nodes = { eat_out: "workplace" };
    expect(WorldSchema.safeParse(w).success).toBe(false);
    w.meal_style.place_nodes = { eat_out: "missing" };
    expect(WorldSchema.safeParse(w).success).toBe(false);
  });

  it("rejects negative meal weights and out-of-range skip probability", () => {
    const w = world();
    w.meal_style.workday.lunch.home = -1;
    expect(WorldSchema.safeParse(w).success).toBe(false);
    const v = world();
    v.meal_style.breakfast_skip_prob = 2;
    expect(WorldSchema.safeParse(v).success).toBe(false);
  });

  it("rejects meal weights summing to 0 or 0.98 and accepts 0.995 (E8)", () => {
    const zero = world();
    zero.meal_style.workday.lunch = {};
    expect(worldInvalid(zero).message).toContain("meal_style.workday.lunch");

    const low = world();
    low.meal_style.rest_day.dinner = { home: 0.77, eat_out: 0.14, others_home: 0.07 };
    expect(worldInvalid(low).issues.map((i) => i.path)).toContain("meal_style.rest_day.dinner");

    const ok = world();
    ok.meal_style.rest_day.dinner = { home: 0.795, eat_out: 0.14, others_home: 0.06 };
    expect(parseWorld(ok).person_id).toBe("p01");
  });

  it("rejects workplace on rest days (E8)", () => {
    const w = world();
    w.meal_style.rest_day.lunch = { home: 0.57, eat_out: 0.27, others_home: 0.06, workplace: 0.1 };
    expect(worldInvalid(w).issues.map((i) => i.path)).toContain("meal_style.rest_day.lunch.workplace");
  });

  it("rejects four hobbies as WORLD_INVALID and joins issues as path: message", () => {
    const w = world();
    for (const k of ["hobby_2", "hobby_3", "hobby_4"]) {
      w.nodes[k] = { kind: "hobby", activity: "x", setting: "outdoor", weekly_freq: 1 };
      w.node_binding[k] = { node: "x" };
    }
    expect(worldInvalid(w).message).toMatch(/^nodes: expected 1-3 hobby nodes, got 4/);
  });

  it("no longer accepts meal_style.weekend", () => {
    const w = world();
    w.meal_style.weekend = w.meal_style.rest_day;
    delete w.meal_style.rest_day;
    expect(worldInvalid(w).issues.map((i) => i.path)).toContain("meal_style.rest_day");
  });

  it("hints at the rest_day rename when meal_style.weekend is present (G25)", () => {
    const w = world();
    w.meal_style.weekend = w.meal_style.rest_day;
    delete w.meal_style.rest_day;
    expect(worldInvalid(w).message).toMatch(/meal_style\.weekend .*renamed.*meal_style\.rest_day/);
    const other = world();
    other.meal_style.rest_day.lunch = { home: 0.9 };
    expect(worldInvalid(other).message).not.toContain("renamed");
  });
});
