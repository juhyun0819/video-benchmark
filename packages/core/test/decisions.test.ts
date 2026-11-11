import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  ExceptionCatalogSchema,
  WorldSchema,
  decideDay,
  decideDays,
  effectiveMealStyle,
  type DayDecisions,
  type ExceptionCatalog,
  type World,
} from "../src/index.js";

function catalog(): ExceptionCatalog {
  const url = new URL("../../../config/timeline/exceptions.v1.json", import.meta.url);
  return ExceptionCatalogSchema.parse(JSON.parse(readFileSync(url, "utf8")));
}

type Over = {
  seed?: number;
  routine?: number;
  slot?: number;
  day?: number;
  start?: string;
  hobbies?: Record<string, number>;
  placeNodes?: Record<string, string>;
  bindFrequent?: boolean;
};

function makeWorld(o: Over = {}): World {
  const hobbies = o.hobbies ?? { hobby_a: 1 };
  const nodes: Record<string, unknown> = {
    home: { kind: "home" },
    kitchen: { kind: "kitchen", default: "home" },
    workplace: { kind: "workplace", job: "clerk" },
    friend_home: { kind: "frequent_place", setting: "home_indoor", optional: true },
    diner: { kind: "frequent_place", setting: "venue_indoor", optional: true },
  };
  const binding: Record<string, unknown> = { home: { node: "h" }, workplace: { node: "w" } };
  for (const [name, freq] of Object.entries(hobbies)) {
    nodes[name] = { kind: "hobby", activity: name, setting: "outdoor", weekly_freq: freq };
    binding[name] = { node: name };
  }
  if (o.bindFrequent) {
    binding.friend_home = { node: "f" };
    binding.diner = { node: "d" };
  }
  return WorldSchema.parse({
    person_id: "p-test",
    attributes: { country: "US", occupation_group: "professional", work_pattern: "day_shift" },
    distribution_source: "atus_2025",
    nodes,
    node_binding: binding,
    meal_style: {
      workday: {
        lunch: { workplace: 0.7, home: 0.1, eat_out: 0.2 },
        dinner: { home: 0.8, eat_out: 0.1, others_home: 0.1 },
      },
      weekend: {
        breakfast: { home: 0.9, eat_out: 0.1 },
        lunch: { home: 0.6, eat_out: 0.3, others_home: 0.1 },
        dinner: { home: 0.7, eat_out: 0.2, others_home: 0.1 },
      },
      breakfast_skip_prob: 0.3,
      place_nodes: o.placeNodes ?? {},
    },
    dials: {
      length_days: 28,
      routine_strength: o.routine ?? 1,
      exception_rate: { slot: o.slot ?? 0, day: o.day ?? 0 },
      wake_sleep: "default_v1",
      start_weekday: o.start ?? "Mon",
    },
    seed: o.seed ?? 42,
  });
}

function bound(w: World): Set<string> {
  return new Set(Object.keys(w.node_binding));
}

function run(o: Over = {}): DayDecisions[] {
  const w = makeWorld(o);
  return decideDays(w, catalog(), { boundNodes: bound(w) });
}

const WEEKEND = [6, 7, 13, 14, 20, 21, 27, 28];

describe("calendar", () => {
  it("a 28-day world starting Mon has days 6, 7, 13, … as weekend", () => {
    const days = run();
    expect(days).toHaveLength(28);
    expect(days.map((d) => d.day_index)).toEqual(Array.from({ length: 28 }, (_, i) => i + 1));
    for (const d of days) {
      const weekend = WEEKEND.includes(d.day_index);
      expect(d.weekday === "Sat" || d.weekday === "Sun").toBe(weekend);
      expect(d.day_type === "workday").toBe(!weekend);
    }
    expect(days[5]!.weekday).toBe("Sat");
    expect(days[6]!.weekday).toBe("Sun");
  });

  it("day 1 is start_weekday", () => {
    const days = run({ start: "Sat" });
    expect(days.slice(0, 3).map((d) => d.weekday)).toEqual(["Sat", "Sun", "Mon"]);
  });
});

describe("hobby budget (routine_strength = 1)", () => {
  it("B=2 → every weekend is 2× hobby_day", () => {
    const days = run({ hobbies: { hobby_a: 1, hobby_b: 1 } });
    for (const d of days) {
      if (WEEKEND.includes(d.day_index)) {
        expect(d.day_type).toBe("hobby_day");
        expect(d.template).toBe("hobby_day");
        expect(["hobby_a", "hobby_b"]).toContain(d.hobby_node);
      } else {
        expect(d.optional.hobby).toBe(false);
      }
    }
  });

  it("B=1 → Saturday is hobby_day, Sunday is home_day", () => {
    const days = run({ hobbies: { hobby_a: 1 } });
    for (const d of days) {
      if (d.weekday === "Sat") expect(d.day_type).toBe("hobby_day");
      if (d.weekday === "Sun") {
        expect(d.day_type).toBe("home_day");
        expect(d.hobby_node).toBeUndefined();
      }
    }
  });

  it("B=3 → one workday per week has optional.hobby", () => {
    const days = run({ hobbies: { hobby_a: 2, hobby_b: 1 } });
    for (let week = 0; week < 4; week++) {
      const wd = days.slice(week * 7, week * 7 + 5);
      const withHobby = wd.filter((d) => d.optional.hobby);
      expect(withHobby).toHaveLength(1);
      expect(withHobby[0]!.weekday).toBe("Mon");
      expect(withHobby[0]!.hobby_node).toBeDefined();
      const weekend = days.slice(week * 7 + 5, week * 7 + 7);
      expect(weekend.map((d) => d.day_type)).toEqual(["hobby_day", "hobby_day"]);
    }
    // Smooth weighted round-robin over (hobby_a:2, hobby_b:1): Sat → hobby_a, Sun → hobby_b, Mon → hobby_a.
    expect(days[5]!.hobby_node).toBe("hobby_a");
    expect(days[6]!.hobby_node).toBe("hobby_b");
    expect(days[0]!.hobby_node).toBe("hobby_a");
  });
});

describe("hobby order (R18)", () => {
  it("A:3, B:1 → A three times and B once per week, in order A, A, B, A", () => {
    const days = run({ hobbies: { hobby_b: 1, hobby_a: 3 } });
    for (let week = 0; week < 4; week++) {
      const wk = days.slice(week * 7, week * 7 + 7);
      const byWeekday = Object.fromEntries(wk.map((d) => [d.weekday, d.hobby_node]));
      // Occurrence order is Sat, Sun, Mon, Tue.
      expect([byWeekday.Sat, byWeekday.Sun, byWeekday.Mon, byWeekday.Tue]).toEqual([
        "hobby_a",
        "hobby_a",
        "hobby_b",
        "hobby_a",
      ]);
      expect(wk.filter((d) => d.hobby_node !== undefined)).toHaveLength(4);
    }
  });
});

function comparable(d: DayDecisions): string {
  return JSON.stringify({
    day_type: d.day_type,
    meal_place: d.meal_place,
    optional: d.optional,
    hobby_node: d.hobby_node ?? null,
  });
}

describe("routine strength", () => {
  it("routine_strength=1 → every Tuesday has equal decisions", () => {
    for (const seed of [1, 2, 3]) {
      const tues = run({ seed, routine: 1, hobbies: { hobby_a: 3, hobby_b: 1 } }).filter(
        (d) => d.weekday === "Tue",
      );
      expect(tues).toHaveLength(4);
      for (const t of tues) expect(comparable(t)).toBe(comparable(tues[0]!));
    }
  });

  it("routine_strength=0 → over 200 seeds, Tuesday decisions differ in ≥ 40% of pairs", () => {
    let pairs = 0;
    let differ = 0;
    for (let seed = 0; seed < 200; seed++) {
      const tues = run({ seed, routine: 0, hobbies: { hobby_a: 3, hobby_b: 1 } })
        .filter((d) => d.weekday === "Tue")
        .map(comparable);
      for (let i = 0; i < tues.length; i++) {
        for (let j = i + 1; j < tues.length; j++) {
          pairs += 1;
          if (tues[i] !== tues[j]) differ += 1;
        }
      }
    }
    expect(differ / pairs).toBeGreaterThanOrEqual(0.4);
  });
});

describe("exceptions", () => {
  it("exception_rate.day=1 → every weekday is leave_day", () => {
    const days = run({ day: 1, slot: 1, bindFrequent: true, placeNodes: { eat_out: "diner" } });
    for (const d of days) {
      if (WEEKEND.includes(d.day_index)) {
        expect(d.day_type).not.toBe("leave_day");
        continue;
      }
      expect(d.day_type).toBe("leave_day");
      expect(d.template).toBe("home_day");
      expect(d.exceptions).toEqual([
        { scope: "day", type: "leave_day", replaces: "workday", day_range: [d.day_index, d.day_index] },
      ]);
    }
  });

  it("exception_rate.slot=1 with no eat_out node → only lunch_at_friend, on weekend days", () => {
    const days = run({ slot: 1, bindFrequent: true, placeNodes: { others_home: "friend_home" } });
    for (const d of days) {
      if (d.template === "workday") {
        expect(d.exceptions).toEqual([]);
      } else {
        expect(d.exceptions).toEqual([
          { scope: "slot", type: "lunch_at_friend", replaces: "lunch", day_range: [d.day_index, d.day_index] },
        ]);
      }
    }
  });

  it("exception_rate.slot=1 with others_home unbound → no exceptions", () => {
    for (const placeNodes of [{}, { others_home: "friend_home" }] as Record<string, string>[]) {
      const days = run({ slot: 1, bindFrequent: false, placeNodes });
      for (const d of days) expect(d.exceptions).toEqual([]);
    }
  });

  it("a slot exception writes its place into meal_place of the replaced role (R19)", () => {
    for (const seed of [1, 2, 3]) {
      const w = makeWorld({ seed, routine: 0, slot: 1, bindFrequent: true, placeNodes: { eat_out: "diner" } });
      const days = decideDays(w, catalog(), { boundNodes: bound(w) });
      for (const d of days) {
        expect(d.exceptions).toHaveLength(1);
        const e = d.exceptions[0]!;
        expect(["dinner_out", "lunch_out"]).toContain(e.type);
        expect(d.meal_place[e.replaces as "lunch" | "dinner"]).toBe("eat_out");
      }
    }
  });

  it("a catalog with duplicate slot types fails to parse", () => {
    const raw = JSON.parse(JSON.stringify(catalog()));
    raw.slot.push({ ...raw.slot[0] });
    expect(ExceptionCatalogSchema.safeParse(raw).success).toBe(false);
  });

  it("no day has both a day and a slot exception", () => {
    let dayExc = 0;
    let slotExc = 0;
    for (let seed = 0; seed < 20; seed++) {
      const days = run({
        seed,
        day: 0.5,
        slot: 1,
        bindFrequent: true,
        placeNodes: { eat_out: "diner", others_home: "friend_home" },
      });
      for (const d of days) {
        expect(d.exceptions.length).toBeLessThanOrEqual(1);
        if (d.day_type === "leave_day") expect(d.exceptions[0]!.scope).toBe("day");
        else expect(d.exceptions[0]!.scope).toBe("slot");
        dayExc += d.exceptions.filter((e) => e.scope === "day").length;
        slotExc += d.exceptions.filter((e) => e.scope === "slot").length;
      }
    }
    expect(dayExc).toBeGreaterThan(0);
    expect(slotExc).toBeGreaterThan(0);
  });
});

describe("meal places", () => {
  it("unbound eat_out mass is dropped and renormalized; zero mass resolves to skip", () => {
    const w = makeWorld({ placeNodes: { eat_out: "diner" } });
    const eff = effectiveMealStyle(w, bound(w));
    expect(Object.keys(eff.workday.lunch)).toEqual(["home", "workplace"]);
    expect(eff.workday.lunch.home).toBeCloseTo(0.125, 12);
    expect(eff.workday.lunch.workplace).toBeCloseTo(0.875, 12);
    expect(eff.weekend.breakfast).toEqual({ home: 1 });

    const onlyOut = makeWorld();
    onlyOut.meal_style.workday.dinner = { eat_out: 1 };
    expect(effectiveMealStyle(onlyOut, bound(onlyOut)).workday.dinner).toEqual({});
    const days = decideDays(onlyOut, catalog(), { boundNodes: bound(onlyOut) });
    for (const d of days.filter((x) => x.template === "workday")) expect(d.meal_place.dinner).toBe("skip");
  });

  it("workdays decide lunch and dinner; weekend templates also decide breakfast", () => {
    for (const d of run({ seed: 5, routine: 0 })) {
      const meals = Object.keys(d.meal_place);
      if (d.template === "workday") {
        expect(meals).toEqual(["lunch", "dinner"]);
        expect(Object.keys(d.optional).sort()).toEqual(["hobby", "home_evening"]);
      } else {
        expect(meals).toEqual(["breakfast", "lunch", "dinner"]);
        expect(d.optional).toEqual({});
      }
      const allowed = d.template === "workday" ? ["home", "workplace", "skip"] : ["home", "skip"];
      for (const p of Object.values(d.meal_place)) expect(allowed).toContain(p);
    }
  });

  it("non-workday templates never eat at the workplace (R16)", () => {
    const w = makeWorld({ routine: 0, day: 0.5, seed: 11 });
    w.meal_style.weekend.breakfast = { home: 0.5, workplace: 0.5 };
    w.meal_style.weekend.lunch = { home: 0.4, workplace: 0.6 };
    w.meal_style.weekend.dinner = { workplace: 1 };
    const eff = effectiveMealStyle(w, bound(w));
    for (const dist of Object.values(eff.weekend)) expect(dist).not.toHaveProperty("workplace");
    expect(eff.weekend.breakfast).toEqual({ home: 1 });
    expect(eff.weekend.lunch).toEqual({ home: 1 });
    expect(eff.weekend.dinner).toEqual({});
    expect(eff.workday.lunch).toHaveProperty("workplace");

    const days = decideDays(w, catalog(), { boundNodes: bound(w) });
    expect(days.some((d) => d.day_type === "leave_day")).toBe(true);
    for (const d of days.filter((x) => x.template !== "workday")) {
      expect(Object.values(d.meal_place)).not.toContain("workplace");
    }
    expect(days.some((d) => d.template === "workday" && d.meal_place.lunch === "workplace")).toBe(true);
  });
});

describe("determinism", () => {
  it("same input → identical output; attempt only changes that day", () => {
    const o = { seed: 9, routine: 0.3, slot: 0.3, day: 0.2, bindFrequent: true, placeNodes: { eat_out: "diner" } };
    expect(JSON.stringify(run(o))).toBe(JSON.stringify(run(o)));
    const w = makeWorld(o);
    const opts = { boundNodes: bound(w) };
    expect(decideDay(w, catalog(), opts, 3, 0)).toEqual(run(o)[2]);
    const differs = [1, 2, 3, 4, 5].some(
      (a) => JSON.stringify(decideDay(w, catalog(), opts, 3, a)) !== JSON.stringify(run(o)[2]),
    );
    expect(differs).toBe(true);
  });
});
