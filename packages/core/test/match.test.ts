import { describe, expect, it } from "vitest";
import {
  TimelineError,
  candidatesFor,
  createTimeline,
  fillTimeline,
  hardFilterFailures,
  preflightCapacity,
  rankSelector,
  type ClipCatalog,
  type ScheduleDay,
  type Selector,
  type Timeline,
} from "../src/index.js";
import {
  HOBBY_DAY,
  WORKDAY,
  clip,
  genId,
  makeClips,
  makeDay,
  makeSchedule,
  makeWorld,
  memoryCatalog,
  type SlotDef,
} from "./fixtures.js";

const neverRegen = (d: number, a: number): ScheduleDay => {
  throw new Error(`unexpected regen(${d}, ${a})`);
};

function errorOf(fn: () => unknown): TimelineError {
  try {
    fn();
  } catch (e) {
    if (e instanceof TimelineError) return e;
    throw e;
  }
  throw new Error("expected a TimelineError");
}

/** Every assignment, in schedule order. */
function assignments(t: Timeline, catalog: ClipCatalog) {
  return t.schedule.days.flatMap((day) =>
    day.slots.flatMap((slot) =>
      t.slots[slot.slot_id]!.clip_ids.map((id) => ({ day: day.day_index, slot, clip: catalog.getClip(id)! })),
    ),
  );
}

describe("fillTimeline", () => {
  const catalog = memoryCatalog(makeClips());
  const filled = () => fillTimeline(createTimeline(makeWorld(), makeSchedule(28), genId), catalog, rankSelector, neverRegen);

  it("RankSelector fills a 28-day fixture with every required slot filled", () => {
    const t = filled();
    expect(t.revision).toBe(1);
    for (const day of t.schedule.days) {
      for (const slot of day.slots) {
        const entry = t.slots[slot.slot_id]!;
        if (slot.required) expect(entry.clip_ids.length, slot.slot_id).toBe(1);
        expect(entry.relaxations).toEqual([]);
      }
    }
    const all = assignments(t, catalog);
    expect(new Set(all.map((a) => a.clip.clip_id)).size).toBe(all.length);
    const daysOfGroup = new Map<string, Set<number>>();
    for (const a of all) daysOfGroup.set(a.clip.day_group, (daysOfGroup.get(a.clip.day_group) ?? new Set()).add(a.day));
    for (const [group, days] of daysOfGroup) expect(days.size, group).toBe(1);
  });

  it("same-node slots of one day share a day_group when possible", () => {
    const t = filled();
    const groups = new Map<string, Set<string>>();
    for (const a of assignments(t, catalog)) {
      const key = `${a.day}/${a.slot.spec.node}`;
      groups.set(key, (groups.get(key) ?? new Set()).add(a.clip.day_group));
    }
    expect(groups.size).toBeGreaterThan(28);
    for (const [key, set] of groups) expect([...set], key).toHaveLength(1);
  });

  it("a day prefers a day_group already used at another node, so 30 groups fill 28 days with non-aligned ids", () => {
    // Work clip ids sort in reverse group order, so home and work ranking ties break toward different groups.
    const clips = makeClips().map((c) =>
      c.node === "work" ? { ...c, clip_id: `w${String(100 - Number(c.clip_id.slice(1, 3))).padStart(2, "0")}${c.clip_id.slice(3)}` } : c,
    );
    const cat = memoryCatalog(clips);
    const t = fillTimeline(createTimeline(makeWorld(), makeSchedule(28), genId), cat, rankSelector, neverRegen);
    const groupsOfDay = new Map<number, Set<string>>();
    for (const a of assignments(t, cat)) {
      if (a.clip.source.dataset !== "nextme800") continue;
      groupsOfDay.set(a.day, (groupsOfDay.get(a.day) ?? new Set()).add(a.clip.day_group));
    }
    expect(groupsOfDay.size).toBe(28);
    for (const [day, set] of groupsOfDay) expect([...set], `day ${day}`).toHaveLength(1);
  });

  it("tags filter drops at level 0 and passes with ignore_tags (recorded)", () => {
    const chores: SlotDef = { role: "chores", category: "home", setting: "home_indoor", node: "home", activity: "doing chores", tags: ["chores"], required: true, window: ["07:00", "08:00"] };
    const t0 = createTimeline(makeWorld(), makeSchedule(1, () => [chores]), genId);
    expect(candidatesFor(t0, catalog, "d1-s1", [])).toEqual([]);
    expect(candidatesFor(t0, catalog, "d1-s1", ["ignore_tags"]).length).toBeGreaterThan(0);
    const t = fillTimeline(t0, catalog, rankSelector, neverRegen);
    expect(t.slots["d1-s1"]!.relaxations).toEqual(["ignore_tags"]);
    expect(t.slots["d1-s1"]!.clip_ids).toHaveLength(1);
  });

  it("evening-only clip fits an afternoon window only with widen_window_30m", () => {
    const evening = clip({ clip_id: "eve", activity: "relaxing at home", time_of_day: "evening" });
    const cat = memoryCatalog([evening]);
    const late: SlotDef = { role: "home_afternoon", category: "home", setting: "home_indoor", node: "home", activity: "relaxing at home", required: true, window: ["16:30", "17:00"] };
    const t0 = createTimeline(makeWorld(), makeSchedule(1, () => [late]), genId);
    expect(hardFilterFailures(t0, cat, "d1-s1", evening, ["ignore_tags"])).toEqual(["time_of_day"]);
    expect(hardFilterFailures(t0, cat, "d1-s1", evening, ["ignore_tags", "widen_window_30m"])).toEqual([]);
    const t = fillTimeline(t0, cat, rankSelector, neverRegen);
    expect(t.slots["d1-s1"]).toEqual({ clip_ids: ["eve"], relaxations: ["ignore_tags", "widen_window_30m"] });
  });

  it("optional slot with no clips is dropped_optional", () => {
    const withDiner = (d: number): SlotDef[] | undefined =>
      d === 1
        ? [WORKDAY[0]!, { role: "dinner", category: "meal", setting: "venue_indoor", node: "diner", activity: "eating dinner", required: false, window: ["18:30", "19:30"] }]
        : undefined;
    const t = fillTimeline(createTimeline(makeWorld(), makeSchedule(2, withDiner), genId), catalog, rankSelector, neverRegen);
    expect(t.slots["d1-s2"]).toEqual({ clip_ids: [], relaxations: ["dropped_optional"] });
    expect(t.slots["d1-s1"]!.clip_ids).toHaveLength(1);
  });

  it("required slot with no clips triggers regen attempts day.attempt+1..+5, then UNFILLABLE lists the slot", () => {
    const noHobbyClips = memoryCatalog(makeClips({ hobbyVideos: 0 }));
    // The generator already built day 2 at attempt 2 (R34: retries continue from there).
    const schedule = makeSchedule(3);
    schedule.days[1] = makeDay(2, 2, HOBBY_DAY);
    const t0 = createTimeline(makeWorld(), schedule, genId);
    const calls: [number, number][] = [];
    const regen = (d: number, a: number): ScheduleDay => {
      calls.push([d, a]);
      // R6: an infeasible attempt counts as a failed attempt; the loop goes on.
      if (a === 5) throw new TimelineError("SCHEDULE_INFEASIBLE", "synthetic infeasible day");
      return makeDay(d, a, HOBBY_DAY);
    };
    const err = errorOf(() => fillTimeline(t0, noHobbyClips, rankSelector, regen));
    expect(err.code).toBe("UNFILLABLE");
    expect(err.details).toMatchObject({ slots: ["d2-s2"] });
    expect(calls).toEqual([[2, 3], [2, 4], [2, 5], [2, 6], [2, 7]]);
    // No partial timeline: the input is untouched.
    expect(t0.revision).toBe(0);
    expect(Object.values(t0.slots).every((s) => s.clip_ids.length === 0)).toBe(true);
  });

  it("regen of day 3 keeps days 1–2 identical and frees only day-3 day_groups", () => {
    // Day 3's first version has a required 5-minute slot no clip fits.
    const broken: SlotDef[] = [
      WORKDAY[0]!,
      { role: "nap", category: "home", setting: "home_indoor", node: "home", activity: "napping", required: true, window: ["08:30", "08:35"] },
      ...WORKDAY.slice(1),
    ];
    const calls: [number, number][] = [];
    const regen = (d: number, a: number) => {
      calls.push([d, a]);
      return makeDay(d, a);
    };
    // Day 3 was built at attempt 1, so the first regeneration is attempt 2 (R34).
    const schedule = makeSchedule(5);
    schedule.days[2] = makeDay(3, 1, broken);
    const a = fillTimeline(createTimeline(makeWorld(), schedule, genId), catalog, rankSelector, regen);
    expect(calls).toEqual([[3, 2]]);

    // Same schedule with day 3 already at attempt 2: identical result means days 1–2 kept their
    // assignments and the day groups day 3's failed first try had taken were released.
    const b0 = createTimeline(makeWorld(), makeSchedule(5), genId);
    b0.schedule.days[2] = makeDay(3, 2);
    const b = fillTimeline(b0, catalog, rankSelector, neverRegen);
    expect(a.schedule).toEqual(b.schedule);
    expect(a.slots).toEqual(b.slots);
    expect(Object.keys(a.slots)).toEqual(Object.keys(b.slots));
    expect(a.revision).toBe(1);

    const day3 = assignments(a, catalog).filter((x) => x.day === 3);
    expect(new Set(day3.map((x) => x.clip.day_group))).toEqual(new Set(["nextme800:g03"]));
    const day12 = assignments(a, catalog).filter((x) => x.day < 3);
    expect(new Set(day12.map((x) => x.clip.day_group))).toEqual(new Set(["nextme800:g01", "nextme800:g02"]));
  });

  it("every clip longer than every window → UNFILLABLE", () => {
    const huge = memoryCatalog(makeClips({ durationS: 25 * 3600 }));
    const t0 = createTimeline(makeWorld(), makeSchedule(28), genId);
    const err = errorOf(() => fillTimeline(t0, huge, rankSelector, (d, a) => makeDay(d, a)));
    expect(err.code).toBe("UNFILLABLE");
    expect(err.details).toMatchObject({ slots: ["d1-s1", "d1-s2", "d1-s4"] });
  });

  it("skips slots that already have clips and hands the selector the top 10 and the last 3 days", () => {
    const seen: { slot: string; n: number; recentDays: number[] }[] = [];
    const spy: Selector = {
      choose(slot, candidates, recent) {
        seen.push({ slot: slot.slot_id, n: candidates.length, recentDays: [...new Set(recent.map((r) => r.day_index))] });
        return rankSelector.choose(slot, candidates, recent);
      },
    };
    const t0 = createTimeline(makeWorld(), makeSchedule(5), genId);
    t0.slots["d1-s1"] = { clip_ids: ["g05-a-morning"], relaxations: [] };
    const t = fillTimeline(t0, catalog, spy, neverRegen);
    expect(t.slots["d1-s1"]!.clip_ids).toEqual(["g05-a-morning"]);
    expect(seen.some((s) => s.slot === "d1-s1")).toBe(false);
    expect(seen.every((s) => s.n <= 10)).toBe(true);
    expect(seen.find((s) => s.slot === "d5-s1")!.recentDays).toEqual([2, 3, 4]);
  });

  it("rejects a selector choice outside the candidates", () => {
    const rogue: Selector = { choose: () => "h01-climb" };
    const t0 = createTimeline(makeWorld(), makeSchedule(1), genId);
    expect(errorOf(() => fillTimeline(t0, catalog, rogue, neverRegen)).code).toBe("CLIP_INELIGIBLE");
  });
});

describe("candidatesFor", () => {
  it("score adds the same-node day_group bonus and routine similarity to yesterday's same-role clip", () => {
    const relax: SlotDef = { role: "home_evening", category: "home", setting: "home_indoor", node: "home", activity: "relaxing", required: true, window: ["20:00", "22:00"] };
    const read = (id: string, group: string) => clip({ clip_id: id, activity: "reading a book", time_of_day: "evening", day_group: group });
    const tv = clip({ clip_id: "tv", activity: "relaxing with tv", time_of_day: "evening", day_group: "g-b" });
    const cat = memoryCatalog([read("read-yesterday", "g-y"), read("read-today", "g-a"), tv, clip({ clip_id: "morning-a", day_group: "g-a" })]);
    const t0 = createTimeline(makeWorld({ routine: 1 }), makeSchedule(2, () => [WORKDAY[0]!, relax]), genId);
    t0.slots["d1-s2"] = { clip_ids: ["read-yesterday"], relaxations: [] };
    t0.slots["d2-s1"] = { clip_ids: ["morning-a"], relaxations: [] };
    const ranked = candidatesFor(t0, cat, "d2-s2", []);
    expect(ranked.map((r) => [r.clip.clip_id, r.score])).toEqual([
      // jaccard({relaxing},{reading,a,book}) = 0, +0.5 day_group g-a used today by home, +0.25 used today at any node,
      // +1 × 1.0 routine
      ["read-today", 1.75],
      // jaccard({relaxing},{relaxing,with,tv}) = 1/3, routine jaccard 0
      ["tv", 1 / 3],
    ]);
    expect(ranked[0]!.reasons.length).toBe(3);
  });

  it("score adds +0.25 when the day_group is used today at another node only", () => {
    const work = clip({ clip_id: "work-a", category: "work", setting: "venue_indoor", node: "work", activity: "work at desk", day_group: "g-a" });
    const cat = memoryCatalog([
      clip({ clip_id: "morning-a", day_group: "g-a" }),
      clip({ clip_id: "morning-b", day_group: "g-b" }),
      work,
      { ...work, clip_id: "work-b", day_group: "g-b" },
    ]);
    const t0 = createTimeline(makeWorld(), makeSchedule(1, () => [WORKDAY[0]!, WORKDAY[1]!]), genId);
    t0.slots["d1-s1"] = { clip_ids: ["morning-b"], relaxations: [] };
    const ranked = candidatesFor(t0, cat, "d1-s2", []);
    // jaccard 1 for both; g-b is used today at home, not at the workplace
    expect(ranked.map((r) => [r.clip.clip_id, r.score])).toEqual([
      ["work-b", 1.25],
      ["work-a", 1],
    ]);
    expect(ranked[0]!.reasons).toEqual(["activity/tags overlap 1.00", "day_group g-b already used today"]);
  });
});

describe("preflightCapacity", () => {
  it('preflight reports BANK_CAPACITY {node:"home", needed_days:28, distinct_day_groups:10}', () => {
    const t = createTimeline(makeWorld(), makeSchedule(28), genId);
    const err = errorOf(() => preflightCapacity(t, memoryCatalog(makeClips({ homeGroups: 10 }))));
    expect(err.code).toBe("BANK_CAPACITY");
    expect(err.details).toEqual({ shortfalls: [{ node: "home", needed_days: 28, distinct_day_groups: 10 }] });
  });

  it("passes when every node has a distinct day_group per day that needs it", () => {
    const t = createTimeline(makeWorld(), makeSchedule(28), genId);
    expect(() => preflightCapacity(t, memoryCatalog(makeClips()))).not.toThrow();
  });
});
