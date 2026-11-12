import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  ExceptionCatalogSchema,
  GapTableSchema,
  OCCUPATION_CODES,
  RULE_IDS,
  ScheduleDefaultsSchema,
  TemplatesSchema,
  WorldSchema,
  atus2025Source,
  generateSchedule,
  verifyTimeline,
  createTimeline,
  type Clip,
  type DonorRow,
  type GapTable,
  type RuleId,
  type ScheduleDay,
  type Timeline,
} from "../src/index.js";
import { clip, genId, makeSchedule, makeWorld, memoryCatalog, type SlotDef } from "./fixtures.js";

const DEFS: SlotDef[] = [
  { role: "home_morning", category: "home", setting: "home_indoor", node: "home", activity: "getting ready", required: true, window: ["07:00", "08:00"] },
  { role: "work_am", category: "work", setting: "venue_indoor", node: "workplace", activity: "work", required: true, window: ["09:00", "12:00"] },
  { role: "lunch", category: "meal", setting: "venue_indoor", node: "workplace", activity: "eating lunch", required: false, window: ["12:00", "12:45"] },
  { role: "work_pm", category: "work", setting: "venue_indoor", node: "workplace", activity: "work", required: true, window: ["13:00", "17:00"] },
];

const HOUR = 3600;
function mk(id: string, over: Partial<Clip>): Clip {
  return clip({ clip_id: id, time_of_day: "unknown", source_end_s: HOUR, ...over });
}
const homeC = (id: string, group: string) => mk(id, { day_group: group });
const workC = (id: string, group: string, over: Partial<Clip> = {}) =>
  mk(id, { category: "work", setting: "venue_indoor", node: "work", day_group: group, ...over });
const lunchC = (id: string, group: string) =>
  mk(id, { category: "meal", setting: "venue_indoor", node: "work", day_group: group, source_end_s: 1200 });

const BASE_CLIPS: Clip[] = [
  homeC("d1-home", "g1"),
  workC("d1-am", "g1"),
  lunchC("d1-lunch", "g1"),
  workC("d1-pm", "g1"),
  homeC("d2-home", "g2"),
  workC("d2-am", "g2"),
  lunchC("d2-lunch", "g2"),
  workC("d2-pm", "g2"),
];

type Built = { t: Timeline; clips: Clip[] };

/** A valid two-day (both workdays) timeline with one clip per slot. */
function base(): Built {
  const t0 = createTimeline(makeWorld(), makeSchedule(2, () => DEFS), genId);
  const slots = { ...t0.slots };
  for (const d of [1, 2]) {
    ["home", "am", "lunch", "pm"].forEach((k, i) => {
      slots[`d${d}-s${i + 1}`] = { clip_ids: [`d${d}-${k}`], relaxations: [] };
    });
  }
  return { t: { ...t0, slots }, clips: BASE_CLIPS };
}

const setSlot = (t: Timeline, id: string, clip_ids: string[], relaxations: Timeline["slots"][string]["relaxations"] = []): Timeline => ({
  ...t,
  slots: { ...t.slots, [id]: { clip_ids, relaxations } },
});
const mapDay = (t: Timeline, d: number, f: (day: ScheduleDay) => ScheduleDay): Timeline => ({
  ...t,
  schedule: { ...t.schedule, days: t.schedule.days.map((day) => (day.day_index === d ? f(day) : day)) },
});
const mapSlot = (t: Timeline, id: string, f: (s: ScheduleDay["slots"][number]) => ScheduleDay["slots"][number]): Timeline => ({
  ...t,
  schedule: {
    ...t.schedule,
    days: t.schedule.days.map((day) => ({ ...day, slots: day.slots.map((s) => (s.slot_id === id ? f(s) : s)) })),
  },
});

function onlyFails(b: Built, rule: RuleId) {
  const r = verifyTimeline(b.t, memoryCatalog(b.clips));
  expect(r.rules[rule].pass).toBe(false);
  expect(r.rules[rule].violations.length).toBeGreaterThan(0);
  for (const other of RULE_IDS) if (other !== rule) expect(r.rules[other].pass, `${other}: ${JSON.stringify(r.rules[other].violations)}`).toBe(true);
  expect(r.pass).toBe(false);
}

const slotExc = { scope: "slot" as const, type: "lunch_out", replaces: "lunch", day_range: [1, 1] as [number, number], meal_place: "eat_out" as const };

/** Day 1 lunch moved to the eat_out node ("diner"). */
function withExceptionLunch(): Built {
  const { t, clips } = base();
  let u = mapSlot(t, "d1-s3", (s) => ({ ...s, spec: { ...s.spec, node: "diner", setting: "venue_indoor" } }));
  u = mapDay(u, 1, (day) => ({ ...day, exceptions: [slotExc] }));
  const diner = mk("d1-diner", { category: "meal", setting: "venue_indoor", node: "diner", day_group: "g1", source_end_s: 1200 });
  return { t: setSlot(u, "d1-s3", ["d1-diner"]), clips: [...clips, diner] };
}

describe("verifyTimeline", () => {
  it("valid timeline passes every rule", () => {
    const { t, clips } = base();
    const r = verifyTimeline(t, memoryCatalog(clips));
    expect(r.pass).toBe(true);
    for (const id of RULE_IDS) expect(r.rules[id], id).toEqual({ pass: true, violations: [] });
    expect(r.missing_clips).toEqual([]);
  });

  it("valid timeline with a slot exception passes", () => {
    const b = withExceptionLunch();
    expect(verifyTimeline(b.t, memoryCatalog(b.clips)).pass).toBe(true);
  });

  it("day_type violation is caught", () => {
    const { t, clips } = base();
    onlyFails({ t: mapDay(t, 1, (d) => ({ ...d, day_type: "hobby_day" })), clips }, "day_type");
  });

  it("slot_window violation is caught", () => {
    const { t, clips } = base();
    onlyFails({ t: setSlot(t, "d1-s2", []), clips }, "slot_window");
  });

  it("slot_window catches a clip longer than its window", () => {
    const { t, clips } = base();
    // d1-pm window is 13:00-17:00 (4 h); with a 5 h second clip the slot overflows.
    const long = workC("d1-long", "g1", { source_end_s: 5 * HOUR });
    onlyFails({ t: setSlot(t, "d1-s4", ["d1-pm", "d1-long"]), clips: [...clips, long] }, "slot_window");
  });

  it("wake_sleep violation is caught", () => {
    const { t, clips } = base();
    onlyFails({ t: mapDay(t, 1, (d) => ({ ...d, sleep: "16:00" })), clips }, "wake_sleep");
  });

  it("no_overlap violation is caught", () => {
    const { t, clips } = base();
    let u = setSlot(t, "d1-s3", []); // lunch is optional; drop it so only work_am/work_pm are compared
    u = mapSlot(u, "d1-s2", (s) => ({ ...s, window: ["09:00", "10:00"] }));
    u = mapSlot(u, "d1-s4", (s) => ({ ...s, window: ["09:30", "10:30"] }));
    onlyFails({ t: u, clips }, "no_overlap");
  });

  it("monotonic violation is caught", () => {
    const { t, clips } = base();
    let u = setSlot(t, "d1-s3", []);
    u = mapSlot(u, "d1-s2", (s) => ({ ...s, window: ["11:00", "12:00"] }));
    u = mapSlot(u, "d1-s4", (s) => ({ ...s, window: ["09:00", "10:00"] }));
    onlyFails({ t: u, clips }, "monotonic");
  });

  it("node_binding violation is caught", () => {
    const { t, clips } = base();
    // A home clip placed in the workplace slot.
    onlyFails({ t: setSlot(t, "d1-s2", ["d1-wrong"]), clips: [...clips, homeC("d1-wrong", "g1")] }, "node_binding");
  });

  it("clip_unique violation is caught", () => {
    const { t, clips } = base();
    onlyFails({ t: setSlot(t, "d1-s4", ["d1-am"]), clips }, "clip_unique");
  });

  it("day_group_single_day violation is caught", () => {
    const { t, clips } = base();
    const stray = workC("d2-stray", "g1");
    onlyFails({ t: setSlot(t, "d2-s2", ["d2-stray"]), clips: [...clips, stray] }, "day_group_single_day");
  });

  it("time_of_day violation is caught", () => {
    const { t, clips } = base();
    const evening = workC("d1-eve", "g1", { time_of_day: "evening" });
    onlyFails({ t: setSlot(t, "d1-s2", ["d1-eve"]), clips: [...clips, evening] }, "time_of_day");
  });

  it("time_of_day honours the recorded widen_window_30m relaxation", () => {
    const { t, clips } = base();
    const midday = workC("d1-mid", "g1", { time_of_day: "midday", source_end_s: 1800 });
    const shrunk = mapSlot(t, "d1-s2", (s) => ({ ...s, window: ["09:00", "10:40"] }));
    const strict = { t: setSlot(shrunk, "d1-s2", ["d1-mid"]), clips: [...clips, midday] };
    onlyFails(strict, "time_of_day");
    const relaxed = setSlot(shrunk, "d1-s2", ["d1-mid"], ["widen_window_30m"]);
    expect(verifyTimeline(relaxed, memoryCatalog(strict.clips)).pass).toBe(true);
  });

  it("exceptions violation is caught when the slot node is not the exception's place node", () => {
    const b = withExceptionLunch();
    const bad = mapSlot(b.t, "d1-s3", (s) => ({ ...s, spec: { ...s.spec, node: "workplace" } }));
    onlyFails({ t: setSlot(bad, "d1-s3", ["d1-lunch"]), clips: b.clips }, "exceptions");
  });

  it("exceptions violation is caught for a slot exception without meal_place", () => {
    const { t, clips } = base();
    const { meal_place: _drop, ...noPlace } = slotExc;
    onlyFails({ t: mapDay(t, 1, (d) => ({ ...d, exceptions: [noPlace] })), clips }, "exceptions");
  });

  it("exceptions violation is caught when the replaced role has no slot", () => {
    const { t, clips } = base();
    const ex = { ...slotExc, type: "dinner_out", replaces: "dinner" };
    onlyFails({ t: mapDay(t, 1, (d) => ({ ...d, exceptions: [ex] })), clips }, "exceptions");
  });

  it("home and workplace meal places resolve to their own node names", () => {
    const { t, clips } = base();
    const ex = { ...slotExc, meal_place: "workplace" as const };
    expect(verifyTimeline(mapDay(t, 1, (d) => ({ ...d, exceptions: [ex] })), memoryCatalog(clips)).pass).toBe(true);
  });

  it("exceptions violation is caught for two exceptions on one day", () => {
    const { t, clips } = base();
    onlyFails({ t: mapDay(t, 1, (d) => ({ ...d, exceptions: [slotExc, { ...slotExc, type: "other" }] })), clips }, "exceptions");
  });

  it("exceptions violation is caught for a day exception with a wrong day_range", () => {
    const { t, clips } = base();
    const ex = { scope: "day" as const, type: "leave_day", replaces: "workday", day_range: [1, 2] as [number, number] };
    onlyFails({ t: mapDay(t, 1, (d) => ({ ...d, exceptions: [ex] })), clips }, "exceptions");
  });

  it("missing clips fail the clip rules with the ids named and do not throw", () => {
    const { t, clips } = base();
    const r = verifyTimeline(t, memoryCatalog(clips.filter((c) => c.clip_id !== "d1-am")));
    expect(r.missing_clips).toEqual(["d1-am"]);
    expect(r.pass).toBe(false);
    for (const id of ["slot_window", "no_overlap", "monotonic", "node_binding", "clip_unique", "day_group_single_day", "time_of_day"] as const) {
      expect(r.rules[id].pass, id).toBe(false);
      expect(r.rules[id].violations[0]!.detail).toContain("d1-am");
    }
    expect(r.rules.day_type.pass).toBe(true);
  });

  it("swap_point flag for same node, different day_group", () => {
    const { t, clips } = base();
    // Day 1 lunch (workplace) comes from another group than the work slots; the day group is still single-day.
    const swapped = clips.map((c) => (c.clip_id === "d1-lunch" ? { ...c, day_group: "g9" } : c));
    const r = verifyTimeline(t, memoryCatalog(swapped));
    expect(r.pass).toBe(true);
    expect(r.flags).toContainEqual({ kind: "swap_point", day_index: 1, slot_ids: ["d1-s2", "d1-s3"] });
    expect(verifyTimeline(t, memoryCatalog(clips)).flags).toEqual([]);
  });

  it("no swap_point for the same node separated by a different node", () => {
    const { t, clips } = base();
    const evening: SlotDef = { role: "home_evening", category: "home", setting: "home_indoor", node: "home", activity: "relaxing", required: false, window: ["20:00", "22:00"] };
    const t0 = createTimeline(makeWorld(), makeSchedule(2, () => [...DEFS, evening]), genId);
    const slots = { ...t0.slots, ...t.slots, "d1-s5": { clip_ids: ["d1-eve"], relaxations: [] }, "d2-s5": { clip_ids: ["d2-eve"], relaxations: [] } };
    const r = verifyTimeline({ ...t0, slots }, memoryCatalog([...clips, homeC("d1-eve", "g3"), homeC("d2-eve", "g2")]));
    expect(r.pass).toBe(true);
    expect(r.flags.filter((f) => f.day_index === 1)).toEqual([]);
    expect(r.seams.filter((x) => x.day_index === 1).map((x) => x.after_clip)).toEqual(["d1-eve"]);
  });

  it("swap_point flag when one slot holds clips of two day groups", () => {
    const { t, clips } = base();
    const extra = workC("d1-pm2", "g3");
    const r = verifyTimeline(setSlot(t, "d1-s4", ["d1-pm", "d1-pm2"]), memoryCatalog([...clips, extra]));
    expect(r.flags).toEqual([{ kind: "swap_point", day_index: 1, slot_ids: ["d1-s4", "d1-s4"] }]);
  });

  it("seam list has the expected at_s", () => {
    const { t, clips } = base();
    const swapped = clips.map((c) => (c.clip_id === "d1-lunch" ? { ...c, day_group: "g9" } : c));
    const r = verifyTimeline(t, memoryCatalog(swapped));
    // Lunch (20 min) fills part of 12:00-12:45; its start is its seam point and so is the work_pm start after it.
    expect(r.seams.map((s) => [s.day_index, s.before_clip, s.after_clip])).toEqual([
      [1, "d1-am", "d1-lunch"],
      [1, "d1-lunch", "d1-pm"],
    ]);
    const lunchStart = r.seams[0]!.at_s;
    expect(lunchStart).toBeGreaterThanOrEqual(12 * 3600);
    expect(lunchStart).toBeLessThanOrEqual(12 * 3600 + 25 * 60);
    expect(r.seams[1]!.at_s).toBeGreaterThanOrEqual(13 * 3600);
  });
});

const readConfig = (rel: string): unknown =>
  JSON.parse(readFileSync(new URL(`../../../config/${rel}`, import.meta.url), "utf8"));

/** Synthetic donor rows: 30 weekday and 30 weekend respondents with fixed times. */
function donorPool(): DonorRow[] {
  const row = (i: number, weekend: boolean): DonorRow => ({
    id: `${weekend ? "we" : "wd"}${i}`,
    daytype: weekend ? "weekend" : "weekday",
    occ: OCCUPATION_CODES["professional"]!,
    w: 1 + (i % 3),
    wake: weekend ? 480 : 400 + (i % 5) * 5,
    bed: 1380,
    work_start: weekend ? null : 510,
    work_end: weekend ? null : 1010,
    work_min: weekend ? 0 : 480,
    worked: !weekend,
    breakfast_time: weekend ? 510 : 430,
    breakfast_loc: 1,
    lunch_time: 750,
    lunch_loc: 2,
    dinner_time: 1110,
    dinner_loc: 1,
    sport_has: false,
    sport_time: null,
    sport_outdoor: null,
    house_min: 30,
    social_out: false,
  });
  return Array.from({ length: 30 }, (_, i) => [row(i, false), row(i, true)]).flat();
}

describe("verifyTimeline on a generated schedule", () => {
  it("slot exceptions placed after windows pass the exceptions rule", () => {
    const fixture = makeWorld({ seed: 11 });
    const world = WorldSchema.parse({ ...fixture, dials: { ...fixture.dials, exception_rate: { slot: 0.5, day: 0.1 } } });
    const schedule = generateSchedule({
      world,
      templates: TemplatesSchema.parse(readConfig("timeline/templates.v3.json")),
      gaps: GapTableSchema.parse(readConfig("timeline/gaps.v1.json")) as GapTable,
      exceptions: ExceptionCatalogSchema.parse(readConfig("timeline/exceptions.v1.json")),
      source: atus2025Source(donorPool()),
      boundNodes: new Set(Object.keys(world.node_binding)),
      defaults: ScheduleDefaultsSchema.parse(readConfig("schedule_defaults_atus2025.json")),
    });
    const slotExceptions = schedule.days.flatMap((d) => d.exceptions.filter((e) => e.scope === "slot"));
    expect(slotExceptions.length).toBeGreaterThan(0);
    const r = verifyTimeline(createTimeline(world, schedule, genId), memoryCatalog([]));
    expect(r.rules.exceptions).toEqual({ pass: true, violations: [] });
  });
});
