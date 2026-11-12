import { spawnSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  TimelineError,
  TimelineManifestSchema,
  createTimeline,
  generateSchedule,
  parseClock,
  parseWorld,
  verifyTimeline,
  type Annotation,
  type Clip,
  type ClipCatalog,
  type ClipCategory,
  type ScheduleDay,
  type Setting,
  type Slot,
  type World,
} from "@vb/core";
import { loadConfig, scheduleInput } from "../src/config.js";
import {
  assignDates,
  collectDates,
  composeBlocks,
  composeBlocksFit,
  deOverlap,
  eveningFill,
  homeFill,
  maxWeightAssignment,
  planDay,
  realStartOf,
  topUpFill,
  type DayLike,
  type DateAssignment,
} from "../src/composeBlocks.js";
import { CONFIG_DIR, EXAMPLE_WORLD, REPO_ROOT, filesIn, makeTempDir, writeAtusPool, writeDataset } from "./helpers.js";

// Synthetic data only: every clip below is made up for these tests.

const PERSON = "nextme800:wearer";
const TRIP = ["2026-06-29", "2026-06-30", "2026-07-01"];

const sec = (hhmm: string) => {
  const [h, m] = hhmm.split(":").map(Number);
  return h! * 3600 + m! * 60;
};

type Spec = { at: string; min: number; node: string; category?: ClipCategory; person?: string };

const DEFAULT_CATEGORY: Record<string, ClipCategory> = {
  home: "home",
  workplace: "work",
  transit: "transit",
  frequent_1: "meal",
  "trip:lodging": "home",
  "trip:outing": "hobby",
  "trip:meal": "meal",
};
const SETTING: Record<string, Setting> = { home: "home_indoor", transit: "outdoor", "trip:outing": "outdoor" };

/** One synthetic recording of `date` that starts at `wall`; clip ids end in the clip's real start. */
function rec(date: string, specs: Spec[], o: { wall?: string; video?: string; group?: string } = {}): Clip[] {
  const wall = o.wall ?? "06:00";
  const video = o.video ?? `rec-${date}`;
  return specs.map((s) => {
    const start = sec(s.at) - sec(wall);
    const category = s.category ?? DEFAULT_CATEGORY[s.node] ?? "home";
    return {
      clip_id: `nextme800:${video}:${s.at}`,
      category,
      setting: SETTING[s.node] ?? "venue_indoor",
      node: s.node,
      // B18: a meal-category fixture clip carries an eating word in its L3 text, as a real one does.
      activity: category === "meal" ? `eating a meal (synthetic ${s.node})` : `synthetic ${s.node}`,
      time_of_day: "unknown",
      tags: [],
      source: { dataset: "nextme800", video_id: video, media: { kind: "frames", approx_fps: 1, tar: `${video}.tar`, start_wallclock: `${date} ${wall}:00` } },
      source_start_s: start,
      source_end_s: start + s.min * 60,
      day_group: o.group ?? date,
      person_source: s.person ?? PERSON,
      fingerprint: null,
      extractor_version: "fixture.v1",
    };
  });
}

/** A work date: home breakfast, commute, work, a two-part lunch, a later snack, commute, home dinner. */
function workDate(date: string, o: { late?: boolean; homeMorning?: boolean; extraWork?: number } = {}): Clip[] {
  const shift = o.late ? 4 * 60 : 0;
  const t = (hhmm: string) => {
    const m = sec(hhmm) / 60 + shift;
    return `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
  };
  const morning: Spec[] = o.homeMorning === false ? [] : [
    { at: t("07:00"), min: 30, node: "home" },
    { at: t("07:30"), min: 15, node: "home", category: "meal" },
  ];
  const day = rec(date, [
    ...morning,
    { at: t("08:00"), min: 30, node: "transit" },
    { at: t("08:40"), min: 120 + (o.extraWork ?? 0), node: "workplace" },
  ]);
  if (o.late) return [...day, ...rec(date, [{ at: "16:00", min: 60, node: "workplace" }], { video: `rec-${date}-pm`, wall: "15:00" })];
  return [
    ...day,
    ...rec(date, [
      { at: "10:45", min: 60, node: "workplace" },
      { at: "12:00", min: 20, node: "workplace", category: "meal" },
      { at: "12:25", min: 15, node: "workplace", category: "meal" },
      { at: "13:00", min: 60, node: "workplace" },
      { at: "14:20", min: 10, node: "workplace", category: "meal" },
      { at: "14:40", min: 90, node: "workplace" },
    ], { video: `rec-${date}-mid`, wall: "10:00" }),
    // The evening recording starts in the afternoon, so real times need the wall clock plus the offset.
    ...rec(date, [
      { at: "17:30", min: 40, node: "transit" },
      { at: "18:20", min: 20, node: "home" },
      { at: "18:45", min: 30, node: "home", category: "meal" },
      { at: "19:20", min: 60, node: "home" },
    ], { video: `rec-${date}-pm`, wall: "17:00" }),
  ];
}

function restDate(date: string): Clip[] {
  return rec(date, [
    { at: "08:00", min: 60, node: "home" },
    { at: "09:10", min: 20, node: "home", category: "meal" },
    { at: "10:00", min: 60, node: "home" },
    { at: "12:30", min: 30, node: "home", category: "meal" },
    { at: "13:10", min: 60, node: "home" },
    { at: "14:20", min: 60, node: "home" },
    { at: "15:30", min: 60, node: "home" },
    { at: "18:00", min: 40, node: "home", category: "meal" },
    { at: "19:00", min: 60, node: "home" },
  ]);
}

function tripDate(date: string): Clip[] {
  return rec(date, [
    { at: "07:30", min: 30, node: "trip:lodging" },
    { at: "08:10", min: 20, node: "trip:meal" },
    { at: "09:00", min: 60, node: "trip:outing" },
    { at: "10:05", min: 60, node: "trip:outing" },
    { at: "11:30", min: 30, node: "trip:outing" },
    { at: "12:10", min: 40, node: "trip:meal" },
    { at: "13:00", min: 90, node: "trip:outing" },
    { at: "17:30", min: 45, node: "trip:meal" },
    { at: "18:30", min: 60, node: "trip:lodging" },
  ]);
}

class MemCatalog implements ClipCatalog {
  private readonly byId: Map<string, Clip>;
  constructor(private readonly all: Clip[]) {
    this.byId = new Map(all.map((c) => [c.clip_id, c]));
  }
  clips() {
    return this.all;
  }
  getClip(id: string) {
    return this.byId.get(id);
  }
  getAnnotations(): Annotation[] {
    return [];
  }
}

function worldJson(o: { trip?: boolean; seed?: number; late?: number } = {}) {
  const base = JSON.parse(readFileSync(EXAMPLE_WORLD, "utf8"));
  base.person_id = "compose-test";
  if (o.seed !== undefined) base.seed = o.seed;
  if (o.late !== undefined) base.dials.late_workdays_per_week = o.late;
  if (o.trip !== false) {
    base.nodes.trip_1 = { kind: "trip", lodging_setting: "venue_indoor", outing_setting: "outdoor", meal_setting: "venue_indoor" };
    base.node_binding.trip_1 = { dataset: ["nextme800"], person_source: PERSON, day_group_sequence: TRIP };
    base.dials.trips = [{ node: "trip_1", length_days: 3, start_day: 10 }];
  }
  return base;
}
const makeWorld = (o: Parameters<typeof worldJson>[0] = {}): World => parseWorld(worldJson(o));

const SPEC_CATEGORY: Record<string, ClipCategory> = { lunch: "meal", dinner: "meal", breakfast: "meal", hobby: "hobby" };

function slot(i: number, role: string, node: string, from: string, to: string): Slot {
  return {
    slot_id: `d1-s${i}`,
    role,
    required: false,
    window: [from, to],
    spec: { category: SPEC_CATEGORY[role] ?? "home", setting: "venue_indoor", node, activity: role, time_of_day: "morning", tags: [] },
  };
}

function mkDay(day_type: ScheduleDay["day_type"], slots: Slot[], transitions: [number, number, number][] = [], trip?: ScheduleDay["trip"]): ScheduleDay {
  return {
    day_index: 1,
    weekday: "Mon",
    day_type,
    ...(trip ? { trip } : {}),
    attempt: 0,
    wake: "06:30",
    sleep: "23:00",
    exceptions: [],
    slot_exception_eligible: false,
    slot_exception_candidates: 0,
    slots,
    dropped_slots: [],
    transitions: transitions.map(([f, to, minutes]) => ({ from_slot: `d1-s${f}`, to_slot: `d1-s${to}`, minutes })),
    anchor_adjustments: [],
    jitter_redraws: 0,
  };
}

const WORKDAY = [
  slot(1, "home_morning", "home", "07:00", "08:00"),
  slot(2, "work_am", "workplace", "08:30", "12:00"),
  slot(3, "lunch", "workplace", "12:00", "12:45"),
  slot(4, "work_pm", "workplace", "12:45", "17:30"),
  slot(5, "dinner", "home", "18:30", "19:15"),
  slot(6, "hobby", "hobby_1", "19:30", "20:30"),
  slot(7, "home_evening", "home", "20:30", "22:00"),
];

/** Plan entries grouped by block id, as real start clocks of their clips. */
function byBlock(world: World, day: ScheduleDay, clips: Clip[], date: string): Record<string, string[]> {
  const dates = collectDates(new MemCatalog(clips), world);
  const plan = planDay(day, dates.get(date), world);
  const ids = new Map(clips.map((c) => [c.clip_id, c]));
  const out: Record<string, string[]> = {};
  for (const e of plan) {
    expect(e.source_start_s).toBe(ids.get(e.clip_id)!.source_start_s);
    expect(e.source_end_s).toBe(ids.get(e.clip_id)!.source_end_s);
    const at = e.clip_id.slice(e.clip_id.lastIndexOf(":") - 2);
    (out[e.slot_id] ??= []).push(at);
  }
  return out;
}

describe("real dates", () => {
  const world = makeWorld();

  it("real start is the recording's wall clock plus the clip offset, from the date's 00:00", () => {
    const [evening] = rec("2026-05-04", [{ at: "18:20", min: 20, node: "home" }], { wall: "17:00" });
    expect(realStartOf(evening!)).toBe(sec("18:20"));
    // A recording that started the evening before the clip's date counts back from that date's midnight.
    const [late] = rec("2026-05-04", [{ at: "23:30", min: 20, node: "home" }], { wall: "23:00", group: "2026-05-05" });
    expect(realStartOf(late!)).toBe(sec("23:30") - 86400);
    const noClock = { ...evening!, source: { dataset: "nextme800" as const, video_id: "v", media: { kind: "video" as const, fps: 30 } } };
    expect(realStartOf(noClock)).toBeUndefined();
  });

  it("classifies work, rest, partial and trip dates from the bound person's clips", () => {
    const clips = [
      ...workDate("2026-05-04"),
      ...workDate("2026-05-05", { late: true }),
      ...restDate("2026-05-09"),
      ...rec("2026-05-12", [{ at: "09:00", min: 30, node: "workplace" }, { at: "10:00", min: 60, node: "home" }]),
      ...tripDate(TRIP[0]!),
      // Another person's work footage does not make a rest date a work date.
      ...rec("2026-05-10", [{ at: "09:00", min: 600, node: "workplace", person: "someone:else" }, { at: "11:00", min: 30, node: "home" }]),
    ];
    const dates = collectDates(new MemCatalog(clips), world);
    const view = Object.fromEntries([...dates].map(([d, x]) => [d, { kind: x.kind, first_work_s: x.first_work_s, home_first: x.home_first }]));
    expect(view).toEqual({
      "2026-05-04": { kind: "work", first_work_s: sec("08:40"), home_first: true },
      "2026-05-05": { kind: "work", first_work_s: sec("12:40"), home_first: true },
      "2026-05-09": { kind: "rest", first_work_s: null, home_first: true },
      "2026-05-10": { kind: "rest", first_work_s: null, home_first: true },
      "2026-05-12": { kind: "partial", first_work_s: sec("09:00"), home_first: false },
      [TRIP[0]!]: { kind: "trip", first_work_s: null, home_first: false },
    });
    expect(dates.get("2026-05-04")!.work_seconds).toBe((120 + 60 + 20 + 15 + 60 + 10 + 90) * 60);
    expect(dates.get("2026-05-10")!.clips.map((c) => c.clip.person_source)).toEqual([PERSON]);
  });
});

describe("overlapping clips of one video", () => {
  const world = makeWorld();

  it("keeps the longest clip of a video and drops the ones nested in or overlapping it", () => {
    const clips = rec("2026-05-04", [
      { at: "08:00", min: 60, node: "workplace" },
      { at: "08:10", min: 20, node: "workplace" }, // nested
      { at: "08:50", min: 30, node: "workplace" }, // partly overlapping
      { at: "09:20", min: 30, node: "workplace" }, // touches the dropped 08:50 clip only: kept
    ]);
    const other = rec("2026-05-04", [{ at: "08:10", min: 20, node: "workplace" }], { video: "rec-b" });
    expect(deOverlap([...clips, ...other]).map((c) => c.clip_id)).toEqual([
      "nextme800:rec-2026-05-04:08:00",
      "nextme800:rec-2026-05-04:09:20",
      "nextme800:rec-b:08:10",
    ]);
    const date = collectDates(new MemCatalog(clips), world).get("2026-05-04")!;
    expect(date.clips.map((c) => c.clip.clip_id)).toEqual(["nextme800:rec-2026-05-04:08:00", "nextme800:rec-2026-05-04:09:20"]);
    expect(date.work_seconds).toBe(90 * 60);
  });
});

describe("date assignment", () => {
  const world = makeWorld();
  const day = (day_index: number, day_type: DayLike["day_type"], trip?: DayLike["trip"]): DayLike => ({
    day_index,
    day_type,
    exceptions: [],
    ...(trip ? { trip } : {}),
  });

  it("takes weekday work dates before Saturday and Sunday ones, for plain and late days alike", () => {
    // 2026-05-09 is a Saturday and ranks best (more footage); 05-04 and 05-05 are Monday and Tuesday.
    const dates = collectDates(
      new MemCatalog([...workDate("2026-05-04"), ...workDate("2026-05-05"), ...workDate("2026-05-09", { extraWork: 600 })]),
      world,
    );
    for (const seed of [1, 2, 3]) {
      const w = makeWorld({ seed });
      const two = assignDates([day(1, "workday"), day(2, "workday")], dates, w).map((a) => a.date).sort();
      expect(two).toEqual(["2026-05-04", "2026-05-05"]);
      const lateTwo = assignDates([day(1, "workday_late"), day(2, "workday")], dates, w).map((a) => a.date).sort();
      expect(lateTwo).toEqual(["2026-05-04", "2026-05-05"]);
      const three = assignDates([day(1, "workday"), day(2, "workday"), day(3, "workday_late")], dates, w);
      expect(three.filter((a) => a.date !== "2026-05-09")).toHaveLength(2);
      expect(three.map((a) => a.date).sort()).toEqual(["2026-05-04", "2026-05-05", "2026-05-09"]);
    }
    // A weekday date on the wrong side of the late preference still beats a Saturday date on the right side.
    const lateSat = collectDates(new MemCatalog([...workDate("2026-05-04"), ...workDate("2026-05-09", { late: true })]), world);
    expect(assignDates([day(1, "workday_late")], lateSat, world)[0]!.date).toBe("2026-05-04");
    expect(assignDates([day(1, "workday_late"), day(2, "workday")], lateSat, world).map((a) => a.date).sort()).toEqual(["2026-05-04", "2026-05-09"]);
  });

  it("uses each date on at most one day and leaves a day without a date when its pool runs out", () => {
    const dates = collectDates(new MemCatalog([...workDate("2026-05-04"), ...workDate("2026-05-06"), ...restDate("2026-05-09")]), world);
    const out = assignDates([day(1, "workday"), day(2, "workday"), day(3, "workday"), day(4, "home_day"), day(5, "hobby_day")], dates, world);
    expect(out.map((a) => a.day_index)).toEqual([1, 2, 3, 4, 5]);
    const used = out.flatMap((a) => (a.date === null ? [] : [a.date]));
    expect(new Set(used).size).toBe(used.length);
    expect(out.filter((a) => a.pool === "work" && a.date !== null).map((a) => a.date).sort()).toEqual(["2026-05-04", "2026-05-06"]);
    expect(out.filter((a) => a.date === null).map((a) => [a.pool, a.day_type])).toEqual([
      ["work", "workday"],
      ["rest", out.find((a) => a.date === null && a.pool === "rest")!.day_type],
    ]);
    expect(out.find((a) => a.pool === "rest" && a.date !== null)!.date).toBe("2026-05-09");
  });

  it("gives late days the late first-work dates and plain workdays the early ones, falling back when one side is empty", () => {
    const dates = collectDates(new MemCatalog([...workDate("2026-05-04"), ...workDate("2026-05-05", { late: true })]), world);
    for (const seed of [1, 2, 3, 4, 5]) {
      const w = makeWorld({ seed });
      const out = assignDates([day(1, "workday_late"), day(2, "workday")], dates, w);
      expect(out.map((a) => a.date)).toEqual(["2026-05-05", "2026-05-04"]);
    }
    const onlyEarly = collectDates(new MemCatalog(workDate("2026-05-04")), world);
    expect(assignDates([day(1, "workday_late")], onlyEarly, world)[0]!.date).toBe("2026-05-04");
    const onlyLate = collectDates(new MemCatalog(workDate("2026-05-05", { late: true })), world);
    expect(assignDates([day(1, "workday")], onlyLate, world)[0]!.date).toBe("2026-05-05");
  });

  it("uses partial work dates for rest days only after the rest dates run out", () => {
    // The partial date ranks higher (home first, more footage) but still comes after the rest date.
    const partial = rec("2026-05-12", [{ at: "06:00", min: 600, node: "home" }, { at: "16:30", min: 30, node: "workplace" }]);
    const dates = collectDates(new MemCatalog([...restDate("2026-05-09"), ...partial]), world);
    expect(dates.get("2026-05-12")!.kind).toBe("partial");
    expect(assignDates([day(1, "home_day")], dates, world).map((a) => a.date)).toEqual(["2026-05-09"]);
    const three = assignDates([day(1, "home_day"), day(2, "leave_day"), day(3, "hobby_day")], dates, world);
    expect(three.flatMap((a) => (a.date === null ? [] : [a.date])).sort()).toEqual(["2026-05-09", "2026-05-12"]);
    expect(three.filter((a) => a.date === null)).toHaveLength(1);
    expect(assignDates([day(1, "workday")], dates, world)[0]!.date).toBeNull();
  });

  it("breaks ties with the seed, so the best date is not always on the first day", () => {
    const clips = Array.from({ length: 6 }, (_, i) => workDate(`2026-05-0${i + 1}`, { extraWork: 10 * i })).flat();
    const dates = collectDates(new MemCatalog(clips), world);
    const days = Array.from({ length: 6 }, (_, i) => day(i + 1, "workday"));
    const best = (seed: number) => assignDates(days, dates, makeWorld({ seed })).find((a) => a.date === "2026-05-06")!.day_index;
    const seen = new Set(Array.from({ length: 12 }, (_, s) => best(s)));
    expect(seen.size).toBeGreaterThan(1);
    expect(best(7)).toBe(best(7));
  });

  /** Day `n` of `day_type` with slot ids d<n>-s<k>. */
  const full = (n: number, day_type: ScheduleDay["day_type"], slots: Slot[]): ScheduleDay => ({
    ...mkDay(day_type, slots.map((s) => ({ ...s, slot_id: s.slot_id.replace("d1-", `d${n}-`) }))),
    day_index: n,
  });
  const work = (at: string, min: number): Spec => ({ at, min, node: "workplace" });

  it("B12: maximizes the total planned footage where a greedy order would not", () => {
    // 05-05 has morning and afternoon work, 05-04 morning work only. Day 1 gains little from the
    // afternoon (a short work_pm), day 2 has only a work_pm: the best total gives 05-05 to day 2,
    // although day 1 alone would fill more with it and 05-05 is the richer date.
    const dates = collectDates(
      new MemCatalog([...rec("2026-05-05", [work("08:40", 240), work("13:00", 240)]), ...rec("2026-05-04", [work("08:40", 240)])]),
      world,
    );
    const d1 = full(1, "workday", [slot(1, "work_am", "workplace", "08:30", "12:00"), slot(2, "work_pm", "workplace", "12:45", "14:00")]);
    const d2 = full(2, "workday", [slot(1, "work_pm", "workplace", "12:45", "17:30")]);
    for (const seed of [1, 2, 3, 4, 5, 6]) {
      for (const days of [[d1, d2], [d2, d1]]) {
        const out = assignDates(days, dates, makeWorld({ seed }));
        expect(Object.fromEntries(out.map((a) => [a.day_index, a.date]))).toEqual({ 1: "2026-05-04", 2: "2026-05-05" });
      }
    }
  });

  it("B12: keeps the tiers hard: weekday before weekend work dates, rest before partial dates", () => {
    const pm = [slot(1, "work_am", "workplace", "08:30", "12:00"), slot(2, "work_pm", "workplace", "12:45", "17:30")];
    // A Saturday date full of work still loses to a Monday date with one hour of it.
    const weekend = collectDates(new MemCatalog([...rec("2026-05-04", [work("09:00", 60)]), ...workDate("2026-05-09", { extraWork: 300 })]), world);
    expect(assignDates([full(1, "workday", pm)], weekend, world).map((a) => a.date)).toEqual(["2026-05-04"]);
    expect(assignDates([full(1, "workday", pm), full(2, "workday", pm)], weekend, world).map((a) => a.date).sort()).toEqual(["2026-05-04", "2026-05-09"]);
    // A partial date full of home footage still loses to a rest date with ten minutes of it.
    const rest = collectDates(
      new MemCatalog([
        ...rec("2026-05-09", [{ at: "08:00", min: 10, node: "home" }]),
        ...rec("2026-05-12", [{ at: "06:00", min: 600, node: "home" }, work("16:30", 30)]),
      ]),
      world,
    );
    const homeDay = full(1, "home_day", [slot(1, "home_morning", "home", "07:00", "09:00"), slot(2, "home_afternoon", "home", "12:00", "17:00")]);
    expect(assignDates([homeDay], rest, world).map((a) => a.date)).toEqual(["2026-05-09"]);
  });

  it("B12: a late day gets one hour of bonus for a first work at or after 11:00, a plain workday for an earlier one", () => {
    const pm = [slot(1, "work_pm", "workplace", "12:45", "17:30")];
    // 05-05: first work 13:00, 2 h in work_pm. 05-04: first work 09:00, more in work_pm.
    const lateDate = rec("2026-05-05", [work("13:00", 120)]);
    const within = collectDates(new MemCatalog([...lateDate, ...rec("2026-05-04", [work("09:00", 10), work("13:00", 150)])]), world);
    const beyond = collectDates(new MemCatalog([...lateDate, ...rec("2026-05-04", [work("09:00", 10), work("13:00", 190)])]), world);
    for (const seed of [1, 2, 3]) {
      const w = makeWorld({ seed });
      expect(assignDates([full(1, "workday_late", pm)], within, w)[0]!.date).toBe("2026-05-05"); // 2 h + 1 h > 2.5 h
      expect(assignDates([full(1, "workday_late", pm)], beyond, w)[0]!.date).toBe("2026-05-04"); // 3 h 10 min > 2 h + 1 h
      expect(assignDates([full(1, "workday", pm)], within, w)[0]!.date).toBe("2026-05-04");
    }
  });

  it("gives trip days the trip sequence in order and leave days rest dates", () => {
    const dates = collectDates(new MemCatalog([...TRIP.flatMap(tripDate), ...restDate("2026-05-09"), ...workDate("2026-05-04")]), world);
    const out = assignDates(
      [
        day(1, "trip_day", { node: "trip_1", index: 0 }),
        day(2, "trip_day", { node: "trip_1", index: 1 }),
        day(3, "trip_day", { node: "trip_1", index: 2 }),
        day(4, "leave_day"),
      ],
      dates,
      world,
    );
    expect(out.map((a) => [a.pool, a.date])).toEqual([
      ["trip", TRIP[0]],
      ["trip", TRIP[1]],
      ["trip", TRIP[2]],
      ["rest", "2026-05-09"],
    ]);
  });
});

describe("max-weight assignment (B12)", () => {
  /** Best total by brute force: every row takes an unused column or none (0). */
  function brute(v: number[][], row = 0, used = new Set<number>()): number {
    if (row === v.length) return 0;
    let best = brute(v, row + 1, used);
    v[row]!.forEach((x, j) => {
      if (used.has(j)) return;
      used.add(j);
      best = Math.max(best, x + brute(v, row + 1, used));
      used.delete(j);
    });
    return best;
  }

  it("matches brute force on small rectangular matrices and uses each column once", () => {
    let state = 12345;
    const next = () => (state = (state * 1103515245 + 12345) % 2147483648) % 50 + 1;
    for (let n = 1; n <= 5; n++) {
      for (let m = 1; m <= 5; m++) {
        for (let k = 0; k < 6; k++) {
          const v = Array.from({ length: n }, () => Array.from({ length: m }, next));
          const got = maxWeightAssignment(v);
          expect(got).toHaveLength(n);
          const cols = got.filter((j): j is number => j !== null);
          expect(new Set(cols).size).toBe(cols.length);
          expect(got.reduce<number>((s, j, i) => s + (j === null ? 0 : v[i]![j]!), 0)).toBe(brute(v));
          expect(cols.length).toBe(Math.min(n, m)); // positive values: every row that can be matched is
        }
      }
    }
  });
});

describe("real day phases", () => {
  const world = makeWorld();

  it("maps a workday's roles to the real phases, with commute clips in the home-work transitions and hobby empty", () => {
    const clips = [
      ...workDate("2026-05-04"),
      // Ignored: a meal at a place the world does not bind, and another person's home footage.
      ...rec("2026-05-04", [{ at: "11:30", min: 20, node: "frequent_1" }], { video: "rec-out", wall: "11:00" }),
      ...rec("2026-05-04", [{ at: "07:10", min: 20, node: "home", person: "someone:else" }], { video: "rec-other" }),
    ];
    const day = mkDay("workday", WORKDAY, [[1, 2, 30], [4, 5, 45]]);
    expect(byBlock(world, day, clips, "2026-05-04")).toEqual({
      "d1-s1": ["07:00"], // B18: the 07:30 breakfast clip (eating word) no longer fills home_morning
      "d1-s2": ["08:40", "10:45"],
      "d1-s3": ["12:00", "12:25"],
      "d1-s4": ["13:00", "14:40"],
      "d1-s5": ["18:45"],
      "d1-s7": ["18:20", "19:20"],
      "d1-t1": ["08:00"],
      "d1-t2": ["17:30"],
    });
  });

  it("fills work_am from the first workplace clips when the real lunch comes before any work", () => {
    const clips = [
      ...rec("2026-06-13", [
        { at: "10:00", min: 30, node: "home" },
        { at: "12:00", min: 30, node: "workplace", category: "meal" },
        { at: "13:38", min: 60, node: "workplace" },
        { at: "15:00", min: 60, node: "workplace" },
        { at: "16:30", min: 60, node: "workplace" },
        { at: "18:00", min: 60, node: "workplace" },
      ]),
    ];
    const day = mkDay("workday_late", [
      slot(1, "work_am", "workplace", "13:00", "15:00"),
      slot(2, "lunch", "workplace", "15:00", "15:30"),
      slot(3, "work_pm", "workplace", "15:30", "20:00"),
    ]);
    const got = byBlock(world, day, clips, "2026-06-13");
    expect(got["d1-s1"]).toEqual(["13:38", "15:00"]);
    expect(got["d1-s3"]).toEqual(["16:30", "18:00"]);
  });

  it("takes commute bounds from the real phases, not from the clips a short slot planned", () => {
    const clips = rec("2026-05-04", [
      { at: "07:00", min: 30, node: "home" },
      { at: "08:00", min: 30, node: "transit" },
      { at: "08:40", min: 120, node: "workplace" },
      { at: "12:00", min: 20, node: "workplace", category: "meal" },
      { at: "13:00", min: 60, node: "workplace" },
      { at: "14:10", min: 15, node: "transit" }, // an errand between two work clips
      { at: "14:30", min: 90, node: "workplace" },
      { at: "17:00", min: 30, node: "workplace", category: "meal" }, // a workplace dinner, placed in the home dinner slot
      { at: "17:30", min: 40, node: "transit" },
      { at: "18:20", min: 20, node: "home" },
      { at: "19:30", min: 20, node: "transit" }, // after the first home clip that follows the last work: not a commute
    ]);
    const day = mkDay("workday", [
      slot(1, "home_morning", "home", "07:00", "08:00"),
      slot(2, "work_am", "workplace", "08:30", "12:00"),
      slot(3, "lunch", "workplace", "12:00", "12:45"),
      slot(4, "work_pm", "workplace", "12:45", "13:45"),
      slot(5, "dinner", "home", "18:30", "19:15"),
    ], [[1, 2, 30], [4, 5, 45]]);
    const got = byBlock(world, day, clips, "2026-05-04");
    expect(got["d1-s4"]).toEqual(["13:00"]);
    expect(got["d1-s5"]).toEqual(["17:00"]);
    expect(got["d1-t1"]).toEqual(["08:00"]);
    expect(got["d1-t2"]).toEqual(["17:30"]);
  });

  it("on a rest date, home_morning takes only home footage before 12:00", () => {
    const day = mkDay("hobby_day", [
      slot(1, "home_morning", "home", "07:00", "13:00"),
      slot(2, "home_afternoon", "home", "13:00", "17:00"),
    ]);
    expect(byBlock(world, day, restDate("2026-05-09"), "2026-05-09")).toEqual({
      // B18: the 09:10 and 12:30 meal clips (eating words) stay out of the home slots
      "d1-s1": ["08:00", "10:00"],
      "d1-s2": ["13:10", "14:20", "15:30"],
    });
  });

  it("splits work at 13:00 when the date has no lunch-time meal", () => {
    const clips = rec("2026-05-04", [
      { at: "09:00", min: 60, node: "workplace" },
      { at: "12:30", min: 20, node: "workplace" },
      { at: "13:30", min: 60, node: "workplace" },
      { at: "18:00", min: 30, node: "workplace", category: "meal" },
    ]);
    const day = mkDay("workday", WORKDAY);
    expect(byBlock(world, day, clips, "2026-05-04")).toEqual({
      "d1-s2": ["09:00", "12:30"],
      "d1-s4": ["13:30"],
      "d1-s5": ["18:00"],
    });
  });

  it("fills a rest day; chores and home_afternoon share the 12:00-17:00 home footage without reuse", () => {
    const day = mkDay("home_day", [
      slot(1, "home_morning", "home", "08:00", "09:00"),
      slot(2, "breakfast", "home", "09:00", "09:30"),
      slot(3, "chores", "home", "09:30", "11:00"),
      slot(4, "lunch", "home", "12:00", "12:45"),
      slot(5, "home_afternoon", "home", "13:00", "15:00"),
      slot(6, "dinner", "home", "18:00", "18:45"),
      slot(7, "home_evening", "home", "19:00", "21:00"),
    ]);
    expect(byBlock(world, day, restDate("2026-05-09"), "2026-05-09")).toEqual({
      "d1-s1": ["08:00"],
      "d1-s2": ["09:10"],
      "d1-s3": ["13:10", "14:20"],
      "d1-s4": ["12:30"],
      "d1-s5": ["15:30"],
      "d1-s6": ["18:00"],
      "d1-s7": ["19:00"],
    });
  });

  it("maps a trip day's roles to trip lodging, outing and meal clips", () => {
    const day = mkDay(
      "trip_day",
      [
        slot(1, "lodging_morning", "trip_1:lodging", "07:00", "08:00"),
        slot(2, "breakfast", "trip_1:meal", "08:00", "08:30"),
        slot(3, "outing", "trip_1:outing", "08:30", "12:00"),
        slot(4, "lunch", "trip_1:meal", "12:00", "13:00"),
        slot(5, "outing_pm", "trip_1:outing", "13:00", "17:00"),
        slot(6, "dinner", "trip_1:meal", "17:30", "18:30"),
        slot(7, "lodging_evening", "trip_1:lodging", "18:30", "22:00"),
      ],
      [],
      { node: "trip_1", index: 0 },
    );
    expect(byBlock(world, day, tripDate(TRIP[0]!), TRIP[0]!)).toEqual({
      "d1-s1": ["07:30"],
      "d1-s2": ["08:10"],
      "d1-s3": ["09:00", "10:05"],
      "d1-s4": ["12:10"],
      "d1-s5": ["13:00"],
      "d1-s6": ["17:30"],
      "d1-s7": ["18:30"],
    });
  });

  it("plans only what fills a block and skips transitions with an empty span", () => {
    const day = mkDay("workday", [
      slot(1, "home_morning", "home", "07:00", "07:20"),
      slot(2, "work_am", "workplace", "07:20", "12:00"),
    ], [[1, 2, 30]]);
    // 20 min block: the 30 min home clip already fills it, so the breakfast clip stays unused.
    expect(byBlock(world, day, workDate("2026-05-04"), "2026-05-04")).toEqual({
      "d1-s1": ["07:00"],
      "d1-s2": ["08:40", "10:45"],
    });
  });

  it("returns no plan for a day without a real date", () => {
    expect(planDay(mkDay("workday", WORKDAY), undefined, world)).toEqual([]);
  });
});

describe("content filters from the L3 text (B18)", () => {
  const world = makeWorld();
  const text = (clips: Clip[], at: string, activity: string): Clip[] =>
    clips.map((c) => (c.clip_id.endsWith(`:${at}`) ? { ...c, activity } : c));

  it("excludes a cooking clip labeled home from home_evening", () => {
    const day = mkDay("home_day", [slot(1, "home_evening", "home", "19:00", "21:00")]);
    let clips = rec("2026-05-09", [{ at: "19:00", min: 20, node: "home" }, { at: "19:30", min: 20, node: "home" }]);
    clips = text(clips, "19:00", "I am Cooking noodles in the kitchen");
    expect(byBlock(world, day, clips, "2026-05-09")).toEqual({ "d1-s1": ["19:30"] });
  });

  it("excludes a mall walk labeled meal from dinner", () => {
    const day = mkDay("home_day", [slot(1, "dinner", "home", "18:00", "19:00")]);
    let clips = rec("2026-05-09", [
      { at: "18:00", min: 20, node: "home", category: "meal" },
      { at: "18:30", min: 20, node: "home", category: "meal" },
    ]);
    clips = text(clips, "18:00", "I walk through the mall");
    clips = text(clips, "18:30", "I eat dinner at the restaurant");
    expect(byBlock(world, day, clips, "2026-05-09")).toEqual({ "d1-s1": ["18:30"] });
  });

  it("uses a canteen lunch clip in lunch and never in work_pm", () => {
    const day = mkDay("workday", [
      slot(1, "work_am", "workplace", "08:30", "12:00"),
      slot(2, "lunch", "workplace", "12:00", "12:45"),
      slot(3, "work_pm", "workplace", "12:45", "17:30"),
    ]);
    let clips = rec("2026-05-04", [
      { at: "09:00", min: 60, node: "workplace" },
      { at: "12:00", min: 30, node: "workplace", category: "meal" },
      { at: "13:30", min: 60, node: "workplace" },
      { at: "15:00", min: 20, node: "workplace", category: "work" },
    ]);
    clips = text(clips, "12:00", "I eat lunch at the canteen");
    clips = text(clips, "15:00", "Eating a Snack at my desk");
    expect(byBlock(world, day, clips, "2026-05-04")).toEqual({ "d1-s1": ["09:00"], "d1-s2": ["12:00"], "d1-s3": ["13:30"] });
  });

  it("excludes a commute sentence labeled home from home_morning", () => {
    const day = mkDay("home_day", [slot(1, "home_morning", "home", "07:00", "09:00")]);
    let clips = rec("2026-05-09", [{ at: "07:00", min: 20, node: "home" }, { at: "07:30", min: 20, node: "home" }]);
    clips = text(clips, "07:00", "I leave home and take the MTR");
    expect(byBlock(world, day, clips, "2026-05-09")).toEqual({ "d1-s1": ["07:30"] });
  });

  it("matches whole words only, so a hobby text without them is kept", () => {
    const day = mkDay("home_day", [slot(1, "home_evening", "home", "19:00", "21:00")]);
    let clips = rec("2026-05-09", [{ at: "19:00", min: 20, node: "home" }, { at: "19:30", min: 20, node: "home" }]);
    clips = text(clips, "19:00", "I build the LEGO set (brice cream)");
    clips = text(clips, "19:30", "Playing Guitar");
    expect(byBlock(world, day, clips, "2026-05-09")).toEqual({ "d1-s1": ["19:00", "19:30"] });
  });
});

describe("narrative coherence (B19, B20)", () => {
  const world = makeWorld();
  const text = (clips: Clip[], at: string, activity: string): Clip[] =>
    clips.map((c) => (c.clip_id.endsWith(`:${at}`) ? { ...c, activity } : c));

  it("rejects a market walk home clip from home_afternoon", () => {
    const day = mkDay("home_day", [slot(1, "home_afternoon", "home", "13:00", "16:00")]);
    let clips = rec("2026-05-09", [{ at: "13:00", min: 20, node: "home" }, { at: "13:30", min: 20, node: "home" }]);
    clips = text(clips, "13:00", "I walk from the market to my apartment");
    expect(byBlock(world, day, clips, "2026-05-09")).toEqual({ "d1-s1": ["13:30"] });
  });

  it("rejects a produce-stall home clip from home_afternoon and a returning-home clip from home_morning", () => {
    const afternoon = mkDay("home_day", [slot(1, "home_afternoon", "home", "13:00", "16:00")]);
    let clips = rec("2026-05-09", [{ at: "13:00", min: 20, node: "home" }, { at: "13:30", min: 20, node: "home" }]);
    clips = text(clips, "13:00", "I purchase pork belly at the produce stall.");
    expect(byBlock(world, afternoon, clips, "2026-05-09")).toEqual({ "d1-s1": ["13:30"] });
    const morning = mkDay("home_day", [slot(1, "home_morning", "home", "08:00", "11:00")]);
    clips = rec("2026-05-09", [{ at: "08:00", min: 20, node: "home" }, { at: "08:30", min: 20, node: "home" }]);
    clips = text(clips, "08:00", "I examine items on the dining table upon returning home.");
    expect(byBlock(world, morning, clips, "2026-05-09")).toEqual({ "d1-s1": ["08:30"] });
  });

  it("rejects a scooter errand workplace clip from work_am", () => {
    const day = mkDay("workday", [slot(1, "work_am", "workplace", "08:30", "12:00")]);
    let clips = rec("2026-05-04", [{ at: "09:00", min: 20, node: "workplace" }, { at: "09:30", min: 20, node: "workplace" }]);
    clips = text(clips, "09:00", "I ride my scooter to McDonald's");
    expect(byBlock(world, day, clips, "2026-05-04")).toEqual({ "d1-s1": ["09:30"] });
  });

  it("leaves dinner empty when the run is drive, abandon, drive", () => {
    const day = mkDay("home_day", [slot(1, "dinner", "home", "18:00", "19:30")]);
    let clips = rec("2026-05-09", [
      { at: "18:00", min: 10, node: "home", category: "meal" },
      { at: "18:15", min: 10, node: "home", category: "meal" },
      { at: "18:30", min: 10, node: "home", category: "meal" },
    ]);
    clips = text(clips, "18:00", "I drive toward a restaurant");
    clips = text(clips, "18:15", "I abandon the restaurant");
    clips = text(clips, "18:30", "I drive to another restaurant");
    expect(byBlock(world, day, clips, "2026-05-09")).toEqual({});
  });

  it("keeps a cafeteria meal run and trims the clips around the eating", () => {
    const day = mkDay("home_day", [slot(1, "dinner", "home", "18:00", "19:30")]);
    let clips = rec("2026-05-09", [
      { at: "18:00", min: 10, node: "home", category: "meal" },
      { at: "18:15", min: 10, node: "home", category: "meal" },
      { at: "18:30", min: 10, node: "home", category: "meal" },
      { at: "18:45", min: 10, node: "home", category: "meal" },
      { at: "19:00", min: 10, node: "home", category: "meal" },
    ]);
    clips = text(clips, "18:00", "I look for a table in the cafeteria");
    clips = text(clips, "18:15", "I look around the cafeteria");
    clips = text(clips, "18:30", "I eat my meal at the cafeteria");
    clips = text(clips, "18:45", "I return the tray to the cafeteria");
    clips = text(clips, "19:00", "I chat in the cafeteria");
    expect(byBlock(world, day, clips, "2026-05-09")).toEqual({ "d1-s1": ["18:15", "18:30", "18:45"] });
  });

  it("trip_day breakfast follows lodging_morning directly, with no anchor", () => {
    const t = JSON.parse(readFileSync(new URL("../../../config/timeline/templates.v3.json", import.meta.url), "utf8"));
    const slots = t.day_types.trip_day as { role: string; anchor?: string }[];
    expect(slots.slice(0, 3).map((x) => x.role)).toEqual(["lodging_morning", "breakfast", "outing"]);
    expect(slots[1]!.anchor).toBeUndefined();
  });
});

describe("home fill from spare dates", () => {
  const world = makeWorld();
  /** Day `n` with ids d<n>-s<k>; `date` is its assigned real date. */
  const dayN = (n: number, slots: Slot[], trip?: ScheduleDay["trip"]): ScheduleDay => {
    const d = mkDay("home_day", slots.map((s) => ({ ...s, slot_id: s.slot_id.replace("d1-", `d${n}-`) })), [], trip);
    return { ...d, day_index: n };
  };
  const ASSIGNED = (rows: [number, string | null][]): DateAssignment[] =>
    rows.map(([day_index, date]) => ({ day_index, day_type: "home_day", pool: "rest", date }));
  const home = (at: string, min: number, category?: ClipCategory): Spec => ({ at, min, node: "home", ...(category ? { category } : {}) });
  /** An assigned date with no home footage, so every home slot stays empty after the primary plan. */
  const bare = (date: string) => rec(date, [{ at: "05:00", min: 10, node: "transit" }]);
  const MORNING = slot(1, "home_morning", "home", "07:00", "08:00");
  const EVENING = slot(2, "home_evening", "home", "20:30", "22:00");

  function fill(days: ScheduleDay[], assigned: [number, string | null][], clips: Clip[]) {
    const dates = collectDates(new MemCatalog(clips), world);
    const assignments = ASSIGNED(assigned);
    const r = homeFill(days, [], assignments, dates, world);
    return { ...r, slots: r.fills.map((f) => `${f.slot_id}<-${f.from_date}`) };
  }

  it("fills an empty home_morning from a spare date within 180 minutes, filling until the block end", () => {
    const clips = [...bare("2026-05-01"), ...rec("2026-05-02", [home("07:30", 30), home("08:05", 30), home("08:45", 30)])];
    const r = fill([dayN(1, [MORNING])], [[1, "2026-05-01"]], clips);
    expect(r.fills).toEqual([{ kind: "home", day_index: 1, slot_id: "d1-s1", role: "home_morning", from_date: "2026-05-02", delta_min: 30, seconds: 3600 }]);
    expect(r.entries.map((e) => e.clip_id.slice(-5))).toEqual(["07:30", "08:05"]);
    expect(r.entries.every((e) => e.slot_id === "d1-s1")).toBe(true);
  });

  it("does not use a candidate more than 180 minutes from the block start", () => {
    const edge = fill([dayN(1, [MORNING])], [[1, "2026-05-01"]], [...bare("2026-05-01"), ...rec("2026-05-02", [home("10:00", 30)])]);
    expect(edge.fills.map((f) => f.delta_min)).toEqual([180]);
    const out = fill([dayN(1, [MORNING])], [[1, "2026-05-01"]], [...bare("2026-05-01"), ...rec("2026-05-02", [home("10:01", 30)])]);
    expect(out.fills.map((f) => f.fallback)).toEqual([true]); // only the B6b last resort takes it
    expect(edge.fills[0]!.fallback).toBeUndefined();
  });

  it("never uses an assigned date as a spare", () => {
    const clips = [...bare("2026-05-01"), ...rec("2026-05-02", [home("07:00", 30)])];
    expect(fill([dayN(1, [MORNING])], [[1, "2026-05-01"], [2, "2026-05-02"]], clips).fills).toEqual([]);
  });

  it("skips slots that already got footage in the primary plan", () => {
    const dates = collectDates(new MemCatalog([...bare("2026-05-01"), ...rec("2026-05-02", [home("07:00", 30)])]), world);
    const primary = [{ slot_id: "d1-s1", clip_id: "x", source_start_s: 0, source_end_s: 60 }];
    expect(homeFill([dayN(1, [MORNING])], primary, ASSIGNED([[1, "2026-05-01"]]), dates, world).fills).toEqual([]);
  });

  it("serves one synthetic day per spare date, but morning and evening of that day alike", () => {
    const clips = [...bare("2026-05-01"), ...bare("2026-05-03"), ...rec("2026-05-02", [home("07:10", 30), home("20:40", 30)])];
    const r = fill([dayN(1, [MORNING, EVENING]), dayN(2, [MORNING, EVENING])], [[1, "2026-05-01"], [2, "2026-05-03"]], clips);
    expect(r.slots).toEqual(["d1-s1<-2026-05-02", "d1-s2<-2026-05-02"]);
  });

  it("fills home_morning slots of all days (B6, then B6b) before any other home slot", () => {
    // 05-02 is the only spare with morning home footage; day 1's evening would take it first in day order.
    const two: [number, string | null][] = [[1, "2026-05-01"], [2, "2026-05-03"]];
    const days = [dayN(1, [EVENING]), dayN(2, [MORNING])];
    const evening = rec("2026-05-04", [home("20:50", 30)]);
    const b6 = fill(days, two, [...bare("2026-05-01"), ...bare("2026-05-03"), ...rec("2026-05-02", [home("07:10", 30), home("20:40", 30)]), ...evening]);
    expect(b6.slots).toEqual(["d2-s1<-2026-05-02", "d1-s2<-2026-05-04"]);
    // the same when day 2's morning can only take 05-02 as the B6b last resort
    const b6b = fill(days, two, [...bare("2026-05-01"), ...bare("2026-05-03"), ...rec("2026-05-02", [home("11:40", 30), home("20:40", 30)]), ...evening]);
    expect(b6b.slots).toEqual(["d2-s1<-2026-05-02", "d1-s2<-2026-05-04"]);
    expect(b6b.fills.map((f) => f.kind)).toEqual(["home_fallback", "home"]);
    // a day's morning spare date is preferred for its later home slots
    const same = fill([dayN(1, [MORNING, EVENING])], [[1, "2026-05-01"]], [...bare("2026-05-01"), ...rec("2026-05-02", [home("11:40", 30), home("20:45", 30)]), ...rec("2026-05-03", [home("20:40", 30)])]);
    expect(same.slots).toEqual(["d1-s1<-2026-05-02", "d1-s2<-2026-05-02"]);
  });

  it("gives the spare date to the first day in index order and the next spare to the second day", () => {
    const clips = [...bare("2026-05-01"), ...bare("2026-05-03"), ...rec("2026-05-02", [home("07:00", 30)]), ...rec("2026-05-04", [home("07:20", 30)])];
    const r = fill([dayN(2, [MORNING]), dayN(1, [MORNING])], [[1, "2026-05-01"], [2, "2026-05-03"]], clips);
    expect(r.slots).toEqual(["d1-s1<-2026-05-02", "d2-s1<-2026-05-04"]);
  });

  it("prefers the smallest time difference, then more footage in the run, then the earlier date", () => {
    const base = bare("2026-05-01");
    const near = fill([dayN(1, [MORNING])], [[1, "2026-05-01"]], [...base, ...rec("2026-05-02", [home("07:10", 30)]), ...rec("2026-05-03", [home("07:20", 30)])]);
    expect(near.fills[0]!.from_date).toBe("2026-05-02");
    const longer = fill([dayN(1, [MORNING])], [[1, "2026-05-01"]], [...base, ...rec("2026-05-02", [home("07:10", 10)]), ...rec("2026-05-03", [home("07:10", 30)])]);
    expect(longer.fills[0]!.from_date).toBe("2026-05-03");
    const tie = fill([dayN(1, [MORNING])], [[1, "2026-05-01"]], [...base, ...rec("2026-05-04", [home("07:10", 30)]), ...rec("2026-05-03", [home("07:10", 30)])]);
    expect(tie.fills[0]!.from_date).toBe("2026-05-03");
  });

  it("fills a home meal slot only with home meal clips", () => {
    const dinner = slot(1, "dinner", "home", "18:30", "19:15");
    const clips = [...bare("2026-05-01"), ...rec("2026-05-02", [home("18:25", 5), home("18:50", 20, "meal")])];
    const r = fill([dayN(1, [dinner])], [[1, "2026-05-01"]], clips);
    expect(r.entries.map((e) => e.clip_id.slice(-5))).toEqual(["18:50"]);
    expect(r.fills[0]).toMatchObject({ role: "dinner", delta_min: 20, seconds: 1200 });
    const none = fill([dayN(1, [dinner])], [[1, "2026-05-01"]], [...bare("2026-05-01"), ...rec("2026-05-02", [home("18:30", 30)])]);
    expect(none.fills).toEqual([]);
  });

  it("leaves empty hobby and work slots empty", () => {
    const slots = [slot(1, "hobby", "hobby_1", "08:30", "09:30"), slot(2, "work_am", "workplace", "09:30", "12:00")];
    const clips = [...bare("2026-05-01"), ...rec("2026-05-02", [home("08:30", 30), home("09:30", 30), { at: "09:30", min: 30, node: "workplace" }])];
    expect(fill([dayN(1, slots)], [[1, "2026-05-01"]], clips).fills).toEqual([]);
  });

  it("leaves trip days untouched and never uses a trip date as a spare", () => {
    const tripDay = dayN(1, [MORNING], { node: "trip_1", index: 0 });
    const clips = [...tripDate("2026-06-29"), ...rec("2026-06-29", [home("07:00", 30)], { video: "extra" }), ...bare("2026-05-01")];
    expect(fill([tripDay, dayN(2, [MORNING])], [[1, "2026-06-29"], [2, "2026-05-01"]], clips).fills).toEqual([]);
    const spare = [...bare("2026-05-01"), ...rec("2026-05-02", [home("07:00", 30)])];
    expect(fill([tripDay], [[1, "2026-05-01"]], spare).fills).toEqual([]);
  });

  describe("B6b last-resort home_morning fill", () => {
    const away = (at: string): Spec => ({ at, min: 30, node: "workplace" });
    const A1 = [[1, "2026-05-01"]] as [number, string | null][];

    it("fills a home_morning with no in-window candidate from a spare date whose home clips precede its first away clip", () => {
      const clips = [...bare("2026-05-01"), ...rec("2026-05-02", [home("11:40", 30), away("12:30")])];
      const r = fill([dayN(1, [MORNING])], A1, clips);
      expect(r.fills).toEqual([{ kind: "home_fallback", day_index: 1, slot_id: "d1-s1", role: "home_morning", from_date: "2026-05-02", delta_min: 280, seconds: 1800, fallback: true }]);
      expect(r.entries.map((e) => e.clip_id.slice(-5))).toEqual(["11:40"]);
    });

    it("does not use home clips after the date's first away clip, but uses ones before it at any clock time", () => {
      const afterAway = [...bare("2026-05-01"), ...rec("2026-05-02", [away("11:00"), home("11:40", 30)])];
      expect(fill([dayN(1, [MORNING])], A1, afterAway).fills).toEqual([]);
      const afternoon = fill([dayN(1, [MORNING])], A1, [...bare("2026-05-01"), ...rec("2026-05-02", [home("13:00", 30), home("13:40", 20), away("15:00")])]);
      expect(afternoon.fills).toEqual([{ kind: "home_fallback", day_index: 1, slot_id: "d1-s1", role: "home_morning", from_date: "2026-05-02", delta_min: 360, seconds: 3000, fallback: true }]);
      expect(afternoon.entries.map((e) => e.clip_id.slice(-5))).toEqual(["13:00", "13:40"]);
      const evening = fill([dayN(1, [MORNING])], A1, [...bare("2026-05-01"), ...rec("2026-05-02", [home("19:00", 30)])]);
      expect(evening.fills.map((f) => f.fallback)).toEqual([true]);
    });

    it("never uses a date already taken by another day, nor fills non-morning slots", () => {
      const clips = [...bare("2026-05-01"), ...bare("2026-05-03"), ...rec("2026-05-02", [home("11:40", 30), away("12:30")])];
      const r = fill([dayN(1, [MORNING]), dayN(2, [MORNING])], [[1, "2026-05-01"], [2, "2026-05-03"]], clips);
      expect(r.slots).toEqual(["d1-s1<-2026-05-02"]);
      const evening = fill([dayN(1, [EVENING])], A1, [...bare("2026-05-01"), ...rec("2026-05-02", [home("11:40", 30)])]);
      expect(evening.fills).toEqual([]);
      // a date taken by a normal B6 fill of an earlier day is not reused by the fallback of a later day
      const mixed = [...bare("2026-05-01"), ...bare("2026-05-03"), ...rec("2026-05-02", [home("07:10", 30), away("12:30")])];
      const m = fill([dayN(1, [MORNING]), dayN(2, [MORNING])], [[1, "2026-05-01"], [2, "2026-05-03"]], mixed);
      expect(m.slots).toEqual(["d1-s1<-2026-05-02"]);
      expect(m.fills[0]!.fallback).toBeUndefined();
    });

    it("picks the date with the longest run, ties to the earlier date, and the run stops at gaps over 15 minutes", () => {
      const base = bare("2026-05-01");
      const longer = fill([dayN(1, [MORNING])], A1, [...base, ...rec("2026-05-02", [home("11:00", 20)]), ...rec("2026-05-03", [home("11:00", 20), home("11:25", 20)])]);
      expect(longer.fills[0]!.from_date).toBe("2026-05-03");
      const gap = fill([dayN(1, [MORNING])], A1, [...base, ...rec("2026-05-02", [home("11:00", 20)]), ...rec("2026-05-03", [home("11:00", 10), home("12:00", 15)])]);
      expect(gap.fills[0]!.from_date).toBe("2026-05-02");
      const tie = fill([dayN(1, [MORNING])], A1, [...base, ...rec("2026-05-04", [home("11:00", 20)]), ...rec("2026-05-03", [home("11:00", 20)])]);
      expect(tie.fills[0]!.from_date).toBe("2026-05-03");
    });
  });
});

describe("top-up fills from spare dates (B13)", () => {
  const world = makeWorld();
  const PM = slot(1, "work_pm", "workplace", "12:45", "17:30");
  const MORNING = slot(2, "home_morning", "home", "07:00", "08:00");
  const dayN = (n: number, slots: Slot[]): ScheduleDay => ({
    ...mkDay("workday", slots.map((s) => ({ ...s, slot_id: s.slot_id.replace("d1-", `d${n}-`) }))),
    day_index: n,
  });
  const work = (at: string, min: number, category?: ClipCategory): Spec => ({ at, min, node: "workplace", ...(category ? { category } : {}) });

  /** Primary plan from the assigned dates, then B6/B6b, then the top-up. */
  function topUp(days: ScheduleDay[], assigned: [number, string][], clips: Clip[]) {
    const dates = collectDates(new MemCatalog(clips), world);
    const assignments: DateAssignment[] = assigned.map(([day_index, date]) => ({ day_index, day_type: "workday", pool: "work", date }));
    const plan = days.flatMap((d) => planDay(d, dates.get(assigned.find(([i]) => i === d.day_index)![1]), world));
    const home = homeFill(days, plan, assignments, dates, world);
    const r = topUpFill(days, [...plan, ...home.entries], assignments, dates, world, home.fills);
    return { plan, home, ...r, slots: r.fills.map((f) => `${f.slot_id}<-${f.from_date}`) };
  }
  /** Day 1's assigned date: one hour of work from 13:00, so work_pm is unfilled from 13:45. */
  const own = rec("2026-05-04", [work("13:00", 60)]);

  it("tops up a half-filled work_pm from a spare date within ±3 h of the unfilled part's start, after the primary clips", () => {
    const r = topUp([dayN(1, [PM])], [[1, "2026-05-04"]], [...own, ...rec("2026-05-06", [work("16:45", 60), work("17:50", 30), work("18:40", 30)])]);
    expect(r.plan.map((e) => e.clip_id.slice(-5))).toEqual(["13:00"]);
    expect(r.fills).toEqual([{ kind: "topup", day_index: 1, slot_id: "d1-s1", role: "work_pm", from_date: "2026-05-06", delta_min: 180, seconds: 90 * 60 }]);
    // a contiguous run: 18:40 is more than 15 min after 18:20
    expect(r.entries.map((e) => `${e.slot_id}@${e.clip_id.slice(-5)}`)).toEqual(["d1-s1@16:45", "d1-s1@17:50"]);
    const before = topUp([dayN(1, [PM])], [[1, "2026-05-04"]], [...own, ...rec("2026-05-06", [work("10:45", 30)])]);
    expect(before.fills.map((f) => f.delta_min)).toEqual([180]);
  });

  it("does not top up from clips more than 3 h away, from workplace meals, or when less than 10 min is unfilled", () => {
    expect(topUp([dayN(1, [PM])], [[1, "2026-05-04"]], [...own, ...rec("2026-05-06", [work("16:46", 60)])]).fills).toEqual([]);
    expect(topUp([dayN(1, [PM])], [[1, "2026-05-04"]], [...own, ...rec("2026-05-06", [work("10:44", 60)])]).fills).toEqual([]);
    expect(topUp([dayN(1, [PM])], [[1, "2026-05-04"]], [...own, ...rec("2026-05-06", [work("14:00", 60, "meal")])]).fills).toEqual([]);
    const spare = rec("2026-05-06", [work("16:00", 60)]);
    // work_pm is 285 min long: 276 min of primary footage leaves 9 min, 275 min leaves 10.
    expect(topUp([dayN(1, [PM])], [[1, "2026-05-04"]], [...rec("2026-05-04", [work("13:00", 276)]), ...spare]).fills).toEqual([]);
    expect(topUp([dayN(1, [PM])], [[1, "2026-05-04"]], [...rec("2026-05-04", [work("13:00", 275)]), ...spare]).fills).toMatchObject([{ kind: "topup", seconds: 600 }]);
  });

  it("never lets a spare date serve two days, counting the dates B6 already gave a day", () => {
    const two: [number, string][] = [[1, "2026-05-04"], [2, "2026-05-05"]];
    const days = [dayN(1, [PM]), dayN(2, [PM])];
    const second = rec("2026-05-05", [work("13:00", 60)]);
    const one = topUp(days, two, [...own, ...second, ...rec("2026-05-06", [work("13:45", 60), work("15:00", 60)])]);
    expect(one.slots).toEqual(["d1-s1<-2026-05-06"]);
    const both = topUp(days, two, [...own, ...second, ...rec("2026-05-06", [work("13:45", 60)]), ...rec("2026-05-07", [work("13:45", 60)])]);
    expect(both.slots).toEqual(["d1-s1<-2026-05-06", "d2-s1<-2026-05-07"]);
    // B6 gives 05-06 to day 1's home_morning: day 2 cannot top up from it, day 1 can.
    const spare = rec("2026-05-06", [{ at: "07:00", min: 30, node: "home" }, work("13:45", 60)]);
    const b6 = topUp([dayN(1, [MORNING]), dayN(2, [PM])], two, [...own, ...second, ...spare]);
    expect(b6.home.fills.map((f) => `${f.slot_id}<-${f.from_date}`)).toEqual(["d1-s2<-2026-05-06"]);
    expect(b6.fills).toEqual([]);
    const sameDay = topUp([dayN(1, [MORNING, PM])], [[1, "2026-05-04"]], [...own, ...spare]);
    expect(sameDay.home.fills.map((f) => f.from_date)).toEqual(["2026-05-06"]);
    expect(sameDay.slots).toEqual(["d1-s1<-2026-05-06"]);
  });

  it("ranks top-up runs by whole minutes placed, then the day's own spare date, then the smaller |Δt|", () => {
    // 05-06 already serves day 1 (B6 morning); 05-05 places 20 s more and starts closer, but not a minute more.
    const ownSpare = rec("2026-05-06", [{ at: "07:00", min: 30, node: "home" }, work("14:15", 60)]);
    const closer = (min: number) => rec("2026-05-05", [work("13:45", min)]);
    const days = [dayN(1, [MORNING, PM])];
    expect(topUp(days, [[1, "2026-05-04"]], [...own, ...ownSpare, ...closer(60 + 1 / 3)]).slots).toEqual(["d1-s1<-2026-05-06"]);
    expect(topUp(days, [[1, "2026-05-04"]], [...own, ...ownSpare, ...closer(61)]).slots).toEqual(["d1-s1<-2026-05-05"]);
  });

  it("tops up a partly filled home slot with the home clips of a spare date", () => {
    const evening = slot(1, "home_evening", "home", "20:00", "22:00");
    const date = rec("2026-05-04", [work("09:00", 60), { at: "20:00", min: 30, node: "home" }]);
    const r = topUp([dayN(1, [evening])], [[1, "2026-05-04"]], [...date, ...rec("2026-05-06", [work("18:00", 60), { at: "20:40", min: 60, node: "home" }])]);
    expect(r.fills).toMatchObject([{ kind: "topup", role: "home_evening", from_date: "2026-05-06", delta_min: 10, seconds: 3600 }]);
    expect(r.entries.map((e) => e.clip_id.slice(-5))).toEqual(["20:40"]);
  });
});

describe("composeBlocks on a generated schedule", () => {
  const dataDir = makeTempDir();
  writeAtusPool(dataDir);
  const config = loadConfig(CONFIG_DIR, dataDir);

  function compose(clips: Clip[]) {
    const world = makeWorld();
    const t = createTimeline(world, generateSchedule(scheduleInput(config, world)), () => "compose-test-1");
    return { t, catalog: new MemCatalog(clips), result: composeBlocks(t, new MemCatalog(clips)) };
  }

  const WORK = Array.from({ length: 30 }, (_, i) => `2026-0${4 + Math.floor(i / 28)}-${String((i % 28) + 1).padStart(2, "0")}`);

  it("assigns every day a date of its pool, passes block-mode verify and reports per-day footage", () => {
    const { t, catalog, result } = compose([...WORK.flatMap((d) => workDate(d)), ...["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05", "2026-06-06", "2026-06-07", "2026-06-08", "2026-06-09", "2026-06-10"].flatMap(restDate), ...TRIP.flatMap(tripDate)]);
    const report = verifyTimeline(t, catalog, { mode: "block", placements: result.placements });
    expect(report.pass).toBe(true);
    expect(result.summary.unassigned_days).toEqual([]);
    const tripDays = t.schedule.days.filter((d) => d.trip !== undefined);
    expect(tripDays.map((d) => result.summary.per_day[d.day_index - 1]!.real_date)).toEqual(TRIP);
    for (const d of result.summary.per_day) {
      const day = t.schedule.days[d.day_index - 1]!;
      expect(d.day_type).toBe(day.day_type);
      expect(d.real_date).not.toBeNull();
      if (day.trip === undefined) expect(d.real_date!.startsWith(day.day_type.startsWith("workday") ? "2026-0" : "2026-06-")).toBe(true);
      expect(d.footage_hours).toBeGreaterThan(0);
      expect(d.empty_roles).not.toContain("home_morning");
    }
    const hobbyRoles = result.summary.per_day.flatMap((d) => d.empty_roles.filter((r) => r === "hobby"));
    expect(hobbyRoles.length).toBe(t.schedule.days.flatMap((d) => d.slots).filter((s) => s.role === "hobby").length);
    expect(result.placements.some((p) => /^d\d+-t\d+$/.test(p.slot_id) && p.clip.node === "transit")).toBe(true);
    const total = Object.values(result.summary.days_by_type).reduce((a, b) => a + b, 0);
    expect(total).toBe(28);
    const day1 = result.summary.per_day[0]!;
    const secs = result.placements.filter((p) => p.day_index === 1).reduce((s, p) => s + p.synthetic_end_s - p.synthetic_start_s, 0);
    expect(day1.footage_hours).toBeCloseTo(secs / 3600, 3);
    expect(day1.first_real).toMatch(/^\d{2}:\d{2}$/);
  });

  it("fills home blocks of a generated schedule from spare dates, passes block-mode verify and lists home_fills", () => {
    const spareRest = Array.from({ length: 20 }, (_, i) => `2026-06-${String(i + 1).padStart(2, "0")}`);
    const { t, catalog, result } = compose([...WORK.flatMap((d) => workDate(d, { homeMorning: false })), ...spareRest.flatMap(restDate), ...TRIP.flatMap(tripDate)]);
    const report = verifyTimeline(t, catalog, { mode: "block", placements: result.placements });
    expect(report.pass).toBe(true);
    const fills = result.summary.fills;
    expect(result.summary.home_fills).toBe(fills);
    expect(fills.filter((f) => f.kind === "home").length).toBeGreaterThan(0);
    const topups = fills.filter((f) => f.kind === "topup");
    expect(topups.length).toBeGreaterThan(0);
    const dateOfDay = new Map(result.assignments.map((a) => [a.day_index, a.date]));
    for (const f of topups.filter((x) => x.role === "work_am" || x.role === "work_pm")) {
      // the top-up follows the slot's primary clips back to back from the window start
      const ps = result.placements.filter((p) => p.slot_id === f.slot_id).sort((a, b) => a.synthetic_start_s - b.synthetic_start_s);
      const slotDef = t.schedule.days[f.day_index - 1]!.slots.find((x) => x.slot_id === f.slot_id)!;
      expect(ps[0]!.synthetic_start_s).toBe((f.day_index - 1) * 86400 + sec(slotDef.window[0]));
      for (let i = 1; i < ps.length; i++) expect(ps[i]!.synthetic_start_s).toBeCloseTo(ps[i - 1]!.synthetic_end_s, 3);
      const groups = ps.map((p) => p.clip.day_group);
      expect(groups.lastIndexOf(dateOfDay.get(f.day_index)!)).toBeLessThan(groups.indexOf(f.from_date));
    }
    const assigned = new Set(result.assignments.map((a) => a.date));
    const dayOf = new Map<string, number>();
    for (const f of fills) {
      expect(assigned.has(f.from_date)).toBe(false);
      expect(TRIP).not.toContain(f.from_date);
      expect(dayOf.get(f.from_date) ?? f.day_index).toBe(f.day_index);
      dayOf.set(f.from_date, f.day_index);
      expect(f.delta_min).toBeLessThanOrEqual(180);
      expect(result.placements.some((p) => p.slot_id === f.slot_id)).toBe(true);
      expect(t.schedule.days[f.day_index - 1]!.trip).toBeUndefined();
    }
  });

  it("reports days left without a date when a pool runs out, with all their blocks empty", () => {
    const { t, catalog, result } = compose([...WORK.slice(0, 5).flatMap((d) => workDate(d)), ...restDate("2026-06-01"), ...TRIP.flatMap(tripDate)]);
    const work = t.schedule.days.filter((d) => d.day_type === "workday" || d.day_type === "workday_late").length;
    const rest = t.schedule.days.filter((d) => d.trip === undefined && !d.day_type.startsWith("workday")).length;
    expect(result.summary.unassigned_days.filter((u) => u.pool === "work")).toHaveLength(work - 5);
    expect(result.summary.unassigned_days.filter((u) => u.pool === "rest")).toHaveLength(rest - 1);
    const u = result.summary.unassigned_days[0]!;
    const row = result.summary.per_day[u.day_index - 1]!;
    expect(row.real_date).toBeNull();
    expect(row.footage_hours).toBe(0);
    expect(row.empty_roles).toEqual(t.schedule.days[u.day_index - 1]!.slots.map((s) => s.role));
    expect(result.placements.some((p) => p.day_index === u.day_index)).toBe(false);
    expect(verifyTimeline(t, catalog, { mode: "block", placements: result.placements }).pass).toBe(true);
  });
});

const TSX = join(REPO_ROOT, "node_modules", "tsx", "dist", "cli.mjs");
const SCRIPT = join(REPO_ROOT, "packages", "mcp", "src", "composeBlocksCli.ts");

function run(args: string[]) {
  const r = spawnSync(process.execPath, [TSX, SCRIPT, ...args], { cwd: REPO_ROOT, encoding: "utf8", timeout: 60_000 });
  return { status: r.status, stdout: r.stdout, stderr: r.stderr };
}

describe("npm run compose-blocks", () => {
  it("writes the timeline, a passing block-mode report and the manifest to --out, and prints a summary", { timeout: 30_000 }, () => {
    const dataDir = makeTempDir();
    writeAtusPool(dataDir);
    const dates = Array.from({ length: 24 }, (_, i) => `2026-05-${String(i + 1).padStart(2, "0")}`);
    writeDataset(dataDir, "nextme800", [...dates.slice(0, 20).flatMap((d) => workDate(d)), ...dates.slice(20).flatMap(restDate), ...TRIP.flatMap(tripDate)]);
    const worldFile = join(makeTempDir(), "world.json");
    writeFileSync(worldFile, JSON.stringify(worldJson()), "utf8");
    const out = makeTempDir();

    const r = run(["--world", worldFile, "--data", dataDir, "--config", CONFIG_DIR, "--out", out]);
    expect(r.stderr).toBe("");
    expect(r.status).toBe(0);
    const s = JSON.parse(r.stdout) as {
      timeline_id: string;
      manifest: string;
      verifier_report: string;
      days_by_type: Record<string, number>;
      per_day: { day_index: number; real_date: string | null; footage_hours: number; empty_roles: string[] }[];
      unassigned_days: { day_index: number; pool: string }[];
      empty_blocks: number;
    };
    expect(s.per_day).toHaveLength(28);
    expect(s.empty_blocks).toBeGreaterThan(0);
    expect(filesIn(out, "timelines")).toEqual([`${s.timeline_id}.json`, `${s.timeline_id}.verifier_report.json`]);
    expect(filesIn(out, "worlds")).toEqual(["compose-test.json"]);
    expect(filesIn(dataDir, "timelines")).toEqual([]);
    expect(s.manifest).toBe(join(out, "exports", `${s.timeline_id}-r0.manifest.json`));
    const manifest = TimelineManifestSchema.parse(JSON.parse(readFileSync(s.manifest, "utf8")));
    expect(manifest.days).toHaveLength(28);
    const report = JSON.parse(readFileSync(s.verifier_report, "utf8")) as { pass: boolean; rules: Record<string, { note?: string }> };
    expect(report.pass).toBe(true);
    expect(report.rules["time_of_day"]!.note).toBeDefined();
  });

  it("exits 1 with an error JSON and writes nothing when a step fails", () => {
    const dataDir = makeTempDir(); // no ATUS pool
    const worldFile = join(makeTempDir(), "world.json");
    writeFileSync(worldFile, JSON.stringify(worldJson()), "utf8");
    const out = makeTempDir();
    const r = run(["--world", worldFile, "--data", dataDir, "--config", CONFIG_DIR, "--out", out]);
    expect(r.status).toBe(1);
    expect(r.stdout).toBe("");
    expect(JSON.parse(r.stderr).error.message).toContain("atus2025_stats_per_respondent.csv");
    expect(filesIn(out, "timelines")).toEqual([]);
    expect(filesIn(out, "worlds")).toEqual([]);
    expect(filesIn(out, "exports")).toEqual([]);
  });

  it("prints usage without --world", () => {
    const r = run([]);
    expect(r.status).toBe(1);
    expect(JSON.parse(r.stderr).error.message).toContain("usage: npm run compose-blocks");
  });
});

describe("evening activities (B10)", () => {
  const EVENING = [
    { node: "gym_1", role: "exercise", position: "before_dinner", duration_min: 60, weekdays: ["Tue", "Thu"] },
    { node: "hobby_2", role: "hobby_2", position: "after_dinner", duration_min: 45, weekdays: ["Mon", "Wed"] },
  ];
  function eveningWorld(seed?: number): World {
    const base = worldJson({ trip: false, ...(seed !== undefined ? { seed } : {}) });
    base.nodes.gym_1 = { kind: "hobby", activity: "gym workout", setting: "venue_indoor", weekly_freq: 0, tags: ["gym"] };
    base.nodes.hobby_2 = { kind: "hobby", activity: "guitar", setting: "venue_indoor", weekly_freq: 0, tags: ["music_guitar"] };
    base.node_binding.gym_1 = { dataset: ["nextme800"], person_source: PERSON, node: "gym" };
    base.node_binding.hobby_2 = { dataset: ["egoexo4d"], tags_any: ["music_guitar"] };
    base.dials.evening_activities = EVENING;
    return parseWorld(base);
  }
  const world = eveningWorld();
  const hobbySlot = (i: number, role: string, node: string, from: string, to: string): Slot => ({ ...slot(i, role, node, from, to), spec: { ...slot(i, role, node, from, to).spec, category: "hobby" } });

  /** A work date with a gym run (transit, then gym clips) after work. */
  const gymDate = (date: string, gap = 5): Clip[] => [
    ...workDate(date),
    ...rec(date, [
      { at: "16:50", min: 10, node: "transit" },
      { at: "17:05", min: 30, node: "gym", category: "hobby" },
      { at: `17:${35 + gap}`, min: 20, node: "gym", category: "hobby" },
      { at: "19:00", min: 20, node: "gym", category: "hobby" }, // a second, shorter run
    ], { video: `rec-${date}-gym`, wall: "16:00" }),
  ];

  const guitar = (take: string, group: string, n = 2, min = 20, activity = "guitar"): Clip[] =>
    Array.from({ length: n }, (_, i): Clip => ({
      clip_id: `egoexo4d:${take}:${i}`,
      category: "hobby",
      setting: "venue_indoor",
      node: "hobby",
      activity,
      time_of_day: "unknown",
      tags: ["music_guitar"],
      source: { dataset: "egoexo4d", video_id: take, media: { kind: "video", fps: 30, path: `${take}.mp4` } },
      source_start_s: i * min * 60,
      source_end_s: (i + 1) * min * 60,
      day_group: group,
      person_source: `egoexo4d:${take}`,
      fingerprint: null,
      extractor_version: "fixture.v1",
    }));

  it("gym dates: an exercise day prefers the work date with gym footage", () => {
    const dates = collectDates(new MemCatalog([...workDate("2026-05-04"), ...gymDate("2026-05-05"), ...workDate("2026-05-06")]), world);
    expect(dates.get("2026-05-05")!.kind).toBe("work");
    const ex = mkDay("workday", [hobbySlot(1, "exercise", "gym_1", "17:30", "18:30")]);
    const plain = mkDay("workday", [slot(1, "work_am", "workplace", "08:30", "12:00")]);
    for (const seed of [1, 2, 3, 4]) {
      const w = eveningWorld(seed);
      const out = assignDates([{ ...plain, day_index: 1 }, { ...ex, day_index: 2 }], dates, w);
      expect(out.find((a) => a.day_index === 2)!.date).toBe("2026-05-05");
    }
  });

  it("planDay: the exercise block takes the date's longest gym run and the transit just before it", () => {
    const day = mkDay("workday", [slot(1, "work_pm", "workplace", "12:45", "17:00"), hobbySlot(2, "exercise", "gym_1", "17:30", "19:30")], [[1, 2, 40]]);
    // the transit goes to the transition block into the exercise slot, the gym clips to the slot
    expect(byBlock(world, day, gymDate("2026-05-05"), "2026-05-05")).toMatchObject({ "d1-s2": ["17:05", "17:40"], "d1-t1": ["16:50"] });
    // a gap over 15 minutes splits the run: the first run (30 min) is then the longest
    const split = byBlock(world, day, gymDate("2026-05-05", 25), "2026-05-05");
    expect(split["d1-s2"]).toEqual(["17:05"]);
  });

  it("planDay: an exercise day keeps its commute home, after the gym run, in the transition into the home slot", () => {
    const clips = rec("2026-05-05", [
      { at: "09:00", min: 480, node: "workplace" },
      { at: "17:50", min: 8, node: "transit" },
      { at: "18:00", min: 70, node: "gym", category: "hobby" },
      { at: "19:20", min: 20, node: "transit" },
      { at: "19:45", min: 60, node: "home" },
    ], { wall: "09:00" });
    const day = mkDay("workday", [
      slot(1, "work_pm", "workplace", "12:45", "17:00"),
      hobbySlot(2, "exercise", "gym_1", "17:30", "19:00"),
      slot(3, "home_evening", "home", "19:30", "21:00"),
    ], [[1, 2, 20], [2, 3, 20]]);
    const got = byBlock(world, day, clips, "2026-05-05");
    expect(got["d1-t1"]).toEqual(["17:50"]);
    expect(got["d1-s2"]).toEqual(["18:00"]);
    expect(got["d1-t2"]).toEqual(["19:20"]);
  });

  it("planDay: no gym clips leaves the exercise block empty (no cross-date fill)", () => {
    const day = mkDay("workday", [slot(1, "work_pm", "workplace", "12:45", "17:00"), hobbySlot(2, "exercise", "gym_1", "17:30", "18:30")], [[1, 2, 40]]);
    const got = byBlock(world, day, workDate("2026-05-04"), "2026-05-04");
    expect(got["d1-s2"]).toBeUndefined();
    expect(got["d1-t1"]).toBeUndefined();
  });

  it("eveningFill: hobby_2 takes egoexo takes, each once, and never reuses a day_group on another day", () => {
    const clips = [...guitar("tA", "gA", 2), ...guitar("tB", "gA", 2), ...guitar("tC", "gC", 2), ...guitar("tD", "gD", 1, 60)];
    const mk = (n: number): ScheduleDay => ({ ...mkDay("workday", [hobbySlot(1, "hobby_2", "hobby_2", "19:30", "20:15")]), day_index: n, slots: [{ ...hobbySlot(1, "hobby_2", "hobby_2", "19:30", "20:15"), slot_id: `d${n}-s1` }] });
    const days = [mk(1), mk(2), mk(3)];
    const r = eveningFill(days, [], new MemCatalog(clips), world);
    expect(r.takes.map((x) => x.take)).toHaveLength(new Set(r.takes.map((x) => x.take)).size);
    const groupDays = new Map<string, Set<number>>();
    for (const x of r.takes) groupDays.set(x.day_group, (groupDays.get(x.day_group) ?? new Set()).add(x.day_index));
    for (const d of groupDays.values()) expect(d.size).toBe(1);
    expect(new Set(r.entries.map((e) => e.clip_id)).size).toBe(r.entries.length);
    expect(r.entries.length).toBeGreaterThan(0);
    // deterministic
    expect(eveningFill(days, [], new MemCatalog(clips), world)).toEqual(r);
  });

  it("planDay: the exercise slot is found by its binding, so a role named workout works", () => {
    const w = parseWorld({
      ...eveningWorld(),
      dials: { ...eveningWorld().dials, evening_activities: EVENING.map((a) => (a.role === "exercise" ? { ...a, role: "workout" } : a)) },
    } as never);
    const day = mkDay("workday", [slot(1, "work_pm", "workplace", "12:45", "17:00"), hobbySlot(2, "workout", "gym_1", "17:30", "19:30")], [[1, 2, 40]]);
    expect(byBlock(w, day, gymDate("2026-05-05"), "2026-05-05")).toMatchObject({ "d1-s2": ["17:05", "17:40"], "d1-t1": ["16:50"] });
  });

  it("eveningFill: each non-wearer node shuffles with its own key, so equal take lists get different orders", () => {
    const base = worldJson({ trip: false });
    base.nodes.hobby_2 = { kind: "hobby", activity: "guitar", setting: "venue_indoor", weekly_freq: 0, tags: ["music_guitar"] };
    base.nodes.hobby_3 = { kind: "hobby", activity: "piano", setting: "venue_indoor", weekly_freq: 0, tags: ["music_piano"] };
    base.node_binding.hobby_2 = { dataset: ["egoexo4d"], tags_any: ["music_guitar"] };
    base.node_binding.hobby_3 = { dataset: ["egoexo4d"], tags_any: ["music_piano"] };
    base.dials.evening_activities = [
      { node: "hobby_2", role: "hobby_2", position: "after_dinner", duration_min: 45, weekdays: ["Mon"] },
      { node: "hobby_3", role: "hobby_3", position: "after_dinner", duration_min: 45, weekdays: ["Tue"] },
    ];
    const w = parseWorld(base);
    const names = Array.from({ length: 8 }, (_, i) => `tk${i}`);
    const clips = names.flatMap((n) => [
      ...guitar(n, `g-${n}`, 1, 30),
      ...guitar(`${n}p`, `p-${n}`, 1, 30).map((c): Clip => ({ ...c, tags: ["music_piano"] })),
    ]);
    const mk = (n: number, role: string): ScheduleDay => {
      const s = { ...hobbySlot(1, role, role, "19:30", "20:15"), slot_id: `d${n}-s1` };
      return { ...mkDay("workday", [s]), day_index: n, slots: [s] };
    };
    const days = [...names.map((_, i) => mk(i + 1, "hobby_2")), ...names.map((_, i) => mk(i + 9, "hobby_3"))];
    const r = eveningFill(days, [], new MemCatalog(clips), w);
    const order = (role: string) => r.takes.filter((x) => x.role === role).sort((a, b) => a.day_index - b.day_index).map((x) => x.take.replace(/p$/, ""));
    expect(order("hobby_2")).toHaveLength(8);
    expect(order("hobby_3")).toHaveLength(8);
    expect(order("hobby_2")).not.toEqual(order("hobby_3"));
  });

  it("B12: an exercise day takes a gym date for its 10 h bonus over a date that fills more of its other blocks", () => {
    const sparseGym = [...rec("2026-05-05", [{ at: "09:00", min: 60, node: "workplace" }]), ...gymDate("2026-05-05").filter((c) => c.clip_id.includes("-gym:"))];
    const dates = collectDates(new MemCatalog([...workDate("2026-05-04", { extraWork: 120 }), ...sparseGym]), world);
    expect(dates.get("2026-05-05")!.work_seconds).toBe(3600);
    const ex = mkDay("workday", [
      slot(1, "work_am", "workplace", "08:30", "12:00"),
      slot(2, "work_pm", "workplace", "12:45", "17:00"),
      hobbySlot(3, "exercise", "gym_1", "17:30", "18:30"),
    ]);
    for (const seed of [1, 2, 3]) expect(assignDates([ex], dates, eveningWorld(seed))[0]!.date).toBe("2026-05-05");
  });

  /** Ego-Exo4D-style guitar takes: one 4-min clip each, the take name (participant prefix) in the activity. */
  const takesOf = (participants: number, perParticipant: number): Clip[] =>
    Array.from({ length: participants * perParticipant }, (_, i) => {
      const p = Math.floor(i / perParticipant);
      const name = `upenn_03${String(p).padStart(2, "0")}_Guitar_1_${i % perParticipant}`;
      return guitar(`take${String(i).padStart(2, "0")}`, `egoexo4d:take${i}`, 1, 4, `Playing Guitar (${name})`);
    }).flat();
  const hobbyDays = (n: number): ScheduleDay[] =>
    Array.from({ length: n }, (_, k) => {
      const s = { ...hobbySlot(1, "hobby_2", "hobby_2", "19:30", "20:15"), slot_id: `d${k + 1}-s1` };
      return { ...mkDay("workday", [s]), day_index: k + 1, slots: [s] };
    });

  it("B14: spreads hobby_2 takes round-robin over the days, each take once, so every day gets one when takes >= days", () => {
    const clips = takesOf(7, 4); // 28 takes of 4 min for 6 days of 45 min
    for (const seed of [1, 2, 3]) {
      const r = eveningFill(hobbyDays(6), [], new MemCatalog(clips), eveningWorld(seed));
      const per = [1, 2, 3, 4, 5, 6].map((d) => r.takes.filter((x) => x.day_index === d).length);
      expect(Math.min(...per)).toBeGreaterThanOrEqual(1);
      expect(Math.max(...per) - Math.min(...per)).toBeLessThanOrEqual(1);
      expect(r.takes).toHaveLength(28);
      expect(new Set(r.takes.map((x) => x.take)).size).toBe(28);
    }
    // fewer takes than days: the first days get one each
    const few = eveningFill(hobbyDays(6), [], new MemCatalog(takesOf(1, 4)), world);
    expect(few.takes.map((x) => x.day_index)).toEqual([1, 2, 3, 4]);
  });

  it("B14: within a day, prefers takes of the same participant", () => {
    const clips = takesOf(3, 4); // 3 participants x 4 takes for 3 days
    for (const seed of [1, 2, 3, 4]) {
      const r = eveningFill(hobbyDays(3), [], new MemCatalog(clips), eveningWorld(seed));
      const byTake = new Map(clips.map((c) => [c.source.video_id, c.activity.slice(16, 26)]));
      for (const d of [1, 2, 3]) {
        const people = r.takes.filter((x) => x.day_index === d).map((x) => byTake.get(x.take));
        expect(people).toHaveLength(4);
        expect(new Set(people).size).toBe(1);
      }
    }
  });

  it("composeBlocks: exercise and hobby_2 fill, the summary lists them and block verify passes", () => {
    const dataDir = makeTempDir();
    writeAtusPool(dataDir);
    const config = loadConfig(CONFIG_DIR, dataDir);
    const w = eveningWorld();
    const t = createTimeline(w, generateSchedule(scheduleInput(config, w)), () => "compose-test-evening");
    const work = Array.from({ length: 30 }, (_, i) => `2026-04-${String(i + 1).padStart(2, "0")}`);
    const rest = Array.from({ length: 10 }, (_, i) => `2026-06-${String(i + 1).padStart(2, "0")}`);
    const takes = Array.from({ length: 12 }, (_, i) => guitar(`tk${i}`, `grp${i % 6}`, 2, 25)).flat();
    const clips = [...work.flatMap((d, i) => (i % 3 === 0 ? gymDate(d) : workDate(d))), ...rest.flatMap(restDate), ...takes];
    const catalog = new MemCatalog(clips);
    const result = composeBlocks(t, catalog);
    const report = verifyTimeline(t, catalog, { mode: "block", placements: result.placements });
    expect(report.pass, JSON.stringify(report).slice(0, 1500)).toBe(true);
    const evening = result.summary.per_day.flatMap((d) => d.evening);
    const exercise = evening.filter((e) => e.role === "exercise");
    const hobby2 = evening.filter((e) => e.role === "hobby_2");
    expect(t.schedule.days.flatMap((d) => d.slots).some((s) => s.role === "exercise")).toBe(true);
    expect(exercise.length).toBeGreaterThan(0);
    expect(exercise.every((e) => /^2026-04-/.test(e.source_date_or_take))).toBe(true);
    expect(hobby2.length).toBeGreaterThan(0);
    expect(hobby2.every((e) => /^tk\d+$/.test(e.source_date_or_take) && e.seconds > 0)).toBe(true);
    const used = result.placements.filter((p) => p.clip.source.dataset === "egoexo4d").map((p) => p.clip.clip_id);
    expect(new Set(used).size).toBe(used.length);
    // exercise blocks come from their day's assigned date
    const dateOf = new Map(result.assignments.map((a) => [a.day_index, a.date]));
    for (const d of result.summary.per_day) for (const e of d.evening.filter((x) => x.role === "exercise")) expect(e.source_date_or_take).toBe(dateOf.get(d.day_index));
  });

  describe("hobby_1 (B15)", () => {
    function hobbyWorld(bound: boolean): World {
      const base = worldJson({ trip: false });
      base.nodes.hobby_1 = { kind: "hobby", activity: "lego building", setting: "home_indoor", weekly_freq: 1, tags: ["lego"] };
      if (bound) base.node_binding.hobby_1 = { dataset: ["youtube"], tags_any: ["lego"] };
      return parseWorld(base);
    }
    const lego = (i: number, min: number): Clip => ({
      clip_id: `youtube:lego:${i}`,
      category: "hobby",
      setting: "home_indoor",
      node: "hobby",
      activity: "lego building",
      time_of_day: "unknown",
      tags: ["lego"],
      source: { dataset: "youtube", video_id: "lego", media: { kind: "video", fps: 30, path: "lego.mp4" } },
      source_start_s: i * 200 * 60,
      source_end_s: i * 200 * 60 + min * 60,
      day_group: `youtube:lego:${i}`,
      person_source: "youtube:lego",
      fingerprint: null,
      extractor_version: "fixture.v1",
    });
    const hobbyDay = (n: number): ScheduleDay => {
      const s = { ...hobbySlot(1, "hobby", "hobby_1", "10:00", "12:00"), slot_id: `d${n}-s1` };
      return { ...mkDay("hobby_day", [s]), day_index: n, slots: [s] };
    };

    it("gives each of 3 hobby days one clip, trims a 132-min clip to the 120-min window, never reuses a clip", () => {
      const w = hobbyWorld(true);
      const clips = [lego(0, 60), lego(1, 132), lego(2, 45)];
      const days = [1, 2, 3].map(hobbyDay);
      const r = eveningFill(days, [], new MemCatalog(clips), w);
      expect(r.takes.map((x) => x.day_index).sort()).toEqual([1, 2, 3]);
      expect(new Set(r.entries.map((e) => e.clip_id)).size).toBe(3);
      expect(r.takes.every((x) => x.role === "hobby")).toBe(true);
      expect(Math.max(...r.takes.map((x) => x.seconds))).toBe(7200);
      expect(eveningFill(days, [], new MemCatalog(clips), w)).toEqual(r);
      // two clips, three days: no reuse
      expect(eveningFill(days, [], new MemCatalog(clips.slice(0, 2)), w).entries).toHaveLength(2);
    });

    it("deals the parts of one video to the hobby days in source order; several videos stay deterministic", () => {
      const days = [1, 2, 3].map(hobbyDay);
      for (const seed of [1, 2, 3]) {
        const w = parseWorld({ ...hobbyWorld(true), seed } as never);
        const r = eveningFill(days, [], new MemCatalog([lego(2, 45), lego(0, 60), lego(1, 132)]), w);
        expect(r.entries.map((e) => e.clip_id)).toEqual(["youtube:lego:0", "youtube:lego:1", "youtube:lego:2"]);
      }
      const other = (i: number): Clip => ({ ...lego(i, 30), clip_id: `youtube:other:${i}`, day_group: `youtube:other:${i}`, source: { ...lego(i, 30).source, video_id: "other" } });
      const two = new MemCatalog([lego(1, 60), other(1), lego(0, 60), other(0)]);
      const r2 = eveningFill([1, 2, 3, 4].map(hobbyDay), [], two, hobbyWorld(true));
      expect(eveningFill([1, 2, 3, 4].map(hobbyDay), [], two, hobbyWorld(true))).toEqual(r2);
      const ids = r2.entries.map((e) => e.clip_id);
      expect(ids.indexOf("youtube:lego:0")).toBeLessThan(ids.indexOf("youtube:lego:1"));
      expect(ids.indexOf("youtube:other:0")).toBeLessThan(ids.indexOf("youtube:other:1"));
    });

    it("leaves hobby slots empty without bound clips", () => {
      const days = [1, 2, 3].map(hobbyDay);
      expect(eveningFill(days, [], new MemCatalog([lego(0, 60)]), hobbyWorld(false)).entries).toEqual([]);
      expect(eveningFill(days, [], new MemCatalog([]), hobbyWorld(true)).entries).toEqual([]);
    });

    it("composeBlocks: fills hobby days from youtube clips, lists them in activities and block verify passes", () => {
      const dataDir = makeTempDir();
      writeAtusPool(dataDir);
      const config = loadConfig(CONFIG_DIR, dataDir);
      const w = hobbyWorld(true);
      const t = createTimeline(w, generateSchedule(scheduleInput(config, w)), () => "compose-test-hobby");
      const work = Array.from({ length: 30 }, (_, i) => `2026-04-${String(i + 1).padStart(2, "0")}`);
      const rest = Array.from({ length: 10 }, (_, i) => `2026-06-${String(i + 1).padStart(2, "0")}`);
      const catalog = new MemCatalog([...work.flatMap((d) => workDate(d)), ...rest.flatMap((d) => restDate(d)), lego(0, 60), lego(1, 132), lego(2, 45)]);
      const result = composeBlocks(t, catalog);
      const report = verifyTimeline(t, catalog, { mode: "block", placements: result.placements });
      expect(report.pass, JSON.stringify(report).slice(0, 1500)).toBe(true);
      const hobbySlots = t.schedule.days.flatMap((d) => d.slots).filter((s) => s.role === "hobby").length;
      const acts = result.summary.per_day.flatMap((d) => d.activities).filter((a) => a.role === "hobby");
      expect(acts.length).toBe(Math.min(3, hobbySlots));
      expect(result.summary.per_day.every((d) => d.evening === d.activities)).toBe(true);
      const used = result.placements.filter((p) => p.clip.source.dataset === "youtube").map((p) => p.clip.clip_id);
      expect(new Set(used).size).toBe(used.length);
      // youtube clips have no wall clock: no summary field may be NaN, and a day with only such clips has no real time
      expect(JSON.stringify(result.summary)).not.toContain("NaN");
      const hobbyDays = result.summary.per_day.filter((d) => d.activities.some((a) => a.role === "hobby"));
      expect(hobbyDays.length).toBeGreaterThan(0);
      for (const d of result.summary.per_day) {
        expect(d.first_real === null || /^\d\d:\d\d/.test(d.first_real)).toBe(true);
        expect(d.last_real === null || /^\d\d:\d\d/.test(d.last_real)).toBe(true);
      }
    });

    describe("fit_roles (B17)", () => {
      const MINUTES = [133, 106, 171];
      const setup = (parts = MINUTES.map((m, i) => lego(i, m))) => {
        const dataDir = makeTempDir();
        writeAtusPool(dataDir);
        const config = loadConfig(CONFIG_DIR, dataDir);
        const w = hobbyWorld(true);
        const build = (x: World) => createTimeline(x, generateSchedule(scheduleInput(config, x)), () => "compose-test-fit");
        const work = Array.from({ length: 30 }, (_, i) => `2026-04-${String(i + 1).padStart(2, "0")}`);
        const rest = Array.from({ length: 10 }, (_, i) => `2026-06-${String(i + 1).padStart(2, "0")}`);
        const catalog = new MemCatalog([...work.flatMap((d) => workDate(d)), ...rest.flatMap((d) => restDate(d)), ...parts]);
        return { w, build, catalog };
      };
      const hobbyIds = (t: ReturnType<typeof createTimeline>) => t.schedule.days.flatMap((d) => d.slots.filter((s) => s.role === "hobby").map((s) => [d.day_index, s.slot_id]));

      it("sizes the hobby slots to ceil(minutes) of their clip, untrimmed, writes the overrides and verifies", () => {
        const { w, build, catalog } = setup();
        const t0 = build(w);
        const r = composeBlocksFit(t0, catalog, ["hobby"], build);
        expect(r.slot_overrides).toHaveLength(3);
        expect(r.slot_overrides.map((o) => o.duration_min)).toEqual([133, 106, 171]);
        expect(r.timeline.world.dials.slot_overrides).toEqual(r.slot_overrides);
        const hobby = r.timeline.schedule.days.flatMap((d) => d.slots.filter((s) => s.role === "hobby").map((s) => ({ d, s })));
        for (const [i, o] of r.slot_overrides.entries()) {
          const slot = hobby.find((h) => h.d.day_index === o.day_index)!.s;
          expect(parseClock(slot.window[1]) - parseClock(slot.window[0])).toBe(o.duration_min);
          const ps = r.placements.filter((p) => p.slot_id === slot.slot_id);
          expect(ps.map((p) => p.clip.clip_id)).toEqual([`youtube:lego:${i}`]);
          expect(ps[0]!.source_start_s).toBe(ps[0]!.clip.source_start_s);
          expect(ps[0]!.source_end_s).toBe(ps[0]!.clip.source_end_s);
        }
        const report = verifyTimeline(r.timeline, catalog, { mode: "block", placements: r.placements });
        expect(report.pass, JSON.stringify(report).slice(0, 1500)).toBe(true);
        // the hobby days (and so the assignment) are the same before and after regenerating
        expect(hobbyIds(r.timeline)).toEqual(hobbyIds(t0));
        expect(composeBlocksFit(t0, catalog, ["hobby"], build)).toEqual(r);
      });

      it("rounds a part up to whole minutes", () => {
        const odd = { ...lego(1, 90), source_end_s: lego(1, 90).source_start_s + 90 * 60 + 20 };
        const { w, build, catalog } = setup([lego(0, 60), odd, lego(2, 45)]);
        const r = composeBlocksFit(build(w), catalog, ["hobby"], build);
        expect(r.slot_overrides.map((o) => o.duration_min)).toEqual([60, 91, 45]);
      });

      it("without fit_roles nothing changes", () => {
        const { w, build, catalog } = setup();
        const t0 = build(w);
        const plain = composeBlocks(t0, catalog);
        const r = composeBlocksFit(t0, catalog, [], build);
        expect(r.slot_overrides).toEqual([]);
        expect(r.timeline).toBe(t0);
        expect(r.placements).toEqual(plain.placements);
        expect(r.summary).toEqual(plain.summary);
      });

      it("CLI --fit-roles: the saved world carries the overrides, the summary reports them, a rerun is accepted", { timeout: 60_000 }, () => {
        const dataDir = makeTempDir();
        writeAtusPool(dataDir);
        const work = Array.from({ length: 30 }, (_, i) => `2026-04-${String(i + 1).padStart(2, "0")}`);
        const rest = Array.from({ length: 10 }, (_, i) => `2026-06-${String(i + 1).padStart(2, "0")}`);
        writeDataset(dataDir, "nextme800", [...work.flatMap((d) => workDate(d)), ...rest.flatMap((d) => restDate(d))]);
        writeDataset(dataDir, "youtube", MINUTES.map((m, i) => lego(i, m)));
        const worldFile = join(makeTempDir(), "world.json");
        writeFileSync(worldFile, JSON.stringify(hobbyWorld(true)), "utf8");
        const out = makeTempDir();
        const args = ["--world", worldFile, "--data", dataDir, "--config", CONFIG_DIR, "--out", out, "--fit-roles", "hobby"];
        const r = run(args);
        expect(r.stderr).toBe("");
        const s = JSON.parse(r.stdout) as { slot_overrides: { duration_min: number }[] };
        expect(s.slot_overrides.map((o) => o.duration_min)).toEqual([133, 106, 171]);
        const saved = JSON.parse(readFileSync(join(out, "worlds", "compose-test.json"), "utf8"));
        expect(saved.dials.slot_overrides).toEqual(s.slot_overrides);
        expect(run(args).status).toBe(0);
      });

      it("names the day, role and length when core cannot hold an override", () => {
        const { w, build, catalog } = setup();
        expect(() =>
          composeBlocksFit(build(w), catalog, ["hobby"], () => {
            throw new Error("SCHEDULE_INFEASIBLE");
          }),
        ).toThrow(/day \d+ hobby 133 min/);
      });

      it("keeps the TimelineError code and details of a core failure", () => {
        const { w, build, catalog } = setup();
        let caught: unknown;
        try {
          composeBlocksFit(build(w), catalog, ["hobby"], () => {
            throw new TimelineError("SCHEDULE_INFEASIBLE", "no room", { day_index: 4 });
          });
        } catch (e) {
          caught = e;
        }
        expect(caught).toBeInstanceOf(TimelineError);
        const e = caught as TimelineError;
        expect(e.code).toBe("SCHEDULE_INFEASIBLE");
        expect(e.details.day_index).toBe(4);
        expect(e.message).toMatch(/day \d+ hobby 133 min.*no room/);
      });

      it("rejects a role that cannot be fitted", () => {
        const { w, build, catalog } = setup();
        expect(() => composeBlocksFit(build(w), catalog, ["exercise"], build)).toThrow(/unsupported role "exercise"/);
      });

      it("CLI --fit-roles: an unsupported role fails; a hobby role without bound clips warns on stderr, stdout stays JSON", { timeout: 120_000 }, () => {
        const dataDir = makeTempDir();
        writeAtusPool(dataDir);
        const work = Array.from({ length: 30 }, (_, i) => `2026-04-${String(i + 1).padStart(2, "0")}`);
        const rest = Array.from({ length: 10 }, (_, i) => `2026-06-${String(i + 1).padStart(2, "0")}`);
        writeDataset(dataDir, "nextme800", [...work.flatMap((d) => workDate(d)), ...rest.flatMap((d) => restDate(d))]);
        writeDataset(dataDir, "youtube", []);
        const worldFile = join(makeTempDir(), "world.json");
        writeFileSync(worldFile, JSON.stringify(hobbyWorld(true)), "utf8");
        const base = ["--world", worldFile, "--data", dataDir, "--config", CONFIG_DIR];
        const bad = run([...base, "--out", makeTempDir(), "--fit-roles", "exercise"]);
        expect(bad.status).toBe(1);
        expect(bad.stderr).toMatch(/unsupported role/);
        const none = run([...base, "--out", makeTempDir(), "--fit-roles", "hobby"]);
        expect(none.status, none.stderr).toBe(0);
        expect(none.stderr).toMatch(/warning: --fit-roles hobby: no clips/);
        expect(JSON.parse(none.stdout).slot_overrides).toEqual([]);
      });
    });
  });
});
