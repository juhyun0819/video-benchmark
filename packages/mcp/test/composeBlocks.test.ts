import { spawnSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  TimelineManifestSchema,
  createTimeline,
  generateSchedule,
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
import { assignDates, collectDates, composeBlocks, deOverlap, homeFill, planDay, realStartOf, type DayLike, type DateAssignment } from "../src/composeBlocks.js";
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
    return {
      clip_id: `nextme800:${video}:${s.at}`,
      category: s.category ?? DEFAULT_CATEGORY[s.node] ?? "home",
      setting: SETTING[s.node] ?? "venue_indoor",
      node: s.node,
      activity: `synthetic ${s.node}`,
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

  it("falls back to the other side's date closest to the preferred first-work time", () => {
    // The 10:00 date ranks lower than the 08:40 one but is the closer fallback for a late day.
    const early = collectDates(
      new MemCatalog([...workDate("2026-05-04"), ...rec("2026-05-06", [{ at: "10:00", min: 120, node: "workplace" }])]),
      world,
    );
    expect(assignDates([day(1, "workday_late")], early, world)[0]!.date).toBe("2026-05-06");
    // The 15:00 date ranks higher than the 12:40 one but a plain workday takes the earliest first work.
    const late = collectDates(
      new MemCatalog([
        ...workDate("2026-05-05", { late: true }),
        ...rec("2026-05-07", [{ at: "07:00", min: 300, node: "home" }, { at: "15:00", min: 120, node: "workplace" }]),
      ]),
      world,
    );
    expect(assignDates([day(1, "workday")], late, world)[0]!.date).toBe("2026-05-05");
  });

  it("ranks dates by home footage before leaving, then by labeled minutes", () => {
    const dates = collectDates(
      new MemCatalog([
        ...workDate("2026-05-04"),
        ...workDate("2026-05-06", { homeMorning: false, extraWork: 600 }),
        ...workDate("2026-05-07", { extraWork: 60 }),
      ]),
      world,
    );
    expect(assignDates([day(1, "workday")], dates, world)[0]!.date).toBe("2026-05-07");
    const two = assignDates([day(1, "workday"), day(2, "workday")], dates, world).map((a) => a.date).sort();
    expect(two).toEqual(["2026-05-04", "2026-05-07"]);
  });

  it("visits days in a seeded shuffle so the best date is not always on the first day", () => {
    const clips = Array.from({ length: 6 }, (_, i) => workDate(`2026-05-0${i + 1}`, { extraWork: 10 * i })).flat();
    const dates = collectDates(new MemCatalog(clips), world);
    const days = Array.from({ length: 6 }, (_, i) => day(i + 1, "workday"));
    const best = (seed: number) => assignDates(days, dates, makeWorld({ seed })).find((a) => a.date === "2026-05-06")!.day_index;
    const seen = new Set(Array.from({ length: 12 }, (_, s) => best(s)));
    expect(seen.size).toBeGreaterThan(1);
    expect(best(7)).toBe(best(7));
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
      "d1-s1": ["07:00", "07:30"],
      "d1-s2": ["08:40", "10:45"],
      "d1-s3": ["12:00", "12:25"],
      "d1-s4": ["13:00", "14:40"],
      "d1-s5": ["18:45"],
      "d1-s7": ["18:20", "19:20"],
      "d1-t1": ["08:00"],
      "d1-t2": ["17:30"],
    });
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
      "d1-s1": ["08:00", "09:10", "10:00"],
      "d1-s2": ["12:30", "13:10", "14:20", "15:30"],
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
    expect(r.fills).toEqual([{ day_index: 1, slot_id: "d1-s1", role: "home_morning", from_date: "2026-05-02", delta_min: 30, seconds: 3600 }]);
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
      expect(r.fills).toEqual([{ day_index: 1, slot_id: "d1-s1", role: "home_morning", from_date: "2026-05-02", delta_min: 280, seconds: 1800, fallback: true }]);
      expect(r.entries.map((e) => e.clip_id.slice(-5))).toEqual(["11:40"]);
    });

    it("does not use home clips after the date's first away clip, or at/after 13:00", () => {
      const afterAway = [...bare("2026-05-01"), ...rec("2026-05-02", [away("11:00"), home("11:40", 30)])];
      expect(fill([dayN(1, [MORNING])], A1, afterAway).fills).toEqual([]);
      const late = [...bare("2026-05-01"), ...rec("2026-05-02", [home("13:00", 30)])];
      expect(fill([dayN(1, [MORNING])], A1, late).fills).toEqual([]);
      const justBefore = fill([dayN(1, [MORNING])], A1, [...bare("2026-05-01"), ...rec("2026-05-02", [home("12:59", 30)])]);
      expect(justBefore.fills.map((f) => f.fallback)).toEqual([true]);
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
    const fills = result.summary.home_fills;
    expect(fills.length).toBeGreaterThan(0);
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
