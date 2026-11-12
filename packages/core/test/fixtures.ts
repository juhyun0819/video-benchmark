import {
  ScheduleSchema,
  WorldSchema,
  WeekdaySchema,
  parseClock,
  timeOfDayAt,
  type Clip,
  type ClipCatalog,
  type ClipCategory,
  type Schedule,
  type ScheduleDay,
  type Setting,
  type Slot,
  type World,
} from "../src/index.js";

// Synthetic data only: one person ("p1") with NextMe-style day groups and Ego4D-style hobby videos.

export function makeWorld(over: { seed?: number; routine?: number } = {}): World {
  return WorldSchema.parse({
    person_id: "p-fixture",
    attributes: { country: "US", occupation_group: "professional", work_pattern: "day_shift" },
    distribution_source: "atus_2025",
    nodes: {
      home: { kind: "home" },
      kitchen: { kind: "kitchen", default: "home" },
      workplace: { kind: "workplace", job: "clerk" },
      climbing: { kind: "hobby", activity: "climbing", setting: "venue_indoor", weekly_freq: 2, tags: ["climbing"] },
      diner: { kind: "frequent_place", setting: "venue_indoor", optional: true },
    },
    node_binding: {
      home: { dataset: ["nextme800"], person_source: "p1", node: "home" },
      workplace: { dataset: ["nextme800"], person_source: "p1", node: "work" },
      climbing: { dataset: ["ego4d"], tags_any: ["climbing"] },
      diner: { dataset: ["nextme800"], person_source: "p1", node: "diner" },
    },
    meal_style: {
      workday: { lunch: { workplace: 1 }, dinner: { home: 1 } },
      rest_day: { breakfast: { home: 1 }, lunch: { home: 1 }, dinner: { home: 1 } },
      breakfast_skip_prob: 0,
      place_nodes: { eat_out: "diner" },
    },
    dials: {
      length_days: 28,
      routine_strength: over.routine ?? 0.5,
      exception_rate: { slot: 0, day: 0 },
      wake_sleep: "default_v1",
    },
    seed: over.seed ?? 4242,
  });
}

export type SlotDef = {
  role: string;
  category: ClipCategory;
  setting: Setting;
  node: string;
  activity: string;
  tags?: string[];
  required: boolean;
  window: [string, string];
};

const home = (role: string, activity: string, window: [string, string], required: boolean): SlotDef => ({
  role,
  category: "home",
  setting: "home_indoor",
  node: "home",
  activity,
  required,
  window,
});

export const WORKDAY: SlotDef[] = [
  home("home_morning", "getting ready", ["07:00", "08:00"], true),
  { role: "work_am", category: "work", setting: "venue_indoor", node: "workplace", activity: "work at desk", required: true, window: ["09:00", "12:00"] },
  { role: "lunch", category: "meal", setting: "venue_indoor", node: "workplace", activity: "eating lunch", required: false, window: ["12:00", "12:45"] },
  { role: "work_pm", category: "work", setting: "venue_indoor", node: "workplace", activity: "work at desk", required: true, window: ["13:00", "17:00"] },
  { role: "dinner", category: "meal", setting: "home_indoor", node: "home", activity: "eating dinner", required: false, window: ["18:30", "19:30"] },
  home("home_evening", "relaxing at home", ["20:00", "22:00"], false),
];

export const HOBBY_DAY: SlotDef[] = [
  home("home_morning", "getting ready", ["08:00", "09:00"], true),
  { role: "hobby", category: "hobby", setting: "venue_indoor", node: "climbing", activity: "climbing", tags: ["climbing"], required: true, window: ["10:00", "12:00"] },
  { role: "dinner", category: "meal", setting: "home_indoor", node: "home", activity: "eating dinner", required: false, window: ["18:30", "19:30"] },
  home("home_evening", "relaxing at home", ["20:00", "22:00"], false),
];

const weekdayOf = (dayIndex: number) => WeekdaySchema.options[(dayIndex - 1) % 7]!;
const isWeekend = (dayIndex: number) => (dayIndex - 1) % 7 >= 5;

/** A schedule day; by default Mon–Fri are workdays and Sat/Sun hobby days (week starts Monday). */
export function makeDay(dayIndex: number, attempt = 0, defs?: SlotDef[]): ScheduleDay {
  const weekend = isWeekend(dayIndex);
  const use = defs ?? (weekend ? HOBBY_DAY : WORKDAY);
  const slots: Slot[] = use.map((def, i) => ({
    slot_id: `d${dayIndex}-s${i + 1}`,
    role: def.role,
    required: def.required,
    window: def.window,
    spec: {
      category: def.category,
      setting: def.setting,
      node: def.node,
      activity: def.activity,
      time_of_day: timeOfDayAt(parseClock(def.window[0])),
      tags: def.tags ?? [],
    },
  }));
  return {
    day_index: dayIndex,
    weekday: weekdayOf(dayIndex),
    day_type: weekend ? "hobby_day" : "workday",
    attempt,
    wake: use[0]!.window[0],
    sleep: "23:00",
    exceptions: [],
    slot_exception_eligible: false,
    slot_exception_candidates: 0,
    slots,
    dropped_slots: [],
    transitions: [],
    anchor_adjustments: [],
    jitter_redraws: 0,
  };
}

const donor = {
  case_id: "synthetic",
  backoff_level: "all" as const,
  filled: [],
  redraws: 0,
  adjustments: [],
  profile: { wake: 420, sleep: 1380, breakfast: 450, lunch: 720, dinner: 1110, meal_loc: {} },
};

/** `days` schedule days; `defsFor(d)` overrides a day's slot list. */
export function makeSchedule(days = 28, defsFor: (d: number) => SlotDef[] | undefined = () => undefined): Schedule {
  return ScheduleSchema.parse({
    person_id: "p-fixture",
    generator_version: "schedule.v1",
    template_version: "templates.fixture",
    gap_table_version: "gaps.fixture",
    gap_table_reviewed: false,
    exception_catalog_version: "exceptions.fixture",
    distribution_source: "atus_2025",
    donors: { weekday: donor, weekend: donor, country_fallback: false, weekend_redraws: 0 },
    meal_style_effective: {
      workday: { lunch: { workplace: 1 }, dinner: { home: 1 } },
      rest_day: { breakfast: { home: 1 }, lunch: { home: 1 }, dinner: { home: 1 } },
    },
    realized: {
      slot_exception_rate: null,
      slot_exception_share: null,
      slot_exception_eligible_days: 0,
      day_exception_rate: 0,
      optional_presence: {},
      shrunk: {},
      composition_consistency: {},
      dials: { routine_strength: 0.5, exception_rate: { slot: 0, day: 0 } },
    },
    days: Array.from({ length: days }, (_, i) => makeDay(i + 1, 0, defsFor(i + 1))),
  });
}

export function clip(over: Partial<Clip> & Pick<Clip, "clip_id">): Clip {
  const start = over.source_start_s ?? 0;
  return {
    category: "home",
    setting: "home_indoor",
    node: "home",
    activity: "getting ready",
    time_of_day: "morning",
    tags: [],
    source: { dataset: "nextme800", video_id: `vid-${over.clip_id}`, media: { kind: "frames", approx_fps: 1, tar: "t.tar" } },
    source_start_s: start,
    source_end_s: start + 1200,
    day_group: `group-${over.clip_id}`,
    person_source: "p1",
    fingerprint: null,
    extractor_version: "fixture.v1",
    ...over,
  };
}

export type ClipBankOptions = {
  /** NextMe-style day groups with work clips (default 30). */
  groups?: number;
  /** How many of those groups also carry home and home-meal clips (default: all). */
  homeGroups?: number;
  /** Ego4D-style climbing videos, one day group each (default 12). */
  hobbyVideos?: number;
  /** Overrides every clip's duration in seconds. */
  durationS?: number;
};

type Part = { key: string; category: ClipCategory; setting: Setting; node: string; activity: string; tod: Clip["time_of_day"]; min: number };

const HOME_PARTS: Part[] = [
  { key: "a-morning", category: "home", setting: "home_indoor", node: "home", activity: "getting ready", tod: "morning", min: 20 },
  { key: "e-dinner", category: "meal", setting: "home_indoor", node: "home", activity: "eating dinner", tod: "evening", min: 30 },
  { key: "f-evening", category: "home", setting: "home_indoor", node: "home", activity: "relaxing at home", tod: "evening", min: 40 },
];
const WORK_PARTS: Part[] = [
  { key: "b-work-am", category: "work", setting: "venue_indoor", node: "work", activity: "work at desk", tod: "morning", min: 60 },
  { key: "c-lunch", category: "meal", setting: "venue_indoor", node: "work", activity: "eating lunch", tod: "midday", min: 20 },
  { key: "d-work-pm", category: "work", setting: "venue_indoor", node: "work", activity: "work at desk", tod: "afternoon", min: 60 },
];

/** Synthetic clips: per group one video with non-overlapping spans; clip ids sort by group first. */
export function makeClips(o: ClipBankOptions = {}): Clip[] {
  const groups = o.groups ?? 30;
  const homeGroups = o.homeGroups ?? groups;
  const out: Clip[] = [];
  for (let g = 1; g <= groups; g++) {
    const id = `g${String(g).padStart(2, "0")}`;
    const parts = g <= homeGroups ? [...HOME_PARTS, ...WORK_PARTS] : WORK_PARTS;
    parts.forEach((p, i) => {
      const start = i * 7200;
      out.push(
        clip({
          clip_id: `${id}-${p.key}`,
          category: p.category,
          setting: p.setting,
          node: p.node,
          activity: p.activity,
          time_of_day: p.tod,
          source: { dataset: "nextme800", video_id: `nm-${id}`, media: { kind: "frames", approx_fps: 1, tar: `${id}.tar` } },
          source_start_s: start,
          source_end_s: start + (o.durationS ?? p.min * 60),
          day_group: `nextme800:${id}`,
          person_source: "p1",
        }),
      );
    });
  }
  for (let v = 1; v <= (o.hobbyVideos ?? 12); v++) {
    const id = `h${String(v).padStart(2, "0")}`;
    out.push(
      clip({
        clip_id: `${id}-climb`,
        category: "hobby",
        setting: "venue_indoor",
        node: "climbing_gym",
        activity: "climbing a bouldering wall",
        time_of_day: "unknown",
        tags: ["climbing"],
        source: { dataset: "ego4d", video_id: `eg-${id}`, media: { kind: "video", fps: 30 } },
        source_start_s: 0,
        source_end_s: o.durationS ?? 1800,
        day_group: `ego4d:eg-${id}`,
        person_source: `ego4d:eg-${id}`,
      }),
    );
  }
  return out;
}

export function memoryCatalog(clips: Clip[]): ClipCatalog {
  const byId = new Map(clips.map((c) => [c.clip_id, c]));
  return {
    clips: () => clips,
    getClip: (id) => byId.get(id),
    getAnnotations: () => [],
  };
}

let next = 0;
/** Deterministic id generator for createTimeline. */
export const genId = (): string => `tl-${++next}`;
