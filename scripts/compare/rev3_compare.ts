// Before/after comparison of the rev3 timeline changes (CR1-CR8) on the same seeds.
//
//   run    — runs one tree's pipeline and writes <out>/<label>.json (plus full-run files under <out>/<label>/):
//            npx tsx scripts/compare/rev3_compare.ts run --root <repo> --label <name> --data <dir> --out <dir>
//              [--seeds 200] [--full-seeds 20] [--rs 0,0.5,1] [--slot 0.5] [--day 0.1] [--world <file>]
//   report — prints markdown tables comparing two run outputs:
//            npx tsx scripts/compare/rev3_compare.ts report --old <out>/old.json --new <out>/new.json
//
// The tree under test is imported from <root>/packages/core/src/index.ts, so the same script runs
// against an older checkout (a git worktree whose node_modules resolves zod). Reference checks
// (E5 bounds, the required-slots check, E2 consistency) always come from the tree this script is
// committed in. Schedules and manifests are read as raw JSON (G7), never through a tree's schemas.
// The clip bank is synthetic (built from the MCP test helpers' clip set); nothing is read from
// data/ except the ATUS per-respondent pool.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

/* eslint-disable @typescript-eslint/no-explicit-any */
type Core = Record<string, any>;

const SCRIPT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const POOL_REL = join("distributions", "atus2025", "atus2025_stats_per_respondent.csv");
const FIXED_NOW = new Date("2026-11-12T00:00:00.000Z");

async function loadCore(root: string): Promise<Core> {
  return (await import(pathToFileURL(join(root, "packages", "core", "src", "index.ts")).href)) as Core;
}

const readJson = (path: string): any => JSON.parse(readFileSync(path, "utf8"));
const writeJson = (path: string, x: unknown): void => {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(x, null, 2) + "\n", "utf8");
};
const raw = <T>(x: T): any => JSON.parse(JSON.stringify(x));

// ---------------------------------------------------------------------------------------------
// Inputs: config, world, bank

type TreeConfig = { templates: any; gaps: any; exceptions: any; defaults: any; source: any; rows: any[]; templateFile: string };

function loadTreeConfig(core: Core, root: string, dataDir: string): TreeConfig {
  const timeline = join(root, "config", "timeline");
  // Newest first: each tree has exactly one of these.
  const templateFile = ["templates.v3.json", "templates.v2.json", "templates.v1.json"].find((f) => existsSync(join(timeline, f)));
  if (templateFile === undefined) throw new Error(`no templates file in ${timeline}`);
  const templatesRaw: unknown = readJson(join(timeline, templateFile));
  const rows = core.parseAtusPool(readFileSync(join(dataDir, POOL_REL), "utf8"));
  return {
    templateFile,
    rows,
    templates: core.TemplatesSchema.parse(templatesRaw),
    gaps: core.GapTableSchema.parse(readJson(join(timeline, "gaps.v1.json"))),
    exceptions: core.ExceptionCatalogSchema.parse(readJson(join(timeline, "exceptions.v1.json"))),
    defaults: core.ScheduleDefaultsSchema.parse(readJson(join(root, "config", "schedule_defaults_atus2025.json"))),
    source: core.atus2025Source(rows),
  };
}

/** The example world plus a bound eat_out frequent place (rev3 shape: meal_style.rest_day). */
function compareWorld(base: any, seed: number, rs: number, slot: number, day: number): any {
  const w = structuredClone(base);
  w.person_id = `cmp-s${seed}-rs${rs}`;
  w.seed = seed;
  w.nodes.frequent_1 = { kind: "frequent_place", setting: "venue_indoor", optional: true };
  w.node_binding.frequent_1 = { dataset: ["nextme800"], person_source: "nextme800:wearer", node: "frequent_1" };
  w.meal_style.place_nodes = { ...w.meal_style.place_nodes, eat_out: "frequent_1" };
  w.dials.routine_strength = rs;
  w.dials.exception_rate = { slot, day };
  return w;
}

/** Parses the world with the tree's own schema: rev3 trees take rest_day, older trees take the same values as `weekend` (G19). */
function treeWorld(core: Core, w: any): any {
  const parse = (x: any) => (core.parseWorld ? core.parseWorld(x) : core.WorldSchema.parse(x));
  try {
    return parse(w);
  } catch (first) {
    const { rest_day, ...rest } = w.meal_style;
    if (rest_day === undefined) throw first;
    return parse({ ...w, meal_style: { ...rest, weekend: rest_day } });
  }
}

type Part = { key: string; category: string; setting: string; node: string; activity: string; tags?: string[]; min: number };

// The MCP test helpers' NextMe clip set (packages/mcp/test/helpers.ts), plus eat_out clips at the
// frequent place and a second workplace meal so a workday can eat lunch and dinner at work.
const NEXTME_PARTS: Part[] = [
  { key: "a-ready", category: "home", setting: "home_indoor", node: "home", activity: "getting ready", min: 5 },
  { key: "b-breakfast", category: "meal", setting: "home_indoor", node: "home", activity: "eating breakfast", min: 5 },
  { key: "c-work", category: "work", setting: "venue_indoor", node: "workplace", activity: "work at desk", min: 20 },
  { key: "d-lunch-work", category: "meal", setting: "venue_indoor", node: "workplace", activity: "eating lunch", min: 5 },
  { key: "e-lunch-home", category: "meal", setting: "home_indoor", node: "home", activity: "eating lunch", min: 5 },
  { key: "f-work", category: "work", setting: "venue_indoor", node: "workplace", activity: "work at desk", min: 20 },
  { key: "g-chores", category: "home", setting: "home_indoor", node: "home", activity: "doing chores", tags: ["chores"], min: 5 },
  { key: "h-resting", category: "home", setting: "home_indoor", node: "home", activity: "resting at home", min: 5 },
  { key: "i-dinner", category: "meal", setting: "home_indoor", node: "home", activity: "eating dinner", min: 5 },
  { key: "j-relaxing", category: "home", setting: "home_indoor", node: "home", activity: "relaxing at home", min: 5 },
  { key: "k-dinner-work", category: "meal", setting: "venue_indoor", node: "workplace", activity: "eating dinner", min: 5 },
  { key: "l-breakfast-out", category: "meal", setting: "venue_indoor", node: "frequent_1", activity: "eating breakfast", min: 5 },
  { key: "m-lunch-out", category: "meal", setting: "venue_indoor", node: "frequent_1", activity: "eating lunch", min: 5 },
  { key: "n-dinner-out", category: "meal", setting: "venue_indoor", node: "frequent_1", activity: "eating dinner", min: 5 },
];

const pad = (n: number) => String(n).padStart(2, "0");

function makeBank(groups = 32, hobbyTakes = 20): any[] {
  const clips: any[] = [];
  for (let g = 1; g <= groups; g++) {
    const video = `nm-g${pad(g)}`;
    NEXTME_PARTS.forEach((p, i) => {
      const start = i * 2000;
      clips.push({
        clip_id: `nextme800:${video}:${p.key}`,
        category: p.category,
        setting: p.setting,
        node: p.node,
        activity: p.activity,
        time_of_day: "unknown",
        tags: p.tags ?? [],
        source: { dataset: "nextme800", video_id: video, media: { kind: "frames", approx_fps: 1, tar: `${video}.tar` } },
        source_start_s: start,
        source_end_s: start + p.min * 60,
        day_group: `nextme800:g${pad(g)}`,
        person_source: "nextme800:wearer",
        fingerprint: null,
        extractor_version: "fixture.v1",
      });
    });
  }
  for (let h = 1; h <= hobbyTakes; h++) {
    const video = `take-${pad(h)}`;
    clips.push({
      clip_id: `egoexo4d:${video}:climb`,
      category: "hobby",
      setting: "venue_indoor",
      node: "hobby",
      activity: "climbing a bouldering wall",
      time_of_day: "unknown",
      tags: ["sport_climbing"],
      source: { dataset: "egoexo4d", video_id: video, media: { kind: "video", fps: 30 } },
      source_start_s: 0,
      source_end_s: 1800,
      day_group: `egoexo4d:${video}`,
      person_source: `egoexo4d:${video}`,
      fingerprint: null,
      extractor_version: "fixture.v1",
    });
  }
  return clips;
}

/** In-memory ClipCatalog over the synthetic bank, one caption per clip. */
function memoryCatalog(core: Core, clips: any[]) {
  const parsed = clips.map((c) => core.ClipSchema.parse(c));
  const byId = new Map(parsed.map((c: any) => [c.clip_id, c]));
  return {
    clips: () => parsed,
    getClip: (id: string) => byId.get(id),
    getAnnotations: (dataset: string, videoId: string) =>
      parsed
        .filter((c: any) => c.source.dataset === dataset && c.source.video_id === videoId)
        .map((c: any) => ({
          id: `cap-${c.clip_id}`,
          track: "captions",
          level: dataset === "nextme800" ? "L3" : "keystep",
          time: { kind: "interval", start: c.source_start_s, end: c.source_end_s },
          text: c.activity,
          original: {},
        })),
  };
}

// ---------------------------------------------------------------------------------------------
// Per-schedule metrics (raw JSON, both trees)

const TEMPLATES = ["workday", "hobby_day", "home_day"] as const;
const templateOf = (dayType: string): string => (dayType === "leave_day" ? "home_day" : dayType);

/** G12: anchors whose bounds are checked per template. */
const CHECKED: Record<string, readonly string[]> = {
  workday: ["wake", "sleep", "work_start", "work_end", "lunch", "dinner"],
  hobby_day: ["wake", "sleep", "breakfast", "lunch", "dinner", "sport"],
  home_day: ["wake", "sleep", "breakfast", "lunch", "dinner"],
};

type Counter = Record<string, number>;
const inc = (c: Counter, k: string, n = 1) => {
  c[k] = (c[k] ?? 0) + n;
};

type Consistency = { optional_jaccard: number; order_similarity: number; start_diff_minutes: number | null; pairs: number };

type SweepRow = {
  seed: number;
  rs: number;
  error?: { code: string; message: string };
  days?: number;
  daysByTemplate?: Counter;
  /** "<template>.<role>" → [attempted, dropped no_time, shrunk and kept]. */
  optional?: Record<string, [number, number, number]>;
  daysWithDrop?: number;
  daysWithDropByTemplate?: Counter;
  workdays?: number;
  workdayNoDinner?: Counter;
  restDinner?: Record<string, [number, number]>;
  /** hobby_day dinner [attempted, dropped] by the hobby slot's start bucket. */
  hobbyDinnerByStart?: Record<string, [number, number]>;
  /** The weekend donor's sport: own diary value, CR8 fill (level / pool), or none (template default used). */
  weekendSport?: { source: "own" | "level" | "pool" | "default"; minutes: number };
  exc?: {
    commonEligible: number;
    commonWithExc: number;
    ownEligible: number | null;
    ownWithExc: number | null;
    candidates: number | null;
    slotExc: number;
    noop: number;
    noopKnown: number;
    dayBase: number;
    dayExc: number;
    drawnEligible?: number;
    drawn?: number;
    r45Removed?: number;
  };
  donor?: Record<"weekday" | "weekend", { recordedRedraws: number | null; adjustments: string[]; violations: string[] }>;
  dayLevel?: {
    days: number;
    redrawHist: Counter | null;
    adjDays: number | null;
    adjItems: Counter;
    wouldDays: number | null;
    wouldItems: Counter;
    silentDays: number | null;
    silentItems: Counter;
    replicaMismatch: number | null;
    /** Baseline: slots with a start anchor (start ≥ anchor, = when the anchor wins) and work_pm's end anchor. */
    anchorCheck: { starts: number; startEqual: number; startBelow: number; ends: number; endEqual: number } | null;
  };
  consistency?: Record<string, { all: Consistency; same: Consistency | null }>;
};

function usablePlace(world: any, place: string, bound: Set<string>): boolean {
  if (place === "home" || place === "workplace") return true;
  const node = world.meal_style.place_nodes[place];
  return node !== undefined && bound.has(node);
}

type Ctx = { core: Core; ref: Core; cfg: TreeConfig; world: any; treeWorld: any; bound: Set<string> };

function sigmaOf(defaults: any, rs: number): number {
  const { routine_strength_0: s0, routine_strength_1: s1 } = defaults.day_level_sigma;
  return s0 + (s1 - s0) * rs;
}

/** The rev3 E5 template check (generate.ts requiredWorkdaySlotsFit), re-run with the reference tree. */
function requiredWorkdaySlotsFit(ref: Core, refCfg: TreeConfig, world: any, profile: any, sigma: number): boolean {
  const decisions = {
    day_index: 0,
    weekday: "Mon",
    day_type: "workday",
    template: "workday",
    meal_place: { breakfast: "home", lunch: "home", dinner: "home" },
    optional: {},
    exceptions: [],
  };
  const required = refCfg.templates.day_types.workday.filter((s: any) => s.required);
  const anchors: any = { wake: profile.wake, sleep: profile.sleep - sigma, breakfast: profile.breakfast, lunch: profile.lunch, dinner: profile.dinner };
  if (profile.work_start !== undefined) anchors.work_start = profile.work_start;
  if (profile.work_end !== undefined) anchors.work_end = profile.work_end;
  try {
    ref.resolveDayWindows(required, anchors, decisions, world, (from: string, to: string) => ref.gapQuantile(refCfg.gaps, from, to, "p90"));
    return true;
  } catch (e: any) {
    if (e?.code === "SCHEDULE_INFEASIBLE") return false;
    throw e;
  }
}

function boundMinutes(templates: any): { lunch: number; hobby: number } {
  return {
    lunch: templates.day_types.workday.find((s: any) => s.anchor === "lunch")?.duration_minutes ?? 0,
    hobby: templates.day_types.hobby_day.find((s: any) => s.anchor === "sport")?.duration_minutes ?? 0,
  };
}

/** E5 violations of a donor profile under the reference tree's bounds, template check and weekend wake floor. */
function donorViolations(ctx: Ctx, refCfg: TreeConfig, kind: "weekday" | "weekend", profile: any, wakeFloor?: number): string[] {
  const { lunch, hobby } = boundMinutes(refCfg.templates);
  const items = ctx.ref.donorBoundViolations(kind, profile, lunch, hobby).map((v: any) => v.item as string);
  const sigma = sigmaOf(refCfg.defaults, ctx.world.dials.routine_strength);
  if (kind === "weekday" && !requiredWorkdaySlotsFit(ctx.ref, refCfg, ctx.world, profile, sigma)) items.push("required_slots");
  if (wakeFloor !== undefined && profile.wake < wakeFloor) items.push("weekend_wake_floor");
  return items;
}

/**
 * Baseline day anchors (ea048f5 generate.ts dayAnchors), recomputed from the donor profile and the
 * day's attempt with the tree's own rngFor: the raw jittered values, then the old silent clamps.
 */
function baselineAnchors(ctx: Ctx, profile: any, day: any): { rawA: Record<string, number>; clamped: Record<string, number> } {
  const { core, cfg } = ctx;
  const seed = ctx.world.seed;
  const sigma = sigmaOf(cfg.defaults, ctx.world.dials.routine_strength);
  const jitter = (name: string, value: number) =>
    Math.round(value + core.rngFor(seed, "day", day.day_index, day.attempt, "jitter", name).normal(0, sigma));
  const workday = templateOf(day.day_type) === "workday";
  const rawA: Record<string, number> = {
    wake: jitter("wake", profile.wake),
    sleep: jitter("sleep", profile.sleep),
    breakfast: jitter("breakfast", profile.breakfast),
    lunch: jitter("lunch", profile.lunch),
    dinner: jitter("dinner", profile.dinner),
    sport: jitter("sport", profile.sport ?? core.parseClock(cfg.defaults.weekend.sports_start.p50)),
  };
  if (workday && profile.work_start !== undefined) rawA.work_start = jitter("work_start", profile.work_start);
  if (workday && profile.work_end !== undefined) rawA.work_end = jitter("work_end", profile.work_end);
  const a = { ...rawA };
  a.wake = Math.max(a.wake!, 240);
  a.sleep = Math.min(a.sleep!, 27 * 60 + 59);
  a.breakfast = Math.max(a.breakfast!, a.wake + 10);
  const { lunch, hobby } = boundMinutes(cfg.templates);
  if (workday && a.work_start !== undefined && a.work_end !== undefined) {
    a.work_end = Math.max(a.work_end, a.work_start + 240);
    a.lunch = Math.min(Math.max(a.lunch!, a.work_start + 60), a.work_end - 60 - lunch);
  }
  a.dinner = Math.min(a.dinner!, a.sleep - 60);
  a.sport = Math.min(a.sport!, a.sleep - hobby - 60);
  return { rawA, clamped: a };
}

function consistencyOf(ctx: Ctx, days: any[]): SweepRow["consistency"] {
  const r = ctx.ref.realizedOf(days, ctx.world, ctx.cfg.templates).composition_consistency;
  const out: NonNullable<SweepRow["consistency"]> = {};
  for (const [name, v] of Object.entries<any>(r)) {
    const { same_reference, ...all } = v;
    out[name] = { all, same: same_reference };
  }
  return out;
}

const START_BUCKETS = ["<10:00", "10-12", "12-14", "14-16", ">=16:00"] as const;
function startBucket(minutes: number): string {
  if (minutes < 600) return START_BUCKETS[0];
  if (minutes < 720) return START_BUCKETS[1];
  if (minutes < 840) return START_BUCKETS[2];
  if (minutes < 960) return START_BUCKETS[3];
  return START_BUCKETS[4];
}

/** hobby_day dinner [attempted, dropped no_time] overall and by hobby slot start bucket, plus no_time drop days per template. */
function dinnerMetrics(core: Core, days: any[]) {
  const hobby: [number, number] = [0, 0];
  const byStart: Record<string, [number, number]> = {};
  const dropDays: Counter = {};
  const daysBy: Counter = {};
  for (const d of days) {
    const tmpl = templateOf(d.day_type);
    inc(daysBy, tmpl);
    const dropped = d.dropped_slots.filter((x: any) => x.reason === "no_time").map((x: any) => x.role as string);
    if (dropped.length > 0) inc(dropDays, tmpl);
    if (tmpl !== "hobby_day") continue;
    const kept = new Set<string>(d.slots.map((s: any) => s.role));
    if (!kept.has("dinner") && !dropped.includes("dinner")) continue;
    const start = d.slots.find((s: any) => s.role === "hobby");
    const c = (byStart[start ? startBucket(core.parseClock(start.window[0])) : "no hobby"] ??= [0, 0]);
    c[0] += 1;
    hobby[0] += 1;
    if (!kept.has("dinner")) {
      c[1] += 1;
      hobby[1] += 1;
    }
  }
  return { hobby, byStart, dropDays, daysBy };
}

function weekendSportOf(cfg: TreeConfig, core: Core, pick: any): NonNullable<SweepRow["weekendSport"]> {
  const fallback = core.parseClock(cfg.defaults.weekend.sports_start.p50);
  if (pick.profile.sport === undefined) return { source: "default", minutes: fallback };
  if ((pick.filled ?? []).includes("sport_time")) return { source: pick.sport_fill ?? "pool", minutes: pick.profile.sport };
  return { source: "own", minutes: pick.profile.sport };
}

function scheduleMetrics(ctx: Ctx, refCfg: TreeConfig, schedule: any, row: SweepRow): void {
  const { core, cfg } = ctx;
  const days: any[] = schedule.days;
  const rev3 = days.every((d) => typeof d.jitter_redraws === "number");
  const optionalRoles = (tmpl: string) =>
    new Set<string>(cfg.templates.day_types[tmpl].filter((s: any) => !s.required).map((s: any) => s.role));

  row.days = days.length;
  row.daysByTemplate = {};
  row.optional = {};
  row.daysWithDrop = 0;
  row.daysWithDropByTemplate = {};
  row.workdays = 0;
  row.workdayNoDinner = {};
  row.restDinner = { hobby_day: [0, 0], home_day: [0, 0] };
  for (const d of days) {
    const tmpl = templateOf(d.day_type);
    inc(row.daysByTemplate, tmpl);
    const kept = new Set<string>(d.slots.map((s: any) => s.role));
    const droppedNoTime = new Set<string>(d.dropped_slots.filter((x: any) => x.reason === "no_time").map((x: any) => x.role));
    const shrunk = new Set<string>(d.dropped_slots.filter((x: any) => x.reason === "shrunk").map((x: any) => x.role));
    if (droppedNoTime.size > 0) {
      row.daysWithDrop += 1;
      inc(row.daysWithDropByTemplate, tmpl);
    }
    for (const role of optionalRoles(tmpl)) {
      if (!kept.has(role) && !droppedNoTime.has(role)) continue;
      const c = (row.optional[`${tmpl}.${role}`] ??= [0, 0, 0]);
      c[0] += 1;
      if (droppedNoTime.has(role) && !kept.has(role)) c[1] += 1;
      if (shrunk.has(role) && kept.has(role)) c[2] += 1;
    }
    if (d.day_type === "workday") {
      row.workdays += 1;
      if (!kept.has("dinner")) inc(row.workdayNoDinner, droppedNoTime.has("dinner") ? "no_time" : "not_planned");
    }
    if (tmpl !== "workday" && (kept.has("dinner") || droppedNoTime.has("dinner"))) {
      const c = row.restDinner[tmpl]!;
      c[0] += 1;
      if (!kept.has("dinner")) c[1] += 1;
    }
  }

  // Exceptions: one E1 definition counted from the schedule for both trees, plus each tree's own view.
  const exc: NonNullable<SweepRow["exc"]> = {
    commonEligible: 0,
    commonWithExc: 0,
    ownEligible: rev3 ? 0 : null,
    ownWithExc: rev3 ? 0 : null,
    candidates: rev3 ? 0 : null,
    slotExc: 0,
    noop: 0,
    noopKnown: 0,
    dayBase: 0,
    dayExc: 0,
  };
  if (!rev3) Object.assign(exc, { drawnEligible: 0, drawn: 0, r45Removed: 0 });
  const decideOpts = { boundNodes: ctx.bound, defaults: cfg.defaults };
  const noSlotWorld = { ...ctx.treeWorld, dials: { ...ctx.treeWorld.dials, exception_rate: { ...ctx.treeWorld.dials.exception_rate, slot: 0 } } };
  for (const d of days) {
    const dayExc = d.exceptions.find((e: any) => e.scope === "day");
    const slotExcs = d.exceptions.filter((e: any) => e.scope === "slot");
    if ((dayExc?.replaces ?? d.day_type) === "workday") {
      exc.dayBase += 1;
      if (dayExc) exc.dayExc += 1;
    }
    const kept = new Set<string>(d.slots.map((s: any) => s.role));
    const eligible =
      !dayExc &&
      cfg.exceptions.slot.some(
        (e: any) => e.day_types.includes(d.day_type) && kept.has(e.replaces_role) && usablePlace(ctx.world, e.meal_place, ctx.bound),
      );
    if (eligible) {
      exc.commonEligible += 1;
      if (slotExcs.length > 0) exc.commonWithExc += 1;
    }
    exc.slotExc += slotExcs.length;
    if (rev3) {
      if (d.slot_exception_eligible) {
        exc.ownEligible! += 1;
        if (slotExcs.length > 0) exc.ownWithExc! += 1;
      }
      exc.candidates! += d.slot_exception_candidates;
      for (const e of slotExcs) {
        exc.noopKnown += 1;
        if (e.routine_place === e.meal_place) exc.noop += 1;
      }
    } else {
      // Baseline: the exception was drawn in decideDay before layout; R45 could remove it afterwards.
      const decided = core.decideDay(ctx.treeWorld, cfg.exceptions, decideOpts, d.day_index, d.attempt);
      const drawn = decided.exceptions.filter((e: any) => e.scope === "slot");
      const oldEligible =
        !decided.exceptions.some((e: any) => e.scope === "day") &&
        cfg.exceptions.slot.some(
          (e: any) =>
            e.day_types.includes(decided.day_type) &&
            (e.meal_place !== "workplace" || decided.template === "workday") &&
            usablePlace(ctx.world, e.meal_place, ctx.bound),
        );
      if (oldEligible) exc.drawnEligible! += 1;
      exc.drawn! += drawn.length;
      exc.r45Removed! += drawn.length - slotExcs.length;
      if (slotExcs.length > 0) {
        // Same rng keys with exception_rate.slot = 0 give the routine meal place the exception replaced.
        const routine = core.decideDay(noSlotWorld, cfg.exceptions, decideOpts, d.day_index, d.attempt);
        for (const e of slotExcs) {
          exc.noopKnown += 1;
          if (routine.meal_place[e.replaces] === e.meal_place) exc.noop += 1;
        }
      }
    }
  }
  row.exc = exc;
  row.hobbyDinnerByStart = dinnerMetrics(core, days).byStart;
  row.weekendSport = weekendSportOf(cfg, core, schedule.donors.weekend);

  // Donor level.
  const wd = schedule.donors.weekday;
  const we = schedule.donors.weekend;
  row.donor = {
    weekday: {
      recordedRedraws: rev3 ? wd.redraws : null,
      adjustments: (wd.adjustments ?? []).map((a: any) => a.item),
      violations: donorViolations(ctx, refCfg, "weekday", wd.profile),
    },
    weekend: {
      recordedRedraws: rev3 ? we.redraws : null,
      adjustments: (we.adjustments ?? []).map((a: any) => a.item),
      violations: donorViolations(ctx, refCfg, "weekend", we.profile, wd.profile.wake - 120),
    },
  };

  // Day level.
  const { lunch, hobby } = boundMinutes(refCfg.templates);
  const dl: NonNullable<SweepRow["dayLevel"]> = {
    days: days.length,
    redrawHist: rev3 ? {} : null,
    adjDays: rev3 ? 0 : null,
    adjItems: {},
    wouldDays: rev3 ? null : 0,
    wouldItems: {},
    silentDays: rev3 ? null : 0,
    silentItems: {},
    replicaMismatch: rev3 ? null : 0,
    anchorCheck: rev3 ? null : { starts: 0, startEqual: 0, startBelow: 0, ends: 0, endEqual: 0 },
  };
  for (const d of days) {
    const tmpl = templateOf(d.day_type);
    if (rev3) {
      inc(dl.redrawHist!, String(Math.min(d.jitter_redraws, 3)));
      if (d.anchor_adjustments.length > 0) dl.adjDays! += 1;
      for (const a of d.anchor_adjustments) inc(dl.adjItems, a.item);
      continue;
    }
    const profile = tmpl === "workday" ? wd.profile : we.profile;
    const { rawA, clamped } = baselineAnchors(ctx, profile, d);
    if (core.formatClock(clamped.wake) !== d.wake || core.formatClock(clamped.sleep) !== d.sleep) dl.replicaMismatch! += 1;
    const ac = dl.anchorCheck!;
    for (const s of d.slots) {
      const t = cfg.templates.day_types[tmpl].find((x: any) => x.role === s.role);
      if (t?.anchor !== undefined && clamped[t.anchor] !== undefined) {
        const start = core.parseClock(s.window[0]);
        ac.starts += 1;
        if (start === clamped[t.anchor]) ac.startEqual += 1;
        if (start < clamped[t.anchor]!) ac.startBelow += 1;
      }
      if (t?.end_anchor !== undefined && clamped[t.end_anchor] !== undefined) {
        ac.ends += 1;
        if (core.parseClock(s.window[1]) === clamped[t.end_anchor]) ac.endEqual += 1;
      }
    }
    const kind = tmpl === "workday" ? "weekday" : "weekend";
    const checked = new Set(CHECKED[tmpl]);
    const would = ctx.ref
      .donorBoundViolations(kind, { ...rawA, meal_loc: {} }, lunch, hobby)
      .map((v: any) => v.item as string)
      .filter((item: string) => checked.has(item));
    if (would.length > 0) dl.wouldDays! += 1;
    for (const item of would) inc(dl.wouldItems, item);
    const silent = [...checked].filter((item) => rawA[item] !== undefined && rawA[item] !== clamped[item]);
    if (silent.length > 0) dl.silentDays! += 1;
    for (const item of silent) inc(dl.silentItems, item);
  }
  row.dayLevel = dl;
  row.consistency = consistencyOf(ctx, days);
}

// ---------------------------------------------------------------------------------------------
// Ablation (rev3 trees only): which weekend change moves hobby_day dinner drops

type AblationVariant = "replay" | "fill_to_default" | "old_pool" | "old_pool+fill_to_default";
const ABLATIONS: readonly AblationVariant[] = ["replay", "fill_to_default", "old_pool", "old_pool+fill_to_default"];

type AblationRow = {
  variant: AblationVariant;
  seed: number;
  rs: number;
  error?: { code: string; message: string };
  /** replay only: days that differ from generateSchedule's own days (must be 0). */
  replayMismatch?: number;
  hobbyDinner?: [number, number];
  hobbyDinnerByStart?: Record<string, [number, number]>;
  dropDays?: Counter;
  daysBy?: Counter;
  weekendSport?: NonNullable<SweepRow["weekendSport"]>;
};

/** The baseline weekend pool (ea048f5 atus2025Source): weekend rows with wake and bed, worked rows included. */
function oldWeekendPoolSource(cfg: TreeConfig): any {
  const base = (r: any) => r.wake !== null && r.bed !== null;
  return {
    id: cfg.source.id,
    country: cfg.source.country,
    pool: (kind: string) => (kind === "weekend" ? cfg.rows.filter((r) => r.daytype === "weekend" && base(r)) : cfg.source.pool(kind)),
  };
}

/** generateSchedule's day loop (attempt 0, then 1..5 on SCHEDULE_INFEASIBLE) with the given donors. */
function replayDays(core: Core, input: any, donors: any, length: number): any[] {
  const days: any[] = [];
  for (let d = 1; d <= length; d++) {
    let day: any;
    let last: any;
    for (let attempt = 0; attempt <= 5 && day === undefined; attempt++) {
      try {
        day = core.generateDay(input, donors, d, attempt);
      } catch (e: any) {
        if (e?.code !== "SCHEDULE_INFEASIBLE") throw e;
        last = e;
      }
    }
    if (day === undefined) throw last;
    days.push(raw(day));
  }
  return days;
}

/**
 * Schedule-level ablation on a rev3 tree. "fill_to_default" sets a CR8-filled weekend sport back to
 * the template default (sports_start p50, what the baseline used for donors without sport) and
 * replays the days through generateDay; "old_pool" selects donors from the baseline weekend pool
 * (worked rows included). The weekend wake floor inside donor acceptance stays (it is not script-side).
 */
function ablationRow(ctx: Ctx, variant: AblationVariant, seed: number, rs: number): AblationRow {
  const { core, cfg } = ctx;
  const row: AblationRow = { variant, seed, rs };
  try {
    const source = variant.startsWith("old_pool") ? oldWeekendPoolSource(cfg) : cfg.source;
    const input = { ...cfg, source, world: ctx.treeWorld, boundNodes: ctx.bound };
    const schedule = raw(core.generateSchedule(input));
    const donors = structuredClone(schedule.donors);
    const selected = weekendSportOf(cfg, core, donors.weekend);
    let days: any[] = schedule.days;
    if (variant === "replay") {
      const again = replayDays(core, input, donors, ctx.treeWorld.dials.length_days);
      row.replayMismatch = again.filter((d, i) => JSON.stringify(d) !== JSON.stringify(schedule.days[i])).length;
    } else if (variant.endsWith("fill_to_default")) {
      if ((donors.weekend.filled ?? []).includes("sport_time")) {
        donors.weekend.profile.sport = core.parseClock(cfg.defaults.weekend.sports_start.p50);
        donors.weekend.filled = donors.weekend.filled.filter((f: string) => f !== "sport_time");
        delete donors.weekend.sport_fill;
      }
      days = replayDays(core, input, donors, ctx.treeWorld.dials.length_days);
    }
    const m = dinnerMetrics(core, days);
    row.hobbyDinner = m.hobby;
    row.hobbyDinnerByStart = m.byStart;
    row.dropDays = m.dropDays;
    row.daysBy = m.daysBy;
    // Source as selected; a filled sport reset to the default is reported as "default".
    const reset = variant.endsWith("fill_to_default") && (selected.source === "level" || selected.source === "pool");
    row.weekendSport = reset ? { source: "default", minutes: donors.weekend.profile.sport } : selected;
  } catch (e) {
    row.error = errInfo(e);
  }
  return row;
}

// ---------------------------------------------------------------------------------------------
// Full runs: fill → verify → manifest

type FullRow = {
  seed: number;
  rs: number;
  stage: "ok" | "schedule" | "preflight" | "fill" | "verify" | "manifest";
  error?: { code: string; message: string };
  days?: number;
  regenDays?: number;
  matchingDropped?: Counter;
  workdays?: number;
  workdayNoDinnerFinal?: number;
  scheduleDropped?: Counter;
  tiling?: { days: number; failures: { day: number; detail: string }[] } | null;
};

const errInfo = (e: any) => ({ code: typeof e?.code === "string" ? e.code : "INTERNAL", message: String(e?.message ?? e).slice(0, 300) });

/** E9: clip spans ∪ no_video_spans tile [wake, sleep] exactly on every manifest day. */
function tilingCheck(core: Core, manifest: any): { days: number; failures: { day: number; detail: string }[] } {
  const failures: { day: number; detail: string }[] = [];
  const EPS = 0.0015;
  for (const day of manifest.days) {
    const base = (day.day_index - 1) * 86400;
    const wake = base + core.parseClock(day.wake) * 60;
    const sleep = base + core.parseClock(day.sleep) * 60;
    const spans = [
      ...day.clips.map((c: any) => [c.synthetic_start_s, c.synthetic_end_s] as [number, number]),
      ...day.no_video_spans.map((s: any) => [s.start, s.end] as [number, number]),
    ].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    let at = wake;
    let bad: string | undefined;
    for (const [s, e] of spans) {
      if (e < s - EPS) bad = `span ${s}-${e} is reversed`;
      else if (Math.abs(s - at) > EPS) bad = `${s > at ? "gap" : "overlap"} at ${at} (next span starts ${s})`;
      if (bad) break;
      at = e;
    }
    if (!bad && Math.abs(at - sleep) > EPS) bad = `covered up to ${at}, sleep is ${sleep}`;
    if (bad) failures.push({ day: day.day_index, detail: bad });
  }
  return { days: manifest.days.length, failures };
}

function fullRun(ctx: Ctx, catalog: any, outDir: string, seed: number, rs: number): FullRow {
  const { core, cfg } = ctx;
  const row: FullRow = { seed, rs, stage: "schedule" };
  const input = { ...cfg, world: ctx.treeWorld, boundNodes: ctx.bound };
  try {
    const schedule = core.generateSchedule(input);
    row.scheduleDropped = {};
    for (const d of schedule.days) for (const x of d.dropped_slots) if (x.reason === "no_time") inc(row.scheduleDropped, x.role);
    row.stage = "preflight";
    const t = core.createTimeline(ctx.treeWorld, schedule, () => `${ctx.treeWorld.person_id}-cmp`);
    core.preflightCapacity(t, catalog);
    row.stage = "fill";
    const regen = (d: number, a: number) => core.generateDay(input, t.schedule.donors, d, a);
    const filled = core.fillTimeline(t, catalog, core.rankSelector, regen);
    row.stage = "verify";
    const report = core.verifyTimeline(filled, catalog);
    if (!report.pass) {
      const failed = Object.entries<any>(report.rules).filter(([, r]) => !r.pass).map(([k]) => k);
      row.error = { code: "VERIFY_FAILED", message: failed.join(", ") };
      return row;
    }
    row.stage = "manifest";
    const manifest = raw(core.buildTimelineManifest(filled, catalog, FIXED_NOW, cfg.templates));
    writeJson(join(outDir, "exports", `${manifest.id}-r${manifest.revision}.manifest.json`), manifest);
    writeJson(join(outDir, "timelines", `${filled.id}.json`), filled);
    row.stage = "ok";
    row.days = manifest.days.length;
    const before = new Map<number, number>(schedule.days.map((d: any) => [d.day_index, d.attempt]));
    row.regenDays = manifest.days.filter((d: any) => d.attempt !== before.get(d.day_index)).length;
    row.matchingDropped = {};
    row.workdays = 0;
    row.workdayNoDinnerFinal = 0;
    const final = new Map<number, any>(filled.schedule.days.map((d: any) => [d.day_index, d]));
    for (const d of manifest.days) {
      for (const role of d.dropped_optional) inc(row.matchingDropped, role);
      if (d.day_type !== "workday") continue;
      row.workdays += 1;
      const hasDinner = final.get(d.day_index).slots.some((s: any) => s.role === "dinner");
      if (!hasDinner || d.dropped_optional.includes("dinner")) row.workdayNoDinnerFinal += 1;
    }
    row.tiling = manifest.days.every((d: any) => Array.isArray(d.no_video_spans)) ? tilingCheck(core, manifest) : null;
  } catch (e) {
    row.error = errInfo(e);
  }
  return row;
}

// ---------------------------------------------------------------------------------------------
// run

async function run(argv: string[]): Promise<void> {
  const { values } = parseArgs({
    args: argv,
    options: {
      root: { type: "string" },
      label: { type: "string" },
      data: { type: "string" },
      out: { type: "string" },
      world: { type: "string" },
      seeds: { type: "string", default: "200" },
      "full-seeds": { type: "string", default: "20" },
      rs: { type: "string", default: "0,0.5,1" },
      slot: { type: "string", default: "0.5" },
      day: { type: "string", default: "0.1" },
    },
    strict: true,
  });
  if (!values.root || !values.label || !values.data || !values.out) {
    throw new Error("run needs --root, --label, --data and --out");
  }
  const root = resolve(values.root);
  const out = resolve(values.out);
  const core = await loadCore(root);
  const ref = await loadCore(SCRIPT_ROOT);
  const cfg = loadTreeConfig(core, root, resolve(values.data));
  const refCfg = loadTreeConfig(ref, SCRIPT_ROOT, resolve(values.data));
  const baseWorld = readJson(values.world ?? join(SCRIPT_ROOT, "config", "worlds", "example-nextme.json"));
  const seeds = Number(values.seeds);
  const fullSeeds = Number(values["full-seeds"]);
  const rsList = values.rs.split(",").map(Number);
  const slot = Number(values.slot);
  const dayRate = Number(values.day);

  const clips = makeBank();
  const catalog = memoryCatalog(core, clips);
  const runDir = join(out, values.label);
  writeJson(join(runDir, "catalog", "bank.json"), clips);

  const ctxFor = (seed: number, rs: number): Ctx => {
    const world = ref.parseWorld(compareWorld(baseWorld, seed, rs, slot, dayRate));
    const tw = treeWorld(core, compareWorld(baseWorld, seed, rs, slot, dayRate));
    return { core, ref, cfg, world, treeWorld: tw, bound: new Set(Object.keys(tw.node_binding)) };
  };

  const sweep: SweepRow[] = [];
  const t0 = Date.now();
  for (const rs of rsList) {
    for (let seed = 1; seed <= seeds; seed++) {
      const ctx = ctxFor(seed, rs);
      const row: SweepRow = { seed, rs };
      let schedule: any;
      try {
        schedule = raw(core.generateSchedule({ ...cfg, world: ctx.treeWorld, boundNodes: ctx.bound }));
      } catch (e) {
        row.error = errInfo(e);
      }
      if (schedule) scheduleMetrics(ctx, refCfg, schedule, row);
      sweep.push(row);
    }
  }
  const ablation: AblationRow[] = [];
  const rev3Tree = sweep.some((r) => r.dayLevel?.redrawHist !== null && r.dayLevel !== undefined);
  if (rev3Tree) {
    for (const variant of ABLATIONS) {
      for (const rs of rsList) {
        for (let seed = 1; seed <= seeds; seed++) ablation.push(ablationRow(ctxFor(seed, rs), variant, seed, rs));
      }
    }
  }
  const t1 = Date.now();
  const full: FullRow[] = [];
  for (const rs of rsList) {
    for (let seed = 1; seed <= fullSeeds; seed++) {
      const ctx = ctxFor(seed, rs);
      full.push(fullRun(ctx, catalog, runDir, seed, rs));
      writeJson(join(runDir, "worlds", `${ctx.treeWorld.person_id}.json`), ctx.treeWorld);
    }
  }
  const t2 = Date.now();
  const result = {
    label: values.label,
    root,
    template_file: cfg.templateFile,
    dials: { routine_strength: rsList, exception_rate: { slot, day: dayRate } },
    seeds,
    full_seeds: fullSeeds,
    seconds: { sweep: (t1 - t0) / 1000, full: (t2 - t1) / 1000 },
    sweep,
    ablation,
    full,
  };
  writeJson(join(out, `${values.label}.json`), result);
  const failed = sweep.filter((r) => r.error).length;
  const fullFailed = full.filter((r) => r.stage !== "ok").length;
  process.stdout.write(
    `${values.label}: ${sweep.length} schedules (${failed} failed), ${full.length} full runs (${fullFailed} failed); ` +
      `${result.seconds.sweep}s + ${result.seconds.full}s → ${join(out, `${values.label}.json`)}\n`,
  );
}

// ---------------------------------------------------------------------------------------------
// report

type Result = { label: string; dials: any; seeds: number; full_seeds: number; sweep: SweepRow[]; ablation?: AblationRow[]; full: FullRow[] };

const pct = (n: number, d: number) => (d === 0 ? "—" : `${((100 * n) / d).toFixed(1)}%`);
const frac = (n: number, d: number) => `${pct(n, d)} (${n}/${d})`;
const num = (x: number | null | undefined, digits = 3) => (x === null || x === undefined || Number.isNaN(x) ? "—" : x.toFixed(digits));
const sum = (xs: number[]) => xs.reduce((s, x) => s + x, 0);
const mean = (xs: number[]) => (xs.length === 0 ? NaN : sum(xs) / xs.length);

function table(head: string[], rows: string[][]): string {
  const line = (cells: string[]) => `| ${cells.join(" | ")} |`;
  return [line(head), line(head.map(() => "---")), ...rows.map(line)].join("\n");
}

function groups<T extends { rs: number }>(rows: T[], rsList: number[]): [string, T[]][] {
  return [...rsList.map((rs) => [`rs ${rs}`, rows.filter((r) => r.rs === rs)] as [string, T[]]), ["all", rows]];
}

function report(argv: string[]): void {
  const { values } = parseArgs({ args: argv, options: { old: { type: "string" }, new: { type: "string" } }, strict: true });
  if (!values.old || !values.new) throw new Error("report needs --old and --new");
  const trees: Result[] = [readJson(values.old), readJson(values.new)];
  const [oldR, newR] = trees as [Result, Result];
  const rsList: number[] = newR.dials.routine_strength;
  const ok = (r: Result) => r.sweep.filter((x) => !x.error);
  const out: string[] = [];
  const h = (s: string) => out.push(`\n### ${s}\n`);

  h("Generation failures");
  const failRow = (r: Result) => {
    const codes: Counter = {};
    for (const x of r.sweep) if (x.error) inc(codes, x.error.code);
    const stages: Counter = {};
    for (const x of r.full) if (x.stage !== "ok") inc(stages, `${x.stage}:${x.error?.code}`);
    return [
      r.label,
      frac(r.sweep.length - ok(r).length, r.sweep.length),
      Object.entries(codes).map(([k, v]) => `${k} ${v}`).join(", ") || "—",
      frac(r.full.filter((x) => x.stage !== "ok").length, r.full.length),
      Object.entries(stages).map(([k, v]) => `${k} ${v}`).join(", ") || "—",
    ];
  };
  out.push(table(["tree", "schedule failures", "codes", "full-run failures", "stage:code"], trees.map(failRow)));

  h("Optional slots dropped at schedule time (no_time)");
  const dropRows: string[][] = [];
  for (const r of trees) {
    for (const [g, rows] of groups(ok(r), rsList)) {
      const days = sum(rows.map((x) => x.days!));
      const byT = (t: string) =>
        frac(sum(rows.map((x) => x.daysWithDropByTemplate![t] ?? 0)), sum(rows.map((x) => x.daysByTemplate![t] ?? 0)));
      dropRows.push([r.label, g, frac(sum(rows.map((x) => x.daysWithDrop!)), days), byT("workday"), byT("hobby_day"), byT("home_day")]);
    }
  }
  out.push(table(["tree", "group", "days with ≥1 no_time drop", "workday", "hobby_day", "home_day (incl. leave_day)"], dropRows));
  const keys = [...new Set(trees.flatMap((r) => ok(r).flatMap((x) => Object.keys(x.optional!))))].sort();
  const roleRows = keys.map((k) => {
    const cells = [k];
    for (const r of trees) {
      const c = ok(r).map((x): [number, number, number] => x.optional![k] ?? [0, 0, 0]);
      const att = sum(c.map((v) => v[0]));
      cells.push(att === 0 ? "—" : frac(sum(c.map((v) => v[1])), att));
      cells.push(att === 0 ? "—" : String(sum(c.map((v) => v[2]))));
    }
    return cells;
  });
  out.push(
    "\n" +
      table(["template.role", `${oldR.label} dropped ÷ attempted`, `${oldR.label} shrunk`, `${newR.label} dropped ÷ attempted`, `${newR.label} shrunk`], roleRows),
  );

  h("Workday dinner and rest-day dinner");
  const dinnerRows: string[][] = [];
  for (const r of trees) {
    for (const [g, rows] of groups(ok(r), rsList)) {
      const wdays = sum(rows.map((x) => x.workdays!));
      const noTime = sum(rows.map((x) => x.workdayNoDinner!["no_time"] ?? 0));
      const notPlanned = sum(rows.map((x) => x.workdayNoDinner!["not_planned"] ?? 0));
      const hob = rows.map((x) => x.restDinner!["hobby_day"]!);
      const hom = rows.map((x) => x.restDinner!["home_day"]!);
      dinnerRows.push([
        r.label,
        g,
        frac(noTime + notPlanned, wdays),
        `${noTime} / ${notPlanned}`,
        frac(sum(hob.map((v) => v[1])), sum(hob.map((v) => v[0]))),
        frac(sum(hom.map((v) => v[1])), sum(hom.map((v) => v[0]))),
      ]);
    }
  }
  out.push(
    table(
      ["tree", "group", "workdays without dinner", "no_time / not planned", "hobby_day dinner dropped", "home_day dinner dropped"],
      dinnerRows,
    ),
  );

  h("Matching time (full runs)");
  const matchRows: string[][] = [];
  for (const r of trees) {
    for (const [g, rows] of groups(r.full.filter((x) => x.stage === "ok"), rsList)) {
      const days = sum(rows.map((x) => x.days!));
      const dropped: Counter = {};
      for (const x of rows) for (const [k, v] of Object.entries(x.matchingDropped!)) inc(dropped, k, v);
      const sched: Counter = {};
      for (const x of rows) for (const [k, v] of Object.entries(x.scheduleDropped!)) inc(sched, k, v);
      matchRows.push([
        r.label,
        g,
        String(rows.length),
        frac(sum(Object.values(dropped)), days),
        Object.entries(dropped).map(([k, v]) => `${k} ${v}`).join(", ") || "—",
        frac(sum(rows.map((x) => x.regenDays!)), days),
        frac(sum(rows.map((x) => x.workdayNoDinnerFinal!)), sum(rows.map((x) => x.workdays!))),
      ]);
    }
  }
  out.push(
    table(
      ["tree", "group", "runs", "dropped_optional slots ÷ days", "by role", "days regenerated at matching", "workdays without a dinner clip"],
      matchRows,
    ),
  );

  h("Exceptions");
  const excRows: string[][] = [];
  for (const r of trees) {
    for (const [g, rows] of groups(ok(r), rsList)) {
      const e = rows.map((x) => x.exc!);
      const s = (f: (v: NonNullable<SweepRow["exc"]>) => number | null | undefined) => sum(e.map((v) => f(v) ?? 0));
      const own = e[0]?.ownEligible !== null;
      excRows.push([
        r.label,
        g,
        frac(s((v) => v.commonWithExc), s((v) => v.commonEligible)),
        own ? frac(s((v) => v.ownWithExc), s((v) => v.ownEligible)) : "—",
        own ? frac(s((v) => v.slotExc), s((v) => v.candidates)) : "—",
        own ? "—" : frac(s((v) => v.drawn), s((v) => v.drawnEligible)),
        own ? "—" : frac(s((v) => v.r45Removed), s((v) => v.drawn)),
        frac(s((v) => v.noop), s((v) => v.noopKnown)),
        frac(s((v) => v.dayExc), s((v) => v.dayBase)),
      ]);
    }
  }
  out.push(
    table(
      [
        "tree",
        "group",
        "slot rate (common E1)",
        "slot rate (rev3 recorded E1)",
        "slot share (E1)",
        "drawn before R45",
        "removed by R45",
        "no-op share",
        "day rate",
      ],
      excRows,
    ),
  );

  h("Donor level");
  const donorRows: string[][] = [];
  for (const r of trees) {
    for (const kind of ["weekday", "weekend"] as const) {
      for (const [g, rows] of groups(ok(r), rsList)) {
        const d = rows.map((x) => x.donor![kind]);
        const rev3 = d[0]?.recordedRedraws !== null;
        const hist: Counter = {};
        for (const v of d) if (v.recordedRedraws !== null) inc(hist, String(v.recordedRedraws));
        const items: Counter = {};
        for (const v of d) for (const it of v.adjustments) inc(items, it);
        const viol: Counter = {};
        for (const v of d) for (const it of v.violations) inc(viol, it);
        donorRows.push([
          r.label,
          kind,
          g,
          rev3 ? frac(d.filter((v) => v.recordedRedraws! > 0).length, d.length) : "—",
          rev3 ? Object.entries(hist).sort().map(([k, v]) => `${k}:${v}`).join(" ") : "—",
          rev3 ? frac(d.filter((v) => v.adjustments.length > 0).length, d.length) : "—",
          rev3 ? Object.entries(items).map(([k, v]) => `${k} ${v}`).join(", ") || "—" : "—",
          frac(d.filter((v) => v.violations.length > 0).length, d.length),
          Object.entries(viol).map(([k, v]) => `${k} ${v}`).join(", ") || "—",
        ]);
      }
    }
  }
  out.push(
    table(
      ["tree", "donor", "group", "redrawn", "redraws 0..5", "adjusted (clamp/repair)", "adjusted items", "final profile violates E5 (measured)", "violation items"],
      donorRows,
    ),
  );

  h("Day level (anchors)");
  const dayRows: string[][] = [];
  const anchorNotes: string[] = [];
  for (const r of trees) {
    for (const [g, rows] of groups(ok(r), rsList)) {
      const dl = rows.map((x) => x.dayLevel!);
      const days = sum(dl.map((v) => v.days));
      const rev3 = dl[0]?.redrawHist !== null;
      const items: Counter = {};
      for (const v of dl) for (const [k, n] of Object.entries(rev3 ? v.adjItems : v.wouldItems)) inc(items, k, n);
      const silent: Counter = {};
      for (const v of dl) for (const [k, n] of Object.entries(v.silentItems)) inc(silent, k, n);
      if (rev3) {
        const hist: Counter = {};
        for (const v of dl) for (const [k, n] of Object.entries(v.redrawHist!)) inc(hist, k, n);
        dayRows.push([
          r.label,
          g,
          frac(days - (hist["0"] ?? 0), days),
          ["0", "1", "2", "3"].map((k) => `${k === "3" ? "3+" : k}:${hist[k] ?? 0}`).join(" "),
          frac(sum(dl.map((v) => v.adjDays!)), days),
          Object.entries(items).map(([k, n]) => `${k} ${n}`).join(", ") || "—",
          "—",
        ]);
      } else {
        const ac = dl.map((v) => v.anchorCheck!);
        const acs = (f: (v: NonNullable<(typeof ac)[number]>) => number) => sum(ac.map(f));
        anchorNotes.push(
          `${r.label} ${g}: start anchors ${acs((v) => v.startEqual)}/${acs((v) => v.starts)} equal, ${acs((v) => v.startBelow)} below; ` +
            `end anchors ${acs((v) => v.endEqual)}/${acs((v) => v.ends)} equal`,
        );
        dayRows.push([
          r.label,
          g,
          `${frac(sum(dl.map((v) => v.wouldDays!)), days)} would`,
          "—",
          `${frac(sum(dl.map((v) => v.silentDays!)), days)} silent`,
          Object.entries(items).map(([k, n]) => `${k} ${n}`).join(", ") || "—",
          `${sum(dl.map((v) => v.replicaMismatch!))} mismatches; silent: ${Object.entries(silent).map(([k, n]) => `${k} ${n}`).join(", ") || "—"}`,
        ]);
      }
    }
  }
  out.push(
    table(
      ["tree", "group", "days needing a redraw", "redraws per day", "days clamped", "items (rev3: clamps, baseline: E5 violations)", "baseline replica check"],
      dayRows,
    ),
  );
  if (anchorNotes.length > 0) out.push("\nBaseline replica vs recorded slot windows:\n\n" + anchorNotes.map((n) => `- ${n}`).join("\n"));

  h("Composition consistency (E2, G15), mean over schedules");
  const consRows: string[][] = [];
  for (const r of trees) {
    for (const tmpl of TEMPLATES) {
      for (const [g, rows] of groups(ok(r), rsList).slice(0, -1)) {
        const c = rows.map((x) => x.consistency![tmpl]).filter((v): v is NonNullable<typeof v> => v !== undefined);
        const same = c.map((v) => v.same).filter((v): v is Consistency => v !== null);
        const m = (xs: Consistency[], f: (v: Consistency) => number | null) =>
          num(mean(xs.map(f).filter((v): v is number => v !== null)), f === startDiff ? 1 : 3);
        consRows.push([
          r.label,
          tmpl,
          g,
          `${m(c.map((v) => v.all), (v) => v.optional_jaccard)} / ${m(same, (v) => v.optional_jaccard)}`,
          `${m(c.map((v) => v.all), (v) => v.order_similarity)} / ${m(same, (v) => v.order_similarity)}`,
          `${m(c.map((v) => v.all), startDiff)} / ${m(same, startDiff)}`,
        ]);
      }
    }
  }
  out.push(table(["tree", "template", "group", "optional_jaccard all / same_ref", "order_similarity all / same_ref", "start_diff min all / same_ref"], consRows));

  const clock = (m: number) => `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(Math.round(m % 60)).padStart(2, "0")}`;
  const quant = (xs: number[], f: number) => {
    const s = [...xs].sort((a, b) => a - b);
    return s.length === 0 ? "—" : clock(s[Math.floor(f * (s.length - 1))]!);
  };
  type DinnerSource = { label: string; rows: { hobbyDinnerByStart?: Record<string, [number, number]>; weekendSport?: SweepRow["weekendSport"]; rs: number }[] };
  const sources: DinnerSource[] = [
    ...trees.map((r) => ({ label: r.label, rows: ok(r) })),
    ...ABLATIONS.filter((v) => v !== "replay").map((v) => ({
      label: `${newR.label} ablation ${v}`,
      rows: (newR.ablation ?? []).filter((x) => x.variant === v && !x.error),
    })),
  ].filter((s) => s.rows.length > 0);

  h("hobby_day dinner drop by hobby slot start");
  const startRows: string[][] = [];
  for (const s of sources) {
    const cells = [s.label];
    let all: [number, number] = [0, 0];
    for (const b of START_BUCKETS) {
      const c = s.rows.map((x): [number, number] => x.hobbyDinnerByStart?.[b] ?? [0, 0]);
      const att = sum(c.map((v) => v[0]));
      const drop = sum(c.map((v) => v[1]));
      all = [all[0] + att, all[1] + drop];
      cells.push(att === 0 ? "— (0)" : frac(drop, att));
    }
    cells.push(frac(all[1], all[0]));
    startRows.push(cells);
  }
  out.push(table(["source", ...START_BUCKETS, "all"], startRows));
  out.push("\nShare of hobby days per bucket (attempted dinners):\n");
  out.push(
    table(
      ["source", ...START_BUCKETS],
      sources.map((s) => {
        const att = START_BUCKETS.map((b) => sum(s.rows.map((x) => x.hobbyDinnerByStart?.[b]?.[0] ?? 0)));
        return [s.label, ...att.map((n) => pct(n, sum(att)))];
      }),
    ),
  );

  h("Weekend donor sport time by source (one donor per schedule, 200 seeds × 3 rs)");
  const sportRows: string[][] = [];
  for (const s of sources) {
    for (const src of ["own", "level", "pool", "default"] as const) {
      const rows = s.rows.filter((x) => x.weekendSport?.source === src);
      if (rows.length === 0) continue;
      const mins = rows.map((x) => x.weekendSport!.minutes);
      const c = rows.map((x) => Object.values(x.hobbyDinnerByStart ?? {}));
      const att = sum(c.flatMap((v) => v.map((p) => p[0])));
      const drop = sum(c.flatMap((v) => v.map((p) => p[1])));
      sportRows.push([s.label, src, frac(rows.length, s.rows.length), quant(mins, 0.25), quant(mins, 0.5), quant(mins, 0.75), frac(drop, att)]);
    }
  }
  out.push(table(["source", "sport source", "schedules", "sport p25", "p50", "p75", "hobby_day dinner dropped"], sportRows));

  const abl = newR.ablation ?? [];
  if (abl.length > 0) {
    h(`Ablation on ${newR.label} (schedule level)`);
    const ablRows: string[][] = [];
    for (const v of ABLATIONS) {
      const rows = abl.filter((x) => x.variant === v);
      const okRows = rows.filter((x) => !x.error);
      const hd = okRows.map((x) => x.hobbyDinner!);
      const drop = (t: string) => frac(sum(okRows.map((x) => x.dropDays![t] ?? 0)), sum(okRows.map((x) => x.daysBy![t] ?? 0)));
      ablRows.push([
        v,
        frac(rows.length - okRows.length, rows.length),
        v === "replay" ? `${sum(okRows.map((x) => x.replayMismatch!))} mismatching days` : "—",
        frac(sum(hd.map((x) => x[1])), sum(hd.map((x) => x[0]))),
        drop("hobby_day"),
        drop("workday"),
        drop("home_day"),
      ]);
    }
    out.push(
      table(["variant", "failures", "replay check", "hobby_day dinner dropped", "hobby_day days with a drop", "workday days with a drop", "home_day days with a drop"], ablRows),
    );
  }

  h("no_video_spans tiling (rev3 full runs)");
  const tileRows: string[][] = [];
  for (const r of trees) {
    const tiled = r.full.filter((x) => x.stage === "ok" && x.tiling);
    if (tiled.length === 0) {
      tileRows.push([r.label, "—", "—", "no no_video_spans in this tree"]);
      continue;
    }
    const fails = tiled.flatMap((x) => x.tiling!.failures.map((f) => `seed ${x.seed} rs ${x.rs} day ${f.day}: ${f.detail}`));
    tileRows.push([r.label, String(tiled.length), String(sum(tiled.map((x) => x.tiling!.days))), fails.length === 0 ? "0" : fails.slice(0, 5).join("; ")]);
  }
  out.push(table(["tree", "manifests", "days checked", "failures"], tileRows));

  process.stdout.write(out.join("\n") + "\n");
}

const startDiff = (v: Consistency) => v.start_diff_minutes;

// ---------------------------------------------------------------------------------------------

const [mode, ...rest] = process.argv.slice(2);
try {
  if (mode === "run") await run(rest);
  else if (mode === "report") report(rest);
  else throw new Error("usage: rev3_compare.ts run --root <repo> --label <name> --data <dir> --out <dir> | report --old <json> --new <json>");
} catch (e) {
  process.stderr.write(`${e instanceof Error ? e.stack ?? e.message : String(e)}\n`);
  process.exitCode = 1;
}
