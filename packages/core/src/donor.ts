import { TimelineError } from "./errors.js";
import { rngFor } from "./rng.js";
import type { MealPlace, World } from "./world.js";

export type DonorRow = {
  id: string;
  daytype: "weekday" | "weekend";
  occ: number;
  w: number;
  wake: number | null;
  bed: number | null;
  work_start: number | null;
  work_end: number | null;
  work_min: number | null;
  worked: boolean;
  breakfast_time: number | null;
  breakfast_loc: number | null;
  lunch_time: number | null;
  lunch_loc: number | null;
  dinner_time: number | null;
  dinner_loc: number | null;
  sport_has: boolean;
  sport_time: number | null;
  sport_outdoor: boolean | null;
  house_min: number;
  social_out: boolean;
  sex?: string;
  age_band?: string;
};

export type DayKind = "weekday" | "weekend";
type Meal = "breakfast" | "lunch" | "dinner";

export type DayProfile = {
  wake: number;
  sleep: number;
  work_start?: number;
  work_end?: number;
  breakfast: number;
  lunch: number;
  dinner: number;
  sport?: number;
  meal_loc: Partial<Record<Meal, MealPlace>>;
};

export type BackoffLevel = "occupation_sex_age" | "occupation_sex" | "occupation" | "sex_age" | "all";

export type DonorViolation = { item: string; bound: number; value: number };
export type DonorAdjustment = { item: string; from: number; to: number };

export type DonorPick = {
  case_id: string;
  backoff_level: BackoffLevel;
  filled: string[];
  profile: DayProfile;
  /** Acceptance redraws spent on this pick (0-5). */
  redraws: number;
  /** Clamps applied after the redraws ran out. */
  adjustments: DonorAdjustment[];
  sport_fill?: "level" | "pool";
};

export interface DistributionSource {
  id: string;
  country: string;
  pool(kind: DayKind): DonorRow[];
}

export const OCCUPATION_CODES: Record<string, number> = {
  mgmt_business_fin: 1,
  professional: 2,
  service: 3,
  sales: 4,
  office_admin: 5,
  farm_fish_forest: 6,
  construction_extraction: 7,
  install_maint_repair: 8,
  production: 9,
  transport_moving: 10,
};

const MIN_CELL = 30;
const EARLY_WEEKEND_MARGIN = 120;
const DAY_SHIFT_START: readonly [number, number] = [300, 660];
const LOC_MAP: Record<number, MealPlace> = { 1: "home", 2: "workplace", 3: "others_home", 4: "eat_out" };

const REQUIRED_COLUMNS = [
  "id", "daytype", "occ", "w", "wake", "bed", "work_start", "work_end", "work_min", "worked",
  "breakfast_has", "breakfast_time", "breakfast_loc", "lunch_has", "lunch_time", "lunch_loc",
  "dinner_has", "dinner_time", "dinner_loc", "sport_has", "sport_time", "sport_outdoor",
  "house_min", "social_out",
] as const;

const num = (s: string | undefined): number | null => {
  if (s === undefined || s === "") return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
};
const bool = (s: string | undefined): boolean | null => (s === "True" ? true : s === "False" ? false : null);

/** Parse the per-respondent ATUS CSV (header-driven; plain comma-separated, no quoting). */
export function parseAtusPool(csvText: string): DonorRow[] {
  const lines = csvText.replace(/^\uFEFF/, "").split(/\r?\n/).filter((l) => l.trim() !== "");
  const header = lines[0]?.split(",");
  if (!header) return [];
  const col = new Map(header.map((h, i) => [h.trim(), i] as const));
  const missing = REQUIRED_COLUMNS.filter((name) => !col.has(name));
  if (missing.length > 0) throw new Error(`ATUS CSV header is missing column(s): ${missing.join(", ")}`);
  const rows: DonorRow[] = [];
  for (const line of lines.slice(1)) {
    const c = line.split(",");
    const get = (name: string): string | undefined => {
      const i = col.get(name);
      return i === undefined ? undefined : c[i];
    };
    const daytype = get("daytype");
    if (daytype !== "weekday" && daytype !== "weekend") continue;
    const meal = (m: Meal) => {
      const has = bool(get(`${m}_has`)) === true;
      return { time: has ? num(get(`${m}_time`)) : null, loc: has ? num(get(`${m}_loc`)) : null };
    };
    const b = meal("breakfast");
    const l = meal("lunch");
    const d = meal("dinner");
    const sportHas = bool(get("sport_has")) === true;
    const row: DonorRow = {
      id: get("id") ?? "",
      daytype,
      occ: num(get("occ")) ?? 0,
      w: num(get("w")) ?? 0,
      wake: num(get("wake")),
      bed: num(get("bed")),
      work_start: num(get("work_start")),
      work_end: num(get("work_end")),
      work_min: num(get("work_min")),
      worked: bool(get("worked")) === true,
      breakfast_time: b.time,
      breakfast_loc: b.loc,
      lunch_time: l.time,
      lunch_loc: l.loc,
      dinner_time: d.time,
      dinner_loc: d.loc,
      sport_has: sportHas,
      sport_time: sportHas ? num(get("sport_time")) : null,
      sport_outdoor: sportHas ? bool(get("sport_outdoor")) : null,
      house_min: num(get("house_min")) ?? 0,
      social_out: bool(get("social_out")) === true,
    };
    rows.push(row);
  }
  return rows;
}

export function atus2025Source(rows: DonorRow[]): DistributionSource {
  const base = (r: DonorRow) => r.wake !== null && r.bed !== null;
  return {
    id: "atus_2025",
    country: "US",
    pool(kind) {
      if (kind === "weekday") {
        return rows.filter((r) => r.daytype === "weekday" && r.worked && r.work_start !== null && base(r));
      }
      return rows.filter((r) => r.daytype === "weekend" && !r.worked && base(r));
    },
  };
}

type Attrs = World["attributes"];

function levelPool(
  level: BackoffLevel,
  pool: DonorRow[],
  attrs: Attrs,
  occCode: number | undefined,
): DonorRow[] | null {
  const needsOcc = level.startsWith("occupation");
  const needsSex = level.includes("sex");
  const needsAge = level.endsWith("age");
  if (needsOcc && occCode === undefined) return null;
  if (needsSex && (attrs.sex === undefined || pool.some((r) => r.sex === undefined))) return null;
  if (needsAge && (attrs.age_band === undefined || pool.some((r) => r.age_band === undefined))) return null;
  return pool.filter(
    (r) =>
      (!needsOcc || r.occ === occCode) &&
      (!needsSex || r.sex === attrs.sex) &&
      (!needsAge || r.age_band === attrs.age_band),
  );
}

const LEVELS: BackoffLevel[] = ["occupation_sex_age", "occupation_sex", "occupation", "sex_age", "all"];

function pickLevel(kind: DayKind, pool: DonorRow[], attrs: Attrs): { level: BackoffLevel; rows: DonorRow[] } {
  if (pool.length === 0) throw new TimelineError("DONOR_POOL_EMPTY", `No donor rows for ${kind}`, { kind });
  const occCode = OCCUPATION_CODES[attrs.occupation_group];
  for (const level of LEVELS) {
    if (level === "all") break;
    const rows = levelPool(level, pool, attrs, occCode);
    if (rows && rows.length >= MIN_CELL) return { level, rows };
  }
  return { level: "all", rows: pool };
}

type FillField = "work_start" | "work_end" | "breakfast_time" | "lunch_time" | "dinner_time";

function buildPick(
  kind: DayKind,
  donor: DonorRow,
  level: BackoffLevel,
  rows: DonorRow[],
  pool: DonorRow[],
  seed: number,
): DonorPick {
  const filled: string[] = [];
  const value = (field: FillField): number => {
    const own = donor[field];
    if (own !== null) return own;
    const cands = rows.filter((r) => r[field] !== null && r.w > 0);
    if (cands.length === 0) {
      throw new TimelineError("DONOR_POOL_EMPTY", `No rows carry ${field}`, { kind, field });
    }
    const v = rngFor(seed, "donor", kind, "fill", field).pickWeighted(
      cands.map((r) => ({ value: r[field] as number, weight: r.w })),
    );
    filled.push(field);
    return v;
  };

  const profile: DayProfile = {
    wake: donor.wake as number,
    sleep: donor.bed as number,
    breakfast: 0,
    lunch: 0,
    dinner: 0,
    meal_loc: {},
  };
  if (kind === "weekday") {
    profile.work_start = value("work_start");
    profile.work_end = value("work_end");
  }
  profile.breakfast = value("breakfast_time");
  profile.lunch = value("lunch_time");
  profile.dinner = value("dinner_time");
  let sportFill: "level" | "pool" | undefined;
  if (donor.sport_has && donor.sport_time !== null) {
    profile.sport = donor.sport_time;
  } else if (kind === "weekend") {
    const withSport = (rs: DonorRow[]) => rs.filter((r) => r.sport_has && r.sport_time !== null && r.w > 0);
    const levelRows = withSport(rows);
    const from = levelRows.length >= MIN_CELL ? levelRows : withSport(pool);
    if (from.length > 0) {
      profile.sport = rngFor(seed, "donor", kind, "fill", "sport_time").pickWeighted(
        from.map((r) => ({ value: r.sport_time as number, weight: r.w })),
      );
      sportFill = levelRows.length >= MIN_CELL ? "level" : "pool";
      filled.push("sport_time");
    }
  }

  for (const m of ["breakfast", "lunch", "dinner"] as const) {
    const code = donor[`${m}_loc`];
    const place = code === null ? undefined : LOC_MAP[code];
    if (place !== undefined && donor[`${m}_time`] !== null) profile.meal_loc[m] = place;
  }
  const pick: DonorPick = { case_id: donor.id, backoff_level: level, filled, profile, redraws: 0, adjustments: [] };
  if (sportFill !== undefined) pick.sport_fill = sportFill;
  return pick;
}

function drawDonor(kind: DayKind, rows: DonorRow[], seed: number, ...key: (string | number)[]): DonorRow {
  const cands = rows.filter((r) => r.w > 0);
  if (cands.length === 0) throw new TimelineError("DONOR_POOL_EMPTY", `No weighted donor rows for ${kind}`, { kind });
  return rngFor(seed, "donor", kind, ...key).pickWeighted(cands.map((r) => ({ value: r, weight: r.w })));
}

const WAKE_MIN = 240;
const SLEEP_MAX = 1679;
const BREAKFAST_AFTER_WAKE = 10;
const DINNER_BEFORE_SLEEP = 60;
const WORK_MIN_SPAN = 240;
const LUNCH_MARGIN = 60;
const SPORT_BEFORE_SLEEP_EXTRA = 60;
const MAX_ACCEPT_REDRAWS = 5;

/** Generic donor bounds (E5); the template-dependent check comes in through `accept`. */
export function donorBoundViolations(
  kind: DayKind,
  profile: DayProfile,
  lunchMinutes: number,
  hobbyMinutes: number,
): DonorViolation[] {
  const out: DonorViolation[] = [];
  const low = (item: string, value: number, bound: number) => {
    if (value < bound) out.push({ item, bound, value });
  };
  const high = (item: string, value: number, bound: number) => {
    if (value > bound) out.push({ item, bound, value });
  };
  low("wake", profile.wake, WAKE_MIN);
  high("sleep", profile.sleep, SLEEP_MAX);
  low("breakfast", profile.breakfast, profile.wake + BREAKFAST_AFTER_WAKE);
  high("dinner", profile.dinner, profile.sleep - DINNER_BEFORE_SLEEP);
  if (kind === "weekday" && profile.work_start !== undefined && profile.work_end !== undefined) {
    low("work_end", profile.work_end, profile.work_start + WORK_MIN_SPAN);
    const lo = profile.work_start + LUNCH_MARGIN;
    const hi = profile.work_end - LUNCH_MARGIN - lunchMinutes;
    if (profile.lunch < lo) out.push({ item: "lunch", bound: lo, value: profile.lunch });
    else if (profile.lunch > hi) out.push({ item: "lunch", bound: hi, value: profile.lunch });
  }
  if (kind === "weekend" && profile.sport !== undefined) {
    high("sport", profile.sport, profile.sleep - hobbyMinutes - SPORT_BEFORE_SLEEP_EXTRA);
  }
  return out;
}

const CLAMP_ORDER = ["wake", "sleep", "breakfast", "dinner", "work_end", "lunch", "sport"] as const;

/** Clamp violated items to their bounds in dependency order, recording each change. */
function clampToBounds(
  kind: DayKind,
  profile: DayProfile,
  lunchMinutes: number,
  hobbyMinutes: number,
): DonorAdjustment[] {
  const adjustments: DonorAdjustment[] = [];
  for (const item of CLAMP_ORDER) {
    const v = donorBoundViolations(kind, profile, lunchMinutes, hobbyMinutes).find((x) => x.item === item);
    if (!v) continue;
    profile[item] = v.bound;
    adjustments.push({ item, from: v.value, to: v.bound });
  }
  return adjustments;
}

export function selectDonors(
  attrs: Attrs,
  src: DistributionSource,
  opts: {
    seed: number;
    allowCountryFallback: boolean;
    lunchMinutes?: number;
    hobbyMinutes?: number;
    accept?: (kind: DayKind, profile: DayProfile) => DonorViolation[];
  },
): { weekday: DonorPick; weekend: DonorPick; country_fallback: boolean; weekend_redraws: number } {
  const mismatch = attrs.country !== src.country;
  if (mismatch && !opts.allowCountryFallback) {
    throw new TimelineError(
      "COUNTRY_SOURCE_MISSING",
      `Distribution source ${src.id} covers ${src.country}, not ${attrs.country}`,
      { country: attrs.country, source: src.id },
    );
  }
  const lunchMinutes = opts.lunchMinutes ?? 45;
  const hobbyMinutes = opts.hobbyMinutes ?? 120;

  let weekdayPool = src.pool("weekday");
  if (attrs.work_pattern === "day_shift") {
    weekdayPool = weekdayPool.filter(
      (r) => r.work_start !== null && r.work_start >= DAY_SHIFT_START[0] && r.work_start <= DAY_SHIFT_START[1],
    );
  }
  const weekendPool = src.pool("weekend");
  const wd = pickLevel("weekday", weekdayPool, attrs);
  const weekdayDonor = drawDonor("weekday", wd.rows, opts.seed, 0);

  const we = pickLevel("weekend", weekendPool, attrs);

  const violationsOf = (kind: DayKind, profile: DayProfile, wakeFloor?: number): DonorViolation[] => [
    ...donorBoundViolations(kind, profile, lunchMinutes, hobbyMinutes),
    ...(opts.accept?.(kind, profile) ?? []),
    ...(wakeFloor !== undefined && profile.wake < wakeFloor
      ? [{ item: "weekend_wake_floor", bound: wakeFloor, value: profile.wake }]
      : []),
  ];
  const accepted = (
    kind: DayKind,
    first: DonorRow,
    level: BackoffLevel,
    rows: DonorRow[],
    pool: DonorRow[],
    wakeFloor?: number,
  ): DonorPick => {
    let donor = first;
    let pick = buildPick(kind, donor, level, rows, pool, opts.seed);
    let k = 0;
    while (violationsOf(kind, pick.profile, wakeFloor).length > 0 && k < MAX_ACCEPT_REDRAWS) {
      k += 1;
      donor = drawDonor(kind, rows, opts.seed, "accept", k);
      pick = buildPick(kind, donor, level, rows, pool, opts.seed);
    }
    pick.redraws = k;
    if (donorBoundViolations(kind, pick.profile, lunchMinutes, hobbyMinutes).length > 0) {
      pick.adjustments = clampToBounds(kind, pick.profile, lunchMinutes, hobbyMinutes);
    }
    return pick;
  };

  const weekday = accepted("weekday", weekdayDonor, wd.level, wd.rows, weekdayPool);
  // The floor follows the final weekday pick and is never clamped.
  const wakeFloor = weekday.profile.wake - EARLY_WEEKEND_MARGIN;
  const weekendFirst = drawDonor("weekend", we.rows, opts.seed, 0);
  const weekend = accepted("weekend", weekendFirst, we.level, we.rows, weekendPool, wakeFloor);

  return {
    weekday,
    weekend,
    country_fallback: mismatch,
    weekend_redraws: weekend.redraws,
  };
}
