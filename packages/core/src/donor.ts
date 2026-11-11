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

export type DonorPick = {
  case_id: string;
  backoff_level: BackoffLevel;
  filled: string[];
  profile: DayProfile;
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
const MAX_WEEKEND_REDRAWS = 5;
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
      return rows.filter((r) => r.daytype === "weekend" && base(r));
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

function buildPick(kind: DayKind, donor: DonorRow, level: BackoffLevel, rows: DonorRow[], seed: number): DonorPick {
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
  if (donor.sport_has && donor.sport_time !== null) profile.sport = donor.sport_time;

  for (const m of ["breakfast", "lunch", "dinner"] as const) {
    const code = donor[`${m}_loc`];
    const place = code === null ? undefined : LOC_MAP[code];
    if (place !== undefined && donor[`${m}_time`] !== null) profile.meal_loc[m] = place;
  }
  return { case_id: donor.id, backoff_level: level, filled, profile };
}

function drawDonor(kind: DayKind, rows: DonorRow[], seed: number, attempt: number): DonorRow {
  const cands = rows.filter((r) => r.w > 0);
  if (cands.length === 0) throw new TimelineError("DONOR_POOL_EMPTY", `No weighted donor rows for ${kind}`, { kind });
  return rngFor(seed, "donor", kind, attempt).pickWeighted(cands.map((r) => ({ value: r, weight: r.w })));
}

export function selectDonors(
  attrs: Attrs,
  src: DistributionSource,
  opts: { seed: number; allowCountryFallback: boolean },
): { weekday: DonorPick; weekend: DonorPick; country_fallback: boolean; weekend_redraws: number } {
  const mismatch = attrs.country !== src.country;
  if (mismatch && !opts.allowCountryFallback) {
    throw new TimelineError(
      "COUNTRY_SOURCE_MISSING",
      `Distribution source ${src.id} covers ${src.country}, not ${attrs.country}`,
      { country: attrs.country, source: src.id },
    );
  }

  let weekdayPool = src.pool("weekday");
  if (attrs.work_pattern === "day_shift") {
    weekdayPool = weekdayPool.filter(
      (r) => r.work_start !== null && r.work_start >= DAY_SHIFT_START[0] && r.work_start <= DAY_SHIFT_START[1],
    );
  }
  const wd = pickLevel("weekday", weekdayPool, attrs);
  const weekdayDonor = drawDonor("weekday", wd.rows, opts.seed, 0);

  const we = pickLevel("weekend", src.pool("weekend"), attrs);
  let weekendDonor = drawDonor("weekend", we.rows, opts.seed, 0);
  let redraws = 0;
  const floor = (weekdayDonor.wake as number) - EARLY_WEEKEND_MARGIN;
  while ((weekendDonor.wake as number) < floor && redraws < MAX_WEEKEND_REDRAWS) {
    redraws += 1;
    weekendDonor = drawDonor("weekend", we.rows, opts.seed, redraws);
  }

  return {
    weekday: buildPick("weekday", weekdayDonor, wd.level, wd.rows, opts.seed),
    weekend: buildPick("weekend", weekendDonor, we.level, we.rows, opts.seed),
    country_fallback: mismatch,
    weekend_redraws: redraws,
  };
}
