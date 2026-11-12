import { describe, expect, it } from "vitest";
import {
  OCCUPATION_CODES,
  TimelineError,
  atus2025Source,
  donorBoundViolations,
  parseAtusPool,
  selectDonors,
  type DayProfile,
  type DonorRow,
} from "../src/index.js";

const ATTRS = { country: "US", occupation_group: "professional", work_pattern: "day_shift" as const };
const OPTS = { seed: 7, allowCountryFallback: false };

let counter = 0;
function row(over: Partial<DonorRow> = {}): DonorRow {
  counter += 1;
  return {
    id: `r${counter}`,
    daytype: "weekday",
    occ: OCCUPATION_CODES["professional"]!,
    w: 1,
    wake: 420,
    bed: 1380,
    work_start: 540,
    work_end: 1020,
    work_min: 480,
    worked: true,
    breakfast_time: 450,
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
    ...over,
  };
}
const many = (n: number, over: Partial<DonorRow> = {}) => Array.from({ length: n }, () => row(over));
const weekendRows = (n: number, over: Partial<DonorRow> = {}) =>
  many(n, { daytype: "weekend", worked: false, work_start: null, work_end: null, ...over });

describe("selectDonors", () => {
  it("same seed gives the same case_id", () => {
    const src = atus2025Source([...many(40), ...weekendRows(40)]);
    const a = selectDonors(ATTRS, src, OPTS);
    const b = selectDonors(ATTRS, src, OPTS);
    expect(a.weekday.case_id).toBe(b.weekday.case_id);
    expect(a.weekend.case_id).toBe(b.weekend.case_id);
  });

  it("occupation cell with 29 rows falls back to all, 30 rows uses occupation", () => {
    const other = many(50, { occ: OCCUPATION_CODES["service"]! });
    const wk = weekendRows(40);
    const small = selectDonors(ATTRS, atus2025Source([...many(29), ...other, ...wk]), OPTS);
    expect(small.weekday.backoff_level).toBe("all");
    const enough = selectDonors(ATTRS, atus2025Source([...many(30), ...other, ...wk]), OPTS);
    expect(enough.weekday.backoff_level).toBe("occupation");
  });

  it("skips sex/age levels when rows lack sex", () => {
    const src = atus2025Source([...many(40), ...weekendRows(40)]);
    const r = selectDonors({ ...ATTRS, sex: "F", age_band: "30-39" }, src, OPTS);
    expect(r.weekday.backoff_level).toBe("occupation");
  });

  it("uses sex/age levels when rows carry them", () => {
    const src = atus2025Source([
      ...many(35, { sex: "F", age_band: "30-39" }),
      ...many(35, { sex: "M", age_band: "30-39" }),
      ...weekendRows(40, { sex: "F", age_band: "30-39" }),
    ]);
    const r = selectDonors({ ...ATTRS, sex: "F", age_band: "30-39" }, src, OPTS);
    expect(r.weekday.backoff_level).toBe("occupation_sex_age");
  });

  it("unknown occupation group falls to all", () => {
    const src = atus2025Source([...many(40), ...weekendRows(40)]);
    const r = selectDonors({ ...ATTRS, occupation_group: "astronaut" }, src, OPTS);
    expect(r.weekday.backoff_level).toBe("all");
  });

  it("weighted draw frequency follows w", () => {
    const rows = [row({ id: "light", w: 1 }), row({ id: "heavy", w: 3 }), ...weekendRows(2)];
    const src = atus2025Source(rows);
    let heavy = 0;
    const n = 20000;
    for (let seed = 0; seed < n; seed++) {
      if (selectDonors(ATTRS, src, { seed, allowCountryFallback: false }).weekday.case_id === "heavy") heavy++;
    }
    expect(heavy / n).toBeGreaterThan(0.73);
    expect(heavy / n).toBeLessThan(0.77);
  });

  it("missing lunch_time is filled and recorded", () => {
    const rows = [row({ lunch_time: null, lunch_loc: null }), row({ lunch_time: 720 }), ...weekendRows(3)];
    const src = atus2025Source(rows);
    const r = selectDonors(ATTRS, src, OPTS);
    expect(r.weekday.profile.lunch).toBeTypeOf("number");
    if (r.weekday.case_id === rows[0]!.id) {
      expect(r.weekday.filled).toContain("lunch_time");
      expect(r.weekday.profile.lunch).toBe(720);
      expect(r.weekday.profile.meal_loc.lunch).toBeUndefined();
    }
    // Force the incomplete donor: single weekday row plus one donor to fill from.
    const only = atus2025Source([
      row({ id: "inc", lunch_time: null, lunch_loc: null, w: 1e9 }),
      row({ id: "src", lunch_time: 720, w: 1e-9 }),
      ...weekendRows(3),
    ]);
    const f = selectDonors(ATTRS, only, OPTS);
    expect(f.weekday.case_id).toBe("inc");
    expect(f.weekday.filled).toContain("lunch_time");
    expect(f.weekday.profile.lunch).toBe(720);
  });

  it("maps location codes and leaves unknown codes out; sleep is bed; sport only when present", () => {
    const src = atus2025Source([
      row({ breakfast_loc: 3, lunch_loc: 4, dinner_loc: 9, bed: 1470, sport_has: true, sport_time: 1140 }),
      ...weekendRows(3),
    ]);
    const p = selectDonors(ATTRS, src, OPTS).weekday.profile;
    expect(p.meal_loc).toEqual({ breakfast: "others_home", lunch: "eat_out" });
    expect(p.sleep).toBe(1470);
    expect(p.sport).toBe(1140);
    const none = selectDonors(ATTRS, atus2025Source([row(), ...weekendRows(3)]), OPTS).weekday.profile;
    expect(none.sport).toBeUndefined();
  });

  it("weekday pool filters worked, work_start window and nulls", () => {
    const rows = [
      row({ id: "ok" }),
      row({ id: "night", work_start: 1320 }),
      row({ id: "early", work_start: 200 }),
      row({ id: "noshift", worked: false }),
      row({ id: "nowake", wake: null }),
      ...weekendRows(2),
    ];
    const src = atus2025Source(rows);
    expect(src.pool("weekday").map((r) => r.id)).toEqual(["ok", "night", "early"]);
    const r = selectDonors(ATTRS, src, OPTS);
    expect(r.weekday.case_id).toBe("ok");
  });

  it("fills work_end when the donor lacks it", () => {
    const src = atus2025Source([
      row({ id: "a", work_end: null, w: 1e9 }),
      row({ id: "b", work_end: 1000, w: 1e-9 }),
      ...weekendRows(3),
    ]);
    const r = selectDonors(ATTRS, src, OPTS);
    expect(r.weekday.filled).toContain("work_end");
    expect(r.weekday.profile.work_end).toBe(1000);
  });

  it("non-US country throws COUNTRY_SOURCE_MISSING, fallback sets country_fallback", () => {
    const src = atus2025Source([...many(40), ...weekendRows(40)]);
    const kr = { ...ATTRS, country: "KR" };
    try {
      selectDonors(kr, src, OPTS);
      expect.unreachable();
    } catch (e) {
      expect((e as TimelineError).code).toBe("COUNTRY_SOURCE_MISSING");
    }
    const r = selectDonors(kr, src, { ...OPTS, allowCountryFallback: true });
    expect(r.country_fallback).toBe(true);
    expect(selectDonors(ATTRS, src, OPTS).country_fallback).toBe(false);
  });

  it("early weekend donor is redrawn", () => {
    const src = atus2025Source([...many(40, { wake: 480, breakfast_time: 510 }), ...weekendRows(20, { wake: 300 }), ...weekendRows(20, { wake: 480, breakfast_time: 510 })]);
    let max = 0;
    for (let seed = 0; seed < 60; seed++) {
      const r = selectDonors(ATTRS, src, { seed, allowCountryFallback: false });
      max = Math.max(max, r.weekend_redraws);
      expect(r.weekend_redraws).toBeLessThanOrEqual(5);
      if (r.weekend_redraws < 5) {
        const wkRow = src.pool("weekend").find((x) => x.id === r.weekend.case_id)!;
        expect(wkRow.wake!).toBeGreaterThanOrEqual(480 - 120);
      }
    }
    expect(max).toBeGreaterThanOrEqual(1);
  });

  it("keeps the last pick with weekend_redraws=5 when every weekend donor is early", () => {
    const src = atus2025Source([...many(40, { wake: 480, breakfast_time: 510 }), ...weekendRows(40, { wake: 300 })]);
    const r = selectDonors(ATTRS, src, OPTS);
    expect(r.weekend_redraws).toBe(5);
  });

  it("empty pools throw DONOR_POOL_EMPTY with kind", () => {
    const noWeekend = atus2025Source(many(40));
    try {
      selectDonors(ATTRS, noWeekend, OPTS);
      expect.unreachable();
    } catch (e) {
      expect((e as TimelineError).code).toBe("DONOR_POOL_EMPTY");
      expect((e as TimelineError).details).toEqual({ kind: "weekend" });
    }
    try {
      selectDonors(ATTRS, atus2025Source(weekendRows(5)), OPTS);
      expect.unreachable();
    } catch (e) {
      expect((e as TimelineError).details).toEqual({ kind: "weekday" });
    }
  });
});

describe("rest-day weekend pool", () => {
  it("excludes worked rows", () => {
    const src = atus2025Source([
      ...many(40),
      ...weekendRows(3, { id: "rest" }),
      ...weekendRows(3, { id: "worked", worked: true }),
    ]);
    const ids = src.pool("weekend").map((r) => r.id);
    expect(ids).toHaveLength(3);
    expect(ids.every((id) => id === "rest")).toBe(true);
  });
});

describe("donor acceptance redraws", () => {
  it("redraws a first-draw bed of 1700 into range without adjustments", () => {
    const src = atus2025Source([
      ...many(40),
      ...weekendRows(1, { id: "late", bed: 1700 }),
      ...weekendRows(39, { id: "ok", bed: 1380 }),
    ]);
    let redrawn = 0;
    for (let seed = 0; seed < 200; seed++) {
      const r = selectDonors(ATTRS, src, { seed, allowCountryFallback: false });
      expect(r.weekend.profile.sleep).toBe(1380);
      expect(r.weekend.adjustments).toEqual([]);
      if (r.weekend.redraws > 0) redrawn++;
    }
    expect(redrawn).toBeGreaterThan(0);
  });

  it("acceptance redraws never land below the weekend wake floor", () => {
    const src = atus2025Source([
      ...many(40, { wake: 480, breakfast_time: 510 }),
      ...weekendRows(1, { wake: 300 }),
      ...weekendRows(1, { wake: 480, breakfast_time: 510, bed: 1700 }),
      ...weekendRows(38, { wake: 480, breakfast_time: 510 }),
    ]);
    let redrawn = 0;
    for (let seed = 0; seed < 200; seed++) {
      const r = selectDonors(ATTRS, src, { seed, allowCountryFallback: false });
      expect(r.weekend.profile.wake).toBeGreaterThanOrEqual(360);
      expect(r.weekend.profile.sleep).toBe(1380);
      expect(r.weekend_redraws).toBe(r.weekend.redraws);
      if (r.weekend.redraws > 0) redrawn++;
    }
    expect(redrawn).toBeGreaterThan(0);
  });

  it("never clamps the wake floor and still reports it", () => {
    const src = atus2025Source([...many(40, { wake: 480, breakfast_time: 510 }), ...weekendRows(40, { wake: 300 })]);
    const r = selectDonors(ATTRS, src, OPTS);
    expect(r.weekend.redraws).toBe(5);
    expect(r.weekend.profile.wake).toBe(300);
    expect(r.weekend.adjustments).toEqual([]);
  });

  it("the wake floor follows the final weekday donor, not the draw-0 one", () => {
    const src = atus2025Source([
      ...many(8, { wake: 700, breakfast_time: 720, bed: 1700 }),
      ...many(32, { wake: 420 }),
      ...weekendRows(40, { wake: 320, breakfast_time: 360 }),
    ]);
    let redrawnWeekday = 0;
    for (let seed = 0; seed < 100; seed++) {
      const r = selectDonors(ATTRS, src, { seed, allowCountryFallback: false });
      expect(r.weekday.profile.wake).toBe(420);
      expect(r.weekend.redraws).toBe(0);
      if (r.weekday.redraws > 0) redrawnWeekday++;
    }
    expect(redrawnWeekday).toBeGreaterThan(0);
  });

  it("clamps the last pick after 5 failed redraws and records it", () => {
    const src = atus2025Source([...many(40), ...weekendRows(40, { bed: 1700 })]);
    const r = selectDonors(ATTRS, src, OPTS);
    expect(r.weekend.redraws).toBe(5);
    expect(r.weekend.adjustments).toEqual([{ item: "sleep", from: 1700, to: 1679 }]);
    expect(r.weekend.profile.sleep).toBe(1679);
    expect(r.weekday.redraws).toBe(0);
  });

  it("uses the accept predicate and returns the pick when it never passes", () => {
    const src = atus2025Source([...many(40), ...weekendRows(40)]);
    const accept = (kind: "weekday" | "weekend") =>
      kind === "weekday" ? [{ item: "slots", bound: 0, value: 1 }] : [];
    const r = selectDonors(ATTRS, src, { ...OPTS, accept });
    expect(r.weekday.redraws).toBe(5);
    expect(r.weekday.adjustments).toEqual([]);
    expect(r.weekend.redraws).toBe(0);
  });

  it("is deterministic per seed", () => {
    const src = atus2025Source([...many(40), ...weekendRows(20, { bed: 1700 }), ...weekendRows(20)]);
    expect(selectDonors(ATTRS, src, OPTS)).toEqual(selectDonors(ATTRS, src, OPTS));
  });
});

describe("sport fill", () => {
  const sportRows = (n: number) => weekendRows(n, { sport_has: true, sport_time: 600 });

  it("fills from the level when it has at least 30 sport rows", () => {
    const src = atus2025Source([...many(40), ...sportRows(30), ...weekendRows(1, { w: 1e9 })]);
    const r = selectDonors(ATTRS, src, OPTS);
    expect(r.weekend.sport_fill).toBe("level");
    expect(r.weekend.profile.sport).toBe(600);
    expect(r.weekend.filled).toContain("sport_time");
  });

  it("falls back to the weekend pool below 30 level sport rows", () => {
    const other = OCCUPATION_CODES["service"]!;
    const src = atus2025Source([
      ...many(40),
      ...weekendRows(40, { w: 1e9 }),
      ...sportRows(29),
      ...weekendRows(40, { occ: other, sport_has: true, sport_time: 900 }),
    ]);
    // occupation level: 40 + 29 rows, only 29 with sport -> pool fill.
    const r = selectDonors(ATTRS, src, { ...OPTS, seed: 3 });
    expect(r.weekend.backoff_level).toBe("occupation");
    if (r.weekend.filled.includes("sport_time")) {
      expect(r.weekend.sport_fill).toBe("pool");
      expect([600, 900]).toContain(r.weekend.profile.sport);
    }
  });

  it("leaves sport undefined when the pool has none", () => {
    const r = selectDonors(ATTRS, atus2025Source([...many(40), ...weekendRows(40)]), OPTS);
    expect(r.weekend.profile.sport).toBeUndefined();
    expect(r.weekend.sport_fill).toBeUndefined();
    expect(r.weekend.filled).not.toContain("sport_time");
  });

  it("does not fill weekday picks and keeps own sport", () => {
    const src = atus2025Source([...many(40), ...sportRows(40)]);
    const r = selectDonors(ATTRS, src, OPTS);
    expect(r.weekday.profile.sport).toBeUndefined();
    expect(r.weekday.sport_fill).toBeUndefined();
    const own = atus2025Source([...many(40), ...weekendRows(40, { sport_has: true, sport_time: 500 })]);
    const o = selectDonors(ATTRS, own, OPTS);
    expect(o.weekend.sport_fill).toBeUndefined();
    expect(o.weekend.filled).not.toContain("sport_time");
  });
});

describe("donorBoundViolations", () => {
  const ok: DayProfile = {
    wake: 420, sleep: 1380, work_start: 540, work_end: 1020, breakfast: 450, lunch: 750, dinner: 1110, meal_loc: {},
  };
  it("is empty for a sane weekday", () => {
    expect(donorBoundViolations("weekday", ok, 45, 120)).toEqual([]);
  });
  it("reports each bound with item, bound and value", () => {
    expect(donorBoundViolations("weekday", { ...ok, wake: 200, sleep: 1700 }, 45, 120)).toEqual([
      { item: "wake", bound: 240, value: 200 },
      { item: "sleep", bound: 1679, value: 1700 },
    ]);
    expect(donorBoundViolations("weekday", { ...ok, breakfast: 425 }, 45, 120)).toEqual([
      { item: "breakfast", bound: 430, value: 425 },
    ]);
    expect(donorBoundViolations("weekday", { ...ok, dinner: 1330 }, 45, 120)).toEqual([
      { item: "dinner", bound: 1320, value: 1330 },
    ]);
    expect(donorBoundViolations("weekday", { ...ok, work_end: 700, lunch: 750 }, 45, 120)).toEqual([
      { item: "work_end", bound: 780, value: 700 },
      { item: "lunch", bound: 595, value: 750 },
    ]);
    expect(donorBoundViolations("weekday", { ...ok, lunch: 590 }, 45, 120)).toEqual([
      { item: "lunch", bound: 600, value: 590 },
    ]);
  });
  it("checks sport only on weekends", () => {
    const p = { ...ok, sport: 1250 };
    expect(donorBoundViolations("weekday", p, 45, 120)).toEqual([]);
    expect(donorBoundViolations("weekend", p, 45, 120)).toEqual([{ item: "sport", bound: 1200, value: 1250 }]);
  });
});

describe("atus2025Source", () => {
  it("exposes id and country", () => {
    const s = atus2025Source([]);
    expect(s.id).toBe("atus_2025");
    expect(s.country).toBe("US");
  });
});

describe("parseAtusPool", () => {
  const HEADER =
    "id,daytype,occ,w,wake,bed,work_start,work_end,work_min,commute_in,commute_out,worked,breakfast_has,breakfast_time,breakfast_loc,lunch_has,lunch_time,lunch_loc,dinner_has,dinner_time,dinner_loc,sport_has,sport_time,sport_outdoor,sport_min,house_min,leisure_min,social_out,away_all_day";

  it("reads the README header line", () => {
    const line =
      "syn-1,weekday,2,1500.5,420.0,1470.0,540.0,1020.0,480.0,10.0,10.0,True,True,450.0,1.0,False,725.0,2.0,True,1110.0,4.0,True,1200.0,True,45,20,60,False,False";
    const rows = parseAtusPool(`${HEADER}\n${line}\n`);
    expect(rows).toHaveLength(1);
    const r = rows[0]!;
    expect(r).toMatchObject({
      id: "syn-1",
      daytype: "weekday",
      occ: 2,
      w: 1500.5,
      wake: 420,
      bed: 1470,
      work_start: 540,
      work_end: 1020,
      work_min: 480,
      worked: true,
      breakfast_time: 450,
      breakfast_loc: 1,
      lunch_time: null,
      lunch_loc: null,
      dinner_time: 1110,
      dinner_loc: 4,
      sport_has: true,
      sport_time: 1200,
      sport_outdoor: true,
      house_min: 20,
      social_out: false,
    });
  });

  it("maps empty cells to null", () => {
    const line = "syn-2,weekend,3,10,,1400.0,,,,,,False,False,,,False,,,False,,,False,,,0,0,30,True,False";
    const r = parseAtusPool(`${HEADER}\n${line}`)[0]!;
    expect(r.wake).toBeNull();
    expect(r.work_start).toBeNull();
    expect(r.sport_outdoor).toBeNull();
    expect(r.social_out).toBe(true);
    expect(r.daytype).toBe("weekend");
  });

  it("throws naming a missing column", () => {
    const header = HEADER.replace(",lunch_loc", "");
    expect(() => parseAtusPool(`${header}\nx`)).toThrow(/lunch_loc/);
  });

  it("strips a UTF-8 BOM from the header", () => {
    const line = "syn-3,weekday,2,10,420.0,1380.0,540.0,1020.0,480.0,0,0,True,False,,,False,,,False,,,False,,,0,0,30,False,False";
    const rows = parseAtusPool(`\uFEFF${HEADER}\n${line}`);
    expect(rows[0]!.id).toBe("syn-3");
  });
});
