import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  ExceptionCatalogSchema,
  GapTableSchema,
  OCCUPATION_CODES,
  ScheduleDefaultsSchema,
  ScheduleSchema,
  TemplatesSchema,
  TimelineError,
  TimelineManifestSchema,
  atus2025Source,
  buildTimelineManifest,
  createTimeline,
  fillTimeline,
  generateDay,
  generateSchedule,
  hardFilterFailures,
  lateWeekdays,
  parseWorld,
  placeTrips,
  rankSelector,
  realizedOf,
  verifyTimeline,
  type Clip,
  type ClipCategory,
  type DonorRow,
  type GapTable,
  type Schedule,
  type ScheduleDay,
  type ScheduleInput,
  type Setting,
  type Timeline,
  type World,
} from "../src/index.js";
import { clip, genId, makeClips, memoryCatalog } from "./fixtures.js";

// Synthetic data only: one person ("p1") with NextMe-style day groups and a made-up 4-day trip.

const readConfig = (rel: string): unknown =>
  JSON.parse(readFileSync(new URL(`../../../config/${rel}`, import.meta.url), "utf8"));

const SEQUENCE = ["2026-05-07", "2026-05-08", "2026-05-09", "2026-05-10"]; // Thu-Sun
const TRIP_NODES = ["trip_1:lodging", "trip_1:outing", "trip_1:meal"];

type Over = { seed?: number; day?: number; slot?: number; late?: number; trips?: unknown; tripNode?: boolean; sequence?: string[] };

function worldJson(o: Over = {}): Record<string, unknown> {
  const tripNode = o.tripNode ?? true;
  return {
    person_id: "p-trip",
    attributes: { country: "US", occupation_group: "professional", work_pattern: "day_shift" },
    distribution_source: "atus_2025",
    nodes: {
      home: { kind: "home" },
      kitchen: { kind: "kitchen", default: "home" },
      workplace: { kind: "workplace", job: "clerk" },
      climbing: { kind: "hobby", activity: "climbing", setting: "venue_indoor", weekly_freq: 2, tags: ["climbing"] },
      ...(tripNode
        ? { trip_1: { kind: "trip", lodging_setting: "venue_indoor", outing_setting: "outdoor", meal_setting: "venue_indoor" } }
        : {}),
    },
    node_binding: {
      home: { dataset: ["nextme800"], person_source: "p1", node: "home" },
      workplace: { dataset: ["nextme800"], person_source: "p1", node: "work" },
      climbing: { dataset: ["ego4d"], tags_any: ["climbing"] },
      ...(tripNode
        ? { trip_1: { dataset: ["nextme800"], person_source: "p1", day_group_sequence: o.sequence ?? SEQUENCE } }
        : {}),
    },
    meal_style: {
      workday: { lunch: { workplace: 1 }, dinner: { home: 1 } },
      rest_day: { breakfast: { home: 1 }, lunch: { home: 1 }, dinner: { home: 1 } },
      breakfast_skip_prob: 0,
      place_nodes: {},
    },
    dials: {
      length_days: 28,
      routine_strength: 0.6,
      exception_rate: { slot: o.slot ?? 0, day: o.day ?? 0 },
      wake_sleep: "default_v1",
      ...(o.late !== undefined ? { late_workdays_per_week: o.late } : {}),
      ...(o.trips !== undefined ? { trips: o.trips } : tripNode ? { trips: [{ node: "trip_1", length_days: 4 }] } : {}),
    },
    seed: o.seed ?? 7,
  };
}

const makeWorld = (o: Over = {}): World => parseWorld(worldJson(o));

let counter = 0;
function row(over: Partial<DonorRow> = {}): DonorRow {
  counter += 1;
  return {
    id: `r${counter}`,
    daytype: "weekday",
    occ: OCCUPATION_CODES["professional"]!,
    w: 1 + (counter % 3),
    wake: 400 + (counter % 5) * 5,
    bed: 1370,
    work_start: 500 + (counter % 4) * 10,
    work_end: 1010,
    work_min: 480,
    worked: true,
    breakfast_time: 430,
    breakfast_loc: 1,
    lunch_time: 740,
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

function pool(): DonorRow[] {
  counter = 0;
  const weekday = Array.from({ length: 40 }, () => row());
  const weekend = Array.from({ length: 40 }, () =>
    row({ daytype: "weekend", worked: false, work_start: null, work_end: null, wake: 470, bed: 1390, breakfast_time: 500, lunch_time: 760, dinner_time: 1100 }),
  );
  const late = Array.from({ length: 40 }, () =>
    row({ work_start: 720, work_end: 1230, wake: 450, bed: 1400, breakfast_time: 480, lunch_time: 800, dinner_time: 1290 }),
  );
  return [...weekday, ...weekend, ...late];
}

function input(o: Over = {}): ScheduleInput {
  const world = makeWorld(o);
  return {
    world,
    templates: TemplatesSchema.parse(readConfig("timeline/templates.v3.json")),
    gaps: GapTableSchema.parse(readConfig("timeline/gaps.v1.json")) as GapTable,
    exceptions: ExceptionCatalogSchema.parse(readConfig("timeline/exceptions.v1.json")),
    source: atus2025Source(pool()),
    boundNodes: new Set(Object.keys(world.node_binding)),
    defaults: ScheduleDefaultsSchema.parse(readConfig("schedule_defaults_atus2025.json")),
  };
}

const tripDays = (s: Schedule): ScheduleDay[] => s.days.filter((d) => d.day_type === "trip_day");
const json = (x: unknown) => JSON.stringify(x, null, 2);

function errorOf(fn: () => unknown): TimelineError {
  try {
    fn();
  } catch (e) {
    if (e instanceof TimelineError) return e;
    throw e;
  }
  throw new Error("expected a TimelineError");
}

describe("trip worlds (N2)", () => {
  it("accepts a trip node, its binding's day_group_sequence and dials.trips; trips default to []", () => {
    const w = makeWorld();
    expect(w.nodes["trip_1"]).toEqual({ kind: "trip", lodging_setting: "venue_indoor", outing_setting: "outdoor", meal_setting: "venue_indoor" });
    expect(w.node_binding["trip_1"]!.day_group_sequence).toEqual(SEQUENCE);
    expect(w.dials.trips).toEqual([{ node: "trip_1", length_days: 4 }]);
    expect(makeWorld({ tripNode: false }).dials.trips).toEqual([]);
  });

  it("rejects a trip whose length differs from its binding's day_group_sequence", () => {
    const e = errorOf(() => makeWorld({ trips: [{ node: "trip_1", length_days: 3 }] }));
    expect(e.code).toBe("WORLD_INVALID");
    expect(e.message).toMatch(/day_group_sequence/);
  });

  it("rejects a trip dial naming a node that is not a trip node", () => {
    expect(errorOf(() => makeWorld({ trips: [{ node: "climbing", length_days: 4 }] })).code).toBe("WORLD_INVALID");
    expect(errorOf(() => makeWorld({ trips: [{ node: "nowhere", length_days: 4 }] })).code).toBe("WORLD_INVALID");
  });

  it("rejects a sequence entry that is not a YYYY-MM-DD date", () => {
    expect(errorOf(() => makeWorld({ sequence: ["2026-05-07", "2026-05-08", "may-9", "2026-05-10"] })).code).toBe("WORLD_INVALID");
  });
});

describe("trip placement (N3)", () => {
  it("a 4-day trip whose sequence starts on a Thursday lands on a synthetic Thursday-Sunday", () => {
    const schedule = generateSchedule(input());
    expect(ScheduleSchema.parse(schedule)).toEqual(schedule);
    const trip = tripDays(schedule);
    expect(trip.map((d) => d.weekday)).toEqual(["Thu", "Fri", "Sat", "Sun"]);
    const s = trip[0]!.day_index;
    expect(trip.map((d) => d.day_index)).toEqual([s, s + 1, s + 2, s + 3]);
    expect(trip.map((d) => d.trip)).toEqual([0, 1, 2, 3].map((index) => ({ node: "trip_1", index })));
    for (const d of schedule.days.filter((x) => x.day_type !== "trip_day")) expect(d).not.toHaveProperty("trip");
  });

  it("the start is drawn per seed among the Thursdays where the trip fits", () => {
    const starts = new Set<number>();
    for (const seed of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) {
      const trip = tripDays(generateSchedule(input({ seed })));
      expect(trip[0]!.weekday).toBe("Thu");
      expect(trip).toHaveLength(4);
      starts.add(trip[0]!.day_index);
    }
    // Mon start: the Thursdays are days 4, 11, 18 and 25, and all of them fit.
    for (const s of starts) expect([4, 11, 18, 25]).toContain(s);
    expect(starts.size).toBeGreaterThan(1);
    expect(json(generateSchedule(input({ seed: 3 })))).toBe(json(generateSchedule(input({ seed: 3 }))));
  });

  it("the exception record is repeated on each trip day and lists the replaced base day types", () => {
    const trip = tripDays(generateSchedule(input()));
    const s = trip[0]!.day_index;
    const record = {
      scope: "multi_day",
      type: "trip",
      replaces: ["workday", "workday", "hobby_day", "hobby_day"],
      day_range: [s, s + 3],
      node: "trip_1",
    };
    for (const d of trip) expect(d.exceptions).toEqual([record]);
  });

  it("trip-day slots use virtual trip nodes with the trip's settings", () => {
    const trip = tripDays(generateSchedule(input()));
    const settings: Record<string, Setting> = {
      "trip_1:lodging": "venue_indoor",
      "trip_1:outing": "outdoor",
      "trip_1:meal": "venue_indoor",
    };
    for (const d of trip) {
      const roles = d.slots.map((s) => s.role);
      expect(roles).toContain("lodging_morning");
      expect(roles).toContain("outing");
      expect(d.slots[0]!.role).toBe("lodging_morning");
      for (const s of d.slots) {
        expect(TRIP_NODES).toContain(s.spec.node);
        expect(s.spec.setting).toBe(settings[s.spec.node]);
      }
      expect(d.slots.find((s) => s.role === "lodging_morning")!.spec.node).toBe("trip_1:lodging");
      expect(d.slots.find((s) => s.role === "outing")!.spec.node).toBe("trip_1:outing");
      for (const role of ["breakfast", "lunch", "dinner"]) {
        const slot = d.slots.find((s) => s.role === role);
        if (slot) expect(slot.spec.node).toBe("trip_1:meal");
      }
    }
  });

  it("no day or slot exception lands on trip days", () => {
    const schedule = generateSchedule(input({ day: 1, slot: 1 }));
    const trip = tripDays(schedule);
    expect(trip).toHaveLength(4);
    for (const d of trip) {
      expect(d.exceptions.map((e) => e.scope)).toEqual(["multi_day"]);
      expect(d.slot_exception_eligible).toBe(false);
      expect(d.slot_exception_candidates).toBe(0);
    }
    // Every workday outside the trip still takes the day exception at rate 1.
    expect(schedule.days.filter((d) => d.day_type === "workday")).toHaveLength(0);
    expect(schedule.days.filter((d) => d.day_type === "leave_day")).toHaveLength(18);
  });

  it("trips override late workdays: late days = late_workdays_per_week x 4 minus the late weekdays the trip covers", () => {
    let coveredLate = 0;
    for (const seed of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) {
      const o = input({ seed, late: 2 });
      const schedule = generateSchedule(o);
      const late = lateWeekdays(o.world);
      const trip = tripDays(schedule);
      const covered = trip.filter((d) => late.has(d.weekday)).length;
      coveredLate += covered;
      expect(schedule.days.filter((d) => d.day_type === "workday_late")).toHaveLength(8 - covered);
      expect(schedule.realized.late_workday_count).toBe(8 - covered);
      expect(trip[0]!.exceptions[0]!.replaces).toEqual(
        trip.map((d) => (d.weekday === "Sat" || d.weekday === "Sun" ? "hobby_day" : late.has(d.weekday) ? "workday_late" : "workday")),
      );
    }
    expect(coveredLate).toBeGreaterThan(0);
  });

  it("a regenerated trip day keeps its trip, record and virtual nodes", () => {
    const o = input();
    const schedule = generateSchedule(o);
    const first = tripDays(schedule)[1]!;
    const again = generateDay(o, schedule.donors, first.day_index, 1);
    expect(again.day_type).toBe("trip_day");
    expect(again.trip).toEqual(first.trip);
    expect(again.exceptions).toEqual(first.exceptions);
    for (const s of again.slots) expect(TRIP_NODES).toContain(s.spec.node);
  });

  it("realized has trip_day presence and consistency; day_exception_rate ignores trip days", () => {
    const o = input({ day: 1 });
    const schedule = generateSchedule(o);
    expect(schedule.realized.day_exception_rate).toBe(1);
    expect(schedule.realized.composition_consistency.trip_day?.pairs).toBe(6);
    expect(realizedOf(schedule.days, o.world, o.templates)).toEqual(schedule.realized);
  });

  it("worlds without trips give byte-identical schedules, with or without a trip node", () => {
    const plain = json(generateSchedule(input({ tripNode: false })));
    expect(json(generateSchedule(input({ trips: [] })))).toBe(plain);
    expect(plain).not.toMatch(/trip_day|multi_day|"trip"/);
  });
});

/** Synthetic trip clips for each sequence date: lodging, outing and meal spans of one video per date. */
function tripClips(dates: readonly string[] = SEQUENCE): Clip[] {
  const parts: { key: string; category: ClipCategory; setting: Setting; node: string; activity: string }[] = [
    { key: "lodging-am", category: "home", setting: "venue_indoor", node: "trip:lodging", activity: "packing in a hotel room" },
    { key: "breakfast", category: "meal", setting: "venue_indoor", node: "trip:meal", activity: "eating breakfast" },
    { key: "outing", category: "hobby", setting: "outdoor", node: "trip:outing", activity: "walking to a waterfall" },
    { key: "lunch", category: "meal", setting: "venue_indoor", node: "trip:meal", activity: "eating lunch" },
    { key: "outing-pm", category: "hobby", setting: "outdoor", node: "trip:outing", activity: "scenic trail" },
    { key: "dinner", category: "meal", setting: "venue_indoor", node: "trip:meal", activity: "eating dinner" },
    { key: "lodging-pm", category: "home", setting: "venue_indoor", node: "trip:lodging", activity: "resting in the hotel" },
  ];
  return dates.flatMap((date) =>
    parts.map((p, i) =>
      clip({
        clip_id: `${date}-${p.key}`,
        category: p.category,
        setting: p.setting,
        node: p.node,
        activity: p.activity,
        time_of_day: "unknown",
        source: { dataset: "nextme800", video_id: `nm-${date}`, media: { kind: "frames", approx_fps: 1, tar: `${date}.tar` } },
        source_start_s: i * 3600,
        source_end_s: i * 3600 + 300,
        day_group: date,
        person_source: "p1",
      }),
    ),
  );
}

const slotOf = (day: ScheduleDay, role: string) => day.slots.find((s) => s.role === role)!;

describe("fixed trip start (B1)", () => {
  const MON_WED = ["2026-05-04", "2026-05-05", "2026-05-06"]; // Mon-Wed
  const fixed = (o: Over = {}) =>
    input({ sequence: MON_WED, trips: [{ node: "trip_1", length_days: 3, start_day: 19 }], ...o });

  it("start_day 19 with a Mon-Wed sequence places the trip on days 19-21 (Fri-Sun), whatever the seed", () => {
    for (const seed of [1, 2, 3]) {
      const trip = tripDays(generateSchedule(fixed({ seed })));
      expect(trip.map((d) => d.day_index)).toEqual([19, 20, 21]);
      expect(trip.map((d) => d.weekday)).toEqual(["Fri", "Sat", "Sun"]);
    }
  });

  it("the exception record has day_range [19, 21] and replaces lists the base day types", () => {
    const trip = tripDays(generateSchedule(fixed()));
    for (const d of trip) {
      expect(d.exceptions).toEqual([
        { scope: "multi_day", type: "trip", replaces: ["workday", "hobby_day", "hobby_day"], day_range: [19, 21], node: "trip_1" },
      ]);
    }
  });

  it("rejects a start_day that does not fit the world or is not a positive integer", () => {
    for (const start_day of [27, 29, 0, 1.5]) {
      const e = errorOf(() => makeWorld({ sequence: MON_WED, trips: [{ node: "trip_1", length_days: 3, start_day }] }));
      expect(e.code).toBe("WORLD_INVALID");
    }
    expect(makeWorld({ sequence: MON_WED, trips: [{ node: "trip_1", length_days: 3, start_day: 26 }] }).dials.trips[0]!.start_day).toBe(26);
  });

  it("rejects two fixed trips that overlap", () => {
    const e = errorOf(() =>
      parseWorld(twoTrips([{ node: "trip_1", length_days: 3, start_day: 5 }, { node: "trip_2", length_days: 3, start_day: 7 }])),
    );
    expect(e.code).toBe("WORLD_INVALID");
    expect(e.message).toMatch(/overlap/);
    expect(parseWorld(twoTrips([{ node: "trip_1", length_days: 3, start_day: 5 }, { node: "trip_2", length_days: 3, start_day: 8 }])).dials.trips).toHaveLength(2);
  });

  it("an unfixed trip listed before a fixed one avoids it", () => {
    for (const seed of [1, 2, 3, 4, 5, 6]) {
      const w = parseWorld({ ...twoTrips([{ node: "trip_1", length_days: 3 }, { node: "trip_2", length_days: 3, start_day: 5 }]), seed });
      const placed = placeTrips(w);
      expect(placed.map((p) => p.node)).toEqual(["trip_1", "trip_2"]);
      expect(placed[1]!.start).toBe(5);
      expect(placed[0]!.start + 2 < 5 || placed[0]!.start > 7).toBe(true);
    }
  });

  it("a world without start_day echoes no start_day", () => {
    expect(makeWorld().dials.trips[0]).not.toHaveProperty("start_day");
  });
});

function twoTrips(trips: unknown): Record<string, unknown> {
  const w = worldJson({ sequence: ["2026-05-04", "2026-05-05", "2026-05-06"], trips }) as Record<string, any>;
  w.nodes.trip_2 = { ...w.nodes.trip_1 };
  w.node_binding.trip_2 = { ...w.node_binding.trip_1 };
  return w;
}

describe("trip matching (N5)", () => {
  const o = input();
  const schedule = generateSchedule(o);
  const t = createTimeline(o.world, schedule, genId);
  const trip = tripDays(schedule);
  const catalog = memoryCatalog([...tripClips(), ...tripClips(["2026-05-11"])]);

  it("trip_sequence accepts only the sequence date at the day's index", () => {
    const day1 = trip[1]!; // index 1: 2026-05-08
    const slot = slotOf(day1, "lodging_morning");
    expect(hardFilterFailures(t, catalog, slot.slot_id, catalog.getClip("2026-05-08-lodging-am")!, [])).toEqual([]);
    expect(hardFilterFailures(t, catalog, slot.slot_id, catalog.getClip("2026-05-07-lodging-am")!, [])).toContain("trip_sequence");
    const relaxed = hardFilterFailures(t, catalog, slot.slot_id, catalog.getClip("2026-05-09-lodging-am")!, ["ignore_tags", "widen_window_30m"]);
    expect(relaxed).toEqual(["trip_sequence"]);
  });

  it("a clip from outside the sequence fails the binding and trip_sequence", () => {
    const slot = slotOf(trip[0]!, "outing");
    const failed = hardFilterFailures(t, catalog, slot.slot_id, catalog.getClip("2026-05-11-outing")!, []);
    expect(failed).toContain("category_setting_node");
    expect(failed).toContain("trip_sequence");
  });

  it("a virtual node only takes clips with its fixed trip label", () => {
    const slot = slotOf(trip[0]!, "outing");
    expect(hardFilterFailures(t, catalog, slot.slot_id, catalog.getClip("2026-05-07-outing")!, [])).toEqual([]);
    const lodging = catalog.getClip("2026-05-07-lodging-am")!;
    const asOuting = { ...lodging, clip_id: "x", category: "hobby" as const, setting: "outdoor" as const };
    expect(hardFilterFailures(t, memoryCatalog([asOuting]), slot.slot_id, asOuting, [])).toContain("category_setting_node");
  });
});

describe("verifier trip rule (N5)", () => {
  const o = input();
  const schedule = generateSchedule(o);
  const trip = tripDays(schedule);
  const clips = tripClips();
  const roleClip: Record<string, string> = {
    lodging_morning: "lodging-am",
    breakfast: "breakfast",
    outing: "outing",
    lunch: "lunch",
    outing_pm: "outing-pm",
    dinner: "dinner",
    lodging_evening: "lodging-pm",
  };
  /** Only the trip days are filled, each slot with its sequence date's clip. */
  function filledTrip(): Timeline {
    const t0 = createTimeline(o.world, schedule, genId);
    const slots = { ...t0.slots };
    trip.forEach((day, k) => {
      for (const s of day.slots) slots[s.slot_id] = { clip_ids: [`${SEQUENCE[k]}-${roleClip[s.role]}`], relaxations: [] };
    });
    return { ...t0, slots };
  }

  it("passes when every trip-day clip comes from the trip's nodes and its sequence date", () => {
    const r = verifyTimeline(filledTrip(), memoryCatalog(clips));
    expect(r.rules.trip).toEqual({ pass: true, violations: [] });
    expect(r.rules.exceptions).toEqual({ pass: true, violations: [] });
    expect(r.rules.day_type).toEqual({ pass: true, violations: [] });
  });

  it("catches a home clip on a trip day", () => {
    const home = clip({ clip_id: "home-1", day_group: "2026-04-01", source_end_s: 300, time_of_day: "unknown" });
    const t = filledTrip();
    const slot = slotOf(trip[2]!, "lodging_morning");
    const bad = { ...t, slots: { ...t.slots, [slot.slot_id]: { clip_ids: ["home-1"], relaxations: [] } } };
    const r = verifyTimeline(bad, memoryCatalog([...clips, home]));
    expect(r.rules.trip.pass).toBe(false);
    expect(r.rules.trip.violations.some((v) => v.slot_id === slot.slot_id && /home/.test(v.detail))).toBe(true);
  });

  it("catches a clip from the wrong sequence date", () => {
    const t = filledTrip();
    const slot = slotOf(trip[0]!, "outing");
    const bad = { ...t, slots: { ...t.slots, [slot.slot_id]: { clip_ids: ["2026-05-08-outing"], relaxations: [] } } };
    const r = verifyTimeline(bad, memoryCatalog(clips));
    expect(r.rules.trip.pass).toBe(false);
    expect(r.rules.trip.violations.some((v) => v.slot_id === slot.slot_id && /2026-05-07/.test(v.detail))).toBe(true);
  });

  it("catches a home slot on a trip day and a trip record that differs between trip days", () => {
    const t = filledTrip();
    const d0 = trip[0]!;
    const withHomeSlot = {
      ...t,
      schedule: {
        ...t.schedule,
        days: t.schedule.days.map((d) =>
          d.day_index === d0.day_index
            ? { ...d, slots: d.slots.map((s, i) => (i === 0 ? { ...s, spec: { ...s.spec, node: "home", setting: "home_indoor" as const } } : s)) }
            : d.day_index === d0.day_index + 1
              ? { ...d, exceptions: [{ ...d.exceptions[0]!, day_range: [d0.day_index + 1, d0.day_index + 3] as [number, number] }] }
              : d,
        ),
      },
    };
    const r = verifyTimeline(withHomeSlot, memoryCatalog(clips));
    expect(r.rules.trip.violations.some((v) => v.day_index === d0.day_index && /home/.test(v.detail))).toBe(true);
    expect(r.rules.exceptions.pass).toBe(false);
  });
});

describe("trip end to end", () => {
  it("fills, verifies and keeps the multi_day record in the manifest", () => {
    const o = input();
    const schedule = generateSchedule(o);
    const catalog = memoryCatalog([...makeClips({ groups: 30, hobbyVideos: 12, durationS: 600 }), ...tripClips()]);
    const t = fillTimeline(createTimeline(o.world, schedule, genId), catalog, rankSelector, (d, a) =>
      generateDay(o, schedule.donors, d, a),
    );
    const report = verifyTimeline(t, catalog);
    expect(report.rules.trip).toEqual({ pass: true, violations: [] });
    const manifest = TimelineManifestSchema.parse(buildTimelineManifest(t, catalog, new Date(0), o.templates));
    const tripManifestDays = manifest.days.filter((d) => d.day_type === "trip_day");
    expect(tripManifestDays).toHaveLength(4);
    tripManifestDays.forEach((d, k) => {
      expect(d.exceptions[0]).toMatchObject({ scope: "multi_day", type: "trip", node: "trip_1" });
      expect(d.trip).toEqual({ node: "trip_1", index: k });
      for (const c of d.clips) expect(c.source.video_id).toBe(`nm-${SEQUENCE[k]}`);
    });
  });
});
