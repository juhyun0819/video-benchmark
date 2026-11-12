import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  TimelineError,
  TimelineManifestSchema,
  buildTimelineManifest,
  createTimeline,
  type Annotation,
  type Clip,
  type ClipCatalog,
  type Timeline,
  TemplatesSchema,
  realizedOf,
} from "../src/index.js";
import { clip, genId, makeSchedule, makeWorld, type SlotDef } from "./fixtures.js";

const TEMPLATES = TemplatesSchema.parse(
  JSON.parse(readFileSync(new URL("../../../config/timeline/templates.v3.json", import.meta.url), "utf8")),
);
const HOUR = 3600;
const DEFS: SlotDef[] = [
  { role: "home_morning", category: "home", setting: "home_indoor", node: "home", activity: "getting ready", required: true, window: ["07:00", "08:00"] },
  { role: "work_am", category: "work", setting: "venue_indoor", node: "workplace", activity: "work", required: true, window: ["09:00", "12:00"] },
  { role: "lunch", category: "meal", setting: "venue_indoor", node: "workplace", activity: "eating lunch", required: false, window: ["12:00", "13:00"] },
  { role: "work_pm", category: "work", setting: "venue_indoor", node: "workplace", activity: "work", required: true, window: ["13:00", "17:00"] },
];

const mk = (id: string, over: Partial<Clip>): Clip =>
  clip({ clip_id: id, time_of_day: "unknown", source_end_s: HOUR, ...over });
const KINDS = ["home", "am", "lunch", "pm"];
const clipsFor = (d: number): Clip[] => [
  mk(`d${d}-home`, { day_group: `g${d}` }),
  mk(`d${d}-am`, { category: "work", setting: "venue_indoor", node: "work", day_group: `g${d}` }),
  mk(`d${d}-lunch`, { category: "meal", setting: "venue_indoor", node: "work", day_group: `g${d}` }),
  mk(`d${d}-pm`, { category: "work", setting: "venue_indoor", node: "work", day_group: `g${d}` }),
];

function build(days: number): { t: Timeline; clips: Clip[] } {
  const t0 = createTimeline(makeWorld(), makeSchedule(days, () => DEFS), genId);
  const slots = { ...t0.slots };
  const clips: Clip[] = [];
  for (let d = 1; d <= days; d++) {
    KINDS.forEach((k, i) => {
      slots[`d${d}-s${i + 1}`] = { clip_ids: [`d${d}-${k}`], relaxations: [] };
    });
    clips.push(...clipsFor(d));
  }
  return { t: { ...t0, slots }, clips };
}

const ann = (id: string, start: number, end: number, text = id): Annotation => ({
  id,
  track: "captions",
  level: "L1",
  time: { kind: "interval", start, end },
  text,
  original: {},
});

function catalogOf(clips: Clip[], annotations: Record<string, Annotation[]> = {}): ClipCatalog {
  const byId = new Map(clips.map((c) => [c.clip_id, c]));
  return {
    clips: () => clips,
    getClip: (id) => byId.get(id),
    getAnnotations: (_d, videoId) => annotations[videoId] ?? [],
  };
}

const NOW = new Date("2026-11-12T00:00:00.000Z");
const SNAPSHOT = join(dirname(fileURLToPath(import.meta.url)), "__snapshots__", "timeline.manifest.json");

describe("buildTimelineManifest", () => {
  it("golden manifest for fixture", () => {
    const { t, clips } = build(2);
    const annotations: Record<string, Annotation[]> = {
      "vid-d1-home": [ann("a1", 0, 30), ann("a2", 100, 200.123456)],
      "vid-d2-am": [ann("b1", 10, 20)],
    };
    const m = buildTimelineManifest(t, catalogOf(clips, annotations), NOW, TEMPLATES);
    expect(TimelineManifestSchema.parse(m)).toEqual(m);
    expect(m.created_at).toBe("2026-11-12T00:00:00.000Z");
    expect(m.schema_version).toBe(2);
    expect(m.versions).toMatchObject({ schema: 2, generator: "schedule.v2" });
    expect(m.days.map((d) => d.clips.length)).toEqual([4, 4]);
    const text = JSON.stringify(m, null, 2) + "\n";
    if (process.env.UPDATE_GOLDEN === "1") {
      mkdirSync(dirname(SNAPSHOT), { recursive: true });
      writeFileSync(SNAPSHOT, text);
    }
    expect(existsSync(SNAPSHOT), `${SNAPSHOT} is missing; run with UPDATE_GOLDEN=1 to write it`).toBe(true);
    expect(text).toBe(readFileSync(SNAPSHOT, "utf8"));
  });

  it("caption clipped at clip edge keeps clipped flag", () => {
    const { t, clips } = build(1);
    clips[0] = mk("d1-home", { day_group: "g1", source_start_s: 100, source_end_s: 100 + HOUR });
    const m = buildTimelineManifest(t, catalogOf(clips, { "vid-d1-home": [ann("edge", 50, 200)] }), NOW, TEMPLATES);
    const cap = m.captions.find((c) => c.provenance.orig_id === "edge");
    expect(cap?.clipped).toEqual({ start: true });
    const first = m.days[0]!.clips[0]!;
    expect(cap?.block_id).toBe(`clip:${first.slot_id}:d1-home`);
    expect(cap?.time).toEqual({ kind: "interval", start: first.synthetic_start_s, end: Math.round((first.synthetic_start_s + 100) * 1000) / 1000 });
  });

  it("lists the roles of slots dropped at matching time as dropped_optional (R48)", () => {
    const { t, clips } = build(2);
    t.slots["d2-s3"] = { clip_ids: [], relaxations: ["dropped_optional"] };
    const m = buildTimelineManifest(t, catalogOf(clips.filter((c) => c.clip_id !== "d2-lunch")), NOW, TEMPLATES);
    expect(TimelineManifestSchema.parse(m)).toEqual(m);
    expect(m.days.map((d) => d.dropped_optional)).toEqual([[], ["lunch"]]);
    expect(m.days[1]!.clips.map((c) => c.slot_id)).toEqual(["d2-s1", "d2-s2", "d2-s4"]);
  });

  it("catalog missing two clips -> MISSING_CLIPS lists both", () => {
    const { t, clips } = build(1);
    const rest = clips.filter((c) => c.clip_id !== "d1-pm" && c.clip_id !== "d1-am");
    try {
      buildTimelineManifest(t, catalogOf(rest), NOW, TEMPLATES);
      expect.unreachable();
    } catch (e) {
      expect(e).toBeInstanceOf(TimelineError);
      expect((e as TimelineError).code).toBe("MISSING_CLIPS");
      expect((e as TimelineError).details["ids"]).toEqual(["d1-am", "d1-pm"]);
    }
  });

  it("failing verifier -> VERIFY_FAILED", () => {
    const { t, clips } = build(1);
    const bad: Timeline = { ...t, slots: { ...t.slots, "d1-s2": { clip_ids: ["d1-home"], relaxations: [] } } };
    try {
      buildTimelineManifest(bad, catalogOf(clips), NOW, TEMPLATES);
      expect.unreachable();
    } catch (e) {
      expect((e as TimelineError).code).toBe("VERIFY_FAILED");
      expect((e as TimelineError).details["failed_rules"]).toContain("clip_unique");
    }
  });

  it("coverage_ratio = 0.25 for 4 h of clips over 16 h awake", () => {
    const { t, clips } = build(1);
    const m = buildTimelineManifest(t, catalogOf(clips), NOW, TEMPLATES);
    expect(m.days[0]!.wake).toBe("07:00");
    expect(m.days[0]!.sleep).toBe("23:00");
    expect(m.days[0]!.coverage_ratio).toBe(0.25);
  });

  it("no_video_spans plus clips tile [wake, sleep] on every day, with all three kinds present (E9)", () => {
    const { t, clips } = build(2);
    // Day 1: a 30 min transition after the home -> work node change; day 2 has none.
    t.schedule.days[0]!.transitions = [{ from_slot: "d1-s1", to_slot: "d1-s2", minutes: 30 }];
    const m = buildTimelineManifest(t, catalogOf(clips), NOW, TEMPLATES);
    expect(TimelineManifestSchema.parse(m)).toEqual(m);
    const kinds = new Set<string>();
    for (const day of m.days) {
      const base = (day.day_index - 1) * 86400;
      const hm = (x: string) => base + Number(x.slice(0, 2)) * 3600 + Number(x.slice(3)) * 60;
      const parts = [
        ...day.clips.map((c) => ({ start: c.synthetic_start_s, end: c.synthetic_end_s })),
        ...day.no_video_spans,
      ].sort((a, b) => a.start - b.start);
      expect(parts[0]!.start).toBe(hm(day.wake));
      expect(parts[parts.length - 1]!.end).toBe(hm(day.sleep));
      for (let i = 1; i < parts.length; i++) expect(parts[i]!.start).toBe(parts[i - 1]!.end);
      expect(day.no_video_spans.every((s) => s.end > s.start)).toBe(true);
      expect(day.no_video_spans.map((s) => s.start)).toEqual(day.no_video_spans.map((s) => s.start).sort((a, b) => a - b));
      day.no_video_spans.forEach((s) => kinds.add(s.kind));
    }
    expect([...kinds].sort()).toEqual(["transition", "unrecorded", "unscheduled"]);
    const d1 = m.days[0]!.no_video_spans;
    // Window s1 ends 08:00, s2 starts 09:00: 30 min unscheduled at s1's place, then the 30 min
    // transition right before s2 (G23, CR6.3).
    expect(d1.filter((s) => s.start >= 8 * HOUR && s.end <= 9 * HOUR)).toEqual([
      { start: 8 * HOUR, end: 8.5 * HOUR, kind: "unscheduled" },
      { start: 8.5 * HOUR, end: 9 * HOUR, kind: "transition" },
    ]);
    // Day 2 has no transition record: the whole 08:00-09:00 gap is unscheduled, and 17:00-23:00 trails.
    const d2 = m.days[1]!.no_video_spans;
    expect(d2.some((s) => s.kind === "transition")).toBe(false);
    expect(d2[d2.length - 1]).toEqual({ start: 86400 + 17 * HOUR, end: 86400 + 23 * HOUR, kind: "unscheduled" });
    // Work window 09:00-12:00 holds a 1 h clip: 2 h unrecorded.
    const unrec = d1.filter((s) => s.kind === "unrecorded" && s.start >= 9 * HOUR && s.end <= 12 * HOUR);
    expect(unrec.reduce((a, s) => a + s.end - s.start, 0)).toBe(2 * HOUR);
  });

  it("copies anchor adjustments, jitter redraws and shrunk roles; donor adjustments at top level", () => {
    const { t, clips } = build(1);
    const day = t.schedule.days[0]!;
    day.anchor_adjustments = [{ item: "work_start", from: 540, to: 545 }];
    day.jitter_redraws = 3;
    day.dropped_slots = [
      { role: "lunch", reason: "shrunk" },
      { role: "gone", reason: "shrunk" },
      { role: "nope", reason: "no_time" },
    ];
    t.schedule.donors.weekend = { ...t.schedule.donors.weekend, sport_fill: "pool", redraws: 2, adjustments: [{ item: "sport", from: 1, to: 2 }] };
    const m = buildTimelineManifest(t, catalogOf(clips), NOW, TEMPLATES);
    expect(m.days[0]!.anchor_adjustments).toEqual([{ item: "work_start", from: 540, to: 545 }]);
    expect(m.days[0]!.jitter_redraws).toBe(3);
    expect(m.days[0]!.shrunk).toEqual(["lunch"]);
    expect(m.donor_adjustments.weekday).toEqual({ case_id: "synthetic", backoff_level: "all", redraws: 0, adjustments: [], filled: [] });
    expect(m.donor_adjustments.weekend).toEqual({
      case_id: "synthetic",
      backoff_level: "all",
      redraws: 2,
      adjustments: [{ item: "sport", from: 1, to: 2 }],
      filled: [],
      sport_fill: "pool",
    });
  });

  it("realized.matching_dropped_slots counts dropped_optional slots, by role", () => {
    const { t, clips } = build(3);
    t.slots["d2-s3"] = { clip_ids: [], relaxations: ["dropped_optional"] };
    t.slots["d3-s3"] = { clip_ids: [], relaxations: ["dropped_optional"] };
    const open = clips.filter((c) => c.clip_id !== "d2-lunch" && c.clip_id !== "d3-lunch");
    const m = buildTimelineManifest(t, catalogOf(open), NOW, TEMPLATES);
    expect(m.realized.matching_dropped_slots).toBe(2);
    expect(m.realized.matching_dropped_by_role).toEqual({ lunch: 2 });
    expect(m.realized.dials.routine_strength).toBe(t.world.dials.routine_strength);
    const none = buildTimelineManifest(build(1).t, catalogOf(build(1).clips), NOW, TEMPLATES);
    expect(none.realized.matching_dropped_slots).toBe(0);
    expect(none.realized.matching_dropped_by_role).toEqual({});
  });

  it("realized.dropped_required counts dropped_required slots by role", () => {
    const { t, clips } = build(3);
    const t2 = {
      ...t,
      world: { ...t.world, dials: { ...t.world.dials, missing_required_allowance: [{ role: "home_morning", max_days: 1 }] } },
      slots: { ...t.slots, "d2-s1": { clip_ids: [], relaxations: ["dropped_required" as const] } },
    };
    const m = buildTimelineManifest(t2, catalogOf(clips.filter((c) => c.clip_id !== "d2-home")), NOW, TEMPLATES);
    expect(TimelineManifestSchema.parse(m)).toEqual(m);
    expect(m.realized.dropped_required).toEqual({ home_morning: 1 });
  });

  it("days carry slot_exception_eligible/candidates and slot windows, so realized recounts from the manifest alone (G24)", () => {
    const { t, clips } = build(3);
    t.schedule.days[1] = { ...t.schedule.days[1]!, slot_exception_eligible: true, slot_exception_candidates: 2 };
    t.schedule.days[2] = { ...t.schedule.days[2]!, slot_exception_eligible: true, slot_exception_candidates: 1 };
    t.slots["d3-s3"] = { clip_ids: [], relaxations: ["dropped_optional"] };
    const m = buildTimelineManifest(t, catalogOf(clips.filter((c) => c.clip_id !== "d3-lunch")), NOW, TEMPLATES);
    expect(TimelineManifestSchema.parse(m)).toEqual(m);
    expect(m.days[1]).toMatchObject({ slot_exception_eligible: true, slot_exception_candidates: 2 });
    expect(m.days[0]!.slots).toEqual(
      t.schedule.days[0]!.slots.map(({ slot_id, role, required, window }) => ({ slot_id, role, required, window })),
    );
    const { matching_dropped_slots: _n, matching_dropped_by_role: _r, dropped_required: _d, ...realized } = m.realized;
    expect(realizedOf(m.days, m.inputs.world, TEMPLATES)).toEqual(realized);
    expect(realized.slot_exception_eligible_days).toBe(2);
    expect(realized.slot_exception_share).toBe(0);
  });
});
