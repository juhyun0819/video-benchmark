import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  RULE_IDS,
  TemplatesSchema,
  TimelineError,
  TimelineManifestSchema,
  WorldSchema,
  buildTimelineManifest,
  createTimeline,
  layoutBlocks,
  transitionBlocks,
  verifyTimeline,
  type Annotation,
  type BlockPlacement,
  type BlockPlanEntry,
  type Clip,
  type ClipCatalog,
  type Timeline,
} from "../src/index.js";
import { clip, genId, makeSchedule, makeWorld, memoryCatalog, type SlotDef } from "./fixtures.js";

// Synthetic data only: one person ("p1"), two workdays, one real-like date ("g1", "g2") per day.

const TEMPLATES = TemplatesSchema.parse(
  JSON.parse(readFileSync(new URL("../../../config/timeline/templates.v3.json", import.meta.url), "utf8")),
);
const NOW = new Date("2026-11-13T00:00:00.000Z");
const H = 3600;
const M = 60;

const DEFS: SlotDef[] = [
  { role: "home_morning", category: "home", setting: "home_indoor", node: "home", activity: "getting ready", required: true, window: ["07:00", "08:00"] },
  { role: "work_am", category: "work", setting: "venue_indoor", node: "workplace", activity: "work", required: true, window: ["09:00", "12:00"] },
  { role: "lunch", category: "meal", setting: "venue_indoor", node: "workplace", activity: "eating lunch", required: false, window: ["12:00", "13:00"] },
  { role: "work_pm", category: "work", setting: "venue_indoor", node: "workplace", activity: "work", required: true, window: ["13:00", "17:00"] },
];

/** A clip of real-like date `g` (one video per date), source span [start, end). */
const at = (id: string, g: string, start: number, end: number, over: Partial<Clip> = {}): Clip =>
  clip({
    clip_id: id,
    time_of_day: "unknown",
    day_group: g,
    source: { dataset: "nextme800", video_id: `nm-${g}`, media: { kind: "frames", approx_fps: 1, tar: `${g}.tar` } },
    source_start_s: start,
    source_end_s: end,
    ...over,
  });
const work = { category: "work", setting: "venue_indoor", node: "work" } as const;
const transit = { category: "transit", setting: "outdoor", node: "transit" } as const;

const CLIPS: Clip[] = [
  at("g1-h1", "g1", 0, 40 * M),
  at("g1-h2", "g1", 40 * M, 80 * M),
  at("g1-h3", "g1", 80 * M, 90 * M),
  at("g1-bus", "g1", 2 * H, 2 * H + 45 * M, transit),
  at("g1-w1", "g1", 3 * H, 5 * H, work),
  at("g2-h1", "g2", 0, 30 * M),
  at("g2-w1", "g2", 3 * H, 4 * H, work),
];

/** Two workdays; day 1 has a 30 min transition from home_morning to work_am (08:30-09:00). */
function timeline(): Timeline {
  const t = createTimeline(makeWorld(), makeSchedule(2, () => DEFS), genId);
  t.schedule.days[0]!.transitions = [{ from_slot: "d1-s1", to_slot: "d1-s2", minutes: 30 }];
  return t;
}

const e = (slot_id: string, c: string, source_start_s?: number, source_end_s?: number): BlockPlanEntry => {
  const k = CLIPS.find((x) => x.clip_id === c)!;
  return { slot_id, clip_id: c, source_start_s: source_start_s ?? k.source_start_s, source_end_s: source_end_s ?? k.source_end_s };
};

const PLAN: BlockPlanEntry[] = [
  e("d1-s1", "g1-h1"),
  e("d1-s1", "g1-h2"), // trimmed to 20 min at 08:00
  e("d1-s1", "g1-h3"), // dropped
  e("d1-t1", "g1-bus"), // 45 min into a 30 min transition: trimmed
  e("d1-s2", "g1-w1", 3 * H, 4 * H),
  e("d2-s1", "g2-h1"),
  e("d2-s2", "g2-w1"),
];

const brief = (p: BlockPlacement) => [p.slot_id, p.clip.clip_id, p.source_start_s, p.source_end_s, p.synthetic_start_s, p.synthetic_end_s, p.trimmed];

describe("layoutBlocks (B2)", () => {
  it("places a block's clips back to back from the window start", () => {
    const placed = layoutBlocks(timeline(), memoryCatalog(CLIPS), [e("d1-s2", "g1-w1", 3 * H + 600, 3 * H + 1800), e("d1-s2", "g1-w1", 4 * H, 4 * H + 600)]);
    // Same clip twice is the verifier's business (clip_unique), not placement's.
    expect(placed.map(brief)).toEqual([
      ["d1-s2", "g1-w1", 3 * H + 600, 3 * H + 1800, 9 * H, 9 * H + 1200, false],
      ["d1-s2", "g1-w1", 4 * H, 4 * H + 600, 9 * H + 1200, 9 * H + 1800, false],
    ]);
    expect(placed.every((p) => p.day_index === 1)).toBe(true);
  });

  it("trims the clip that passes the block end and drops the clips after it", () => {
    const placed = layoutBlocks(timeline(), memoryCatalog(CLIPS), PLAN.filter((x) => x.slot_id === "d1-s1"));
    expect(placed.map(brief)).toEqual([
      ["d1-s1", "g1-h1", 0, 40 * M, 7 * H, 7 * H + 40 * M, false],
      ["d1-s1", "g1-h2", 40 * M, 60 * M, 7 * H + 40 * M, 8 * H, true],
    ]);
  });

  it("drops a clip that would start exactly at the block end (no zero-length placement)", () => {
    const placed = layoutBlocks(timeline(), memoryCatalog(CLIPS), [e("d2-s1", "g2-h1", 0, 30 * M), e("d2-s1", "g1-h2", 40 * M, 70 * M), e("d2-s1", "g1-h3")]);
    expect(placed.map((p) => [p.clip.clip_id, p.trimmed])).toEqual([["g2-h1", false], ["g1-h2", false]]);
  });

  it("places a transition block d<N>-t<k> from the transition start", () => {
    const placed = layoutBlocks(timeline(), memoryCatalog(CLIPS), [e("d1-t1", "g1-bus")]);
    expect(placed.map(brief)).toEqual([["d1-t1", "g1-bus", 2 * H, 2 * H + 30 * M, 8.5 * H, 9 * H, true]]);
    expect(placed[0]!.day_index).toBe(1);
  });

  it("orders placements by day, then by time, whatever the plan order", () => {
    // Blocks listed latest first; each block's own clips keep their order.
    const blocksFirst = ["d2-s2", "d2-s1", "d1-s2", "d1-t1", "d1-s1"].flatMap((id) => PLAN.filter((x) => x.slot_id === id));
    const placed = layoutBlocks(timeline(), memoryCatalog(CLIPS), blocksFirst);
    expect(placed.map((p) => p.slot_id)).toEqual(["d1-s1", "d1-s1", "d1-t1", "d1-s2", "d2-s1", "d2-s2"]);
    for (let i = 1; i < placed.length; i++) expect(placed[i]!.synthetic_start_s).toBeGreaterThanOrEqual(placed[i - 1]!.synthetic_end_s);
  });

  it("rejects unknown blocks, unknown clips and spans outside the clip", () => {
    const t = timeline();
    const cat = memoryCatalog(CLIPS);
    const code = (f: () => unknown) => {
      try {
        f();
      } catch (err) {
        expect(err).toBeInstanceOf(TimelineError);
        return (err as TimelineError).code;
      }
      return "none";
    };
    expect(code(() => layoutBlocks(t, cat, [e("d1-t2", "g1-bus")]))).toBe("SLOT_NOT_FOUND");
    expect(code(() => layoutBlocks(t, cat, [{ slot_id: "d1-s1", clip_id: "nope", source_start_s: 0, source_end_s: 1 }]))).toBe("CLIP_NOT_FOUND");
    expect(code(() => layoutBlocks(t, cat, [e("d1-s1", "g1-h1", 0, 41 * M)]))).toBe("BLOCK_PLAN_INVALID");
    expect(code(() => layoutBlocks(t, cat, [e("d1-s1", "g1-h1", 600, 600)]))).toBe("BLOCK_PLAN_INVALID");
  });

  it("transition spans match the manifest's transition no-video spans", () => {
    const t = timeline();
    expect(transitionBlocks(t.schedule.days[0]!)).toEqual([
      { transition_id: "d1-t1", from_slot: "d1-s1", to_slot: "d1-s2", span: [8.5 * H, 9 * H] },
    ]);
    expect(transitionBlocks(t.schedule.days[1]!)).toEqual([]);
    // The default manifest needs every required slot filled; reuse the B3 path's own output instead.
    const m = buildTimelineManifest(t, memoryCatalog(CLIPS), NOW, TEMPLATES, { placements: [] });
    expect(m.days[0]!.no_video_spans.filter((s) => s.kind === "transition")).toEqual([{ start: 8.5 * H, end: 9 * H, kind: "transition" }]);
  });
});

const ann = (id: string, start: number, end: number): Annotation => ({
  id,
  track: "captions",
  level: "L1",
  time: { kind: "interval", start, end },
  text: id,
  original: {},
});

function catalogWith(clips: Clip[], annotations: Record<string, Annotation[]>): ClipCatalog {
  const byId = new Map(clips.map((c) => [c.clip_id, c]));
  return { clips: () => clips, getClip: (id) => byId.get(id), getAnnotations: (_d, v) => annotations[v] ?? [] };
}

describe("buildTimelineManifest with placements (B3)", () => {
  const t = timeline();
  const annotations = {
    // g1 video: one caption inside h2's kept part, one in its trimmed-off part, one on the bus' kept part.
    "nm-g1": [ann("keep", 45 * M, 50 * M), ann("cut", 65 * M, 70 * M), ann("bus", 2 * H + 10 * M, 2 * H + 20 * M), ann("bus-cut", 2 * H + 40 * M, 2 * H + 44 * M)],
  };
  const cat = catalogWith(CLIPS, annotations);
  const placements = layoutBlocks(t, cat, PLAN);
  const m = buildTimelineManifest(t, cat, NOW, TEMPLATES, { placements });

  it("passes the manifest schema", () => {
    expect(TimelineManifestSchema.parse(m)).toEqual(m);
    expect(m.verifier_summary.pass).toBe(true);
  });

  it("records the trimmed source spans and uses d<N>-t<k> for commute clips", () => {
    const d1 = m.days[0]!.clips;
    expect(d1.map((c) => [c.slot_id, c.source_start_s, c.source_end_s, c.synthetic_start_s, c.synthetic_end_s])).toEqual([
      ["d1-s1", 0, 40 * M, 7 * H, 7 * H + 40 * M],
      ["d1-s1", 40 * M, 60 * M, 7 * H + 40 * M, 8 * H],
      ["d1-t1", 2 * H, 2 * H + 30 * M, 8.5 * H, 9 * H],
      ["d1-s2", 3 * H, 4 * H, 9 * H, 10 * H],
    ]);
    const bus = d1[2]!;
    expect([bus.category, bus.node, bus.setting, bus.relaxations]).toEqual(["transit", "transit", "outdoor", []]);
  });

  it("keeps only the captions inside the trimmed spans", () => {
    expect(m.captions.map((c) => c.provenance.orig_id)).toEqual(["keep", "bus"]);
    const keep = m.captions[0]!;
    expect(keep.block_id).toBe("clip:d1-s1:g1-h2");
    expect(keep.time).toEqual({ kind: "interval", start: 7 * H + 45 * M, end: 7 * H + 50 * M });
    expect(m.captions[1]!.time).toEqual({ kind: "interval", start: 8.5 * H + 10 * M, end: 8.5 * H + 20 * M });
  });

  it("computes no_video_spans and coverage from the placements", () => {
    const day = m.days[0]!;
    const parts = [...day.clips.map((c) => ({ start: c.synthetic_start_s, end: c.synthetic_end_s })), ...day.no_video_spans].sort((a, b) => a.start - b.start);
    expect(parts[0]!.start).toBe(7 * H);
    expect(parts[parts.length - 1]!.end).toBe(23 * H);
    for (let i = 1; i < parts.length; i++) expect(parts[i]!.start).toBe(parts[i - 1]!.end);
    // The bus fills the whole transition: no "transition" span is left on day 1.
    expect(day.no_video_spans.filter((s) => s.start >= 8 * H && s.end <= 9 * H)).toEqual([{ start: 8 * H, end: 8.5 * H, kind: "unscheduled" }]);
    expect(day.no_video_spans.filter((s) => s.kind === "unrecorded" && s.start >= 9 * H && s.end <= 12 * H)).toEqual([
      { start: 10 * H, end: 12 * H, kind: "unrecorded" },
    ]);
    // 60 + 30 + 60 min of footage over 16 h awake.
    expect(day.coverage_ratio).toBe(Math.round((150 / 960) * 1000) / 1000);
  });

  it("leaves the rest of a transition as a transition span when its clip is short", () => {
    const short = layoutBlocks(t, cat, [e("d1-t1", "g1-bus", 2 * H, 2 * H + 10 * M)]);
    const mm = buildTimelineManifest(t, cat, NOW, TEMPLATES, { placements: short });
    expect(mm.days[0]!.no_video_spans.filter((s) => s.start >= 8.5 * H && s.end <= 9 * H)).toEqual([{ start: 8.5 * H + 10 * M, end: 9 * H, kind: "transition" }]);
  });

  it("throws VERIFY_FAILED when the block-mode verifier fails", () => {
    const twice = layoutBlocks(t, cat, [e("d1-s1", "g1-h1"), e("d2-s1", "g1-h1")]);
    expect(() => buildTimelineManifest(t, cat, NOW, TEMPLATES, { placements: twice })).toThrow(/clip_unique/);
  });
});

describe("verifyTimeline block mode (B4)", () => {
  const t = timeline();
  const cat = memoryCatalog(CLIPS);
  const placements = layoutBlocks(t, cat, PLAN);
  const block = (p: readonly BlockPlacement[]) => verifyTimeline(t, cat, { mode: "block", placements: p });

  it("passes a valid placement set with empty t.slots, and skips time_of_day with a note", () => {
    // Off-bucket clips would fail time_of_day in the default mode.
    const night = CLIPS.map((c) => ({ ...c, time_of_day: "night" as const }));
    const r = verifyTimeline(t, memoryCatalog(night), { mode: "block", placements: layoutBlocks(t, memoryCatalog(night), PLAN) });
    for (const id of RULE_IDS) expect(r.rules[id].pass, `${id}: ${JSON.stringify(r.rules[id].violations)}`).toBe(true);
    expect(r.pass).toBe(true);
    expect(r.rules.time_of_day.note).toMatch(/block mode/);
    expect(r.rules.slot_window.note).toBeUndefined();
  });

  it("catches a placement outside its window", () => {
    const moved = placements.map((p) => (p.slot_id === "d1-t1" ? { ...p, synthetic_start_s: p.synthetic_start_s - 60, synthetic_end_s: p.synthetic_end_s - 60 } : p));
    const r = block(moved);
    expect(r.rules.slot_window.pass).toBe(false);
    expect(r.rules.slot_window.violations).toEqual([{ day_index: 1, slot_id: "d1-t1", detail: "clip g1-bus lies outside window" }]);
    expect(r.rules.monotonic.pass && r.rules.no_overlap.pass).toBe(true);
  });

  it("catches a date used on two days, counting transitions as their day", () => {
    const t2 = { ...t, schedule: { ...t.schedule, days: t.schedule.days.map((d) => (d.day_index === 2 ? { ...d, transitions: [{ from_slot: "d2-s1", to_slot: "d2-s2", minutes: 20 }] } : d)) } };
    const p = layoutBlocks(t2, cat, [...PLAN.filter((x) => x.slot_id !== "d1-t1"), e("d2-t1", "g1-bus")]);
    const r = verifyTimeline(t2, cat, { mode: "block", placements: p });
    expect(r.rules.day_group_single_day.pass).toBe(false);
    expect(r.rules.day_group_single_day.violations.map((v) => v.slot_id)).toContain("d2-t1");
    const others = RULE_IDS.filter((id) => id !== "day_group_single_day" && !r.rules[id].pass);
    expect(others).toEqual([]);
  });

  it("catches a clip used twice and overlapping source spans of one video", () => {
    expect(block(layoutBlocks(t, cat, [e("d1-s1", "g1-h1"), e("d1-s2", "g1-h1")])).rules.clip_unique.pass).toBe(false);
    // Two different clips whose placed spans overlap in the same video.
    const r = block(layoutBlocks(t, cat, [e("d1-s1", "g1-h1"), e("d1-s2", "g1-h2", 40 * M, 50 * M), e("d1-s3", "g1-h2", 45 * M, 55 * M)]));
    expect(r.rules.clip_unique.pass).toBe(false);
    // A clip is placed once, even as disjoint parts.
    const parts = block(layoutBlocks(t, cat, [e("d1-s2", "g1-w1", 3 * H, 4 * H), e("d1-s4", "g1-w1", 4 * H, 5 * H)]));
    expect(parts.rules.clip_unique.pass).toBe(false);
    // Different clips of one video with disjoint spans are fine (PLAN uses g1-h1 and g1-h2).
    expect(block(placements).rules.clip_unique.pass).toBe(true);
  });

  it("checks node binding over slot placements", () => {
    const r = block(layoutBlocks(t, cat, [e("d1-s2", "g1-h1")]));
    expect(r.rules.node_binding.pass).toBe(false);
  });

  describe("node_binding checks the place only", () => {
    const breakfast = at("g1-bf", "g1", 100 * M, 110 * M, { category: "meal" });
    const lunchAtWork = at("g1-lunch", "g1", 5 * H, 5 * H + 20 * M, { category: "meal", node: "work" });
    const strangerBus = at("x-bus", "g1", 6 * H, 6 * H + 10 * M, { ...transit, person_source: "someone-else" });
    const nc = memoryCatalog([...CLIPS, breakfast, lunchAtWork, strangerBus]);
    const nb = (tt: Timeline, plan: BlockPlanEntry[]) => {
      const r = verifyTimeline(tt, nc, { mode: "block", placements: layoutBlocks(tt, nc, plan) });
      return r.rules.node_binding;
    };
    const span = (slot_id: string, c: Clip): BlockPlanEntry => ({ slot_id, clip_id: c.clip_id, source_start_s: c.source_start_s, source_end_s: c.source_end_s });

    it("a breakfast-at-home clip may sit in home_morning (category is not checked)", () => {
      expect(nb(t, [span("d1-s1", breakfast)])).toEqual({ pass: true, violations: [] });
    });

    it("a lunch clip from the workplace may sit in a lunch slot at frequent_1 (meals use the real place)", () => {
      const w = makeWorld();
      const world = WorldSchema.parse({
        ...w,
        nodes: { ...w.nodes, frequent_1: { kind: "frequent_place", setting: "venue_indoor", optional: true } },
        node_binding: { ...w.node_binding, frequent_1: { dataset: ["nextme800"], person_source: "p1", node: "cafe" } },
      });
      const tt: Timeline = {
        ...t,
        world,
        schedule: {
          ...t.schedule,
          days: t.schedule.days.map((d) => ({ ...d, slots: d.slots.map((s) => (s.slot_id === "d1-s3" ? { ...s, spec: { ...s.spec, node: "frequent_1" } } : s)) })),
        },
      };
      expect(nb(tt, [span("d1-s3", lunchAtWork)])).toEqual({ pass: true, violations: [] });
      // A meal clip bound to no place of the world is still rejected.
      const nowhere = at("g1-nowhere", "g1", 7 * H, 7 * H + 10 * M, { category: "meal", node: "street-stall" });
      const nc2 = memoryCatalog([...nc.clips(), nowhere]);
      const r = verifyTimeline(tt, nc2, { mode: "block", placements: layoutBlocks(tt, nc2, [span("d1-s3", nowhere)]) });
      expect(r.rules.node_binding.violations).toEqual([{ day_index: 1, slot_id: "d1-s3", detail: "clip g1-nowhere: not accepted by any bound place" }]);
    });

    it("a workplace clip in home_morning fails", () => {
      const r = nb(t, [e("d1-s1", "g1-w1", 3 * H, 3 * H + 10 * M)]);
      expect(r.pass).toBe(false);
      expect(r.violations).toEqual([{ day_index: 1, slot_id: "d1-s1", detail: "clip g1-w1: not accepted by the binding of node home" }]);
    });

    it("a transition takes only transit clips of the bound person", () => {
      expect(nb(t, [e("d1-t1", "g1-bus")])).toEqual({ pass: true, violations: [] });
      const home = nb(t, [e("d1-t1", "g1-h1")]);
      expect(home.pass).toBe(false);
      expect(home.violations).toEqual([{ day_index: 1, slot_id: "d1-t1", detail: "clip g1-h1: not a transit clip of the bound person" }]);
      expect(nb(t, [span("d1-t1", strangerBus)]).pass).toBe(false);
    });
  });

  it("does not require empty slots to be filled in block mode", () => {
    const r = block([]);
    expect(r.pass).toBe(true);
  });

  it("checks the trip rule over placements", () => {
    const world = WorldSchema.parse({
      ...makeWorld(),
      nodes: { ...makeWorld().nodes, trip_1: { kind: "trip", lodging_setting: "venue_indoor", outing_setting: "outdoor", meal_setting: "venue_indoor" } },
      node_binding: { ...makeWorld().node_binding, trip_1: { dataset: ["nextme800"], person_source: "p1", day_group_sequence: ["2026-06-29"] } },
    });
    const tripDefs: SlotDef[] = [
      { role: "lodging_morning", category: "home", setting: "venue_indoor", node: "trip_1:lodging", activity: "getting ready", required: true, window: ["08:00", "09:00"] },
    ];
    const tt = createTimeline(world, makeSchedule(1, () => tripDefs), genId);
    tt.schedule.days[0] = { ...tt.schedule.days[0]!, day_type: "trip_day", trip: { node: "trip_1", index: 0 } };
    const lodging = (id: string, g: string) => at(id, g, 0, 600, { setting: "venue_indoor", node: "trip:lodging" });
    const tc = memoryCatalog([lodging("g7-l", "2026-06-29"), lodging("g8-l", "2026-06-30"), at("home-l", "2026-06-29", 700, 900)]);
    const run = (id: string) =>
      verifyTimeline(tt, tc, { mode: "block", placements: layoutBlocks(tt, tc, [{ slot_id: "d1-s1", clip_id: id, source_start_s: tc.getClip(id)!.source_start_s, source_end_s: tc.getClip(id)!.source_end_s }]) });
    expect(run("g7-l").rules.trip.pass).toBe(true);
    const bad = run("g8-l").rules.trip;
    expect(bad.pass).toBe(false);
    expect(bad.violations[0]!.detail).toMatch(/comes from 2026-06-30/);
    // A home clip on a trip day is caught too.
    expect(run("home-l").rules.trip.violations.map((v) => v.detail)).toEqual(["clip home-l is a home clip on a trip day"]);
  });

  it("trip rule, block mode: a meal slot takes a clip of any place of the trip; a non-meal slot still needs its place", () => {
    const world = WorldSchema.parse({
      ...makeWorld(),
      nodes: { ...makeWorld().nodes, trip_1: { kind: "trip", lodging_setting: "venue_indoor", outing_setting: "outdoor", meal_setting: "venue_indoor" } },
      node_binding: { ...makeWorld().node_binding, trip_1: { dataset: ["nextme800"], person_source: "p1", day_group_sequence: ["2026-06-29"] } },
    });
    const defs: SlotDef[] = [
      { role: "lodging_morning", category: "home", setting: "venue_indoor", node: "trip_1:lodging", activity: "getting ready", required: true, window: ["08:00", "09:00"] },
      { role: "dinner", category: "meal", setting: "venue_indoor", node: "trip_1:meal", activity: "dinner", required: true, window: ["18:00", "19:00"] },
    ];
    const tt = createTimeline(world, makeSchedule(1, () => defs), genId);
    tt.schedule.days[0] = { ...tt.schedule.days[0]!, day_type: "trip_day", trip: { node: "trip_1", index: 0 } };
    const lodging = (id: string) => at(id, "2026-06-29", 0, 600, { setting: "venue_indoor", node: "trip:lodging" });
    const tc = memoryCatalog([lodging("l1"), lodging("l2"), at("home-l", "2026-06-29", 700, 900)]);
    const run = (slotId: string, id: string) =>
      verifyTimeline(tt, tc, { mode: "block", placements: layoutBlocks(tt, tc, [{ slot_id: slotId, clip_id: id, source_start_s: tc.getClip(id)!.source_start_s, source_end_s: tc.getClip(id)!.source_end_s }]) });
    expect(run("d1-s2", "l1").rules.trip.pass).toBe(true);
    expect(run("d1-s2", "home-l").rules.trip.violations.map((v) => v.detail)).toEqual(["clip home-l is a home clip on a trip day"]);
  });

  it("without the option, the verifier ignores placements and checks t.slots", () => {
    const r = verifyTimeline(t, cat);
    expect(r.rules.slot_window.pass).toBe(false); // required slots are empty in t.slots
    expect(Object.values(r.rules).every((x) => !("note" in x))).toBe(true);
  });
});
