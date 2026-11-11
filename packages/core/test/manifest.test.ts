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
} from "../src/index.js";
import { clip, genId, makeSchedule, makeWorld, type SlotDef } from "./fixtures.js";

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
    const m = buildTimelineManifest(t, catalogOf(clips, annotations), NOW);
    expect(TimelineManifestSchema.parse(m)).toEqual(m);
    expect(m.created_at).toBe("2026-11-12T00:00:00.000Z");
    expect(m.days.map((d) => d.clips.length)).toEqual([4, 4]);
    const text = JSON.stringify(m, null, 2) + "\n";
    if (!existsSync(SNAPSHOT)) {
      mkdirSync(dirname(SNAPSHOT), { recursive: true });
      writeFileSync(SNAPSHOT, text);
    }
    expect(text).toBe(readFileSync(SNAPSHOT, "utf8"));
  });

  it("caption clipped at clip edge keeps clipped flag", () => {
    const { t, clips } = build(1);
    clips[0] = mk("d1-home", { day_group: "g1", source_start_s: 100, source_end_s: 100 + HOUR });
    const m = buildTimelineManifest(t, catalogOf(clips, { "vid-d1-home": [ann("edge", 50, 200)] }), NOW);
    const cap = m.captions.find((c) => c.provenance.orig_id === "edge");
    expect(cap?.clipped).toEqual({ start: true });
    const first = m.days[0]!.clips[0]!;
    expect(cap?.block_id).toBe(`clip:${first.slot_id}:d1-home`);
    expect(cap?.time).toEqual({ kind: "interval", start: first.synthetic_start_s, end: Math.round((first.synthetic_start_s + 100) * 1000) / 1000 });
  });

  it("catalog missing two clips -> MISSING_CLIPS lists both", () => {
    const { t, clips } = build(1);
    const rest = clips.filter((c) => c.clip_id !== "d1-pm" && c.clip_id !== "d1-am");
    try {
      buildTimelineManifest(t, catalogOf(rest), NOW);
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
      buildTimelineManifest(bad, catalogOf(clips), NOW);
      expect.unreachable();
    } catch (e) {
      expect((e as TimelineError).code).toBe("VERIFY_FAILED");
      expect((e as TimelineError).details["failed_rules"]).toContain("clip_unique");
    }
  });

  it("coverage_ratio = 0.25 for 4 h of clips over 16 h awake", () => {
    const { t, clips } = build(1);
    const m = buildTimelineManifest(t, catalogOf(clips), NOW);
    expect(m.days[0]!.wake).toBe("07:00");
    expect(m.days[0]!.sleep).toBe("23:00");
    expect(m.days[0]!.coverage_ratio).toBe(0.25);
  });
});
