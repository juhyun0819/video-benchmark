import { describe, expect, it } from "vitest";
import {
  assignClip,
  createTimeline,
  fillTimeline,
  layoutTimeline,
  parseClock,
  rankSelector,
  type ClipCatalog,
  type PlacedClip,
  type Timeline,
} from "../src/index.js";
import { clip, genId, makeClips, makeSchedule, makeWorld, memoryCatalog, type SlotDef } from "./fixtures.js";

const noRegen = () => {
  throw new Error("unexpected regen");
};

function filled(seed: number, catalog: ClipCatalog, days = 28, defsFor?: (d: number) => SlotDef[] | undefined): Timeline {
  return fillTimeline(createTimeline(makeWorld({ seed }), makeSchedule(days, defsFor), genId), catalog, rankSelector, noRegen);
}

/** Minutes since 00:00 of the clip's own day. */
const minuteOfDay = (p: PlacedClip, s: number) => (s - (p.day_index - 1) * 86400) / 60;

function windowOf(t: Timeline, slotId: string): [number, number] {
  const slot = t.schedule.days.flatMap((d) => d.slots).find((s) => s.slot_id === slotId)!;
  return [parseClock(slot.window[0]), parseClock(slot.window[1])];
}

describe("layoutTimeline", () => {
  it("clips stay inside window and do not overlap", () => {
    const catalog = memoryCatalog(makeClips());
    let t = filled(7, catalog);
    // Three clips in one evening slot.
    const extra = memoryCatalog([
      ...catalog.clips(),
      clip({ clip_id: "g01-x1", activity: "relaxing at home", time_of_day: "evening", day_group: "nextme800:g01", source_start_s: 50000, source_end_s: 51200 }),
      clip({ clip_id: "g01-x2", activity: "relaxing at home", time_of_day: "evening", day_group: "nextme800:g01", source_start_s: 60000, source_end_s: 61800 }),
    ]);
    t = assignClip(t, extra, { slotId: "d1-s6", clipId: "g01-x1" });
    t = assignClip(t, extra, { slotId: "d1-s6", clipId: "g01-x2" });
    expect(t.slots["d1-s6"]!.clip_ids).toHaveLength(3);

    const placed = layoutTimeline(t, extra);
    const total = Object.values(t.slots).reduce((n, s) => n + s.clip_ids.length, 0);
    expect(placed).toHaveLength(total);
    for (const p of placed) {
      const [ws, we] = windowOf(t, p.slot_id);
      expect(p.slot_id.startsWith(`d${p.day_index}-`)).toBe(true);
      expect(minuteOfDay(p, p.synthetic_start_s)).toBeGreaterThanOrEqual(ws);
      expect(minuteOfDay(p, p.synthetic_end_s)).toBeLessThanOrEqual(we);
      expect(p.synthetic_end_s - p.synthetic_start_s).toBeCloseTo(p.clip.source_end_s - p.clip.source_start_s, 2);
    }
    const sorted = [...placed].sort((a, b) => a.synthetic_start_s - b.synthetic_start_s);
    expect(sorted).toEqual(placed);
    for (let i = 1; i < placed.length; i++) {
      expect(placed[i]!.synthetic_start_s).toBeGreaterThanOrEqual(placed[i - 1]!.synthetic_end_s);
    }
    // Assignment order within a slot.
    expect(placed.filter((p) => p.slot_id === "d1-s6").map((p) => p.clip.clip_id)).toEqual(t.slots["d1-s6"]!.clip_ids);
  });

  it("deterministic for same seed", () => {
    const catalog = memoryCatalog(makeClips());
    const a = layoutTimeline(filled(7, catalog), catalog);
    const b = layoutTimeline(filled(7, catalog), catalog);
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
    const c = layoutTimeline(filled(8, catalog), catalog);
    expect(c.map((p) => p.clip.clip_id)).toEqual(a.map((p) => p.clip.clip_id));
    expect(c.map((p) => p.synthetic_start_s)).not.toEqual(a.map((p) => p.synthetic_start_s));
  });

  it("unknown time_of_day samples within the full window", () => {
    const long = (node: "home" | "climbing"): SlotDef =>
      node === "home"
        ? { role: "home_day", category: "home", setting: "home_indoor", node: "home", activity: "getting ready", required: true, window: ["08:00", "20:00"] }
        : { role: "hobby", category: "hobby", setting: "venue_indoor", node: "climbing", activity: "climbing", tags: ["climbing"], required: true, window: ["08:00", "20:00"] };
    const catalog = memoryCatalog(makeClips({ hobbyVideos: 20 }));

    // Unknown-bucket climbing clips: anywhere in 08:00–20:00.
    const hobby = layoutTimeline(filled(7, catalog, 20, () => [long("climbing")]), catalog);
    expect(hobby).toHaveLength(20);
    expect(hobby.every((p) => p.clip.time_of_day === "unknown")).toBe(true);
    const starts = hobby.map((p) => minuteOfDay(p, p.synthetic_start_s));
    expect(Math.min(...starts)).toBeGreaterThanOrEqual(480);
    expect(Math.max(...hobby.map((p) => minuteOfDay(p, p.synthetic_end_s)))).toBeLessThanOrEqual(1200);
    expect(Math.min(...starts)).toBeLessThan(660);
    expect(Math.max(...starts)).toBeGreaterThan(1020);

    // Morning clips in the same window are narrowed to 08:00–11:00.
    const home = layoutTimeline(filled(7, catalog, 20, () => [long("home")]), catalog);
    expect(home).toHaveLength(20);
    for (const p of home) {
      expect(p.clip.time_of_day).toBe("morning");
      expect(minuteOfDay(p, p.synthetic_start_s)).toBeGreaterThanOrEqual(480);
      expect(minuteOfDay(p, p.synthetic_end_s)).toBeLessThanOrEqual(660);
    }
  });

  it("falls back to the full window when the bucket overlap is shorter than the clips", () => {
    // Window 10:30–12:00, a 40-minute morning clip: overlap with morning (to 11:00) is only 30 min.
    const late: SlotDef = { role: "home_late", category: "home", setting: "home_indoor", node: "home", activity: "getting ready", required: true, window: ["10:30", "12:00"] };
    const c = clip({ clip_id: "m40", source_start_s: 0, source_end_s: 2400 });
    const catalog = memoryCatalog([c]);
    const t = filled(7, catalog, 1, () => [late]);
    const [p] = layoutTimeline(t, catalog);
    expect(minuteOfDay(p!, p!.synthetic_start_s)).toBeGreaterThanOrEqual(630);
    expect(minuteOfDay(p!, p!.synthetic_end_s)).toBeLessThanOrEqual(720);
  });
});
