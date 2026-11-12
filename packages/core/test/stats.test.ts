import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { TimelineManifestSchema, summarizeManifests, type TimelineManifest } from "../src/index.js";

const SNAPSHOT = join(dirname(fileURLToPath(import.meta.url)), "__snapshots__", "timeline.manifest.json");
const golden = (): TimelineManifest => TimelineManifestSchema.parse(JSON.parse(readFileSync(SNAPSHOT, "utf8")));

function make(rs: number, redraws: [number, number], opts: { jaccard: number | null; shrunk: number; adj?: boolean }): TimelineManifest {
  const m = golden();
  m.inputs.dials.routine_strength = rs;
  m.realized.shrunk = { hobby: opts.shrunk };
  m.realized.composition_consistency.workday!.same_reference =
    opts.jaccard === null ? null : { optional_jaccard: opts.jaccard, order_similarity: 1, start_diff_minutes: 0, pairs: 1 };
  m.donor_adjustments.weekday.redraws = redraws[0];
  m.donor_adjustments.weekend.redraws = redraws[1];
  if (opts.adj) m.donor_adjustments.weekday.adjustments = [{ item: "work_end", from: 1, to: 2 }];
  m.days[0]!.jitter_redraws = 7;
  m.days[0]!.anchor_adjustments = opts.adj ? [{ item: "x", from: 1, to: 2 } as never] : [];
  return m;
}

describe("summarizeManifests", () => {
  const a = make(0.8, [0, 2], { jaccard: 0.5, shrunk: 2, adj: true });
  const b = make(0.2, [9, 5], { jaccard: null, shrunk: 0 });
  const r = summarizeManifests([a, b]);

  it("groups by dials sorted ascending", () => {
    expect(r.manifests).toBe(2);
    expect(r.groups.map((g) => g.dials.routine_strength)).toEqual([0.2, 0.8]);
    expect(r.groups.map((g) => g.n)).toEqual([1, 1]);
  });

  it("skips null values, reports n, flattens shrunk per day, excludes dials echo", () => {
    const hi = r.groups[1]!.realized;
    expect(hi["composition_consistency.workday.same_reference.optional_jaccard"]).toEqual({ mean: 0.5, min: 0.5, max: 0.5, n: 1 });
    expect(hi["optional_presence.hobby"]).toEqual({ mean: 0, min: 0, max: 0, n: 1 });
    expect(hi["shrunk.hobby"]).toEqual({ mean: 2, min: 2, max: 2, n: 1 });
    expect(hi["shrunk.hobby_per_day"]).toEqual({ mean: 1, min: 1, max: 1, n: 1 });
    expect(hi["slot_exception_rate"]).toBeUndefined();
    expect(Object.keys(hi).some((k) => k.startsWith("dials"))).toBe(false);
    expect(r.groups[0]!.realized["composition_consistency.workday.same_reference.optional_jaccard"]).toBeUndefined();
  });

  it("means over a shared group", () => {
    const c = make(0.8, [1, 1], { jaccard: 0.9, shrunk: 4 });
    const g = summarizeManifests([a, c]).groups;
    expect(g).toHaveLength(1);
    expect(g[0]!.realized["composition_consistency.workday.same_reference.optional_jaccard"]).toEqual({ mean: 0.7, min: 0.5, max: 0.9, n: 2 });
    expect(g[0]!.realized["shrunk.hobby"]).toEqual({ mean: 3, min: 2, max: 4, n: 2 });
  });

  it("donor histograms and shares", () => {
    expect(r.donors.n).toBe(4);
    expect(r.donors.redraws_histogram).toEqual({ "0": 1, "1": 0, "2": 1, "3": 0, "4": 0, "5": 2 });
    expect(r.donors.redraws_histogram_by_kind.weekday).toEqual({ "0": 1, "1": 0, "2": 0, "3": 0, "4": 0, "5": 1 });
    expect(r.donors.redraws_histogram_by_kind.weekend).toEqual({ "0": 0, "1": 0, "2": 1, "3": 0, "4": 0, "5": 1 });
    expect(r.donors.adjusted_share).toBe(0.25);
    expect(r.donors.work_end_repairs).toBe(1);
  });

  it("day shares and jitter histogram", () => {
    expect(r.days.n).toBe(4);
    expect(r.days.anchor_adjusted_share).toBe(0.25);
    expect(r.days.jitter_redraws_histogram).toEqual({ "0": 2, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6+": 2 });
  });
});
