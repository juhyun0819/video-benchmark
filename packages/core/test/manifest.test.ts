import { describe, expect, it } from "vitest";
import {
  CompositionError,
  LEVEL_ORDER,
  LICENSES,
  buildManifest,
  createComposition,
  insertEvent,
  type Catalog,
  type Composition,
  type TimeSpan,
} from "../src/index.js";
import { makeFixtureCatalog } from "./fixtures.js";

const COOK = "hdepic:P01-x:10.000-70.000";
const WALK = "ego4d:uid-1:0.000-120.000";
const NOW = new Date("2026-11-03T00:00:00Z");

function counter(): () => string {
  let n = 0;
  return () => `r${++n}`;
}

function goldenComposition(catalog: ReturnType<typeof makeFixtureCatalog>): Composition {
  const gen = counter();
  let comp = createComposition(catalog.baseDay, { genId: () => "c1" });
  comp = insertEvent(comp, catalog, { slot: 1, candidateId: COOK }, gen);
  comp = insertEvent(comp, catalog, { slot: 3, candidateId: WALK }, gen);
  return comp;
}

const startOf = (t: TimeSpan) => (t.kind === "interval" ? t.start : t.t);
const endOf = (t: TimeSpan) => (t.kind === "interval" ? t.end : t.t);

describe("buildManifest", () => {
  it("golden manifest", async () => {
    const catalog = makeFixtureCatalog();
    const m = buildManifest(goldenComposition(catalog), catalog, NOW);
    await expect(JSON.stringify(m, null, 2)).toMatchFileSnapshot("__snapshots__/golden.manifest.json");
  });

  it("every caption lies within its block", () => {
    const catalog = makeFixtureCatalog();
    const m = buildManifest(goldenComposition(catalog), catalog, NOW);
    expect(m.tracks.captions.length).toBeGreaterThan(0);
    for (const c of m.tracks.captions) {
      const b = m.blocks.find((x) => x.block_id === c.block_id)!;
      expect(b.composed_start).toBeLessThanOrEqual(startOf(c.time));
      expect(endOf(c.time)).toBeLessThanOrEqual(b.composed_end);
    }
  });

  it("duration equals sum of block lengths", () => {
    const catalog = makeFixtureCatalog();
    const m = buildManifest(goldenComposition(catalog), catalog, NOW);
    const sum = m.blocks.reduce((s, b) => s + (b.composed_end - b.composed_start), 0);
    expect(m.duration).toBeCloseTo(sum, 6);
    expect(m.duration).toBe(1980);
  });

  it("sorts captions by start, then level order, then id", () => {
    const catalog = makeFixtureCatalog();
    const m = buildManifest(goldenComposition(catalog), catalog, NOW);
    const keys = m.tracks.captions.map((c) => [startOf(c.time), LEVEL_ORDER.indexOf(c.level), c.id] as const);
    for (let i = 1; i < keys.length; i++) {
      const a = keys[i - 1]!;
      const b = keys[i]!;
      const cmp = a[0] - b[0] || a[1] - b[1] || (a[2] < b[2] ? -1 : a[2] > b[2] ? 1 : 0);
      expect(cmp).toBeLessThanOrEqual(0);
    }
  });

  it("strips slot/ref_id/base_index from blocks and sets header fields", () => {
    const catalog = makeFixtureCatalog();
    const m = buildManifest(goldenComposition(catalog), catalog, NOW);
    for (const b of m.blocks) {
      expect(Object.keys(b).sort()).toEqual([
        "block_id",
        "category",
        "composed_end",
        "composed_start",
        "role",
        "source",
        "src_end",
        "src_start",
        "summary",
      ]);
    }
    expect(m.created_at).toBe("2026-11-03T00:00:00.000Z");
    expect(m.composition_id).toBe("c1");
  });

  it("removing all events yields base captions offset only", () => {
    const catalog = makeFixtureCatalog();
    const comp = createComposition(catalog.baseDay, { genId: () => "c1" });
    const m = buildManifest(comp, catalog, NOW);
    expect(m.blocks.map((b) => b.role)).toEqual(["base", "base", "base"]);
    const block0 = m.tracks.captions.filter((c) => c.block_id === "base:2026-04-15:0");
    expect(block0.length).toBeGreaterThan(0);
    for (const c of block0) {
      const src = c.provenance.src_time;
      expect(startOf(c.time)).toBe(Math.max(startOf(src), 0));
      expect(endOf(c.time)).toBe(Math.min(endOf(src), 600));
    }
    expect(m.duration).toBe(1800);
  });

  it("licenses only for used datasets", () => {
    const catalog = makeFixtureCatalog();
    const base = buildManifest(createComposition(catalog.baseDay, { genId: () => "c1" }), catalog, NOW);
    expect(base.licenses).toEqual([{ dataset: "nextme800", ...LICENSES.nextme800 }]);
    const full = buildManifest(goldenComposition(catalog), catalog, NOW);
    expect(full.licenses.map((l) => l.dataset)).toEqual(["nextme800", "hdepic", "ego4d"]);
  });

  it("exports an Ego-Exo4D block with its ego video path, license and keystep before atomic at one start", () => {
    const catalog = makeFixtureCatalog();
    const TAKE = "egoexo4d:take-1:5.000-65.000";
    const media = { kind: "video", fps: 30, path: "takes/t1/frame_aligned_videos/aria01_214-1.mp4" } as const;
    const withTake: Catalog = {
      getBaseDay: (d) => catalog.getBaseDay(d),
      getAnnotations: (ds, v) =>
        ds === "egoexo4d"
          ? [
              { id: "a0", track: "captions", level: "atomic", time: { kind: "point", t: 10 }, text: "C cracks an egg.", original: {} },
              { id: "k0", track: "captions", level: "keystep", time: { kind: "interval", start: 10, end: 20 }, text: "Crack eggs", original: {} },
            ]
          : catalog.getAnnotations(ds, v),
      getCandidate: (id) =>
        id === TAKE
          ? {
              id: TAKE,
              role: "event",
              category: "cooking",
              source: { dataset: "egoexo4d", video_id: "take-1", media },
              src_start: 5,
              src_end: 65,
              summary: "Cooking an Omelet",
            }
          : catalog.getCandidate(id),
    };
    const comp = insertEvent(
      createComposition(catalog.baseDay, { genId: () => "c1" }),
      withTake,
      { slot: 0, candidateId: TAKE },
      () => "r1",
    );

    const m = buildManifest(comp, withTake, NOW);

    expect(m.blocks.find((b) => b.block_id === "evt:r1")?.source.media).toEqual(media);
    expect(m.licenses.find((l) => l.dataset === "egoexo4d")?.license).toBe("Ego-Exo4D License Agreement");
    expect(m.tracks.captions.filter((c) => c.block_id === "evt:r1").map((c) => c.level)).toEqual(["keystep", "atomic"]);
  });

  it("rounds composed values to 3 decimals but keeps src values verbatim", () => {
    const catalog = makeFixtureCatalog();
    const odd: Catalog = {
      getBaseDay: (d) => catalog.getBaseDay(d),
      getAnnotations: (ds, v) => catalog.getAnnotations(ds, v),
      getCandidate: (id) => {
        const c = catalog.getCandidate(id);
        return id === COOK && c ? { ...c, src_start: 10.0004, src_end: 70.00049 } : c;
      },
    };
    const comp = insertEvent(
      createComposition(catalog.baseDay, { genId: () => "c1" }),
      catalog,
      { slot: 1, candidateId: COOK },
      () => "r1",
    );
    const m = buildManifest(comp, odd, NOW);
    const evt = m.blocks.find((b) => b.block_id === "evt:r1")!;
    expect(evt.src_start).toBe(10.0004);
    expect(evt.src_end).toBe(70.00049);
    expect(evt.composed_start).toBe(600);
    expect(evt.composed_end).toBe(660);
    expect(m.duration).toBe(1860);
  });

  it("fails with MISSING_CANDIDATES when catalog lost a candidate", () => {
    const catalog = makeFixtureCatalog();
    const comp = goldenComposition(catalog);
    const lost: Catalog = {
      getBaseDay: (d) => catalog.getBaseDay(d),
      getAnnotations: (ds, v) => catalog.getAnnotations(ds, v),
      getCandidate: (id) => (id === COOK ? undefined : catalog.getCandidate(id)),
    };
    try {
      buildManifest(comp, lost, NOW);
      expect.unreachable();
    } catch (e) {
      expect(e).toBeInstanceOf(CompositionError);
      expect((e as CompositionError).code).toBe("MISSING_CANDIDATES");
      expect((e as CompositionError).details["ids"]).toEqual([COOK]);
    }
  });
});
