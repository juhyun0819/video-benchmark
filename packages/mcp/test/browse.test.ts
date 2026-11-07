import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { CompositionError, type Composition, type Manifest } from "@vb/core";
import { existsSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { FsCatalog, CatalogMissingError } from "../src/fsCatalog.js";
import { toToolError } from "../src/server.js";
import { CompositionStore } from "../src/store.js";
import { connect, IDS, makeTempDir, textOf, writeFixtureCatalog } from "./helpers.js";

let dir: string;
let session: Awaited<ReturnType<typeof connect>>;

beforeEach(async () => {
  dir = makeTempDir();
  writeFixtureCatalog(dir);
  session = await connect(dir);
});
afterEach(async () => {
  await session.close();
});

describe("browse tools", () => {
  it("list_base_days returns fixture day with 3 blocks", async () => {
    const r = await session.call("list_base_days");
    expect(r.isError).toBeFalsy();
    expect(r.structuredContent).toEqual({
      days: [{ day: "2026-04-15", blocks: 3, duration: 600 + 900 + 300 }],
    });
    expect(textOf(r)).toContain("2026-04-15");
  });

  it("get_base_day returns block index, length, summary and source range", async () => {
    const r = await session.call("get_base_day", { day: "2026-04-15" });
    const sc = r.structuredContent as { day: string; blocks: Array<Record<string, unknown>> };
    expect(sc.blocks).toHaveLength(3);
    expect(sc.blocks[1]).toMatchObject({
      index: 1,
      id: "nextme800:rec-A:600.000-1500.000",
      duration: 900,
      summary: "Commute and arrival",
      video_id: "rec-A",
      src_start: 600,
      src_end: 1500,
    });
    const missing = await session.call("get_base_day", { day: "1999-01-01" });
    expect(missing.isError).toBe(true);
  });

  it("list_categories returns cooking and walking with counts", async () => {
    const r = await session.call("list_categories");
    expect(r.structuredContent).toEqual({
      categories: [
        { category: "cooking", count: 2 },
        { category: "walking", count: 1 },
      ],
    });
  });

  it("list_event_candidates filters by category and query and pages", async () => {
    const all = await session.call("list_event_candidates", { category: "cooking" });
    const sc = all.structuredContent as { total: number; items: Array<{ id: string; duration: number; dataset: string; summary: string }> };
    expect(sc.total).toBe(2);
    expect(sc.items.map((i) => i.id)).toEqual([IDS.cooking, IDS.cooking2]);
    expect(sc.items[0]).toMatchObject({ duration: 60, dataset: "hdepic", summary: "Cooking onions in a pan" });

    const q = await session.call("list_event_candidates", { category: "cooking", query: "OMELETTE" });
    expect((q.structuredContent as { total: number }).total).toBe(1);

    const dur = await session.call("list_event_candidates", { category: "cooking", min_sec: 100 });
    expect((dur.structuredContent as { items: Array<{ id: string }> }).items.map((i) => i.id)).toEqual([IDS.cooking2]);
    const dur2 = await session.call("list_event_candidates", { category: "cooking", max_sec: 100 });
    expect((dur2.structuredContent as { items: Array<{ id: string }> }).items.map((i) => i.id)).toEqual([IDS.cooking]);

    const page = await session.call("list_event_candidates", { category: "cooking", limit: 1, offset: 1 });
    const psc = page.structuredContent as { total: number; items: Array<{ id: string }> };
    expect(psc.total).toBe(2);
    expect(psc.items.map((i) => i.id)).toEqual([IDS.cooking2]);

    const none = await session.call("list_event_candidates", { category: "nope" });
    expect((none.structuredContent as { total: number }).total).toBe(0);
  });

  it("get_candidate returns caption preview limited to preview", async () => {
    const r = await session.call("get_candidate", { candidate_id: IDS.cooking, preview: 2 });
    expect(r.isError).toBeFalsy();
    const sc = r.structuredContent as {
      candidate: { id: string };
      preview: Array<{ text: string; start: number }>;
    };
    expect(sc.candidate.id).toBe(IDS.cooking);
    // window [10,70): overlapping narrations are 5-15, 20-28, 40-55, 60-70; first 2 by start.
    expect(sc.preview.map((p) => p.text)).toEqual(["take the pan from the shelf", "chop the onion"]);

    const full = await session.call("get_candidate", { candidate_id: IDS.cooking });
    expect((full.structuredContent as { preview: unknown[] }).preview).toHaveLength(4);

    const walk = await session.call("get_candidate", { candidate_id: IDS.walking, preview: 10 });
    // point annotations 5, 30, 90 plus summary interval 0-120
    expect((walk.structuredContent as { preview: unknown[] }).preview).toHaveLength(4);
  });

  it("get_candidate resolves base blocks and reports unknown ids", async () => {
    const ok = await session.call("get_candidate", { candidate_id: IDS.base0 });
    expect(ok.isError).toBeFalsy();
    const bad = await session.call("get_candidate", { candidate_id: "nope" });
    expect(bad.isError).toBe(true);
    expect(textOf(bad)).toBe("candidate_id not found: nope — check with list_event_candidates");
  });

  it("missing catalog yields an ingest hint", async () => {
    const empty = makeTempDir();
    const s = await connect(empty);
    try {
      const r = await s.call("list_base_days");
      expect(r.isError).toBe(true);
      expect(textOf(r)).toContain("npm run ingest");
      expect(textOf(r)).toBe("Catalog not found. Run npm run ingest first");
    } finally {
      await s.close();
    }
  });
});

describe("FsCatalog", () => {
  it("throws CatalogMissingError without index.json", () => {
    expect(() => new FsCatalog(makeTempDir())).toThrow(CatalogMissingError);
  });

  it("implements the core Catalog interface", () => {
    const c = new FsCatalog(dir);
    expect(c.getBaseDay("2026-04-15")?.blocks).toHaveLength(3);
    expect(c.getBaseDay("1999-01-01")).toBeUndefined();
    expect(c.getBaseDay("../etc")).toBeUndefined();
    expect(c.getCandidate(IDS.walking)?.category).toBe("walking");
    expect(c.getCandidate(IDS.base0)?.role).toBe("base");
    expect(c.getCandidate("zzz")).toBeUndefined();
    expect(c.getAnnotations("ego4d", "uid-1")).toHaveLength(4);
    expect(c.getAnnotations("ego4d", "unknown")).toEqual([]);
  });
});

describe("toToolError", () => {
  it("maps codes to their messages", () => {
    const t = (e: unknown) => (toToolError(e).content[0] as { text: string }).text;
    expect(toToolError(new Error("x")).isError).toBe(true);
    expect(t(new CompositionError("SLOT_OUT_OF_RANGE", "m", { slot: 21, max: 20 }))).toBe("slot 21 out of range (0..20)");
    expect(t(new CompositionError("CANDIDATE_NOT_FOUND", "m", { candidateId: "a" }))).toBe("candidate_id not found: a — check with list_event_candidates");
    expect(t(new CompositionError("REVISION_MISMATCH", "m", { expected: 1, current: 3 }))).toBe("revision mismatch (current 3)");
    expect(t(new CompositionError("MISSING_CANDIDATES", "m", { ids: ["a", "b"] }))).toBe("candidates not in catalog: a, b");
    expect(t(new CompositionError("REF_NOT_FOUND", "boom", { refId: "r" }))).toBe("ref_id not found: r — check with get_composition");
    expect(t(new CompositionError("NOT_EVENT", "m", { candidateId: "c" }))).toBe("not an event candidate: c");
    expect(t(new CompositionError("POSITION_OUT_OF_RANGE", "m", { position: 5, max: 2 }))).toBe("position 5 out of range (0..2)");
    expect(t(new CompositionError("COMPOSITION_NOT_FOUND", "m", { id: "x1" }))).toBe("composition not found: x1 — check with list_compositions");
    expect(t(new CompositionError("BASE_DAY_NOT_FOUND", "m", { day: "d" }))).toBe("base day not found: d — check with list_base_days");
    expect(t(new CompositionError("BASE_DAY_CHANGED", "m", { day: "d", expected: 3, actual: 4 }))).toBe(
      "base day d block layout changed (slots: 4, required now: 3) — create a new composition",
    );
    expect(t(new CatalogMissingError("x"))).toBe("Catalog not found. Run npm run ingest first");
  });
});

describe("CompositionStore", () => {
  const comp: Composition = {
    schema_version: 1,
    id: "c1",
    name: "demo",
    base_day: "2026-04-15",
    revision: 2,
    slots: [[], [], [], []],
  };

  it("round-trips, lists, and throws COMPOSITION_NOT_FOUND", () => {
    const s = new CompositionStore(dir);
    expect(s.list()).toEqual([]);
    s.save(comp);
    expect(s.load("c1")).toEqual(comp);
    expect(s.list()).toEqual([{ id: "c1", name: "demo", base_day: "2026-04-15", revision: 2 }]);
    expect(readdirSync(join(dir, "compositions")).filter((f) => f.includes(".tmp-"))).toEqual([]);
    try {
      s.load("zzz");
      throw new Error("expected throw");
    } catch (e) {
      expect(e).toBeInstanceOf(CompositionError);
      expect((e as CompositionError).code).toBe("COMPOSITION_NOT_FOUND");
      expect((e as CompositionError).details).toEqual({ id: "zzz" });
    }
  });

  it("writes exports atomically as <id>-r<rev>.manifest.json", () => {
    const s = new CompositionStore(dir);
    const m = { schema_version: 1, composition_id: "c1", composition_revision: 2 } as unknown as Manifest;
    const p = s.writeExport(m);
    expect(p).toBe(join(dir, "exports", "c1-r2.manifest.json"));
    expect(existsSync(p)).toBe(true);
  });
});
