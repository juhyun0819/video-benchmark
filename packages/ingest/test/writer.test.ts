import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { Annotation, BaseDay, BlockCandidate } from "@vb/core";
import { CatalogWriter, rebuildIndex } from "../src/writer.js";

let dir: string;
beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "vb-ingest-writer-"));
});
afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

const nextmeSource = {
  dataset: "nextme800",
  video_id: "2026-04-15_0900-1000",
  media: { kind: "frames", approx_fps: 1, tar: "raw/x.tar", start_wallclock: "2026-04-15 09:00:00" },
} as const;

const baseDay: BaseDay = {
  day: "2026-04-15",
  blocks: [
    {
      id: "nextme800:2026-04-15_0900-1000:0.000-600.000",
      role: "base",
      category: "base",
      source: nextmeSource,
      src_start: 0,
      src_end: 600,
      summary: "I cook.",
    },
  ],
};

const ann = (id: string, text: string): Annotation => ({
  id,
  track: "captions",
  level: "L4",
  time: { kind: "interval", start: 0, end: 10 },
  text,
  original: { id, text },
});

const cand = (id: string, category: string): BlockCandidate => ({
  id,
  role: "event",
  category,
  source: { dataset: "hdepic", video_id: "P01-x", media: { kind: "video", fps: 30 } },
  src_start: 0,
  src_end: 60,
  summary: category,
});

describe("CatalogWriter", () => {
  it("writes catalog layout and index", () => {
    const w = new CatalogWriter(dir);
    w.writeBaseDay(baseDay);
    w.writeBaseDay({ ...baseDay, day: "2026-04-14" });
    w.writeAnnotations("nextme800", "2026-04-15_0900-1000", [ann("a1", "x"), ann("a2", "y")]);
    w.writeCandidates("hdepic", [cand("hdepic:P01-x:0.000-60.000", "cooking"), cand("hdepic:P01-x:60.000-120.000", "cooking")]);
    w.writeCandidates("ego4d", [cand("ego4d:u:0.000-60.000", "walking")]);
    rebuildIndex(dir, new Date("2026-11-03T12:00:00.000Z"));

    const cat = join(dir, "catalog");
    expect(JSON.parse(readFileSync(join(cat, "nextme800/days/2026-04-15.json"), "utf8"))).toEqual(baseDay);

    const annLines = readFileSync(join(cat, "nextme800/annotations/2026-04-15_0900-1000.jsonl"), "utf8")
      .trimEnd()
      .split("\n");
    expect(annLines.map((l) => (JSON.parse(l) as Annotation).id)).toEqual(["a1", "a2"]);

    const candLines = readFileSync(join(cat, "hdepic/candidates.jsonl"), "utf8").trimEnd().split("\n");
    expect(candLines).toHaveLength(2);
    expect(existsSync(join(cat, "ego4d/candidates.jsonl"))).toBe(true);

    expect(JSON.parse(readFileSync(join(cat, "index.json"), "utf8"))).toEqual({
      updated_at: "2026-11-03T12:00:00.000Z",
      base_days: ["2026-04-14", "2026-04-15"],
      categories: { cooking: 2, walking: 1 },
    });
  });

  it("is idempotent: same input gives byte-identical files", () => {
    const w = new CatalogWriter(dir);
    const read = () => readFileSync(join(dir, "catalog/nextme800/days/2026-04-15.json"), "utf8");
    w.writeBaseDay(baseDay);
    const first = read();
    w.writeBaseDay(baseDay);
    expect(read()).toBe(first);
    rebuildIndex(dir, new Date(0));
    const idx = readFileSync(join(dir, "catalog/index.json"), "utf8");
    rebuildIndex(dir, new Date(0));
    expect(readFileSync(join(dir, "catalog/index.json"), "utf8")).toBe(idx);
  });

  it("rebuilds an empty index when nothing is ingested", () => {
    rebuildIndex(dir, new Date("2026-11-03T00:00:00.000Z"));
    expect(JSON.parse(readFileSync(join(dir, "catalog/index.json"), "utf8"))).toEqual({
      updated_at: "2026-11-03T00:00:00.000Z",
      base_days: [],
      categories: {},
    });
  });

  it("validates before writing and rejects path-like ids", () => {
    const w = new CatalogWriter(dir);
    expect(() => w.writeBaseDay({ day: "2026-04-15", blocks: [{ nope: 1 }] } as unknown as BaseDay)).toThrow();
    expect(existsSync(join(dir, "catalog/nextme800/days/2026-04-15.json"))).toBe(false);
    expect(() => w.writeAnnotations("nextme800", "../evil", [])).toThrow(/video id/i);
    expect(() => w.writeBaseDay({ ...baseDay, day: "../x" })).toThrow(/day/i);
  });
});

describe("CatalogWriter.writeEventCatalog", () => {
  const annDir = () => join(dir, "catalog", "ego4d", "annotations");
  const anns = (...ids: string[]) => new Map<string, Annotation[]>(ids.map((id) => [id, [ann(`${id}:a`, id)]]));

  it("writes candidates and annotations, and removes annotation files of videos it did not write", () => {
    const w = new CatalogWriter(dir);
    w.writeEventCatalog("ego4d", [cand("ego4d:u:0.000-60.000", "walking")], anns("old", "kept"));
    w.writeEventCatalog("ego4d", [cand("ego4d:u:0.000-60.000", "walking")], anns("kept", "new"));
    expect(readdirSync(annDir()).sort()).toEqual(["kept.jsonl", "new.jsonl"]);
    expect(readFileSync(join(dir, "catalog/ego4d/candidates.jsonl"), "utf8").trimEnd().split("\n")).toHaveLength(1);
  });

  it("removes every annotation file when the catalog is empty, and leaves other files alone", () => {
    const w = new CatalogWriter(dir);
    w.writeEventCatalog("ego4d", [], anns("old"));
    writeFileSync(join(annDir(), "notes.txt"), "keep me");
    w.writeEventCatalog("ego4d", [], new Map());
    expect(readdirSync(annDir())).toEqual(["notes.txt"]);
    expect(readFileSync(join(dir, "catalog/ego4d/candidates.jsonl"), "utf8")).toBe("");
  });

  it("does not touch annotations of other datasets", () => {
    const w = new CatalogWriter(dir);
    w.writeAnnotations("hdepic", "P01-x", [ann("h", "x")]);
    w.writeEventCatalog("ego4d", [], new Map());
    expect(existsSync(join(dir, "catalog/hdepic/annotations/P01-x.jsonl"))).toBe(true);
  });

  it("validates everything before deleting anything", () => {
    const w = new CatalogWriter(dir);
    w.writeEventCatalog("ego4d", [], anns("old"));
    expect(() => w.writeEventCatalog("ego4d", [], anns("../evil"))).toThrow(/video id/i);
    expect(() =>
      w.writeEventCatalog("ego4d", [{ nope: 1 } as unknown as BlockCandidate], new Map()),
    ).toThrow();
    expect(readdirSync(annDir())).toEqual(["old.jsonl"]);
  });

  it("refuses nextme800, whose ingest adds days one at a time", () => {
    const w = new CatalogWriter(dir);
    mkdirSync(join(dir, "catalog/nextme800/annotations"), { recursive: true });
    expect(() => w.writeEventCatalog("nextme800", [], new Map())).toThrow(/nextme800/);
  });
});
