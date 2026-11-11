import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { Annotation, Clip } from "@vb/core";
import type { BlockCandidate } from "../src/legacy.js";
import { CatalogWriter, rebuildIndex } from "../src/writer.js";

let dir: string;
beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "vb-ingest-writer-"));
});
afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

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
    w.writeAnnotations("nextme800", "2026-04-15_0900-1000", [ann("a1", "x"), ann("a2", "y")]);
    w.writeCandidates("hdepic", [cand("hdepic:P01-x:0.000-60.000", "cooking"), cand("hdepic:P01-x:60.000-120.000", "cooking")]);
    w.writeCandidates("ego4d", [cand("ego4d:u:0.000-60.000", "walking")]);
    rebuildIndex(dir, new Date("2026-11-03T12:00:00.000Z"));

    const cat = join(dir, "catalog");
    const annLines = readFileSync(join(cat, "nextme800/annotations/2026-04-15_0900-1000.jsonl"), "utf8")
      .trimEnd()
      .split("\n");
    expect(annLines.map((l) => (JSON.parse(l) as Annotation).id)).toEqual(["a1", "a2"]);

    const candLines = readFileSync(join(cat, "hdepic/candidates.jsonl"), "utf8").trimEnd().split("\n");
    expect(candLines).toHaveLength(2);
    expect(existsSync(join(cat, "ego4d/candidates.jsonl"))).toBe(true);

    expect(JSON.parse(readFileSync(join(cat, "index.json"), "utf8"))).toEqual({
      updated_at: "2026-11-03T12:00:00.000Z",
      categories: { cooking: 2, walking: 1 },
      clips: {},
    });
  });

  it("is idempotent: same input gives byte-identical files", () => {
    const w = new CatalogWriter(dir);
    w.writeCandidates("ego4d", [cand("ego4d:u:0.000-60.000", "walking")]);
    const read = () => readFileSync(join(dir, "catalog/ego4d/candidates.jsonl"), "utf8");
    const first = read();
    w.writeCandidates("ego4d", [cand("ego4d:u:0.000-60.000", "walking")]);
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
      categories: {},
      clips: {},
    });
  });

  it("validates before writing and rejects path-like ids", () => {
    const w = new CatalogWriter(dir);
    expect(() => w.writeCandidates("ego4d", [{ nope: 1 } as unknown as BlockCandidate])).toThrow();
    expect(existsSync(join(dir, "catalog/ego4d/candidates.jsonl"))).toBe(false);
    expect(() => w.writeAnnotations("nextme800", "../evil", [])).toThrow(/video id/i);
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

  describe("writeClipCatalog", () => {
    const clip = (id: string, category: Clip["category"], videoId = "vid00000001"): Clip => ({
      clip_id: id,
      category,
      setting: "home_indoor",
      node: "kitchen",
      activity: "cooking",
      time_of_day: "unknown",
      tags: [],
      source: { dataset: "youtube", video_id: videoId, media: { kind: "video", fps: 30 } },
      source_start_s: 0,
      source_end_s: 60,
      day_group: "g1",
      person_source: "p1",
      fingerprint: null,
      extractor_version: "test",
    });

    it("writes clips.jsonl, annotations, and counts clips per dataset and category in the index", () => {
      const w = new CatalogWriter(dir);
      w.writeClipCatalog("youtube", [clip("c1", "home"), clip("c2", "home"), clip("c3", "work", "vid00000002")], new Map([["vid00000001", [ann("a1", "x")]]]));
      rebuildIndex(dir, new Date(0));
      const cat = join(dir, "catalog");
      expect(readFileSync(join(cat, "youtube/clips.jsonl"), "utf8").trimEnd().split("\n")).toHaveLength(3);
      expect(existsSync(join(cat, "youtube/annotations/vid00000001.jsonl"))).toBe(true);
      expect((JSON.parse(readFileSync(join(cat, "index.json"), "utf8")) as { clips: unknown }).clips).toEqual({
        youtube: { home: 2, work: 1 },
      });
    });

    it("removes stale annotation files and validates before writing", () => {
      const w = new CatalogWriter(dir);
      w.writeClipCatalog("youtube", [clip("c1", "home")], new Map([["old", [ann("a1", "x")]]]));
      w.writeClipCatalog("youtube", [clip("c1", "home")], new Map([["new", [ann("a1", "x")]]]));
      const annDir = join(dir, "catalog/youtube/annotations");
      expect(readdirSync(annDir)).toEqual(["new.jsonl"]);
      expect(() =>
        w.writeClipCatalog("youtube", [{ ...clip("c2", "home"), source_end_s: -1 }], new Map()),
      ).toThrow();
      expect(readdirSync(annDir)).toEqual(["new.jsonl"]);
      expect(readFileSync(join(dir, "catalog/youtube/clips.jsonl"), "utf8")).toContain('"c1"');
    });
  });
});
