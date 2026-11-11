import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { Annotation, Clip, Dataset } from "@vb/core";
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

const clip = (
  id: string,
  category: Clip["category"],
  videoId = "vid00000001",
  dataset: Dataset = "youtube",
): Clip => ({
  clip_id: id,
  category,
  setting: "home_indoor",
  node: "kitchen",
  activity: "cooking",
  time_of_day: "unknown",
  tags: [],
  source: { dataset, video_id: videoId, media: { kind: "video", fps: 30 } },
  source_start_s: 0,
  source_end_s: 60,
  day_group: "g1",
  person_source: "p1",
  fingerprint: null,
  extractor_version: "test",
});

describe("CatalogWriter", () => {
  it("writes annotations and an index with per-dataset and summed category counts", () => {
    const w = new CatalogWriter(dir);
    w.writeAnnotations("nextme800", "2026-04-15_0900-1000", [ann("a1", "x"), ann("a2", "y")]);
    w.writeClipCatalog("ego4d", [clip("ego4d:u:0.000-60.000", "hobby", "u", "ego4d")], new Map());
    w.writeClipCatalog("youtube", [clip("c2", "hobby"), clip("c3", "home")], new Map());
    rebuildIndex(dir, new Date("2026-11-03T12:00:00.000Z"));

    const cat = join(dir, "catalog");
    const annLines = readFileSync(join(cat, "nextme800/annotations/2026-04-15_0900-1000.jsonl"), "utf8")
      .trimEnd()
      .split("\n");
    expect(annLines.map((l) => (JSON.parse(l) as Annotation).id)).toEqual(["a1", "a2"]);

    expect(JSON.parse(readFileSync(join(cat, "index.json"), "utf8"))).toEqual({
      updated_at: "2026-11-03T12:00:00.000Z",
      categories: { hobby: 2, home: 1 },
      clips: { ego4d: { hobby: 1 }, youtube: { hobby: 1, home: 1 } },
    });
  });

  it("is idempotent: same input gives byte-identical files", () => {
    const w = new CatalogWriter(dir);
    const clips = [clip("ego4d:u:0.000-60.000", "hobby", "u", "ego4d")];
    w.writeClipCatalog("ego4d", clips, new Map());
    const read = () => readFileSync(join(dir, "catalog/ego4d/clips.jsonl"), "utf8");
    const first = read();
    w.writeClipCatalog("ego4d", clips, new Map());
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

  it("rejects path-like video ids", () => {
    const w = new CatalogWriter(dir);
    expect(() => w.writeAnnotations("nextme800", "../evil", [])).toThrow(/video id/i);
  });

  describe("writeClipCatalog", () => {
    it("writes clips.jsonl and annotations, and counts clips per dataset and category in the index", () => {
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

    it("removes stale annotation files, leaves other files and datasets alone, and validates before writing", () => {
      const w = new CatalogWriter(dir);
      w.writeAnnotations("nextme800", "keep", [ann("k", "x")]);
      w.writeClipCatalog("youtube", [clip("c1", "home")], new Map([["old", [ann("a1", "x")]]]));
      const annDir = join(dir, "catalog/youtube/annotations");
      writeFileSync(join(annDir, "notes.txt"), "keep me");
      w.writeClipCatalog("youtube", [clip("c1", "home")], new Map([["new", [ann("a1", "x")]]]));
      expect(readdirSync(annDir).sort()).toEqual(["new.jsonl", "notes.txt"]);
      expect(existsSync(join(dir, "catalog/nextme800/annotations/keep.jsonl"))).toBe(true);
      expect(() =>
        w.writeClipCatalog("youtube", [{ ...clip("c2", "home"), source_end_s: -1 }], new Map()),
      ).toThrow();
      expect(() => w.writeClipCatalog("youtube", [], new Map([["../evil", []]]))).toThrow(/video id/i);
      expect(readdirSync(annDir).sort()).toEqual(["new.jsonl", "notes.txt"]);
      expect(readFileSync(join(dir, "catalog/youtube/clips.jsonl"), "utf8")).toContain('"c1"');
    });

    it("writes an empty clips.jsonl and removes every annotation file for an empty catalog", () => {
      const w = new CatalogWriter(dir);
      w.writeClipCatalog("youtube", [clip("c1", "home")], new Map([["old", [ann("a1", "x")]]]));
      w.writeClipCatalog("youtube", [], new Map());
      expect(readdirSync(join(dir, "catalog/youtube/annotations"))).toEqual([]);
      expect(readFileSync(join(dir, "catalog/youtube/clips.jsonl"), "utf8")).toBe("");
    });
  });
});
