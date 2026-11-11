import { mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { ingestBank } from "../src/bank.js";
import { IngestFormatError } from "../src/jsonl.js";
import { CatalogWriter, isSafeVideoIdSegment, rebuildIndex } from "../src/writer.js";

let dir: string;
beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "vb-ingest-bank-"));
});
afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

const row = (clipId: string, over: Record<string, unknown> = {}, captions: unknown[] = []) => ({
  clip_id: clipId,
  category: "home",
  setting: "home_indoor",
  node: "kitchen",
  activity: "cooking",
  time_of_day: "unknown",
  tags: [],
  source: { dataset: "youtube", video_id: "fakeVid0001", media: { kind: "video", fps: 30 } },
  source_start_s: 0,
  source_end_s: 60,
  day_group: "g1",
  person_source: "p1",
  fingerprint: null,
  extractor_version: "test",
  captions,
  ...over,
});
const cap = (id: string, text: string) => ({
  id,
  level: "L4",
  time: { kind: "interval", start: 0, end: 10 },
  text,
});
const text = (...rows: unknown[]) => rows.map((r) => JSON.stringify(r)).join("\n") + "\n";

describe("ingestBank", () => {
  it("two rows of one video share one annotation file", () => {
    const t = text(
      row("c1", {}, [cap("k1", "a"), cap("k2", "b")]),
      row("c2", { source_start_s: 60, source_end_s: 120 }, [cap("k2", "b"), cap("k3", "c")]),
    );
    const { clips, annotations, report } = ingestBank(t, "bank.jsonl", "youtube");
    expect(clips).toHaveLength(2);
    expect(clips[0]).not.toHaveProperty("captions");
    expect([...annotations.keys()]).toEqual(["fakeVid0001"]);
    expect(annotations.get("fakeVid0001")?.map((a) => a.id)).toEqual([
      "youtube:fakeVid0001:k1",
      "youtube:fakeVid0001:k2",
      "youtube:fakeVid0001:k3",
    ]);
    expect(annotations.get("fakeVid0001")?.[0]).toMatchObject({ track: "captions", level: "L4", text: "a", original: cap("k1", "a") });
    expect(report.ingested).toEqual({ home: 2 });

    new CatalogWriter(dir).writeClipCatalog("youtube", clips, annotations);
    expect(readdirSync(join(dir, "catalog/youtube/annotations"))).toEqual(["fakeVid0001.jsonl"]);
  });

  it("unlabeled row is skipped and counted (report.skipped.unlabeled === 1)", () => {
    const draft = row("c9", { category: null, setting: null, node: null });
    const { clips, report } = ingestBank(text(row("c1"), draft), "bank.jsonl", "youtube");
    expect(clips).toHaveLength(1);
    expect(report.skipped.unlabeled).toBe(1);
  });

  it("bad row reports file:line", () => {
    const t = text(row("c1"), row("c2", { source_end_s: "x" }));
    expect(() => ingestBank(t, "bank.jsonl", "youtube")).toThrow(IngestFormatError);
    expect(() => ingestBank(t, "bank.jsonl", "youtube")).toThrow(/^bank\.jsonl:2:/);
  });

  it("dataset mismatch throws", () => {
    expect(() => ingestBank(text(row("c1")), "bank.jsonl", "nextme800")).toThrow(/bank\.jsonl:1: .*does not match/);
  });

  it("duplicate clip_id throws", () => {
    expect(() => ingestBank(text(row("c1"), row("c1")), "bank.jsonl", "youtube")).toThrow(/bank\.jsonl:2: duplicate clip_id/);
  });

  it("re-ingest removes annotation files of videos no longer present", () => {
    const w = new CatalogWriter(dir);
    const other = { source: { dataset: "youtube", video_id: "fakeVid0002", media: { kind: "video", fps: 30 } } };
    const first = ingestBank(text(row("c1", {}, [cap("k1", "a")]), row("c2", other, [cap("k1", "a")])), "b.jsonl", "youtube");
    w.writeClipCatalog("youtube", first.clips, first.annotations);
    const annDir = join(dir, "catalog/youtube/annotations");
    expect(readdirSync(annDir).sort()).toEqual(["fakeVid0001.jsonl", "fakeVid0002.jsonl"]);
    const second = ingestBank(text(row("c1", {}, [cap("k1", "a")])), "b.jsonl", "youtube");
    w.writeClipCatalog("youtube", second.clips, second.annotations);
    expect(readdirSync(annDir)).toEqual(["fakeVid0001.jsonl"]);
  });

  it("index.json has clips.youtube.home === 1", () => {
    const { clips, annotations } = ingestBank(text(row("c1")), "b.jsonl", "youtube");
    new CatalogWriter(dir).writeClipCatalog("youtube", clips, annotations);
    rebuildIndex(dir, new Date(0));
    const idx = JSON.parse(readFileSync(join(dir, "catalog/index.json"), "utf8")) as { clips: Record<string, Record<string, number>> };
    expect(idx.clips.youtube?.home).toBe(1);
  });

  it("a video id with leading '-' or '_' round-trips; path-like ids are rejected", () => {
    const id = "-abc_DEF123";
    const src = { source: { dataset: "youtube", video_id: id, media: { kind: "video", fps: 30 } } };
    const { clips, annotations } = ingestBank(text(row("c1", src, [cap("k1", "a")])), "b.jsonl", "youtube");
    new CatalogWriter(dir).writeClipCatalog("youtube", clips, annotations);
    expect(readdirSync(join(dir, "catalog/youtube/annotations"))).toEqual([`${id}.jsonl`]);
    expect(isSafeVideoIdSegment("_x-1")).toBe(true);
    expect(isSafeVideoIdSegment("../x")).toBe(false);
    expect(isSafeVideoIdSegment("a..b")).toBe(false);
    const bad = ingestBank(text(row("c1", { source: { ...src.source, video_id: "../x" } })), "b.jsonl", "youtube");
    expect(() => new CatalogWriter(dir).writeClipCatalog("youtube", bad.clips, bad.annotations)).toThrow(/Unsafe video id/);
  });
});
