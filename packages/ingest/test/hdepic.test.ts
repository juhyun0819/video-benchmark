import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AnnotationSchema, BlockCandidateSchema } from "@vb/core";
import { ingestHdepic, main } from "../src/cli.js";
import {
  buildHdepicCatalog,
  HdNarrationSchema,
  parseErratumCsv,
  parseHighLevel,
} from "../src/hdepic/adapter.js";
import { newReport, readJsonl, type IngestReport } from "../src/jsonl.js";

const FIX = join(dirname(fileURLToPath(import.meta.url)), "fixtures", "hdepic");
const ANN = join(FIX, "annotations");
const V1 = "P01-20990101-000001";
const V2 = "P01-20990101-000002";
const V3 = "P01-20990101-000003";

const read = (...p: string[]) => readFileSync(join(FIX, ...p), "utf8");
const segments = () =>
  parseHighLevel(read("annotations/high-level/activities/P01_recipe_timestamps.csv"), "P01.csv");
const narrations = () => readJsonl(read("HD_EPIC_Narrations.jsonl"), "n.jsonl", HdNarrationSchema);
const recipes = () => JSON.parse(read("annotations/high-level/complete_recipes.json"));
const erratum = () =>
  parseErratumCsv(read("annotations/narrations-and-action-segments/HD_EPIC_Narrations_erratum.csv"), "e.csv");

function build(report: IngestReport = newReport()) {
  const out = buildHdepicCatalog(segments(), narrations(), recipes(), report, erratum(), 0);
  return { ...out, report };
}

describe("parseHighLevel", () => {
  it("reads quoted comma lists and labels with the csv parser, keeping 'end' literal", () => {
    const segs = segments();
    expect(segs).toHaveLength(11);
    expect(segs[2]).toEqual({
      video_id: V1,
      start: 103,
      end: "end",
      label: "Plate toast, serve tea together",
      recipe_id: "R02, R01",
    });
    expect(segs[0]).toMatchObject({ start: 0, end: 48, recipe_id: "R01" });
    expect(segs[1]?.recipe_id).toBeUndefined();
    expect(Number.isNaN(segs[8]?.start)).toBe(true);
  });

  it("rejects a file without the expected columns", () => {
    expect(() => parseHighLevel("a,b\n1,2\n", "x.csv")).toThrow(/video_id/);
  });
});

describe("buildHdepicCatalog", () => {
  it("builds one cooking candidate per high-level segment with recipe name in summary", () => {
    const { candidates } = build();
    expect(candidates).toHaveLength(7);
    for (const c of candidates) BlockCandidateSchema.parse(c);
    const first = candidates.find((c) => c.id === `hdepic:${V1}:0.000-48.000`);
    expect(first).toEqual({
      id: `hdepic:${V1}:0.000-48.000`,
      role: "event",
      category: "cooking",
      source: { dataset: "hdepic", video_id: V1, media: { kind: "video", fps: 30 } },
      src_start: 0,
      src_end: 48,
      summary: "P01 · Make tea (Test Tea)",
    });
    // no recipe -> no parenthesis
    expect(candidates.find((c) => c.src_start === 48)?.summary).toBe("P01 · Wipe the counter");
    // multi-recipe row lists all names in listed order
    expect(candidates.find((c) => c.src_start === 103)?.summary).toBe(
      "P01 · Plate toast, serve tea together (Test Toast, Test Tea)",
    );
    // empty label falls back
    expect(candidates.find((c) => c.id === `hdepic:${V2}:0.000-100.000`)?.summary).toBe("P01 · (unlabeled)");
  });

  it("resolves literal end: next sorted row's start mid-video, max narration end for the last row", () => {
    const { candidates } = build();
    const ids = candidates.map((c) => c.id);
    expect(ids).toContain(`hdepic:${V1}:103.000-115.000`); // erratum (71) and backwards narration ignored
    expect(ids).toContain(`hdepic:${V2}:100.000-300.000`); // next row by start_time, not file order
    expect(ids).toContain(`hdepic:${V2}:300.000-420.500`);
  });

  it("reports skips: open_end, end_before_start, erratum, invalid_time", () => {
    const { report } = build();
    expect(report.skipped).toEqual({
      open_end: 1,
      end_before_start: 2, // one segment + one narration
      erratum: 1,
      invalid_time: 1,
      duplicate_span: 1,
    });
    expect(report.ingested["hdepic:candidates"]).toBe(7);
    expect(report.ingested["hdepic:narrations"]).toBe(5);
  });

  it("attaches narrations by video_id", () => {
    const { annotations } = build();
    expect([...annotations.keys()].sort()).toEqual([V1, V2]); // V3 has no candidate; V4 has no narrations; V9 has no segments
    const a1 = annotations.get(V1) ?? [];
    expect(a1.map((a) => a.id)).toEqual([`hdepic:${V1}-1`, `hdepic:${V1}-2`, `hdepic:${V1}-5`]);
    for (const a of a1) AnnotationSchema.parse(a);
    expect(a1[0]).toMatchObject({
      track: "captions",
      level: "narration",
      time: { kind: "interval", start: 5.5, end: 8.25 },
      text: " Pour water into the kettle.", // verbatim, leading space kept
    });
    expect(a1[0]?.original).toMatchObject({ unique_narration_id: `${V1}-1`, hands: ["left hand"] });
    // a narration outside every candidate is still kept (remap filters later)
    expect(annotations.get(V2)?.map((a) => a.id)).toEqual([`hdepic:${V2}-1`, `hdepic:${V2}-2`]);
    expect(V3 in Object.fromEntries(annotations)).toBe(false);
  });

  it("skips narration with end_before_start", () => {
    const { annotations } = build();
    expect(annotations.get(V1)?.some((a) => a.id === `hdepic:${V1}-4`)).toBe(false);
  });

  it("excludes erratum narrations", () => {
    const { annotations } = build();
    expect(annotations.get(V1)?.some((a) => a.id === `hdepic:${V1}-3`)).toBe(false);
  });

  it("merges rows with the same video and span into one candidate (recipes unioned)", () => {
    const { candidates } = build();
    const shared = candidates.filter((c) => c.id === "hdepic:P01-20990101-000004:10.000-20.000");
    expect(shared).toHaveLength(1);
    expect(shared[0]?.summary).toBe("P01 · Shared span (Test Tea, Test Toast)");
  });

  it("candidate ids are deterministic across runs", () => {
    const a = build().candidates.map((c) => c.id);
    const b = build().candidates.map((c) => c.id);
    expect(a).toEqual(b);
    expect(new Set(a).size).toBe(a.length);
  });
});

describe("ingestHdepic", () => {
  let dir: string;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "vb-ingest-hdepic-"));
  });
  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  const opts = () => ({
    annotationsDir: ANN,
    narrationsPath: join(FIX, "HD_EPIC_Narrations.jsonl"),
    dataDir: dir,
    now: new Date("2026-11-03T00:00:00.000Z"),
    minSec: 0,
  });

  it("writes candidates, per-video annotations and the index", () => {
    const report = ingestHdepic(opts());
    expect(report.ingested["hdepic:candidates"]).toBe(7);

    const cat = join(dir, "catalog");
    const cands = readFileSync(join(cat, "hdepic/candidates.jsonl"), "utf8").trim().split("\n");
    expect(cands).toHaveLength(7);
    expect(readFileSync(join(cat, `hdepic/annotations/${V1}.jsonl`), "utf8")).toContain(`hdepic:${V1}-1`);
    const index = JSON.parse(readFileSync(join(cat, "index.json"), "utf8"));
    expect(index.categories).toEqual({ cooking: 7 });
  });

  it("rerunning gives identical output (idempotent)", () => {
    ingestHdepic(opts());
    const path = join(dir, "catalog/hdepic/candidates.jsonl");
    const first = readFileSync(path, "utf8");
    ingestHdepic(opts());
    expect(readFileSync(path, "utf8")).toBe(first);
  });

  it("cli applies --min-sec and rejects a bad value", async () => {
    const log = vi.spyOn(console, "log").mockImplementation(() => {});
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    const args = ["hdepic", "--annotations", ANN, "--narrations", join(FIX, "HD_EPIC_Narrations.jsonl"), "--data", dir];

    await expect(main([...args, "--min-sec", "soon"], {})).resolves.toBe(1);
    expect(err.mock.calls[0]?.[0]).toMatch(/--min-sec/);

    await expect(main([...args, "--min-sec", "150"], {})).resolves.toBe(0);
    const ids = readFileSync(join(dir, "catalog/hdepic/candidates.jsonl"), "utf8");
    expect(ids).toContain(`hdepic:${V2}:100.000-300.000`);
    expect(ids).not.toContain(`hdepic:${V2}:0.000-100.000`);
    expect(JSON.parse(log.mock.calls[0]?.[0] as string).skipped["too_short"]).toBeGreaterThan(0);
    log.mockRestore();
    err.mockRestore();
  });
});

describe("buildHdepicCatalog minimum length", () => {
  const seg = (start: number, end: number | "end", video_id = "P01-x") => ({ video_id, start, end, label: "step" });
  const narr = (id: string, end_timestamp: number, video_id = "P01-x") => ({
    unique_narration_id: id,
    video_id,
    narration: "Stir the pan",
    start_timestamp: 1,
    end_timestamp,
  });

  it("drops candidates under 60 seconds by default and keeps exactly 60", () => {
    const report = newReport();
    const { candidates } = buildHdepicCatalog([seg(0, 59.9), seg(100, 160)], [], {}, report);
    expect(candidates.map((c) => c.id)).toEqual(["hdepic:P01-x:100.000-160.000"]);
    expect(report.skipped["too_short"]).toBe(1);
  });

  it("measures an open end after resolving it", () => {
    const short = newReport();
    expect(buildHdepicCatalog([seg(0, "end")], [narr("a", 30)], {}, short).candidates).toEqual([]);
    expect(short.skipped["too_short"]).toBe(1);

    const long = buildHdepicCatalog([seg(0, "end")], [narr("a", 90)], {}, newReport());
    expect(long.candidates.map((c) => c.src_end)).toEqual([90]);
  });

  it("keeps no annotations for a video left without candidates", () => {
    const { annotations } = buildHdepicCatalog(
      [seg(0, 30, "P01-short"), seg(0, 90, "P01-long")],
      [narr("a", 5, "P01-short"), narr("b", 5, "P01-long")],
      {},
      newReport(),
    );
    expect([...annotations.keys()]).toEqual(["P01-long"]);
  });

  it("honours a custom minimum length", () => {
    const segs = [seg(0, 30)];
    expect(buildHdepicCatalog(segs, [], {}, newReport(), new Set(), 30).candidates).toHaveLength(1);
    expect(buildHdepicCatalog(segs, [], {}, newReport(), new Set(), 31).candidates).toHaveLength(0);
  });
});
