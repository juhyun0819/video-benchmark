import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AnnotationSchema, BlockCandidateSchema } from "@vb/core";
import { loadCategories } from "../src/categories.js";
import { ingestEgo4d, main } from "../src/cli.js";
import {
  buildEgo4dCatalog,
  listScenarios,
  parseEgo4dMeta,
  parseEgo4dNarrations,
  readJsonFile,
} from "../src/ego4d/adapter.js";
import { newReport } from "../src/jsonl.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIX = join(HERE, "fixtures", "ego4d");
const META_PATH = join(FIX, "ego4d.json");
const NARR_PATH = join(FIX, "narration.json");
const CATS_PATH = resolve(HERE, "..", "..", "..", "config", "categories.json");
const V1 = "aaaaaaaa-0000-0000-0000-000000000001";
const V2 = "bbbbbbbb-0000-0000-0000-000000000002";
const V3 = "cccccccc-0000-0000-0000-000000000003";

const meta = () => parseEgo4dMeta(JSON.parse(readFileSync(META_PATH, "utf8")), META_PATH);
const narrations = () => parseEgo4dNarrations(JSON.parse(readFileSync(NARR_PATH, "utf8")), NARR_PATH);
const categories = () => loadCategories(CATS_PATH);

let dir: string;
beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "vb-ingest-ego4d-"));
});
afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
  vi.restoreAllMocks();
});

describe("loadCategories", () => {
  it("loads the shipped config with Korean labels", () => {
    const cats = categories().categories;
    expect(cats.map((c) => c.id)).toEqual(["walking", "transit", "hobby_lego", "hobby_tent"]);
    expect(cats.map((c) => c.label)).toEqual(["산책·보행", "차량·대중교통 이동", "취미: 레고", "취미: 텐트"]);
    expect(cats[2]).toMatchObject({ scenarios: ["Playing board games"], keywords: ["lego"] });
  });

  it("rejects categories.json with empty scenarios, naming the file", () => {
    const bad = join(dir, "categories.json");
    writeFileSync(
      bad,
      JSON.stringify({ categories: [{ id: "x", label: "X", dataset: "ego4d", scenarios: [], keywords: [] }] }),
    );
    expect(() => loadCategories(bad)).toThrow(bad);
    expect(() => loadCategories(bad)).toThrow(/categories\.0\.scenarios/);
  });
});

describe("listScenarios", () => {
  it("lists scenarios with counts, sorted by count desc then name", () => {
    const m = parseEgo4dMeta(
      {
        videos: [
          { video_uid: "a", scenarios: ["Cooking", "Camping"] },
          { video_uid: "b", scenarios: ["Cooking", "Cooking"] },
          { video_uid: "c", scenarios: ["Baking"] },
        ],
      },
      "m.json",
    );
    expect(listScenarios(m)).toEqual([
      { scenario: "Cooking", videos: 2 },
      { scenario: "Baking", videos: 1 },
      { scenario: "Camping", videos: 1 },
    ]);
  });
});

describe("buildEgo4dCatalog", () => {
  function build() {
    const report = newReport();
    const out = buildEgo4dCatalog(meta(), narrations(), categories(), report);
    return { ...out, report };
  }

  it("creates candidates only for matching scenario and keyword", () => {
    const { candidates } = build();
    expect(candidates.map((c) => [c.id, c.category])).toEqual([
      [`ego4d:${V1}:0.000-30.000`, "walking"],
      [`ego4d:${V1}:60.000-90.500`, "walking"],
      // keyword match is case-insensitive; "lego" + "tent" both present -> first category in config order
      [`ego4d:${V2}:0.000-45.000`, "hobby_lego"],
      [`ego4d:${V2}:60.000-120.000`, "hobby_lego"],
      [`ego4d:${V2}:120.000-150.000`, "hobby_tent"],
    ]);
    for (const c of candidates) BlockCandidateSchema.parse(c);
    expect(candidates[0]).toMatchObject({
      role: "event",
      source: { dataset: "ego4d", video_id: V1, media: { kind: "video", fps: 30 } },
      src_start: 0,
      src_end: 30,
      summary: "C walked along a quiet street.",
    });
    // "C played a card game." matches the scenario but no keyword
    expect(candidates.some((c) => c.summary.includes("card game"))).toBe(false);
  });

  it("keeps narrations as points, never intervals, and summaries as intervals", () => {
    const { annotations } = build();
    const v1 = annotations.get(V1) ?? [];
    for (const a of v1) AnnotationSchema.parse(a);
    const narr = v1.filter((a) => a.level === "narration");
    expect(narr.map((a) => a.id)).toEqual([`ego4d:${V1}:n:0`, `ego4d:${V1}:n:1`, `ego4d:${V1}:n:2`]);
    expect(narr.every((a) => a.time.kind === "point")).toBe(true);
    expect(narr[1]).toMatchObject({
      track: "captions",
      time: { kind: "point", t: 11.25 },
      text: "#C C looks at the road",
    });
    expect(narr[1]?.original).toMatchObject({ annotation_uid: "n-a-1", _unmapped_timestamp_sec: 11.3 });
    const sums = v1.filter((a) => a.level === "summary");
    expect(sums.every((a) => a.time.kind === "interval")).toBe(true);
    // index keeps the position in the file; the end==start summary (index 2) is omitted
    expect(sums.map((a) => a.id)).toEqual([`ego4d:${V1}:s:0`, `ego4d:${V1}:s:1`, `ego4d:${V1}:s:3`]);
    expect(sums[0]).toMatchObject({ time: { kind: "interval", start: 0, end: 30 }, text: "C walked along a quiet street." });
  });

  it("includes all pass-1 content of a candidate video (not only the matching summaries), ignoring pass 2", () => {
    const { annotations } = build();
    const v2 = annotations.get(V2) ?? [];
    expect(v2.filter((a) => a.level === "summary")).toHaveLength(4);
    expect(v2.filter((a) => a.level === "narration")).toHaveLength(2);
    expect((annotations.get(V1) ?? []).some((a) => a.text.includes("pass two"))).toBe(false);
  });

  it("skips video without narration_pass_1, counts reason, and writes no annotations for it", () => {
    const { annotations, report } = build();
    expect(report.skipped.no_narration).toBe(1);
    expect(annotations.has(V3)).toBe(false);
    expect([...annotations.keys()].sort()).toEqual([V1, V2]);
  });

  it("counts duplicate_span (keep first) and end_before_start, and totals", () => {
    const { report, candidates } = build();
    expect(report.skipped).toEqual({ duplicate_span: 1, end_before_start: 1, no_narration: 1 });
    expect(report.ingested["ego4d:candidates"]).toBe(5);
    expect(report.ingested["ego4d:narrations"]).toBe(5);
    expect(candidates.find((c) => c.id === `ego4d:${V1}:0.000-30.000`)?.summary).toBe("C walked along a quiet street.");
  });

  it("skips narrations with non-finite timestamps as invalid_time", () => {
    const m = parseEgo4dMeta({ videos: [{ video_uid: "v", scenarios: ["Walking on street"] }] }, "m.json");
    const n = {
      v: {
        narration_pass_1: {
          narrations: [
            { timestamp_sec: Number.NaN, narration_text: "bad" },
            { timestamp_sec: 3, narration_text: "good" },
          ],
          summaries: [{ start_sec: 0, end_sec: 10, summary_text: "s" }],
        },
      },
    } as unknown as ReturnType<typeof narrations>;
    const report = newReport();
    const { annotations } = buildEgo4dCatalog(m, n, categories(), report);
    expect(report.skipped.invalid_time).toBe(1);
    expect((annotations.get("v") ?? []).filter((a) => a.level === "narration").map((a) => a.text)).toEqual(["good"]);
  });

  it("is deterministic", () => {
    expect(JSON.stringify(build().candidates)).toBe(JSON.stringify(build().candidates));
  });
});

describe("buildEgo4dCatalog with ego4d.json video fields", () => {
  const WALK = ["Walking on street"];
  const summaries = (...spans: Array<[number, number]>) => ({
    v: {
      narration_pass_1: {
        narrations: [{ timestamp_sec: 1, narration_text: "#C C walks" }],
        summaries: spans.map(([start_sec, end_sec], i) => ({ start_sec, end_sec, summary_text: `walk ${i}` })),
      },
    },
  });
  function buildOne(video: Record<string, unknown>, spans: Array<[number, number]>) {
    const m = parseEgo4dMeta({ videos: [{ video_uid: "v", scenarios: WALK, ...video }] }, "m.json");
    const report = newReport();
    const out = buildEgo4dCatalog(m, parseEgo4dNarrations(summaries(...spans), "n.json"), categories(), report);
    return { ...out, report };
  }

  it("takes media fps from video_metadata.fps", () => {
    const { candidates } = buildOne({ video_metadata: { fps: 29.97, num_frames: 8991 } }, [[0, 10]]);
    expect(candidates[0]?.source.media).toEqual({ kind: "video", fps: 29.97 });
  });

  it("falls back to fps 30 when video_metadata or its fps is missing", () => {
    expect(buildOne({}, [[0, 10]]).candidates[0]?.source.media).toEqual({ kind: "video", fps: 30 });
    expect(buildOne({ video_metadata: {} }, [[0, 10]]).candidates[0]?.source.media).toEqual({ kind: "video", fps: 30 });
  });

  it("names the key path when video_metadata.fps is not a number", () => {
    expect(() =>
      parseEgo4dMeta({ videos: [{ video_uid: "v", scenarios: WALK, video_metadata: { fps: "30" } }] }, "m.json"),
    ).toThrow(/m\.json: videos\.0\.video_metadata\.fps/);
  });

  it("clips a summary end that runs past duration_sec, and the id uses the clipped end", () => {
    const { candidates, report } = buildOne({ duration_sec: 100 }, [[90, 100.4]]);
    expect(candidates.map((c) => [c.id, c.src_start, c.src_end])).toEqual([["ego4d:v:90.000-100.000", 90, 100]]);
    expect(report.skipped).toEqual({});
  });

  it("skips a summary starting at or after duration_sec as beyond_duration", () => {
    const { candidates, annotations, report } = buildOne({ duration_sec: 100 }, [[100, 120], [150, 160]]);
    expect(candidates).toEqual([]);
    expect(report.skipped).toEqual({ beyond_duration: 2 });
    expect(annotations.has("v")).toBe(false);
  });

  it("skips a summary overlapping a redacted interval as redacted, keeping ones that only touch it", () => {
    const { candidates, report } = buildOne(
      { has_redacted_regions: true, redacted_intervals: [{ start_sec: 20, end_sec: 30, start_frame: 600, end_frame: 900 }] },
      [[0, 20], [15, 25], [25, 40], [30, 50]],
    );
    expect(candidates.map((c) => [c.src_start, c.src_end])).toEqual([[0, 20], [30, 50]]);
    expect(report.skipped).toEqual({ redacted: 2 });
  });

  it("treats null redacted_intervals as no redaction", () => {
    expect(buildOne({ redacted_intervals: null }, [[0, 10]]).candidates).toHaveLength(1);
  });
});

describe("parse errors", () => {
  it("names the file and key path on a narration shape mismatch", () => {
    const bad = { [V1]: { narration_pass_1: { narrations: [{ timestamp_sec: "x", narration_text: "t" }], summaries: [] } } };
    expect(() => parseEgo4dNarrations(bad, "narr.json")).toThrow(/narr\.json/);
    expect(() => parseEgo4dNarrations(bad, "narr.json")).toThrow(new RegExp(`${V1}\\.narration_pass_1\\.narrations\\.0\\.timestamp_sec`));
  });

  it("names the file on a metadata shape mismatch", () => {
    expect(() => parseEgo4dMeta({ videos: [{ video_uid: "a" }] }, "meta.json")).toThrow(/meta\.json: videos\.0\.scenarios/);
  });

  it("readJsonFile exits on JSON parse failure naming the file", () => {
    const p = join(dir, "broken.json");
    writeFileSync(p, "{ not json");
    expect(() => readJsonFile(p)).toThrow(p);
  });

  it("readJsonFile refuses files over the size limit naming the file", () => {
    const p = join(dir, "big.json");
    writeFileSync(p, JSON.stringify({ a: "x".repeat(100) }));
    expect(() => readJsonFile(p, 50)).toThrow(p);
    expect(() => readJsonFile(p, 50)).toThrow(/512 MB|too large/);
  });
});

describe("ingestEgo4d / CLI", () => {
  it("writes candidates, per-video annotations and the index", () => {
    const report = ingestEgo4d({
      metadataPath: META_PATH,
      narrationsPath: NARR_PATH,
      categoriesPath: CATS_PATH,
      dataDir: dir,
      now: new Date("2026-11-03T00:00:00Z"),
    });
    expect(report.ingested["ego4d:candidates"]).toBe(5);
    const cat = join(dir, "catalog", "ego4d");
    const lines = readFileSync(join(cat, "candidates.jsonl"), "utf8").trim().split("\n");
    expect(lines).toHaveLength(5);
    expect(existsSync(join(cat, "annotations", `${V1}.jsonl`))).toBe(true);
    expect(existsSync(join(cat, "annotations", `${V2}.jsonl`))).toBe(true);
    expect(existsSync(join(cat, "annotations", `${V3}.jsonl`))).toBe(false);
    const index = JSON.parse(readFileSync(join(dir, "catalog", "index.json"), "utf8"));
    expect(index.categories).toMatchObject({ walking: 2, hobby_lego: 2, hobby_tent: 1 });
  });

  it("main --list-scenarios prints scenario<TAB>count and writes nothing", async () => {
    const out: string[] = [];
    vi.spyOn(console, "log").mockImplementation((s: unknown) => {
      out.push(String(s));
    });
    const code = await main(["ego4d", "--list-scenarios", "--metadata", META_PATH, "--data", dir], {});
    expect(code).toBe(0);
    expect(out).toEqual([
      "Camping\t1",
      "Car - commuting, road trip\t1",
      "Cooking\t1",
      "Playing board games\t1",
      "Walking on street\t1",
    ]);
    expect(existsSync(join(dir, "catalog"))).toBe(false);
  });

  it("main ego4d requires --metadata, and --narrations unless listing", async () => {
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    expect(await main(["ego4d", "--list-scenarios"], {})).toBe(1);
    expect(await main(["ego4d", "--metadata", META_PATH], {})).toBe(1);
    expect(err).toHaveBeenCalledTimes(2);
  });

  it("main ego4d ingests end to end with an explicit --categories", async () => {
    vi.spyOn(console, "log").mockImplementation(() => {});
    const code = await main(
      ["ego4d", "--metadata", META_PATH, "--narrations", NARR_PATH, "--categories", CATS_PATH, "--data", dir],
      {},
    );
    expect(code).toBe(0);
    expect(existsSync(join(dir, "catalog", "ego4d", "candidates.jsonl"))).toBe(true);
  });
});
