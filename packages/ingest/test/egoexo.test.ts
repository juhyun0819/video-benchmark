import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { loadCategories, type Category } from "../src/categories.js";
import { ingestEgoExo, main } from "../src/cli.js";
import { buildEgo4dCatalog, parseEgo4dMeta, parseEgo4dNarrations } from "../src/ego4d/adapter.js";
import {
  buildEgoExoCatalog,
  parseEgoExoAtomic,
  parseEgoExoKeysteps,
  parseEgoExoTakes,
} from "../src/egoexo/adapter.js";
import { newReport } from "../src/jsonl.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const CATS_PATH = resolve(HERE, "..", "..", "..", "config", "categories.json");

const CATS: Category[] = [
  { id: "cooking", label: "Cooking", dataset: "egoexo4d", scenarios: ["Cooking"], keywords: [] },
  { id: "bike_repair", label: "Bike repair", dataset: "egoexo4d", scenarios: ["Bike Repair"], keywords: [] },
];

const take = (over: Record<string, unknown> = {}) => ({
  take_uid: "take-a",
  take_name: "lab_cooking_01_1",
  parent_task_name: "Cooking",
  task_name: "Cooking an Omelet",
  duration_sec: 130,
  task_start_sec: 10,
  task_end_sec: 120,
  root_dir: "takes/lab_cooking_01_1",
  frame_aligned_videos: {
    cam01: { "0": { relative_path: "frame_aligned_videos/cam01.mp4" } },
    aria01: { rgb: { relative_path: "frame_aligned_videos/aria01_214-1.mp4" } },
  },
  ...over,
});

const desc = (timestamp: number, text: string) => ({ text, timestamp, subject: "C", ego_visible: true, unsure: false });
const run = (descriptions: unknown[], rejected = false) => ({ annotation_uid: "ann", rejected, descriptions });
const seg = (start_time: number, end_time: number, step_name: string) => ({
  start_time,
  end_time,
  step_id: 1,
  step_name,
  step_description: `${step_name} carefully`,
  is_essential: true,
});

const takesOf = (...t: unknown[]) => parseEgoExoTakes(t, "takes.json");
const atomicOf = (annotations: Record<string, unknown[]>) => parseEgoExoAtomic({ annotations }, "atomic.json");
const keystepsOf = (annotations: Record<string, unknown[]>) =>
  parseEgoExoKeysteps(
    { annotations: Object.fromEntries(Object.entries(annotations).map(([k, segments]) => [k, { segments }])) },
    "keystep.json",
  );

const ATOMIC_A = { "take-a": [run([desc(12, "C cracks an egg.")])] };

function build(takes = takesOf(take()), atomic = atomicOf(ATOMIC_A), keysteps = keystepsOf({}), minSec?: number) {
  const report = newReport();
  return { ...buildEgoExoCatalog(takes, atomic, keysteps, CATS, report, minSec), report };
}

let dir: string;
beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "vb-ingest-egoexo-"));
});
afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
  vi.restoreAllMocks();
});

describe("buildEgoExoCatalog candidates", () => {
  it("makes one event candidate per take over its task span", () => {
    const { candidates } = build();
    expect(candidates).toEqual([
      {
        id: "egoexo4d:take-a:10.000-120.000",
        role: "event",
        category: "cooking",
        source: {
          dataset: "egoexo4d",
          video_id: "take-a",
          media: { kind: "video", fps: 30, path: "takes/lab_cooking_01_1/frame_aligned_videos/aria01_214-1.mp4" },
        },
        src_start: 10,
        src_end: 120,
        summary: "Cooking an Omelet (lab_cooking_01_1)",
      },
    ]);
  });

  it.each([
    ["missing task span", { task_start_sec: null, task_end_sec: null }],
    ["inverted task span", { task_start_sec: 90, task_end_sec: 40 }],
    ["task span starting past the video", { task_start_sec: 200, task_end_sec: 210 }],
    ["negative task start", { task_start_sec: -5, task_end_sec: 120 }],
  ])("falls back to the whole take for a %s", (_name, over) => {
    const { candidates, report } = build(takesOf(take(over)));
    expect(candidates.map((c) => [c.src_start, c.src_end])).toEqual([[0, 130]]);
    expect(report.ingested["egoexo4d:whole_take_span"]).toBe(1);
  });

  it("clips a task end past the video to its duration", () => {
    const { candidates } = build(takesOf(take({ task_end_sec: 500 })));
    expect(candidates.map((c) => [c.src_start, c.src_end])).toEqual([[10, 130]]);
  });

  it("measures the minimum length on the block span, not the video duration", () => {
    const short = build(takesOf(take({ task_start_sec: 10, task_end_sec: 69.9 })));
    expect(short.candidates).toEqual([]);
    expect(short.report.skipped["too_short"]).toBe(1);

    const exact = build(takesOf(take({ task_start_sec: 10, task_end_sec: 70 })));
    expect(exact.candidates).toHaveLength(1);
  });

  it("honours a custom minimum length", () => {
    const t = takesOf(take({ task_start_sec: 10, task_end_sec: 40 }));
    expect(build(t, atomicOf(ATOMIC_A), keystepsOf({}), 30).candidates).toHaveLength(1);
    expect(build(t, atomicOf(ATOMIC_A), keystepsOf({}), 31).candidates).toHaveLength(0);
  });

  it("skips a take with no ego rgb stream", () => {
    const { candidates, report } = build(
      takesOf(take({ frame_aligned_videos: { cam01: { "0": { relative_path: "x.mp4" } }, aria01: { slam: {} } } })),
    );
    expect(candidates).toEqual([]);
    expect(report.skipped["no_ego_video"]).toBe(1);
  });

  it("skips a take with no atomic description run", () => {
    const { candidates, annotations, report } = build(takesOf(take()), atomicOf({}));
    expect(candidates).toEqual([]);
    expect(annotations.size).toBe(0);
    expect(report.skipped["no_atomic"]).toBe(1);
  });

  it("treats a take whose runs are all rejected or empty as having no atomic descriptions", () => {
    const { candidates, report } = build(
      takesOf(take()),
      atomicOf({ "take-a": [run([desc(1, "bad")], true), run([])] }),
    );
    expect(candidates).toEqual([]);
    expect(report.skipped["no_atomic"]).toBe(1);
  });

  it("skips a take with a non-positive duration", () => {
    const { candidates, report } = build(takesOf(take({ duration_sec: 0 })));
    expect(candidates).toEqual([]);
    expect(report.skipped["invalid_time"]).toBe(1);
  });

  it("ignores takes in domains no category maps", () => {
    const { candidates, report } = build(takesOf(take({ parent_task_name: "Soccer" })));
    expect(candidates).toEqual([]);
    expect(report.skipped).toEqual({});
  });

  it("ignores atomic runs of takes missing from takes.json", () => {
    const { annotations } = build(
      takesOf(take()),
      atomicOf({ ...ATOMIC_A, "take-ghost": [run([desc(1, "ghost")])] }),
    );
    expect([...annotations.keys()]).toEqual(["take-a"]);
  });

  it("uses category keywords against the task name", () => {
    const cats: Category[] = [
      { id: "eggs", label: "Eggs", dataset: "egoexo4d", scenarios: ["Cooking"], keywords: ["omelet"] },
      { id: "salad", label: "Salad", dataset: "egoexo4d", scenarios: ["Cooking"], keywords: ["salad"] },
    ];
    const takes = takesOf(
      take({ take_uid: "take-a", task_name: "Cooking an OMELET" }),
      take({ take_uid: "take-b", task_name: "Making Noodles" }),
    );
    const atomic = atomicOf({ ...ATOMIC_A, "take-b": [run([desc(12, "C boils water.")])] });
    const { candidates } = buildEgoExoCatalog(takes, atomic, keystepsOf({}), cats, newReport());
    expect(candidates.map((c) => [c.source.video_id, c.category])).toEqual([["take-a", "eggs"]]);
  });

  it("does not match Ego4D categories even when the scenario name is the same", () => {
    const cats: Category[] = [{ id: "ego", label: "E", dataset: "ego4d", scenarios: ["Cooking"], keywords: [] }];
    const { candidates } = buildEgoExoCatalog(takesOf(take()), atomicOf(ATOMIC_A), keystepsOf({}), cats, newReport());
    expect(candidates).toEqual([]);
  });
});

describe("Ego4D adapter next to Ego-Exo4D categories", () => {
  it("does not match an Ego-Exo4D category on a shared scenario name", () => {
    const meta = parseEgo4dMeta({ videos: [{ video_uid: "v1", scenarios: ["Cooking"] }] }, "m.json");
    const narrations = parseEgo4dNarrations(
      { v1: { narration_pass_1: { narrations: [], summaries: [{ start_sec: 0, end_sec: 90, summary_text: "C cooks" }] } } },
      "n.json",
    );
    const { candidates } = buildEgo4dCatalog(meta, narrations, CATS, newReport());
    expect(candidates).toEqual([]);
  });
});

describe("buildEgoExoCatalog summary", () => {
  it("adds the first three distinct keysteps in time order", () => {
    const keysteps = keystepsOf({
      "take-a": [seg(40, 50, "Fry"), seg(10, 20, "Crack eggs"), seg(20, 30, "Whisk"), seg(25, 28, "Crack eggs"), seg(60, 70, "Serve")],
    });
    const { candidates } = build(takesOf(take()), atomicOf(ATOMIC_A), keysteps);
    expect(candidates[0]?.summary).toBe("Cooking an Omelet: Crack eggs, Whisk, Fry");
  });

  it("ignores keysteps with an invalid span", () => {
    const keysteps = keystepsOf({ "take-a": [seg(30, 30, "Broken"), seg(10, 20, "Whisk")] });
    const { candidates } = build(takesOf(take()), atomicOf(ATOMIC_A), keysteps);
    expect(candidates[0]?.summary).toBe("Cooking an Omelet: Whisk");
  });
});

describe("buildEgoExoCatalog annotations", () => {
  it("copies atomic descriptions verbatim as points from the first accepted run only", () => {
    const first = desc(12, "C cracks an egg.");
    const { annotations } = build(
      takesOf(take()),
      atomicOf({
        "take-a": [run([desc(1, "rejected")], true), run([first, desc(14.5, "C whisks.")]), run([desc(13, "second annotator")])],
      }),
    );
    expect(annotations.get("take-a")).toEqual([
      {
        id: "egoexo4d:take-a:a:0",
        track: "captions",
        level: "atomic",
        time: { kind: "point", t: 12 },
        text: "C cracks an egg.",
        original: first,
      },
      {
        id: "egoexo4d:take-a:a:1",
        track: "captions",
        level: "atomic",
        time: { kind: "point", t: 14.5 },
        text: "C whisks.",
        original: desc(14.5, "C whisks."),
      },
    ]);
  });

  it("copies keysteps verbatim as intervals and counts the invalid ones", () => {
    const whisk = seg(20, 30, "Whisk");
    const { annotations, report } = build(
      takesOf(take()),
      atomicOf(ATOMIC_A),
      keystepsOf({ "take-a": [seg(30, 30, "Broken"), whisk] }),
    );
    expect(annotations.get("take-a")?.filter((a) => a.level === "keystep")).toEqual([
      {
        id: "egoexo4d:take-a:k:1",
        track: "captions",
        level: "keystep",
        time: { kind: "interval", start: 20, end: 30 },
        text: "Whisk",
        original: whisk,
      },
    ]);
    expect(report.skipped["end_before_start"]).toBe(1);
  });

  it("reports what was ingested", () => {
    const { report } = build(
      takesOf(take()),
      atomicOf(ATOMIC_A),
      keystepsOf({ "take-a": [seg(20, 30, "Whisk")] }),
    );
    expect(report.ingested).toEqual({ "egoexo4d:atomic": 1, "egoexo4d:keysteps": 1, "egoexo4d:candidates": 1 });
  });
});

describe("parsers", () => {
  it("names the file and the key path of a malformed take", () => {
    expect(() => parseEgoExoTakes([{ take_uid: 1 }], "/x/takes.json")).toThrow(/\/x\/takes\.json: 0\.take_uid/);
  });

  it("names the file of malformed annotations", () => {
    expect(() => parseEgoExoAtomic({}, "/x/atomic.json")).toThrow(/\/x\/atomic\.json: annotations/);
    expect(() => parseEgoExoKeysteps({}, "/x/keystep.json")).toThrow(/\/x\/keystep\.json: annotations/);
  });
});

describe("shipped categories", () => {
  it("maps every Ego-Exo4D domain, with Cooking sharing the cooking category", () => {
    const egoexo = loadCategories(CATS_PATH).categories.filter((c) => c.dataset === "egoexo4d");
    const byDomain = Object.fromEntries(egoexo.flatMap((c) => c.scenarios.map((s) => [s, c.id])));
    expect(Object.keys(byDomain).sort()).toEqual([
      "Basketball",
      "Bike Repair",
      "Cooking",
      "Dance",
      "Health",
      "Music",
      "Rock Climbing",
      "Soccer",
    ]);
    expect(byDomain["Cooking"]).toBe("cooking");
  });
});

function writeRoot(root: string): void {
  mkdirSync(join(root, "annotations"), { recursive: true });
  const write = (name: string, value: unknown) => writeFileSync(join(root, name), JSON.stringify(value));
  write("takes.json", [
    take({ take_uid: "take-a" }),
    take({ take_uid: "take-b", parent_task_name: "Bike Repair", task_name: "Remove a Wheel" }),
  ]);
  write("annotations/atomic_descriptions_train.json", { annotations: ATOMIC_A });
  write("annotations/atomic_descriptions_val.json", {
    annotations: { "take-b": [run([desc(15, "C loosens the bolt.")])] },
  });
  write("annotations/keystep_train.json", { annotations: { "take-a": { segments: [seg(20, 30, "Whisk")] } } });
  write("annotations/keystep_val.json", { annotations: {} });
}

describe("ingestEgoExo", () => {
  const CATS_FILE = () => {
    const path = join(dir, "categories.json");
    writeFileSync(path, JSON.stringify({ categories: CATS }));
    return path;
  };

  it("merges train and val and writes candidates, annotations and the index", () => {
    const root = join(dir, "egoexo");
    writeRoot(root);
    const dataDir = join(dir, "data");

    const report = ingestEgoExo({
      rootDir: root,
      categoriesPath: CATS_FILE(),
      dataDir,
      now: new Date("2026-11-07T00:00:00Z"),
    });

    expect(report.ingested["egoexo4d:candidates"]).toBe(2);
    const catalog = join(dataDir, "catalog");
    expect(readFileSync(join(catalog, "egoexo4d", "candidates.jsonl"), "utf8").trimEnd().split("\n")).toHaveLength(2);
    expect(readFileSync(join(catalog, "egoexo4d", "annotations", "take-b.jsonl"), "utf8")).toContain("C loosens the bolt.");
    expect(JSON.parse(readFileSync(join(catalog, "index.json"), "utf8")).categories).toEqual({ bike_repair: 1, cooking: 1 });
  });

  it("writes nothing when an annotation file is missing, naming the file", () => {
    const root = join(dir, "egoexo");
    writeRoot(root);
    rmSync(join(root, "annotations", "keystep_val.json"));
    const dataDir = join(dir, "data");

    expect(() => ingestEgoExo({ rootDir: root, categoriesPath: CATS_FILE(), dataDir, now: new Date() })).toThrow(
      /keystep_val\.json/,
    );
    expect(existsSync(join(dataDir, "catalog"))).toBe(false);
  });
});

describe("cli egoexo", () => {
  it("requires --root", async () => {
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    await expect(main(["egoexo"], {})).resolves.toBe(1);
    expect(err.mock.calls[0]?.[0]).toMatch(/--root/);
  });

  it("rejects a --min-sec that is not a non-negative number", async () => {
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    await expect(main(["egoexo", "--root", dir, "--min-sec", "soon"], {})).resolves.toBe(1);
    expect(err.mock.calls[0]?.[0]).toMatch(/--min-sec/);
  });

  it("ingests with a custom --min-sec and prints the report", async () => {
    const root = join(dir, "egoexo");
    writeRoot(root);
    const cats = join(dir, "categories.json");
    writeFileSync(cats, JSON.stringify({ categories: CATS }));
    const log = vi.spyOn(console, "log").mockImplementation(() => {});

    const code = await main(
      ["egoexo", "--root", root, "--categories", cats, "--data", join(dir, "data"), "--min-sec", "111"],
      {},
    );

    expect(code).toBe(0);
    expect(JSON.parse(log.mock.calls[0]?.[0] as string).skipped).toEqual({ too_short: 2 });
  });
});
