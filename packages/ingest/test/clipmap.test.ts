import { mkdtempSync, readFileSync, rmSync, existsSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Annotation } from "@vb/core";
import type { BlockCandidate } from "../src/blocks.js";
import { loadCategories, type Category } from "../src/categories.js";
import { candidatesToClips, candidateToClip } from "../src/clipmap.js";
import { ingestEgo4d } from "../src/cli.js";
import { newReport } from "../src/jsonl.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const CATS_PATH = resolve(HERE, "..", "..", "..", "config", "categories.json");
const FIX = join(HERE, "fixtures", "ego4d");

const cand = (category: string, videoId = "take-a"): BlockCandidate => ({
  id: `egoexo4d:${videoId}:10.000-120.000`,
  role: "event",
  category,
  source: { dataset: "egoexo4d", video_id: videoId, media: { kind: "video", fps: 30, path: "takes/x.mp4" } },
  src_start: 10,
  src_end: 120,
  summary: "Rock climbing: route one",
});

const climbing = () => loadCategories(CATS_PATH).categories.find((c) => c.id === "sport_climbing") as Category;
const ann = (id: string): Annotation => ({
  id,
  track: "captions",
  level: "L4",
  time: { kind: "point", t: 12 },
  text: "C climbs.",
  original: { id, text: "C climbs." },
});

let dir: string;
beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "vb-ingest-clipmap-"));
});
afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
  vi.restoreAllMocks();
});

describe("candidateToClip", () => {
  it("climbing candidate becomes hobby/venue_indoor with tags [\"sport_climbing\"]", () => {
    const clipCfg = climbing().clip;
    expect(clipCfg).toBeDefined();
    const clip = candidateToClip(cand("sport_climbing"), clipCfg!);
    expect(clip).toMatchObject({
      clip_id: "egoexo4d:take-a:10.000-120.000",
      category: "hobby",
      setting: "venue_indoor",
      node: "hobby",
      activity: "Rock climbing: route one",
      time_of_day: "unknown",
      tags: ["sport_climbing"],
      source_start_s: 10,
      source_end_s: 120,
      day_group: "egoexo4d:take-a",
      person_source: "egoexo4d:take-a",
      fingerprint: null,
      extractor_version: "egoexo4d-ingest.v1",
    });
    expect(clip.source.video_id).toBe("take-a");
  });

  it("maps the shipped categories as agreed", () => {
    const cats = loadCategories(CATS_PATH).categories;
    const got = Object.fromEntries(cats.map((c) => [c.id, c.clip ? [c.clip.category, c.clip.setting, c.clip.node] : null]));
    expect(got).toEqual({
      meal_out: ["meal", "venue_indoor", "eat_out"],
      hobby_guitar_e4d: ["hobby", "home_indoor", "hobby"],
      walking: ["hobby", "outdoor", "hobby"],
      hobby_tent: ["hobby", "outdoor", "hobby"],
      sport_soccer: ["hobby", "outdoor", "hobby"],
      sport_climbing: ["hobby", "venue_indoor", "hobby"],
      sport_basketball: ["hobby", "venue_indoor", "hobby"],
      hobby_dance: ["hobby", "venue_indoor", "hobby"],
      hobby_music: ["hobby", "venue_indoor", "hobby"],
      hobby_guitar: ["hobby", "venue_indoor", "hobby"],
      transit: ["transit", "outdoor", "transit"],
      hobby_lego: null,
      cooking: null,
      health: null,
      bike_repair: null,
    });
  });
});

describe("candidatesToClips", () => {
  it("cooking candidate is skipped as no_clip_mapping, with its annotations", () => {
    const cats = loadCategories(CATS_PATH).categories;
    const report = newReport();
    const { clips, annotations } = candidatesToClips(
      [cand("cooking", "take-c"), cand("sport_climbing", "take-a")],
      new Map([
        ["take-c", [ann("c1")]],
        ["take-a", [ann("a1")]],
      ]),
      cats,
      report,
    );
    expect(clips.map((c) => c.source.video_id)).toEqual(["take-a"]);
    expect([...annotations.keys()]).toEqual(["take-a"]);
    expect(report.skipped).toEqual({ no_clip_mapping: 1 });
  });
});

describe("ego4d ingest", () => {
  it("ego4d ingest writes clips.jsonl and no candidates.jsonl", () => {
    ingestEgo4d({
      metadataPath: join(FIX, "ego4d.json"),
      narrationsPath: join(FIX, "narration.json"),
      categoriesPath: CATS_PATH,
      dataDir: dir,
      now: new Date("2026-11-12T00:00:00Z"),
    });
    const cat = join(dir, "catalog", "ego4d");
    expect(existsSync(join(cat, "clips.jsonl"))).toBe(true);
    expect(existsSync(join(cat, "candidates.jsonl"))).toBe(false);
    const rows = readFileSync(join(cat, "clips.jsonl"), "utf8").trim().split("\n").map((l) => JSON.parse(l));
    expect(rows.length).toBeGreaterThan(0);
    expect(rows.every((r) => r.category === "hobby" || r.category === "transit")).toBe(true);
    expect(readdirSync(join(cat, "annotations")).length).toBeGreaterThan(0);
  });
});
