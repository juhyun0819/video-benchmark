import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import type { Annotation, Clip, ClipCategory, Dataset, Setting } from "@vb/core";
import { CatalogWriter, rebuildIndex } from "@vb/ingest";
import { createServer, type ServerOptions } from "../src/server.js";

// Synthetic data only: nothing here is copied from data/.

export const REPO_ROOT = fileURLToPath(new URL("../../../", import.meta.url));
export const CONFIG_DIR = join(REPO_ROOT, "config");
export const EXAMPLE_WORLD = join(CONFIG_DIR, "worlds", "example-nextme.json");
export const PERSON_ID = "example-nextme";
export const FIXTURE_NOW = new Date("2026-11-12T00:00:00.000Z");
export const POOL_REL = join("distributions", "atus2025", "atus2025_stats_per_respondent.csv");

export function makeTempDir(): string {
  return mkdtempSync(join(tmpdir(), "vb-mcp-"));
}

/** Files directly under `<dataDir>/<sub>`, or [] when the folder does not exist. */
export function filesIn(dataDir: string, sub: string): string[] {
  const dir = join(dataDir, sub);
  return existsSync(dir) ? readdirSync(dir).sort() : [];
}

const ATUS_HEADER = [
  "id", "daytype", "occ", "w", "wake", "bed", "work_start", "work_end", "work_min", "worked",
  "breakfast_has", "breakfast_time", "breakfast_loc", "lunch_has", "lunch_time", "lunch_loc",
  "dinner_has", "dinner_time", "dinner_loc", "sport_has", "sport_time", "sport_outdoor",
  "house_min", "social_out",
];

/** A synthetic ATUS per-respondent pool: 40 weekday workers and 40 weekend rows, all occupation 2. */
export function atusCsv(): string {
  const lines = [ATUS_HEADER.join(",")];
  for (let i = 0; i < 40; i++) {
    lines.push(
      [
        `syn-wd-${i}`, "weekday", 2, 1 + (i % 3), 390 + (i % 5) * 5, 1380, 510 + (i % 4) * 10, 1050, 480, "True",
        "True", 420, 1, "True", 740, 2, "True", 1120, 1, "False", "", "", 30, "False",
      ].join(","),
    );
  }
  for (let i = 0; i < 40; i++) {
    lines.push(
      [
        `syn-we-${i}`, "weekend", 2, 1 + (i % 3), 480 + (i % 4) * 5, 1410, "", "", "", "False",
        "True", 510, 1, "True", 760, 1, "True", 1110, 1, "True", 600, "False", 90, "False",
      ].join(","),
    );
  }
  return lines.join("\n") + "\n";
}

export function writeAtusPool(dataDir: string): void {
  const file = join(dataDir, POOL_REL);
  mkdirSync(join(file, ".."), { recursive: true });
  writeFileSync(file, atusCsv(), "utf8");
}

/** Copies the example world into `<dataDir>/worlds`. */
export function installExampleWorld(dataDir: string): void {
  mkdirSync(join(dataDir, "worlds"), { recursive: true });
  copyFileSync(EXAMPLE_WORLD, join(dataDir, "worlds", `${PERSON_ID}.json`));
}

type Part = { key: string; category: ClipCategory; setting: Setting; node: string; activity: string; tags?: string[]; min: number };

const NEXTME_PARTS: Part[] = [
  { key: "a-ready", category: "home", setting: "home_indoor", node: "home", activity: "getting ready", min: 5 },
  { key: "b-breakfast", category: "meal", setting: "home_indoor", node: "home", activity: "eating breakfast", min: 5 },
  { key: "c-work", category: "work", setting: "venue_indoor", node: "workplace", activity: "work at desk", min: 20 },
  { key: "d-lunch-work", category: "meal", setting: "venue_indoor", node: "workplace", activity: "eating lunch", min: 5 },
  { key: "e-lunch-home", category: "meal", setting: "home_indoor", node: "home", activity: "eating lunch", min: 5 },
  { key: "f-work", category: "work", setting: "venue_indoor", node: "workplace", activity: "work at desk", min: 20 },
  { key: "g-chores", category: "home", setting: "home_indoor", node: "home", activity: "doing chores", tags: ["chores"], min: 5 },
  { key: "h-resting", category: "home", setting: "home_indoor", node: "home", activity: "resting at home", min: 5 },
  { key: "i-dinner", category: "meal", setting: "home_indoor", node: "home", activity: "eating dinner", min: 5 },
  { key: "j-relaxing", category: "home", setting: "home_indoor", node: "home", activity: "relaxing at home", min: 5 },
];

export type BankOptions = {
  /** NextMe-style day groups (default 32; the example world needs 28). */
  groups?: number;
  /** Ego-Exo4D-style climbing takes, one day group each (default 20). */
  hobbyTakes?: number;
  /** time_of_day of every NextMe clip (default "unknown"). */
  timeOfDay?: Clip["time_of_day"];
};

export type Bank = Record<"nextme800" | "egoexo4d", Clip[]>;

const pad = (n: number) => String(n).padStart(2, "0");

/** Synthetic clip bank matching the example world's bindings. */
export function makeBank(o: BankOptions = {}): Bank {
  const nextme: Clip[] = [];
  for (let g = 1; g <= (o.groups ?? 32); g++) {
    const video = `nm-g${pad(g)}`;
    NEXTME_PARTS.forEach((p, i) => {
      const start = i * 2000;
      nextme.push({
        clip_id: `nextme800:${video}:${p.key}`,
        category: p.category,
        setting: p.setting,
        node: p.node,
        activity: p.activity,
        time_of_day: o.timeOfDay ?? "unknown",
        tags: p.tags ?? [],
        source: { dataset: "nextme800", video_id: video, media: { kind: "frames", approx_fps: 1, tar: `${video}.tar` } },
        source_start_s: start,
        source_end_s: start + p.min * 60,
        day_group: `nextme800:g${pad(g)}`,
        person_source: "nextme800:wearer",
        fingerprint: null,
        extractor_version: "fixture.v1",
      });
    });
  }
  const egoexo: Clip[] = [];
  for (let h = 1; h <= (o.hobbyTakes ?? 20); h++) {
    const video = `take-${pad(h)}`;
    egoexo.push({
      clip_id: `egoexo4d:${video}:climb`,
      category: "hobby",
      setting: "venue_indoor",
      node: "hobby",
      activity: "climbing a bouldering wall",
      time_of_day: "unknown",
      tags: ["sport_climbing"],
      source: { dataset: "egoexo4d", video_id: video, media: { kind: "video", fps: 30 } },
      source_start_s: 0,
      source_end_s: 1800,
      day_group: `egoexo4d:${video}`,
      person_source: `egoexo4d:${video}`,
      fingerprint: null,
      extractor_version: "fixture.v1",
    });
  }
  return { nextme800: nextme, egoexo4d: egoexo };
}

/** One synthetic caption per clip, keyed by video id. */
function annotationsFor(clips: Clip[]): Map<string, Annotation[]> {
  const out = new Map<string, Annotation[]>();
  for (const c of clips) {
    const rows = out.get(c.source.video_id) ?? [];
    rows.push({
      id: `cap-${c.clip_id}`,
      track: "captions",
      level: c.source.dataset === "nextme800" ? "L3" : "keystep",
      time: { kind: "interval", start: c.source_start_s, end: c.source_end_s },
      text: c.activity,
      original: {},
    });
    out.set(c.source.video_id, rows);
  }
  return out;
}

/** Replaces one dataset's clip catalog (as a re-ingest would) and rebuilds index.json. */
export function writeDataset(dataDir: string, dataset: Dataset, clips: Clip[]): void {
  new CatalogWriter(dataDir).writeClipCatalog(dataset, clips, annotationsFor(clips));
  rebuildIndex(dataDir, FIXTURE_NOW);
}

export function writeBank(dataDir: string, o: BankOptions = {}): Bank {
  const bank = makeBank(o);
  writeDataset(dataDir, "nextme800", bank.nextme800);
  writeDataset(dataDir, "egoexo4d", bank.egoexo4d);
  return bank;
}

/** A data dir with the example world, the synthetic pool and a bank. */
export function setupDataDir(o: BankOptions & { pool?: boolean; world?: boolean } = {}): string {
  const dataDir = makeTempDir();
  writeBank(dataDir, o);
  if (o.pool !== false) writeAtusPool(dataDir);
  if (o.world !== false) installExampleWorld(dataDir);
  return dataDir;
}

let nextId = 0;

export async function connect(dataDir: string, opts: ServerOptions = {}) {
  const server = createServer(
    { dataDir, configDir: CONFIG_DIR },
    { now: () => FIXTURE_NOW, genId: () => `tl${++nextId}`, ...opts },
  );
  const [ct, st] = InMemoryTransport.createLinkedPair();
  const client = new Client({ name: "test", version: "0.0.0" });
  await Promise.all([server.connect(st), client.connect(ct)]);
  return {
    async call(name: string, args: Record<string, unknown> = {}) {
      return (await client.callTool({ name, arguments: args })) as CallToolResult;
    },
    async close() {
      await client.close();
      await server.close();
    },
  };
}

export function textOf(r: CallToolResult): string {
  return r.content.map((c) => (c.type === "text" ? c.text : "")).join("\n");
}

/** structuredContent of a successful call; fails the test with the error text otherwise. */
export function dataOf<T = Record<string, unknown>>(r: CallToolResult): T {
  if (r.isError) throw new Error(`tool failed: ${textOf(r)}`);
  return r.structuredContent as T;
}
