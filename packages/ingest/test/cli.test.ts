import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { LabelRules } from "../src/nextme/draft.js";
import { ingestNextme, resolveDataDir, resolveHfToken } from "../src/cli.js";

const FIX = join(dirname(fileURLToPath(import.meta.url)), "fixtures", "nextme");

let dir: string;
beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "vb-ingest-cli-"));
});
afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

// Serves the synthetic fixture the way HF paths are laid out.
const fakeFetch = async (path: string): Promise<string> => {
  if (path === "raw/recordings.csv") return readFileSync(join(FIX, "recordings.csv"), "utf8");
  const m = /^captions\/1_nextme\/([^/]+)\/([^/]+)\.jsonl$/.exec(path);
  if (!m) throw new Error(`unexpected path ${path}`);
  return readFileSync(join(FIX, m[1] as string, `${m[2]}.jsonl`), "utf8");
};

const RULES: LabelRules = {
  version: 1,
  person_source: "nextme800:wearer",
  places: [
    { words: ["station", "train"], node: "transit", setting: "outdoor" },
    { words: ["home"], node: "home", setting: "home_indoor" },
  ],
  activities: [],
  default_category_by_node: { home: "home", transit: "transit" },
};

describe("ingestNextme", () => {
  const out = () => join(dir, "nested", "draft.jsonl");
  const opts = (days: string[]) => ({ days, outPath: out(), rules: RULES, fetchText: fakeFetch });

  it("writes one draft JSONL (creating parent dirs), reports counts and leaves the catalog alone", async () => {
    const report = await ingestNextme(opts(["2026-04-15", "2026-04-15"]));
    const rows = readFileSync(out(), "utf8").trimEnd().split("\n").map((l) => JSON.parse(l));
    expect(rows.map((r) => r.day_group)).toEqual(["2026-04-15", "2026-04-15", "2026-04-15", "2026-04-15"]);
    expect(rows.slice(0, 4).map((r) => r.source_start_s)).toEqual([0, 1200, 2460, 0]);
    expect(rows.every((r) => r.extractor_version === "nextme-draft.v1")).toBe(true);
    expect(report.ingested["nextme800:labeled"]! + report.ingested["nextme800:unlabeled"]!).toBe(rows.length);
    expect(report.ingested["nextme800:labeled"]).toBe(rows.filter((r) => r.category !== null).length);
    expect(existsSync(join(dir, "catalog"))).toBe(false);
  });

  it("rerunning gives identical output (idempotent)", async () => {
    await ingestNextme(opts(["2026-04-15"]));
    const first = readFileSync(out(), "utf8");
    await ingestNextme(opts(["2026-04-15"]));
    expect(readFileSync(out(), "utf8")).toBe(first);
  });

  it("fails with the day in the message when it has no recordings", async () => {
    await expect(ingestNextme(opts(["2026-05-01"]))).rejects.toThrow(/2026-05-01/);
    expect(existsSync(out())).toBe(false);
  });

  it("rejects malformed day arguments", async () => {
    await expect(ingestNextme(opts(["15-04-2026"]))).rejects.toThrow(/YYYY-MM-DD/);
  });
});

describe("resolveHfToken", () => {
  it("prefers HF_TOKEN, trims the cached token file, and returns undefined if neither exists", () => {
    const home = join(dir, "home");
    expect(resolveHfToken({}, home)).toBeUndefined();

    mkdirSync(join(home, ".cache/huggingface"), { recursive: true });
    writeFileSync(join(home, ".cache/huggingface/token"), "file-token\n");
    expect(resolveHfToken({}, home)).toBe("file-token");
    expect(resolveHfToken({ HF_TOKEN: " env-token " }, home)).toBe("env-token");
    expect(resolveHfToken({ HF_TOKEN: "" }, home)).toBe("file-token");
  });
});

describe("resolveDataDir", () => {
  it("uses flag, then VIDEO_BENCH_DATA_DIR, then ./data", () => {
    expect(resolveDataDir("/a", { VIDEO_BENCH_DATA_DIR: "/b" })).toBe("/a");
    expect(resolveDataDir(undefined, { VIDEO_BENCH_DATA_DIR: "/b" })).toBe("/b");
    expect(resolveDataDir(undefined, {})).toBe("./data");
  });
});
