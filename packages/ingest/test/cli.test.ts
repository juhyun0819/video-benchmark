import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
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

describe("ingestNextme", () => {
  it("fetches, writes the catalog, rebuilds the index and returns the report", async () => {
    const report = await ingestNextme({
      days: ["2026-04-15"],
      dataDir: dir,
      fetchText: fakeFetch,
      now: new Date("2026-11-03T00:00:00.000Z"),
    });
    expect(report.ingested["nextme800:recordings"]).toBe(2);
    expect(report.ingested["nextme800:blocks"]).toBe(4);

    const cat = join(dir, "catalog");
    const day = JSON.parse(readFileSync(join(cat, "nextme800/days/2026-04-15.json"), "utf8"));
    expect(day.blocks).toHaveLength(4);
    expect(readFileSync(join(cat, "nextme800/annotations/2026-04-15_0900-1000.jsonl"), "utf8")).toContain(
      "nextme800:2026-04-15_0900-1000:L4:L4_00001",
    );
    expect(JSON.parse(readFileSync(join(cat, "index.json"), "utf8")).base_days).toEqual(["2026-04-15"]);
  });

  it("rerunning gives identical output (idempotent)", async () => {
    const opts = { days: ["2026-04-15"], dataDir: dir, fetchText: fakeFetch, now: new Date(0) };
    await ingestNextme(opts);
    const read = () => readFileSync(join(dir, "catalog/nextme800/days/2026-04-15.json"), "utf8");
    const first = read();
    await ingestNextme(opts);
    expect(read()).toBe(first);
  });

  it("fails with the day in the message when it has no recordings", async () => {
    await expect(
      ingestNextme({ days: ["2026-05-01"], dataDir: dir, fetchText: fakeFetch, now: new Date(0) }),
    ).rejects.toThrow(/2026-05-01/);
  });

  it("rejects malformed day arguments", async () => {
    await expect(
      ingestNextme({ days: ["15-04-2026"], dataDir: dir, fetchText: fakeFetch, now: new Date(0) }),
    ).rejects.toThrow(/YYYY-MM-DD/);
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
