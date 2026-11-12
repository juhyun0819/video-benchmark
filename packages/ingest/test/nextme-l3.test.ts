import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { ClipContractRowSchema, type ClipContractRow } from "@vb/core";
import { main, makeLocalFetcher, ingestNextme } from "../src/cli.js";
import { loadLabelRules } from "../src/nextme/draft.js";
import { labelText, loadL3Rules, loadOverrides } from "../src/nextme/l3.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..", "..", "..");
const SRC = join(HERE, "fixtures", "nextme-l3");
const OVERRIDES = join(HERE, "fixtures", "nextme", "l3-overrides.json");
const L3_RULES = join(ROOT, "config", "nextme_label_rules.l3.json");

let dir: string;
beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "vb-l3-"));
});
afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

type Row = Omit<ClipContractRow, "category" | "setting" | "node"> & {
  category: string | null;
  setting: string | null;
  node: string | null;
};

async function run(days: string[], withOverrides = false): Promise<Row[]> {
  const out = join(dir, "l3.jsonl");
  await ingestNextme({
    days,
    outPath: out,
    rules: loadLabelRules(join(ROOT, "config", "nextme_label_rules.json")),
    l3Rules: loadL3Rules(L3_RULES),
    clipLevel: "L3",
    ...(withOverrides ? { overrides: loadOverrides(OVERRIDES) } : {}),
    fetchText: makeLocalFetcher(SRC),
  });
  return readFileSync(out, "utf8").trim().split("\n").map((l) => JSON.parse(l));
}
const byActivity = (rows: Row[], text: string): Row => {
  const r = rows.find((x) => x.activity === text);
  if (!r) throw new Error(`no clip for "${text}"`);
  return r;
};

const EXPECTED_L4: unknown[] = [
  ["nextme800:2026-04-15_0900-1000:0.000-1200.000", 0, 1200, "home", "meal", "nextme-draft.v1"],
  ["nextme800:2026-04-15_0900-1000:1200.000-2460.000", 1200, 2460, null, null, "nextme-draft.v1"],
  ["nextme800:2026-04-15_0900-1000:2460.000-3600.000", 2460, 3600, "transit", "transit", "nextme-draft.v1"],
  ["nextme800:2026-04-15_1100-1105:0.000-300.000", 0, 300, null, null, "nextme-draft.v1"],
];

describe("L3 labeler", () => {
  it("yields one clip per L3 segment of at least 120 s", async () => {
    const rows = await run(["2026-06-02"]);
    expect(rows).toHaveLength(6); // the 60 s "home screen" segment is dropped
    expect(rows.every((r) => r.source_end_s - r.source_start_s >= 120)).toBe(true);
    expect(rows.every((r) => r.extractor_version === "nextme-l3.v1")).toBe(true);
    expect(rows.every((r) => ClipContractRowSchema.safeParse(r).success)).toBe(true);
    expect(byActivity(rows, "I wash dishes in the kitchen.")).toMatchObject({
      source_start_s: 0,
      source_end_s: 600,
      category: "home",
      node: "home",
      setting: "home_indoor",
      time_of_day: "morning",
    });
  });

  it("falls back from L3 to L4 to L5 for the place", async () => {
    const rows = await run(["2026-06-02"]);
    expect(byActivity(rows, "I walk through a quiet area.")).toMatchObject({ node: "transit", category: "transit" }); // via L4
    expect(byActivity(rows, "I look around.")).toMatchObject({ node: "workplace", category: "work" }); // via L5
  });

  it("applies hobby tags and the meal override of the category", async () => {
    const rows = await run(["2026-06-02"]);
    expect(byActivity(rows, "I hike along a trail.")).toMatchObject({ node: "hobby", category: "hobby", setting: "outdoor", tags: ["walking"] });
    expect(byActivity(rows, "I eat lunch in the canteen.")).toMatchObject({ node: "workplace", category: "meal" });
    // meal words do not change hobby (or transit) clips
    expect(byActivity(rows, "I eat lunch on the hiking trail.")).toMatchObject({ node: "hobby", category: "hobby", setting: "outdoor", tags: ["walking"] });
  });

  it("labels trip-date clips with trip:* nodes and leaves other places unlabeled", async () => {
    const rows = await run(["2026-05-07"]);
    expect(byActivity(rows, "I check in at the hotel lobby.")).toMatchObject({ node: "trip:lodging", category: "home", setting: "venue_indoor" });
    expect(byActivity(rows, "I stroll along the beach.")).toMatchObject({ node: "trip:outing", category: "hobby", setting: "outdoor" });
    expect(byActivity(rows, "I eat lunch at a restaurant.")).toMatchObject({ node: "trip:meal", category: "meal", setting: "venue_indoor" });
    expect(byActivity(rows, "I take the train.")).toMatchObject({ node: "transit" });
    expect(byActivity(rows, "I sit at home.")).toMatchObject({ node: null, category: null });
  });

  it("an override relabels unlabeled clips in range and leaves labeled ones alone", async () => {
    const rows = await run(["2026-06-21"], true);
    expect(byActivity(rows, "I scroll through my phone.")).toMatchObject({ node: null }); // 11:00, before 11:12
    expect(byActivity(rows, "I read the home screen.")).toMatchObject({ node: "home", setting: "home_indoor", category: "home" });
    expect(byActivity(rows, "I sit in the library.")).toMatchObject({ node: "workplace", category: "work" });
  });

  it("without overrides the same clip stays unlabeled", async () => {
    const rows = await run(["2026-06-21"]);
    expect(byActivity(rows, "I read the home screen.")).toMatchObject({ node: null, category: null });
    expect(byActivity(rows, "I sit in the library.")).toMatchObject({ node: "workplace" });
  });

  it("--overrides is passed through the CLI, and is rejected with L4", async () => {
    const out = join(dir, "ov.jsonl");
    expect(await main(["nextme", "--days", "2026-06-21", "--clip-level", "L3", "--source-dir", SRC, "--out", out, "--rules", L3_RULES, "--overrides", OVERRIDES])).toBe(0);
    const rows: Row[] = readFileSync(out, "utf8").trim().split("\n").map((l) => JSON.parse(l));
    expect(byActivity(rows, "I read the home screen.")).toMatchObject({ node: "home" });
    expect(await main(["nextme", "--days", "2026-06-21", "--source-dir", SRC, "--overrides", OVERRIDES])).toBe(1);
  });

  it("validates the overrides file", () => {
    const bad = join(dir, "bad.json");
    writeFileSync(bad, JSON.stringify([{ day_group: "2026-06-21", from: "12:00", to: "11:00", node: "home", setting: "home_indoor", category: "home" }]));
    expect(() => loadOverrides(bad)).toThrow(/from must not be after to/);
  });

  it("never-home phrases do not count as home and the clip stays unlabeled", async () => {
    const rows = await run(["2026-06-03"]);
    expect(rows).toHaveLength(2);
    expect(rows.every((r) => r.node === null && r.category === null)).toBe(true);
  });

  it("the default L4 clip level output is unchanged", async () => {
    const out = join(dir, "l4.jsonl");
    await ingestNextme({
      days: ["2026-04-15"],
      outPath: out,
      rules: loadLabelRules(join(ROOT, "config", "nextme_label_rules.json")),
      fetchText: async (path: string) => {
        const m = /^captions\/1_nextme\/([^/]+)\/([^/]+)$/.exec(path);
        const flat = path === "raw/recordings.csv" ? "recordings.csv" : m ? `${m[1]}/${m[2]}` : path;
        return readFileSync(join(HERE, "fixtures", "nextme", flat), "utf8");
      },
    });
    const rows: Row[] = readFileSync(out, "utf8").trim().split("\n").map((l) => JSON.parse(l));
    expect(rows.map((r) => [r.clip_id, r.source_start_s, r.source_end_s, r.node, r.category, r.extractor_version])).toEqual(EXPECTED_L4);
  });

  it("--source-dir reads local files through the CLI", async () => {
    const out = join(dir, "cli.jsonl");
    const code = await main(["nextme", "--days", "2026-05-07", "--clip-level", "L3", "--source-dir", SRC, "--out", out, "--rules", L3_RULES]);
    expect(code).toBe(0);
    const rows = readFileSync(out, "utf8").trim().split("\n").map((l) => JSON.parse(l));
    expect(rows).toHaveLength(5);
    expect(rows[0].extractor_version).toBe("nextme-l3.v1");
  });

  it("rejects an unknown --clip-level", async () => {
    expect(await main(["nextme", "--days", "2026-05-07", "--clip-level", "L9", "--source-dir", SRC])).toBe(1);
  });
});

describe("gym place rule (B9)", () => {
  const rules = loadL3Rules(L3_RULES);
  const label = (t: string) => labelText(t, rules, false);
  it("labels gym words as the gym node, before workplace", () => {
    for (const t of ["I lift a dumbbell.", "I do a bench press.", "I run on the treadmill.", "I walk from the office to the gym.", "I do pull-up exercises at the fitness center."]) {
      expect(label(t), t).toMatchObject({ node: "gym", category: "hobby", setting: "venue_indoor", tags: ["gym"] });
    }
  });
  it("leaves plain office text as workplace", () => {
    expect(label("I sit in the office.")).toMatchObject({ node: "workplace" });
  });
});
