import { randomUUID } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { parseArgs } from "node:util";
import {
  RULE_IDS,
  TimelineError,
  parseWorld,
  buildTimelineManifest,
  fillTimeline,
  rankSelector,
  verifyTimeline,
} from "@vb/core";
import { loadConfig, regenFor, resolveDirs, type Dirs } from "./config.js";
import { FsClipCatalog } from "./fsCatalog.js";
import { describeError, manifestSummary, newTimeline } from "./server.js";
import { TimelineStore, WorldStore, isSafeId, writeFileAtomic } from "./store.js";

// Batch mode: create → auto_fill (rankSelector) → verify → export for one world file.
// Everything runs in memory; files are written only after every step succeeded.

const USAGE = "usage: npm run generate -- --world <path> [--data <dir>] [--config <dir>]";

type GenerateResult = {
  timeline_id: string;
  revision: number;
  manifest: string;
  days: number;
  clips: number;
  coverage_mean: number;
};

function generate(worldPath: string, { dataDir, configDir }: Dirs, now: Date): GenerateResult {
  if (!existsSync(worldPath)) throw new Error(`world file not found: ${worldPath}`);
  const worldText = readFileSync(worldPath, "utf8");
  const world = parseWorld(JSON.parse(worldText));
  if (!isSafeId(world.person_id)) throw new Error(`person_id ${JSON.stringify(world.person_id)} is not a safe file name`);

  const worldFile = new WorldStore(dataDir).file(world.person_id);
  const existing = existsSync(worldFile) ? readFileSync(worldFile, "utf8") : undefined;
  if (existing !== undefined && existing !== worldText) {
    throw new Error(`${worldFile} already exists with different content; remove it or change person_id in ${worldPath}`);
  }

  const config = loadConfig(configDir, dataDir);
  const catalog = new FsClipCatalog(dataDir);
  const created = newTimeline(world, config, catalog, () => randomUUID().slice(0, 8));
  const filled = fillTimeline(created, catalog, rankSelector, regenFor(config, created));
  const report = verifyTimeline(filled, catalog);
  if (!report.pass) {
    const failed_rules = RULE_IDS.filter((r) => !report.rules[r].pass);
    const violations = Object.fromEntries(failed_rules.map((r) => [r, report.rules[r].violations]));
    throw new TimelineError("VERIFY_FAILED", `timeline fails rules: ${failed_rules.join(", ")}`, { failed_rules, violations });
  }
  const manifest = buildTimelineManifest(filled, catalog, now, config.templates);

  if (existing === undefined) writeFileAtomic(worldFile, worldText);
  const store = new TimelineStore(dataDir);
  store.save(filled);
  store.writeReport(filled.id, report);
  const path = store.writeExport(manifest);
  const { days, clips, coverage_mean } = manifestSummary(manifest);
  return { timeline_id: filled.id, revision: filled.revision, manifest: path, days, clips, coverage_mean };
}

try {
  const { values } = parseArgs({
    options: { world: { type: "string" }, data: { type: "string" }, config: { type: "string" } },
    strict: true,
  });
  if (values.world === undefined) throw new Error(USAGE);
  const result = generate(values.world, resolveDirs(values, process.env), new Date());
  process.stdout.write(JSON.stringify(result, null, 2) + "\n");
} catch (e) {
  process.stderr.write(JSON.stringify({ error: describeError(e) }, null, 2) + "\n");
  process.exitCode = 1;
}
