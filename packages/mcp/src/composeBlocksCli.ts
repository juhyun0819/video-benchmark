import { randomUUID } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { RULE_IDS, TimelineError, buildTimelineManifest, createTimeline, generateSchedule, parseWorld, verifyTimeline, type SlotOverride, type World } from "@vb/core";
import { loadConfig, resolveDirs, scheduleInput, type Dirs } from "./config.js";
import { composeBlocksFit, type ComposeSummary } from "./composeBlocks.js";
import { FsClipCatalog } from "./fsCatalog.js";
import { describeError } from "./server.js";
import { TimelineStore, WorldStore, isSafeId, writeFileAtomic } from "./store.js";

// B5 batch mode: generate the schedule → one real date per day → block plan → layoutBlocks →
// block-mode verify → manifest. Reads the catalog and ATUS pool from --data, writes the world,
// timeline, verifier report and manifest under --out (default: --data). Nothing is written unless
// every step succeeded.

const USAGE = "usage: npm run compose-blocks -- --world <path> [--data <dir>] [--config <dir>] [--out <dir>] [--fit-roles hobby[,role...]]";

type ComposeBlocksOutput = {
  timeline_id: string;
  revision: number;
  manifest: string;
  verifier_report: string;
  /** B17: the slot lengths fit_roles wrote into the world. */
  slot_overrides: SlotOverride[];
} & ComposeSummary;

function composeBlocksCli(worldPath: string, { dataDir, configDir }: Dirs, outDir: string, now: Date, fitRoles: string[] = []): ComposeBlocksOutput {
  if (!existsSync(worldPath)) throw new Error(`world file not found: ${worldPath}`);
  const worldText = readFileSync(worldPath, "utf8");
  const world = parseWorld(JSON.parse(worldText));
  if (!isSafeId(world.person_id)) throw new Error(`person_id ${JSON.stringify(world.person_id)} is not a safe file name`);

  const worldFile = new WorldStore(outDir).file(world.person_id);
  const existing = existsSync(worldFile) ? readFileSync(worldFile, "utf8") : undefined;

  const config = loadConfig(configDir, dataDir);
  const catalog = new FsClipCatalog(dataDir);
  // Block mode: t.slots stay empty; the placements are the source of truth.
  const id = `${world.person_id}-${randomUUID().slice(0, 8)}`;
  const build = (w: World) => createTimeline(w, generateSchedule(scheduleInput(config, w)), () => id);
  const { timeline: t, placements, summary, slot_overrides } = composeBlocksFit(build(world), catalog, fitRoles, build);
  const report = verifyTimeline(t, catalog, { mode: "block", placements });
  if (!report.pass) {
    const failed_rules = RULE_IDS.filter((r) => !report.rules[r].pass);
    const violations = Object.fromEntries(failed_rules.map((r) => [r, report.rules[r].violations]));
    throw new TimelineError("VERIFY_FAILED", `timeline fails rules: ${failed_rules.join(", ")}`, { failed_rules, violations });
  }
  const manifest = buildTimelineManifest(t, catalog, now, config.templates, { placements });

  // B17: the saved world carries the computed overrides, so the manifest is reproducible.
  const savedText = fitRoles.length > 0 && slot_overrides.length > 0 ? JSON.stringify(t.world, null, 2) + "\n" : worldText;
  if (existing !== undefined && existing !== worldText && existing !== savedText) {
    throw new Error(`${worldFile} already exists with different content; remove it or change person_id in ${worldPath}`);
  }
  if (existing === undefined) writeFileAtomic(worldFile, savedText);
  const store = new TimelineStore(outDir);
  store.save(t);
  const verifier_report = store.writeReport(t.id, report);
  const path = store.writeExport(manifest);
  return { timeline_id: t.id, revision: t.revision, manifest: path, verifier_report, ...summary, slot_overrides };
}

try {
  const { values } = parseArgs({
    options: { world: { type: "string" }, data: { type: "string" }, config: { type: "string" }, out: { type: "string" }, "fit-roles": { type: "string" } },
    strict: true,
  });
  if (values.world === undefined) throw new Error(USAGE);
  const dirs = resolveDirs(values, process.env);
  const out = values.out !== undefined ? resolve(values.out) : dirs.dataDir;
  const fitRoles = (values["fit-roles"] ?? "").split(",").map((r) => r.trim()).filter((r) => r !== "");
  if (values["fit-roles"] !== undefined && fitRoles.length === 0) throw new Error(USAGE);
  const result = composeBlocksCli(values.world, dirs, out, new Date(), fitRoles);
  process.stdout.write(JSON.stringify(result, null, 2) + "\n");
} catch (e) {
  process.stderr.write(JSON.stringify({ error: describeError(e) }, null, 2) + "\n");
  process.exitCode = 1;
}
