import { existsSync, readdirSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import type { Annotation, BaseDay } from "@vb/core";
import {
  buildHdepicCatalog,
  HdNarrationSchema,
  parseErratumCsv,
  parseHighLevel,
  type HighLevelSegment,
} from "./hdepic/adapter.js";
import { bump, newReport, readJsonl, type IngestReport } from "./jsonl.js";
import { buildNextmeDay, parseRecordingsCsv } from "./nextme/adapter.js";
import { makeHfFetcher } from "./nextme/hf.js";
import { CatalogWriter, rebuildIndex } from "./writer.js";

const USAGE = `Usage:
  npm run ingest -- nextme --days 2026-04-15[,2026-04-16...] [--data <dir>]
  npm run ingest -- hdepic --annotations <hd-epic-annotations dir> --narrations <converted.jsonl> [--data <dir>]

HD-EPIC narrations come from a one-off pkl conversion:
  uv run --isolated --no-project --with pandas --with "numpy>=2" \\
    python scripts/convert_hdepic.py <HD_EPIC_Narrations.pkl> <converted.jsonl>

Environment:
  HF_TOKEN                 Hugging Face token (falls back to ~/.cache/huggingface/token)
  VIDEO_BENCH_DATA_DIR     catalog root (default ./data; --data overrides)`;

const CAPTION_FILES = ["raw_caption", "L1", "L2", "L3", "L4", "L5"] as const;
const DAY_PATTERN = /^\d{4}-\d{2}-\d{2}$/;

/** HF_TOKEN env first, then the trimmed contents of ~/.cache/huggingface/token. Never logged. */
export function resolveHfToken(
  env: Record<string, string | undefined>,
  home: string = homedir(),
): string | undefined {
  const fromEnv = env.HF_TOKEN?.trim();
  if (fromEnv) return fromEnv;
  const file = join(home, ".cache", "huggingface", "token");
  if (existsSync(file)) {
    const fromFile = readFileSync(file, "utf8").trim();
    if (fromFile) return fromFile;
  }
  return undefined;
}

export function resolveDataDir(flag: string | undefined, env: Record<string, string | undefined>): string {
  return flag || env.VIDEO_BENCH_DATA_DIR || "./data";
}

export interface IngestNextmeOptions {
  days: string[];
  dataDir: string;
  /** Downloads a text file from the HF dataset repo by repo-relative path. */
  fetchText: (path: string) => Promise<string>;
  now: Date;
}

/** Fetch, build and write the given NextMe-800 days, then rebuild the index. */
export async function ingestNextme(opts: IngestNextmeOptions): Promise<IngestReport> {
  for (const day of opts.days) {
    if (!DAY_PATTERN.test(day)) throw new Error(`Invalid day "${day}": expected YYYY-MM-DD`);
  }
  const report = newReport();

  const recordings = parseRecordingsCsv(await opts.fetchText("raw/recordings.csv"));

  // Prefetch every caption file of the requested days, then build with a sync getter.
  const cache = new Map<string, string>();
  const key = (recId: string, name: string) => `${recId}/${name}`;
  for (const day of opts.days) {
    for (const rec of recordings.filter((r) => r.day === day)) {
      const texts = await Promise.all(
        CAPTION_FILES.map((name) =>
          opts.fetchText(`captions/1_nextme/${rec.recording_id}/${name}.jsonl`),
        ),
      );
      CAPTION_FILES.forEach((name, i) => cache.set(key(rec.recording_id, name), texts[i] as string));
    }
  }
  const files = (recId: string, name: string): string => {
    const text = cache.get(key(recId, name));
    if (text === undefined) throw new Error(`Caption file not fetched: ${recId}/${name}.jsonl`);
    return text;
  };

  // Build everything first so a bad day writes nothing.
  const built: { baseDay: BaseDay; annotations: Map<string, Annotation[]> }[] = [];
  for (const day of opts.days) built.push(buildNextmeDay(day, recordings, files, report));

  const writer = new CatalogWriter(opts.dataDir);
  for (const { baseDay, annotations } of built) {
    writer.writeBaseDay(baseDay);
    for (const [recId, anns] of annotations) writer.writeAnnotations("nextme800", recId, anns);
  }
  rebuildIndex(opts.dataDir, opts.now);
  bump(report.ingested, "nextme800:days", built.length);
  return report;
}

export interface IngestHdepicOptions {
  /** Root of the cloned hd-epic-annotations repo. */
  annotationsDir: string;
  /** Converted HD_EPIC_Narrations.jsonl (scripts/convert_hdepic.py output). */
  narrationsPath: string;
  dataDir: string;
  now: Date;
}

const HIGH_LEVEL_FILE = /^P\d+_recipe_timestamps\.csv$/;

/** Read the HD-EPIC annotations, build cooking candidates + narration annotations, write the catalog. */
export function ingestHdepic(opts: IngestHdepicOptions): IngestReport {
  const report = newReport();
  const activitiesDir = join(opts.annotationsDir, "high-level", "activities");
  if (!existsSync(activitiesDir)) {
    throw new Error(`HD-EPIC: ${activitiesDir} not found (is --annotations the hd-epic-annotations repo root?)`);
  }
  const segments: HighLevelSegment[] = [];
  for (const f of readdirSync(activitiesDir).filter((n) => HIGH_LEVEL_FILE.test(n)).sort()) {
    segments.push(...parseHighLevel(readFileSync(join(activitiesDir, f), "utf8"), f));
  }
  if (segments.length === 0) throw new Error(`HD-EPIC: no PXX_recipe_timestamps.csv files in ${activitiesDir}`);

  const recipes = JSON.parse(
    readFileSync(join(opts.annotationsDir, "high-level", "complete_recipes.json"), "utf8"),
  ) as Record<string, { name: string }>;
  const erratumFile = "HD_EPIC_Narrations_erratum.csv";
  const erratum = parseErratumCsv(
    readFileSync(join(opts.annotationsDir, "narrations-and-action-segments", erratumFile), "utf8"),
    erratumFile,
  );
  const narrations = readJsonl(readFileSync(opts.narrationsPath, "utf8"), opts.narrationsPath, HdNarrationSchema);

  // Build everything first so a malformed input writes nothing.
  const { candidates, annotations } = buildHdepicCatalog(segments, narrations, recipes, report, erratum);

  const writer = new CatalogWriter(opts.dataDir);
  writer.writeCandidates("hdepic", candidates);
  for (const [videoId, anns] of annotations) writer.writeAnnotations("hdepic", videoId, anns);
  rebuildIndex(opts.dataDir, opts.now);
  return report;
}

export async function main(
  argv: string[],
  env: Record<string, string | undefined> = process.env,
): Promise<number> {
  const { positionals, values } = parseArgs({
    args: argv,
    allowPositionals: true,
    options: {
      days: { type: "string" },
      data: { type: "string" },
      annotations: { type: "string" },
      narrations: { type: "string" },
      help: { type: "boolean", short: "h" },
    },
  });
  const [command] = positionals;
  if (values.help || !command) {
    console.log(USAGE);
    return command || values.help ? 0 : 1;
  }

  const dataDir = resolveDataDir(values.data, env);

  if (command === "nextme") {
    const days = (values.days ?? "")
      .split(",")
      .map((d) => d.trim())
      .filter(Boolean);
    if (days.length === 0) {
      console.error("nextme: --days <YYYY-MM-DD[,YYYY-MM-DD...]> is required");
      return 1;
    }
    const token = resolveHfToken(env);
    if (!token) {
      console.error(
        "HF_TOKEN is not set and ~/.cache/huggingface/token was not found. " +
          "Set HF_TOKEN to a Hugging Face token with approved access to mmm8383/NextMe-800.",
      );
      return 1;
    }
    const report = await ingestNextme({
      days,
      dataDir,
      fetchText: makeHfFetcher(token),
      now: new Date(),
    });
    console.log(JSON.stringify(report, null, 2));
    return 0;
  }

  if (command === "hdepic") {
    if (!values.annotations || !values.narrations) {
      console.error("hdepic: --annotations <repo dir> and --narrations <converted.jsonl> are required");
      return 1;
    }
    const report = ingestHdepic({
      annotationsDir: values.annotations,
      narrationsPath: values.narrations,
      dataDir,
      now: new Date(),
    });
    console.log(JSON.stringify(report, null, 2));
    return 0;
  }

  console.error(`Unknown command: ${command}\n\n${USAGE}`);
  return 1;
}

// Run only when executed directly (tsx packages/ingest/src/cli.ts), not when imported by tests.
if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  main(process.argv.slice(2)).then(
    (code) => {
      process.exitCode = code;
    },
    (err: unknown) => {
      console.error(err instanceof Error ? err.message : String(err));
      process.exitCode = 1;
    },
  );
}
