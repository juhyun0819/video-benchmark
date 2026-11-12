import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { DatasetSchema, type Annotation, type Dataset } from "@vb/core";
import { loadCategories } from "./categories.js";
import { candidatesToClips } from "./clipmap.js";
import {
  buildEgo4dCatalog,
  listScenarios,
  parseEgo4dMeta,
  parseEgo4dNarrations,
  readJsonFile,
} from "./ego4d/adapter.js";
import {
  buildEgoExoCatalog,
  parseEgoExoAtomic,
  parseEgoExoKeysteps,
  parseEgoExoTakes,
} from "./egoexo/adapter.js";
import { ingestBank } from "./bank.js";
import { bump, newReport, readJsonl, type IngestReport } from "./jsonl.js";
import { buildNextmeDay, parseRecordingsCsv } from "./nextme/adapter.js";
import { loadLabelRules, nextmeDraftRows, type DraftRow, type LabelRules } from "./nextme/draft.js";
import { makeHfFetcher } from "./nextme/hf.js";
import { loadL3Rules, nextmeL3Rows, type L3Rules } from "./nextme/l3.js";
import { parseHkt } from "./time.js";
import { CatalogWriter, rebuildIndex } from "./writer.js";

const USAGE = `Usage:
  npm run ingest -- nextme --days 2026-04-15[,2026-04-16...] [--out data/raw/bank/nextme800.draft.jsonl] [--rules config/nextme_label_rules.json] [--overwrite]
                       [--clip-level L4|L3] [--source-dir <dir with raw/ and captions/>]   (L3 default rules: config/nextme_label_rules.l3.json)
  npm run ingest -- ego4d --metadata <ego4d.json> --narrations <narration.json> [--categories config/categories.json] [--data <dir>]
  npm run ingest -- ego4d --metadata <ego4d.json> --list-scenarios
  npm run ingest -- bank --input <clips.jsonl> --dataset <nextme800|ego4d|egoexo4d|youtube> [--data <dir>]
  npm run ingest -- egoexo --root <egoexo download dir> [--min-sec 60] [--categories config/categories.json] [--data <dir>]

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

/** Serves repo-relative dataset paths (raw/recordings.csv, captions/1_nextme/...) from a local directory. */
export function makeLocalFetcher(dir: string): (path: string) => Promise<string> {
  return async (path: string) => {
    const full = join(dir, path);
    if (!existsSync(full)) throw new Error(`${full}: file not found in --source-dir`);
    return readFileSync(full, "utf8");
  };
}

export interface IngestNextmeOptions {
  days: string[];
  /** Draft JSONL to write (parent directories are created). */
  outPath: string;
  rules: LabelRules;
  /** Clip granularity: L4 blocks (default) or experimental L3 segments of at least 120 s. */
  clipLevel?: "L3" | "L4";
  /** Rules used when `clipLevel` is L3. */
  l3Rules?: L3Rules;
  /** Downloads a text file from the HF dataset repo by repo-relative path. */
  fetchText: (path: string) => Promise<string>;
  /** Replace an existing `outPath`; without it an existing draft (possibly reviewed by hand) is refused. */
  overwrite?: boolean;
}

const rowStartEpoch = (r: DraftRow): number =>
  (r.source.media.kind === "frames" && r.source.media.start_wallclock
    ? parseHkt(r.source.media.start_wallclock)
    : 0) + r.source_start_s;

/**
 * Fetch and build the given NextMe-800 days and write one labeled clip-contract draft JSONL for review
 * (rows sorted by day, then clip start). Rows the rules cannot label keep null labels; `ingest bank` skips them.
 * Nothing is written to the catalog. An existing `outPath` is refused unless `overwrite` is set.
 */
export async function ingestNextme(opts: IngestNextmeOptions): Promise<IngestReport> {
  for (const day of opts.days) {
    if (!DAY_PATTERN.test(day)) throw new Error(`Invalid day "${day}": expected YYYY-MM-DD`);
  }
  if (!opts.overwrite && existsSync(opts.outPath)) {
    throw new Error(
      `${opts.outPath} already exists and may hold reviewed labels; keep a copy and pass --overwrite to replace it, or choose another --out`,
    );
  }
  const clipLevel = opts.clipLevel ?? "L4";
  if (clipLevel === "L3" && !opts.l3Rules) throw new Error("clip level L3 needs l3Rules");
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
  const rows: DraftRow[] = [];
  for (const day of [...new Set(opts.days)].sort()) {
    const built = buildNextmeDay(day, recordings, files, report);
    const dayRows =
      clipLevel === "L3" ? nextmeL3Rows(day, built, opts.l3Rules as L3Rules) : nextmeDraftRows(day, built, opts.rules);
    rows.push(...dayRows.sort((a, b) => rowStartEpoch(a) - rowStartEpoch(b)));
  }

  mkdirSync(dirname(opts.outPath), { recursive: true });
  writeFileSync(opts.outPath, rows.map((r) => JSON.stringify(r)).join("\n") + (rows.length > 0 ? "\n" : ""), "utf8");
  bump(report.ingested, "nextme800:labeled", rows.filter((r) => r.category !== null).length);
  bump(report.ingested, "nextme800:unlabeled", rows.filter((r) => r.category === null).length);
  return report;
}

export interface IngestEgo4dOptions {
  metadataPath: string;
  narrationsPath: string;
  categoriesPath: string;
  dataDir: string;
  now: Date;
}

/** Read the Ego4D metadata + narrations, build scenario/keyword candidates, map them to clips and write the clip catalog. */
export function ingestEgo4d(opts: IngestEgo4dOptions): IngestReport {
  const report = newReport();
  const categories = loadCategories(opts.categoriesPath);
  const meta = parseEgo4dMeta(readJsonFile(opts.metadataPath), opts.metadataPath);
  const narrations = parseEgo4dNarrations(readJsonFile(opts.narrationsPath), opts.narrationsPath);

  // Build everything first so a malformed input writes nothing.
  const built = buildEgo4dCatalog(meta, narrations, categories, report);
  const { clips, annotations } = candidatesToClips(built.candidates, built.annotations, categories.categories, report);
  bump(report.ingested, "ego4d:clips", clips.length);

  const writer = new CatalogWriter(opts.dataDir);
  writer.writeClipCatalog("ego4d", clips, annotations);
  rebuildIndex(opts.dataDir, opts.now);
  return report;
}

export interface IngestEgoExoOptions {
  /** Ego-Exo4D download root: holds takes.json and annotations/. */
  rootDir: string;
  categoriesPath: string;
  dataDir: string;
  now: Date;
  /** Minimum block length in seconds (adapter default when omitted). */
  minSec?: number | undefined;
}

/** Read the Ego-Exo4D takes and train/val annotations, build one candidate per take, map them to clips and write the clip catalog. */
export function ingestEgoExo(opts: IngestEgoExoOptions): IngestReport {
  const report = newReport();
  const categories = loadCategories(opts.categoriesPath);
  const takesPath = join(opts.rootDir, "takes.json");
  const takes = parseEgoExoTakes(readJsonFile(takesPath), takesPath);
  const splitPath = (name: string, split: string) => join(opts.rootDir, "annotations", `${name}_${split}.json`);
  const atomicTrain = splitPath("atomic_descriptions", "train");
  const atomicVal = splitPath("atomic_descriptions", "val");
  const keystepTrain = splitPath("keystep", "train");
  const keystepVal = splitPath("keystep", "val");
  const atomic = {
    ...parseEgoExoAtomic(readJsonFile(atomicTrain), atomicTrain),
    ...parseEgoExoAtomic(readJsonFile(atomicVal), atomicVal),
  };
  const keysteps = {
    ...parseEgoExoKeysteps(readJsonFile(keystepTrain), keystepTrain),
    ...parseEgoExoKeysteps(readJsonFile(keystepVal), keystepVal),
  };

  // Build everything first so a malformed input writes nothing.
  const built = buildEgoExoCatalog(takes, atomic, keysteps, categories, report, opts.minSec);
  const { clips, annotations } = candidatesToClips(built.candidates, built.annotations, categories.categories, report);
  bump(report.ingested, "egoexo4d:clips", clips.length);

  const writer = new CatalogWriter(opts.dataDir);
  writer.writeClipCatalog("egoexo4d", clips, annotations);
  rebuildIndex(opts.dataDir, opts.now);
  return report;
}

export interface IngestBankOptions {
  inputPath: string;
  dataset: Dataset;
  dataDir: string;
  now: Date;
}

/** Read a content-contract clip JSONL, write the dataset's clip catalog and caption annotations. */
export function ingestBankFile(opts: IngestBankOptions): IngestReport {
  const { clips, annotations, report } = ingestBank(readFileSync(opts.inputPath, "utf8"), opts.inputPath, opts.dataset);
  const writer = new CatalogWriter(opts.dataDir);
  writer.writeClipCatalog(opts.dataset, clips, annotations);
  rebuildIndex(opts.dataDir, opts.now);
  return report;
}

/** The parsed --min-sec value, undefined when absent, or null (after logging) when it is not a non-negative number. */
function parseMinSec(command: string, value: string | undefined): number | undefined | null {
  if (value === undefined) return undefined;
  const minSec = Number(value);
  if (Number.isFinite(minSec) && minSec >= 0) return minSec;
  console.error(`${command}: --min-sec must be a non-negative number, got "${value}"`);
  return null;
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
      out: { type: "string" },
      rules: { type: "string" },
      overwrite: { type: "boolean" },
      "clip-level": { type: "string" },
      "source-dir": { type: "string" },
      data: { type: "string" },
      annotations: { type: "string" },
      narrations: { type: "string" },
      metadata: { type: "string" },
      categories: { type: "string" },
      "list-scenarios": { type: "boolean" },
      root: { type: "string" },
      input: { type: "string" },
      dataset: { type: "string" },
      "min-sec": { type: "string" },
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
    const clipLevel = values["clip-level"] ?? "L4";
    if (clipLevel !== "L4" && clipLevel !== "L3") {
      console.error(`nextme: --clip-level must be L4 or L3, got "${clipLevel}"`);
      return 1;
    }
    const sourceDir = values["source-dir"];
    const token = sourceDir ? undefined : resolveHfToken(env);
    if (!sourceDir && !token) {
      console.error(
        "HF_TOKEN is not set and ~/.cache/huggingface/token was not found. " +
          "Set HF_TOKEN to a Hugging Face token with approved access to mmm8383/NextMe-800.",
      );
      return 1;
    }
    const report = await ingestNextme({
      days,
      outPath: values.out ?? resolve("data", "raw", "bank", "nextme800.draft.jsonl"),
      rules:
        clipLevel === "L3"
          ? loadLabelRules(resolve("config", "nextme_label_rules.json"))
          : loadLabelRules(values.rules ?? resolve("config", "nextme_label_rules.json")),
      clipLevel,
      ...(clipLevel === "L3"
        ? { l3Rules: loadL3Rules(values.rules ?? resolve("config", "nextme_label_rules.l3.json")) }
        : {}),
      fetchText: sourceDir ? makeLocalFetcher(sourceDir) : makeHfFetcher(token as string),
      overwrite: values.overwrite ?? false,
    });
    console.log(JSON.stringify(report, null, 2));
    return 0;
  }

  if (command === "ego4d") {
    if (!values.metadata) {
      console.error("ego4d: --metadata <ego4d.json> is required");
      return 1;
    }
    if (values["list-scenarios"]) {
      const meta = parseEgo4dMeta(readJsonFile(values.metadata), values.metadata);
      for (const { scenario, videos } of listScenarios(meta)) console.log(`${scenario}\t${videos}`);
      return 0;
    }
    if (!values.narrations) {
      console.error("ego4d: --narrations <narration.json> is required (or use --list-scenarios)");
      return 1;
    }
    const report = ingestEgo4d({
      metadataPath: values.metadata,
      narrationsPath: values.narrations,
      categoriesPath: values.categories ?? resolve("config", "categories.json"),
      dataDir,
      now: new Date(),
    });
    console.log(JSON.stringify(report, null, 2));
    return 0;
  }

  if (command === "egoexo") {
    if (!values.root) {
      console.error("egoexo: --root <egoexo download dir> is required");
      return 1;
    }
    const minSec = parseMinSec("egoexo", values["min-sec"]);
    if (minSec === null) return 1;
    const report = ingestEgoExo({
      rootDir: values.root,
      categoriesPath: values.categories ?? resolve("config", "categories.json"),
      dataDir,
      now: new Date(),
      minSec,
    });
    console.log(JSON.stringify(report, null, 2));
    return 0;
  }

  if (command === "bank") {
    if (!values.input || !values.dataset) {
      console.error("bank: --input <clips.jsonl> and --dataset <ds> are required");
      return 1;
    }
    const dataset = DatasetSchema.safeParse(values.dataset);
    if (!dataset.success) {
      console.error(`bank: --dataset must be one of ${DatasetSchema.options.join(", ")}, got "${values.dataset}"`);
      return 1;
    }
    const report = ingestBankFile({
      inputPath: values.input,
      dataset: dataset.data,
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
