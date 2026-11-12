import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import {
  ExceptionCatalogSchema,
  GapTableSchema,
  ScheduleDefaultsSchema,
  TemplatesSchema,
  atus2025Source,
  generateDay,
  parseAtusPool,
  type GapTable,
  type ScheduleDay,
  type ScheduleInput,
  type Timeline,
  type World,
} from "@vb/core";

export type GeneratorConfig = Omit<ScheduleInput, "world" | "boundNodes">;

export const POOL_PATH = join("distributions", "atus2025", "atus2025_stats_per_respondent.csv");

function readText(path: string): string {
  if (!existsSync(path)) throw new Error(`config file not found: ${path}`);
  return readFileSync(path, "utf8");
}

function parseFile<T>(path: string, parse: (raw: unknown) => T): T {
  const text = readText(path);
  try {
    return parse(JSON.parse(text));
  } catch (e) {
    throw new Error(`invalid config file ${path}: ${e instanceof Error ? e.message : String(e)}`);
  }
}

/** Generator inputs other than the world: timeline configs, schedule defaults and the ATUS donor pool. */
export function loadConfig(configDir: string, dataDir: string): GeneratorConfig {
  const timeline = join(configDir, "timeline");
  const templates = parseFile(join(timeline, "templates.v2.json"), (r) => TemplatesSchema.parse(r));
  // GapTableSchema's inferred type is wider than GapTable (known mismatch).
  const gaps = parseFile(join(timeline, "gaps.v1.json"), (r) => GapTableSchema.parse(r) as GapTable);
  const exceptions = parseFile(join(timeline, "exceptions.v1.json"), (r) => ExceptionCatalogSchema.parse(r));
  const defaults = parseFile(join(configDir, "schedule_defaults_atus2025.json"), (r) => ScheduleDefaultsSchema.parse(r));
  return { templates, gaps, exceptions, defaults, source: atus2025Source(readPool(join(dataDir, POOL_PATH))) };
}

function readPool(path: string) {
  const text = readText(path);
  try {
    return parseAtusPool(text);
  } catch (e) {
    throw new Error(`invalid ATUS pool ${path}: ${e instanceof Error ? e.message : String(e)}`);
  }
}

export function scheduleInput(config: GeneratorConfig, world: World): ScheduleInput {
  return { ...config, world, boundNodes: new Set(Object.keys(world.node_binding)) };
}

/** Day regeneration for fillTimeline, from the timeline's own world and donors (R27). */
export function regenFor(config: GeneratorConfig, t: Timeline): (dayIndex: number, attempt: number) => ScheduleDay {
  const input = scheduleInput(config, t.world);
  return (d, a) => generateDay(input, t.schedule.donors, d, a);
}

export type Dirs = { dataDir: string; configDir: string };

/** `--data` / `--config` override VIDEO_BENCH_DATA_DIR / VIDEO_BENCH_CONFIG_DIR (defaults ./data, ./config). */
export function resolveDirs(flags: { data?: string; config?: string }, env: NodeJS.ProcessEnv): Dirs {
  return {
    dataDir: resolve(flags.data ?? env["VIDEO_BENCH_DATA_DIR"] ?? "./data"),
    configDir: resolve(flags.config ?? env["VIDEO_BENCH_CONFIG_DIR"] ?? "./config"),
  };
}
