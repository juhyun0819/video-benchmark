import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import {
  AnnotationSchema,
  BaseDaySchema,
  BlockCandidateSchema,
  DatasetSchema,
  type Annotation,
  type BaseDay,
  type BlockCandidate,
  type Dataset,
} from "@vb/core";
import { readJsonl } from "./jsonl.js";

const SAFE_SEGMENT = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

function safeSegment(kind: string, value: string): string {
  if (!SAFE_SEGMENT.test(value) || value.includes("..")) {
    throw new Error(`Unsafe ${kind} for catalog path: ${JSON.stringify(value)}`);
  }
  return value;
}

function writeFileAtomic(path: string, content: string): void {
  mkdirSync(dirname(path), { recursive: true });
  const tmp = `${path}.tmp`;
  writeFileSync(tmp, content, "utf8");
  renameSync(tmp, path);
}

function jsonl(rows: unknown[]): string {
  return rows.map((r) => JSON.stringify(r)).join("\n") + (rows.length > 0 ? "\n" : "");
}

/** Writes the normalized catalog under `<dataDir>/catalog` (spec 6.5). All output is zod-validated. */
export class CatalogWriter {
  private readonly root: string;

  constructor(dataDir: string) {
    this.root = join(dataDir, "catalog");
  }

  writeBaseDay(d: BaseDay): void {
    const parsed = BaseDaySchema.parse(d);
    const day = safeSegment("day", parsed.day);
    writeFileAtomic(
      join(this.root, "nextme800", "days", `${day}.json`),
      JSON.stringify(parsed, null, 2) + "\n",
    );
  }

  writeAnnotations(dataset: Dataset, videoId: string, a: Annotation[]): void {
    const ds = DatasetSchema.parse(dataset);
    const id = safeSegment("video id", videoId);
    const rows = a.map((x) => AnnotationSchema.parse(x));
    writeFileAtomic(join(this.root, ds, "annotations", `${id}.jsonl`), jsonl(rows));
  }

  writeCandidates(dataset: Dataset, c: BlockCandidate[]): void {
    const ds = DatasetSchema.parse(dataset);
    const rows = c.map((x) => BlockCandidateSchema.parse(x));
    writeFileAtomic(join(this.root, ds, "candidates.jsonl"), jsonl(rows));
  }
}

/**
 * Scans `<dataDir>/catalog` and writes `index.json`:
 * `base_days` = days under nextme800/days, `categories` = event-candidate counts
 * per category from every dataset's candidates.jsonl (base blocks are not event candidates).
 */
export function rebuildIndex(dataDir: string, now: Date): void {
  const root = join(dataDir, "catalog");

  const daysDir = join(root, "nextme800", "days");
  const base_days = existsSync(daysDir)
    ? readdirSync(daysDir)
        .filter((f) => f.endsWith(".json"))
        .map((f) => f.slice(0, -".json".length))
        .sort()
    : [];

  const counts = new Map<string, number>();
  for (const ds of DatasetSchema.options) {
    const file = join(root, ds, "candidates.jsonl");
    if (!existsSync(file)) continue;
    for (const c of readJsonl(readFileSync(file, "utf8"), `${ds}/candidates.jsonl`, BlockCandidateSchema)) {
      counts.set(c.category, (counts.get(c.category) ?? 0) + 1);
    }
  }
  const categories = Object.fromEntries([...counts].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)));

  writeFileAtomic(
    join(root, "index.json"),
    JSON.stringify({ updated_at: now.toISOString(), base_days, categories }, null, 2) + "\n",
  );
}
