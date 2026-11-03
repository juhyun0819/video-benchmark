import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import {
  AnnotationSchema,
  BaseDaySchema,
  BlockCandidateSchema,
  DatasetSchema,
  type Annotation,
  type BaseDay,
  type BlockCandidate,
  type Catalog,
  type Dataset,
} from "@vb/core";

export class CatalogMissingError extends Error {
  constructor(dataDir: string) {
    super(`Catalog not found under ${join(dataDir, "catalog")} (index.json missing)`);
    this.name = "CatalogMissingError";
  }
}

export type CandidateFilter = {
  category: string;
  query?: string;
  minSec?: number;
  maxSec?: number;
  limit: number;
  offset: number;
};

const SAFE_SEGMENT = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;
const isSafe = (s: string) => SAFE_SEGMENT.test(s) && !s.includes("..");

function readJsonl<T>(file: string, parse: (row: unknown) => T): T[] {
  const out: T[] = [];
  const lines = readFileSync(file, "utf8").split("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!.trim();
    if (line === "") continue;
    try {
      out.push(parse(JSON.parse(line)));
    } catch (e) {
      throw new Error(`${file}:${i + 1}: ${e instanceof Error ? e.message : String(e)}`);
    }
  }
  return out;
}

/** Read-only filesystem view of the catalog written by `@vb/ingest` (spec 6.5). */
export class FsCatalog implements Catalog {
  private readonly root: string;
  private events?: BlockCandidate[];
  private eventsById?: Map<string, BlockCandidate>;
  private baseDays = new Map<string, BaseDay | undefined>();
  private baseBlocksById?: Map<string, BlockCandidate>;
  private annotations = new Map<string, Annotation[]>();

  constructor(dataDir: string) {
    this.root = join(dataDir, "catalog");
    if (!existsSync(join(this.root, "index.json"))) throw new CatalogMissingError(dataDir);
  }

  private loadEvents(): BlockCandidate[] {
    if (this.events) return this.events;
    const all: BlockCandidate[] = [];
    for (const ds of DatasetSchema.options) {
      const file = join(this.root, ds, "candidates.jsonl");
      if (existsSync(file)) all.push(...readJsonl(file, (r) => BlockCandidateSchema.parse(r)));
    }
    this.events = all;
    this.eventsById = new Map(all.map((c) => [c.id, c]));
    return all;
  }

  private dayFiles(): string[] {
    const dir = join(this.root, "nextme800", "days");
    if (!existsSync(dir)) return [];
    return readdirSync(dir)
      .filter((f) => f.endsWith(".json"))
      .map((f) => f.slice(0, -".json".length))
      .sort();
  }

  getBaseDay(day: string): BaseDay | undefined {
    if (!isSafe(day)) return undefined;
    if (this.baseDays.has(day)) return this.baseDays.get(day);
    const file = join(this.root, "nextme800", "days", `${day}.json`);
    const parsed = existsSync(file)
      ? BaseDaySchema.parse(JSON.parse(readFileSync(file, "utf8")))
      : undefined;
    this.baseDays.set(day, parsed);
    return parsed;
  }

  getCandidate(id: string): BlockCandidate | undefined {
    this.loadEvents();
    const ev = this.eventsById!.get(id);
    if (ev) return ev;
    if (!this.baseBlocksById) {
      this.baseBlocksById = new Map();
      for (const day of this.dayFiles()) {
        for (const b of this.getBaseDay(day)?.blocks ?? []) this.baseBlocksById.set(b.id, b);
      }
    }
    return this.baseBlocksById.get(id);
  }

  getAnnotations(dataset: Dataset, videoId: string): Annotation[] {
    if (!isSafe(videoId)) return [];
    const key = `${dataset}/${videoId}`;
    let rows = this.annotations.get(key);
    if (!rows) {
      const file = join(this.root, dataset, "annotations", `${videoId}.jsonl`);
      rows = existsSync(file) ? readJsonl(file, (r) => AnnotationSchema.parse(r)) : [];
      this.annotations.set(key, rows);
    }
    return structuredClone(rows);
  }

  listBaseDays(): Array<{ day: string; blocks: number; duration: number }> {
    return this.dayFiles().flatMap((day) => {
      const d = this.getBaseDay(day);
      if (!d) return [];
      const duration = d.blocks.reduce((s, b) => s + (b.src_end - b.src_start), 0);
      return [{ day: d.day, blocks: d.blocks.length, duration }];
    });
  }

  /** Event categories only (base blocks are not event candidates). */
  listCategories(): Array<{ category: string; count: number }> {
    const counts = new Map<string, number>();
    for (const c of this.loadEvents()) counts.set(c.category, (counts.get(c.category) ?? 0) + 1);
    return [...counts]
      .map(([category, count]) => ({ category, count }))
      .sort((a, b) => (a.category < b.category ? -1 : a.category > b.category ? 1 : 0));
  }

  listCandidates(f: CandidateFilter): { total: number; items: BlockCandidate[] } {
    const q = f.query?.toLowerCase();
    const matched = this.loadEvents().filter((c) => {
      if (c.category !== f.category) return false;
      if (q !== undefined && !c.summary.toLowerCase().includes(q)) return false;
      const dur = c.src_end - c.src_start;
      if (f.minSec !== undefined && dur < f.minSec) return false;
      if (f.maxSec !== undefined && dur > f.maxSec) return false;
      return true;
    });
    return { total: matched.length, items: matched.slice(f.offset, f.offset + f.limit) };
  }
}
