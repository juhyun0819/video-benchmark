import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, unlinkSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import {
  AnnotationSchema,
  ClipSchema,
  DatasetSchema,
  type Annotation,
  type Clip,
  type Dataset,
} from "@vb/core";
import { readJsonl } from "./jsonl.js";

const SAFE_VIDEO_ID = /^[A-Za-z0-9_-][A-Za-z0-9._-]*$/;

/** Video ids become annotation file names; YouTube ids may start with '-' or '_' (R28). */
export function isSafeVideoIdSegment(id: string): boolean {
  return SAFE_VIDEO_ID.test(id) && !id.includes("..");
}

function safeVideoId(value: string): string {
  if (!isSafeVideoIdSegment(value)) throw new Error(`Unsafe video id for catalog path: ${JSON.stringify(value)}`);
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

  writeAnnotations(dataset: Dataset, videoId: string, a: Annotation[]): void {
    const ds = DatasetSchema.parse(dataset);
    const id = safeVideoId(videoId);
    const rows = a.map((x) => AnnotationSchema.parse(x));
    writeFileAtomic(join(this.root, ds, "annotations", `${id}.jsonl`), jsonl(rows));
  }

  /**
   * Replaces a dataset's clip catalog: clips.jsonl, one annotation file per video in `annotations`, and
   * removal of stale `<id>.jsonl` annotation files. Everything is validated before any file is written or removed.
   */
  writeClipCatalog(dataset: Dataset, clips: Clip[], annotations: Map<string, Annotation[]>): void {
    const ds = DatasetSchema.parse(dataset);
    const clipRows = clips.map((x) => ClipSchema.parse(x));
    this.replaceCatalog(ds, "clips.jsonl", clipRows, annotations);
  }

  /** Validate all annotations, write the list file and annotations, drop stale annotation files. */
  private replaceCatalog(
    ds: Dataset,
    listFile: string,
    rows: unknown[],
    annotations: Map<string, Annotation[]>,
  ): void {
    const annRows = [...annotations].map(
      ([videoId, a]) => [safeVideoId(videoId), a.map((x) => AnnotationSchema.parse(x))] as const,
    );

    writeFileAtomic(join(this.root, ds, listFile), jsonl(rows));
    const annDir = join(this.root, ds, "annotations");
    for (const [id, annotationRows] of annRows) writeFileAtomic(join(annDir, `${id}.jsonl`), jsonl(annotationRows));

    if (!existsSync(annDir)) return;
    const keep = new Set(annRows.map(([id]) => `${id}.jsonl`));
    for (const f of readdirSync(annDir)) {
      if (f.endsWith(".jsonl") && !keep.has(f)) unlinkSync(join(annDir, f));
    }
  }
}

/**
 * Scans `<dataDir>/catalog` and writes `index.json`:
 * `clips` = clip counts per dataset and category from every dataset's clips.jsonl,
 * `categories` = the same counts summed across datasets.
 */
export function rebuildIndex(dataDir: string, now: Date): void {
  const root = join(dataDir, "catalog");

  const clips: Record<string, Record<string, number>> = {};
  for (const ds of [...DatasetSchema.options].sort()) {
    const file = join(root, ds, "clips.jsonl");
    if (!existsSync(file)) continue;
    const perCategory = new Map<string, number>();
    for (const c of readJsonl(readFileSync(file, "utf8"), `${ds}/clips.jsonl`, ClipSchema)) {
      perCategory.set(c.category, (perCategory.get(c.category) ?? 0) + 1);
    }
    clips[ds] = Object.fromEntries([...perCategory].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)));
  }

  const totals = new Map<string, number>();
  for (const perCategory of Object.values(clips)) {
    for (const [category, n] of Object.entries(perCategory)) totals.set(category, (totals.get(category) ?? 0) + n);
  }
  const categories = Object.fromEntries([...totals].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)));

  writeFileAtomic(
    join(root, "index.json"),
    JSON.stringify({ updated_at: now.toISOString(), categories, clips }, null, 2) + "\n",
  );
}
