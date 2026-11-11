import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { AnnotationSchema, ClipSchema, DatasetSchema, type Annotation, type Clip, type ClipCatalog, type Dataset } from "@vb/core";
import { isSafeVideoIdSegment, readJsonl } from "@vb/ingest";

/** Read-only view of every `catalog/<dataset>/clips.jsonl` and its annotation files under the data dir. */
export class FsClipCatalog implements ClipCatalog {
  private readonly root: string;
  private readonly all: Clip[];
  private readonly byId: Map<string, Clip>;
  private readonly annotations = new Map<string, Annotation[]>();

  constructor(dataDir: string) {
    this.root = join(dataDir, "catalog");
    this.all = [];
    this.byId = new Map();
    for (const ds of DatasetSchema.options) {
      const file = join(this.root, ds, "clips.jsonl");
      if (!existsSync(file)) continue;
      for (const clip of readJsonl(readFileSync(file, "utf8"), file, ClipSchema)) {
        if (this.byId.has(clip.clip_id)) throw new Error(`duplicate clip_id ${clip.clip_id} in ${file}`);
        this.byId.set(clip.clip_id, clip);
        this.all.push(clip);
      }
    }
  }

  clips(): readonly Clip[] {
    return this.all;
  }

  getClip(id: string): Clip | undefined {
    return this.byId.get(id);
  }

  getAnnotations(dataset: Dataset, videoId: string): Annotation[] {
    if (!isSafeVideoIdSegment(videoId)) return [];
    const key = `${dataset}/${videoId}`;
    let rows = this.annotations.get(key);
    if (!rows) {
      const file = join(this.root, dataset, "annotations", `${videoId}.jsonl`);
      rows = existsSync(file) ? readJsonl(readFileSync(file, "utf8"), file, AnnotationSchema) : [];
      this.annotations.set(key, rows);
    }
    return structuredClone(rows);
  }
}
