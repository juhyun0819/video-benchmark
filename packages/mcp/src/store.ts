import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { CompositionError, CompositionSchema, type Composition, type Manifest } from "@vb/core";

const SAFE_ID = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

function writeFileAtomic(path: string, content: string): void {
  mkdirSync(dirname(path), { recursive: true });
  const tmp = `${path}.tmp-${process.pid}`;
  writeFileSync(tmp, content, "utf8");
  renameSync(tmp, path);
}

/** Persists compositions and exported manifests under the data dir (spec 7.5). */
export class CompositionStore {
  private readonly compDir: string;
  private readonly exportDir: string;

  constructor(dataDir: string) {
    this.compDir = join(dataDir, "compositions");
    this.exportDir = join(dataDir, "exports");
  }

  private file(id: string): string {
    if (!SAFE_ID.test(id) || id.includes("..")) {
      throw new CompositionError("COMPOSITION_NOT_FOUND", `Composition ${id} not found`, { id });
    }
    return join(this.compDir, `${id}.json`);
  }

  load(id: string): Composition {
    const file = this.file(id);
    if (!existsSync(file)) {
      throw new CompositionError("COMPOSITION_NOT_FOUND", `Composition ${id} not found`, { id });
    }
    return CompositionSchema.parse(JSON.parse(readFileSync(file, "utf8")));
  }

  save(c: Composition): void {
    const parsed = CompositionSchema.parse(c);
    writeFileAtomic(this.file(parsed.id), JSON.stringify(parsed, null, 2) + "\n");
  }

  list(): Array<{ id: string; name?: string; base_day: string; revision: number }> {
    if (!existsSync(this.compDir)) return [];
    return readdirSync(this.compDir)
      .filter((f) => f.endsWith(".json"))
      .sort()
      .map((f) => this.load(f.slice(0, -".json".length)))
      .map((c) => ({
        id: c.id,
        ...(c.name !== undefined ? { name: c.name } : {}),
        base_day: c.base_day,
        revision: c.revision,
      }));
  }

  /** Writes `<dataDir>/exports/<id>-r<revision>.manifest.json`; returns its path. */
  writeExport(m: Manifest): string {
    if (!SAFE_ID.test(m.composition_id) || m.composition_id.includes("..")) {
      throw new Error(`Unsafe composition id for export: ${JSON.stringify(m.composition_id)}`);
    }
    const path = join(this.exportDir, `${m.composition_id}-r${m.composition_revision}.manifest.json`);
    writeFileAtomic(path, JSON.stringify(m, null, 2) + "\n");
    return path;
  }
}
