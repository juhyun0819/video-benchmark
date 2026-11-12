import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import {
  TimelineError,
  TimelineSchema,
  parseWorld,
  type Timeline,
  type TimelineManifest,
  type VerifierReport,
  type World,
} from "@vb/core";

const SAFE_ID = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;
export const isSafeId = (id: string): boolean => SAFE_ID.test(id) && !id.includes("..");

export function writeFileAtomic(path: string, content: string): void {
  mkdirSync(dirname(path), { recursive: true });
  const tmp = `${path}.tmp-${process.pid}`;
  writeFileSync(tmp, content, "utf8");
  renameSync(tmp, path);
}

const json = (x: unknown): string => JSON.stringify(x, null, 2) + "\n";
const REPORT_SUFFIX = ".verifier_report.json";

/** Worlds under `<dataDir>/worlds/<person_id>.json`. */
export class WorldStore {
  readonly dir: string;

  constructor(dataDir: string) {
    this.dir = join(dataDir, "worlds");
  }

  /** Path of a world file; unsafe ids are reported as not found. */
  file(personId: string): string {
    if (!isSafeId(personId)) {
      throw new TimelineError("WORLD_NOT_FOUND", `world ${personId} not found`, { person_id: personId });
    }
    return join(this.dir, `${personId}.json`);
  }

  load(personId: string): World {
    const file = this.file(personId);
    if (!existsSync(file)) {
      throw new TimelineError("WORLD_NOT_FOUND", `world ${personId} not found under ${this.dir}`, { person_id: personId });
    }
    return parseWorld(JSON.parse(readFileSync(file, "utf8")));
  }

  list(): World[] {
    if (!existsSync(this.dir)) return [];
    return readdirSync(this.dir)
      .filter((f) => f.endsWith(".json"))
      .sort()
      .map((f) => this.load(f.slice(0, -".json".length)));
  }
}

export type TimelineSummary = { id: string; name?: string; person_id: string; revision: number; days: number };

/** Timelines and verifier reports under `<dataDir>/timelines`, manifests under `<dataDir>/exports`. */
export class TimelineStore {
  private readonly dir: string;
  private readonly exportDir: string;

  constructor(dataDir: string) {
    this.dir = join(dataDir, "timelines");
    this.exportDir = join(dataDir, "exports");
  }

  private file(id: string, suffix = ".json"): string {
    if (!isSafeId(id)) throw new TimelineError("TIMELINE_NOT_FOUND", `timeline ${id} not found`, { id });
    return join(this.dir, `${id}${suffix}`);
  }

  load(id: string): Timeline {
    const file = this.file(id);
    if (!existsSync(file)) throw new TimelineError("TIMELINE_NOT_FOUND", `timeline ${id} not found`, { id });
    return TimelineSchema.parse(JSON.parse(readFileSync(file, "utf8")));
  }

  save(t: Timeline): string {
    const parsed = TimelineSchema.parse(t);
    const path = this.file(parsed.id);
    writeFileAtomic(path, json(parsed));
    return path;
  }

  list(): TimelineSummary[] {
    if (!existsSync(this.dir)) return [];
    return readdirSync(this.dir)
      .filter((f) => f.endsWith(".json") && !f.endsWith(REPORT_SUFFIX))
      .sort()
      .map((f) => this.load(f.slice(0, -".json".length)))
      .map((t) => ({
        id: t.id,
        ...(t.name !== undefined ? { name: t.name } : {}),
        person_id: t.world.person_id,
        revision: t.revision,
        days: t.schedule.days.length,
      }));
  }

  /** Writes `<dataDir>/timelines/<id>.verifier_report.json`; returns its path. */
  writeReport(id: string, report: VerifierReport): string {
    const path = this.file(id, REPORT_SUFFIX);
    writeFileAtomic(path, json(report));
    return path;
  }

  /** Writes `<dataDir>/exports/<id>-r<revision>.manifest.json`; returns its path. */
  writeExport(m: TimelineManifest): string {
    if (!isSafeId(m.id)) throw new Error(`unsafe timeline id for export: ${JSON.stringify(m.id)}`);
    const path = join(this.exportDir, `${m.id}-r${m.revision}.manifest.json`);
    writeFileAtomic(path, json(m));
    return path;
  }
}
