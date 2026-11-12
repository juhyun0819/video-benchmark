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

/** A file list() could not load (G25): it is skipped and reported instead of failing the listing. */
export type SkippedFile = { file: string; code: string; message: string };

function skippedOf(file: string, e: unknown): SkippedFile {
  if (e instanceof TimelineError) return { file, code: e.code, message: e.message };
  return { file, code: "INTERNAL", message: e instanceof Error ? e.message : String(e) };
}

/** Loads every `.json` name in `names` (sorted), collecting failures as skipped files. */
function loadAll<T>(names: string[], load: (name: string) => T): { items: T[]; skipped: SkippedFile[] } {
  const items: T[] = [];
  const skipped: SkippedFile[] = [];
  for (const f of [...names].sort()) {
    try {
      items.push(load(f.slice(0, -".json".length)));
    } catch (e) {
      skipped.push(skippedOf(f, e));
    }
  }
  return { items, skipped };
}

function readJson(file: string, invalid: (message: string) => TimelineError): unknown {
  try {
    return JSON.parse(readFileSync(file, "utf8"));
  } catch (e) {
    throw invalid(`not valid JSON (${e instanceof Error ? e.message : String(e)})`);
  }
}

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
    const raw = readJson(file, (m) => new TimelineError("WORLD_INVALID", `${file}: ${m}`, { person_id: personId }));
    return parseWorld(raw);
  }

  /** Worlds that load; files that do not are skipped and reported (G25). */
  list(): { worlds: World[]; skipped: SkippedFile[] } {
    if (!existsSync(this.dir)) return { worlds: [], skipped: [] };
    const names = readdirSync(this.dir).filter((f) => f.endsWith(".json"));
    const { items, skipped } = loadAll(names, (id) => this.load(id));
    return { worlds: items, skipped };
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

  /** A saved timeline; one that does not match the current schema is TIMELINE_INVALID (G25). */
  load(id: string): Timeline {
    const file = this.file(id);
    if (!existsSync(file)) throw new TimelineError("TIMELINE_NOT_FOUND", `timeline ${id} not found`, { id });
    const invalid = (m: string, issues?: { path: string; message: string }[]) =>
      new TimelineError(
        "TIMELINE_INVALID",
        `timeline ${id} does not match the current schema (${m}); it was probably saved before rev3 — regenerate it with create_timeline`,
        { id, file, ...(issues ? { issues } : {}) },
      );
    const r = TimelineSchema.safeParse(readJson(file, (m) => invalid(m)));
    if (r.success) return r.data;
    const issues = r.error.issues.slice(0, 5).map((i) => ({ path: i.path.join("."), message: i.message }));
    throw invalid(issues.map((i) => `${i.path}: ${i.message}`).join("; "), issues);
  }

  save(t: Timeline): string {
    const parsed = TimelineSchema.parse(t);
    const path = this.file(parsed.id);
    writeFileAtomic(path, json(parsed));
    return path;
  }

  /** Summaries of the timelines that load; files that do not are skipped and reported (G25). */
  list(): { timelines: TimelineSummary[]; skipped: SkippedFile[] } {
    if (!existsSync(this.dir)) return { timelines: [], skipped: [] };
    const names = readdirSync(this.dir).filter((f) => f.endsWith(".json") && !f.endsWith(REPORT_SUFFIX));
    const { items, skipped } = loadAll(names, (id) => this.load(id));
    const timelines = items.map((t) => ({
      id: t.id,
      ...(t.name !== undefined ? { name: t.name } : {}),
      person_id: t.world.person_id,
      revision: t.revision,
      days: t.schedule.days.length,
    }));
    return { timelines, skipped };
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
