import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { TimelineManifestSchema, summarizeManifests, type TimelineManifest } from "@vb/core";

// Stats report over exported manifests: prints JSON to stdout.

const USAGE = "usage: npm run stats -- <manifest.json | dir> [...]";

function manifestFiles(path: string): string[] {
  let st;
  try {
    st = statSync(path);
  } catch {
    throw new Error(`cannot read ${path}`);
  }
  if (!st.isDirectory()) return [path];
  return readdirSync(path)
    .filter((f) => f.endsWith(".manifest.json"))
    .sort()
    .map((f) => join(path, f));
}

const isObject = (x: unknown): x is Record<string, unknown> => typeof x === "object" && x !== null && !Array.isArray(x);

/** A timeline manifest, or undefined (with a stderr warning) for a composition manifest (G26). */
function load(file: string): TimelineManifest | undefined {
  let text: string;
  try {
    text = readFileSync(file, "utf8");
  } catch {
    throw new Error(`cannot read ${file}`);
  }
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (e) {
    throw new Error(`invalid manifest ${file}: ${e instanceof Error ? e.message.slice(0, 200) : String(e)}`);
  }
  if (isObject(raw) && "composition_id" in raw && !("days" in raw)) {
    process.stderr.write(`skipping ${file}: a composition manifest, not a timeline manifest\n`);
    return undefined;
  }
  const r = TimelineManifestSchema.safeParse(raw);
  if (r.success) return r.data;
  if (isObject(raw) && raw["schema_version"] === 1 && "days" in raw) {
    throw new Error(`invalid manifest ${file}: schema_version 1 was exported before rev3; re-export the timeline`);
  }
  throw new Error(`invalid manifest ${file}: ${r.error.message.slice(0, 200)}`);
}

function main(): void {
  const { positionals } = parseArgs({ allowPositionals: true, options: {} });
  if (positionals.length === 0) throw new Error(USAGE);
  const manifests = positionals
    .flatMap(manifestFiles)
    .map(load)
    .filter((m): m is TimelineManifest => m !== undefined);
  process.stdout.write(`${JSON.stringify(summarizeManifests(manifests), null, 2)}\n`);
}

try {
  main();
} catch (e) {
  process.stderr.write(`${e instanceof Error ? e.message : String(e)}\n`);
  process.exit(1);
}
