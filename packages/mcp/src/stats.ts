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

function load(file: string): TimelineManifest {
  let text: string;
  try {
    text = readFileSync(file, "utf8");
  } catch {
    throw new Error(`cannot read ${file}`);
  }
  try {
    return TimelineManifestSchema.parse(JSON.parse(text));
  } catch (e) {
    throw new Error(`invalid manifest ${file}: ${e instanceof Error ? e.message.slice(0, 200) : String(e)}`);
  }
}

function main(): void {
  const { positionals } = parseArgs({ allowPositionals: true, options: {} });
  if (positionals.length === 0) throw new Error(USAGE);
  const manifests = positionals.flatMap(manifestFiles).map(load);
  process.stdout.write(`${JSON.stringify(summarizeManifests(manifests), null, 2)}\n`);
}

try {
  main();
} catch (e) {
  process.stderr.write(`${e instanceof Error ? e.message : String(e)}\n`);
  process.exit(1);
}
