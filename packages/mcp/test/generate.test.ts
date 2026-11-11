import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { TimelineManifestSchema } from "@vb/core";
import { CONFIG_DIR, EXAMPLE_WORLD, PERSON_ID, REPO_ROOT, filesIn, setupDataDir } from "./helpers.js";

const TSX = join(REPO_ROOT, "node_modules", "tsx", "dist", "cli.mjs");
const SCRIPT = join(REPO_ROOT, "packages", "mcp", "src", "generate.ts");

function generate(dataDir: string, world = EXAMPLE_WORLD) {
  const r = spawnSync(process.execPath, [TSX, SCRIPT, "--world", world, "--data", dataDir, "--config", CONFIG_DIR], {
    cwd: REPO_ROOT,
    encoding: "utf8",
    timeout: 60_000,
  });
  return { status: r.status, stdout: r.stdout, stderr: r.stderr };
}

describe("npm run generate", () => {
  it("generate CLI exits 0 and prints manifest path", () => {
    const dataDir = setupDataDir({ world: false });
    const r = generate(dataDir);
    expect(r.stderr).toBe("");
    expect(r.status).toBe(0);

    const out = JSON.parse(r.stdout) as {
      timeline_id: string;
      revision: number;
      manifest: string;
      days: number;
      clips: number;
      coverage_mean: number;
    };
    expect(out.revision).toBe(1);
    expect(out.days).toBe(28);
    expect(out.clips).toBeGreaterThan(0);
    expect(out.manifest).toBe(join(dataDir, "exports", `${out.timeline_id}-r1.manifest.json`));
    const manifest = TimelineManifestSchema.parse(JSON.parse(readFileSync(out.manifest, "utf8")));
    expect(manifest.id).toBe(out.timeline_id);

    expect(readFileSync(join(dataDir, "worlds", `${PERSON_ID}.json`), "utf8")).toBe(readFileSync(EXAMPLE_WORLD, "utf8"));
    expect(filesIn(dataDir, "timelines")).toEqual([
      `${out.timeline_id}.json`,
      `${out.timeline_id}.verifier_report.json`,
    ]);

    // A second run with the identical world file succeeds and adds a second timeline.
    const again = generate(dataDir);
    expect(again.status).toBe(0);
    expect(filesIn(dataDir, "timelines")).toHaveLength(4);
  });

  it("generate with unfillable bank exits 1 and leaves data/timelines empty", () => {
    // Every NextMe clip is a night clip, so no morning slot can be filled at any ladder level.
    const dataDir = setupDataDir({ world: false, timeOfDay: "night" });
    const r = generate(dataDir);
    expect(r.status).toBe(1);
    expect(r.stdout).toBe("");
    const err = JSON.parse(r.stderr) as { error: { code: string; message: string; details: Record<string, unknown> } };
    expect(err.error.code).toBe("UNFILLABLE");
    expect(err.error.details).toMatchObject({ day: 1, attempts: 5 });
    expect(filesIn(dataDir, "timelines")).toEqual([]);
    expect(filesIn(dataDir, "worlds")).toEqual([]);
    expect(filesIn(dataDir, "exports")).toEqual([]);
  });

  it("refuses to overwrite a different world file with the same person_id", () => {
    const dataDir = setupDataDir({ world: false });
    mkdirSync(join(dataDir, "worlds"), { recursive: true });
    const existing = join(dataDir, "worlds", `${PERSON_ID}.json`);
    writeFileSync(existing, "{}\n", "utf8");
    const r = generate(dataDir);
    expect(r.status).toBe(1);
    const err = JSON.parse(r.stderr) as { error: { code: string; message: string } };
    expect(err.error.message).toContain(existing);
    expect(readFileSync(existing, "utf8")).toBe("{}\n");
    expect(existsSync(join(dataDir, "timelines"))).toBe(false);
  });
});
