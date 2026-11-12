import { spawnSync } from "node:child_process";
import { copyFileSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { REPO_ROOT } from "./helpers.js";

const TSX = join(REPO_ROOT, "node_modules", "tsx", "dist", "cli.mjs");
const SCRIPT = join(REPO_ROOT, "packages", "mcp", "src", "stats.ts");
const GOLDEN = join(REPO_ROOT, "packages", "core", "test", "__snapshots__", "timeline.manifest.json");

function stats(...args: string[]) {
  const r = spawnSync(process.execPath, [TSX, SCRIPT, ...args], { cwd: REPO_ROOT, encoding: "utf8", timeout: 60_000 });
  return { status: r.status, stdout: r.stdout, stderr: r.stderr };
}

describe("npm run stats", () => {
  it("prints a JSON report for a directory", { timeout: 30_000 }, () => {
    const dir = mkdtempSync(join(tmpdir(), "vb-stats-"));
    copyFileSync(GOLDEN, join(dir, "a-r1.manifest.json"));
    const r = stats(dir);
    expect(r.stderr).toBe("");
    expect(r.status).toBe(0);
    const out = JSON.parse(r.stdout) as { manifests: number; groups: unknown[] };
    expect(out.manifests).toBe(1);
    expect(out.groups).toHaveLength(1);
  });

  it("exits 1 naming an unreadable or invalid path", { timeout: 30_000 }, () => {
    const missing = join(tmpdir(), "vb-stats-missing.manifest.json");
    const a = stats(missing);
    expect(a.status).toBe(1);
    expect(a.stderr).toContain(missing);
    const dir = mkdtempSync(join(tmpdir(), "vb-stats-"));
    const bad = join(dir, "bad.manifest.json");
    writeFileSync(bad, "{}");
    const b = stats(bad);
    expect(b.status).toBe(1);
    expect(b.stderr).toContain(bad);
  });
});
