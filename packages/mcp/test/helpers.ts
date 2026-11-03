import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import type { BlockCandidate, Dataset } from "@vb/core";
import { CatalogWriter, rebuildIndex } from "@vb/ingest";
import { createServer } from "../src/server.js";
import { makeFixtureCatalog } from "../../core/test/fixtures.js";

export const FIXTURE_NOW = new Date("2026-11-03T00:00:00.000Z");

export const IDS = {
  cooking: "hdepic:P01-x:10.000-70.000",
  walking: "ego4d:uid-1:0.000-120.000",
  /** Extra cooking candidate added by writeFixtureCatalog (1800 s). */
  cooking2: "hdepic:P01-y:0.000-1800.000",
  base0: "nextme800:rec-A:0.000-600.000",
};

export function makeTempDir(): string {
  return mkdtempSync(join(tmpdir(), "vb-mcp-"));
}

/** Writes the core fixture catalog (plus one extra cooking candidate) into `dataDir` via CatalogWriter. */
export function writeFixtureCatalog(dataDir: string): void {
  const fx = makeFixtureCatalog();
  const w = new CatalogWriter(dataDir);
  w.writeBaseDay(fx.baseDay);

  const cooking = fx.getCandidate(IDS.cooking);
  const walking = fx.getCandidate(IDS.walking);
  if (!cooking || !walking) throw new Error("fixture candidates missing");
  const cooking2: BlockCandidate = {
    id: IDS.cooking2,
    role: "event",
    category: "cooking",
    source: { dataset: "hdepic", video_id: "P01-y", media: { kind: "video", fps: 30 } },
    src_start: 0,
    src_end: 1800,
    summary: "Making breakfast (omelette)",
  };
  w.writeCandidates("hdepic", [cooking, cooking2]);
  w.writeCandidates("ego4d", [walking]);

  const videos: Array<[Dataset, string]> = [
    ["nextme800", "rec-A"],
    ["nextme800", "rec-B"],
    ["hdepic", "P01-x"],
    ["ego4d", "uid-1"],
  ];
  for (const [ds, vid] of videos) w.writeAnnotations(ds, vid, fx.getAnnotations(ds, vid));
  rebuildIndex(dataDir, FIXTURE_NOW);
}

export async function connect(dataDir: string, opts: Parameters<typeof createServer>[1] = {}) {
  const server = createServer(dataDir, opts);
  const [ct, st] = InMemoryTransport.createLinkedPair();
  const client = new Client({ name: "test", version: "0.0.0" });
  await Promise.all([server.connect(st), client.connect(ct)]);
  return {
    client,
    async call(name: string, args: Record<string, unknown> = {}) {
      return (await client.callTool({ name, arguments: args })) as CallToolResult;
    },
    async close() {
      await client.close();
      await server.close();
    },
  };
}

export function textOf(r: CallToolResult): string {
  return r.content
    .map((c) => (c.type === "text" ? c.text : ""))
    .join("\n");
}
