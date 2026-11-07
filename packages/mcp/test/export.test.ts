import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { isAbsolute, join, relative } from "node:path";
import { ManifestSchema } from "@vb/core";
import { connect, FIXTURE_NOW, IDS, makeTempDir, textOf, writeFixtureCatalog } from "./helpers.js";

let dir: string;
let session: Awaited<ReturnType<typeof connect>>;

function counter() {
  let n = 0;
  return () => `id${++n}`;
}

type ExportResult = {
  path: string;
  duration: number;
  blocks: number;
  captions_by_dataset: Record<string, number>;
};

beforeEach(async () => {
  dir = makeTempDir();
  writeFixtureCatalog(dir);
  session = await connect(dir, { genId: counter(), now: () => FIXTURE_NOW });
});
afterEach(async () => {
  await session.close();
});

async function createWithBoth(): Promise<string> {
  const c = await session.call("create_composition", { day: "2026-04-15" });
  const id = (c.structuredContent as { composition: { id: string } }).composition.id;
  const a = await session.call("insert_event", { composition_id: id, slot: 1, candidate_id: IDS.cooking });
  expect(a.isError).toBeFalsy();
  const b = await session.call("insert_event", { composition_id: id, slot: 2, candidate_id: IDS.walking });
  expect(b.isError).toBeFalsy();
  return id;
}

describe("export_manifest", () => {
  it("end-to-end: browse → create → insert cooking → insert walking → export", async () => {
    const days = await session.call("list_base_days");
    expect(days.isError).toBeFalsy();
    const cats = await session.call("list_categories");
    const names = (cats.structuredContent as { categories: Array<{ category: string }> }).categories.map(
      (c) => c.category,
    );
    expect(names).toEqual(expect.arrayContaining(["cooking", "walking"]));
    const cand = await session.call("list_event_candidates", { category: "cooking" });
    expect(textOf(cand)).toContain(IDS.cooking);

    const id = await createWithBoth();
    const tt = await session.call("get_composition", { composition_id: id });
    const timeline = (tt.structuredContent as { timeline: Array<{ composed_end: number }> }).timeline;
    const total = timeline[timeline.length - 1]!.composed_end;
    const header = textOf(tt).split("\n")[0]!;
    const headerTotal = /총 (\d+:\d{2}:\d{2})/.exec(header)![1]!;

    const r = await session.call("export_manifest", { composition_id: id });
    expect(r.isError).toBeFalsy();
    const sc = r.structuredContent as ExportResult;
    expect(sc.path).toBe(join(dir, "exports", `${id}-r2.manifest.json`));
    expect(existsSync(sc.path)).toBe(true);

    const manifest = ManifestSchema.parse(JSON.parse(readFileSync(sc.path, "utf8")));
    expect(manifest.created_at).toBe(FIXTURE_NOW.toISOString());
    expect(manifest.composition_id).toBe(id);
    expect(manifest.composition_revision).toBe(2);
    expect(manifest.duration).toBe(total);
    expect(sc.duration).toBe(total);
    expect(sc.blocks).toBe(manifest.blocks.length);
    expect(sc.blocks).toBe(timeline.length);
    expect(Object.keys(sc.captions_by_dataset).sort()).toEqual(["ego4d", "egoexo4d", "hdepic", "nextme800"]);
    expect(sc.captions_by_dataset["hdepic"]).toBeGreaterThan(0);
    const counted: Record<string, number> = { nextme800: 0, hdepic: 0, ego4d: 0, egoexo4d: 0 };
    for (const c of manifest.tracks.captions) counted[c.provenance.dataset]! += 1;
    expect(sc.captions_by_dataset).toEqual(counted);

    const text = textOf(r);
    expect(text).toContain(sc.path);
    expect(text).toContain(`총 ${headerTotal}`);
    expect(text).toContain(`${sc.blocks}개 블록`);
    expect(text).toContain(`hdepic ${counted["hdepic"]}`);
  });

  it("export twice at same revision overwrites same path; after an edit writes r<n+1>", async () => {
    const id = await createWithBoth();
    const a = await session.call("export_manifest", { composition_id: id });
    const b = await session.call("export_manifest", { composition_id: id });
    const pa = (a.structuredContent as ExportResult).path;
    expect((b.structuredContent as ExportResult).path).toBe(pa);
    expect(readdirSync(join(dir, "exports"))).toEqual([`${id}-r2.manifest.json`]);

    const ins = await session.call("insert_event", { composition_id: id, slot: 3, candidate_id: IDS.cooking2 });
    expect(ins.isError).toBeFalsy();
    const c = await session.call("export_manifest", { composition_id: id });
    const pc = (c.structuredContent as ExportResult).path;
    expect(pc).toBe(join(dir, "exports", `${id}-r3.manifest.json`));
    expect(readdirSync(join(dir, "exports")).sort()).toEqual([
      `${id}-r2.manifest.json`,
      `${id}-r3.manifest.json`,
    ]);
    expect(ManifestSchema.parse(JSON.parse(readFileSync(pc, "utf8"))).composition_revision).toBe(3);
  });

  it("export with missing candidate returns 카탈로그에 없는 후보 and writes nothing", async () => {
    const c = await session.call("create_composition", { day: "2026-04-15" });
    const id = (c.structuredContent as { composition: { id: string } }).composition.id;
    await session.call("insert_event", { composition_id: id, slot: 0, candidate_id: IDS.cooking });

    const file = join(dir, "catalog", "hdepic", "candidates.jsonl");
    const lines = readFileSync(file, "utf8").split("\n").filter((l) => l.trim() !== "");
    const kept = lines.filter((l) => !l.includes(IDS.cooking));
    expect(kept.length).toBe(lines.length - 1);
    writeFileSync(file, kept.join("\n") + "\n");

    const r = await session.call("export_manifest", { composition_id: id });
    expect(r.isError).toBe(true);
    expect(textOf(r)).toContain(`카탈로그에 없는 후보: ${IDS.cooking}`);
    expect(existsSync(join(dir, "exports"))).toBe(false);
  });

  it("base day that changed shape after composition creation: get_composition and export_manifest fail in Korean, export writes nothing", async () => {
    const c = await session.call("create_composition", { day: "2026-04-15" });
    const id = (c.structuredContent as { composition: { id: string } }).composition.id;

    const file = join(dir, "catalog", "nextme800", "days", "2026-04-15.json");
    const day = JSON.parse(readFileSync(file, "utf8")) as { blocks: unknown[] };
    expect(day.blocks).toHaveLength(3);
    day.blocks.pop();
    writeFileSync(file, JSON.stringify(day));

    const expected = "base day 2026-04-15의 블록 구성이 바뀜 (슬롯 4개, 현재 필요 3개) — 새 composition을 만드세요";
    const g = await session.call("get_composition", { composition_id: id });
    expect(g.isError).toBe(true);
    expect(textOf(g)).toBe(expected);

    const r = await session.call("export_manifest", { composition_id: id });
    expect(r.isError).toBe(true);
    expect(textOf(r)).toBe(expected);
    expect(existsSync(join(dir, "exports"))).toBe(false);
  });

  it("unknown composition id is a tool error and writes nothing", async () => {
    const r = await session.call("export_manifest", { composition_id: "nope" });
    expect(r.isError).toBe(true);
    expect(existsSync(join(dir, "exports"))).toBe(false);
  });

  it("returns an absolute path even when the server is created with a relative data dir", async () => {
    await session.close();
    session = await connect(relative(process.cwd(), dir), { genId: counter(), now: () => FIXTURE_NOW });
    const id = await createWithBoth();
    const r = await session.call("export_manifest", { composition_id: id });
    expect(r.isError).toBeFalsy();
    const path = (r.structuredContent as ExportResult).path;
    expect(isAbsolute(path)).toBe(true);
    expect(path).toBe(join(dir, "exports", `${id}-r2.manifest.json`));
    expect(existsSync(path)).toBe(true);
  });
});
