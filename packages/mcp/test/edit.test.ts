import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { connect, IDS, makeTempDir, textOf, writeFixtureCatalog } from "./helpers.js";

let dir: string;
let session: Awaited<ReturnType<typeof connect>>;

function counter() {
  let n = 0;
  return () => `id${++n}`;
}

beforeEach(async () => {
  dir = makeTempDir();
  writeFixtureCatalog(dir);
  session = await connect(dir, { genId: counter() });
});
afterEach(async () => {
  await session.close();
});

const onDisk = (id: string) => JSON.parse(readFileSync(join(dir, "compositions", `${id}.json`), "utf8"));

describe("edit tools", () => {
  it("create → insert → move → remove round trip updates revision 0→1→2→3 and persists to data/compositions/<id>.json", async () => {
    const c = await session.call("create_composition", { day: "2026-04-15", name: "demo" });
    expect(c.isError).toBeFalsy();
    const created = c.structuredContent as { composition: { id: string; revision: number; name: string } };
    const id = created.composition.id;
    expect(id).toBe("id1");
    expect(created.composition.revision).toBe(0);
    expect(existsSync(join(dir, "compositions", `${id}.json`))).toBe(true);
    expect(textOf(c)).toContain("rev 0 · 총 0:30:00");

    const ins = await session.call("insert_event", {
      composition_id: id,
      slot: 1,
      candidate_id: IDS.cooking,
      expected_revision: 0,
    });
    expect(ins.isError).toBeFalsy();
    const insSc = ins.structuredContent as {
      composition: { revision: number; slots: Array<Array<{ ref_id: string }>> };
      timeline: unknown[];
    };
    expect(insSc.composition.revision).toBe(1);
    expect(insSc.timeline).toHaveLength(4);
    const refId = insSc.composition.slots[1]![0]!.ref_id;
    expect(refId).toBe("id2");
    expect(textOf(ins)).toContain(`S1  00:10:00–00:11:00  cooking   evt:${refId}  Cooking onions in a pan`);
    expect(onDisk(id).revision).toBe(1);

    const mv = await session.call("move_event", { composition_id: id, ref_id: refId, to_slot: 3 });
    expect(mv.isError).toBeFalsy();
    expect((mv.structuredContent as { composition: { revision: number } }).composition.revision).toBe(2);
    expect(textOf(mv)).toContain(`S3  00:30:00–00:31:00  cooking   evt:${refId}`);
    expect(onDisk(id).slots[3]).toHaveLength(1);

    const rm = await session.call("remove_event", { composition_id: id, ref_id: refId, expected_revision: 2 });
    expect(rm.isError).toBeFalsy();
    expect((rm.structuredContent as { composition: { revision: number } }).composition.revision).toBe(3);
    expect(onDisk(id).revision).toBe(3);
    expect(onDisk(id).slots.flat()).toHaveLength(0);

    const got = await session.call("get_composition", { composition_id: id });
    expect(textOf(got)).toContain("rev 3 · 총 0:30:00");

    const list = await session.call("list_compositions");
    expect(list.structuredContent).toEqual({
      compositions: [{ id, name: "demo", base_day: "2026-04-15", revision: 3 }],
    });
  });

  it("insert with stale expected_revision returns revision 불일치 and does not save", async () => {
    const c = await session.call("create_composition", { day: "2026-04-15" });
    const id = (c.structuredContent as { composition: { id: string } }).composition.id;
    const r = await session.call("insert_event", {
      composition_id: id,
      slot: 0,
      candidate_id: IDS.cooking,
      expected_revision: 5,
    });
    expect(r.isError).toBe(true);
    expect(textOf(r)).toContain("revision 불일치");
    expect(textOf(r)).toContain("현재 0");
    expect(onDisk(id).revision).toBe(0);
    expect(onDisk(id).slots.flat()).toHaveLength(0);
  });

  it("slot out of range returns Korean range message", async () => {
    const c = await session.call("create_composition", { day: "2026-04-15" });
    const id = (c.structuredContent as { composition: { id: string } }).composition.id;
    const r = await session.call("insert_event", { composition_id: id, slot: 21, candidate_id: IDS.cooking });
    expect(r.isError).toBe(true);
    expect(textOf(r)).toBe("slot 21은 범위 밖 (0..3)");
    expect(onDisk(id).revision).toBe(0);
  });

  it("create_composition with unknown day is an error", async () => {
    const r = await session.call("create_composition", { day: "1999-01-01" });
    expect(r.isError).toBe(true);
    expect(textOf(r)).toContain("BASE_DAY_NOT_FOUND");
  });

  it("get_composition on composition referencing removed candidate lists missing ids", async () => {
    const c = await session.call("create_composition", { day: "2026-04-15" });
    const id = (c.structuredContent as { composition: { id: string } }).composition.id;
    await session.call("insert_event", { composition_id: id, slot: 0, candidate_id: IDS.cooking });

    const file = join(dir, "catalog", "hdepic", "candidates.jsonl");
    const lines = readFileSync(file, "utf8").split("\n").filter((l) => l.trim() !== "");
    const kept = lines.filter((l) => !l.includes(IDS.cooking));
    expect(kept.length).toBe(lines.length - 1);
    writeFileSync(file, kept.join("\n") + "\n");

    const r = await session.call("get_composition", { composition_id: id });
    expect(r.isError).toBe(true);
    expect(textOf(r)).toContain(`카탈로그에 없는 후보: ${IDS.cooking}`);
  });

  /** composition with a walking ref (id2) in slot 0; walking then removed from the catalog. */
  async function staleComposition(): Promise<{ id: string; refId: string }> {
    const c = await session.call("create_composition", { day: "2026-04-15" });
    const id = (c.structuredContent as { composition: { id: string } }).composition.id;
    const ins = await session.call("insert_event", { composition_id: id, slot: 0, candidate_id: IDS.walking });
    const refId = (ins.structuredContent as { composition: { slots: Array<Array<{ ref_id: string }>> } })
      .composition.slots[0]![0]!.ref_id;
    const file = join(dir, "catalog", "ego4d", "candidates.jsonl");
    const lines = readFileSync(file, "utf8").split("\n").filter((l) => l.trim() !== "");
    const kept = lines.filter((l) => !l.includes(IDS.walking));
    expect(kept.length).toBe(lines.length - 1);
    writeFileSync(file, kept.join("\n") + (kept.length ? "\n" : ""));
    return { id, refId };
  }

  it("editing an already-stale composition saves, then reports the saved revision with the resolve error", async () => {
    const { id } = await staleComposition();
    const r = await session.call("insert_event", { composition_id: id, slot: 1, candidate_id: IDS.cooking });
    expect(r.isError).toBe(true);
    expect(textOf(r)).toContain(`카탈로그에 없는 후보: ${IDS.walking}`);
    expect(textOf(r)).toContain(" (변경 사항은 rev 2으로 저장됨)");
    expect(onDisk(id).revision).toBe(2);
    expect(onDisk(id).slots[1]).toHaveLength(1);
  });

  it("remove_event on the stale ref repairs the composition and returns a normal timetable", async () => {
    const { id, refId } = await staleComposition();
    const r = await session.call("remove_event", { composition_id: id, ref_id: refId });
    expect(r.isError).toBeFalsy();
    expect(textOf(r)).toContain("rev 2 · 총 0:30:00");
    expect(onDisk(id).revision).toBe(2);
    const got = await session.call("get_composition", { composition_id: id });
    expect(got.isError).toBeFalsy();
  });
});
