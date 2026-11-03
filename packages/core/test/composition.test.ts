import { describe, expect, it } from "vitest";
import {
  CompositionError,
  assertRevision,
  createComposition,
  insertEvent,
  moveEvent,
  removeEvent,
  type Composition,
} from "../src/index.js";
import { makeFixtureCatalog } from "./fixtures.js";

const COOK = "hdepic:P01-x:10.000-70.000";
const WALK = "ego4d:uid-1:0.000-120.000";

function counter(prefix = "r") {
  let n = 0;
  return () => `${prefix}${++n}`;
}

function expectError(fn: () => unknown, code: string, details?: Record<string, unknown>) {
  try {
    fn();
  } catch (e) {
    expect(e).toBeInstanceOf(CompositionError);
    expect((e as CompositionError).code).toBe(code);
    if (details) expect((e as CompositionError).details).toEqual(details);
    return;
  }
  throw new Error(`expected ${code} to be thrown`);
}

function setup() {
  const catalog = makeFixtureCatalog();
  const comp = createComposition(catalog.baseDay, { genId: () => "comp-1", name: "t" });
  return { catalog, comp };
}

const refIds = (c: Composition, slot: number) => c.slots[slot]!.map((r) => r.ref_id);

describe("composition ops", () => {
  it("creates composition with n+1 empty slots", () => {
    const { comp } = setup();
    expect(comp.slots.length).toBe(4);
    expect(comp.slots.every((s) => s.length === 0)).toBe(true);
    expect(comp.revision).toBe(0);
    expect(comp.schema_version).toBe(1);
    expect(comp.id).toBe("comp-1");
    expect(comp.name).toBe("t");
    expect(comp.base_day).toBe("2026-04-15");
  });

  it("inserts event at end of slot and bumps revision", () => {
    const { catalog, comp } = setup();
    const gen = counter();
    const a = insertEvent(comp, catalog, { slot: 1, candidateId: COOK }, gen);
    const b = insertEvent(a, catalog, { slot: 1, candidateId: WALK }, gen);
    expect(a.revision).toBe(1);
    expect(b.revision).toBe(2);
    expect(b.slots[1]).toEqual([
      { ref_id: "r1", candidate_id: COOK },
      { ref_id: "r2", candidate_id: WALK },
    ]);
  });

  it("inserts at explicit position", () => {
    const { catalog, comp } = setup();
    const gen = counter();
    let c = insertEvent(comp, catalog, { slot: 0, candidateId: COOK }, gen);
    c = insertEvent(c, catalog, { slot: 0, candidateId: WALK, position: 0 }, gen);
    expect(refIds(c, 0)).toEqual(["r2", "r1"]);
  });

  it("allows same candidate twice with distinct ref_ids", () => {
    const { catalog, comp } = setup();
    const gen = counter();
    let c = insertEvent(comp, catalog, { slot: 0, candidateId: COOK }, gen);
    c = insertEvent(c, catalog, { slot: 0, candidateId: COOK }, gen);
    expect(c.slots[0]!.map((r) => r.candidate_id)).toEqual([COOK, COOK]);
    expect(new Set(refIds(c, 0)).size).toBe(2);
  });

  it("retries genId on ref_id collision", () => {
    const { catalog, comp } = setup();
    const c = insertEvent(comp, catalog, { slot: 0, candidateId: COOK }, () => "dup");
    const ids = ["dup", "fresh"];
    const d = insertEvent(c, catalog, { slot: 2, candidateId: WALK }, () => ids.shift()!);
    expect(d.slots[2]).toEqual([{ ref_id: "fresh", candidate_id: WALK }]);
  });

  it("throws when genId keeps colliding", () => {
    const { catalog, comp } = setup();
    const c = insertEvent(comp, catalog, { slot: 0, candidateId: COOK }, () => "dup");
    let calls = 0;
    expect(() =>
      insertEvent(c, catalog, { slot: 1, candidateId: WALK }, () => {
        calls++;
        return "dup";
      }),
    ).toThrow("ref_id generation collided 10 times");
    expect(calls).toBe(10);
  });

  it("rejects slot out of range", () => {
    const { catalog, comp } = setup();
    expectError(() => insertEvent(comp, catalog, { slot: 4, candidateId: COOK }, counter()), "SLOT_OUT_OF_RANGE", { slot: 4, max: 3 });
    expectError(() => insertEvent(comp, catalog, { slot: -1, candidateId: COOK }, counter()), "SLOT_OUT_OF_RANGE", { slot: -1, max: 3 });
    expectError(() => insertEvent(comp, catalog, { slot: 1.5, candidateId: COOK }, counter()), "SLOT_OUT_OF_RANGE");
  });

  it("rejects position beyond slot length", () => {
    const { catalog, comp } = setup();
    expectError(() => insertEvent(comp, catalog, { slot: 0, candidateId: COOK, position: 1 }, counter()), "POSITION_OUT_OF_RANGE", { position: 1, max: 0 });
    expectError(() => insertEvent(comp, catalog, { slot: 0, candidateId: COOK, position: -1 }, counter()), "POSITION_OUT_OF_RANGE", { position: -1, max: 0 });
    expectError(() => insertEvent(comp, catalog, { slot: 0, candidateId: COOK, position: 0.5 }, counter()), "POSITION_OUT_OF_RANGE");
  });

  it("rejects unknown candidate", () => {
    const { catalog, comp } = setup();
    expectError(() => insertEvent(comp, catalog, { slot: 0, candidateId: "nope" }, counter()), "CANDIDATE_NOT_FOUND", { candidateId: "nope" });
  });

  it("rejects base candidate", () => {
    const { catalog, comp } = setup();
    const baseId = catalog.baseDay.blocks[0]!.id;
    expectError(() => insertEvent(comp, catalog, { slot: 0, candidateId: baseId }, counter()), "NOT_EVENT", { candidateId: baseId });
  });

  it("moves event across slots and within a slot", () => {
    const { catalog, comp } = setup();
    const gen = counter();
    let c = insertEvent(comp, catalog, { slot: 0, candidateId: COOK }, gen); // r1
    c = insertEvent(c, catalog, { slot: 0, candidateId: WALK }, gen); // r2
    const across = moveEvent(c, { refId: "r1", toSlot: 2 });
    expect(refIds(across, 0)).toEqual(["r2"]);
    expect(refIds(across, 2)).toEqual(["r1"]);
    expect(across.revision).toBe(c.revision + 1);

    const within = moveEvent(c, { refId: "r1", toSlot: 0 });
    expect(refIds(within, 0)).toEqual(["r2", "r1"]);
    const withinPos = moveEvent(within, { refId: "r1", toSlot: 0, toPosition: 0 });
    expect(refIds(withinPos, 0)).toEqual(["r1", "r2"]);

    expectError(() => moveEvent(c, { refId: "r1", toSlot: 9 }), "SLOT_OUT_OF_RANGE", { slot: 9, max: 3 });
    expectError(() => moveEvent(c, { refId: "r1", toSlot: 0, toPosition: 2 }), "POSITION_OUT_OF_RANGE", { position: 2, max: 1 });
    expectError(() => moveEvent(c, { refId: "r1", toSlot: 1, toPosition: 1 }), "POSITION_OUT_OF_RANGE", { position: 1, max: 0 });
  });

  it("move/remove unknown ref", () => {
    const { comp } = setup();
    expectError(() => moveEvent(comp, { refId: "zz", toSlot: 0 }), "REF_NOT_FOUND", { refId: "zz" });
    expectError(() => removeEvent(comp, { refId: "zz" }), "REF_NOT_FOUND", { refId: "zz" });
  });

  it("removes an event and bumps revision", () => {
    const { catalog, comp } = setup();
    const c = insertEvent(comp, catalog, { slot: 3, candidateId: COOK }, counter());
    const r = removeEvent(c, { refId: "r1" });
    expect(r.slots[3]).toEqual([]);
    expect(r.revision).toBe(c.revision + 1);
  });

  it("does not mutate input", () => {
    const { catalog, comp } = setup();
    const c = insertEvent(comp, catalog, { slot: 0, candidateId: COOK }, counter());
    const before = structuredClone(c);
    insertEvent(c, catalog, { slot: 0, candidateId: WALK }, counter("x"));
    moveEvent(c, { refId: "r1", toSlot: 2 });
    removeEvent(c, { refId: "r1" });
    expect(c).toEqual(before);
  });

  it("assertRevision mismatch", () => {
    const { comp } = setup();
    expectError(() => assertRevision(comp, 5), "REVISION_MISMATCH", { expected: 5, current: comp.revision });
    expect(() => assertRevision(comp, undefined)).not.toThrow();
    expect(() => assertRevision(comp, comp.revision)).not.toThrow();
  });
});
