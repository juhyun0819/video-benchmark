import { describe, expect, it } from "vitest";
import {
  CompositionError,
  createComposition,
  insertEvent,
  resolveTimeline,
  type Composition,
} from "../src/index.js";
import { makeFixtureCatalog } from "./fixtures.js";

const COOK = "hdepic:P01-x:10.000-70.000";

describe("resolveTimeline", () => {
  it("orders slot events before following base block", () => {
    const catalog = makeFixtureCatalog();
    const empty = createComposition(catalog.baseDay, { genId: () => "c1" });
    const comp = insertEvent(empty, catalog, { slot: 1, candidateId: COOK }, () => "ref1");
    const blocks = resolveTimeline(comp, catalog);
    expect(
      blocks.map((b) => [b.block_id, b.composed_start, b.composed_end, b.src_start, b.src_end]),
    ).toEqual([
      ["base:2026-04-15:0", 0, 600, 0, 600],
      ["evt:ref1", 600, 660, 10, 70],
      ["base:2026-04-15:1", 660, 1560, 600, 1500],
      ["base:2026-04-15:2", 1560, 1860, 0, 300],
    ]);
    expect(blocks[1]).toMatchObject({
      role: "event",
      category: "cooking",
      slot: 1,
      ref_id: "ref1",
      summary: "Cooking onions in a pan",
    });
    expect(blocks[1]!.base_index).toBeUndefined();
    expect(blocks[2]).toMatchObject({ role: "base", base_index: 1 });
    expect(blocks[2]!.ref_id).toBeUndefined();
    expect(blocks[0]!.source.video_id).toBe("rec-A");
  });

  it("empty composition equals base blocks back to back", () => {
    const catalog = makeFixtureCatalog();
    const comp = createComposition(catalog.baseDay, { genId: () => "c1" });
    const blocks = resolveTimeline(comp, catalog);
    expect(blocks.map((b) => [b.composed_start, b.composed_end])).toEqual([
      [0, 600],
      [600, 1500],
      [1500, 1800],
    ]);
  });

  it("lists all missing candidate ids", () => {
    const catalog = makeFixtureCatalog();
    const comp: Composition = {
      ...createComposition(catalog.baseDay, { genId: () => "c1" }),
      slots: [
        [{ ref_id: "a", candidate_id: "hdepic:gone-1:0.000-1.000" }],
        [
          { ref_id: "b", candidate_id: COOK },
          { ref_id: "c", candidate_id: "hdepic:gone-2:0.000-1.000" },
        ],
        [{ ref_id: "d", candidate_id: "hdepic:gone-1:0.000-1.000" }],
        [],
      ],
    };
    try {
      resolveTimeline(comp, catalog);
    } catch (e) {
      expect(e).toBeInstanceOf(CompositionError);
      expect((e as CompositionError).code).toBe("MISSING_CANDIDATES");
      expect((e as CompositionError).details).toEqual({
        ids: ["hdepic:gone-1:0.000-1.000", "hdepic:gone-2:0.000-1.000"],
      });
      return;
    }
    throw new Error("expected MISSING_CANDIDATES");
  });

  it("throws BASE_DAY_NOT_FOUND for an unknown day", () => {
    const catalog = makeFixtureCatalog();
    const comp = { ...createComposition(catalog.baseDay, { genId: () => "c1" }), base_day: "1999-01-01" };
    try {
      resolveTimeline(comp, catalog);
    } catch (e) {
      expect((e as CompositionError).code).toBe("BASE_DAY_NOT_FOUND");
      expect((e as CompositionError).details).toEqual({ day: "1999-01-01" });
      return;
    }
    throw new Error("expected BASE_DAY_NOT_FOUND");
  });
});
