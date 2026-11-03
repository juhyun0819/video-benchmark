import { describe, expect, it } from "vitest";
import {
  AnnotationSchema,
  BaseDaySchema,
  BlockCandidateSchema,
  SourceRefSchema,
  CompositionSchema,
  CompositionError,
} from "../src/index.js";
import { makeFixtureCatalog } from "./fixtures.js";

describe("core schemas", () => {
  it("parses fixture catalog base day and candidates", () => {
    const catalog = makeFixtureCatalog();
    expect(BaseDaySchema.safeParse(catalog.baseDay).success).toBe(true);
    expect(catalog.baseDay.blocks).toHaveLength(3);
    for (const b of catalog.baseDay.blocks) {
      expect(BlockCandidateSchema.safeParse(b).success).toBe(true);
      expect(catalog.getCandidate(b.id)).toEqual(b);
    }
    for (const id of ["hdepic:P01-x:10.000-70.000", "ego4d:uid-1:0.000-120.000"]) {
      const c = catalog.getCandidate(id);
      expect(c?.role).toBe("event");
      expect(BlockCandidateSchema.safeParse(c).success).toBe(true);
    }
    for (const b of catalog.baseDay.blocks) {
      for (const a of catalog.getAnnotations(b.source.dataset, b.source.video_id)) {
        expect(AnnotationSchema.safeParse(a).success).toBe(true);
      }
    }
    for (const [ds, vid] of [
      ["hdepic", "P01-x"],
      ["ego4d", "uid-1"],
    ] as const) {
      const anns = catalog.getAnnotations(ds, vid);
      expect(anns.length).toBeGreaterThan(0);
      for (const a of anns) expect(AnnotationSchema.safeParse(a).success).toBe(true);
    }
    expect(catalog.getBaseDay("2026-04-15")).toBe(catalog.baseDay);
    expect(catalog.getBaseDay("nope")).toBeUndefined();
    expect(catalog.getCandidate("nope")).toBeUndefined();
  });

  it("rejects interval annotation without end", () => {
    const valid = {
      id: "a1",
      track: "captions",
      level: "L1",
      time: { kind: "interval", start: 0, end: 5 },
      text: "hello",
      original: {},
    };
    expect(AnnotationSchema.safeParse(valid).success).toBe(true);
    const bad = { ...valid, time: { kind: "interval", start: 1 } };
    expect(AnnotationSchema.safeParse(bad).success).toBe(false);
  });

  it("rejects composition with schema_version 2", () => {
    const valid = {
      schema_version: 1,
      id: "c1",
      base_day: "2026-04-15",
      revision: 0,
      slots: [[], [], [], []],
    };
    expect(CompositionSchema.safeParse(valid).success).toBe(true);
    expect(CompositionSchema.safeParse({ ...valid, schema_version: 2 }).success).toBe(false);
  });

  it("CompositionError carries code and details", () => {
    const e = new CompositionError("SLOT_OUT_OF_RANGE", "bad slot", { slot: 9 });
    expect(e).toBeInstanceOf(Error);
    expect(e.code).toBe("SLOT_OUT_OF_RANGE");
    expect(e.details).toEqual({ slot: 9 });
    expect(new CompositionError("NOT_EVENT", "x").details).toEqual({});
  });

  it("accepts optional start_wallclock on frames media", () => {
    const base = { dataset: "nextme800", video_id: "r" } as const;
    const withClock = SourceRefSchema.safeParse({
      ...base,
      media: { kind: "frames", approx_fps: 1, tar: "t.tar", start_wallclock: "2026-04-15 09:00:00" },
    });
    expect(withClock.success).toBe(true);
    expect(withClock.data?.media).toMatchObject({ start_wallclock: "2026-04-15 09:00:00" });
    const without = SourceRefSchema.safeParse({
      ...base,
      media: { kind: "frames", approx_fps: 1, tar: "t.tar" },
    });
    expect(without.success).toBe(true);
  });
});
