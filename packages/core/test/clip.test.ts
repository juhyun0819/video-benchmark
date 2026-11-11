import { describe, expect, it } from "vitest";
import { ClipContractRowSchema, ClipSchema, ClipSpecSchema } from "../src/clip.js";

const clip = {
  category: "meal",
  setting: "home_indoor",
  node: "home",
  activity: "cooking dinner",
  time_of_day: "evening",
  clip_id: "yt-abc-0001",
  source: { dataset: "youtube", video_id: "abc", media: { kind: "video", fps: 30 } },
  source_start_s: 10,
  source_end_s: 70,
  day_group: "abc",
  person_source: "creator-1",
  fingerprint: { duration_s: 600, phash: ["a1b2c3d4e5f60718"] },
  extractor_version: "0.1.0",
};

describe("clip contract", () => {
  it("parses a valid clip and defaults tags", () => {
    const c = ClipSchema.parse(clip);
    expect(c.tags).toEqual([]);
  });

  it("accepts a null fingerprint", () => {
    expect(ClipSchema.safeParse({ ...clip, fingerprint: null }).success).toBe(true);
  });

  it("accepts unknown time_of_day on Clip but not on ClipSpec", () => {
    expect(ClipSchema.safeParse({ ...clip, time_of_day: "unknown" }).success).toBe(true);
    expect(ClipSpecSchema.safeParse({ ...clip, time_of_day: "unknown" }).success).toBe(false);
  });

  it("rejects an end that is not after the start", () => {
    expect(ClipSchema.safeParse({ ...clip, source_end_s: 10 }).success).toBe(false);
    expect(ClipSchema.safeParse({ ...clip, source_end_s: 5 }).success).toBe(false);
  });

  it("rejects an unknown category", () => {
    expect(ClipSchema.safeParse({ ...clip, category: "sleep" }).success).toBe(false);
  });

  it("parses a contract row with captions", () => {
    const row = ClipContractRowSchema.parse({
      ...clip,
      captions: [{ id: "c1", level: "L2", time: { kind: "interval", start: 10, end: 20 }, text: "stirs a pot" }],
    });
    expect(row.captions).toHaveLength(1);
  });
});
