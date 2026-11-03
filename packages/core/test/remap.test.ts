import { describe, expect, it } from "vitest";
import {
  remapAnnotations,
  type Annotation,
  type TimelineBlock,
  type TimeSpan,
} from "../src/index.js";

const block: TimelineBlock = {
  block_id: "base:2026-04-15:1",
  role: "base",
  category: "base",
  summary: "Commute and arrival",
  source: {
    dataset: "nextme800",
    video_id: "rec-A",
    media: { kind: "frames", approx_fps: 1, tar: "frames/rec-A.tar" },
  },
  src_start: 600,
  src_end: 1500,
  composed_start: 660,
  composed_end: 1560,
  base_index: 1,
};

function ann(id: string, time: TimeSpan, extra: Record<string, unknown> = {}): Annotation {
  return {
    id,
    track: "captions",
    level: "L1",
    time,
    text: `text ${id}`,
    original: { text: `text ${id}`, ...extra },
  };
}
const iv = (start: number, end: number): TimeSpan => ({ kind: "interval", start, end });
const pt = (t: number): TimeSpan => ({ kind: "point", t });

describe("remapAnnotations", () => {
  it("shifts interval by block offset", () => {
    const [c] = remapAnnotations(block, [ann("a", iv(700, 710))]);
    expect(c!.time).toEqual(iv(760, 770));
    expect(c!.id).toBe("base:2026-04-15:1:a");
    expect(c!.block_id).toBe(block.block_id);
    expect(c!.level).toBe("L1");
    expect("clipped" in c!).toBe(false);
    expect(c!.provenance).toMatchObject({
      dataset: "nextme800",
      video_id: "rec-A",
      orig_id: "a",
      src_time: iv(700, 710),
    });
    expect("orig_time_string" in c!.provenance).toBe(false);
  });

  it("clips interval crossing block start and marks clipped.start", () => {
    const [c] = remapAnnotations(block, [ann("a", iv(590, 610))]);
    expect(c!.time).toEqual(iv(660, 670));
    expect(c!.clipped).toEqual({ start: true });
    expect(c!.provenance.src_time).toEqual(iv(590, 610));
  });

  it("clips interval crossing block end", () => {
    const [c] = remapAnnotations(block, [ann("a", iv(1490, 1510))]);
    expect(c!.time).toEqual(iv(1550, 1560));
    expect(c!.clipped).toEqual({ end: true });
    expect(c!.provenance.src_time).toEqual(iv(1490, 1510));
  });

  it("marks both sides when the interval spans the whole block", () => {
    const [c] = remapAnnotations(block, [ann("a", iv(500, 1600))]);
    expect(c!.time).toEqual(iv(660, 1560));
    expect(c!.clipped).toEqual({ start: true, end: true });
  });

  it("point at src_end excluded, at src_start included", () => {
    const out = remapAnnotations(block, [ann("end", pt(1500)), ann("start", pt(600)), ann("mid", pt(700))]);
    expect(out.map((c) => c.provenance.orig_id)).toEqual(["start", "mid"]);
    expect(out[0]!.time).toEqual(pt(660));
  });

  it("interval ending exactly at src_start excluded", () => {
    expect(remapAnnotations(block, [ann("a", iv(500, 600))])).toEqual([]);
    expect(remapAnnotations(block, [ann("b", iv(1500, 1600))])).toEqual([]);
  });

  it("zero-length interval treated as point", () => {
    const out = remapAnnotations(block, [
      ann("in", iv(700, 700)),
      ann("atStart", iv(600, 600)),
      ann("atEnd", iv(1500, 1500)),
    ]);
    expect(out.map((c) => c.provenance.orig_id)).toEqual(["in", "atStart"]);
    expect(out[0]!.time).toEqual(iv(760, 760));
    expect("clipped" in out[0]!).toBe(false);
  });

  it("copies text and original verbatim and NextMe time strings into provenance", () => {
    const original = {
      recording_id: "rec-A",
      start: "2026-04-15 09:10:00",
      end: "2026-04-15 09:11:00",
      text: "text a",
    };
    const a: Annotation = { id: "a", track: "captions", level: "L2", time: iv(600, 660), text: "text a", original };
    const [c] = remapAnnotations(block, [a]);
    expect(c!.text).toBe("text a");
    expect(c!.provenance.original).toEqual(original);
    expect(c!.provenance.orig_time_string).toEqual({
      start: "2026-04-15 09:10:00",
      end: "2026-04-15 09:11:00",
    });
  });

  it("includes only the string-valued orig_time_string sides", () => {
    const [c] = remapAnnotations(block, [ann("a", iv(700, 710), { start: "x", end: 710 })]);
    expect(c!.provenance.orig_time_string).toEqual({ start: "x" });
  });

  it("keeps annotation input order", () => {
    const out = remapAnnotations(block, [ann("z", pt(900)), ann("a", pt(700))]);
    expect(out.map((c) => c.provenance.orig_id)).toEqual(["z", "a"]);
  });
});
