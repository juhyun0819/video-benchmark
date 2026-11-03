import { describe, expect, it } from "vitest";
import { tileRecording, type Annotation, type SourceRef } from "../src/index.js";

const source: SourceRef = {
  dataset: "nextme800",
  video_id: "v1",
  media: { kind: "frames", approx_fps: 1, tar: "v1.tar" },
};

function l4(start: number, end: number, text = `act-${start}`): Annotation {
  return {
    id: `l4-${start}`,
    track: "captions",
    level: "L4",
    time: { kind: "interval", start, end },
    text,
    original: {},
  };
}

const spans = (bs: { src_start: number; src_end: number }[]) =>
  bs.map((b) => [b.src_start, b.src_end]);

describe("tileRecording", () => {
  it("tiles at L4 starts covering whole recording", () => {
    const blocks = tileRecording({
      source,
      duration: 3600,
      l4: [l4(300, 1800, "a"), l4(1800, 3600, "b")],
    });
    expect(spans(blocks)).toEqual([
      [0, 300],
      [300, 1800],
      [1800, 3600],
    ]);
    for (const b of blocks) {
      expect(b.role).toBe("base");
      expect(b.category).toBe("base");
      expect(b.source).toEqual(source);
      expect(b.id).toBe(`nextme800:v1:${b.src_start.toFixed(3)}-${b.src_end.toFixed(3)}`);
    }
    expect(blocks[0]?.id).toBe("nextme800:v1:0.000-300.000");
    expect(blocks.map((b) => b.summary)).toEqual(["(unlabeled)", "a", "b"]);
  });

  it("first block before any L4 is unlabeled", () => {
    const blocks = tileRecording({ source, duration: 3600, l4: [l4(300, 1800)] });
    expect(blocks[0]?.summary).toBe("(unlabeled)");
  });

  it("merges sub-60s block into previous", () => {
    const blocks = tileRecording({
      source,
      duration: 3600,
      l4: [l4(300, 330, "x"), l4(330, 3600, "y")],
    });
    // [300,330] is 30 s, so it merges into the previous block [0,300] -> [0,330].
    expect(spans(blocks)).toEqual([
      [0, 330],
      [330, 3600],
    ]);
    expect(blocks[0]?.summary).toBe("(unlabeled)");
    expect(blocks[1]?.summary).toBe("y");
  });

  it("merges sub-60s first block into next", () => {
    const blocks = tileRecording({ source, duration: 3600, l4: [l4(30, 3600, "z")] });
    expect(spans(blocks)).toEqual([[0, 3600]]);
    expect(blocks[0]?.summary).toBe("(unlabeled)");
  });

  it("recording shorter than 60s stays one block", () => {
    const blocks = tileRecording({ source, duration: 40, l4: [] });
    expect(spans(blocks)).toEqual([[0, 40]]);
  });

  it("no L4 -> single unlabeled block", () => {
    const blocks = tileRecording({ source, duration: 3600, l4: [] });
    expect(blocks).toHaveLength(1);
    expect(blocks[0]?.summary).toBe("(unlabeled)");
  });

  it("ignores L4 starts outside (0, duration)", () => {
    const blocks = tileRecording({
      source,
      duration: 3600,
      l4: [l4(-5, 10), l4(4000, 4100)],
    });
    expect(spans(blocks)).toEqual([[0, 3600]]);
  });

  it("ignores non-interval annotations", () => {
    const point: Annotation = { ...l4(0, 0), time: { kind: "point", t: 500 } };
    const blocks = tileRecording({ source, duration: 3600, l4: [point] });
    expect(spans(blocks)).toEqual([[0, 3600]]);
  });
});
