import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { GapTableSchema, TimelineError, rngFor, sampleGap, type GapTable } from "../src/index.js";

const Q = { p10: 4, p25: 7, p50: 10, p75: 16, p90: 25 };

function table(): GapTable {
  return GapTableSchema.parse({
    version: "test",
    reviewed: false,
    transitions: { "home->workplace": Q, "workplace->home": Q },
  });
}

describe("sampleGap", () => {
  it("same kind → 0", () => {
    expect(sampleGap(table(), "hobby", "hobby", rngFor(1, "g"))).toBe(0);
  });

  it('missing pair throws GAP_MISSING naming "home->hobby"', () => {
    let err: unknown;
    try {
      sampleGap(table(), "home", "hobby", rngFor(1, "g"));
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(TimelineError);
    expect((err as TimelineError).code).toBe("GAP_MISSING");
    expect((err as TimelineError).details).toEqual({ transition: "home->hobby" });
  });

  it("samples stay within [p10,p90]", () => {
    const rng = rngFor(2, "g");
    for (let i = 0; i < 1000; i++) {
      const g = sampleGap(table(), "home", "workplace", rng);
      expect(Number.isInteger(g)).toBe(true);
      expect(g).toBeGreaterThanOrEqual(Q.p10);
      expect(g).toBeLessThanOrEqual(Q.p90);
    }
  });

  it("median of 10 000 draws within ±1 of p50", () => {
    const rng = rngFor(3, "g");
    const xs = Array.from({ length: 10_000 }, () => sampleGap(table(), "workplace", "home", rng)).sort(
      (a, b) => a - b,
    );
    const median = xs[5000]!;
    expect(Math.abs(median - Q.p50)).toBeLessThanOrEqual(1);
  });
});

describe("gaps.v1.json", () => {
  it("parses and has all 12 ordered pairs", () => {
    const url = new URL("../../../config/timeline/gaps.v1.json", import.meta.url);
    const t = GapTableSchema.parse(JSON.parse(readFileSync(url, "utf8")));
    expect(t.version).toBe("gaps.v1");
    expect(t.reviewed).toBe(false);
    const kinds = ["home", "workplace", "hobby", "frequent_place"];
    const pairs = kinds.flatMap((a) => kinds.filter((b) => b !== a).map((b) => `${a}->${b}`));
    expect(Object.keys(t.transitions).sort()).toEqual(pairs.sort());
    expect(t.transitions["home->workplace"]).toEqual({ p10: 7, p25: 15, p50: 20, p75: 35, p90: 60 });
    expect(t.transitions["workplace->home"]).toEqual({ p10: 6, p25: 15, p50: 20, p75: 30, p90: 55 });
  });
});
