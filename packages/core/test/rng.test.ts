import { describe, expect, it } from "vitest";
import { hash32, Rng, rngFor } from "../src/rng.js";

const take = (r: Rng, n: number) => Array.from({ length: n }, () => r.next());

describe("rng", () => {
  it("hash32 matches FNV-1a reference values", () => {
    expect(hash32("")).toBe(0x811c9dc5);
    expect(hash32("a")).toBe(0xe40c292c);
    expect(hash32("foobar")).toBe(0xbf9cf968);
  });
  it("is reproducible per key set", () => {
    expect(take(rngFor(1, "a", 2), 5)).toEqual(take(rngFor(1, "a", 2), 5));
  });
  it("differs across keys", () => {
    expect(take(rngFor(1, "a", 2), 5)).not.toEqual(take(rngFor(1, "a", 3), 5));
  });
  it("keeps next() in [0,1)", () => {
    const r = rngFor(7, "range");
    for (let i = 0; i < 10000; i++) {
      const v = r.next();
      expect(v >= 0 && v < 1).toBe(true);
    }
  });
  it("int stays in [lo, hiExcl)", () => {
    const r = rngFor(3, "int");
    for (let i = 0; i < 2000; i++) {
      const v = r.int(5, 9);
      expect(Number.isInteger(v) && v >= 5 && v < 9).toBe(true);
    }
  });
  it("pickWeighted follows weights", () => {
    const r = rngFor(2, "pick");
    const items = [{ value: "a", weight: 1 }, { value: "b", weight: 3 }] as const;
    let b = 0;
    const n = 100000;
    for (let i = 0; i < n; i++) if (r.pickWeighted(items) === "b") b++;
    expect(b / n).toBeGreaterThanOrEqual(0.74);
    expect(b / n).toBeLessThanOrEqual(0.76);
  });
  it("normal has the requested mean and sd", () => {
    const r = rngFor(4, "normal");
    const n = 50000;
    const xs = Array.from({ length: n }, () => r.normal(10, 2));
    const mean = xs.reduce((s, x) => s + x, 0) / n;
    const sd = Math.sqrt(xs.reduce((s, x) => s + (x - mean) ** 2, 0) / n);
    expect(Math.abs(mean - 10)).toBeLessThan(0.05);
    expect(Math.abs(sd - 2)).toBeLessThan(0.05);
  });
  it("bernoulli respects extremes", () => {
    const r = rngFor(5, "b");
    expect(r.bernoulli(0)).toBe(false);
    expect(r.bernoulli(1)).toBe(true);
  });
});
