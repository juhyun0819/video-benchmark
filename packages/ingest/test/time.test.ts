import { describe, expect, it } from "vitest";
import { parseHkt } from "../src/time.js";

describe("parseHkt", () => {
  it("parses HKT independent of TZ", () => {
    expect(parseHkt("2026-04-15 09:00:00")).toBe(1776214800);
    expect(parseHkt("2026-04-15 09:00:01")).toBe(1776214801);
    // midnight rollover keeps working across days
    expect(parseHkt("2026-04-16 00:00:00") - parseHkt("2026-04-15 23:59:59")).toBe(1);
  });

  it("throws on malformed", () => {
    for (const bad of [
      "",
      "2026-04-15",
      "2026-04-15T09:00:00",
      "2026-04-15 9:00:00",
      "2026-13-01 00:00:00",
      "2026-02-30 00:00:00",
      "2026-04-15 25:00:00",
      "2026-04-15 09:00:00Z",
    ]) {
      expect(() => parseHkt(bad), bad).toThrow();
    }
  });
});
