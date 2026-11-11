import { describe, expect, it } from "vitest";
import { bucketOverlaps, formatClock, parseClock, timeOfDayAt } from "../src/clock.js";

describe("clock", () => {
  it("parses HH:MM including hours past midnight", () => {
    expect(parseClock("07:52")).toBe(472);
    expect(parseClock("24:30")).toBe(1470);
    expect(parseClock("27:59")).toBe(1679);
  });
  it("rejects malformed clocks", () => {
    for (const s of ["7:5", "28:00", "12:60", "1200", "ab:cd", "12:30 ", ""]) {
      expect(() => parseClock(s)).toThrow();
    }
  });
  it("formats without wrapping and rounds to the nearest minute", () => {
    expect(formatClock(1470)).toBe("24:30");
    expect(formatClock(472.4)).toBe("07:52");
    expect(formatClock(472.6)).toBe("07:53");
  });
  it("maps minutes to time-of-day buckets", () => {
    expect(timeOfDayAt(30)).toBe("night");
    expect(timeOfDayAt(420)).toBe("morning");
    expect(timeOfDayAt(240)).toBe("early_morning");
    expect(timeOfDayAt(1260)).toBe("night");
    expect(timeOfDayAt(1679)).toBe("night");
    expect(timeOfDayAt(1680)).toBe("early_morning");
    expect(timeOfDayAt(1700)).toBe("early_morning");
  });
  it("checks half-open bucket overlap", () => {
    expect(bucketOverlaps("evening", [1000, 1030])).toBe(true);
    expect(bucketOverlaps("evening", [900, 1010])).toBe(false);
    expect(bucketOverlaps("evening", [900, 1010], 30)).toBe(true);
    expect(bucketOverlaps("evening", [1000, 1020])).toBe(false);
    expect(bucketOverlaps("evening", [1260, 1300])).toBe(false);
    expect(bucketOverlaps("unknown", [0, 1])).toBe(true);
  });
});
