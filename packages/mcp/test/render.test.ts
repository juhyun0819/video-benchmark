import { describe, expect, it } from "vitest";
import { createComposition, insertEvent, resolveTimeline } from "@vb/core";
import { makeFixtureCatalog } from "../../core/test/fixtures.js";
import { renderTimetable } from "../src/render.js";

describe("renderTimetable", () => {
  it("renders empty slots and base lines in order", () => {
    const catalog = makeFixtureCatalog();
    let comp = createComposition(catalog.baseDay, { genId: () => "c" });
    comp = insertEvent(comp, catalog, { slot: 1, candidateId: "hdepic:P01-x:10.000-70.000" }, () => "a1");
    const text = renderTimetable(comp, resolveTimeline(comp, catalog));
    expect(text).toBe(
      [
        "rev 1 · total 0:31:00",
        "S0  (empty)",
        "B0  00:00:00–00:10:00  base      Morning at home",
        "S1  00:10:00–00:11:00  cooking   evt:a1  Cooking onions in a pan",
        "B1  00:11:00–00:26:00  base      Commute and arrival",
        "S2  (empty)",
        "B2  00:26:00–00:31:00  base      Early afternoon desk work",
        "S3  (empty)",
      ].join("\n"),
    );
  });

  it("gives each event in a slot its own line", () => {
    const catalog = makeFixtureCatalog();
    let comp = createComposition(catalog.baseDay, { genId: () => "c" });
    comp = insertEvent(comp, catalog, { slot: 0, candidateId: "hdepic:P01-x:10.000-70.000" }, () => "a1");
    comp = insertEvent(comp, catalog, { slot: 0, candidateId: "ego4d:uid-1:0.000-120.000" }, () => "a2");
    const lines = renderTimetable(comp, resolveTimeline(comp, catalog)).split("\n");
    expect(lines[1]).toBe("S0  00:00:00–00:01:00  cooking   evt:a1  Cooking onions in a pan");
    expect(lines[2]).toBe("S0  00:01:00–00:03:00  walking   evt:a2  Walking down a street");
    expect(lines[3]).toBe("B0  00:03:00–00:13:00  base      Morning at home");
  });
});
