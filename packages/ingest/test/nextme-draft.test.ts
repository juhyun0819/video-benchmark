import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { ClipContractRowSchema } from "@vb/core";
import { ingestBank } from "../src/bank.js";
import { newReport } from "../src/jsonl.js";
import { buildNextmeDay, parseRecordingsCsv } from "../src/nextme/adapter.js";
import { loadLabelRules, nextmeDraftRows, type LabelRules } from "../src/nextme/draft.js";
import { containsWord } from "../src/words.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
const RULES: LabelRules = loadLabelRules(join(ROOT, "config", "nextme_label_rules.json"));
const DAY = "2026-04-15";

const CSV = `recording_id,date,tar_path\nrec1,${DAY},raw/rec1.tar\n`;

/** One recording starting at `start` with the given L4 blocks (text per 10 minute block) and an L5 caption. */
function draft(start: string, l4: string[], l5: string) {
  const hhmm = (min: number) => {
    const t = Date.UTC(2026, 3, 15, Number(start.slice(11, 13)), Number(start.slice(14, 16)) + min);
    return new Date(t).toISOString().slice(0, 19).replace("T", " ");
  };
  const line = (o: object) => JSON.stringify({ recording_id: "rec1", ...o });
  const l4Rows = l4.map((text, i) => line({ id: `L4_${i}`, start: hhmm(i * 10), end: hhmm(i * 10 + 10), text }));
  const files = (_: string, name: string): string => {
    if (name === "L4") return l4Rows.join("\n");
    if (name === "L5") return line({ id: "L5_0", start: hhmm(0), end: hhmm(l4.length * 10), text: l5 });
    if (name === "raw_caption") return "";
    return line({ id: `${name}_0`, start: hhmm(0), end: hhmm(0), text: `${name} text` });
  };
  const recordings = parseRecordingsCsv(CSV);
  const built = buildNextmeDay(DAY, recordings, files, newReport());
  return { built, rows: nextmeDraftRows(DAY, built, RULES) };
}

describe("containsWord", () => {
  it("matches whole words case-insensitively", () => {
    expect(containsWord("the Car.", "car")).toBe(true);
    expect(containsWord("a card", "car")).toBe(false);
  });
});

describe("nextmeDraftRows", () => {
  it("labels a home meal block", () => {
    const { rows } = draft("2026-04-15 13:23:00", ["I conduct my meal routine at my apartment."], "I do things.");
    expect(rows[0]).toMatchObject({ category: "meal", node: "home", setting: "home_indoor" });
    expect(ClipContractRowSchema.safeParse(rows[0]).success).toBe(true);
  });

  it("labels a workplace block by its default category", () => {
    const { rows } = draft("2026-04-15 10:00:00", ["I study my notes at the library desk."], "I work.");
    expect(rows[0]).toMatchObject({ category: "work", node: "workplace", setting: "venue_indoor" });
  });

  it("uses the L5 text when the block has no place word, and nulls when nothing matches", () => {
    const viaL5 = draft("2026-04-15 10:00:00", ["I read a paper."], "I spend the day at the office.");
    expect(viaL5.rows[0]).toMatchObject({ category: "work", node: "workplace" });
    const none = draft("2026-04-15 10:00:00", ["I read a paper."], "I read.");
    expect(none.rows[0]).toMatchObject({ category: null, node: null, setting: null });
  });

  it("fills contract fields and captions with id suffixes", () => {
    const { rows } = draft("2026-04-15 09:00:00", ["I cook at home.", "I walk to the bus."], "A day.");
    const r = rows[0]!;
    expect(r).toMatchObject({
      day_group: DAY,
      person_source: "nextme800:wearer",
      fingerprint: null,
      tags: [],
      extractor_version: "nextme-draft.v1",
      source_start_s: 0,
      source_end_s: 600,
      activity: "I cook at home.",
    });
    const ids = r.captions.map((c) => c.id);
    expect(ids).toContain("L4:L4_0");
    expect(ids).toContain("L5:L5_0");
    expect(ids).not.toContain("L4:L4_1");
    expect(rows[1]).toMatchObject({ category: "transit", node: "transit" });
  });

  it("computes time_of_day from HKT wall clock, independent of TZ, advancing by src_start", () => {
    const { rows } = draft("2026-04-15 13:23:00", ["I cook at home."], "x");
    expect(rows.every((r) => r.day_group === DAY)).toBe(true);
    expect(rows[0]?.time_of_day).toBe("midday"); // 803 min is before the 14:00 afternoon boundary
    expect(draft("2026-04-15 14:23:00", ["I cook at home."], "x").rows[0]?.time_of_day).toBe("afternoon");
    const late = draft("2026-04-15 13:55:00", ["I cook at home.", "I cook at home."], "x");
    expect(late.rows.map((r) => r.time_of_day)).toEqual(["midday", "afternoon"]);
  });

  it("feeds ingestBank, which skips exactly the unlabeled rows", () => {
    const { rows } = draft("2026-04-15 09:00:00", ["I cook at home.", "I read a paper."], "I read.");
    const text = rows.map((r) => JSON.stringify(r)).join("\n");
    const out = ingestBank(text, "draft.jsonl", "nextme800");
    expect(rows.map((r) => r.category)).toEqual(["meal", null]);
    expect(out.clips).toHaveLength(1);
    expect(out.report.skipped.unlabeled).toBe(1);
  });
});
