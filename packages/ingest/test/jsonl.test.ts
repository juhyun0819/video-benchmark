import { describe, expect, it } from "vitest";
import { z } from "zod";
import { IngestFormatError, readJsonl } from "../src/jsonl.js";

const Row = z.object({ id: z.string(), start: z.string() });

describe("readJsonl", () => {
  it("parses rows and skips blank lines", () => {
    const text = '{"id":"a","start":"s"}\n\n{"id":"b","start":"t"}\n';
    expect(readJsonl(text, "L4.jsonl", Row)).toEqual([
      { id: "a", start: "s" },
      { id: "b", start: "t" },
    ]);
    expect(readJsonl("", "L4.jsonl", Row)).toEqual([]);
  });

  it("reports file and line on schema mismatch", () => {
    const text = '{"id":"a","start":"s"}\n{"id":"b"}\n';
    expect(() => readJsonl(text, "L4.jsonl", Row)).toThrow(/L4\.jsonl:2/);
    try {
      readJsonl(text, "L4.jsonl", Row);
    } catch (e) {
      expect(e).toBeInstanceOf(IngestFormatError);
      const err = e as IngestFormatError;
      expect(err.file).toBe("L4.jsonl");
      expect(err.line).toBe(2);
      expect(err.issue).toContain("start");
    }
  });

  it("reports invalid JSON with its line", () => {
    expect(() => readJsonl('{"id":"a","start":"s"}\n{oops\n', "raw_caption.jsonl", Row)).toThrow(
      /raw_caption\.jsonl:2/,
    );
  });
});
