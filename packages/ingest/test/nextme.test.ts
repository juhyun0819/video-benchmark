import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it, vi } from "vitest";
import { AnnotationSchema, BaseDaySchema } from "@vb/core";
import { buildNextmeDay, parseRecordingsCsv } from "../src/nextme/adapter.js";
import { makeHfFetcher } from "../src/nextme/hf.js";
import { newReport, type IngestReport } from "../src/jsonl.js";
import { parseHkt } from "../src/time.js";

const FIX = join(dirname(fileURLToPath(import.meta.url)), "fixtures", "nextme");
const REC_A = "2026-04-15_0900-1000";
const REC_B = "2026-04-15_1100-1105";
const DAY = "2026-04-15";

const csv = () => readFileSync(join(FIX, "recordings.csv"), "utf8");
const fixtureFiles = (recId: string, name: string): string =>
  readFileSync(join(FIX, recId, `${name}.jsonl`), "utf8");

function build(files = fixtureFiles, report: IngestReport = newReport()) {
  const out = buildNextmeDay(DAY, parseRecordingsCsv(csv()), files, report);
  return { ...out, report };
}

describe("parseRecordingsCsv", () => {
  it("reads id, day and tar_path columns", () => {
    const recs = parseRecordingsCsv(csv());
    expect(recs).toHaveLength(3);
    expect(recs[0]).toEqual({
      recording_id: REC_A,
      day: "2026-04-15",
      tar: "raw/2026-04-15/2026-04-15_0900-1000.tar",
    });
    expect(recs[2]?.day).toBe("2026-04-16");
  });

  it("fails clearly when a required column is missing", () => {
    expect(() => parseRecordingsCsv("recording_id,date\nx,2026-04-15\n")).toThrow(/tar_path/);
  });
});

describe("buildNextmeDay", () => {
  it("builds base day from fixture", () => {
    const { baseDay, annotations, report } = build();
    expect(BaseDaySchema.safeParse(baseDay).success).toBe(true);
    expect(baseDay.day).toBe(DAY);

    const a = baseDay.blocks.filter((b) => b.source.video_id === REC_A);
    expect(a.map((b) => [b.src_start, b.src_end])).toEqual([
      [0, 1200],
      [1200, 2460],
      [2460, 3600],
    ]);
    expect(a.map((b) => b.summary)).toEqual([
      "I prepare breakfast at home.",
      "I walk to the station.",
      "I ride the train.",
    ]);

    const b = baseDay.blocks.filter((x) => x.source.video_id === REC_B);
    expect(b).toHaveLength(1);
    expect(b[0]).toMatchObject({ src_start: 0, src_end: 300, summary: "(unlabeled)", role: "base" });

    // recordings in time order, 2026-04-16 recording excluded
    expect(baseDay.blocks.map((x) => x.source.video_id)).toEqual([REC_A, REC_A, REC_A, REC_B]);

    expect(a[0]?.source).toEqual({
      dataset: "nextme800",
      video_id: REC_A,
      media: {
        kind: "frames",
        approx_fps: 1,
        tar: "raw/2026-04-15/2026-04-15_0900-1000.tar",
        start_wallclock: "2026-04-15 09:00:00",
      },
    });
    expect(b[0]?.source.media).toMatchObject({ start_wallclock: "2026-04-15 11:00:00" });

    expect([...annotations.keys()].sort()).toEqual([REC_A, REC_B]);
    const annA = annotations.get(REC_A) ?? [];
    expect(annA).toHaveLength(4 + 4 + 2 + 1 + 3 + 1);
    for (const x of annA) expect(AnnotationSchema.safeParse(x).success).toBe(true);
    expect(new Set(annA.map((x) => x.id)).size).toBe(annA.length);

    expect(report.ingested["nextme800:recordings"]).toBe(2);
    expect(report.ingested["nextme800:blocks"]).toBe(4);
    expect(report.ingested["nextme800:annotations"]).toBe(annA.length + (annotations.get(REC_B)?.length ?? 0));
    expect(report.skipped).toEqual({});
  });

  it("converts caption times to recording-relative seconds", () => {
    const { annotations } = build();
    const annA = annotations.get(REC_A) ?? [];
    const raw2 = annA.find((x) => x.id === `nextme800:${REC_A}:raw:S000002`);
    expect(raw2).toMatchObject({
      level: "raw",
      track: "captions",
      text: "I pick up my bag.",
      time: { kind: "interval", start: 1200, end: 1204 },
    });
    // whole source row is preserved, incl. approx_time and the original time strings
    expect(raw2?.original).toMatchObject({
      seg_id: "S000002",
      approx_time: true,
      start: "2026-04-15 09:20:00",
      end: "2026-04-15 09:20:04",
      recording_id: REC_A,
    });

    const l4 = annA.find((x) => x.id === `nextme800:${REC_A}:L4:L4_00002`);
    expect(l4).toMatchObject({
      level: "L4",
      text: "I ride the train.",
      time: { kind: "interval", start: 2460, end: 3600 },
    });
    expect(l4?.original).toMatchObject({ id: "L4_00002", text: "I ride the train." });

    // zero-length rows survive as zero-length intervals
    const blink = annA.find((x) => x.id === `nextme800:${REC_A}:raw:S000001`);
    expect(blink?.time).toEqual({ kind: "interval", start: 3, end: 3 });
    expect(parseHkt("2026-04-15 09:00:03") - parseHkt("2026-04-15 09:00:00")).toBe(3);
  });

  it("derives recording span from the earliest start and latest end across all six files", () => {
    const files = (rec: string, name: string) => {
      const t = fixtureFiles(rec, name);
      if (rec === REC_A && name === "raw_caption") {
        // starts 10 s before everything else; ends 30 s after L4
        const early = JSON.stringify({
          recording_id: REC_A, seg_id: "S900000", start: "2026-04-15 08:59:50", end: "2026-04-15 08:59:55",
          approx_time: false, action: "I wake up.", action_brief: "Wake.", objects: [], environment: "x",
          text_visible: [], speech: null, details: "",
        });
        const late = JSON.stringify({
          recording_id: REC_A, seg_id: "S900001", start: "2026-04-15 10:00:00", end: "2026-04-15 10:00:30",
          approx_time: false, action: "I lie down.", action_brief: "Lie.", objects: [], environment: "x",
          text_visible: [], speech: null, details: "",
        });
        return `${early}\n${t}${late}\n`;
      }
      return t;
    };
    const { baseDay, annotations } = build(files);
    const blocks = baseDay.blocks.filter((b) => b.source.video_id === REC_A);
    expect(blocks[0]?.source.media).toMatchObject({ start_wallclock: "2026-04-15 08:59:50" });
    expect(blocks.at(-1)?.src_end).toBe(3640);
    const first = annotations.get(REC_A)?.find((x) => x.id.endsWith(":raw:S000000"));
    expect(first?.time).toEqual({ kind: "interval", start: 10, end: 13 });
  });

  it("skips record with unparseable time and counts reason", () => {
    const files = (rec: string, name: string) => {
      const t = fixtureFiles(rec, name);
      if (rec === REC_A && name === "L1") {
        const rows = t.trimEnd().split("\n");
        rows[1] = JSON.stringify({ ...JSON.parse(rows[1] ?? "{}"), start: "not a time" });
        rows[2] = JSON.stringify({
          ...JSON.parse(rows[2] ?? "{}"),
          start: "2026-04-15 09:30:10",
          end: "2026-04-15 09:30:05",
        });
        return rows.join("\n") + "\n";
      }
      return t;
    };
    const { annotations, report } = build(files);
    expect(report.skipped).toEqual({ invalid_time: 1, end_before_start: 1 });
    const ids = (annotations.get(REC_A) ?? []).map((x) => x.id);
    expect(ids).not.toContain(`nextme800:${REC_A}:L1:L1_00001`);
    expect(ids).not.toContain(`nextme800:${REC_A}:L1:L1_00002`);
    expect(ids).toContain(`nextme800:${REC_A}:L1:L1_00000`);
  });

  it("keeps caption extending past the L4 tiling verbatim (clipping happens at remap)", () => {
    const files = (rec: string, name: string) => {
      const t = fixtureFiles(rec, name);
      if (rec === REC_A && name === "L2") {
        const extra = JSON.stringify({
          recording_id: REC_A, id: "L2_00002", start: "2026-04-15 09:59:00", end: "2026-04-15 10:00:45",
          text: "I keep going.",
        });
        return t + extra + "\n";
      }
      return t;
    };
    const { annotations, baseDay } = build(files);
    const x = annotations.get(REC_A)?.find((a) => a.id === `nextme800:${REC_A}:L2:L2_00002`);
    expect(x?.time).toEqual({ kind: "interval", start: 3540, end: 3645 });
    expect(baseDay.blocks.filter((b) => b.source.video_id === REC_A).at(-1)?.src_end).toBe(3645);
  });

  it("uses action_brief as text when a raw speech event has action null", () => {
    const files = (rec: string, name: string) => {
      const t = fixtureFiles(rec, name);
      if (rec === REC_A && name === "raw_caption") {
        const speech = JSON.stringify({
          recording_id: REC_A, seg_id: "S000010", start: "2026-04-15 09:30:00", end: "2026-04-15 09:30:00",
          approx_time: false, action: null, action_brief: "Present in scene", objects: [], environment: "No change.",
          text_visible: [], speech: "hello there", details: "Speech event corresponding to the audio transcript.",
        });
        return t + speech + "\n";
      }
      return t;
    };
    const { annotations } = build(files);
    const x = annotations.get(REC_A)?.find((a) => a.id === `nextme800:${REC_A}:raw:S000010`);
    expect(x?.text).toBe("Present in scene");
    expect(x?.original).toMatchObject({ action: null, speech: "hello there" });
  });

  it("falls back to speech when action and action_brief are both null", () => {
    const files = (rec: string, name: string) => {
      const t = fixtureFiles(rec, name);
      if (rec === REC_A && name === "raw_caption") {
        const row = {
          recording_id: REC_A, seg_id: "S000011", start: "2026-04-15 09:31:00", end: "2026-04-15 09:31:00",
          approx_time: false, action: null, action_brief: null, objects: [], environment: "No change.",
          text_visible: [], speech: "spoken words", details: "Speech event.",
        };
        return t + JSON.stringify(row) + "\n";
      }
      return t;
    };
    const { annotations } = build(files);
    const x = annotations.get(REC_A)?.find((a) => a.id === `nextme800:${REC_A}:raw:S000011`);
    expect(x?.text).toBe("spoken words");
    expect(x?.original).toMatchObject({ action: null, action_brief: null, speech: "spoken words" });
  });

  it("fails clearly when day has no recordings", () => {
    expect(() => buildNextmeDay("2026-05-01", parseRecordingsCsv(csv()), fixtureFiles, newReport())).toThrow(
      /2026-05-01/,
    );
  });

  it("surfaces file and line for malformed caption rows", () => {
    const files = (rec: string, name: string) =>
      rec === REC_A && name === "L4" ? '{"recording_id":"x","id":"L4_00000","end":"2026-04-15 09:00:01","text":"t"}\n' : fixtureFiles(rec, name);
    expect(() => build(files)).toThrow(new RegExp(`${REC_A}/L4\\.jsonl:1`));
  });
});

describe("makeHfFetcher", () => {
  it("requests the resolve URL with a bearer token", async () => {
    const fetchImpl = vi.fn(async () => new Response("hello", { status: 200 }));
    const get = makeHfFetcher("tok", fetchImpl as unknown as typeof fetch);
    expect(await get("raw/recordings.csv")).toBe("hello");
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    const [url, init] = fetchImpl.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("https://huggingface.co/datasets/mmm8383/NextMe-800/resolve/main/raw/recordings.csv");
    expect((init.headers as Record<string, string>).Authorization).toBe("Bearer tok");
  });

  it("explains access denial on 401/403 without leaking the token", async () => {
    for (const status of [401, 403]) {
      const get = makeHfFetcher("secret-token", (async () => new Response("no", { status })) as typeof fetch);
      const err = await get("raw/recordings.csv").then(
        () => undefined,
        (e: Error) => e,
      );
      expect(err?.message).toBe("HF access denied: check NextMe-800 access approval and HF_TOKEN");
      expect(err?.message).not.toContain("secret-token");
    }
  });

  it("reports other HTTP failures with the path", async () => {
    const get = makeHfFetcher("t", (async () => new Response("x", { status: 404 })) as typeof fetch);
    await expect(get("captions/missing.jsonl")).rejects.toThrow(/404.*captions\/missing\.jsonl/);
  });
});
