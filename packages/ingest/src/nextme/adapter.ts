import { parse } from "csv-parse/sync";
import { z } from "zod";
import {
  AnnotationSchema,
  BaseDaySchema,
  tileRecording,
  type Annotation,
  type AnnotationLevel,
  type BaseDay,
  type BlockCandidate,
  type SourceRef,
} from "@vb/core";
import { bump, readJsonl, type IngestReport } from "../jsonl.js";
import { parseHkt } from "../time.js";

/** Column names in raw/recordings.csv (spec section 11, NextMe-800 findings). */
export const RECORDINGS_COLUMNS = {
  recording_id: "recording_id",
  day: "date",
  tar: "tar_path",
} as const;

export type RecordingInfo = { recording_id: string; day: string; tar: string };

export function parseRecordingsCsv(text: string): RecordingInfo[] {
  const required = Object.values(RECORDINGS_COLUMNS);
  const rows = parse(text, {
    columns: (header: string[]) => {
      for (const col of required) {
        if (!header.includes(col)) throw new Error(`recordings.csv: missing column "${col}"`);
      }
      return header;
    },
    skip_empty_lines: true,
    bom: true,
  }) as Record<string, string>[];
  return rows.map((r) => ({
    recording_id: r[RECORDINGS_COLUMNS.recording_id] ?? "",
    day: r[RECORDINGS_COLUMNS.day] ?? "",
    tar: r[RECORDINGS_COLUMNS.tar] ?? "",
  }));
}

const RawRow = z
  .object({
    recording_id: z.string(),
    seg_id: z.string(),
    start: z.string(),
    end: z.string(),
    approx_time: z.boolean(),
    // Speech-only events (about 1% of rows in the real data) have action: null; text falls back
    // to action_brief, then speech.
    action: z.string().nullable(),
    action_brief: z.string().nullable().optional(),
    speech: z.string().nullable(),
  })
  .passthrough();

const LevelRow = z
  .object({
    recording_id: z.string(),
    id: z.string(),
    start: z.string(),
    end: z.string(),
    text: z.string(),
  })
  .passthrough();

const LEVELS = ["L1", "L2", "L3", "L4", "L5"] as const;

type Row = {
  level: AnnotationLevel;
  id: string;
  text: string;
  start: string;
  end: string;
  startEpoch: number;
  endEpoch: number;
  original: Record<string, unknown>;
};

function readRows(recId: string, files: (recId: string, name: string) => string, report: IngestReport): Row[] {
  const rows: Row[] = [];
  const accept = (
    level: AnnotationLevel,
    id: string,
    text: string,
    start: string,
    end: string,
    original: Record<string, unknown>,
  ) => {
    let startEpoch: number;
    let endEpoch: number;
    try {
      startEpoch = parseHkt(start);
      endEpoch = parseHkt(end);
    } catch {
      bump(report.skipped, "invalid_time");
      return;
    }
    if (endEpoch < startEpoch) {
      bump(report.skipped, "end_before_start");
      return;
    }
    rows.push({ level, id, text, start, end, startEpoch, endEpoch, original });
  };

  for (const r of readJsonl(files(recId, "raw_caption"), `${recId}/raw_caption.jsonl`, RawRow)) {
    accept("raw", r.seg_id, r.action ?? r.action_brief ?? r.speech ?? "", r.start, r.end, r);
  }
  for (const level of LEVELS) {
    for (const r of readJsonl(files(recId, level), `${recId}/${level}.jsonl`, LevelRow)) {
      accept(level, r.id, r.text, r.start, r.end, r);
    }
  }
  return rows;
}

type BuiltRecording = {
  recId: string;
  startEpoch: number;
  blocks: BlockCandidate[];
  annotations: Annotation[];
};

function buildRecording(
  rec: RecordingInfo,
  files: (recId: string, name: string) => string,
  report: IngestReport,
): BuiltRecording | undefined {
  const rows = readRows(rec.recording_id, files, report);
  if (rows.length === 0) {
    bump(report.skipped, "no_valid_captions");
    return undefined;
  }

  // Recording span = earliest start / latest end over raw_caption + L1..L5 (spec 5.1).
  let first = rows[0] as Row;
  let endEpoch = first.endEpoch;
  for (const r of rows) {
    if (r.startEpoch < first.startEpoch) first = r;
    if (r.endEpoch > endEpoch) endEpoch = r.endEpoch;
  }
  const startEpoch = first.startEpoch;
  const duration = endEpoch - startEpoch;

  const annotations = rows.map((r) =>
    AnnotationSchema.parse({
      id: `nextme800:${rec.recording_id}:${r.level}:${r.id}`,
      track: "captions",
      level: r.level,
      time: { kind: "interval", start: r.startEpoch - startEpoch, end: r.endEpoch - startEpoch },
      text: r.text,
      original: r.original,
    }),
  );

  const source: SourceRef = {
    dataset: "nextme800",
    video_id: rec.recording_id,
    media: { kind: "frames", approx_fps: 1, tar: rec.tar, start_wallclock: first.start },
  };
  const blocks = tileRecording({
    source,
    duration,
    l4: annotations.filter((a) => a.level === "L4"),
  });
  return { recId: rec.recording_id, startEpoch, blocks, annotations };
}

/**
 * Build the NextMe-800 base day: tile every recording of `day` at its L4
 * boundaries and normalize all caption levels to recording-relative seconds.
 * `files(recId, name)` returns the text of `<name>.jsonl` in
 * captions/1_nextme/<recId>/ (name is "raw_caption" or "L1".."L5").
 */
export function buildNextmeDay(
  day: string,
  recordings: RecordingInfo[],
  files: (recId: string, name: string) => string,
  report: IngestReport,
): { baseDay: BaseDay; annotations: Map<string, Annotation[]> } {
  const todays = recordings.filter((r) => r.day === day);
  if (todays.length === 0) {
    throw new Error(`NextMe-800: recordings.csv has no recordings for day ${day}`);
  }

  const built: BuiltRecording[] = [];
  for (const rec of todays) {
    const b = buildRecording(rec, files, report);
    if (b) built.push(b);
  }
  built.sort((a, b) => a.startEpoch - b.startEpoch || (a.recId < b.recId ? -1 : a.recId > b.recId ? 1 : 0));

  const blocks = built.flatMap((b) => b.blocks);
  const annotations = new Map<string, Annotation[]>();
  for (const b of built) annotations.set(b.recId, b.annotations);

  bump(report.ingested, "nextme800:recordings", built.length);
  bump(report.ingested, "nextme800:blocks", blocks.length);
  bump(
    report.ingested,
    "nextme800:annotations",
    built.reduce((n, b) => n + b.annotations.length, 0),
  );

  return { baseDay: BaseDaySchema.parse({ day, blocks }), annotations };
}
