import type {
  Annotation,
  AnnotationLevel,
  BaseDay,
  BlockCandidate,
  Catalog,
  Dataset,
  SourceRef,
  TimeSpan,
} from "../src/index.js";

// Synthetic data only. Never copy real dataset content into fixtures.

const frames = (video_id: string): SourceRef => ({
  dataset: "nextme800",
  video_id,
  media: { kind: "frames", approx_fps: 1, tar: `frames/${video_id}.tar` },
});

const video = (dataset: Dataset, video_id: string, fps: number): SourceRef => ({
  dataset,
  video_id,
  media: { kind: "video", fps },
});

const candidateId = (s: SourceRef, start: number, end: number) =>
  `${s.dataset}:${s.video_id}:${start.toFixed(3)}-${end.toFixed(3)}`;

function baseBlock(video_id: string, start: number, end: number, summary: string): BlockCandidate {
  const source = frames(video_id);
  return {
    id: candidateId(source, start, end),
    role: "base",
    category: "base",
    source,
    src_start: start,
    src_end: end,
    summary,
  };
}

function event(
  source: SourceRef,
  category: string,
  start: number,
  end: number,
  summary: string,
): BlockCandidate {
  return {
    id: candidateId(source, start, end),
    role: "event",
    category,
    source,
    src_start: start,
    src_end: end,
    summary,
  };
}

// NextMe-style wall-clock annotation: rec-A is assumed to start at 09:00:00 (HKT).
const REC_A_START = "2026-04-15 09:00:00";
const wall = (s: string) => s; // kept as a string in `original`, as in the raw CSV

function nextme(
  id: string,
  level: AnnotationLevel,
  start: number,
  end: number,
  startStr: string,
  endStr: string,
  text: string,
  recording: string,
): Annotation {
  const time: TimeSpan = { kind: "interval", start, end };
  return {
    id,
    track: "captions",
    level,
    time,
    text,
    original: { recording_id: recording, level, start: wall(startStr), end: wall(endStr), text },
  };
}

const annotationsByVideo: Record<string, Annotation[]> = {
  "nextme800:rec-A": [
    nextme("nm:rec-A:raw:0", "raw", 30, 40, "2026-04-15 09:00:30", "2026-04-15 09:00:40", "picks up a mug", "rec-A"),
    nextme("nm:rec-A:raw:1", "raw", 590, 610, "2026-04-15 09:09:50", "2026-04-15 09:10:10", "opens the door", "rec-A"),
    nextme("nm:rec-A:L1:0", "L1", 100, 160, "2026-04-15 09:01:40", "2026-04-15 09:02:40", "makes coffee at the counter", "rec-A"),
    nextme("nm:rec-A:L4:0", "L4", 0, 600, REC_A_START, "2026-04-15 09:10:00", "morning routine at home", "rec-A"),
    nextme("nm:rec-A:L4:1", "L4", 600, 1500, "2026-04-15 09:10:00", "2026-04-15 09:25:00", "commute and arrival", "rec-A"),
    // L5 interval crossing the rec-A 600 s block boundary.
    nextme("nm:rec-A:L5:0", "L5", 540, 780, "2026-04-15 09:09:00", "2026-04-15 09:13:00", "leaves home and starts walking to the station", "rec-A"),
  ],
  "nextme800:rec-B": [
    nextme("nm:rec-B:raw:0", "raw", 10, 20, "2026-04-15 13:00:10", "2026-04-15 13:00:20", "sits at the desk", "rec-B"),
    nextme("nm:rec-B:L1:0", "L1", 60, 200, "2026-04-15 13:01:00", "2026-04-15 13:03:20", "reads documents", "rec-B"),
    nextme("nm:rec-B:L5:0", "L5", 0, 300, "2026-04-15 13:00:00", "2026-04-15 13:05:00", "early afternoon desk work", "rec-B"),
  ],
  "hdepic:P01-x": [
    // Straddles the start of the candidate window [10, 70].
    ann("hd:P01-x:0", "narration", { kind: "interval", start: 5, end: 15 }, "take the pan from the shelf", { narration_id: "P01-x_0", start_timestamp: 5, end_timestamp: 15 }),
    ann("hd:P01-x:1", "narration", { kind: "interval", start: 20, end: 28 }, "chop the onion", { narration_id: "P01-x_1", start_timestamp: 20, end_timestamp: 28 }),
    ann("hd:P01-x:2", "narration", { kind: "interval", start: 40, end: 55 }, "fry the onion in the pan", { narration_id: "P01-x_2", start_timestamp: 40, end_timestamp: 55 }),
    ann("hd:P01-x:3", "narration", { kind: "interval", start: 60, end: 70 }, "plate the food", { narration_id: "P01-x_3", start_timestamp: 60, end_timestamp: 70 }),
  ],
  "ego4d:uid-1": [
    ann("ego:uid-1:0", "narration", { kind: "point", t: 5 }, "#C walks along the road", { narration_timestamp_sec: 5 }),
    ann("ego:uid-1:1", "narration", { kind: "point", t: 30 }, "#C looks at a shop window", { narration_timestamp_sec: 30 }),
    ann("ego:uid-1:2", "narration", { kind: "point", t: 90 }, "#C crosses the street", { narration_timestamp_sec: 90 }),
    ann("ego:uid-1:sum", "summary", { kind: "interval", start: 0, end: 120 }, "The camera wearer walks down a street.", { start_time: 0, end_time: 120 }),
  ],
};

function ann(
  id: string,
  level: AnnotationLevel,
  time: TimeSpan,
  text: string,
  original: Record<string, unknown>,
): Annotation {
  return { id, track: "captions", level, time, text, original: { ...original, text } };
}

export function makeFixtureCatalog(): Catalog & { baseDay: BaseDay } {
  const baseDay: BaseDay = {
    day: "2026-04-15",
    blocks: [
      baseBlock("rec-A", 0, 600, "Morning at home"),
      baseBlock("rec-A", 600, 1500, "Commute and arrival"),
      baseBlock("rec-B", 0, 300, "Early afternoon desk work"),
    ],
  };

  const events: BlockCandidate[] = [
    event(video("hdepic", "P01-x", 30), "cooking", 10, 70, "Cooking onions in a pan"),
    event(video("ego4d", "uid-1", 30), "walking", 0, 120, "Walking down a street"),
  ];

  const candidates = new Map<string, BlockCandidate>();
  for (const b of baseDay.blocks) candidates.set(b.id, b);
  for (const e of events) candidates.set(e.id, e);

  return {
    baseDay,
    getBaseDay: (day) => (day === baseDay.day ? baseDay : undefined),
    getCandidate: (id) => candidates.get(id),
    getAnnotations: (dataset, videoId) =>
      structuredClone(annotationsByVideo[`${dataset}:${videoId}`] ?? []),
  };
}
