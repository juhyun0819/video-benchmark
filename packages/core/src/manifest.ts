import { z } from "zod";
import { ClipCategorySchema, ClipSpecSchema, SettingSchema, type ClipCatalog } from "./clip.js";
import { TIME_OF_DAY } from "./clock.js";
import { TimelineError } from "./errors.js";
import { layoutTimeline } from "./layout.js";
import { LICENSES } from "./licenses.js";
import { remapAnnotations } from "./remap.js";
import { ScheduleSchema, DayTypeSchema, ExceptionSchema } from "./schedule/types.js";
import {
  DatasetSchema,
  ManifestCaptionSchema,
  SourceRefSchema,
  type AnnotationLevel,
  type ManifestCaption,
  type TimeSpan,
} from "./schema.js";
import { RelaxationSchema, type Timeline } from "./timeline.js";
import { verifyTimeline, RULE_IDS } from "./verify.js";
import { WeekdaySchema, WorldSchema } from "./world.js";

export const LEVEL_ORDER = ["L5", "L4", "L3", "L2", "L1", "raw", "summary", "keystep", "narration", "atomic"] as const;

const round3 = (x: number): number => Math.round(x * 1000) / 1000;

const roundTime = (t: TimeSpan): TimeSpan =>
  t.kind === "interval" ? { kind: "interval", start: round3(t.start), end: round3(t.end) } : { kind: "point", t: round3(t.t) };

const startOf = (t: TimeSpan): number => (t.kind === "interval" ? t.start : t.t);
const levelRank = (l: AnnotationLevel): number => LEVEL_ORDER.indexOf(l);

function compareCaptions(a: ManifestCaption, b: ManifestCaption): number {
  const ds = startOf(a.time) - startOf(b.time);
  if (ds !== 0) return ds;
  const dl = levelRank(a.level) - levelRank(b.level);
  if (dl !== 0) return dl;
  return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
}

const SpecWithUnknownSchema = ClipSpecSchema.omit({ time_of_day: true }).extend({
  time_of_day: z.union([z.enum(TIME_OF_DAY), z.literal("unknown")]),
});

export const ManifestClipSchema = z.object({
  source: SourceRefSchema,
  video_id: z.string(),
  source_start_s: z.number(),
  source_end_s: z.number(),
  synthetic_start_s: z.number(),
  synthetic_end_s: z.number(),
  slot_id: z.string(),
  category: ClipCategorySchema,
  node: z.string(),
  setting: SettingSchema,
  relaxations: z.array(RelaxationSchema),
  fingerprint: z.object({ duration_s: z.number(), phash: z.array(z.string()) }).nullable(),
  provenance: z.object({ spec: SpecWithUnknownSchema, extractor_version: z.string() }),
});
export type ManifestClip = z.infer<typeof ManifestClipSchema>;

export const ManifestDaySchema = z.object({
  day_index: z.number().int().min(1),
  weekday: WeekdaySchema,
  day_type: DayTypeSchema,
  attempt: z.number().int().min(0),
  wake: z.string(),
  sleep: z.string(),
  coverage_ratio: z.number(),
  exceptions: z.array(ExceptionSchema),
  dropped_slots: ScheduleSchema.shape.days.element.shape.dropped_slots,
  clips: z.array(ManifestClipSchema),
});
export type ManifestDay = z.infer<typeof ManifestDaySchema>;

export const TimelineManifestSchema = z.object({
  schema_version: z.literal(1),
  id: z.string(),
  revision: z.number().int().min(0),
  created_at: z.string(),
  versions: z.object({
    schema: z.literal(1),
    generator: z.string(),
    template: z.string(),
    gap_table: z.string(),
    exception_catalog: z.string(),
    extractor: z.array(z.string()),
  }),
  inputs: z.object({
    world: WorldSchema,
    dials: WorldSchema.innerType().shape.dials,
    seed: z.number().int(),
    node_binding: WorldSchema.innerType().shape.node_binding,
    distribution_source: z.string(),
    donors: ScheduleSchema.shape.donors,
  }),
  days: z.array(ManifestDaySchema),
  captions: z.array(ManifestCaptionSchema),
  verifier_summary: z.object({
    pass: z.boolean(),
    failed_rules: z.array(z.string()),
    flags: z.array(z.object({ kind: z.literal("swap_point"), day_index: z.number(), slot_ids: z.tuple([z.string(), z.string()]) })),
    seams: z.number().int(),
  }),
  licenses: z.array(z.object({ dataset: DatasetSchema, license: z.string(), note: z.string().optional() })),
  qa_tracks: z.object({}),
});
export type TimelineManifest = z.infer<typeof TimelineManifestSchema>;

const minutes = (hhmm: string): number => {
  const [h, m] = hhmm.split(":").map(Number);
  return h! * 60 + m!;
};

/** Public manifest of a verified timeline (§8): source spans, synthetic times, remapped captions, provenance. */
export function buildTimelineManifest(t: Timeline, catalog: ClipCatalog, now: Date): TimelineManifest {
  const missing = new Set<string>();
  for (const a of Object.values(t.slots)) for (const id of a.clip_ids) if (!catalog.getClip(id)) missing.add(id);
  if (missing.size > 0) {
    const ids = [...missing].sort();
    throw new TimelineError("MISSING_CLIPS", `clips missing from the catalog: ${ids.join(", ")}`, { ids });
  }

  const report = verifyTimeline(t, catalog);
  if (!report.pass) {
    const failed_rules = RULE_IDS.filter((r) => !report.rules[r].pass);
    throw new TimelineError("VERIFY_FAILED", `timeline fails rules: ${failed_rules.join(", ")}`, { failed_rules });
  }

  const placed = layoutTimeline(t, catalog);
  const captions: ManifestCaption[] = [];
  const extractors = new Set<string>();
  const datasets = new Set<string>();

  const days: ManifestDay[] = t.schedule.days.map((day) => {
    const clips: ManifestClip[] = [];
    let seconds = 0;
    for (const p of placed.filter((x) => x.day_index === day.day_index)) {
      const c = p.clip;
      extractors.add(c.extractor_version);
      datasets.add(c.source.dataset);
      seconds += c.source_end_s - c.source_start_s;
      clips.push({
        source: c.source,
        video_id: c.source.video_id,
        source_start_s: c.source_start_s,
        source_end_s: c.source_end_s,
        synthetic_start_s: round3(p.synthetic_start_s),
        synthetic_end_s: round3(p.synthetic_end_s),
        slot_id: p.slot_id,
        category: c.category,
        node: c.node,
        setting: c.setting,
        relaxations: t.slots[p.slot_id]?.relaxations ?? [],
        fingerprint: c.fingerprint,
        provenance: {
          spec: {
            category: c.category,
            setting: c.setting,
            node: c.node,
            activity: c.activity,
            time_of_day: c.time_of_day,
            tags: c.tags,
          },
          extractor_version: c.extractor_version,
        },
      });
      const remapped = remapAnnotations(
        {
          block_id: `clip:${p.slot_id}:${c.clip_id}`,
          source: c.source,
          src_start: c.source_start_s,
          src_end: c.source_end_s,
          composed_start: p.synthetic_start_s,
        },
        catalog.getAnnotations(c.source.dataset, c.source.video_id),
      );
      for (const cap of remapped) captions.push({ ...cap, time: roundTime(cap.time) });
    }
    const awake = (minutes(day.sleep) - minutes(day.wake)) * 60;
    return {
      day_index: day.day_index,
      weekday: day.weekday,
      day_type: day.day_type,
      attempt: day.attempt,
      wake: day.wake,
      sleep: day.sleep,
      coverage_ratio: awake > 0 ? round3(seconds / awake) : 0,
      exceptions: day.exceptions,
      dropped_slots: day.dropped_slots,
      clips,
    };
  });

  captions.sort(compareCaptions);
  const s = t.schedule;
  return {
    schema_version: 1,
    id: t.id,
    revision: t.revision,
    created_at: now.toISOString(),
    versions: {
      schema: 1,
      generator: s.generator_version,
      template: s.template_version,
      gap_table: s.gap_table_version,
      exception_catalog: s.exception_catalog_version,
      extractor: [...extractors].sort(),
    },
    inputs: {
      world: t.world,
      dials: t.world.dials,
      seed: t.world.seed,
      node_binding: t.world.node_binding,
      distribution_source: s.distribution_source,
      donors: s.donors,
    },
    days,
    captions,
    verifier_summary: {
      pass: true,
      failed_rules: [],
      flags: report.flags,
      seams: report.seams.length,
    },
    licenses: DatasetSchema.options.filter((d) => datasets.has(d)).map((dataset) => ({ dataset, ...LICENSES[dataset] })),
    qa_tracks: {},
  };
}
