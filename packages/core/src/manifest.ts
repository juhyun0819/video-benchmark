import { z } from "zod";
import { ClipCategorySchema, ClipSpecSchema, SettingSchema, type ClipCatalog } from "./clip.js";
import { TIME_OF_DAY } from "./clock.js";
import { TimelineError } from "./errors.js";
import { layoutTimeline } from "./layout.js";
import { LICENSES } from "./licenses.js";
import { remapAnnotations } from "./remap.js";
import { realizedOf } from "./schedule/realized.js";
import {
  ScheduleSchema,
  DayTypeSchema,
  ExceptionSchema,
  RealizedSchema,
  SlotSchema,
  type ScheduleDay,
  type Templates,
} from "./schedule/types.js";
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

/**
 * Time on a day that no clip covers (E9), in absolute seconds since day 1 00:00.
 * "transition": travel between two slots at different places, right before the later
 * slot; "unscheduled": time between windows (or before the first / after the last)
 * that no slot claims, at the previous slot's place (G23);
 * "unrecorded": time inside a slot's window not covered by its clips.
 */
export const NoVideoSpanSchema = z.object({
  start: z.number(),
  end: z.number(),
  kind: z.enum(["transition", "unrecorded", "unscheduled"]),
});
export type NoVideoSpan = z.infer<typeof NoVideoSpanSchema>;

export const ManifestDaySchema = z.object({
  day_index: z.number().int().min(1),
  weekday: WeekdaySchema,
  day_type: DayTypeSchema,
  attempt: z.number().int().min(0),
  wake: z.string(),
  sleep: z.string(),
  coverage_ratio: z.number(),
  exceptions: z.array(ExceptionSchema),
  /** As in schedule.json (E1); with `slots` they let realized be recounted from the manifest (G24). */
  slot_exception_eligible: z.boolean(),
  slot_exception_candidates: z.number().int().min(0),
  /** The day's scheduled slots, including ones left empty at matching time. */
  slots: z.array(SlotSchema.pick({ slot_id: true, role: true, required: true, window: true })),
  dropped_slots: ScheduleSchema.shape.days.element.shape.dropped_slots,
  /** Roles of slots left empty at matching time (relaxation "dropped_optional", D10, R27, R48). */
  dropped_optional: z.array(z.string()),
  /** Clamps applied after an anchor's jitter redraws ran out (E6). */
  anchor_adjustments: ScheduleSchema.shape.days.element.shape.anchor_adjustments,
  jitter_redraws: z.number().int().min(0),
  /** Roles cut to their min_minutes whose slot survived (E4). */
  shrunk: z.array(z.string()),
  clips: z.array(ManifestClipSchema),
  /** Together with the clips, tiles [wake, sleep] exactly; sorted by start (E9). */
  no_video_spans: z.array(NoVideoSpanSchema),
});
export type ManifestDay = z.infer<typeof ManifestDaySchema>;

const DonorAdjustmentSchema = ScheduleSchema.shape.donors.shape.weekday.pick({
  case_id: true,
  backoff_level: true,
  redraws: true,
  adjustments: true,
  filled: true,
  sport_fill: true,
});

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
  /** Dials realized by the final days (G17); schedule.realized is pre-matching. */
  realized: RealizedSchema.extend({
    /** Slots left empty at matching time (relaxation "dropped_optional"). */
    matching_dropped_slots: z.number().int().min(0),
    matching_dropped_by_role: z.record(z.string(), z.number().int().min(1)),
  }),
  donor_adjustments: z.object({ weekday: DonorAdjustmentSchema, weekend: DonorAdjustmentSchema }),
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

/**
 * E9: the parts of [wake, sleep] that no clip covers. `clipSpans` are the day's placed
 * clips per slot id as [start, end] in absolute seconds (already rounded).
 */
function noVideoSpans(day: ScheduleDay, clipSpans: ReadonlyMap<string, [number, number][]>): NoVideoSpan[] {
  const base = (day.day_index - 1) * 86400;
  const out: NoVideoSpan[] = [];
  const push = (start: number, end: number, kind: NoVideoSpan["kind"]) => {
    const a = round3(start);
    const b = round3(end);
    if (b > a) out.push({ start: a, end: b, kind });
  };
  const wake = base + minutes(day.wake) * 60;
  const sleep = base + minutes(day.sleep) * 60;
  const slots = [...day.slots].sort((a, b) => minutes(a.window[0]) - minutes(b.window[0]));
  if (slots.length === 0) {
    push(wake, sleep, "unscheduled");
    return out;
  }
  push(wake, base + minutes(slots[0]!.window[0]) * 60, "unscheduled");
  slots.forEach((slot, i) => {
    const ws = base + minutes(slot.window[0]) * 60;
    const we = base + minutes(slot.window[1]) * 60;
    let cursor = ws;
    for (const [a, b] of [...(clipSpans.get(slot.slot_id) ?? [])].sort((x, y) => x[0] - y[0])) {
      if (a > cursor) push(cursor, Math.min(a, we), "unrecorded");
      cursor = Math.max(cursor, b);
    }
    if (cursor < we) push(cursor, we, "unrecorded");
    const next = slots[i + 1];
    const gapEnd = next === undefined ? sleep : base + minutes(next.window[0]) * 60;
    if (gapEnd > we) {
      const moved = next === undefined
        ? 0
        : (day.transitions.find((x) => x.from_slot === slot.slot_id && x.to_slot === next.slot_id)?.minutes ?? 0) * 60;
      // G23 (CR6.3): unscheduled time is spent at this slot's place; the move comes right before the next slot.
      const t = Math.min(moved, gapEnd - we);
      push(we, gapEnd - t, "unscheduled");
      push(gapEnd - t, gapEnd, "transition");
    }
  });
  return out;
}

/**
 * Public manifest of a verified timeline (§8): source spans, synthetic times, remapped captions, provenance.
 * `realized` is recomputed from the final days (G17): schedule.realized is the pre-matching value and goes
 * stale when matching regenerates a day.
 */
export function buildTimelineManifest(t: Timeline, catalog: ClipCatalog, now: Date, templates: Templates): TimelineManifest {
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
    const clipSpans = new Map<string, [number, number][]>();
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
      clipSpans.set(p.slot_id, [...(clipSpans.get(p.slot_id) ?? []), [round3(p.synthetic_start_s), round3(p.synthetic_end_s)]]);
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
      slot_exception_eligible: day.slot_exception_eligible,
      slot_exception_candidates: day.slot_exception_candidates,
      slots: day.slots.map(({ slot_id, role, required, window }) => ({ slot_id, role, required, window })),
      dropped_slots: day.dropped_slots,
      dropped_optional: day.slots
        .filter((s) => t.slots[s.slot_id]?.relaxations.includes("dropped_optional"))
        .map((s) => s.role),
      anchor_adjustments: day.anchor_adjustments,
      jitter_redraws: day.jitter_redraws,
      shrunk: day.dropped_slots
        .filter((x) => x.reason === "shrunk" && day.slots.some((s) => s.role === x.role))
        .map((x) => x.role),
      clips,
      no_video_spans: noVideoSpans(day, clipSpans),
    };
  });

  captions.sort(compareCaptions);
  const s = t.schedule;
  const matchingDropped: Record<string, number> = {};
  for (const d of t.schedule.days) {
    for (const slot of d.slots) {
      if (t.slots[slot.slot_id]?.relaxations.includes("dropped_optional")) {
        matchingDropped[slot.role] = (matchingDropped[slot.role] ?? 0) + 1;
      }
    }
  }
  const pickOf = (p: typeof s.donors.weekday) => ({
    case_id: p.case_id,
    backoff_level: p.backoff_level,
    redraws: p.redraws,
    adjustments: p.adjustments,
    filled: p.filled,
    ...(p.sport_fill !== undefined ? { sport_fill: p.sport_fill } : {}),
  });
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
    realized: {
      ...realizedOf(s.days, t.world, templates),
      matching_dropped_slots: Object.values(matchingDropped).reduce((a, b) => a + b, 0),
      matching_dropped_by_role: matchingDropped,
    },
    donor_adjustments: { weekday: pickOf(s.donors.weekday), weekend: pickOf(s.donors.weekend) },
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
