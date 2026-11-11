import { z } from "zod";

export const DatasetSchema = z.enum(["nextme800", "hdepic", "ego4d", "egoexo4d", "youtube"]);
export type Dataset = z.infer<typeof DatasetSchema>;

export const SourceRefSchema = z.object({
  dataset: DatasetSchema,
  video_id: z.string(),
  media: z.discriminatedUnion("kind", [
    z.object({ kind: z.literal("frames"), approx_fps: z.literal(1), tar: z.string(), start_wallclock: z.string().optional() }),
    z.object({ kind: z.literal("video"), fps: z.number(), path: z.string().optional() }),
  ]),
});
export type SourceRef = z.infer<typeof SourceRefSchema>;

export const TimeSpanSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("interval"), start: z.number(), end: z.number() }),
  z.object({ kind: z.literal("point"), t: z.number() }),
]);
export type TimeSpan = z.infer<typeof TimeSpanSchema>;

export const AnnotationLevelSchema = z.enum([
  "raw",
  "L1",
  "L2",
  "L3",
  "L4",
  "L5",
  "narration",
  "summary",
  "keystep",
  "atomic",
]);
export type AnnotationLevel = z.infer<typeof AnnotationLevelSchema>;

export const AnnotationSchema = z.object({
  id: z.string(),
  track: z.literal("captions"),
  level: AnnotationLevelSchema,
  time: TimeSpanSchema,
  text: z.string(),
  original: z.record(z.string(), z.unknown()),
});
export type Annotation = z.infer<typeof AnnotationSchema>;

const ClippedSchema = z.object({
  start: z.literal(true).optional(),
  end: z.literal(true).optional(),
});

export const ManifestCaptionSchema = z.object({
  id: z.string(),
  block_id: z.string(),
  level: AnnotationLevelSchema,
  time: TimeSpanSchema,
  text: z.string(),
  clipped: ClippedSchema.optional(),
  provenance: z.object({
    dataset: DatasetSchema,
    video_id: z.string(),
    orig_id: z.string(),
    src_time: TimeSpanSchema,
    orig_time_string: z
      .object({ start: z.string().optional(), end: z.string().optional() })
      .optional(),
    original: z.record(z.string(), z.unknown()),
  }),
});
export type ManifestCaption = z.infer<typeof ManifestCaptionSchema>;
