import { z } from "zod";
import { TIME_OF_DAY } from "./clock.js";
import {
  AnnotationLevelSchema,
  SourceRefSchema,
  TimeSpanSchema,
  type Annotation,
  type Dataset,
} from "./schema.js";

export const ClipCategorySchema = z.enum(["home", "work", "meal", "hobby", "transit"]);
export type ClipCategory = z.infer<typeof ClipCategorySchema>;

export const SettingSchema = z.enum(["home_indoor", "venue_indoor", "outdoor"]);
export type Setting = z.infer<typeof SettingSchema>;

const TimeOfDaySchema = z.enum(TIME_OF_DAY);

/** Slot spec: the content contract shared by schedule slots and catalog clips (spec 4.1). */
export const ClipSpecSchema = z.object({
  category: ClipCategorySchema,
  setting: SettingSchema,
  node: z.string(),
  activity: z.string(),
  time_of_day: TimeOfDaySchema,
  tags: z.array(z.string()).default([]),
});
export type ClipSpec = z.infer<typeof ClipSpecSchema>;

export const ClipSchema = ClipSpecSchema.omit({ time_of_day: true })
  .extend({
    time_of_day: z.union([TimeOfDaySchema, z.literal("unknown")]),
    clip_id: z.string(),
    source: SourceRefSchema,
    source_start_s: z.number(),
    source_end_s: z.number(),
    day_group: z.string(),
    person_source: z.string(),
    fingerprint: z.object({ duration_s: z.number(), phash: z.array(z.string()) }).nullable(),
    extractor_version: z.string(),
  })
  .refine((c) => c.source_end_s > c.source_start_s, {
    message: "source_end_s must be greater than source_start_s",
    path: ["source_end_s"],
  });
export type Clip = z.infer<typeof ClipSchema>;

/** One line of the `ingest bank` output. Caption times are source seconds. */
export const ClipContractRowSchema = ClipSchema.and(
  z.object({
    captions: z.array(
      z.object({
        id: z.string(),
        level: AnnotationLevelSchema,
        time: TimeSpanSchema,
        text: z.string(),
      }),
    ),
  }),
);
export type ClipContractRow = z.infer<typeof ClipContractRowSchema>;

export interface ClipCatalog {
  clips(): readonly Clip[];
  getClip(id: string): Clip | undefined;
  getAnnotations(dataset: Dataset, videoId: string): Annotation[];
}
