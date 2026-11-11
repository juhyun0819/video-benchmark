import { z } from "zod";
import { ClipCategorySchema, ClipSpecSchema } from "../clip.js";
import type { DonorPick } from "../donor.js";
import { MealPlaceSchema, WeekdaySchema, type MealPlace, type Weekday } from "../world.js";

export const MealSchema = z.enum(["breakfast", "lunch", "dinner"]);
export type Meal = z.infer<typeof MealSchema>;

export const DayTypeSchema = z.enum(["workday", "hobby_day", "home_day", "leave_day"]);
export type DayType = z.infer<typeof DayTypeSchema>;

/** Day types that have their own template; leave_day uses home_day. */
export const TemplateNameSchema = z.enum(["workday", "hobby_day", "home_day"]);
export type TemplateName = z.infer<typeof TemplateNameSchema>;

export const ExceptionSchema = z.object({
  scope: z.enum(["slot", "day", "multi_day"]),
  type: z.string(),
  /** Slot role for slot exceptions, replaced day type for day exceptions. */
  replaces: z.string(),
  /** 1-based inclusive day indexes. */
  day_range: z.tuple([z.number().int().min(1), z.number().int().min(1)]),
});
export type Exception = z.infer<typeof ExceptionSchema>;

export const ExceptionCatalogSchema = z.object({
  version: z.string(),
  day: z.array(
    z.object({
      type: z.literal("leave_day"),
      from_day_type: z.literal("workday"),
      template: z.literal("home_day"),
    }),
  ),
  slot: z
    .array(
      z.object({
        type: z.string(),
        replaces_role: MealSchema,
        meal_place: MealPlaceSchema,
        day_types: z.array(DayTypeSchema).min(1),
      }),
    )
    .refine((entries) => new Set(entries.map((e) => e.type)).size === entries.length, {
      message: "slot exception types must be unique",
    }),
});
export type ExceptionCatalog = z.infer<typeof ExceptionCatalogSchema>;

export type DayDecisions = {
  day_index: number;
  weekday: Weekday;
  day_type: DayType;
  template: TemplateName;
  meal_place: Partial<Record<Meal, MealPlace | "skip">>;
  /** Template optional keys ("hobby", "home_evening"); workdays only. */
  optional: Record<string, boolean>;
  hobby_node?: string;
  exceptions: Exception[];
};

const ClockSchema = z.string().regex(/^\d{2}:\d{2}$/, "expected HH:MM");

/** The subset of schedule_defaults_*.json the generator reads; other keys pass through (R5). */
export const ScheduleDefaultsSchema = z
  .object({
    day_level_sigma: z
      .object({ routine_strength_1: z.number().min(0), routine_strength_0: z.number().min(0) })
      .passthrough(),
    weekend: z.object({ sports_start: z.object({ p50: ClockSchema }).passthrough() }).passthrough(),
  })
  .passthrough();
export type ScheduleDefaults = z.infer<typeof ScheduleDefaultsSchema>;

export const AnchorNameSchema = z.enum(["wake", "work_start", "work_end", "breakfast", "lunch", "dinner", "sport"]);
export type AnchorName = z.infer<typeof AnchorNameSchema>;

/** Jittered, sanitized anchor times for one day, in integer minutes since 00:00. */
export type DayAnchors = { wake: number; sleep: number } & Partial<Record<AnchorName, number>>;

export const TemplateSlotSchema = z
  .object({
    role: z.string(),
    category: ClipCategorySchema,
    /** World node name, or "$meal" / "$hobby" resolved per day. */
    node: z.string(),
    activity: z.string(),
    tags: z.array(z.string()).default([]),
    required: z.boolean().default(false),
    anchor: AnchorNameSchema.optional(),
    end_anchor: AnchorNameSchema.optional(),
    duration_minutes: z.number().int().positive().optional(),
    min_minutes: z.number().int().positive().default(10),
    meal: MealSchema.optional(),
    optional_key: z.string().optional(),
  })
  .refine((s) => s.node !== "$meal" || s.meal !== undefined, { message: "$meal slots need `meal`", path: ["meal"] });
export type TemplateSlot = z.infer<typeof TemplateSlotSchema>;

export const TemplatesSchema = z.object({
  version: z.string(),
  day_types: z.object({
    workday: z.array(TemplateSlotSchema).min(1),
    hobby_day: z.array(TemplateSlotSchema).min(1),
    home_day: z.array(TemplateSlotSchema).min(1),
  }),
});
export type Templates = z.infer<typeof TemplatesSchema>;

export const SlotSchema = z.object({
  slot_id: z.string().regex(/^d\d+-s\d+$/),
  role: z.string(),
  required: z.boolean(),
  window: z.tuple([ClockSchema, ClockSchema]),
  spec: ClipSpecSchema,
});
export type Slot = z.infer<typeof SlotSchema>;

/** A place change between two consecutive slots; `minutes` is the sampled transition gap. */
export const SlotTransitionSchema = z.object({
  from_slot: z.string(),
  to_slot: z.string(),
  minutes: z.number().int().min(0),
});
export type SlotTransition = z.infer<typeof SlotTransitionSchema>;

export const ScheduleDaySchema = z.object({
  day_index: z.number().int().min(1),
  weekday: WeekdaySchema,
  day_type: DayTypeSchema,
  attempt: z.number().int().min(0),
  wake: ClockSchema,
  sleep: ClockSchema,
  exceptions: z.array(ExceptionSchema),
  slots: z.array(SlotSchema),
  transitions: z.array(SlotTransitionSchema),
});
export type ScheduleDay = z.infer<typeof ScheduleDaySchema>;

const DonorPickSchema = z.object({
  case_id: z.string(),
  backoff_level: z.enum(["occupation_sex_age", "occupation_sex", "occupation", "sex_age", "all"]),
  filled: z.array(z.string()),
  profile: z.object({
    wake: z.number(),
    sleep: z.number(),
    work_start: z.number().optional(),
    work_end: z.number().optional(),
    breakfast: z.number(),
    lunch: z.number(),
    dinner: z.number(),
    sport: z.number().optional(),
    meal_loc: z.object({
      breakfast: MealPlaceSchema.optional(),
      lunch: MealPlaceSchema.optional(),
      dinner: MealPlaceSchema.optional(),
    }),
  }),
});
// Compile-time check that the schema matches donor.ts.
const _donorPickMatches: z.infer<typeof DonorPickSchema> extends DonorPick
  ? DonorPick extends z.infer<typeof DonorPickSchema>
    ? true
    : false
  : false = true;
void _donorPickMatches;

const MealDistributionSchema = z.record(MealPlaceSchema, z.number());

export const ScheduleSchema = z.object({
  person_id: z.string(),
  generator_version: z.literal("schedule.v1"),
  template_version: z.string(),
  gap_table_version: z.string(),
  gap_table_reviewed: z.boolean(),
  exception_catalog_version: z.string(),
  distribution_source: z.string(),
  donors: z.object({
    weekday: DonorPickSchema,
    weekend: DonorPickSchema,
    country_fallback: z.boolean(),
    weekend_redraws: z.number().int().min(0),
  }),
  meal_style_effective: z.object({
    workday: z.object({ lunch: MealDistributionSchema, dinner: MealDistributionSchema }),
    weekend: z.object({
      breakfast: MealDistributionSchema,
      lunch: MealDistributionSchema,
      dinner: MealDistributionSchema,
    }),
  }),
  days: z.array(ScheduleDaySchema),
});
export type Schedule = z.infer<typeof ScheduleSchema>;
