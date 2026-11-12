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
  /** Where the replaced meal happens; set for slot exceptions (R39). */
  meal_place: MealPlaceSchema.optional(),
  /** Slot exceptions: the day's routine place for the replaced meal before the exception (equal to meal_place for a no-op). */
  routine_place: MealPlaceSchema.optional(),
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

/** Exactly what parseClock accepts: HH 00–27, MM 00–59 (R44). */
const ClockSchema = z.string().regex(/^(?:[01]\d|2[0-7]):[0-5]\d$/, "expected HH:MM with HH 00-27 and MM 00-59");

const Prob01 = z.number().min(0).max(1);

/** The subset of schedule_defaults_*.json the generator reads; other keys pass through (R5). */
export const ScheduleDefaultsSchema = z
  .object({
    day_level_sigma: z
      .object({ routine_strength_1: z.number().min(0), routine_strength_0: z.number().min(0) })
      .passthrough(),
    weekend: z.object({ sports_start: z.object({ p50: ClockSchema }).passthrough() }).passthrough(),
    weekday_evening_home: z
      .object({
        slot_probability: Prob01,
        minutes: z.object({ p25: z.number().min(0), p50: z.number().min(0), p75: z.number().min(0) }).passthrough(),
      })
      .passthrough(),
    meal_location_priors: z.object({ rest_day: z.object({}).passthrough() }).passthrough(),
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

/** One step of a day type's reduce order (E4): shrink sets the slot to its min_minutes, drop removes it. */
export const ReduceStepSchema = z.object({ role: z.string(), action: z.enum(["shrink", "drop"]) });
export type ReduceStep = z.infer<typeof ReduceStepSchema>;

export const TemplatesSchema = z
  .object({
    version: z.string(),
    day_types: z.object({
      workday: z.array(TemplateSlotSchema).min(1),
      hobby_day: z.array(TemplateSlotSchema).min(1),
      home_day: z.array(TemplateSlotSchema).min(1),
    }),
    /** Per day type; a day type without one keeps R24's drop rule. */
    reduce_order: z.record(TemplateNameSchema, z.array(ReduceStepSchema)).optional(),
  })
  .superRefine((t, ctx) => {
    for (const [name, steps] of Object.entries(t.reduce_order ?? {})) {
      const slots = t.day_types[name as TemplateName];
      steps.forEach((step, i) => {
        const slot = slots.find((s) => s.role === step.role);
        if (slot === undefined || slot.required) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            path: ["reduce_order", name, i, "role"],
            message: `${step.role} is not an optional ${name} slot`,
          });
        }
      });
    }
  });
export type Templates = z.infer<typeof TemplatesSchema>;

const AdjustmentSchema = z.object({ item: z.string(), from: z.number(), to: z.number() });

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

/** An optional slot removed ("no_time", R21) or cut to its min_minutes ("shrunk", E4) because the day had no time for it. */
export const DroppedSlotSchema = z.object({ role: z.string(), reason: z.enum(["no_time", "shrunk"]) });
export type DroppedSlot = z.infer<typeof DroppedSlotSchema>;

export const ScheduleDaySchema = z.object({
  day_index: z.number().int().min(1),
  weekday: WeekdaySchema,
  day_type: DayTypeSchema,
  attempt: z.number().int().min(0),
  wake: ClockSchema,
  sleep: ClockSchema,
  exceptions: z.array(ExceptionSchema),
  /** No day exception and at least one feasible slot-exception candidate (E1). */
  slot_exception_eligible: z.boolean(),
  /** Catalog slot exceptions that could replace a kept slot today and still lay out (E3); 0 on day-exception days. */
  slot_exception_candidates: z.number().int().min(0),
  slots: z.array(SlotSchema),
  dropped_slots: z.array(DroppedSlotSchema),
  transitions: z.array(SlotTransitionSchema),
  /** Clamps applied after an anchor's jitter redraws ran out (E6, G13). */
  anchor_adjustments: z.array(AdjustmentSchema),
  /** Per-anchor jitter redraws spent on the day, summed over anchors (G13). */
  jitter_redraws: z.number().int().min(0),
});
export type ScheduleDay = z.infer<typeof ScheduleDaySchema>;

const DonorPickSchema = z.object({
  case_id: z.string(),
  backoff_level: z.enum(["occupation_sex_age", "occupation_sex", "occupation", "sex_age", "all"]),
  filled: z.array(z.string()),
  redraws: z.number().int().min(0).max(5),
  adjustments: z.array(AdjustmentSchema),
  sport_fill: z.enum(["level", "pool"]).optional(),
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

/** Routine consistency over pairs of days (E2). */
const ConsistencyValuesSchema = z.object({
  /** Mean Jaccard of the sets of kept optional roles. */
  optional_jaccard: Prob01,
  /** Mean of 1 − edit distance ÷ max length of the kept role sequences. */
  order_similarity: Prob01,
  /** Mean |start difference| in minutes over roles kept on both days; null when no pair shares a role. */
  start_diff_minutes: z.number().min(0).nullable(),
  pairs: z.number().int().min(1),
});

/**
 * E2 over all pairs of days that use one template, plus (G15) the same values
 * over pairs that share a routine reference: the same weekday on workdays, the
 * same day type otherwise; null when no pair shares one.
 */
export const CompositionConsistencySchema = ConsistencyValuesSchema.extend({
  same_reference: ConsistencyValuesSchema.nullable(),
});
export type CompositionConsistency = z.infer<typeof CompositionConsistencySchema>;

/** What the dials produced (CR1). A ratio is null when its denominator is 0. */
export const RealizedSchema = z.object({
  /** Days with a slot exception ÷ eligible days (E1); the number that should match exception_rate.slot. */
  slot_exception_rate: Prob01.nullable(),
  /** Slot exceptions ÷ feasible candidate slots, summed over days (E1). */
  slot_exception_share: Prob01.nullable(),
  /** Slot exceptions whose meal_place equals their routine_place ÷ slot exceptions. */
  slot_exception_noop_share: Prob01.nullable(),
  slot_exception_eligible_days: z.number().int().min(0),
  /** Day exceptions ÷ days whose base day type has a catalog day exception (workdays). */
  day_exception_rate: Prob01.nullable(),
  /** Per role: days the role is kept ÷ days whose template has it as an optional slot. */
  optional_presence: z.record(z.string(), Prob01),
  /** Per role with a shrink step: days it was shrunk and still kept (G14). */
  shrunk: z.record(z.string(), z.number().int().min(0)),
  /** Templates used on fewer than 2 days are omitted. */
  composition_consistency: z.record(TemplateNameSchema, CompositionConsistencySchema),
  dials: z.object({ routine_strength: Prob01, exception_rate: z.object({ slot: Prob01, day: Prob01 }) }),
});
export type Realized = z.infer<typeof RealizedSchema>;

export const ScheduleSchema = z.object({
  person_id: z.string(),
  generator_version: z.literal("schedule.v2"),
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
    rest_day: z.object({
      breakfast: MealDistributionSchema,
      lunch: MealDistributionSchema,
      dinner: MealDistributionSchema,
    }),
  }),
  realized: RealizedSchema,
  days: z.array(ScheduleDaySchema),
});
export type Schedule = z.infer<typeof ScheduleSchema>;
