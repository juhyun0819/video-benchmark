import { z } from "zod";
import { MealPlaceSchema, type MealPlace, type Weekday } from "../world.js";

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
