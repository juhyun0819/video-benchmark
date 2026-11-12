import { z } from "zod";
import { TimelineError } from "./errors.js";
import { SettingSchema } from "./clip.js";
import { DatasetSchema } from "./schema.js";

export const WeekdaySchema = z.enum(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]);
export type Weekday = z.infer<typeof WeekdaySchema>;

export const MealPlaceSchema = z.enum(["home", "workplace", "eat_out", "others_home"]);
export type MealPlace = z.infer<typeof MealPlaceSchema>;

export const NodeDefSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("home") }),
  z.object({ kind: z.literal("kitchen"), default: z.literal("home") }),
  /** `setting` defaults to venue_indoor; outdoor or in-vehicle jobs (roofer, truck driver) set it explicitly. */
  z.object({ kind: z.literal("workplace"), job: z.string(), setting: SettingSchema.optional() }),
  z.object({
    kind: z.literal("hobby"),
    activity: z.string(),
    setting: SettingSchema,
    weekly_freq: z.number().int().min(0).max(7),
    tags: z.array(z.string()).default([]),
  }),
  z.object({ kind: z.literal("frequent_place"), setting: SettingSchema, optional: z.literal(true) }),
]);
export type NodeDef = z.infer<typeof NodeDefSchema>;

export const NodeBindingSchema = z.object({
  dataset: z.array(DatasetSchema).optional(),
  person_source: z.string().optional(),
  node: z.string().optional(),
  tags_any: z.array(z.string()).optional(),
});
export type NodeBinding = z.infer<typeof NodeBindingSchema>;

/** Non-negative weights per place; missing places count as 0. */
const MealWeightsSchema = z.record(MealPlaceSchema, z.number().min(0));
export type MealWeights = z.infer<typeof MealWeightsSchema>;

const Prob = z.number().min(0).max(1);
/** E8: every meal distribution must sum to 1 within this tolerance. */
const MEAL_SUM_TOLERANCE = 0.01;

export const WorldSchema = z
  .object({
    person_id: z.string(),
    attributes: z.object({
      country: z.string(),
      occupation_group: z.string(),
      sex: z.string().optional(),
      age_band: z.string().optional(),
      work_pattern: z.enum(["day_shift"]),
    }),
    distribution_source: z.string(),
    allow_country_fallback: z.boolean().default(false),
    nodes: z.record(z.string(), NodeDefSchema),
    node_binding: z.record(z.string(), NodeBindingSchema),
    meal_style: z.object({
      workday: z.object({ lunch: MealWeightsSchema, dinner: MealWeightsSchema }),
      rest_day: z.object({
        breakfast: MealWeightsSchema,
        lunch: MealWeightsSchema,
        dinner: MealWeightsSchema,
      }),
      breakfast_skip_prob: Prob,
      place_nodes: z.object({ eat_out: z.string(), others_home: z.string() }).partial(),
    }),
    dials: z.object({
      length_days: z.number().int().min(28),
      routine_strength: Prob,
      exception_rate: z.object({ slot: Prob, day: Prob }),
      wake_sleep: z.string(),
      start_weekday: WeekdaySchema.default("Mon"),
    }),
    seed: z.number().int().min(0),
  })
  .superRefine((w, ctx) => {
    const hobbies = Object.values(w.nodes).filter((n) => n.kind === "hobby").length;
    if (hobbies < 1 || hobbies > 3) {
      ctx.addIssue({ code: "custom", path: ["nodes"], message: `expected 1-3 hobby nodes, got ${hobbies}` });
    }
    for (const [place, name] of Object.entries(w.meal_style.place_nodes)) {
      if (w.nodes[name]?.kind !== "frequent_place") {
        ctx.addIssue({
          code: "custom",
          path: ["meal_style", "place_nodes", place],
          message: `"${name}" is not a frequent_place node`,
        });
      }
    }
    const { workday, rest_day } = w.meal_style;
    const meals: [string, MealWeights][] = [
      ["workday.lunch", workday.lunch],
      ["workday.dinner", workday.dinner],
      ["rest_day.breakfast", rest_day.breakfast],
      ["rest_day.lunch", rest_day.lunch],
      ["rest_day.dinner", rest_day.dinner],
    ];
    for (const [key, weights] of meals) {
      const sum = Object.values(weights).reduce((s, v) => s + v, 0);
      if (Math.abs(sum - 1) > MEAL_SUM_TOLERANCE + 1e-9) {
        ctx.addIssue({
          code: "custom",
          path: ["meal_style", ...key.split(".")],
          message: `meal weights must sum to 1 +/- ${MEAL_SUM_TOLERANCE}, got ${Number(sum.toFixed(6))}`,
        });
      }
      if (key.startsWith("rest_day.") && (weights.workplace ?? 0) > 0) {
        ctx.addIssue({
          code: "custom",
          path: ["meal_style", ...key.split("."), "workplace"],
          message: "rest_day meals cannot take place at the workplace",
        });
      }
    }
    for (const [name, def] of Object.entries(w.nodes)) {
      if (def.kind === "kitchen" || (def.kind === "frequent_place" && def.optional)) continue;
      if (!(name in w.node_binding)) {
        ctx.addIssue({ code: "custom", path: ["node_binding", name], message: `node "${name}" has no binding` });
      }
    }
  });
export type World = z.infer<typeof WorldSchema>;

const WEEKEND_RENAMED_HINT = "hint: meal_style.weekend was renamed to meal_style.rest_day in rev3; rename it";

/**
 * Parses a world file; schema problems become WORLD_INVALID listing each `path: message`.
 * A world written before rev3 (meal_style.weekend) gets a rename hint (G25).
 */
export function parseWorld(json: unknown): World {
  const r = WorldSchema.safeParse(json);
  if (r.success) return r.data;
  const issues = r.error.issues.map((i) => ({ path: i.path.join("."), message: i.message }));
  const mealStyle = (json as { meal_style?: unknown } | null)?.meal_style;
  const legacy = typeof mealStyle === "object" && mealStyle !== null && "weekend" in mealStyle;
  const message = issues.map((i) => `${i.path}: ${i.message}`).join("; ");
  throw new TimelineError("WORLD_INVALID", legacy ? `${message}; ${WEEKEND_RENAMED_HINT}` : message, {
    issues,
    ...(legacy ? { hint: WEEKEND_RENAMED_HINT } : {}),
  });
}
