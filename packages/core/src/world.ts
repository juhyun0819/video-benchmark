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
  z.object({ kind: z.literal("workplace"), job: z.string() }),
  z.object({
    kind: z.literal("hobby"),
    activity: z.string(),
    setting: SettingSchema,
    weekly_freq: z.number().int().min(0).max(7),
    tags: z.array(z.string()).default([]),
  }),
  z.object({ kind: z.literal("frequent_place"), setting: SettingSchema, optional: z.literal(true) }),
  /** N2: one trip destination, used through its virtual places `<trip>:lodging|outing|meal`. */
  z.object({
    kind: z.literal("trip"),
    lodging_setting: SettingSchema,
    outing_setting: SettingSchema,
    meal_setting: SettingSchema,
  }),
]);
export type NodeDef = z.infer<typeof NodeDefSchema>;
export type TripNodeDef = Extract<NodeDef, { kind: "trip" }>;

/** The virtual places of a trip node (N2). */
export const TRIP_PLACES = ["lodging", "outing", "meal"] as const;
export type TripPlace = (typeof TRIP_PLACES)[number];

/** Fixed node label of a trip place on catalog clips (N2): `trip:lodging`, `trip:outing`, `trip:meal`. */
export const tripClipNode = (place: TripPlace): string => `trip:${place}`;

/** `<trip>:<place>` resolved against the world's trip nodes; undefined for any other name. */
export function tripPlaceOf(
  world: Pick<World, "nodes">,
  name: string,
): { trip: string; place: TripPlace; def: TripNodeDef } | undefined {
  const at = name.lastIndexOf(":");
  if (at < 0) return undefined;
  const trip = name.slice(0, at);
  const place = name.slice(at + 1) as TripPlace;
  const def = world.nodes[trip];
  if (def?.kind !== "trip" || !TRIP_PLACES.includes(place)) return undefined;
  return { trip, place, def };
}

/** Weekday of a YYYY-MM-DD date by pure date arithmetic (no time zone). */
export function weekdayOfDate(date: string): Weekday {
  const [y, m, d] = date.split("-").map(Number);
  return WeekdaySchema.options[(new Date(Date.UTC(y!, m! - 1, d!)).getUTCDay() + 6) % 7]!;
}

const IsoDateSchema = z
  .string()
  .regex(/^\d{4}-\d{2}-\d{2}$/, "expected a YYYY-MM-DD date")
  .refine((s) => {
    const [y, m, d] = s.split("-").map(Number);
    const t = new Date(Date.UTC(y!, m! - 1, d!));
    return t.getUTCFullYear() === y && t.getUTCMonth() === m! - 1 && t.getUTCDate() === d;
  }, "not a calendar date");

export const NodeBindingSchema = z.object({
  dataset: z.array(DatasetSchema).optional(),
  person_source: z.string().optional(),
  node: z.string().optional(),
  tags_any: z.array(z.string()).optional(),
  /** Trip bindings only (N2): the source day_group used on each day of the trip, in order. */
  day_group_sequence: z.array(IsoDateSchema).min(1).optional(),
});
export type NodeBinding = z.infer<typeof NodeBindingSchema>;

/** Non-negative weights per place; missing places count as 0. */
const MealWeightsSchema = z.record(MealPlaceSchema, z.number().min(0));
export type MealWeights = z.infer<typeof MealWeightsSchema>;

const Prob = z.number().min(0).max(1);

/**
 * B8: an activity inserted on routine workdays (template workday, no exception) after work_pm
 * (before_dinner) or right after dinner (after_dinner). Days come from exactly one of `weekdays`
 * (every such day) or `count` (the days whose work_pm ends earliest, at most `max_per_week` per ISO week).
 */
export const EveningActivitySchema = z.object({
  /** A hobby node; its place, setting, activity and tags make the slot spec. */
  node: z.string(),
  /** Slot role, unique among the activities and not a template role. */
  role: z.string(),
  position: z.enum(["before_dinner", "after_dinner"]),
  duration_min: z.number().int().positive(),
  weekdays: z.array(WeekdaySchema).min(1).optional(),
  count: z.number().int().min(1).optional(),
  max_per_week: z.number().int().min(1).optional(),
});
export type EveningActivity = z.infer<typeof EveningActivitySchema>;
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
      /** Workdays per week that start late (workday_late); the weekdays are fixed per person. */
      late_workdays_per_week: z.number().int().min(0).max(5).default(0),
      /** Multi-day trip exceptions (N3); each length equals its binding's day_group_sequence length. */
      trips: z
        .array(
          z.object({
            node: z.string(),
            length_days: z.number().int().min(1),
            /** B1: fixed 1-based synthetic start day; no weekday match and no draw. */
            start_day: z.number().int().min(1).optional(),
          }),
        )
        .default([]),
      /** N6: a required slot of `role` may be left empty on up to `max_days` distinct days when no clip fits. */
      missing_required_allowance: z.array(z.object({ role: z.string(), max_days: z.number().int().min(1) })).default([]),
      /** B8: evening activities on routine workdays. Optional with no default, so worlds without it keep their bytes. */
      evening_activities: z.array(EveningActivitySchema).optional(),
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
    for (const [name, b] of Object.entries(w.node_binding)) {
      const trip = w.nodes[name]?.kind === "trip";
      if (b.day_group_sequence !== undefined && !trip) {
        ctx.addIssue({
          code: "custom",
          path: ["node_binding", name, "day_group_sequence"],
          message: `only trip nodes take a day_group_sequence`,
        });
      }
      if (trip && b.node !== undefined) {
        ctx.addIssue({
          code: "custom",
          path: ["node_binding", name, "node"],
          message: `trip bindings take no node: trip clips carry the labels trip:lodging, trip:outing and trip:meal`,
        });
      }
    }
    const roles = new Set<string>();
    (w.dials.evening_activities ?? []).forEach((a, i) => {
      const path = ["dials", "evening_activities", i];
      if (w.nodes[a.node]?.kind !== "hobby") {
        ctx.addIssue({ code: "custom", path: [...path, "node"], message: `"${a.node}" is not a hobby node` });
      }
      if ((a.weekdays === undefined) === (a.count === undefined)) {
        ctx.addIssue({ code: "custom", path, message: "give exactly one of weekdays and count" });
      }
      if (a.max_per_week !== undefined && a.count === undefined) {
        ctx.addIssue({ code: "custom", path: [...path, "max_per_week"], message: "max_per_week needs count" });
      }
      if (roles.has(a.role)) {
        ctx.addIssue({ code: "custom", path: [...path, "role"], message: `role "${a.role}" is listed twice` });
      }
      roles.add(a.role);
    });
    const seen = new Set<string>();
    w.dials.trips.forEach((t, i) => {
      const path = ["dials", "trips", i];
      if (w.nodes[t.node]?.kind !== "trip") {
        ctx.addIssue({ code: "custom", path: [...path, "node"], message: `"${t.node}" is not a trip node` });
        return;
      }
      if (seen.has(t.node)) {
        ctx.addIssue({ code: "custom", path: [...path, "node"], message: `trip "${t.node}" is listed twice` });
      }
      seen.add(t.node);
      const sequence = w.node_binding[t.node]?.day_group_sequence;
      if (sequence === undefined || sequence.length !== t.length_days) {
        ctx.addIssue({
          code: "custom",
          path: [...path, "length_days"],
          message: `length_days ${t.length_days} must equal the length of node_binding.${t.node}.day_group_sequence (${sequence?.length ?? "missing"})`,
        });
      }
      if (t.length_days > w.dials.length_days) {
        ctx.addIssue({ code: "custom", path: [...path, "length_days"], message: `trip is longer than the world` });
      }
      if (t.start_day !== undefined) {
        const end = t.start_day + t.length_days - 1;
        if (end > w.dials.length_days) {
          ctx.addIssue({
            code: "custom",
            path: [...path, "start_day"],
            message: `trip ${t.node} starting on day ${t.start_day} ends on day ${end}, past length_days ${w.dials.length_days}`,
          });
        }
        w.dials.trips.slice(0, i).forEach((o) => {
          if (o.start_day === undefined) return;
          if (t.start_day! <= o.start_day + o.length_days - 1 && o.start_day <= end) {
            ctx.addIssue({
              code: "custom",
              path: [...path, "start_day"],
              message: `trip ${t.node} (days ${t.start_day}-${end}) overlaps trip ${o.node} (days ${o.start_day}-${o.start_day + o.length_days - 1})`,
            });
          }
        });
      }
    });
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
