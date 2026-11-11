import { rngFor, type Rng } from "../rng.js";
import { MealPlaceSchema, WeekdaySchema, type MealPlace, type MealWeights, type Weekday, type World } from "../world.js";
import type { DayDecisions, DayType, Exception, ExceptionCatalog, Meal, TemplateName } from "./types.js";

/** Renormalized meal place probabilities; {} when no usable place has mass (R12). */
export type MealDistribution = Partial<Record<MealPlace, number>>;
export type EffectiveMealStyle = {
  workday: { lunch: MealDistribution; dinner: MealDistribution };
  weekend: { breakfast: MealDistribution; lunch: MealDistribution; dinner: MealDistribution };
};

export type DecideOptions = { boundNodes: Set<string> };

const WEEKDAYS = WeekdaySchema.options;
const WORKDAYS: readonly Weekday[] = ["Mon", "Tue", "Wed", "Thu", "Fri"];
/** Presence probability of the optional workday home_evening slot (no source gives one). */
const HOME_EVENING_PROB = 0.5;

/** D4: eat_out / others_home are usable only when their place node exists and is bound. */
function mealPlaceUsable(world: World, place: MealPlace, boundNodes: Set<string>): boolean {
  if (place === "home" || place === "workplace") return true;
  const node = world.meal_style.place_nodes[place];
  return node !== undefined && boundNodes.has(node);
}

function renormalize(weights: MealWeights, world: World, boundNodes: Set<string>): MealDistribution {
  const kept: [MealPlace, number][] = [];
  for (const place of MealPlaceSchema.options) {
    const w = weights[place] ?? 0;
    if (w > 0 && mealPlaceUsable(world, place, boundNodes)) kept.push([place, w]);
  }
  const total = kept.reduce((s, [, w]) => s + w, 0);
  return Object.fromEntries(kept.map(([place, w]) => [place, w / total]));
}

/** meal_style after dropping unusable places and renormalizing (D4, R12). */
export function effectiveMealStyle(world: World, boundNodes: Set<string>): EffectiveMealStyle {
  const { workday, weekend } = world.meal_style;
  const norm = (w: MealWeights) => renormalize(w, world, boundNodes);
  return {
    workday: { lunch: norm(workday.lunch), dinner: norm(workday.dinner) },
    weekend: { breakfast: norm(weekend.breakfast), lunch: norm(weekend.lunch), dinner: norm(weekend.dinner) },
  };
}

type HobbyPlan = {
  /** Weekend hobby days per week: Sat when ≥ 1, Sun too when 2. */
  weekendDays: number;
  /** Workday hobby occurrences per week in the reference week, capped at 5. */
  workdayCount: number;
  /** Reference hobby node per weekday that holds an occurrence. */
  byWeekday: Map<Weekday, string>;
  weights: { value: string; weight: number }[];
};

function hobbyPlan(world: World): HobbyPlan {
  const hobbies = Object.entries(world.nodes)
    .flatMap(([name, def]) => (def.kind === "hobby" && def.weekly_freq > 0 ? [{ name, freq: def.weekly_freq }] : []))
    .sort((a, b) => b.freq - a.freq || (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  const budget = hobbies.reduce((s, h) => s + h.freq, 0);
  const weekendDays = Math.min(budget, 2);
  const workdayCount = Math.min(budget - weekendDays, WORKDAYS.length);
  const occurrences: Weekday[] = [...(["Sat", "Sun"] as const).slice(0, weekendDays), ...WORKDAYS.slice(0, workdayCount)];
  const byWeekday = new Map(occurrences.map((wd, i) => [wd, hobbies[i % hobbies.length]!.name]));
  const weights = hobbies.map((h) => ({ value: h.name, weight: h.freq }));
  return { weekendDays, workdayCount, byWeekday, weights };
}

function baseDayType(weekday: Weekday, plan: HobbyPlan): DayType {
  if (weekday === "Sat") return plan.weekendDays >= 1 ? "hobby_day" : "home_day";
  if (weekday === "Sun") return plan.weekendDays >= 2 ? "hobby_day" : "home_day";
  return "workday";
}

function mealDraw(dist: MealDistribution): (rng: Rng) => MealPlace | "skip" {
  const items = Object.entries(dist).map(([place, weight]) => ({ value: place as MealPlace, weight }));
  return (rng) => (items.length === 0 ? "skip" : rng.pickWeighted(items));
}

/**
 * Decisions for one day (§5.2 steps 1-3, 6, 7). Each decision copies the
 * reference with probability routine_strength and is redrawn otherwise (D14).
 */
export function decideDay(
  world: World,
  catalog: ExceptionCatalog,
  opts: DecideOptions,
  dayIndex: number,
  attempt: number,
): DayDecisions {
  const { seed, dials } = world;
  const weekday = WEEKDAYS[(WEEKDAYS.indexOf(dials.start_weekday) + dayIndex - 1) % 7]!;
  const plan = hobbyPlan(world);
  const dayRng = (name: string) => rngFor(seed, "day", dayIndex, attempt, name);

  let dayType = baseDayType(weekday, plan);
  let template = dayType as TemplateName;
  const exceptions: Exception[] = [];

  const dayEntries = catalog.day.filter((e) => e.from_day_type === dayType);
  if (dayEntries.length > 0) {
    const rng = dayRng("day_exception");
    if (rng.bernoulli(dials.exception_rate.day)) {
      const entry = dayEntries[rng.int(0, dayEntries.length)]!;
      exceptions.push({ scope: "day", type: entry.type, replaces: dayType, day_range: [dayIndex, dayIndex] });
      dayType = entry.type;
      template = entry.template;
    }
  }

  // Reference decisions: per weekday on workdays, per day type otherwise.
  const refKey = template === "workday" ? weekday : dayType;
  function decide<T>(name: string, draw: (rng: Rng) => T, reference?: T): T {
    const rng = dayRng(name);
    if (rng.bernoulli(1 - dials.routine_strength)) return draw(rng);
    return reference !== undefined ? reference : draw(rngFor(seed, "reference", refKey, name));
  }

  const style = effectiveMealStyle(world, opts.boundNodes);
  const meal_place: Partial<Record<Meal, MealPlace | "skip">> = {};
  const optional: Record<string, boolean> = {};
  let hobbyPresent = template === "hobby_day";

  if (template === "workday") {
    meal_place.lunch = decide("meal_lunch", mealDraw(style.workday.lunch));
    meal_place.dinner = decide("meal_dinner", mealDraw(style.workday.dinner));
    optional.hobby = decide(
      "optional_hobby",
      (rng) => rng.bernoulli(plan.workdayCount / WORKDAYS.length),
      plan.byWeekday.has(weekday),
    );
    optional.home_evening = decide("optional_home_evening", (rng) => rng.bernoulli(HOME_EVENING_PROB));
    hobbyPresent = optional.hobby;
  } else {
    const skip = decide("breakfast_skip", (rng) => rng.bernoulli(world.meal_style.breakfast_skip_prob));
    const breakfast = decide("meal_breakfast", mealDraw(style.weekend.breakfast));
    meal_place.breakfast = skip ? "skip" : breakfast;
    meal_place.lunch = decide("meal_lunch", mealDraw(style.weekend.lunch));
    meal_place.dinner = decide("meal_dinner", mealDraw(style.weekend.dinner));
  }

  const hobby_node = hobbyPresent
    ? decide("hobby_node", (rng) => rng.pickWeighted(plan.weights), plan.byWeekday.get(weekday))
    : undefined;

  if (exceptions.length === 0) {
    const eligible = catalog.slot.filter(
      (e) => e.day_types.includes(dayType) && mealPlaceUsable(world, e.meal_place, opts.boundNodes),
    );
    if (eligible.length > 0) {
      const rng = dayRng("slot_exception");
      if (rng.bernoulli(dials.exception_rate.slot)) {
        const entry = eligible[rng.int(0, eligible.length)]!;
        exceptions.push({ scope: "slot", type: entry.type, replaces: entry.replaces_role, day_range: [dayIndex, dayIndex] });
      }
    }
  }

  return {
    day_index: dayIndex,
    weekday,
    day_type: dayType,
    template,
    meal_place,
    optional,
    ...(hobby_node !== undefined ? { hobby_node } : {}),
    exceptions,
  };
}

/** One DayDecisions per day of the world, all at attempt 0. */
export function decideDays(world: World, catalog: ExceptionCatalog, opts: DecideOptions): DayDecisions[] {
  return Array.from({ length: world.dials.length_days }, (_, i) => decideDay(world, catalog, opts, i + 1, 0));
}
