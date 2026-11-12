import { parseClock } from "../clock.js";
import type { World } from "../world.js";
import {
  TemplateNameSchema,
  type CompositionConsistency,
  type DayType,
  type ExceptionCatalog,
  type Realized,
  type ScheduleDay,
  type TemplateName,
  type Templates,
} from "./types.js";

/** Base day type of every catalog day exception (ExceptionCatalogSchema fixes it to workday). */
const DAY_EXCEPTION_FROM: ExceptionCatalog["day"][number]["from_day_type"] = "workday";

/** leave_day uses the home_day template. */
const templateOf = (dayType: DayType): TemplateName => (dayType === "leave_day" ? "home_day" : dayType);

const ratio = (n: number, d: number): number | null => (d === 0 ? null : n / d);

/** Days that copy the same routine reference (D14): the same weekday on workdays, the same day type otherwise (G15). */
const sameReference = (a: ScheduleDay, b: ScheduleDay): boolean =>
  a.day_type === b.day_type && (a.day_type !== "workday" || a.weekday === b.weekday);

function jaccard(a: ReadonlySet<string>, b: ReadonlySet<string>): number {
  const union = new Set([...a, ...b]).size;
  if (union === 0) return 1;
  let both = 0;
  for (const x of a) if (b.has(x)) both += 1;
  return both / union;
}

function editDistance(a: readonly string[], b: readonly string[]): number {
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const row = [i];
    for (let j = 1; j <= b.length; j++) {
      row.push(Math.min(prev[j]! + 1, row[j - 1]! + 1, prev[j - 1]! + (a[i - 1] === b[j - 1] ? 0 : 1)));
    }
    prev = row;
  }
  return prev[b.length]!;
}

type ConsistencyValues = Omit<CompositionConsistency, "same_reference">;

/** E2 over the pairs of the given days, which share one template, for which `paired` holds; null without pairs. */
function consistency(
  days: readonly ScheduleDay[],
  paired: (a: ScheduleDay, b: ScheduleDay) => boolean = () => true,
): ConsistencyValues | null {
  const views = days.map((d) => ({
    optional: new Set(d.slots.filter((s) => !s.required).map((s) => s.role)),
    order: d.slots.map((s) => s.role),
    start: new Map(d.slots.map((s) => [s.role, parseClock(s.window[0])])),
  }));
  let pairs = 0;
  let jac = 0;
  let order = 0;
  let diff = 0;
  let diffPairs = 0;
  for (let i = 0; i < views.length; i++) {
    for (let j = i + 1; j < views.length; j++) {
      if (!paired(days[i]!, days[j]!)) continue;
      const a = views[i]!;
      const b = views[j]!;
      pairs += 1;
      jac += jaccard(a.optional, b.optional);
      order += 1 - editDistance(a.order, b.order) / Math.max(a.order.length, b.order.length, 1);
      const shared = [...a.start].filter(([role]) => b.start.has(role));
      if (shared.length > 0) {
        diff += shared.reduce((s, [role, t]) => s + Math.abs(t - b.start.get(role)!), 0) / shared.length;
        diffPairs += 1;
      }
    }
  }
  if (pairs === 0) return null;
  return {
    optional_jaccard: jac / pairs,
    order_similarity: order / pairs,
    start_diff_minutes: ratio(diff, diffPairs),
    pairs,
  };
}

/**
 * `realized` for a schedule's days (CR1, E1, E2, G14): what the dials
 * produced, counted from the days alone. Key order is fixed; a ratio with a zero denominator is null.
 */
export function realizedOf(days: readonly ScheduleDay[], world: World, templates: Templates): Realized {
  const hasSlotException = (d: ScheduleDay) => d.exceptions.some((e) => e.scope === "slot");
  const eligible = days.filter((d) => d.slot_exception_eligible);
  const slotExceptionList = days.flatMap((d) => d.exceptions.filter((e) => e.scope === "slot"));
  const slotExceptions = slotExceptionList.length;
  const noops = slotExceptionList.filter((e) => e.routine_place === e.meal_place).length;
  const candidates = days.reduce((s, d) => s + d.slot_exception_candidates, 0);

  const dayException = (d: ScheduleDay) => d.exceptions.find((e) => e.scope === "day");
  const baseType = (d: ScheduleDay) => dayException(d)?.replaces ?? d.day_type;
  const dayExceptionBase = days.filter((d) => baseType(d) === DAY_EXCEPTION_FROM);

  // Roles in template order; [kept, days whose template has the role as optional].
  const presence = new Map<string, [number, number]>();
  for (const name of TemplateNameSchema.options) {
    for (const slot of templates.day_types[name]) if (!slot.required && !presence.has(slot.role)) presence.set(slot.role, [0, 0]);
  }
  for (const d of days) {
    const kept = new Set(d.slots.map((s) => s.role));
    for (const slot of templates.day_types[templateOf(d.day_type)]) {
      if (slot.required) continue;
      const c = presence.get(slot.role)!;
      c[1] += 1;
      if (kept.has(slot.role)) c[0] += 1;
    }
  }

  // G14: a shrunk slot that was later dropped counts as dropped, not shrunk.
  const shrunk: Record<string, number> = {};
  for (const name of TemplateNameSchema.options) {
    for (const step of templates.reduce_order?.[name] ?? []) {
      if (step.action === "shrink") shrunk[step.role] = 0;
    }
  }
  for (const d of days) {
    for (const role of Object.keys(shrunk)) {
      const cut = d.dropped_slots.some((x) => x.role === role && x.reason === "shrunk");
      if (cut && d.slots.some((s) => s.role === role)) shrunk[role]! += 1;
    }
  }

  const composition: Realized["composition_consistency"] = {};
  for (const name of TemplateNameSchema.options) {
    const same = days.filter((d) => templateOf(d.day_type) === name);
    if (same.length < 2) continue;
    composition[name] = { ...consistency(same)!, same_reference: consistency(same, sameReference) };
  }

  return {
    slot_exception_rate: ratio(eligible.filter(hasSlotException).length, eligible.length),
    slot_exception_share: ratio(slotExceptions, candidates),
    slot_exception_noop_share: ratio(noops, slotExceptions),
    slot_exception_eligible_days: eligible.length,
    day_exception_rate: ratio(dayExceptionBase.filter((d) => dayException(d) !== undefined).length, dayExceptionBase.length),
    optional_presence: Object.fromEntries(
      [...presence].filter(([, [, n]]) => n > 0).map(([role, [kept, n]]) => [role, kept / n]),
    ),
    shrunk,
    composition_consistency: composition,
    dials: {
      routine_strength: world.dials.routine_strength,
      exception_rate: { slot: world.dials.exception_rate.slot, day: world.dials.exception_rate.day },
    },
  };
}
