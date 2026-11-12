import { readFileSync } from "node:fs";
import { z } from "zod";
import { ClipCategorySchema, SettingSchema, overlapsSpan, timeOfDayAt, type Annotation, type Setting, type ClipCategory } from "@vb/core";
import { containsWord } from "../words.js";
import type { buildNextmeDay } from "./adapter.js";
import { hktMinuteOfDay, type DraftRow } from "./draft.js";

const HHMM = z.string().regex(/^\d{2}:\d{2}$/, "expected HH:MM");

/** Experimental L3 label rules; separate from the L4 `LabelRulesSchema`. */
export const L3RulesSchema = z.object({
  version: z.number(),
  person_source: z.string(),
  min_clip_seconds: z.number().positive(),
  places: z.array(
    z.object({
      words: z.array(z.string()),
      node: z.string(),
      setting: SettingSchema,
      category: ClipCategorySchema,
      tags: z.array(z.string()),
    }),
  ),
  meal_words: z.array(z.string()),
  never_home_phrases: z.array(z.string()),
  trip_dates: z.array(z.string()),
  trip: z.object({ lodging_words: z.array(z.string()), outing_words: z.array(z.string()) }),
});
export type L3Rules = z.infer<typeof L3RulesSchema>;

/** Per-person relabel of still-unlabeled clips starting in [from, to] (HKT, inclusive minutes) on `day_group`. Kept out of git. */
export const OverrideSchema = z
  .object({
    day_group: z.string().regex(/^\d{4}-\d{2}-\d{2}$/, "expected YYYY-MM-DD"),
    from: HHMM,
    to: HHMM,
    node: z.string(),
    setting: SettingSchema,
    category: ClipCategorySchema,
  })
  .refine((o) => o.from <= o.to, { message: "from must not be after to" });
export type L3Override = z.infer<typeof OverrideSchema>;

export function loadOverrides(path: string): L3Override[] {
  let json: unknown;
  try {
    json = JSON.parse(readFileSync(path, "utf8"));
  } catch (e) {
    throw new Error(`${path}: cannot read overrides: ${(e as Error).message}`);
  }
  const parsed = z.array(OverrideSchema).safeParse(json);
  if (!parsed.success) {
    const issue = parsed.error.issues.map((i) => `${i.path.join(".") || "(root)"}: ${i.message}`).join("; ");
    throw new Error(`${path}: ${issue}`);
  }
  return parsed.data;
}

export function loadL3Rules(path: string): L3Rules {
  let json: unknown;
  try {
    json = JSON.parse(readFileSync(path, "utf8"));
  } catch (e) {
    throw new Error(`${path}: cannot read L3 label rules: ${(e as Error).message}`);
  }
  const parsed = L3RulesSchema.safeParse(json);
  if (!parsed.success) {
    const issue = parsed.error.issues.map((i) => `${i.path.join(".") || "(root)"}: ${i.message}`).join("; ");
    throw new Error(`${path}: ${issue}`);
  }
  return parsed.data;
}

const hasAny = (text: string, words: string[]): boolean => words.some((w) => containsWord(text, w));
const minutes = (hhmm: string): number => Number(hhmm.slice(0, 2)) * 60 + Number(hhmm.slice(3, 5));

type Label = { node: string; setting: Setting; category: ClipCategory; tags: string[] };

/** Label one text level (no fallback). Trip rules come first on trip dates; transit still applies there. */
function labelText(text: string, rules: L3Rules, trip: boolean): Label | undefined {
  const transit = rules.places.find((p) => p.node === "transit");
  const transitLabel = (): Label | undefined =>
    transit && hasAny(text, transit.words)
      ? { node: transit.node, setting: transit.setting, category: transit.category, tags: transit.tags }
      : undefined;

  if (trip) {
    if (hasAny(text, rules.trip.lodging_words)) {
      return { node: "trip:lodging", setting: "venue_indoor", category: "home", tags: [] };
    }
    if (hasAny(text, rules.trip.outing_words)) {
      return { node: "trip:outing", setting: "outdoor", category: "hobby", tags: [] };
    }
    if (hasAny(text, rules.meal_words)) {
      return { node: "trip:meal", setting: "venue_indoor", category: "meal", tags: [] };
    }
    return transitLabel();
  }

  let cleaned = text;
  for (const phrase of rules.never_home_phrases) {
    cleaned = cleaned.replace(new RegExp(phrase.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "giu"), " ");
  }
  const place = rules.places.find((p) => hasAny(cleaned, p.words));
  if (!place) return undefined;
  const category = place.node !== "transit" && place.node !== "hobby" && hasAny(text, rules.meal_words) ? "meal" : place.category;
  return { node: place.node, setting: place.setting, category, tags: place.tags };
}

/**
 * One draft clip row per L3 segment of at least `min_clip_seconds`. The place is read from the L3
 * text, then the enclosing L4, then the enclosing L5 (first match wins). Rows without a label keep
 * null labels, then date/time overrides relabel the ones that are still unlabeled.
 */
export function nextmeL3Rows(day: string, built: ReturnType<typeof buildNextmeDay>, rules: L3Rules, overrides: L3Override[] = []): DraftRow[] {
  const trip = rules.trip_dates.includes(day);
  const rows: DraftRow[] = [];
  for (const [recId, anns] of built.annotations) {
    const block = built.baseDay.blocks.find((b) => b.source.video_id === recId);
    if (!block) continue;
    const source = block.source;
    const wallclock = source.media.kind === "frames" ? source.media.start_wallclock : undefined;
    const prefix = `nextme800:${recId}:`;
    const enclosing = (level: "L4" | "L5", t: number): Annotation | undefined =>
      anns.find((a) => a.level === level && a.time.kind === "interval" && a.time.start <= t && t < a.time.end);

    for (const l3 of anns) {
      if (l3.level !== "L3" || l3.time.kind !== "interval") continue;
      const { start, end } = l3.time;
      if (end - start < rules.min_clip_seconds) continue;

      const label = [l3.text, enclosing("L4", start)?.text, enclosing("L5", start)?.text]
        .filter((t): t is string => t !== undefined)
        .map((t) => labelText(t, rules, trip))
        .find((l) => l !== undefined);

      rows.push({
        clip_id: `nextme800:${recId}:${start.toFixed(3)}-${end.toFixed(3)}`,
        category: label?.category ?? null,
        setting: label?.setting ?? null,
        node: label?.node ?? null,
        activity: l3.text,
        time_of_day: wallclock ? timeOfDayAt(hktMinuteOfDay(wallclock, start)) : "unknown",
        tags: label?.tags ?? [],
        source,
        source_start_s: start,
        source_end_s: end,
        day_group: day,
        person_source: rules.person_source,
        fingerprint: null,
        extractor_version: "nextme-l3.v1",
        captions: anns
          .filter((a) => overlapsSpan(a.time, start, end))
          .map((a) => ({
            id: a.id.startsWith(prefix) ? a.id.slice(prefix.length) : a.id,
            level: a.level,
            time: a.time,
            text: a.text,
          })),
      });
    }
  }

  for (const row of rows) {
    if (row.node !== null) continue;
    const wallclock = row.source.media.kind === "frames" ? row.source.media.start_wallclock : undefined;
    if (!wallclock) continue;
    const m = hktMinuteOfDay(wallclock, row.source_start_s);
    const o = overrides.find((x) => x.day_group === row.day_group && minutes(x.from) <= m && m <= minutes(x.to));
    if (o) {
      row.node = o.node;
      row.setting = o.setting;
      row.category = o.category;
    }
  }
  return rows;
}
