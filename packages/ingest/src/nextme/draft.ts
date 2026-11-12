import { readFileSync } from "node:fs";
import { z } from "zod";
import {
  ClipCategorySchema,
  SettingSchema,
  overlapsSpan,
  timeOfDayAt,
  type Annotation,
  type ClipContractRow,
} from "@vb/core";
import { parseHkt } from "../time.js";
import { containsWord } from "../words.js";
import type { buildNextmeDay } from "./adapter.js";

export const LabelRulesSchema = z.object({
  version: z.number(),
  person_source: z.string(),
  places: z.array(z.object({ words: z.array(z.string()), node: z.string(), setting: SettingSchema })),
  activities: z.array(z.object({ words: z.array(z.string()), category: ClipCategorySchema })),
  default_category_by_node: z.record(z.string(), ClipCategorySchema),
});
export type LabelRules = z.infer<typeof LabelRulesSchema>;

/** A clip-contract row whose labels may still be missing; `ingest bank` skips rows with a null category. */
export type DraftRow = Omit<ClipContractRow, "category" | "setting" | "node"> & {
  category: ClipContractRow["category"] | null;
  setting: ClipContractRow["setting"] | null;
  node: string | null;
};

export function loadLabelRules(path: string): LabelRules {
  let json: unknown;
  try {
    json = JSON.parse(readFileSync(path, "utf8"));
  } catch (e) {
    throw new Error(`${path}: cannot read label rules: ${(e as Error).message}`);
  }
  const parsed = LabelRulesSchema.safeParse(json);
  if (!parsed.success) {
    const issue = parsed.error.issues.map((i) => `${i.path.join(".") || "(root)"}: ${i.message}`).join("; ");
    throw new Error(`${path}: ${issue}`);
  }
  return parsed.data;
}

const hasAny = (text: string, words: string[]): boolean => words.some((w) => containsWord(text, w));

/** HKT minutes of day (floored) at `offsetSec` after the HKT wall-clock string `wallclock`. */
export function hktMinuteOfDay(wallclock: string, offsetSec: number): number {
  const epoch = parseHkt(wallclock) + offsetSec + 8 * 3600;
  return Math.floor((((epoch % 86400) + 86400) % 86400) / 60);
}

/** One draft clip row per L4-tiled block of the built day, labeled by keyword rules (spec: NextMe draft). */
export function nextmeDraftRows(day: string, built: ReturnType<typeof buildNextmeDay>, rules: LabelRules): DraftRow[] {
  const rows: DraftRow[] = [];
  for (const block of built.baseDay.blocks) {
    const source = block.source;
    const recId = source.video_id;
    const anns = built.annotations.get(recId) ?? [];
    const prefix = `nextme800:${recId}:`;

    const l5 = anns.find(
      (a) => a.level === "L5" && a.time.kind === "interval" && a.time.start <= block.src_start && block.src_start < a.time.end,
    );
    const place = [block.summary, l5?.text]
      .filter((t): t is string => t !== undefined)
      .map((t) => rules.places.find((p) => hasAny(t, p.words)))
      .find((p) => p !== undefined);
    const activity = rules.activities.find((a) => hasAny(block.summary, a.words));
    const category = place ? (activity?.category ?? rules.default_category_by_node[place.node] ?? null) : null;

    const wallclock = source.media.kind === "frames" ? source.media.start_wallclock : undefined;
    rows.push({
      clip_id: block.id,
      category,
      setting: place?.setting ?? null,
      node: place?.node ?? null,
      activity: block.summary,
      time_of_day: wallclock ? timeOfDayAt(hktMinuteOfDay(wallclock, block.src_start)) : "unknown",
      tags: [],
      source,
      source_start_s: block.src_start,
      source_end_s: block.src_end,
      day_group: day,
      person_source: rules.person_source,
      fingerprint: null,
      extractor_version: "nextme-draft.v1",
      captions: anns
        .filter((a) => overlapsSpan(a.time, block.src_start, block.src_end))
        .map((a) => ({
          id: a.id.startsWith(prefix) ? a.id.slice(prefix.length) : a.id,
          level: a.level,
          time: a.time,
          text: a.text,
        })),
    });
  }
  return rows;
}
