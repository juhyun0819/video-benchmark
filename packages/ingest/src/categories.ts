import { readFileSync } from "node:fs";
import { z } from "zod";
import { ClipCategorySchema, SettingSchema } from "@vb/core";
import { containsWord } from "./words.js";

export const CategorySchema = z.object({
  id: z.string(),
  label: z.string(),
  dataset: z.enum(["ego4d", "egoexo4d"]),
  scenarios: z.array(z.string()).min(1),
  keywords: z.array(z.string()),
  /** When set, the text must also contain one of these words (e.g. an eating word next to a place word). */
  require_keywords: z.array(z.string()).optional(),
  /** When set, text containing any of these words never matches (and never continues a run). */
  exclude_keywords: z.array(z.string()).optional(),
  /**
   * Ego4D only. When set, a matching summary starts a run: the video's next summaries that start at or before
   * the run's end continue it while they contain a `continue_keywords` word (and no exclude word). The run is
   * one candidate spanning all its summaries, with the first summary's text. Runs shorter than
   * `min_seconds` (default 0) are dropped.
   */
  merge_runs: z
    .object({ continue_keywords: z.array(z.string()).min(1), min_seconds: z.number().nonnegative().optional() })
    .optional(),
  /** How candidates of this category become catalog clips; categories without it are not ingested as clips. */
  clip: z
    .object({ category: ClipCategorySchema, setting: SettingSchema, node: z.string(), tags: z.array(z.string()) })
    .optional(),
});
export type Category = z.infer<typeof CategorySchema>;

export const CategoriesSchema = z.object({ categories: z.array(CategorySchema) });
export type Categories = z.infer<typeof CategoriesSchema>;

/**
 * True when `text` has no exclude_keywords word, has a require_keywords word (when set), and the category
 * has no keywords or one of them appears in `text` as a whole word (case-insensitive).
 * "car" matches "the Car." but not "carried" or "card"; list inflections ("tent", "tents") as separate keywords.
 */
export function keywordMatches(cat: Category, text: string): boolean {
  if (excluded(cat, text)) return false;
  if (cat.require_keywords !== undefined && !cat.require_keywords.some((k) => containsWord(text, k))) return false;
  if (cat.keywords.length === 0) return true;
  return cat.keywords.some((k) => containsWord(text, k));
}

/** True when `text` contains one of the category's exclude_keywords. */
export function excluded(cat: Category, text: string): boolean {
  return (cat.exclude_keywords ?? []).some((k) => containsWord(text, k));
}

/** True when `text` continues a merge_runs run of the category. */
export function continuesRun(cat: Category, text: string): boolean {
  const run = cat.merge_runs;
  return run !== undefined && !excluded(cat, text) && run.continue_keywords.some((k) => containsWord(text, k));
}

/** Read and validate config/categories.json. Errors name the file and the offending key path. */
export function loadCategories(path: string): Categories {
  let json: unknown;
  try {
    json = JSON.parse(readFileSync(path, "utf8"));
  } catch (e) {
    throw new Error(`${path}: cannot read categories: ${(e as Error).message}`);
  }
  const parsed = CategoriesSchema.safeParse(json);
  if (!parsed.success) {
    const issue = parsed.error.issues
      .map((iss) => `${iss.path.join(".") || "(root)"}: ${iss.message}`)
      .join("; ");
    throw new Error(`${path}: ${issue}`);
  }
  return parsed.data;
}
