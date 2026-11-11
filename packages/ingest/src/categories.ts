import { readFileSync } from "node:fs";
import { z } from "zod";

export const CategorySchema = z.object({
  id: z.string(),
  label: z.string(),
  dataset: z.enum(["ego4d", "egoexo4d"]),
  scenarios: z.array(z.string()).min(1),
  keywords: z.array(z.string()),
});
export type Category = z.infer<typeof CategorySchema>;

export const CategoriesSchema = z.object({ categories: z.array(CategorySchema) });
export type Categories = z.infer<typeof CategoriesSchema>;

const escapeRegExp = (s: string): string => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/**
 * True when the category has no keywords or one of them appears in `text` as a whole word (case-insensitive).
 * "car" matches "the Car." but not "carried" or "card"; list inflections ("tent", "tents") as separate keywords.
 */
export function keywordMatches(cat: Category, text: string): boolean {
  if (cat.keywords.length === 0) return true;
  return cat.keywords.some((k) => new RegExp(`(?<![\\p{L}\\p{N}])${escapeRegExp(k)}(?![\\p{L}\\p{N}])`, "iu").test(text));
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
