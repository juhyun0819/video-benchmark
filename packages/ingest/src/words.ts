const escapeRegExp = (s: string): string => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** True when `word` appears in `text` as a whole word, case-insensitive ("car" matches "the Car." but not "card"). */
export function containsWord(text: string, word: string): boolean {
  return new RegExp(`(?<![\\p{L}\\p{N}])${escapeRegExp(word)}(?![\\p{L}\\p{N}])`, "iu").test(text);
}
