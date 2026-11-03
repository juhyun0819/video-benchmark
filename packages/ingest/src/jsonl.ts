import type { z } from "zod";

export class IngestFormatError extends Error {
  readonly file: string;
  readonly line: number;
  readonly issue: string;

  constructor(file: string, line: number, issue: string) {
    super(`${file}:${line}: ${issue}`);
    this.name = "IngestFormatError";
    this.file = file;
    this.line = line;
    this.issue = issue;
  }
}

/** Counts per category; skip reasons include "invalid_time" and "end_before_start". */
export type IngestReport = {
  ingested: Record<string, number>;
  skipped: Record<string, number>;
};

export function newReport(): IngestReport {
  return { ingested: {}, skipped: {} };
}

export function bump(bucket: Record<string, number>, key: string, n = 1): void {
  bucket[key] = (bucket[key] ?? 0) + n;
}

/**
 * Parse JSON Lines text with a zod schema. Blank lines are skipped but still
 * counted for line numbers. Throws IngestFormatError(file, line, issue).
 */
export function readJsonl<T>(text: string, file: string, schema: z.ZodType<T, z.ZodTypeDef, unknown>): T[] {
  const out: T[] = [];
  const lines = text.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i];
    if (raw === undefined || raw.trim() === "") continue;
    let json: unknown;
    try {
      json = JSON.parse(raw);
    } catch (e) {
      throw new IngestFormatError(file, i + 1, `invalid JSON: ${(e as Error).message}`);
    }
    const parsed = schema.safeParse(json);
    if (!parsed.success) {
      const issue = parsed.error.issues
        .map((iss) => `${iss.path.join(".") || "(row)"}: ${iss.message}`)
        .join("; ");
      throw new IngestFormatError(file, i + 1, issue);
    }
    out.push(parsed.data);
  }
  return out;
}
