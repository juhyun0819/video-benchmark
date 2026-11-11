import { z } from "zod";
import { SourceRefSchema } from "@vb/core";

// Block candidates and base days: internal shapes of the adapters (the catalog itself stores clips).
export const BlockCandidateSchema = z.object({
  id: z.string(),
  role: z.enum(["base", "event"]),
  category: z.string(),
  source: SourceRefSchema,
  src_start: z.number(),
  src_end: z.number(),
  summary: z.string(),
});
export type BlockCandidate = z.infer<typeof BlockCandidateSchema>;

export const BaseDaySchema = z.object({
  day: z.string(),
  blocks: z.array(BlockCandidateSchema),
});
export type BaseDay = z.infer<typeof BaseDaySchema>;
