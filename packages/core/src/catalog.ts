import type { Annotation, BaseDay, BlockCandidate, Dataset } from "./schema.js";

/**
 * Read-only view of the normalized catalog. Implementations (filesystem,
 * in-memory fixtures) live outside @vb/core so core stays I/O-free.
 */
export interface Catalog {
  getBaseDay(day: string): BaseDay | undefined;
  /** Resolves both event candidates and base-day blocks. */
  getCandidate(id: string): BlockCandidate | undefined;
  getAnnotations(dataset: Dataset, videoId: string): Annotation[];
}
