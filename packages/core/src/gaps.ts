import { z } from "zod";
import { TimelineError } from "./errors.js";
import type { Rng } from "./rng.js";
import type { NodeDef } from "./world.js";

export type PlaceKind = "home" | "workplace" | "hobby" | "frequent_place";
export type Transition = `${PlaceKind}->${PlaceKind}`;

/** Place kind of a world node; a kitchen counts as home. */
export function placeKindOf(kind: NodeDef["kind"]): PlaceKind {
  return kind === "kitchen" ? "home" : kind;
}

const GapQuantilesSchema = z
  .object({
    p10: z.number().min(0),
    p25: z.number().min(0),
    p50: z.number().min(0),
    p75: z.number().min(0),
    p90: z.number().min(0),
  })
  .refine((q) => q.p10 <= q.p25 && q.p25 <= q.p50 && q.p50 <= q.p75 && q.p75 <= q.p90, {
    message: "quantiles must be non-decreasing",
  });

export type GapQuantiles = z.infer<typeof GapQuantilesSchema>;

const TransitionKeySchema = z
  .string()
  .regex(/^(home|workplace|hobby|frequent_place)->(home|workplace|hobby|frequent_place)$/, "bad transition key");

export const GapTableSchema = z.object({
  version: z.string(),
  reviewed: z.boolean(),
  transitions: z.record(TransitionKeySchema, GapQuantilesSchema),
});
/** Entries may be absent; sampleGap reports the missing transition. */
export type GapTable = {
  version: string;
  reviewed: boolean;
  transitions: Partial<Record<Transition, GapQuantiles>>;
};

/**
 * Transition gap in minutes. Draws u in [0.10, 0.90] and interpolates
 * piecewise-linearly between the table quantiles.
 */
export function sampleGap(table: GapTable, from: PlaceKind, to: PlaceKind, rng: Rng): number {
  if (from === to) return 0;
  const transition: Transition = `${from}->${to}`;
  const q = table.transitions[transition];
  if (!q) throw new TimelineError("GAP_MISSING", `gap table has no entry for ${transition}`, { transition });
  const points: [number, number][] = [
    [0.1, q.p10],
    [0.25, q.p25],
    [0.5, q.p50],
    [0.75, q.p75],
    [0.9, q.p90],
  ];
  const u = 0.1 + 0.8 * rng.next();
  for (let i = 1; i < points.length; i++) {
    const [u1, v1] = points[i]!;
    if (u <= u1 || i === points.length - 1) {
      const [u0, v0] = points[i - 1]!;
      return Math.round(v0 + ((u - u0) / (u1 - u0)) * (v1 - v0));
    }
  }
  return Math.round(q.p90);
}

/** The table's median gap in minutes (no draw); 0 for the same place kind. */
export function medianGap(table: GapTable, from: PlaceKind, to: PlaceKind): number {
  if (from === to) return 0;
  const transition: Transition = `${from}->${to}`;
  const q = table.transitions[transition];
  if (!q) throw new TimelineError("GAP_MISSING", `gap table has no entry for ${transition}`, { transition });
  return q.p50;
}
