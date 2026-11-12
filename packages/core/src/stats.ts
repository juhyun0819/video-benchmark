import type { TimelineManifest } from "./manifest.js";

// Pure aggregation over exported manifests (CR1.4, CR5.3; E10, G18).

export type FieldStat = { mean: number; min: number; max: number; n: number };
export type StatsGroup = {
  dials: { routine_strength: number; exception_rate: { slot: number; day: number } };
  n: number;
  realized: Record<string, FieldStat>;
};
export type RedrawHistogram = Record<"0" | "1" | "2" | "3" | "4" | "5", number>;
export type DonorStats = {
  n: number;
  /** Redraw counts above 5 are counted in "5". */
  redraws_histogram: RedrawHistogram;
  redraws_histogram_by_kind: { weekday: RedrawHistogram; weekend: RedrawHistogram };
  adjusted_share: number;
  work_end_repairs: number;
};
export type DayStats = {
  n: number;
  anchor_adjusted_share: number;
  /** Keys "0".."5" and "6+". */
  jitter_redraws_histogram: Record<string, number>;
};
export type StatsReport = { manifests: number; groups: StatsGroup[]; donors: DonorStats; days: DayStats };

const emptyHist = (): RedrawHistogram => ({ "0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0 });

/** Flattens numbers recursively into dot paths; null/absent values are skipped; the `dials` echo is excluded. */
function flatten(value: unknown, path: string, out: Map<string, number>): void {
  if (typeof value === "number") {
    if (Number.isFinite(value)) out.set(path, value);
  } else if (value !== null && typeof value === "object" && !Array.isArray(value)) {
    for (const [k, v] of Object.entries(value)) {
      if (path === "" && k === "dials") continue;
      flatten(v, path === "" ? k : `${path}.${k}`, out);
    }
  }
}

function realizedFields(m: TimelineManifest): Map<string, number> {
  const out = new Map<string, number>();
  flatten(m.realized, "", out);
  const days = m.days.length;
  if (days > 0) {
    for (const [role, count] of Object.entries(m.realized.shrunk)) out.set(`shrunk.${role}_per_day`, count / days);
  }
  return out;
}

const share = (a: number, b: number): number => (b === 0 ? 0 : a / b);

export function summarizeManifests(ms: TimelineManifest[]): StatsReport {
  const byKey = new Map<string, { dials: StatsGroup["dials"]; items: TimelineManifest[] }>();
  for (const m of ms) {
    const d = m.inputs.dials;
    const dials = { routine_strength: d.routine_strength, exception_rate: { slot: d.exception_rate.slot, day: d.exception_rate.day } };
    const key = `${dials.routine_strength}|${dials.exception_rate.slot}|${dials.exception_rate.day}`;
    const g = byKey.get(key) ?? { dials, items: [] };
    g.items.push(m);
    byKey.set(key, g);
  }
  const groups: StatsGroup[] = [...byKey.values()]
    .sort(
      (a, b) =>
        a.dials.routine_strength - b.dials.routine_strength ||
        a.dials.exception_rate.slot - b.dials.exception_rate.slot ||
        a.dials.exception_rate.day - b.dials.exception_rate.day,
    )
    .map(({ dials, items }) => {
      const acc = new Map<string, number[]>();
      for (const m of items) for (const [k, v] of realizedFields(m)) (acc.get(k) ?? acc.set(k, []).get(k)!).push(v);
      const realized: Record<string, FieldStat> = {};
      for (const k of [...acc.keys()].sort()) {
        const xs = acc.get(k)!;
        realized[k] = { mean: xs.reduce((s, x) => s + x, 0) / xs.length, min: Math.min(...xs), max: Math.max(...xs), n: xs.length };
      }
      return { dials, n: items.length, realized };
    });

  const hist = emptyHist();
  const byKind = { weekday: emptyHist(), weekend: emptyHist() };
  let donorN = 0;
  let adjusted = 0;
  let repairs = 0;
  const jitter: Record<string, number> = { "0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6+": 0 };
  let dayN = 0;
  let anchorAdjusted = 0;
  for (const m of ms) {
    for (const kind of ["weekday", "weekend"] as const) {
      const p = m.donor_adjustments[kind];
      const bucket = String(Math.min(p.redraws, 5)) as keyof RedrawHistogram;
      hist[bucket] += 1;
      byKind[kind][bucket] += 1;
      donorN += 1;
      if (p.adjustments.length > 0) adjusted += 1;
      repairs += p.adjustments.filter((a) => a.item === "work_end").length;
    }
    for (const d of m.days) {
      dayN += 1;
      if (d.anchor_adjustments.length > 0) anchorAdjusted += 1;
      jitter[d.jitter_redraws >= 6 ? "6+" : String(d.jitter_redraws)]! += 1;
    }
  }
  return {
    manifests: ms.length,
    groups,
    donors: {
      n: donorN,
      redraws_histogram: hist,
      redraws_histogram_by_kind: byKind,
      adjusted_share: share(adjusted, donorN),
      work_end_repairs: repairs,
    },
    days: { n: dayN, anchor_adjusted_share: share(anchorAdjusted, dayN), jitter_redraws_histogram: jitter },
  };
}
