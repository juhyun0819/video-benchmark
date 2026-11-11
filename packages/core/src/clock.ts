export const TIME_OF_DAY = ["early_morning", "morning", "midday", "afternoon", "evening", "night"] as const;
export type TimeOfDay = (typeof TIME_OF_DAY)[number];

/** Half-open minute ranges [start, end) since 00:00; night runs past midnight to 28:00. */
export const TIME_OF_DAY_RANGES: Record<TimeOfDay, [number, number]> = {
  early_morning: [240, 420],
  morning: [420, 660],
  midday: [660, 840],
  afternoon: [840, 1020],
  evening: [1020, 1260],
  night: [1260, 1680],
};

/** Parses `HH:MM` (HH 00-27, MM 00-59) into minutes since 00:00. */
export function parseClock(s: string): number {
  const m = /^(\d{2}):(\d{2})$/.exec(s);
  if (!m) throw new Error(`invalid clock "${s}": expected HH:MM`);
  const h = Number(m[1]);
  const min = Number(m[2]);
  if (h > 27 || min > 59) throw new Error(`invalid clock "${s}": hours 00-27, minutes 00-59`);
  return h * 60 + min;
}

/** Formats minutes as `HH:MM`, rounding to the nearest minute and never wrapping past 24h. */
export function formatClock(min: number): string {
  const total = Math.round(min);
  const h = Math.floor(total / 60);
  const m = total - h * 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

/** Bucket containing the minute. Values < 240 count as the next day's night; values >= 1680 as the next morning. */
export function timeOfDayAt(min: number): TimeOfDay {
  let m = min;
  if (m < 240) m += 1440;
  else if (m >= 1680) m -= 1440;
  for (const tod of TIME_OF_DAY) {
    const [bs, be] = TIME_OF_DAY_RANGES[tod];
    if (m >= bs && m < be) return tod;
  }
  return "night";
}

/** True when the padded window [ws-pad, we+pad) overlaps the half-open bucket; "unknown" always passes. */
export function bucketOverlaps(tod: TimeOfDay | "unknown", window: [number, number], padMin = 0): boolean {
  if (tod === "unknown") return true;
  const [bs, be] = TIME_OF_DAY_RANGES[tod];
  const [ws, we] = window;
  return ws - padMin < be && we + padMin > bs;
}
