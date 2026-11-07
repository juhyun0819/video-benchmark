/** Event candidates shorter than this are too small for a timetable block. */
export const DEFAULT_MIN_EVENT_SEC = 60;

const HKT_OFFSET_SECONDS = 8 * 3600;
const PATTERN = /^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})$/;

/**
 * Epoch seconds for a "YYYY-MM-DD HH:MM:SS" wall-clock string in UTC+08:00 (HKT).
 * Independent of the process time zone. Throws on a malformed or non-existent time.
 */
export function parseHkt(s: string): number {
  const m = PATTERN.exec(s);
  if (!m) throw new Error(`Invalid HKT timestamp: ${JSON.stringify(s)}`);
  const [y, mo, d, h, mi, se] = m.slice(1).map(Number) as [number, number, number, number, number, number];
  const ms = Date.UTC(y, mo - 1, d, h, mi, se);
  const back = new Date(ms);
  if (
    back.getUTCFullYear() !== y ||
    back.getUTCMonth() !== mo - 1 ||
    back.getUTCDate() !== d ||
    back.getUTCHours() !== h ||
    back.getUTCMinutes() !== mi ||
    back.getUTCSeconds() !== se
  ) {
    throw new Error(`Invalid HKT timestamp: ${JSON.stringify(s)}`);
  }
  return ms / 1000 - HKT_OFFSET_SECONDS;
}
