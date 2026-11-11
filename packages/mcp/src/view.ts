import type { ClipCatalog, ScheduleDay, Timeline } from "@vb/core";

function dayOf(t: Timeline, day: number): ScheduleDay {
  const found = t.schedule.days.find((d) => d.day_index === day);
  if (!found) throw new Error(`day ${day} is not in timeline ${t.id} (1..${t.schedule.days.length})`);
  return found;
}

const orDash = (xs: readonly string[]): string => (xs.length > 0 ? xs.join(",") : "-");

/**
 * One header line for the day, then one line per slot:
 * `slot_id window role node clip_id(s) relaxations`. A clip missing from the catalog is marked `(missing)`.
 */
export function renderDay(t: Timeline, catalog: ClipCatalog, day: number): string {
  const d = dayOf(t, day);
  const header = [
    `day ${d.day_index} ${d.weekday} ${d.day_type} attempt ${d.attempt} wake ${d.wake} sleep ${d.sleep}`,
    ...d.exceptions.map((e) => `exception ${e.type} replaces ${e.replaces}${e.meal_place ? ` at ${e.meal_place}` : ""}`),
    ...d.dropped_slots.map((s) => `dropped ${s.role} (${s.reason})`),
  ];
  const lines = d.slots.map((s) => {
    const entry = t.slots[s.slot_id] ?? { clip_ids: [], relaxations: [] };
    const clips = entry.clip_ids.map((id) => (catalog.getClip(id) ? id : `${id}(missing)`));
    return [s.slot_id, `${s.window[0]}-${s.window[1]}`, s.role, s.spec.node, orDash(clips), orDash(entry.relaxations)].join(" ");
  });
  return [...header, ...lines].join("\n");
}

/** One line per day: index, weekday, day type and filled/total slots. */
export function renderOverview(t: Timeline): string {
  const head = `timeline ${t.id}${t.name ? ` (${t.name})` : ""} person ${t.world.person_id} revision ${t.revision}`;
  const days = t.schedule.days.map((d) => {
    const filled = d.slots.filter((s) => (t.slots[s.slot_id]?.clip_ids.length ?? 0) > 0).length;
    return `day ${d.day_index} ${d.weekday} ${d.day_type} ${filled}/${d.slots.length} slots filled`;
  });
  return [head, ...days].join("\n");
}
