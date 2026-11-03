import type { Composition, TimelineBlock } from "@vb/core";

const pad2 = (n: number) => String(n).padStart(2, "0");

function parts(sec: number): [number, number, number] {
  const s = Math.max(0, Math.floor(sec));
  return [Math.floor(s / 3600), Math.floor((s % 3600) / 60), s % 60];
}

/** `HH:MM:SS`, floored to whole seconds. */
function clock(sec: number): string {
  const [h, m, s] = parts(sec);
  return `${pad2(h)}:${pad2(m)}:${pad2(s)}`;
}

/** `H:MM:SS`, floored to whole seconds. */
function total(sec: number): string {
  const [h, m, s] = parts(sec);
  return `${h}:${pad2(m)}:${pad2(s)}`;
}

const range = (b: TimelineBlock) => `${clock(b.composed_start)}–${clock(b.composed_end)}`;

/** Text timetable (spec 8.2): S0, B0, S1, B1, ..., SN. Each event gets its own `S<slot>` line. */
export function renderTimetable(comp: Composition, timeline: TimelineBlock[]): string {
  const totalSec = timeline.length > 0 ? timeline[timeline.length - 1]!.composed_end : 0;
  const lines = [`rev ${comp.revision} · 총 ${total(totalSec)}`];
  for (let slot = 0; slot < comp.slots.length; slot++) {
    const events = timeline.filter((b) => b.role === "event" && b.slot === slot);
    if (events.length === 0) {
      lines.push(`S${slot}  (비어 있음)`);
    } else {
      for (const e of events) {
        lines.push(`S${slot}  ${range(e)}  ${e.category.padEnd(8)}  evt:${e.ref_id ?? ""}  ${e.summary}`);
      }
    }
    const base = timeline.find((b) => b.role === "base" && b.base_index === slot);
    if (base) lines.push(`B${slot}  ${range(base)}  ${"base".padEnd(8)}  ${base.summary}`);
  }
  return lines.join("\n");
}
