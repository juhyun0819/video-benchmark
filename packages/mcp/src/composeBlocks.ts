import {
  bindingAccepts,
  formatClock,
  layoutBlocks,
  parseClock,
  rngFor,
  transitionBlocks,
  type BlockPlacement,
  type BlockPlanEntry,
  type Clip,
  type ClipCatalog,
  type DayType,
  type ScheduleDay,
  type Slot,
  type Timeline,
  type TransitionBlock,
  type World,
} from "@vb/core";

// B5: fills a generated schedule's blocks with the real footage of one real date per synthetic day
// (spec 2026-11-13-nextme-28day-blocks §3.2). Pure functions; the CLI only does I/O.

const H = (hours: number) => hours * 3600;
/** A work date has at least this much workplace footage. */
const WORK_DATE_MIN_S = H(1);
/** Late workdays prefer dates whose first workplace clip starts at or after this time. */
const LATE_FIRST_WORK_S = H(11);
/** Consecutive clips of a meal (or outing) run are at most this far apart. */
const RUN_GAP_S = 15 * 60;
/** B6: a home block takes spare-date footage whose real start is at most this far from the block start. */
const FILL_WINDOW_S = 180 * 60;
/** B6b: the last-resort home_morning fill only takes home clips that start before this time of day. */
const FALLBACK_BEFORE_S = H(13);
const MEAL_ROLES = new Set(["breakfast", "lunch", "dinner"]);

/** A catalog clip with its real start and end, in seconds since its date's 00:00. */
export type RealClip = { clip: Clip; start: number; end: number };

/** work: ≥ 60 min of workplace footage; rest: none; partial: some, but less; trip: a trip sequence date. */
export type DateKind = "work" | "rest" | "partial" | "trip";

export type RealDate = {
  date: string;
  kind: DateKind;
  /** The bound person's timed clips of the date, in real-time order. */
  clips: RealClip[];
  work_seconds: number;
  first_work_s: number | null;
  /** Home footage before the first non-home clip (ranking key a). */
  home_first: boolean;
  /** Total labeled footage (ranking key b). */
  labeled_seconds: number;
};

const DATE_RE = /^(\d{4})-(\d{2})-(\d{2})$/;
const WALL_RE = /^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})$/;

/** Seconds of the date's 00:00 on a naive (zone-free) clock, or undefined for a non-date. */
function midnightOf(date: string): number | undefined {
  const m = DATE_RE.exec(date);
  return m ? Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3])) / 1000 : undefined;
}

/**
 * Real start of a clip: `source.media.start_wallclock` + `source_start_s`, in seconds since the
 * 00:00 of the clip's day_group (or of the wall clock's own date when day_group is not a date).
 * Undefined when the recording has no wall clock.
 */
export function realStartOf(clip: Clip): number | undefined {
  const media = clip.source.media;
  if (media.kind !== "frames" || media.start_wallclock === undefined) return undefined;
  const w = WALL_RE.exec(media.start_wallclock);
  if (!w) return undefined;
  const [y, mo, d, h, mi, s] = w.slice(1).map(Number) as [number, number, number, number, number, number];
  const wall = Date.UTC(y, mo - 1, d, h, mi, s) / 1000;
  const base = midnightOf(clip.day_group) ?? Date.UTC(y, mo - 1, d) / 1000;
  return wall - base + clip.source_start_s;
}

/** Name of the world's first node of `kind`. */
const nodeOfKind = (world: World, kind: "home" | "workplace"): string | undefined =>
  Object.entries(world.nodes).find(([, def]) => def.kind === kind)?.[0];

/** The person source the world binds its home (B5: the wearer whose dates are used). */
export function personSourceOf(world: World): string {
  const home = nodeOfKind(world, "home");
  const person = home !== undefined ? world.node_binding[home]?.person_source : undefined;
  if (person === undefined) throw new Error("the world's home binding names no person_source; compose-blocks needs one");
  return person;
}

const tripDatesOf = (world: World): Set<string> =>
  new Set(Object.values(world.node_binding).flatMap((b) => b.day_group_sequence ?? []));

/** Accepted by the binding of the world's home node (home footage). */
const isHome = (world: World, c: Clip): boolean => {
  const home = nodeOfKind(world, "home");
  return home !== undefined && bindingAccepts(world, home, c);
};
const isWork = (world: World, c: Clip): boolean => {
  const work = nodeOfKind(world, "workplace");
  return work !== undefined && bindingAccepts(world, work, c);
};

/** The bound person's dates (day_groups), classified and ranked by B5 step 2; clips without a wall clock are left out. */
export function collectDates(catalog: ClipCatalog, world: World): Map<string, RealDate> {
  const person = personSourceOf(world);
  const trips = tripDatesOf(world);
  const groups = new Map<string, RealClip[]>();
  for (const clip of deOverlap(catalog.clips().filter((c) => c.person_source === person))) {
    const start = realStartOf(clip);
    if (start === undefined) continue;
    const rc = { clip, start, end: start + clip.source_end_s - clip.source_start_s };
    groups.set(clip.day_group, [...(groups.get(clip.day_group) ?? []), rc]);
  }
  const out = new Map<string, RealDate>();
  for (const date of [...groups.keys()].sort()) {
    const clips = groups.get(date)!.sort(byRealTime);
    const work = clips.filter((c) => isWork(world, c.clip));
    const work_seconds = sum(work);
    const firstAway = clips.find((c) => !isHome(world, c.clip));
    const kind: DateKind = trips.has(date) ? "trip" : work_seconds >= WORK_DATE_MIN_S ? "work" : work_seconds > 0 ? "partial" : "rest";
    out.set(date, {
      date,
      kind,
      clips,
      work_seconds,
      first_work_s: kind === "trip" || work.length === 0 ? null : work[0]!.start,
      home_first: clips.some((c) => isHome(world, c.clip) && (firstAway === undefined || c.start < firstAway.start)),
      labeled_seconds: sum(clips),
    });
  }
  return out;
}

/**
 * Fix ruling 2: real L3 clips can nest or overlap inside one video. Per video, the longest clip is
 * kept and every clip overlapping a kept one is dropped (ties: earlier start, then clip_id), so no
 * two clips that could be planned share footage.
 */
export function deOverlap(clips: readonly Clip[]): Clip[] {
  const byVideo = new Map<string, Clip[]>();
  for (const c of clips) {
    const key = `${c.source.dataset}:${c.source.video_id}`;
    byVideo.set(key, [...(byVideo.get(key) ?? []), c]);
  }
  const keep = new Set<string>();
  for (const group of byVideo.values()) {
    const kept: Clip[] = [];
    const longestFirst = [...group].sort(
      (a, b) =>
        b.source_end_s - b.source_start_s - (a.source_end_s - a.source_start_s) ||
        a.source_start_s - b.source_start_s ||
        (a.clip_id < b.clip_id ? -1 : a.clip_id > b.clip_id ? 1 : 0),
    );
    for (const c of longestFirst) {
      if (kept.some((k) => k.source_start_s < c.source_end_s && c.source_start_s < k.source_end_s)) continue;
      kept.push(c);
      keep.add(c.clip_id);
    }
  }
  return clips.filter((c) => keep.has(c.clip_id));
}

const byRealTime = (a: RealClip, b: RealClip) =>
  a.start - b.start || (a.clip.clip_id < b.clip.clip_id ? -1 : a.clip.clip_id > b.clip.clip_id ? 1 : 0);
const sum = (cs: readonly RealClip[]) => cs.reduce((s, c) => s + c.end - c.start, 0);

export type Pool = "work" | "rest" | "trip";
export type DayLike = Pick<ScheduleDay, "day_index" | "day_type" | "trip" | "exceptions">;
export type DateAssignment = { day_index: number; day_type: DayType; pool: Pool; date: string | null };

/**
 * Trip days take trip dates; workdays (plain or late) take work dates; leave days and the other rest
 * days take rest dates, then partial work dates once the rest dates run out (fix ruling 4).
 */
export function poolOf(day: DayLike): Pool {
  if (day.trip !== undefined) return "trip";
  if (day.exceptions.some((e) => e.type === "leave_day")) return "rest";
  return day.day_type === "workday" || day.day_type === "workday_late" ? "work" : "rest";
}

/** Best first: home footage before leaving, then more labeled footage, then the earlier date. */
const rank = (a: RealDate, b: RealDate) =>
  Number(b.home_first) - Number(a.home_first) || b.labeled_seconds - a.labeled_seconds || (a.date < b.date ? -1 : a.date > b.date ? 1 : 0);

const isWeekend = (date: string) => [0, 6].includes(new Date(`${date}T00:00:00Z`).getUTCDay());

/**
 * B5 step 3: one real date per synthetic day, each date used once. Trip days take their trip's
 * day_group_sequence entry at `trip.index`. Other days are visited in an order shuffled with
 * rngFor(seed, "block_dates") and take the best unused date of their pool; late workdays prefer
 * first work at or after 11:00, plain workdays earlier; when that side is empty, the day takes the
 * other side's date whose first work is closest to its preference (latest for late days, earliest
 * otherwise), each within the weekday dates first and the Sat/Sun dates only after (B7). Rest days fall back to partial dates. A day whose pool is empty gets `date: null`.
 */
export function assignDates(days: readonly DayLike[], dates: ReadonlyMap<string, RealDate>, world: World): DateAssignment[] {
  const used = new Set<string>();
  const out = new Map<number, DateAssignment>();
  const order = [...days];
  const rng = rngFor(world.seed, "block_dates");
  for (let i = order.length - 1; i > 0; i--) {
    const j = rng.int(0, i + 1);
    [order[i], order[j]] = [order[j]!, order[i]!];
  }
  const sorted = [...dates.values()].sort(rank);
  for (const day of order) {
    const pool = poolOf(day);
    let date: string | null = null;
    if (pool === "trip") {
      const d = world.node_binding[day.trip!.node]?.day_group_sequence?.[day.trip!.index];
      date = d !== undefined && !used.has(d) ? d : null;
    } else {
      const free = (kind: DateKind) => sorted.filter((x) => x.kind === kind && !used.has(x.date));
      if (pool === "work") {
        const late = day.day_type === "workday_late";
        // B7: real Mon-Fri dates first; Sat/Sun dates only once no weekday work date is left.
        const pick = (work: RealDate[]) => {
          const preferred = work.filter((x) => (x.first_work_s! >= LATE_FIRST_WORK_S) === late);
          const fallback = [...work].sort((a, b) => (late ? b.first_work_s! - a.first_work_s! : a.first_work_s! - b.first_work_s!) || rank(a, b));
          return (preferred[0] ?? fallback[0])?.date;
        };
        const work = free("work");
        date = pick(work.filter((x) => !isWeekend(x.date))) ?? pick(work.filter((x) => isWeekend(x.date))) ?? null;
      } else {
        date = (free("rest")[0] ?? free("partial")[0])?.date ?? null;
      }
    }
    if (date !== null) used.add(date);
    out.set(day.day_index, { day_index: day.day_index, day_type: day.day_type, pool, date });
  }
  return days.map((d) => out.get(d.day_index)!);
}

const windowSeconds = (s: Slot): [number, number] => [parseClock(s.window[0]) * 60, parseClock(s.window[1]) * 60];

/** Ruling 2: the places a meal slot accepts (any bound non-trip place; the trip's meal place on a trip day). */
function mealAccepts(world: World, day: ScheduleDay, c: Clip): boolean {
  if (day.trip !== undefined) return bindingAccepts(world, `${day.trip.node}:meal`, c);
  return (
    c.category === "meal" &&
    Object.entries(world.nodes).some(([name, def]) => def.kind !== "trip" && bindingAccepts(world, name, c))
  );
}

/** Ruling 2: a commute clip is a transit clip of a person bound to the world's places (verify's placeAccepts). */
function transitAccepts(world: World, c: Clip): boolean {
  if (world.node_binding["transit"] !== undefined) return bindingAccepts(world, "transit", c);
  const persons = new Set(Object.values(world.node_binding).flatMap((b) => (b.person_source !== undefined ? [b.person_source] : [])));
  return c.node === "transit" && (persons.size === 0 || persons.has(c.person_source));
}

/** The first run of `cs` (time-ordered): consecutive clips at most RUN_GAP_S apart. */
function firstRun(cs: readonly RealClip[]): RealClip[] {
  const run: RealClip[] = [];
  for (const c of cs) {
    if (run.length > 0 && c.start - run[run.length - 1]!.end > RUN_GAP_S) break;
    run.push(c);
  }
  return run;
}

type SlotKind = "home" | "work" | "other";
function slotKind(world: World, slot: Slot): SlotKind {
  const def = world.nodes[slot.spec.node];
  if (def?.kind === "home" || def?.kind === "kitchen") return "home";
  if (def?.kind === "workplace") return "work";
  return "other";
}

/**
 * B5 step 4–5: the block plan of one synthetic day from its real date. Meal slots are planned first,
 * then the other slots in window order, then the morning and evening commutes. Each slot takes the unused clips
 * of its real phase in real-time order until they fill its window (layoutBlocks trims the last); a
 * clip is used once. Only clips the world binds to the slot's place are used (ruling 2).
 */
export function planDay(day: ScheduleDay, date: RealDate | undefined, world: World): BlockPlanEntry[] {
  if (date === undefined) return [];
  const clips = date.clips;
  const used = new Set<string>();
  const planned = new Map<string, RealClip[]>();
  // Partial dates are used only by rest days (fix ruling 4), so they follow the rest-date rules.
  const rest = date.kind === "rest" || date.kind === "partial";

  const home = (c: RealClip) => isHome(world, c.clip);
  const away = clips.filter((c) => !home(c));
  const firstAway = away[0]?.start ?? Infinity;
  const lastAway = away[away.length - 1]?.start ?? -Infinity;
  const meals = clips.filter((c) => mealAccepts(world, day, c.clip));
  const lunchStart = meals.find((c) => c.start >= H(11) && c.start < H(15))?.start ?? H(13);
  const inRange = (from: number, to: number) => (c: RealClip) => c.start >= from && c.start < to;

  /** Real-phase candidates of a slot, before removing used clips; `run` keeps only the first run. */
  function phase(slot: Slot): { cands: RealClip[]; run: boolean } {
    const bound = (c: RealClip) => bindingAccepts(world, slot.spec.node, c.clip);
    switch (slot.role) {
      case "breakfast":
        return { cands: meals.filter(inRange(-Infinity, H(11))), run: true };
      case "lunch":
        return { cands: meals.filter(inRange(H(11), H(15))), run: true };
      case "dinner":
        return { cands: meals.filter(inRange(H(17), Infinity)), run: true };
      case "home_morning":
        // Fix ruling 3: on rest dates only the morning (before 12:00).
        return { cands: clips.filter((c) => bound(c) && c.start < firstAway && (!rest || c.start < H(12))), run: false };
      case "work_am":
      {
        const work = clips.filter((c) => bound(c) && c.clip.category !== "meal");
        const before = work.filter((c) => c.start < lunchStart);
        // B7: lunch before any work leaves work_am empty, so it takes the date's work clips from the start.
        return { cands: before.length > 0 ? before : work, run: false };
      }
      case "work_pm":
        return { cands: clips.filter((c) => bound(c) && c.clip.category !== "meal" && c.start >= lunchStart), run: false };
      case "home_afternoon":
      case "chores":
        return { cands: clips.filter((c) => bound(c) && inRange(H(12), H(17))(c)), run: false };
      case "home_evening":
        return { cands: clips.filter((c) => bound(c) && c.start >= (rest ? H(17) : lastAway)), run: false };
      case "lodging_morning":
        return { cands: clips.filter((c) => bound(c) && c.start < H(12)), run: false };
      case "lodging_evening":
        return { cands: clips.filter((c) => bound(c) && c.start >= H(17)), run: false };
      case "outing":
        return { cands: clips.filter(bound), run: true };
      case "outing_pm":
        return { cands: clips.filter((c) => bound(c) && c.start >= lunchStart), run: false };
      default:
        // hobby (spec §3.2.4) and any other role stay empty.
        return { cands: [], run: false };
    }
  }

  function take(id: string, cands: readonly RealClip[], length: number): void {
    const out: RealClip[] = [];
    let filled = 0;
    for (const c of cands) {
      if (filled >= length) break;
      out.push(c);
      used.add(c.clip.clip_id);
      filled += c.end - c.start;
    }
    if (out.length > 0) planned.set(id, out);
  }

  const slots = [...day.slots].sort((a, b) => windowSeconds(a)[0] - windowSeconds(b)[0]);
  const ordered = [...slots.filter((s) => MEAL_ROLES.has(s.role)), ...slots.filter((s) => !MEAL_ROLES.has(s.role))];
  for (const slot of ordered) {
    const { cands, run } = phase(slot);
    const free = cands.filter((c) => !used.has(c.clip.clip_id));
    const [ws, we] = windowSeconds(slot);
    take(slot.slot_id, run ? firstRun(free) : free, we - ws);
  }

  // Fix ruling 1: commute windows come from the date's real phases, not from what the slots planned.
  // Morning: after the real home_morning phase, before the first workplace clip. Evening: after the
  // last workplace clip (any category), before the first home clip after it (open end if none).
  const work = clips.filter((c) => isWork(world, c.clip));
  if (work.length > 0) {
    const kinds = new Map(day.slots.map((s) => [s.slot_id, slotKind(world, s)] as const));
    // Ruling 1 (empty spans drop their clips silently).
    const spans = transitionBlocks(day).filter((tr) => tr.span[1] > tr.span[0]);
    const between = (from: SlotKind, to: SlotKind) =>
      spans.filter((tr) => kinds.get(tr.from_slot) === from && kinds.get(tr.to_slot) === to).sort((a, b) => a.span[0] - b.span[0]);
    const lastWorkEnd = Math.max(...work.map((c) => c.end));
    const homeMorning = clips.filter((c) => home(c) && c.start < firstAway);
    const windows: [TransitionBlock | undefined, number, number][] = [
      [between("home", "work")[0], homeMorning.length > 0 ? Math.max(...homeMorning.map((c) => c.end)) : -Infinity, work[0]!.start],
      [between("work", "home").at(-1), lastWorkEnd, clips.find((c) => home(c) && c.start >= lastWorkEnd)?.start ?? Infinity],
    ];
    const transit = clips.filter((c) => transitAccepts(world, c.clip));
    // Only the day's first home→work and last work→home transitions are commutes; midday ones stay empty.
    for (const [tr, lo, hi] of windows) {
      if (tr === undefined) continue;
      const free = transit.filter((c) => !used.has(c.clip.clip_id) && c.start >= lo && c.start < hi);
      take(tr.transition_id, free, tr.span[1] - tr.span[0]);
    }
  }

  const order = [...ordered.map((s) => s.slot_id), ...transitionBlocks(day).map((x) => x.transition_id)];
  return order.flatMap((id) =>
    (planned.get(id) ?? []).map((c) => ({
      slot_id: id,
      clip_id: c.clip.clip_id,
      source_start_s: c.clip.source_start_s,
      source_end_s: c.clip.source_end_s,
    })),
  );
}

export type HomeFill = { day_index: number; slot_id: string; role: string; from_date: string; delta_min: number; seconds: number; fallback?: true };

/**
 * B6: every non-trip day's home slot (node kind home or kitchen) with no planned footage is filled from
 * a spare date: not assigned to a day, not a trip date, and serving one day only (it may fill several
 * home blocks of that day, and is preferred for the day's later ones). Days go in index order, slots in
 * window order. Candidates are the date's unused home clips (home meal clips for meal slots) starting
 * within ±180 min of the window start; the smallest |Δt| wins, then the longer run, then the earlier
 * date. The block takes that clip and the following run of candidates (gap ≤ 15 min) until it is full.
 * B6b: afterwards every home_morning still empty gets a last-resort fill (`fallback: true`; normal fills
 * leave it absent) from a spare date not yet taken: its home clips that start before its first non-home
 * clip and before 13:00, no time window; the date whose home run from the first such clip is longest wins
 * (ties: earlier date); the block takes that run until it is full.
 */
export function homeFill(
  days: readonly ScheduleDay[],
  plan: readonly BlockPlanEntry[],
  assignments: readonly DateAssignment[],
  dates: ReadonlyMap<string, RealDate>,
  world: World,
): { entries: BlockPlanEntry[]; fills: HomeFill[] } {
  const assigned = new Set(assignments.flatMap((a) => (a.date === null ? [] : [a.date])));
  const spares = [...dates.values()].filter((d) => d.kind !== "trip" && !assigned.has(d.date)).sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
  const planned = new Set(plan.map((e) => e.slot_id));
  const taken = new Set<string>(); // spare dates already serving a day
  const entries: BlockPlanEntry[] = [];
  const fills: HomeFill[] = [];

  for (const day of [...days].sort((a, b) => a.day_index - b.day_index)) {
    if (day.trip !== undefined) continue;
    const empty = day.slots.filter((s) => slotKind(world, s) === "home" && !planned.has(s.slot_id)).sort((a, b) => windowSeconds(a)[0] - windowSeconds(b)[0]);
    const used = new Set<string>();
    let own: RealDate | undefined;
    for (const slot of empty) {
      const [ws, we] = windowSeconds(slot);
      const meal = MEAL_ROLES.has(slot.role);
      /** The best run start of `d` for this slot: smallest |Δt|, then the longer run, then the earlier clip. */
      const best = (d: RealDate) => {
        const pool = d.clips.filter((c) => isHome(world, c.clip) && (!meal || c.clip.category === "meal") && !used.has(c.clip.clip_id));
        let pick: { run: RealClip[]; delta: number; footage: number } | undefined;
        pool.forEach((c, i) => {
          const delta = Math.abs(c.start - ws);
          if (delta > FILL_WINDOW_S) return;
          const run = [c];
          for (const n of pool.slice(i + 1)) {
            if (n.start - run[run.length - 1]!.end > RUN_GAP_S) break;
            run.push(n);
          }
          const footage = sum(run);
          if (pick === undefined || delta < pick.delta || (delta === pick.delta && footage > pick.footage)) pick = { run, delta, footage };
        });
        return pick;
      };
      let chosen: { date: RealDate; run: RealClip[]; delta: number } | undefined;
      const ownPick = own === undefined ? undefined : best(own);
      if (own !== undefined && ownPick !== undefined) chosen = { date: own, ...ownPick };
      else {
        let top: { date: RealDate; run: RealClip[]; delta: number; footage: number } | undefined;
        for (const d of spares) {
          if (taken.has(d.date)) continue;
          const b = best(d);
          if (b !== undefined && (top === undefined || b.delta < top.delta || (b.delta === top.delta && b.footage > top.footage))) top = { date: d, ...b };
        }
        chosen = top;
      }
      if (chosen === undefined) continue;
      own = chosen.date;
      taken.add(own.date);
      let filled = 0;
      for (const c of chosen.run) {
        if (filled >= we - ws) break;
        entries.push({ slot_id: slot.slot_id, clip_id: c.clip.clip_id, source_start_s: c.clip.source_start_s, source_end_s: c.clip.source_end_s });
        used.add(c.clip.clip_id);
        filled += c.end - c.start;
      }
      fills.push({ day_index: day.day_index, slot_id: slot.slot_id, role: slot.role, from_date: own.date, delta_min: Math.round(chosen.delta / 60), seconds: Math.min(filled, we - ws) });
    }
  }

  // B6b: last resort for home_morning slots B6 left empty.
  const filledSlots = new Set(fills.map((f) => f.slot_id));
  for (const day of [...days].sort((a, b) => a.day_index - b.day_index)) {
    if (day.trip !== undefined) continue;
    const morning = day.slots.filter((s) => s.role === "home_morning" && slotKind(world, s) === "home" && !planned.has(s.slot_id) && !filledSlots.has(s.slot_id));
    for (const slot of morning) {
      const [ws, we] = windowSeconds(slot);
      let top: { date: RealDate; run: RealClip[]; footage: number } | undefined;
      for (const d of spares) {
        if (taken.has(d.date)) continue;
        const firstAway = d.clips.find((c) => !isHome(world, c.clip));
        const pool = d.clips.filter((c) => isHome(world, c.clip) && c.start < FALLBACK_BEFORE_S && (firstAway === undefined || c.start < firstAway.start));
        if (pool.length === 0) continue;
        const run = [pool[0]!];
        for (const n of pool.slice(1)) {
          if (n.start - run[run.length - 1]!.end > RUN_GAP_S) break;
          run.push(n);
        }
        const footage = sum(run);
        if (top === undefined || footage > top.footage) top = { date: d, run, footage };
      }
      if (top === undefined) continue;
      taken.add(top.date.date);
      let filled = 0;
      for (const c of top.run) {
        if (filled >= we - ws) break;
        entries.push({ slot_id: slot.slot_id, clip_id: c.clip.clip_id, source_start_s: c.clip.source_start_s, source_end_s: c.clip.source_end_s });
        filled += c.end - c.start;
      }
      fills.push({ day_index: day.day_index, slot_id: slot.slot_id, role: slot.role, from_date: top.date.date, delta_min: Math.round(Math.abs(top.run[0]!.start - ws) / 60), seconds: Math.min(filled, we - ws), fallback: true });
    }
  }
  return { entries, fills };
}

export type DaySummary = {
  day_index: number;
  day_type: DayType;
  real_date: string | null;
  /** Placed (trimmed) footage. */
  footage_hours: number;
  /** Earliest / latest real clock time of the footage used (HH:MM, may pass 24:00). */
  first_real: string | null;
  last_real: string | null;
  /** Roles of the day's slots with no placement, in slot order. */
  empty_roles: string[];
};

export type ComposeSummary = {
  days_by_type: Partial<Record<DayType, number>>;
  footage_hours_total: number;
  /** Slot blocks with no placement over all days. */
  empty_blocks: number;
  /** Transition blocks that got commute footage. */
  commute_blocks: number;
  unassigned_days: { day_index: number; day_type: DayType; pool: Pool }[];
  per_day: DaySummary[];
  /** B6: home blocks filled from spare dates. */
  home_fills: HomeFill[];
};

export type ComposeResult = {
  assignments: DateAssignment[];
  plan: BlockPlanEntry[];
  placements: BlockPlacement[];
  summary: ComposeSummary;
};

const round3 = (x: number) => Math.round(x * 1000) / 1000;

/** B5 steps 2–5 for a timeline whose schedule is already generated: dates, plan, placements and summary. */
export function composeBlocks(t: Timeline, catalog: ClipCatalog): ComposeResult {
  const world = t.world;
  const dates = collectDates(catalog, world);
  const assignments = assignDates(t.schedule.days, dates, world);
  const dateOf = new Map(assignments.map((a) => [a.day_index, a.date]));
  const plan = t.schedule.days.flatMap((day) => {
    const date = dateOf.get(day.day_index);
    return planDay(day, date === null || date === undefined ? undefined : dates.get(date), world);
  });
  const filled = homeFill(t.schedule.days, plan, assignments, dates, world);
  const full = [...plan, ...filled.entries];
  const placements = layoutBlocks(t, catalog, full);
  return { assignments, plan: full, placements, summary: { ...summarize(t, assignments, placements), home_fills: filled.fills } };
}

function summarize(t: Timeline, assignments: readonly DateAssignment[], placements: readonly BlockPlacement[]): Omit<ComposeSummary, "home_fills"> {
  const days_by_type: Partial<Record<DayType, number>> = {};
  for (const d of t.schedule.days) days_by_type[d.day_type] = (days_by_type[d.day_type] ?? 0) + 1;
  const dateOf = new Map(assignments.map((a) => [a.day_index, a.date]));
  const per_day = t.schedule.days.map((day): DaySummary => {
    const ps = placements.filter((p) => p.day_index === day.day_index);
    const filled = new Set(ps.map((p) => p.slot_id));
    const real = ps.map((p) => {
      const at = realStartOf(p.clip)! - p.clip.source_start_s;
      return [at + p.source_start_s, at + p.source_end_s] as const;
    });
    return {
      day_index: day.day_index,
      day_type: day.day_type,
      real_date: dateOf.get(day.day_index) ?? null,
      footage_hours: round3(ps.reduce((s, p) => s + p.synthetic_end_s - p.synthetic_start_s, 0) / 3600),
      first_real: real.length > 0 ? formatClock(Math.min(...real.map((r) => r[0])) / 60) : null,
      last_real: real.length > 0 ? formatClock(Math.max(...real.map((r) => r[1])) / 60) : null,
      empty_roles: day.slots.filter((s) => !filled.has(s.slot_id)).map((s) => s.role),
    };
  });
  return {
    days_by_type,
    footage_hours_total: round3(per_day.reduce((s, d) => s + d.footage_hours, 0)),
    empty_blocks: per_day.reduce((n, d) => n + d.empty_roles.length, 0),
    commute_blocks: new Set(placements.filter((p) => /^d\d+-t\d+$/.test(p.slot_id)).map((p) => p.slot_id)).size,
    unassigned_days: assignments.filter((a) => a.date === null).map((a) => ({ day_index: a.day_index, day_type: a.day_type, pool: a.pool })),
    per_day,
  };
}
