import {
  bindingAccepts,
  formatClock,
  layoutBlocks,
  parseClock,
  rngFor,
  TimelineError,
  transitionBlocks,
  type BlockPlacement,
  type BlockPlanEntry,
  type Clip,
  type ClipCatalog,
  type DayType,
  type ScheduleDay,
  type SlotOverride,
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
const MEAL_ROLES = new Set(["breakfast", "lunch", "dinner"]);

/** B18: whole-word, case-insensitive word sets matched against a clip's own L3 sentence (`activity`). */
const EATING = new Set(
  "eat eats eating ate meal meals breakfast lunch dinner supper snack snacks cook cooks cooking cooked restaurant canteen cafeteria diner eatery noodle noodles dumpling dumplings rice bun buns fry fries frying pancake pancakes food sushi sashimi".split(" "),
);
const COMMUTE = new Set("commute commuting travel travels traveling depart departs leave leaves return returns mtr bus train taxi scooter station".split(" "));
/** B19: extra words a home slot rejects (outdoor, shopping and arrival scenes). */
const HOME_REJECT = new Set(
  [...COMMUTE, ..."walk walks walking street streets market campus university library mall store shop shopping supermarket plaza plazas navigate navigates outdoor outdoors tram elevator arrive arrives arrival journey departure returning returned arriving arrived stall stalls vendor vendors transit".split(" ")],
);
/** B19: words a work slot rejects (errands, commutes, food runs). */
const WORK_REJECT = new Set([...COMMUTE, ..."tram exit exits departure supermarket mall store shopping food".split(" ")]);
/** B19: a meal run must contain one of these. */
const EAT_ACT = new Set("eat eats eating ate dine dines dining consume consumes consuming cook cooks cooking cooked buffet".split(" "));
const DRIVE = new Set("drive drives driving car".split(" "));
const hasWord = (text: string, set: ReadonlySet<string>): boolean => (text.toLowerCase().match(/[a-z0-9]+/g) ?? []).some((w) => set.has(w));

/**
 * B18/B19: the one content predicate per slot role. Meal slots need an EATING word in the clip's activity;
 * every other slot rejects one; home slots (home_*, chores, or a home/kitchen node) reject HOME_REJECT words
 * and work slots (work_am, work_pm) reject WORK_REJECT words.
 */
function contentOk(world: World, slot: Slot, c: Clip): boolean {
  if (MEAL_ROLES.has(slot.role)) return hasWord(c.activity, EATING);
  if (hasWord(c.activity, EATING)) return false;
  if (slot.role === "work_am" || slot.role === "work_pm") return !hasWord(c.activity, WORK_REJECT);
  const kind = world.nodes[slot.spec.node]?.kind;
  const home = slot.role.startsWith("home_") || slot.role === "chores" || kind === "home" || kind === "kitchen";
  return !(home && hasWord(c.activity, HOME_REJECT));
}

/**
 * B19: trims a meal run to one clip before the first EAT_ACT clip through one after the last, and drops
 * drive/car clips. No EAT_ACT clip: empty.
 */
function trimMeal<T extends { clip: Clip }>(run: readonly T[]): T[] {
  const idx = run.flatMap((c, i) => (hasWord(c.clip.activity, EAT_ACT) ? [i] : []));
  if (idx.length === 0) return [];
  return run.slice(Math.max(0, idx[0]! - 1), idx[idx.length - 1]! + 2).filter((c) => !hasWord(c.clip.activity, DRIVE));
}

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
export type DayLike = Pick<ScheduleDay, "day_index" | "day_type" | "trip" | "exceptions"> & Partial<Pick<ScheduleDay, "slots">>;
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

const isWeekend = (date: string) => [0, 6].includes(new Date(`${date}T00:00:00Z`).getUTCDay());

/** B12: an exercise day that gets a date with gym footage scores this much more. */
const GYM_BONUS_S = H(10);
/** B12: a late day whose date's first work is at or after 11:00, or a plain workday whose is earlier, scores this much more. */
const LATE_BONUS_S = H(1);

/**
 * B12: max-weight assignment of rows to columns (Hungarian method, O(n²(n+m))). `value[i][j]` is the
 * worth of giving column j to row i; a row may also stay unassigned (worth 0), and a column serves
 * at most one row. Returns the column of each row, or null. Deterministic: with equal totals the
 * result depends only on the row and column order. Values must be safe integers for an exact optimum.
 */
export function maxWeightAssignment(value: readonly (readonly number[])[]): (number | null)[] {
  const n = value.length;
  if (n === 0) return [];
  const real = value[0]!.length;
  // n dummy columns of worth 0 let any row stay unassigned and make the problem n × (m + n), m + n ≥ n.
  const m = real + n;
  const cost = (i: number, j: number) => (j < real ? -value[i]![j]! : 0);
  // 1-based potentials u (rows), v (columns); p[j]: row matched to column j (0: none); way: augmenting path.
  const u = new Array<number>(n + 1).fill(0);
  const v = new Array<number>(m + 1).fill(0);
  const p = new Array<number>(m + 1).fill(0);
  const way = new Array<number>(m + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Array<number>(m + 1).fill(Infinity);
    const used = new Array<boolean>(m + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0]!;
      let delta = Infinity;
      let j1 = 0;
      for (let j = 1; j <= m; j++) {
        if (used[j]) continue;
        const cur = cost(i0 - 1, j - 1) - u[i0]! - v[j]!;
        if (cur < minv[j]!) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j]! < delta) {
          delta = minv[j]!;
          j1 = j;
        }
      }
      for (let j = 0; j <= m; j++) {
        if (used[j]) {
          u[p[j]!] = u[p[j]!]! + delta;
          v[j] = v[j]! - delta;
        } else minv[j] = minv[j]! - delta;
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0]!;
      p[j0] = p[j1]!;
      j0 = j1;
    } while (j0 !== 0);
  }
  const out = new Array<number | null>(n).fill(null);
  for (let j = 1; j <= real; j++) if (p[j] !== 0) out[p[j]! - 1] = j - 1;
  return out;
}

/** B12: placed seconds of a block plan: per block the planned footage, cut at the block length as layoutBlocks does. */
function plannedSeconds(day: ScheduleDay, plan: readonly BlockPlanEntry[]): number {
  const len = new Map<string, number>(day.slots.map((s) => [s.slot_id, windowSeconds(s)[1] - windowSeconds(s)[0]] as const));
  for (const tr of transitionBlocks(day)) len.set(tr.transition_id, tr.span[1] - tr.span[0]);
  const got = new Map<string, number>();
  for (const e of plan) got.set(e.slot_id, (got.get(e.slot_id) ?? 0) + e.source_end_s - e.source_start_s);
  let total = 0;
  for (const [id, s] of got) total += Math.min(s, len.get(id) ?? 0);
  return total;
}

/**
 * B12 score s(d, x) in whole seconds: the placed seconds of planDay(d, x) (before any spare-date fill),
 * plus 10 h when an exercise day gets a date with gym footage, plus 1 h when a late day gets a date
 * whose first work is at or after 11:00 or a plain workday one whose first work is earlier. A day
 * without slots (or transitions) plans nothing and scores its bonuses only.
 */
function dateScore(day: DayLike, date: RealDate, world: World, gymNode: string | undefined): number {
  const full = day.slots !== undefined && "transitions" in day ? (day as ScheduleDay) : undefined;
  let s = full === undefined ? 0 : Math.round(plannedSeconds(full, planDay(full, date, world)));
  if (gymNode !== undefined && date.clips.some((c) => bindingAccepts(world, gymNode, c.clip))) s += GYM_BONUS_S;
  if (poolOf(day) === "work" && date.first_work_s !== null) {
    const late = day.day_type === "workday_late";
    if ((date.first_work_s >= LATE_FIRST_WORK_S) === late) s += LATE_BONUS_S;
  }
  return s;
}

/**
 * B5 step 3 with B12: one real date per synthetic day, each date used once. Trip days take their
 * trip's day_group_sequence entry at `trip.index`. The work days (plain or late) and the rest days
 * (leave days included) are each solved as one max-weight assignment over their pool:
 * - work days: Mon–Fri work dates (tier 1), then Sat/Sun work dates (tier 2, B7);
 * - rest days: rest dates (tier 1), then partial work dates (tier 2, fix ruling 4).
 * The worth of (day, date) is Q + P·[tier 1] + s(d, x) (see dateScore), with P larger than any
 * possible score total and Q larger than any tier and score total, so the assignment is lexicographic:
 * first as many days with a date as possible, then as many tier-1 dates as possible, then the most
 * score. Ties go by row and column order, which is a permutation of days and dates drawn with
 * rngFor(seed, "block_dates"). A day whose pool runs out gets `date: null`.
 */
export function assignDates(days: readonly DayLike[], dates: ReadonlyMap<string, RealDate>, world: World): DateAssignment[] {
  const out = new Map<number, DateAssignment>();
  const byIndex = [...days].sort((a, b) => a.day_index - b.day_index);
  const all = [...dates.values()].sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
  const rng = rngFor(world.seed, "block_dates");
  const shuffle = <T>(xs: T[]): T[] => {
    for (let i = xs.length - 1; i > 0; i--) {
      const j = rng.int(0, i + 1);
      [xs[i], xs[j]] = [xs[j]!, xs[i]!];
    }
    return xs;
  };

  const usedTrip = new Set<string>();
  for (const day of byIndex.filter((d) => poolOf(d) === "trip")) {
    const d = world.node_binding[day.trip!.node]?.day_group_sequence?.[day.trip!.index];
    const date = d !== undefined && !usedTrip.has(d) ? d : null;
    if (date !== null) usedTrip.add(date);
    out.set(day.day_index, { day_index: day.day_index, day_type: day.day_type, pool: "trip", date });
  }

  for (const pool of ["work", "rest"] as const) {
    const rows = shuffle(byIndex.filter((d) => poolOf(d) === pool));
    const tier1 = pool === "work" ? all.filter((x) => x.kind === "work" && !isWeekend(x.date)) : all.filter((x) => x.kind === "rest");
    const tier2 = pool === "work" ? all.filter((x) => x.kind === "work" && isWeekend(x.date)) : all.filter((x) => x.kind === "partial");
    const cols = shuffle([...tier1.map((x) => ({ x, tier1: true })), ...tier2.map((x) => ({ x, tier1: false }))]);
    if (rows.length === 0) continue;
    const scores = rows.map((day) => {
      const gym = exerciseNodeOf(day, world);
      return cols.map((c) => dateScore(day, c.x, world, gym));
    });
    const maxScore = Math.max(0, ...scores.flat());
    const P = rows.length * maxScore + 1;
    const Q = rows.length * (P + maxScore) + 1;
    if (!Number.isSafeInteger(rows.length * (Q + P + maxScore))) throw new Error("assignDates: date scores too large for an exact assignment");
    const pick = maxWeightAssignment(scores.map((r) => r.map((s, j) => Q + (cols[j]!.tier1 ? P : 0) + s)));
    rows.forEach((day, i) => {
      const j = pick[i];
      out.set(day.day_index, { day_index: day.day_index, day_type: day.day_type, pool, date: j === null || j === undefined ? null : cols[j]!.x.date });
    });
  }
  return days.map((d) => out.get(d.day_index)!);
}

/** B15: the rest-day hobby role; its slots are filled from clips like the evening activities. */
const HOBBY_ROLE = "hobby";

/** B10: the roles of the world's evening activities, with the node each one uses. */
const eveningNodes = (world: World): Map<string, string> => new Map((world.dials.evening_activities ?? []).map((a) => [a.role, a.node]));

/** B10: the node of the day's exercise slot (the evening activity bound to the NextMe person), if any. */
function exerciseNodeOf(day: DayLike, world: World): string | undefined {
  const person = personSourceOf(world);
  return (day.slots ?? []).find((s) => eveningNodes(world).has(s.role) && world.node_binding[s.spec.node]?.person_source === person)?.spec.node;
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
 * `reserved` (B22): seconds of a slot's window already taken by pinned clips.
 */
export function planDay(day: ScheduleDay, date: RealDate | undefined, world: World, reserved: ReadonlyMap<string, number> = new Map()): BlockPlanEntry[] {
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
  const meals = clips.filter((c) => mealAccepts(world, day, c.clip) && hasWord(c.clip.activity, EATING));
  const lunchStart = meals.find((c) => c.start >= H(11) && c.start < H(15))?.start ?? H(13);
  const inRange = (from: number, to: number) => (c: RealClip) => c.start >= from && c.start < to;

  /**
   * B10: the date's longest gym run (gap <= 15 min; ties: the earlier one) and the transit clips right
   * before it (each at most 15 min before the next clip of the chain), kept for the transition block
   * that ends at the slot. No gym clips: nothing.
   */
  const exerciseTransit = new Map<string, RealClip[]>();
  function exerciseCands(slot: Slot): RealClip[] {
    const gym = clips.filter((c) => bindingAccepts(world, slot.spec.node, c.clip));
    const runs: RealClip[][] = [];
    for (const c of gym) {
      const last = runs[runs.length - 1];
      if (last !== undefined && c.start - last[last.length - 1]!.end <= RUN_GAP_S) last.push(c);
      else runs.push([c]);
    }
    const run = runs.reduce<RealClip[] | undefined>((best, r) => (best === undefined || sum(r) > sum(best) ? r : best), undefined);
    if (run === undefined) return [];
    const before: RealClip[] = [];
    let next = run[0]!.start;
    for (const c of clips.filter((x) => x.end <= run[0]!.start && transitAccepts(world, x.clip)).reverse()) {
      if (next - c.end > RUN_GAP_S) break;
      before.unshift(c);
      next = c.start;
    }
    // The transit is not a gym clip: it goes to the transition block right before the exercise slot.
    exerciseTransit.set(slot.slot_id, before);
    return run;
  }

  /** Real-phase candidates of a slot, before removing used clips; `run` keeps only the first run. */
  function phase(slot: Slot): { cands: RealClip[]; run: boolean } {
    const bound = (c: RealClip) => bindingAccepts(world, slot.spec.node, c.clip);
    // The exercise slot is found by binding (as exerciseNodeOf does), not by its role name.
    if (eveningNodes(world).has(slot.role) && world.node_binding[slot.spec.node]?.person_source === personSourceOf(world)) {
      return { cands: exerciseCands(slot), run: false };
    }
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
        // hobby (B15: eveningFill) and any other role stay empty here.
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
    const free = cands.filter((c) => !used.has(c.clip.clip_id) && contentOk(world, slot, c.clip));
    const [ws, we] = windowSeconds(slot);
    const picked = run ? firstRun(free) : free;
    // B22: seconds already taken by pinned clips come off the window.
    take(slot.slot_id, MEAL_ROLES.has(slot.role) ? trimMeal(picked) : picked, we - ws - (reserved.get(slot.slot_id) ?? 0));
  }

  // B10: the transit right before a gym run fills the transition into its exercise slot.
  for (const [slotId, before] of exerciseTransit) {
    const tr = transitionBlocks(day).find((x) => x.to_slot === slotId && x.span[1] > x.span[0]);
    if (tr === undefined || !planned.has(slotId)) continue;
    take(tr.transition_id, before.filter((c) => !used.has(c.clip.clip_id)), tr.span[1] - tr.span[0]);
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
    const gymEnds = [...exerciseTransit.keys()].flatMap((id) => (planned.get(id) ?? []).map((c) => c.end));
    // B10: on an exercise day the commute home is the last transition into a home slot, after the gym run too.
    const eveningTr = gymEnds.length > 0 ? spans.filter((tr) => kinds.get(tr.to_slot) === "home").sort((a, b) => a.span[0] - b.span[0]).at(-1) : between("work", "home").at(-1);
    const lastWorkEnd = Math.max(...work.map((c) => c.end), ...gymEnds);
    const homeMorning = clips.filter((c) => home(c) && c.start < firstAway);
    const windows: [TransitionBlock | undefined, number, number][] = [
      [between("home", "work")[0], homeMorning.length > 0 ? Math.max(...homeMorning.map((c) => c.end)) : -Infinity, work[0]!.start],
      [eveningTr, lastWorkEnd, clips.find((c) => home(c) && c.start >= lastWorkEnd)?.start ?? Infinity],
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

/** home: B6; home_fallback: B6b (also `fallback: true`, kept for compatibility); topup: B13. */
export type FillKind = "home" | "home_fallback" | "topup";
/** A block filled (or topped up) from a spare date: real-time distance of the first clip and placed seconds. */
export type Fill = { kind: FillKind; day_index: number; slot_id: string; role: string; from_date: string; delta_min: number; seconds: number; fallback?: true };
/** @deprecated the B6 name of Fill. */
export type HomeFill = Fill;

/**
 * B6: every non-trip day's home slot (node kind home or kitchen) with no planned footage is filled from
 * a spare date: not assigned to a day, not a trip date, and serving one day only (it may fill several
 * home blocks of that day, and the date a day used last is preferred for its next ones). Candidates are
 * the date's unused home clips (home meal clips for meal slots) starting within ±180 min of the window
 * start; the smallest |Δt| wins, then the longer run, then the earlier date. The block takes that clip
 * and the following run of candidates (gap ≤ 15 min) until it is full.
 * B6b: every home_morning B6 left empty gets a last-resort fill (`kind: "home_fallback"`, `fallback: true`)
 * from a spare date not yet taken: its home clips that start before its first non-home clip, at any
 * clock time (the start of that date's footage; user decision 2026-11-13), no time window; the date whose home run from the first such clip is longest wins (ties: earlier
 * date); the block takes that run until it is full.
 * Order (the scarce pre-departure morning footage goes to home_morning first): B6 for the home_morning
 * slots of all days, then B6b for all of them, then B6 for the other home slots. Within each phase days
 * go in index order, slots in window order.
 */
export function homeFill(
  days: readonly ScheduleDay[],
  plan: readonly BlockPlanEntry[],
  assignments: readonly DateAssignment[],
  dates: ReadonlyMap<string, RealDate>,
  world: World,
): { entries: BlockPlanEntry[]; fills: Fill[] } {
  const assigned = new Set(assignments.flatMap((a) => (a.date === null ? [] : [a.date])));
  const spares = [...dates.values()].filter((d) => d.kind !== "trip" && !assigned.has(d.date)).sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
  const planned = new Set(plan.map((e) => e.slot_id));
  const taken = new Set<string>(); // spare dates already serving a day
  const own = new Map<number, RealDate>(); // the spare date a day used last
  const used = new Set<string>(); // spare-date clips already used (a spare date serves one day)
  const entries: BlockPlanEntry[] = [];
  const fills: Fill[] = [];
  const byIndex = [...days].sort((a, b) => a.day_index - b.day_index).filter((d) => d.trip === undefined);
  const emptyHome = (day: ScheduleDay) =>
    day.slots.filter((s) => slotKind(world, s) === "home" && !planned.has(s.slot_id)).sort((a, b) => windowSeconds(a)[0] - windowSeconds(b)[0]);
  const place = (slot: Slot, run: readonly RealClip[]): number => {
    const [ws, we] = windowSeconds(slot);
    let filled = 0;
    for (const c of run) {
      if (filled >= we - ws) break;
      entries.push({ slot_id: slot.slot_id, clip_id: c.clip.clip_id, source_start_s: c.clip.source_start_s, source_end_s: c.clip.source_end_s });
      used.add(c.clip.clip_id);
      filled += c.end - c.start;
    }
    return Math.min(filled, we - ws);
  };

  function b6(day: ScheduleDay, slot: Slot): void {
    const [ws] = windowSeconds(slot);
    const meal = MEAL_ROLES.has(slot.role);
    /** The best run start of `d` for this slot: smallest |Δt|, then the longer run, then the earlier clip. */
    const best = (d: RealDate) => {
      const pool = d.clips.filter((c) => isHome(world, c.clip) && (!meal || c.clip.category === "meal") && contentOk(world, slot, c.clip) && !used.has(c.clip.clip_id));
      let pick: { run: RealClip[]; delta: number; footage: number } | undefined;
      pool.forEach((c, i) => {
        const delta = Math.abs(c.start - ws);
        if (delta > FILL_WINDOW_S) return;
        let run = [c];
        for (const n of pool.slice(i + 1)) {
          if (n.start - run[run.length - 1]!.end > RUN_GAP_S) break;
          run.push(n);
        }
        if (meal) run = trimMeal(run);
        if (run.length === 0) return;
        const footage = sum(run);
        if (pick === undefined || delta < pick.delta || (delta === pick.delta && footage > pick.footage)) pick = { run, delta, footage };
      });
      return pick;
    };
    let chosen: { date: RealDate; run: RealClip[]; delta: number } | undefined;
    const mine = own.get(day.day_index);
    const minePick = mine === undefined ? undefined : best(mine);
    if (mine !== undefined && minePick !== undefined) chosen = { date: mine, ...minePick };
    else {
      let top: { date: RealDate; run: RealClip[]; delta: number; footage: number } | undefined;
      for (const d of spares) {
        if (taken.has(d.date)) continue;
        const b = best(d);
        if (b !== undefined && (top === undefined || b.delta < top.delta || (b.delta === top.delta && b.footage > top.footage))) top = { date: d, ...b };
      }
      chosen = top;
    }
    if (chosen === undefined) return;
    own.set(day.day_index, chosen.date);
    taken.add(chosen.date.date);
    const seconds = place(slot, chosen.run);
    fills.push({ kind: "home", day_index: day.day_index, slot_id: slot.slot_id, role: slot.role, from_date: chosen.date.date, delta_min: Math.round(chosen.delta / 60), seconds });
  }

  function b6b(day: ScheduleDay, slot: Slot): void {
    const [ws] = windowSeconds(slot);
    let top: { date: RealDate; run: RealClip[]; footage: number } | undefined;
    for (const d of spares) {
      if (taken.has(d.date)) continue;
      const firstAway = d.clips.find((c) => !isHome(world, c.clip));
      const pool = d.clips.filter((c) => isHome(world, c.clip) && (firstAway === undefined || c.start < firstAway.start) && contentOk(world, slot, c.clip));
      if (pool.length === 0) continue;
      const run = [pool[0]!];
      for (const n of pool.slice(1)) {
        if (n.start - run[run.length - 1]!.end > RUN_GAP_S) break;
        run.push(n);
      }
      const footage = sum(run);
      if (top === undefined || footage > top.footage) top = { date: d, run, footage };
    }
    if (top === undefined) return;
    own.set(day.day_index, top.date);
    taken.add(top.date.date);
    const seconds = place(slot, top.run);
    fills.push({ kind: "home_fallback", day_index: day.day_index, slot_id: slot.slot_id, role: slot.role, from_date: top.date.date, delta_min: Math.round(Math.abs(top.run[0]!.start - ws) / 60), seconds, fallback: true });
  }

  const morning = (s: Slot) => s.role === "home_morning";
  for (const day of byIndex) for (const slot of emptyHome(day).filter(morning)) b6(day, slot);
  const filledSlots = () => new Set(fills.map((f) => f.slot_id));
  for (const day of byIndex) {
    const done = filledSlots();
    for (const slot of emptyHome(day).filter((s) => morning(s) && !done.has(s.slot_id))) b6b(day, slot);
  }
  for (const day of byIndex) for (const slot of emptyHome(day).filter((s) => !morning(s))) b6(day, slot);
  return { entries, fills };
}

/** B13: a work_am, work_pm or home slot with at least this much of its window unfilled is topped up. */
const TOPUP_MIN_S = 10 * 60;

/**
 * B13: after the primary plan, B6 and B6b, every non-trip day's work_am, work_pm and non-meal home
 * slot (node kind home or kitchen) whose window still has ≥ 10 min unfilled is topped up from a spare
 * date (not assigned to a day, not a trip date). A spare date serves one synthetic day only, shared
 * with B6/B6b (`prior` fills say which day each spare date already serves). Candidates are the
 * date's unused clips that the slot's node binding accepts (no meal clips for work slots) whose real
 * start is within ±180 min of the unfilled part's start (window start + placed seconds); each starts a
 * run of the following candidates (gap ≤ 15 min). The run that places the most whole minutes wins, then a
 * date already serving the day, then the smaller |Δt|, then the earlier date. The slot takes the run
 * until full; its entries come after the slot's earlier ones, so layoutBlocks places them right after
 * the primary clips. Days go in index order, slots in window order.
 */
export function topUpFill(
  days: readonly ScheduleDay[],
  plan: readonly BlockPlanEntry[],
  assignments: readonly DateAssignment[],
  dates: ReadonlyMap<string, RealDate>,
  world: World,
  prior: readonly Fill[],
): { entries: BlockPlanEntry[]; fills: Fill[] } {
  const assigned = new Set(assignments.flatMap((a) => (a.date === null ? [] : [a.date])));
  const spares = [...dates.values()].filter((d) => d.kind !== "trip" && !assigned.has(d.date)).sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
  const owner = new Map(prior.map((f) => [f.from_date, f.day_index] as const)); // spare date -> the day it serves
  const usedClips = new Set(plan.map((e) => e.clip_id));
  const placed = new Map<string, number>();
  for (const e of plan) placed.set(e.slot_id, (placed.get(e.slot_id) ?? 0) + e.source_end_s - e.source_start_s);
  const entries: BlockPlanEntry[] = [];
  const fills: Fill[] = [];

  for (const day of [...days].sort((a, b) => a.day_index - b.day_index)) {
    if (day.trip !== undefined) continue;
    const slots = day.slots
      .filter((s) => !MEAL_ROLES.has(s.role) && (s.role === "work_am" || s.role === "work_pm" || slotKind(world, s) === "home"))
      .sort((a, b) => windowSeconds(a)[0] - windowSeconds(b)[0]);
    for (const slot of slots) {
      const [ws, we] = windowSeconds(slot);
      const filled = Math.min(placed.get(slot.slot_id) ?? 0, we - ws);
      const need = we - ws - filled;
      if (need < TOPUP_MIN_S) continue;
      const at = ws + filled;
      const work = slot.role === "work_am" || slot.role === "work_pm";
      const accepts = (c: RealClip) => bindingAccepts(world, slot.spec.node, c.clip) && (!work || c.clip.category !== "meal") && contentOk(world, slot, c.clip) && !usedClips.has(c.clip.clip_id);
      let top: { date: RealDate; run: RealClip[]; seconds: number; own: boolean; delta: number } | undefined;
      for (const d of spares) {
        const by = owner.get(d.date);
        if (by !== undefined && by !== day.day_index) continue;
        const own = by === day.day_index;
        const pool = d.clips.filter(accepts);
        pool.forEach((c, i) => {
          const delta = Math.abs(c.start - at);
          if (delta > FILL_WINDOW_S) return;
          const run = [c];
          for (const n of pool.slice(i + 1)) {
            if (sum(run) >= need || n.start - run[run.length - 1]!.end > RUN_GAP_S) break;
            run.push(n);
          }
          const seconds = Math.min(sum(run), need);
          // Whole minutes first, so a few seconds more never beat the day's own date or a closer start.
          const [mins, topMins] = [Math.round(seconds / 60), top === undefined ? -1 : Math.round(top.seconds / 60)];
          const better =
            top === undefined ||
            mins > topMins ||
            (mins === topMins && (Number(own) > Number(top.own) || (own === top.own && delta < top.delta)));
          if (better) top = { date: d, run, seconds, own, delta };
        });
      }
      if (top === undefined) continue;
      owner.set(top.date.date, day.day_index);
      for (const c of top.run) {
        entries.push({ slot_id: slot.slot_id, clip_id: c.clip.clip_id, source_start_s: c.clip.source_start_s, source_end_s: c.clip.source_end_s });
        usedClips.add(c.clip.clip_id);
      }
      placed.set(slot.slot_id, (placed.get(slot.slot_id) ?? 0) + sum(top.run));
      fills.push({ kind: "topup", day_index: day.day_index, slot_id: slot.slot_id, role: slot.role, from_date: top.date.date, delta_min: Math.round(top.delta / 60), seconds: top.seconds });
    }
  }
  return { entries, fills };
}

export type EveningTake = { day_index: number; slot_id: string; role: string; take: string; day_group: string; seconds: number };

/**
 * B14: the participant of an Ego-Exo4D take. Catalog clips carry no participant field, so it is the
 * take-name prefix `<site>_<number>` (e.g. `upenn_0331` of `upenn_0331_Guitar_2_4`) that the ingest
 * writes into the activity as `<task> (<take_name>)`; without one, the take is its own participant.
 */
export function participantOf(c: Clip): string {
  const m = /\(([A-Za-z]+_\d+)_[^()]*\)\s*$/.exec(c.activity);
  return m !== null ? m[1]! : `${c.source.dataset}:${c.source.video_id}`;
}

/**
 * B15: role `hobby` (rest-day slot, e.g. node hobby_1) is filled the same way, but from the node's
 * binding with any dataset, and a unit is one clip (not a whole video), so a few clips of one video
 * spread over the hobby days, each with its own day_group.
 *
 * B10 with B14: every evening-activity slot whose node is not bound to the NextMe person (hobby_2:
 * Ego-Exo4D) and that has no planned footage takes whole catalog takes accepted by its binding. Takes
 * are the binding's clips grouped by video, in an order shuffled with rngFor(seed, "activity", node). The
 * takes are dealt round-robin over the node's slots (days in index order, slots in window order), one
 * take per slot per round, until every slot is full or no take is left, so the takes spread evenly.
 * A slot takes a take of the participant it already has (participantOf) first; otherwise, of the
 * participant that no other day uses yet and that has the most takes left (ties: shuffled order). A
 * take is used once and its day_group is never used on another day. The block takes a take's clips
 * from its start (source order) until it is full.
 */
export function eveningFill(
  days: readonly ScheduleDay[],
  plan: readonly BlockPlanEntry[],
  catalog: ClipCatalog,
  world: World,
  /** B17: roles whose slots take exactly one whole clip each (the slot is then sized to the clip). */
  fitRoles: ReadonlySet<string> = new Set(),
): { entries: BlockPlanEntry[]; takes: EveningTake[] } {
  const person = personSourceOf(world);
  const roles = new Set([...eveningNodes(world).keys(), HOBBY_ROLE]);
  const planned = new Set(plan.map((e) => e.slot_id));
  const groupDay = new Map<string, number>(); // day_group -> the day that uses it
  const dayOfSlot = new Map(days.flatMap((d) => d.slots.map((s) => [s.slot_id, d.day_index] as const)));
  for (const e of plan) {
    const c = catalog.getClip(e.clip_id);
    if (c !== undefined && !groupDay.has(c.day_group)) groupDay.set(c.day_group, dayOfSlot.get(e.slot_id)!);
  }
  const catalogClips = catalog.clips();
  /** The node's takes in shuffled order. */
  const takesOf = (node: string, perClip: boolean): Clip[][] => {
    const byTake = new Map<string, Clip[]>();
    for (const c of deOverlap(catalogClips.filter((x) => x.person_source !== person && bindingAccepts(world, node, x)))) {
      const key = perClip ? c.clip_id : `${c.source.dataset}:${c.source.video_id}`;
      byTake.set(key, [...(byTake.get(key) ?? []), c]);
    }
    const cmp = (a: Clip, b: Clip) => a.source_start_s - b.source_start_s || (a.clip_id < b.clip_id ? -1 : 1);
    if (perClip) {
      // B15: clips of one video stay in source order (a build progresses over the days); videos are shuffled.
      const byVideo = new Map<string, Clip[]>();
      for (const [, [c]] of byTake) byVideo.set(`${c!.source.dataset}:${c!.source.video_id}`, []);
      for (const [, [c]] of byTake) byVideo.get(`${c!.source.dataset}:${c!.source.video_id}`)!.push(c!);
      const videos = [...byVideo.keys()].sort();
      const vr = rngFor(world.seed, "hobby");
      for (let i = videos.length - 1; i > 0; i--) {
        const j = vr.int(0, i + 1);
        [videos[i], videos[j]] = [videos[j]!, videos[i]!];
      }
      return videos.flatMap((v) => byVideo.get(v)!.sort(cmp).map((c) => [c]));
    }
    const list = [...byTake.keys()].sort().map((k) => byTake.get(k)!.sort(cmp));
    const rng = rngFor(world.seed, "activity", node);
    for (let i = list.length - 1; i > 0; i--) {
      const j = rng.int(0, i + 1);
      [list[i], list[j]] = [list[j]!, list[i]!];
    }
    return list;
  };

  type Open = { day: number; slot: Slot; len: number; filled: number; participant?: string; entries: BlockPlanEntry[]; takes: EveningTake[] };
  const open: Open[] = [];
  for (const day of [...days].sort((a, b) => a.day_index - b.day_index)) {
    const slots = day.slots.filter((s) => roles.has(s.role) && world.node_binding[s.spec.node]?.person_source !== person && !planned.has(s.slot_id));
    for (const slot of slots.sort((a, b) => windowSeconds(a)[0] - windowSeconds(b)[0])) {
      const [ws, we] = windowSeconds(slot);
      open.push({ day: day.day_index, slot, len: we - ws, filled: 0, entries: [], takes: [] });
    }
  }
  const usedTakes = new Set<string>();
  for (const node of [...new Set(open.map((o) => o.slot.spec.node))].sort()) {
    const slots = open.filter((o) => o.slot.spec.node === node);
    const perClip = slots.every((o) => o.slot.role === HOBBY_ROLE);
    const takes = takesOf(node, perClip);
    const keyOf = (take: Clip[]) => (perClip ? take[0]!.clip_id : `${take[0]!.source.dataset}:${take[0]!.source.video_id}`);
    const usable = (o: Open) => (take: Clip[]) => {
      const group = take[0]!.day_group;
      return !usedTakes.has(keyOf(take)) && (!groupDay.has(group) || groupDay.get(group) === o.day);
    };
    /** B14: the next take for slot `o`: its participant's first, else the free participant with the most takes left. */
    const pick = (o: Open): Clip[] | undefined => {
      const left = takes.filter(usable(o));
      if (left.length === 0) return undefined;
      if (perClip) return left[0];
      const same = o.participant === undefined ? undefined : left.find((t) => participantOf(t[0]!) === o.participant);
      if (same !== undefined) return same;
      const others = new Set(slots.filter((x) => x.day !== o.day && x.participant !== undefined).map((x) => x.participant!));
      const count = new Map<string, number>();
      for (const t of left) count.set(participantOf(t[0]!), (count.get(participantOf(t[0]!)) ?? 0) + 1);
      let best: Clip[] | undefined;
      for (const t of left) {
        if (best === undefined) {
          best = t;
          continue;
        }
        const p = participantOf(t[0]!);
        const q = participantOf(best[0]!);
        const key = (x: string) => [Number(!others.has(x)), count.get(x)!] as const;
        const [pf, pc] = key(p);
        const [qf, qc] = key(q);
        if (pf > qf || (pf === qf && pc > qc)) best = t;
      }
      return best;
    };
    for (let progress = true; progress; ) {
      progress = false;
      for (const o of slots) {
        if (o.filled >= o.len) continue;
        const take = pick(o);
        if (take === undefined) continue;
        progress = true;
        if (perClip && fitRoles.has(o.slot.role)) {
          // B17: one whole, untrimmed clip; the slot is sized to it afterwards.
          const c = take[0]!;
          usedTakes.add(keyOf(take));
          groupDay.set(c.day_group, o.day);
          o.participant = participantOf(c);
          o.entries.push({ slot_id: o.slot.slot_id, clip_id: c.clip_id, source_start_s: c.source_start_s, source_end_s: c.source_end_s });
          o.takes.push({ day_index: o.day, slot_id: o.slot.slot_id, role: o.slot.role, take: c.source.video_id, day_group: c.day_group, seconds: c.source_end_s - c.source_start_s });
          o.filled = o.len;
          continue;
        }
        usedTakes.add(keyOf(take));
        groupDay.set(take[0]!.day_group, o.day);
        o.participant = participantOf(take[0]!);
        let seconds = 0;
        for (const c of take) {
          if (o.filled >= o.len) break;
          o.entries.push({ slot_id: o.slot.slot_id, clip_id: c.clip_id, source_start_s: c.source_start_s, source_end_s: c.source_end_s });
          const len = c.source_end_s - c.source_start_s;
          seconds += Math.min(len, o.len - o.filled);
          o.filled += len;
        }
        o.takes.push({ day_index: o.day, slot_id: o.slot.slot_id, role: o.slot.role, take: take[0]!.source.video_id, day_group: take[0]!.day_group, seconds });
      }
    }
  }
  return { entries: open.flatMap((o) => o.entries), takes: open.flatMap((o) => o.takes) };
}

export type EveningSource = { role: string; source_date_or_take: string; seconds: number };

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
  /** B10 and B15: what filled the day's activity blocks (exercise, hobby_2, hobby): the real date or the take, with placed seconds. */
  activities: EveningSource[];
  /** @deprecated alias of `activities` (the B10 name), the same list. */
  evening: EveningSource[];
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
  /** B6, B6b and B13: blocks filled or topped up from spare dates, by `kind`. */
  fills: Fill[];
  /** @deprecated alias of `fills` (the B6 name), the same list. */
  home_fills: Fill[];
  /** B22: the pins applied, in the order given. */
  pins: PinApplied[];
};

/** B17: a fitted slot: its whole clip and the whole-minute length the slot is sized to. */
export type FittedSlot = { day_index: number; slot_id: string; role: string; clip_id: string; seconds: number; duration_min: number };

export type ComposeResult = {
  /** B17: the fitted slots (empty unless fit_roles is given). */
  fitted: FittedSlot[];
  assignments: DateAssignment[];
  plan: BlockPlanEntry[];
  placements: BlockPlacement[];
  summary: ComposeSummary;
};

/** B22: pin a video span into one day's slot. */
export type Pin = { day_index: number; role: string; video_id: string; from_s: number; to_s: number };
export type PinApplied = Pin & { slot_id: string; day_group: string; clips: number; seconds: number };
type ResolvedPin = { pin: Pin; slot: Slot; clips: Clip[] };

/** B22: validates the pins against the schedule and the catalog and resolves each to its slot and clips (source order). */
export function resolvePins(t: Timeline, catalog: ClipCatalog, pins: readonly Pin[]): ResolvedPin[] {
  const seen = new Set<string>();
  return pins.map((pin) => {
    const label = `pin day ${pin.day_index} ${pin.role} ${pin.video_id} [${pin.from_s}, ${pin.to_s}]`;
    const day = t.schedule.days.find((d) => d.day_index === pin.day_index);
    if (day === undefined) throw new Error(`${label}: unknown day`);
    const slot = day.slots.find((s) => s.role === pin.role);
    if (slot === undefined) throw new Error(`${label}: day ${pin.day_index} has no ${pin.role} slot`);
    const clips = deOverlap(catalog.clips().filter((c) => c.source.video_id === pin.video_id))
      .filter((c) => c.source_start_s >= pin.from_s && c.source_end_s <= pin.to_s)
      .sort((a, b) => a.source_start_s - b.source_start_s || (a.clip_id < b.clip_id ? -1 : 1));
    if (clips.length === 0) throw new Error(`${label}: the span matches no clips`);
    for (const c of clips) {
      if (seen.has(c.clip_id)) throw new Error(`${label}: clip ${c.clip_id} is pinned twice`);
      seen.add(c.clip_id);
    }
    return { pin, slot, clips };
  });
}

/** A catalog without the pinned clips (B22: they are in no other candidate pool). */
function withoutClips(catalog: ClipCatalog, ids: ReadonlySet<string>): ClipCatalog {
  return {
    clips: () => catalog.clips().filter((c) => !ids.has(c.clip_id)),
    getClip: (id) => catalog.getClip(id),
    getAnnotations: (d, v) => catalog.getAnnotations(d, v),
  };
}

const round3 = (x: number) => Math.round(x * 1000) / 1000;

/** B5 steps 2–5 for a timeline whose schedule is already generated: dates, plan, placements and summary. */
export function composeBlocks(t: Timeline, fullCatalog: ClipCatalog, opts: { fit_roles?: readonly string[]; pins?: readonly Pin[] } = {}): ComposeResult {
  const world = t.world;
  const resolved = resolvePins(t, fullCatalog, opts.pins ?? []);
  const catalog = resolved.length === 0 ? fullCatalog : withoutClips(fullCatalog, new Set(resolved.flatMap((r) => r.clips.map((c) => c.clip_id))));
  const dates = collectDates(catalog, world);
  const assignments = assignDates(t.schedule.days, dates, world);
  const dateOf = new Map(assignments.map((a) => [a.day_index, a.date]));
  // B22: a pinned clip's day_group belongs to the pinned day only.
  const groupOwner = new Map<string, number>();
  for (const r of resolved) {
    for (const g of new Set(r.clips.map((c) => c.day_group))) {
      const other = assignments.find((a) => a.date === g && a.day_index !== r.pin.day_index);
      if (other !== undefined) throw new Error(`pin day ${r.pin.day_index} ${r.pin.role} ${r.pin.video_id}: day_group ${g} is day ${other.day_index}'s assigned real date`);
      const prev = groupOwner.get(g);
      if (prev !== undefined && prev !== r.pin.day_index) throw new Error(`pins on days ${prev} and ${r.pin.day_index} share day_group ${g}`);
      groupOwner.set(g, r.pin.day_index);
    }
  }
  const spareDates: ReadonlyMap<string, RealDate> = groupOwner.size === 0 ? dates : new Map([...dates].filter(([d]) => !groupOwner.has(d)));
  const pinnedEntries: BlockPlanEntry[] = resolved.flatMap((r) =>
    r.clips.map((c) => ({ slot_id: r.slot.slot_id, clip_id: c.clip_id, source_start_s: c.source_start_s, source_end_s: c.source_end_s })),
  );
  const reserved = new Map<string, number>();
  for (const e of pinnedEntries) reserved.set(e.slot_id, (reserved.get(e.slot_id) ?? 0) + e.source_end_s - e.source_start_s);
  const pinApplied: PinApplied[] = resolved.map((r) => ({
    ...r.pin,
    slot_id: r.slot.slot_id,
    day_group: r.clips[0]!.day_group,
    clips: r.clips.length,
    seconds: r.clips.reduce((s, c) => s + c.source_end_s - c.source_start_s, 0),
  }));
  // Pinned entries come first, so layoutBlocks places them first in their slot and the normal fill follows.
  const plan = [
    ...pinnedEntries,
    ...t.schedule.days.flatMap((day) => {
      const date = dateOf.get(day.day_index);
      return planDay(day, date === null || date === undefined ? undefined : dates.get(date), world, reserved);
    }),
  ];
  const filled = homeFill(t.schedule.days, plan, assignments, spareDates, world);
  // B13: top-up entries come after the slot's primary (and B6) entries, so layoutBlocks places them right after.
  const topped = topUpFill(t.schedule.days, [...plan, ...filled.entries], assignments, spareDates, world, filled.fills);
  const fitRoles = new Set(opts.fit_roles ?? []);
  const evening = eveningFill(t.schedule.days, [...plan, ...filled.entries, ...topped.entries], catalog, world, fitRoles);
  const slotOf = new Map(t.schedule.days.flatMap((d) => d.slots.map((s) => [s.slot_id, { day: d.day_index, role: s.role }] as const)));
  const fitted = evening.takes
    .filter((x) => fitRoles.has(x.role))
    .map((x): FittedSlot => {
      const clip_id = evening.entries.find((e) => e.slot_id === x.slot_id)!.clip_id;
      return { day_index: slotOf.get(x.slot_id)!.day, slot_id: x.slot_id, role: x.role, clip_id, seconds: x.seconds, duration_min: Math.ceil(x.seconds / 60) };
    });
  const full = [...plan, ...filled.entries, ...topped.entries, ...evening.entries];
  const placements = layoutBlocks(t, fullCatalog, full);
  const fills = [...filled.fills, ...topped.fills];
  return { assignments, plan: full, placements, fitted, summary: { ...summarize(t, assignments, placements, world), fills, home_fills: fills, pins: pinApplied } };
}

function summarize(t: Timeline, assignments: readonly DateAssignment[], placements: readonly BlockPlacement[], world: World): Omit<ComposeSummary, "fills" | "home_fills" | "pins"> {
  const days_by_type: Partial<Record<DayType, number>> = {};
  for (const d of t.schedule.days) days_by_type[d.day_type] = (days_by_type[d.day_type] ?? 0) + 1;
  const dateOf = new Map(assignments.map((a) => [a.day_index, a.date]));
  const person = personSourceOf(world);
  const eveningRoles = new Set([...eveningNodes(world).keys(), HOBBY_ROLE]);
  const per_day = t.schedule.days.map((day): DaySummary => {
    const ps = placements.filter((p) => p.day_index === day.day_index);
    const filled = new Set(ps.map((p) => p.slot_id));
    // Only the wearer's own clips have a real time (youtube / Ego-Exo4D clips have no wall clock).
    const real = ps.flatMap((p) => {
      const start = realStartOf(p.clip);
      if (start === undefined) return [];
      const at = start - p.clip.source_start_s;
      return [[at + p.source_start_s, at + p.source_end_s] as const];
    });
    const activities = day.slots
      .filter((s) => eveningRoles.has(s.role))
      .flatMap((s) => {
        const by = new Map<string, number>();
        for (const p of ps.filter((x) => x.slot_id === s.slot_id)) {
          const src = s.role === HOBBY_ROLE ? p.clip.clip_id : p.clip.person_source === person ? p.clip.day_group : p.clip.source.video_id;
          by.set(src, (by.get(src) ?? 0) + p.synthetic_end_s - p.synthetic_start_s);
        }
        return [...by].map(([source_date_or_take, seconds]) => ({ role: s.role, source_date_or_take, seconds: round3(seconds) }));
      });
    return {
      day_index: day.day_index,
      day_type: day.day_type,
      real_date: dateOf.get(day.day_index) ?? null,
      footage_hours: round3(ps.reduce((s, p) => s + p.synthetic_end_s - p.synthetic_start_s, 0) / 3600),
      first_real: real.length > 0 ? formatClock(Math.min(...real.map((r) => r[0])) / 60) : null,
      last_real: real.length > 0 ? formatClock(Math.max(...real.map((r) => r[1])) / 60) : null,
      empty_roles: day.slots.filter((s) => !filled.has(s.slot_id)).map((s) => s.role),
      activities,
      evening: activities,
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

/** B17: the roles that can be fitted (the per-clip hobby path). */
export const FIT_ROLES: ReadonlySet<string> = new Set([HOBBY_ROLE]);

export type FitResult = ComposeResult & {
  timeline: Timeline;
  /** B17: the overrides written into the world (empty when nothing was fitted). */
  slot_overrides: SlotOverride[];
};

const fitError = (overrides: readonly SlotOverride[], cause: unknown): Error => {
  const list = overrides.map((o) => `day ${o.day_index} ${o.role} ${o.duration_min} min`).join(", ");
  const why = cause instanceof Error ? cause.message : String(cause);
  const message = `fit_roles: the schedule cannot hold the fitted slots (${list}): ${why}`;
  // Keep the machine-readable code and details of a core failure.
  if (cause instanceof TimelineError) return new TimelineError(cause.code, message, { ...cause.details, fitted: overrides.map((o) => ({ ...o })) });
  return new Error(message);
};

/**
 * B17: composes, sizes each fit-role slot to its one whole clip (length rounded up to whole minutes), writes the
 * lengths into world.dials.slot_overrides, regenerates the schedule with `regenerate` and composes again. The
 * overrides apply last in core, so the days that have fit slots (and so the clip assignment) do not change; that
 * is checked. With no fit_roles, or nothing to fit, this is composeBlocks on `t`.
 */
export function composeBlocksFit(
  t: Timeline,
  catalog: ClipCatalog,
  fitRoles: readonly string[],
  regenerate: (world: World) => Timeline,
  pins: readonly Pin[] = [],
): FitResult {
  const bad = fitRoles.filter((r) => !FIT_ROLES.has(r));
  if (bad.length > 0) throw new Error(`fit_roles: unsupported role ${bad.map((r) => JSON.stringify(r)).join(", ")}; supported: ${[...FIT_ROLES].join(", ")}`);
  const first = composeBlocks(t, catalog, { fit_roles: fitRoles, pins });
  if (fitRoles.length === 0 || first.fitted.length === 0) return { ...first, timeline: t, slot_overrides: [] };
  const computed: SlotOverride[] = first.fitted.map((f) => ({ day_index: f.day_index, role: f.role, duration_min: f.duration_min }));
  const keep = (t.world.dials.slot_overrides ?? []).filter((o) => !computed.some((c) => c.day_index === o.day_index && c.role === o.role));
  const slot_overrides = [...keep, ...computed].sort((a, b) => a.day_index - b.day_index || (a.role < b.role ? -1 : a.role > b.role ? 1 : 0));
  const world: World = { ...t.world, dials: { ...t.world.dials, slot_overrides } };
  let timeline: Timeline;
  try {
    timeline = regenerate(world);
  } catch (e) {
    throw fitError(computed, e);
  }
  const second = composeBlocks(timeline, catalog, { fit_roles: fitRoles, pins });
  const key = (r: ComposeResult) => JSON.stringify(r.fitted.map((f) => [f.day_index, f.slot_id, f.clip_id]));
  if (key(first) !== key(second)) throw new Error("fit_roles: the clip assignment changed after the schedule was regenerated with the slot overrides");
  const pinKey = (r: ComposeResult) => JSON.stringify(r.summary.pins.map((p) => [p.day_index, p.slot_id, p.video_id, p.clips, p.seconds]));
  if (pinKey(first) !== pinKey(second)) throw new Error("fit_roles: the pinned assignment changed after the schedule was regenerated with the slot overrides");
  return { ...second, timeline, slot_overrides };
}
