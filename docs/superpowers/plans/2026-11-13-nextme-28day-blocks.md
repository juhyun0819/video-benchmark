# NextMe 28-day Block Fill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build one 28-day NextMe manifest by keeping the ATUS-driven schedule and filling every block with whole, contiguous real footage from one real NextMe date per synthetic day.

**Spec:** `docs/superpowers/specs/2026-11-13-nextme-28day-blocks-design.md` (approved 2026-11-13). Earlier: `docs/superpowers/specs/2026-11-12-nextme-28day-design.md` and plan `2026-11-12-nextme-28day.md` (Tasks 1–4 done).

**Tech:** Node 24, TS strict ESM, zod 3, vitest. No new deps.

## Global Constraints
- **Language:** English code, messages and tests; Korean docs.
- **Purity and determinism:**
  - `@vb/core` stays pure. Every draw uses `rngFor` with stable keys.
  - Existing worlds and the existing slot-matching path (`npm run generate`, auto_fill, layoutTimeline, buildTimelineManifest default, verifyTimeline default) must produce byte-identical results to HEAD. New options default to off.
- **Data:** fixtures are synthetic. Never commit `data/`.
- **Commits:** use `git -c user.name=juhyun0819 -c user.email=p3hyun2@gmail.com commit` with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never push.

## Decisions
- **B1. Fixed trip start.** `dials.trips[].start_day` is an optional int ≥ 1. When set, the trip starts on that synthetic day. There is no weekday match and no rng draw. It must fit within `length_days` and must not overlap another trip; otherwise WORLD_INVALID, raised from `WorldSchema` where possible. `replaces` is computed as today.
- **B2. Block placement (core).** A pure function `layoutBlocks(t, catalog, plan)` turns a block plan into placed clips. The plan gives an ordered list of `{ slot_id, clip_id, source_start_s, source_end_s }` for each slot and transition. Placement rules:
  - Clips go back to back from the block start (the slot window start, or the transition start) in the given order.
  - The last clip that would pass the block end is trimmed. Clips after it are dropped.
  - Trimmed spans are recorded.
- **B3. Manifest from placements.** `buildTimelineManifest` gains an optional `placements` argument.
  - When given, it uses those placements, which may be trimmed, instead of `layoutTimeline`.
  - Each manifest clip's `source_start_s`/`source_end_s` is the trimmed span.
  - Commute clips use a transition id `d<N>-t<k>` as `slot_id`.
  - `no_video_spans` is computed from the placements.
  - Without the argument, output is byte-identical to HEAD.
- **B4. Verifier block mode.** `verifyTimeline` gains an optional `{ mode: "block", placements }`. In block mode:
  - `time_of_day` is skipped and reported as pass with a note.
  - `slot_window` checks that each placement lies inside its window.
  - `clip_unique` and `day_group_single_day` are checked over placements. Transitions count as their day.
  - `trip` is checked as today.
  - The other rules are unchanged.
  - Without the option, the verifier is byte-identical.
- **B5. compose-blocks (mcp).** `npm run compose-blocks -- --world <path> [--data <dir>] [--config <dir>] [--out <dir>]`:
  1. Generate the schedule with the existing generator.
  2. **Real dates.** Use the clips in the catalog for the world's bound `person_source` (nextme800:wearer), grouped by `day_group`. The real wall-clock time of a clip is `source.media.start_wallclock` + `source_start_s`.
     - A work date has ≥ 60 min of `workplace` clips.
     - A rest date has none.
     - Trip dates are the trip binding's `day_group_sequence`, used in order on trip days.
     - Each date is used on at most one synthetic day.
  3. **Assignment.**
     - workday and workday_late days take work dates. Late days prefer dates whose first workplace clip starts ≥ 11:00. Plain workdays prefer earlier first-work dates.
     - hobby_day and home_day take rest dates.
     - leave_day takes a rest date.
     - Within a pool, rank dates by: (a) having home footage before the first non-home clip, then (b) total labeled minutes, descending.
     - Visit synthetic days in an order shuffled with `rngFor(seed, "block_dates")` so the best dates do not all land in week 1.
     - When a pool runs out, the day keeps no real date and all its blocks are unrecorded. Report this.
  4. **Real day phases.** Map each slot role to the real clips of the assigned date, in real-time order. "real lunch" is the first meal clip starting between 11:00 and 15:00; when there is none, split at 13:00.

     | role | real clips used |
     |---|---|
     | home_morning | home clips before the first non-home labeled clip |
     | breakfast | meal clips before 11:00 |
     | work_am | workplace clips (any category except meal) before the real lunch |
     | lunch | meal clips 11:00–15:00 |
     | work_pm | workplace clips (non-meal) after the real lunch |
     | dinner | meal clips at or after 17:00 |
     | home_afternoon, chores | home clips 12:00–17:00 |
     | home_evening | home clips at or after the last non-home clip (or at or after 17:00 on rest dates) |
     | lodging_morning | trip:lodging before 12:00 |
     | lodging_evening | trip:lodging at or after 17:00 |
     | outing | the first run of trip:outing clips |
     | outing_pm | trip:outing clips after the trip lunch |
     | trip breakfast/lunch/dinner | trip:meal by the same time rules |
     | hobby | none (left empty, spec §3.2.4) |

     A clip is used once.
  5. **Commute.** For each schedule transition from a home slot to a work slot (and back), place the date's transit clips that sit between those real phases. They go into the transition span as the block.
  6. **Output.** Write the timeline, a verifier report (block mode) and the manifest through the existing stores. Print a JSON summary: days by type, real date per synthetic day, footage hours per day, empty blocks, unassigned days.

## Review Focus
1. The default paths are byte-identical, which the existing golden tests and snapshot show.
2. Placements never leave their window, and the trimmed spans are recorded.
3. One real date per synthetic day, and trip days use the sequence in order.
4. Hobby slots stay empty.

---

### Task 1: Fixed trip start (B1)
**Files:**
- `packages/core/src/world.ts`
- `packages/core/src/schedule/decisions.ts`
- tests

Steps (TDD):
- [ ] Write tests:
  - `start_day: 19` with a Mon–Wed sequence places the trip on days 19–21 (Fri–Sun);
  - the exception record `day_range` is [19, 21] and `replaces` lists the base types;
  - `start_day` out of range or overlapping gives WORLD_INVALID;
  - worlds without it are unchanged (golden).
- [ ] Implement. Run `npm test && npm run typecheck`. Commit.

### Task 2: Block placement, manifest and verifier options (B2–B4)
**Files:**
- `packages/core/src/layout.ts` (or a new `blocks.ts`)
- `packages/core/src/manifest.ts`
- `packages/core/src/verify.ts`
- tests with synthetic fixtures

Steps (TDD):
- [ ] Write tests:
  - back-to-back placement from the block start;
  - trimming the last clip at the block end;
  - dropping the clips after it;
  - transition-block placement;
  - the manifest with placements records trimmed spans and computes no_video_spans;
  - verifier block mode skips time_of_day and catches a placement outside its window and a date used on two days;
  - default manifest and verifier outputs are unchanged (existing snapshot and golden).
- [ ] Implement. Run tests. Commit.

### Task 3: compose-blocks CLI (B5)
**Files:**
- `packages/mcp/src/composeBlocks.ts` (logic)
- `packages/mcp/src/composeBlocksCli.ts`
- `package.json` script `compose-blocks`
- tests with a synthetic catalog (a few real-like dates with home/transit/work/meal clips)

Steps (TDD):
- [ ] Write tests:
  - date classification and one-date-per-day;
  - late days get later first-work dates;
  - the role→phase mapping table, including the lunch split and the 13:00 fallback;
  - commute clips go into transitions;
  - hobby is empty;
  - trip days use the sequence in order;
  - an exhausted pool gives an unassigned day in the summary;
  - the end-to-end CLI writes a manifest whose block-mode verify passes.
- [ ] Implement. Run tests. Commit.

---

## Addendum (2026-11-13): home fill from spare dates

- **B6. Home fill.** After the primary plan, every slot whose `spec.node` is `home` and that received no footage gets filled from a spare date. This covers home_morning, home_afternoon, home_evening, chores, and meals at home.
  - **Spare dates:** dates not assigned to any synthetic day and not trip dates. Each spare date serves at most one synthetic day. It may fill several home blocks of that day.
  - **Candidates:** home-bound clips of a spare date whose real start is within ±90 min of the block's synthetic start (window start). Meal slots take only home meal clips. Other home slots take any home clip.
  - **Selection:**
    - Visit days in index order and their empty home slots in time order.
    - Pick the candidate clip with the smallest |Δt|. Break ties by more footage in the run, then by earlier date.
    - From that clip, take the contiguous run of the date's home clips (gap ≤ 15 min, real-time order) and fill until the block end.
    - A spare date chosen for a day is preferred for that day's later empty home slots.
  - **Bounds:** deterministic, and trip days are excluded.
  - **Summary:** add `home_fills: [{ day_index, slot_id, role, from_date, delta_min, seconds }]`.

### Task 4: Home fill from spare dates (B6)
**Files:**
- `packages/mcp/src/composeBlocks.ts`
- `packages/mcp/test/composeBlocks.test.ts`

Steps (TDD):
- [ ] Write tests:
  - an empty home_morning is filled from a spare date within ±90 min;
  - a candidate outside ±90 min is not used;
  - an assigned date is never used as a spare;
  - a spare date never serves two synthetic days, but can serve morning and evening of one day;
  - meal-at-home slots take only home meal clips;
  - non-home empty slots stay empty (hobby, work);
  - trip days are untouched;
  - block-mode verify passes;
  - the summary lists home_fills.
- [ ] Implement. Run `npm test && npm run typecheck`. Commit.
