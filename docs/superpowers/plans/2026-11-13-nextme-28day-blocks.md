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
  - **Candidates:** home-bound clips of a spare date whose real start is within ±180 min (widened from ±90 by the user, 2026-11-13) of the block's synthetic start (window start). Meal slots take only home meal clips. Other home slots take any home clip.
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

- **B6b. Home-morning last resort.** This applies after B6 to every home_morning that is still empty. Use a spare date not yet taken, under the same one-day-per-date rule. Its candidates are home clips that start before that date's first non-home labeled clip (any clock time; the 13:00 cap was dropped by the user, 2026-11-13), with no time window. Pick the date whose run is longest, breaking ties by earlier date. Record it in `home_fills` with `fallback: true`.

## Addendum 2 (2026-11-13): weekday dates, gym, hobby 2

- **B7. Weekday dates and work_am fallback.**
  - workday and workday_late days take real Mon–Fri dates first. Sat/Sun work dates are used only when the weekday pool runs out.
  - If the real lunch comes before the first workplace clip (so work_am would be empty), work_am takes the date's workplace clips from the start, in order, until its window is full. work_pm continues with the rest.
- **B8. Evening activities in the schedule (core).** A new optional dial is `dials.evening_activities: [{ node, role, position: "before_dinner" | "after_dinner", duration_min, weekdays?: Weekday[], count?: int, max_per_week?: int }]`. The default is [].
  - **Eligible days:** template `workday` days only. Late, trip, leave and exception days are excluded.
  - **Days chosen:**
    - With `weekdays`, every eligible day on those weekdays.
    - With `count`, the eligible days with the earliest work_pm window end. At most `max_per_week` per ISO week. Ties are broken by `rngFor(seed, "evening", node)`.
  - **Slot:** inserted after work_pm (before_dinner) or right after dinner (after_dinner).
  - **Window:** starts after the previous slot plus the gap-table travel, and lasts `duration_min`.
  - **Following slots:** they shift later by the overlap (a before_dinner activity pushes dinner, and an after_dinner one pushes home_evening). Nothing may pass sleep. If it would, shrink home_evening first, then the activity, to no less than 20 min.
  - **Record:** the slot spec uses the activity node's place, setting, category hobby and tags.
  - Existing worlds are byte-identical.
- **B9. Gym label.** `config/nextme_label_rules.l3.json` gains a place rule `gym` (venue_indoor, category hobby, tags ["gym"]) placed before workplace. Its words are gym, fitness center, fitness centre, workout, dumbbell, dumbbells, bench press, pull-up, cable machine, treadmill.
- **B10. compose-blocks for evening activities.**
  - Exercise days prefer real dates that have gym clips. The exercise block takes that date's gym run (contiguous, gap ≤ 15 min, including the transit just before it).
  - hobby_2 blocks take Ego-Exo4D clips bound to the hobby_2 node. Each take is used once. The take's `day_group` must not be used on another day.
  - Commute stays as now.
- **B11. Ego-Exo4D guitar category.** `config/categories.json` gains `hobby_guitar` (egoexo4d, scenario Music, keywords ["guitar"], clip hobby/venue_indoor/node "hobby", tags ["music_guitar"]) placed before `hobby_music`.

### Task 5: Weekday dates and work_am fallback (B7)
Files: `packages/mcp/src/composeBlocks.ts`, tests.
- Tests:
  - a Saturday work date is used only after the weekday dates;
  - a date whose lunch precedes work fills work_am from its first workplace clips.

### Task 6: Evening activities in the schedule (B8)
Files:
- `packages/core/src/world.ts`
- `packages/core/src/schedule/{decisions,windows,generate,types}.ts`
- tests

Tests:
- Tue/Thu exercise is inserted before dinner on Tue/Thu workdays only;
- count 6 with max 2 per week picks the earliest work-end days;
- an after_dinner activity pushes home_evening later and never past sleep;
- late, trip and leave days get nothing;
- the golden test is unchanged for worlds without the dial.

### Task 7: Labels and compose for gym and hobby_2 (B9–B11)
Files:
- `config/nextme_label_rules.l3.json`
- `config/categories.json`
- `packages/mcp/src/composeBlocks.ts`
- ingest tests
- compose tests

Tests:
- gym words label a clip as `gym` before workplace;
- the egoexo guitar take maps to hobby_guitar;
- exercise days get gym dates and the gym run;
- hobby_2 takes egoexo clips, each once;
- block verify passes.

## Addendum 3 (2026-11-13): fuller blocks

- **B12. Fill-maximizing date assignment.**
  - **Score:** for each synthetic day d and each candidate date x in d's allowed pool, the score s(d,x) is the planned seconds of planDay(d, x) before any spare-date fills.
  - **Assignment:** choose the one-to-one assignment that maximizes Σ s plus bonuses:
    - +10 h when an exercise day gets a gym date;
    - +1 h when a late day gets a date whose first work is ≥ 11:00, or a plain workday gets one whose first work is < 11:00.
  - **Method:** Hungarian or min-cost assignment, deterministic. The tiers stay hard: weekday dates before weekend dates for work days; rest, then partial, for rest days; trip dates fixed.
  - **Tie-breaks:** use `rngFor(seed, "block_dates")`. The old shuffle order is no longer needed.
- **B13. Top-up fills.**
  - **When:** after the primary plan, B6 and B6b. A work_am, work_pm or home-place slot whose window still has ≥ 10 min unfilled is topped up.
  - **Where from:** the same-place clips of a spare date (workplace non-meal for work slots, home for home slots), whose real start is within ±180 min of the unfilled part's start.
  - **How:** a contiguous run (gap ≤ 15 min), with each clip once.
  - **Rules:** a spare date serves at most one synthetic day, shared with B6. Placement follows the primary clips back to back. Record in `home_fills`, renamed `fills` with `kind: "home" | "home_fallback" | "topup"`; keep `home_fills` as an alias.
- **B14. hobby_2 distribution.**
  - Split the takes evenly across the hobby_2 days (round-robin by day, in day order).
  - Within a day, prefer takes of the same participant.
  - Use each take once.

### Task 8: Fuller blocks (B12–B14)
Files: `packages/mcp/src/composeBlocks.ts`, tests.

Tests:
- the assignment beats a greedy order on a crafted case;
- tiers and bonuses are respected;
- a half-filled work_pm is topped up from a spare date within ±3 h, and not beyond;
- a spare date never serves two days;
- hobby_2 takes are spread so that every hobby_2 day gets at least one when takes ≥ days;
- block verify passes.

## Addendum 4 (2026-11-13): hobby_1 = LEGO from YouTube

- **B15. hobby_1 fill.** The role `hobby` (the rest-day hobby slot, node hobby_1) is no longer left empty. It is filled like hobby_2 (B14):
  - The clips come from the catalog clips accepted by the slot node's binding, from any dataset.
  - Clips are spread round-robin over the hobby days in day order. Each clip is used once.
  - A clip's day_group is never used on another day.
  - Clips are placed from the block start and trimmed at the block end.
  - The summary's per-day `evening` list is renamed `activities` and includes hobby. `evening` is kept as an alias.
  - World: `hobby_1: { kind: "hobby", activity: "lego building", setting: "home_indoor", weekly_freq: 1, tags: ["lego"] }`, bound `{ dataset: ["youtube"], tags_any: ["lego"] }`. The bank has 3 YouTube clips (MALryOlc3TU parts 1–3, each with its own day_group).

### Task 9: hobby_1 fill (B15)
Files: `packages/mcp/src/composeBlocks.ts`, tests.

Tests:
- 3 bound clips over 3 hobby days give one clip each;
- a 132-min clip in a 120-min window is trimmed;
- a clip is never reused;
- hobby slots with no bound clips stay empty;
- block verify passes.

## Addendum 5 (2026-11-13): whole LEGO parts

- **B16. Per-day slot duration override (core).** A new optional dial is `dials.slot_overrides: [{ day_index, role, duration_min }]`. Use zod `.optional()` and no default, so existing worlds stay byte-identical.
  - On that day, the slot with that role gets exactly `duration_min`. Its start stays where the layout puts it.
  - The following slots shift later.
  - **Sleep limit:** nothing may pass sleep. If it would, shrink the home-place slots after it (home_afternoon, chores, home_evening) first, down to 20 min, and then drop them as `no_time`. Meals keep their minimum. The overridden slot is never shrunk or dropped.
  - **Errors:** if the day still cannot fit, generation fails with a clear INFEASIBLE error that names the day and role.
  - **Validation:** world validation rejects unknown roles and day indexes out of range. Other days are unchanged.
  - The verifier accepts the overridden duration.
- **B17. Fit hobby slots (mcp).**
  - **Setting:** a world dial or CLI flag `--fit-roles hobby`. A world dial `dials.fit_roles: ["hobby"]` is preferred, also `.optional()`.
  - **Process:**
    1. compose-blocks first assigns bound clips to the slots of those roles (B15 order).
    2. It sets `slot_overrides` so each slot's duration equals the total length of its assigned clips, rounded up to whole minutes.
    3. It regenerates the schedule with those overrides and composes again.
  - **Outputs:** the saved world in the outputs includes the computed overrides, so the manifest can be reproduced. The summary reports the overrides.
  - **Result:** fitted hobby clips are never trimmed.

### Task 10: slot_overrides (B16)
Files:
- `packages/core/src/world.ts`
- `packages/core/src/schedule/{generate,windows}.ts`
- `packages/core/src/verify.ts`
- tests

Tests:
- an override lengthens a rest-day hobby to 171 min and shifts later slots;
- home slots shrink before anything else;
- the overridden slot never shrinks;
- an infeasible override errors;
- worlds without the dial are byte-identical.

### Task 11: fit_roles in compose-blocks (B17)
Files: `packages/mcp/src/composeBlocks.ts`, `composeBlocksCli.ts`, tests.

Tests:
- with fit_roles hobby, 3 parts (133/106/171 min) give hobby slots of exactly those lengths with untrimmed placements;
- the written world carries the overrides;
- block verify passes.
