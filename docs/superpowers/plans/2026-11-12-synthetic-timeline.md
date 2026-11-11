# Synthetic Timeline Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the "one base day + inserted events" composer with a pipeline that turns a world file (one person) into a verified, person × N-day timeline manifest. The pipeline is world → donor → schedule → matching → rule verifier → manifest. It runs today on NextMe-800 (home and work in one wearer's days) plus Ego4D/Ego-Exo4D hobby clips. YouTube clips from the extraction framework must later flow through the same `ingest bank` path with no code change.

**Architecture:** We keep the three workspaces.
- `@vb/core` stays pure (no I/O). It gains the clip/world/schedule/timeline/manifest schemas, a seeded RNG, the ATUS donor source, the schedule generator, matching, the verifier rules and the manifest builder. The old composition, timeline and manifest modules are removed.
- `@vb/ingest` writes clip catalogs. Sources are any 4.1-contract JSONL (`ingest bank`), a NextMe draft-label step, and Ego4D/Ego-Exo4D mapped through `config/categories.json`.
- `@vb/mcp` swaps its tools for timeline tools. It also adds a `generate` CLI for batch runs with a deterministic selector.

**Tech Stack:** Node 24, TypeScript 5 (ESM, NodeNext, strict), npm workspaces, zod 3, `@modelcontextprotocol/sdk` 1.x, `csv-parse`, vitest, tsx. No new npm dependencies.

**Spec:** `docs/superpowers/specs/2026-11-12-synthetic-timeline-generation-design.md`

**Not in this plan (follow-up plans):**
- Plan 2: rendering (spec §9, `@vb/render`). It reads the manifest from Task 12.
- Plan 3: seam VLM check (spec §7.2). It needs open item 2 (model and threshold). Task 11 already emits the seam list it consumes.
- `multi_day` exceptions: schema only (§5.4).

## Global Constraints

- Code, config, messages, scripts and tests are in English. README and spec bodies are in Korean (CLAUDE.md).
- `@vb/core` must not import `node:fs`, `node:path`, `fetch`, `Date.now`, `Math.random`, or any other I/O or ambient randomness. Pass RNG, ids and `now` in.
- **Determinism:** the same world + config + pool + seed produces a byte-identical `schedule.json` (`JSON.stringify(x, null, 2) + "\n"`). Every random draw comes from `rngFor(seed, ...keys)`, so draws for one day never shift draws for another day.
- Times inside a day are integer minutes since 00:00 of that synthetic day. Values ≥ 1440 mean after midnight (sleep `24:30` = 1470). Manifest times are seconds since day 1 00:00, rounded to 3 decimals.
- `data/` is never committed. Test fixtures are synthetic. ATUS outputs live in `data/distributions/atus2025/` and reach other servers through `scripts/sync_data.py`.
- Never print token or key values.
- Hard filters 1 (category/setting/node), 3 (clip unused) and 4 (one `day_group` per synthetic day) are never relaxed (§6.4).
- Failures write nothing and report what is missing (§10).
- Commit after each task. Messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never push.

## Decisions This Plan Makes Where the Spec Is Silent or Conflicts

The user approved the direction on 2026-11-12. Each line names the task that implements it, so the user can veto it there.

| # | Decision | Task |
|---|---|---|
| D1 | Weekend times come from the weekend donor (§5.6). The `weekend_shift_from_weekday` value in `schedule_defaults_atus2025.json` is used only if no weekend pool exists. | 4 |
| D2 | `time_of_day` buckets in minutes: early_morning 240–420, morning 420–660, midday 660–840, afternoon 840–1020, evening 1020–1260, night 1260–1680. A clip fits a slot when its bucket overlaps the slot window. | 2 |
| D3 | The workday template follows §5.2 exactly, with no breakfast slot. `breakfast_skip_prob` applies to the weekend breakfast slot. | 6 |
| D4 | `eat_out` and `others_home` resolve to the world nodes named in `meal_style.place_nodes`. If a node is missing or unbound, its probability mass is dropped and the rest is renormalized. The schedule records the effective distribution. | 3, 5 |
| D5 | No template has a transit slot in v1. Transit clips are ingested but unused. | 9 |
| D6 | Gaps are sampled by the schedule generator, not during matching, so windows already account for transitions. The gap table is a generator input and its version is recorded. Matching only places clips inside windows. | 5 |
| D7 | Before matching, a bank-capacity preflight counts distinct eligible `day_group`s per bound node. It fails with the numbers when there are fewer than the days that need that node. | 10 |
| D8 | `fingerprint` is nullable for interim sources (NextMe drafts, Ego4D/Ego-Exo4D). The extraction framework fills it for YouTube. | 3 |
| D9 | NextMe goes through the contract path. `ingest nextme` writes a draft JSONL labeled by keyword rules. The user reviews it, then runs `ingest bank`. Unlabeled rows are skipped and counted. | 8 |
| D10 | An optional slot still empty after ladder step 2 is dropped (recorded). Only required slots trigger day regeneration (step 3). | 10 |
| D11 | Day regeneration uses `attempt` 1..5 as an extra RNG key and records it on the day. | 6, 10 |
| D12 | HD-EPIC ingest is removed. Its kitchens are other people's homes, so its clips cannot bind to any world node. Ego4D/Ego-Exo4D stay as hobby sources. The verifier tolerates person-level differences (§2). | 9 |
| D13 | Ladder step 1 ("ignore descriptive fields") drops the level-0 filter "a clip must share ≥ 1 tag with the slot when the slot has tags". Ladder step 2 widens the window by ±30 min for the time-of-day check only. Clips must still fit the real window. | 10 |
| D14 | Routine strength (§5.2 step 6) applies to these decisions: optional-slot presence, meal place, breakfast skip, and which hobby. Reference decisions are drawn once per weekday (workdays) or per day type (weekends). Slot order is fixed by the template in v1. | 5 |
| D15 | Clip `time_of_day` may be `"unknown"` (Ego4D/Ego-Exo4D have no wall clock). Unknown passes the time-of-day filter, and placement samples a position in the window (§6.5). | 3, 10 |

## Review Focus

1. **After-midnight times.** A donor `bed` of 1470 or a slot ending at `24:30` must format, bucket (night) and validate correctly, with no wrap to 00:30. Tests are in Task 2 (`formatClock(1470) === "24:30"`, `timeOfDayAt(30) === "night"`) and Task 6 (sleep 1470 schedule).
2. **Day regeneration bookkeeping.** Regenerating day d must free only day d's clips and `day_group`s, and keep days < d intact. Test is in Task 10.
3. **Stale catalog.** A clip referenced by a saved timeline is gone after re-ingest. `export_manifest` and `verify_timeline` must fail and list every missing id, writing nothing. Tests are in Tasks 12 and 13.
4. **Impossible bank.** When every clip is longer than every window, or a node has no eligible clips, the result is `UNFILLABLE` or `BANK_CAPACITY`, listing slot ids or counts, with no partial file. Tests are in Tasks 10 and 13.
5. **Machine timezone.** NextMe HKT bucketing and the schedule bytes are identical under `TZ=UTC` and `TZ=America/New_York`. Tests are in Task 6 (determinism test) and Task 8.

---

## File Structure

```
config/
  categories.json                 + "clip" mapping per category (Task 9)
  schedule_defaults_atus2025.json (from ATUS bundle; fallback + commute gaps)
  timeline/templates.v1.json      day-type slot templates (Task 6)
  timeline/gaps.v1.json           transition gap table (Task 5)
  timeline/exceptions.v1.json     exception catalog (Task 5)
  nextme_label_rules.json         NextMe draft labeling rules (Task 8)
  worlds/example-nextme.json      example world bound to NextMe + Ego-Exo4D (Task 13)
scripts/atus/atus_stats.py        (from ATUS bundle)
scripts/sync_data.py              folders: catalog, worlds, timelines, exports, renders, distributions
packages/core/src/
  schema.ts       Dataset(+youtube), SourceRef, TimeSpan, Annotation, ManifestCaption (kept)
  licenses.ts     LICENSES (moved from old manifest.ts, + youtube)
  errors.ts       TimelineError + codes (replaces CompositionError)
  clock.ts        parseClock/formatClock/time-of-day buckets
  rng.ts          hash32, Rng, rngFor
  clip.ts         ClipSpec, Clip, ClipContractRow, ClipCatalog
  world.ts        World, NodeBinding, MealPlace
  donor.ts        DistributionSource, atus2025 pool parse, selectDonors
  gaps.ts         GapTable, sampleGap, empirical quantile sampling
  schedule/
    types.ts      Schedule, ScheduleDay, Slot, Exception, Transition, Template schemas
    decisions.ts  calendar, day types, routine reference, exceptions
    windows.ts    resolveDayWindows
    generate.ts   generateSchedule, generateDay
  match.ts        bindingAccepts, candidatesFor, RankSelector, fillTimeline, preflightCapacity
  layout.ts       layoutTimeline (clip placement → synthetic seconds)
  timeline.ts     Timeline schema + assignClip/unassignClip/assertRevision (rewritten)
  verify.ts       verifyTimeline → VerifierReport
  remap.ts        remapAnnotations (kept; parameter type generalized)
  manifest.ts     buildTimelineManifest (rewritten)
packages/ingest/src/
  legacy.ts       BlockCandidate/BaseDay schemas used only inside adapters (Task 1; renamed blocks.ts in Task 9)
  nextme/tiling.ts  moved from core
  nextme/draft.ts   NextMe blocks → contract draft rows (Task 8)
  bank.ts           contract JSONL → clips + annotations (Task 7)
  clipmap.ts        BlockCandidate + category clip mapping → Clip (Task 9)
  writer.ts         + writeClipCatalog; rebuildIndex counts clips
packages/mcp/src/
  config.ts       load templates/gaps/exceptions/defaults/pool from disk
  fsCatalog.ts    FsClipCatalog implements ClipCatalog (rewritten)
  store.ts        TimelineStore, WorldStore (rewritten)
  view.ts         text day view (replaces render.ts)
  server.ts       timeline tools (rewritten)
  generate.ts     batch CLI: create → auto_fill → verify → export
```

---

### Task 0: Land pending work and the ATUS bundle

**Files:**
- Commit as-is: the 15 modified files in the working tree, plus `scripts/convert_ego4d_narrations.py` and `scripts/test_convert_ego4d_narrations.py` (Ego4D keyword work from 2026-11-11)
- Commit: `docs/superpowers/specs/2026-11-12-synthetic-timeline-generation-design.md`, `scripts/atus/atus_stats.py`, `config/schedule_defaults_atus2025.json`, this plan
- Modify: `scripts/sync_data.py:18`, `scripts/test_sync_data.py`

- [ ] **Step 1: Verify the pending work passes, then commit it on the current branch**

Run: `npm test && npm run typecheck && python3 -m pytest scripts/test_sync_data.py scripts/test_convert_ego4d_narrations.py`
Expected: all pass. Then commit with `feat(ingest): word-level Ego4D keywords and narration pre-filter`.

- [ ] **Step 2: Create the branch `feat/synthetic-timeline` from that commit and commit the spec, ATUS script, defaults and plan** (`docs: add synthetic timeline spec, ATUS artifacts and plan`).

- [ ] **Step 3: Write failing tests in `scripts/test_sync_data.py`**

`test_push_folders_are_the_timeline_set`: `FOLDERS == ("catalog", "worlds", "timelines", "exports", "renders", "distributions")`.
`test_pull_refuses_when_local_timelines_exist`: a tmp data dir with `timelines/x.json` makes `assert_pull_safe(dir, overwrite=False)` raise `SyncError` matching `"timelines"`.

- [ ] **Step 4: Change `FOLDERS`, and in `assert_pull_safe` guard `timelines` instead of `compositions`.** Run pytest; expected PASS. Commit `feat(scripts): sync worlds, timelines and distributions`.

---

### Task 1: Remove the base-day composition pipeline

The rest of the plan rebuilds these pieces on the new model. This task leaves the repo green with the MCP server gone and the ingest adapters still producing their current output.

**Files:**
- Delete: `packages/core/src/{composition,timeline,manifest,catalog,tiling}.ts`, `packages/core/test/{composition,timeline,manifest,schema}.test.ts`, `packages/core/test/__snapshots__/golden.manifest.json`, all of `packages/mcp/src/*` and `packages/mcp/test/*`
- Move: `packages/core/src/tiling.ts` → `packages/ingest/src/nextme/tiling.ts`, and `packages/core/test/tiling.test.ts` → `packages/ingest/test/tiling.test.ts`
- Create: `packages/ingest/src/legacy.ts` (moves `BlockCandidateSchema`, `BaseDaySchema` and their types out of core), `packages/core/src/licenses.ts`
- Modify: `packages/core/src/schema.ts`, `errors.ts`, `remap.ts`, `index.ts`; the imports in `packages/core/test/{remap.test.ts,fixtures.ts}` and the ingest adapters and writer; `package.json` (drop the `mcp` script until Task 13)

**Interfaces:**
- Produces:
  - `remapAnnotations(span: PlacedSpan, annotations: Annotation[]): ManifestCaption[]`, where `PlacedSpan = { block_id: string; source: SourceRef; src_start: number; src_end: number; composed_start: number }`. The behavior is unchanged.
  - `LICENSES: Record<Dataset, { license: string; note?: string }>`
  - `class TimelineError extends Error { code: TimelineErrorCode; details: Record<string, unknown> }`, with codes `COUNTRY_SOURCE_MISSING | DONOR_POOL_EMPTY | SCHEDULE_INFEASIBLE | GAP_MISSING | EXCEPTION_OVERLAP | BANK_CAPACITY | UNFILLABLE | SLOT_NOT_FOUND | CLIP_NOT_FOUND | CLIP_INELIGIBLE | REVISION_MISMATCH | TIMELINE_NOT_FOUND | WORLD_NOT_FOUND | MISSING_CLIPS | VERIFY_FAILED`
  - `DatasetSchema` = `["nextme800","hdepic","ego4d","egoexo4d","youtube"]`. `hdepic` leaves in Task 9.
  - `LICENSES.youtube = { license: "YouTube creator content, not redistributed", note: "Manifest lists video ids and spans only" }`

- [ ] **Step 1:** Do the deletions and moves. Change the `remap.ts` parameter type to `PlacedSpan` and keep the existing `remap.test.ts` cases unchanged.
- [ ] **Step 2:** Run `npm test && npm run typecheck`. Expected: PASS. The remaining suites are remap, ingest (incl. moved tiling) and jsonl/time.
- [ ] **Step 3:** Commit `refactor: remove base-day composition pipeline`.

---

### Task 2: Clock and seeded RNG

**Files:**
- Create: `packages/core/src/clock.ts`, `packages/core/src/rng.ts`
- Test: `packages/core/test/clock.test.ts`, `packages/core/test/rng.test.ts`

**Interfaces:**
- Produces:
  - `parseClock(s: string): number`: accepts `HH:MM` with `00 ≤ HH ≤ 27`, and throws on anything else.
  - `formatClock(min: number): string`: rounds to the nearest minute and never wraps.
  - `TIME_OF_DAY = ["early_morning","morning","midday","afternoon","evening","night"] as const`, `type TimeOfDay`
  - `TIME_OF_DAY_RANGES: Record<TimeOfDay, [number, number]>` (D2)
  - `timeOfDayAt(min: number): TimeOfDay`: values < 240 are treated as +1440.
  - `bucketOverlaps(tod: TimeOfDay | "unknown", window: [number, number], padMin = 0): boolean`
  - `hash32(s: string): number` (FNV-1a)
  - `class Rng { constructor(seed: number); next(): number; int(lo: number, hiExcl: number): number; normal(mean: number, sd: number): number; bernoulli(p: number): boolean; pickWeighted<T>(items: readonly { value: T; weight: number }[]): T }` (mulberry32; Box–Muller)
  - `rngFor(seed: number, ...keys: (string | number)[]): Rng`, which is `new Rng(hash32(`${seed}|${keys.join("|")}`))`

- [ ] **Step 1: Write failing tests**

```ts
expect(parseClock("07:52")).toBe(472);
expect(parseClock("24:30")).toBe(1470);
expect(() => parseClock("7:5")).toThrow();
expect(formatClock(1470)).toBe("24:30");
expect(formatClock(472.4)).toBe("07:52");
expect(timeOfDayAt(30)).toBe("night");
expect(timeOfDayAt(420)).toBe("morning");
expect(bucketOverlaps("evening", [1000, 1019])).toBe(true);
expect(bucketOverlaps("evening", [900, 1010])).toBe(false);
expect(bucketOverlaps("evening", [900, 1010], 30)).toBe(true);
expect(bucketOverlaps("unknown", [0, 1])).toBe(true);
```

RNG cases:
- `rngFor(1,"a",2)` reproduces the same 5 numbers on two instances.
- `rngFor(1,"a",2)` and `rngFor(1,"a",3)` differ.
- All `next()` values are in `[0,1)`.
- `pickWeighted` with weights 1:3 over 100 000 draws gives a share of the second item in `[0.74, 0.76]`.
- `normal(10, 2)` over 50 000 draws has a mean within ±0.05 and an sd within ±0.05.

- [ ] **Step 2:** Run `npx vitest run packages/core/test/clock.test.ts packages/core/test/rng.test.ts`. Expected: FAIL (module not found).
- [ ] **Step 3:** Implement and export from `index.ts`.
- [ ] **Step 4:** Run the same command. Expected: PASS.
- [ ] **Step 5:** Commit `feat(core): add clock buckets and seeded rng`.

---

### Task 3: Clip contract and world schemas

**Files:**
- Create: `packages/core/src/clip.ts`, `packages/core/src/world.ts`
- Test: `packages/core/test/clip.test.ts`, `packages/core/test/world.test.ts`

**Interfaces:**
- Produces (zod schema + inferred type for each):
  - `CategorySchema = enum("home","work","meal","hobby","transit")`, `SettingSchema = enum("home_indoor","venue_indoor","outdoor")`
  - `ClipSpecSchema = { category, setting, node: string, activity: string, time_of_day: TimeOfDay, tags: string[] (default []) }`. This is the slot spec (§4.1).
  - `ClipSchema = ClipSpec.omit(time_of_day).extend({ time_of_day: TimeOfDay | "unknown", clip_id, source: SourceRef, source_start_s, source_end_s (> start), day_group: string, person_source: string, fingerprint: { duration_s: number; phash: string[] } | null, extractor_version: string })`
  - `ClipContractRowSchema = Clip.extend({ captions: { id: string; level: AnnotationLevel; time: TimeSpan; text: string }[] })`. This is the line format of `ingest bank`. Caption times are source seconds.
  - `interface ClipCatalog { clips(): readonly Clip[]; getClip(id: string): Clip | undefined; getAnnotations(dataset: Dataset, videoId: string): Annotation[] }`
  - `MealPlaceSchema = enum("home","workplace","eat_out","others_home")`
  - `NodeDefSchema`: discriminated on `kind`, with the variants:
    - `home`
    - `kitchen { default: "home" }`
    - `workplace { job: string }`
    - `hobby { activity: string; setting: Setting; weekly_freq: int 0..7; tags: string[] }`
    - `frequent_place { setting: Setting; optional: true }`
  - `NodeBindingSchema = { dataset?: Dataset[]; person_source?: string; node?: string; tags_any?: string[] }`
  - `WorldSchema` = spec §4.2 minus `donors`, plus `allow_country_fallback: boolean (default false)`, `meal_style.place_nodes: Partial<Record<"eat_out"|"others_home", string>>`, `meal_style.weekend.breakfast: Record<MealPlace, number>`, `dials.start_weekday: "Mon".."Sun" (default "Mon")`, and `node_binding: Record<string, NodeBinding>`. Refinements:
    - 1–3 hobby nodes
    - `length_days ≥ 28`
    - `routine_strength ∈ [0,1]`
    - every `place_nodes` value names a `frequent_place` node
    - every node except `kitchen` has a binding, unless the node is `optional`

- [ ] **Step 1: Write failing tests**
  - `clip.test.ts`:
    - A valid synthetic YouTube row parses.
    - `fingerprint: null` parses (D8).
    - `time_of_day: "unknown"` parses for `Clip` but fails `ClipSpecSchema` (D15).
    - `source_end_s <= source_start_s` fails.
    - An unknown category fails.
  - `world.test.ts`:
    - The spec §4.2 example (made valid by adding bindings and `place_nodes`) parses.
    - 0 hobbies fails, and 4 hobbies fail.
    - `length_days: 27` fails.
    - A missing binding for `workplace` fails with the path `node_binding.workplace`.
    - `frequent_1` with `optional: true` and no binding passes.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement and export.
- [ ] **Step 4:** Run them. Expected: PASS.
- [ ] **Step 5:** Commit `feat(core): add clip contract and world schemas`.

---

### Task 4: ATUS donor source

**Files:**
- Create: `packages/core/src/donor.ts`
- Test: `packages/core/test/donor.test.ts`, with the synthetic CSV built in the test (no ATUS rows)

**Interfaces:**
- Consumes: `World["attributes"]`, `rngFor`, `TimelineError`
- Produces:
  - `type DonorRow = { id: string; daytype: "weekday"|"weekend"; occ: number; w: number; wake: number|null; bed: number|null; work_start: number|null; work_end: number|null; work_min: number|null; worked: boolean; breakfast_time: number|null; breakfast_loc: number|null; lunch_time: number|null; lunch_loc: number|null; dinner_time: number|null; dinner_loc: number|null; sport_has: boolean; sport_time: number|null; sport_outdoor: boolean|null; house_min: number; social_out: boolean; sex?: string; age_band?: string }`
  - `parseAtusPool(csvText: string): DonorRow[]`. Columns are the ones listed in `ATUS_ARTIFACTS_README.md`. Empty cell → null; `True`/`False` → boolean; `*_time` is only kept when `*_has` is true.
  - `OCCUPATION_CODES: Record<string, number>` (mgmt_business_fin 1 … transport_moving 10, as in `atus_stats.py` `OCC`)
  - `interface DistributionSource { id: string; country: string; pool(kind: "weekday"|"weekend"): DonorRow[] }`
  - `atus2025Source(rows: DonorRow[]): DistributionSource`. Its `id` is `"atus_2025"` and its `country` is `"US"`.
  - `type DonorPick = { case_id: string; backoff_level: "occupation_sex_age"|"occupation_sex"|"occupation"|"sex_age"|"all"; filled: string[]; profile: DayProfile }`
  - `type DayProfile = { wake: number; sleep: number; work_start?: number; work_end?: number; breakfast: number; lunch: number; dinner: number; sport?: number; meal_loc: Partial<Record<"breakfast"|"lunch"|"dinner", MealPlace>> }`
  - `selectDonors(attrs: World["attributes"], src: DistributionSource, opts: { seed: number; allowCountryFallback: boolean }): { weekday: DonorPick; weekend: DonorPick; country_fallback: boolean; weekend_redraws: number }`

**Rules (§5.6):**
- Weekday pool: `daytype=weekday ∧ worked ∧ work_start≠null ∧ wake≠null ∧ bed≠null`. When `work_pattern="day_shift"`, also `300 ≤ work_start ≤ 660`.
- Weekend pool: `daytype=weekend ∧ wake≠null ∧ bed≠null`.
- Backoff levels run in order and stop at the first level with ≥ 30 rows. A level whose attributes are missing from the rows (no `sex`/`age_band` until Roster arrives) is skipped.
- Weighted draw by `w` using `rngFor(seed,"donor",kind,attempt)`.
- Empty fields are filled by a weighted draw among same-level rows that have the field, and their names go to `filled`.
- TEWHERE location map: `1→home, 2→workplace, 3→others_home, 4→eat_out`. Any other code is treated as missing.
- Weekend redraw: redraw while `weekend.wake < weekday.wake − 120`, at most 5 times. After that, keep the last pick and leave `weekend_redraws = 5`.
- Country: `country ≠ src.country` throws `COUNTRY_SOURCE_MISSING` unless `allowCountryFallback`.

- [ ] **Step 1: Write failing tests**
  - `same seed → same case_id`
  - `occupation cell with 29 rows falls back to "all"`, and `with 30 rows uses "occupation"`
  - `sex/age levels are skipped when rows lack sex` (`backoff_level === "occupation"`)
  - `weighted draw frequency`: 2 rows with w 1 and 3, 20 000 seeds, share of the heavy row ∈ [0.73, 0.77]
  - `missing lunch_time is filled and recorded` (`filled` contains `"lunch_time"`)
  - `non-US country throws COUNTRY_SOURCE_MISSING`, and `fallback sets country_fallback`
  - `early weekend donor is redrawn` (`weekend_redraws ≥ 1`)
  - `parseAtusPool reads the README header line` (one synthetic line, with `bed` 1470 kept as 1470)
- [ ] **Step 2:** Run `npx vitest run packages/core/test/donor.test.ts`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run it. Expected: PASS.
- [ ] **Step 5:** Commit `feat(core): add ATUS donor selection with backoff`.

---

### Task 5: Gap table, exception catalog and day decisions

**Files:**
- Create: `packages/core/src/gaps.ts`, `packages/core/src/schedule/types.ts`, `packages/core/src/schedule/decisions.ts`, `config/timeline/gaps.v1.json`, `config/timeline/exceptions.v1.json`
- Test: `packages/core/test/gaps.test.ts`, `packages/core/test/decisions.test.ts`

**Interfaces:**
- Produces:
  - `type PlaceKind = "home"|"workplace"|"hobby"|"frequent_place"`. `kitchen` counts as `home`.
  - `GapTableSchema = { version: string; reviewed: boolean; transitions: Record<`${PlaceKind}->${PlaceKind}`, { p10; p25; p50; p75; p90 }> }`
  - `sampleGap(table, from: PlaceKind, to: PlaceKind, rng: Rng): number`. It returns 0 when `from === to` and throws `GAP_MISSING { transition }` when the entry is absent. Otherwise it draws `u ∈ [0.10, 0.90]` and interpolates piecewise-linearly between the quantiles, rounded to a minute.
  - `ExceptionSchema = { scope: "slot"|"day"|"multi_day"; type: string; replaces: string; day_range: [number, number] }`
  - `ExceptionCatalogSchema = { version; day: { type: "leave_day"; from_day_type: "workday"; template: "home_day" }[]; slot: { type: string; replaces_role: string; meal_place: MealPlace; day_types: DayType[] }[] }`
  - `type DayType = "workday"|"hobby_day"|"home_day"|"leave_day"`
  - `type DayDecisions = { day_index: number; weekday: Weekday; day_type: DayType; template: "workday"|"hobby_day"|"home_day"; meal_place: Partial<Record<Meal, MealPlace | "skip">>; optional: Record<string, boolean>; hobby_node?: string; exceptions: Exception[] }`
  - `decideDays(world: World, catalog: ExceptionCatalog, opts: { boundNodes: Set<string> }): DayDecisions[]` (one per day)
  - `decideDay(world, catalog, opts, dayIndex: number, attempt: number): DayDecisions`

**`gaps.v1.json`:**
- `home->workplace` and `workplace->home` are copied from `schedule_defaults_atus2025.json` `transition_gaps`.
- The other 10 ordered pairs are an initial draft and need review:
  - `home<->hobby` p50 20
  - `home<->frequent_place` p50 20
  - `workplace<->frequent_place` p50 10
  - `workplace<->hobby` p50 25
  - `hobby<->frequent_place` p50 15
  - Use p10/p25/p75/p90 = 0.4/0.7/1.6/2.5 × p50, rounded.
- The file has `"reviewed": false`. Spec open item 1 assigns the review.

**`exceptions.v1.json`:**
- Day exception: `leave_day`.
- Slot exceptions:
  - `dinner_out` (replaces `dinner` → `eat_out`, all day types)
  - `lunch_out` (`lunch` → `eat_out`, workday)
  - `lunch_at_friend` (`lunch` → `others_home`, hobby_day and home_day)

**Rules (§5.2 steps 1–3, 6, 7):**
- Calendar: day 1 is `dials.start_weekday`. Saturday and Sunday are weekend days.
- Hobby budget: `B = Σ weekly_freq`.
  - Each week, the first `min(B, 2)` weekend days (Sat first) are `hobby_day` and the rest are `home_day`.
  - The remaining `B − h` occurrences become `optional.hobby = true` on the reference week's workdays, starting Mon.
  - The hobby node per occurrence comes from cycling the hobbies sorted by `weekly_freq` desc, then by name.
- Reference decisions: drawn once per weekday for workdays and once per day type for weekends, with `rngFor(seed,"reference",key)`. Each day re-draws each decision with probability `1 − routine_strength`, using `rngFor(seed,"day",d,attempt,decisionName)`, and otherwise copies the reference (D14).
- Meal place: drawn from `meal_style` (workday or weekend) after D4 renormalization. The weekend breakfast is `"skip"` with `breakfast_skip_prob`.
- Day exception: a weekday becomes `leave_day` with `exception_rate.day` (template `home_day`).
- Slot exception: a day without a day exception gets at most one slot exception, with `exception_rate.slot`, chosen uniformly among the eligible catalog entries. An entry is eligible when its day type matches and its `meal_place` resolves to a bound node.
- `boundNodes` is the set of world nodes with a binding. An unbound `frequent_place` makes exceptions that need it ineligible.

- [ ] **Step 1: Write failing tests**
  - `gaps.test.ts`:
    - `same kind → 0`
    - `missing pair throws GAP_MISSING naming "home->hobby"`
    - `samples stay within [p10,p90]` (1 000 draws)
    - `median of 10 000 draws within ±1 of p50`
    - `gaps.v1.json parses and has all 12 ordered pairs`
  - `decisions.test.ts`:
    - A 28-day world starting Mon has days 6, 7, 13, … as weekend.
    - `B=2` → every weekend is 2× `hobby_day`. `B=3` → one workday per week has `optional.hobby`.
    - `routine_strength=1` → every Tuesday has equal decisions. `routine_strength=0` → over 200 seeds, decisions on Tuesdays differ in ≥ 40% of pairs.
    - `exception_rate.day=1` → every weekday is `leave_day` with exception `{scope:"day",type:"leave_day",replaces:"workday",day_range:[d,d]}`.
    - `exception_rate.slot=1` with no `eat_out` node bound → only `lunch_at_friend` appears, and only when `others_home` is bound. Otherwise no exceptions.
    - No day has both a day and a slot exception.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them. Expected: PASS.
- [ ] **Step 5:** Commit `feat(core): add gap table, exception catalog and day decisions`.

---

### Task 6: Day windows and schedule generation

**Files:**
- Create: `packages/core/src/schedule/windows.ts`, `packages/core/src/schedule/generate.ts`, `config/timeline/templates.v1.json`
- Test: `packages/core/test/windows.test.ts`, `packages/core/test/generate.test.ts`

**Interfaces:**
- Consumes: Tasks 2–5
- Produces:
  - `TemplateSlotSchema = { role: string; category: Category; node: string /* world node name, or "$meal" | "$hobby" */; activity: string; tags: string[]; required: boolean; anchor?: "wake"|"work_start"|"work_end"|"breakfast"|"lunch"|"dinner"|"sport"; end_anchor?: same enum; duration_minutes?: number; min_minutes: number /* default 10 */; meal?: Meal; optional_key?: string }`
  - `TemplatesSchema = { version; day_types: Record<"workday"|"hobby_day"|"home_day", TemplateSlot[]> }`
  - `SlotSchema = { slot_id: `d${day}-s${i}`; role; required; window: [string, string]; spec: ClipSpec }`
  - `ScheduleDaySchema = { day_index; weekday; day_type; attempt: number; wake: string; sleep: string; exceptions: Exception[]; slots: Slot[]; transitions: { from_slot; to_slot; minutes }[] }`
  - `ScheduleSchema = { person_id; generator_version: "schedule.v1"; template_version; gap_table_version; gap_table_reviewed; exception_catalog_version; distribution_source; donors: { weekday: DonorPick; weekend: DonorPick; country_fallback; weekend_redraws }; meal_style_effective; days: ScheduleDay[] }`
  - `type ScheduleInput = { world: World; templates: Templates; gaps: GapTable; exceptions: ExceptionCatalog; source: DistributionSource; boundNodes: Set<string> }`
  - `resolveDayWindows(template: TemplateSlot[], anchors: DayAnchors, decisions: DayDecisions, world: World, gap: (from: PlaceKind, to: PlaceKind) => number): { slots: Slot[]; transitions: Transition[] }`. It throws `SCHEDULE_INFEASIBLE { day, role, reason }`.
  - `generateSchedule(input: ScheduleInput): Schedule`
  - `generateDay(input: ScheduleInput, donors: Schedule["donors"], dayIndex: number, attempt: number): ScheduleDay`

**`templates.v1.json` (§5.2 step 5):**

- `workday`:
  1. `home_morning` (home, required, anchor wake)
  2. `work_am` (work, workplace, required, anchor work_start)
  3. `lunch` ($meal lunch, anchor lunch, duration 45)
  4. `work_pm` (work, workplace, required, end_anchor work_end)
  5. `dinner` ($meal dinner, anchor dinner, duration 60)
  6. `hobby` ($hobby, optional_key hobby, duration 90)
  7. `home_evening` (home, optional_key home_evening)
- `hobby_day`:
  1. `home_morning` (required, anchor wake)
  2. `breakfast` ($meal breakfast, anchor breakfast, duration 20)
  3. `hobby` ($hobby, required, anchor sport, duration 120)
  4. `lunch` (anchor lunch, 45)
  5. `home_afternoon` (home)
  6. `dinner` (anchor dinner, 60)
  7. `home_evening` (home)
- `home_day`: like `hobby_day`, but slot 3 is `chores` (home, required, tags `["cleaning","chores"]`) instead of `hobby`.

Slot activities are short English phrases (`"getting ready"`, `"work"`, …).

**Anchors:**
- `DayAnchors` = the donor profile for the day kind. Weekday donor for `workday`; weekend donor for the others, including `leave_day` (D1).
- Every anchor is jittered by `rngFor(seed,"day",d,attempt,"jitter",name).normal(0, σ)` with `σ = 45 − 25 × routine_strength` (from `schedule_defaults.day_level_sigma`), then rounded.
- `sport` defaults to `parseClock(defaults.weekend.sports_start.p50)` when the donor has none.

**Node resolution:**
- `$meal` resolves through `decisions.meal_place`: `home → "home"`, `workplace → "workplace"`, else `place_nodes[...]`. A `"skip"` drops the slot.
- `$hobby` resolves to `decisions.hobby_node`.
- An optional slot with `optional[optional_key] === false` is dropped.
- Setting comes from the node: home → `home_indoor`, workplace → `venue_indoor`, hobby/frequent_place → their `setting`.
- `spec.time_of_day = timeOfDayAt(window start)`.
- A slot exception replaces `meal_place` for its role before resolution.

**Window algorithm** (the signature does not determine it):

```
kept = template slots after dropping
start[0] = anchors.wake
for i ≥ 1:
  g = gap(kind(kept[i-1]), kind(kept[i]))           // sampled once per (day, attempt, i)
  prevEnd = end_anchor(i-1) ?? (duration(i-1) ? start[i-1]+duration(i-1) : start[i-1]+min(i-1))
  start[i] = max(anchor(i) ?? -∞, prevEnd + g)
end[i] = end_anchor(i) ?? (duration(i) ? start[i]+duration(i) : start[i+1] - g(i,i+1))
end[last] = anchors.sleep (when the last slot has no duration)
infeasible if end[i] - start[i] < min(i), or end[i] > start[i+1] - g(i,i+1), or end[last] > sleep
```

- [ ] **Step 1: Write failing tests**
  - `windows.test.ts`, with a hand-built workday (wake 390, work_start 454, lunch 720, work_end 1020, dinner 1110, sleep 1338, every gap 20, `optional.hobby=false`, `optional.home_evening=true`, lunch at workplace, dinner at home):
    - Windows are exactly `home_morning 06:30–07:14`, `work_am 07:34–12:00`, `lunch 12:00–12:45` (lunch at workplace, so no gap), `work_pm 12:45–17:00`, `dinner 18:30–19:30` at home (start = max(1110, 1020+20)), `home_evening 19:30–22:18`.
    - The transitions list holds two 20-min entries.
    - Dinner at `eat_out` adds gaps on both sides.
    - `work_end` before `lunch + 45` throws `SCHEDULE_INFEASIBLE` with role `work_pm`.
    - Sleep 1470 gives a last window ending `"24:30"`.
  - `generate.test.ts`:
    - `byte-identical`: `JSON.stringify(generateSchedule(x))` is equal twice, and equal under `process.env.TZ="UTC"` vs `"America/New_York"` (set before the call).
    - `28 days, slot ids unique, windows inside [wake, sleep], transitions match node changes`
    - `generateDay(…, d, 1)` differs from attempt 0 but leaves the other days of `generateSchedule` untouched.
    - `meal_style_effective` drops `eat_out` when no `place_nodes.eat_out` is set and sums to 1.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement. `generateSchedule` calls `selectDonors` once, `decideDays`, then for each day `resolveDayWindows`. It retries that day with attempts 1..5 on `SCHEDULE_INFEASIBLE` and rethrows after the 5th, listing the day.
- [ ] **Step 4:** Run them. Expected: PASS.
- [ ] **Step 5:** Commit `feat(core): generate deterministic schedules from templates`.

---

### Task 7: `ingest bank` for contract JSONL

**Files:**
- Create: `packages/ingest/src/bank.ts`
- Modify: `packages/ingest/src/writer.ts` (add `writeClipCatalog`; `rebuildIndex` also counts `clips.jsonl` as `clips: { "<dataset>": { "<category>": n } }`), `packages/ingest/src/cli.ts` (subcommand `bank`)
- Test: `packages/ingest/test/bank.test.ts`, `packages/ingest/test/writer.test.ts`

**Interfaces:**
- Consumes: `ClipContractRowSchema`, `ClipSchema`, `AnnotationSchema`
- Produces:
  - `ingestBank(text: string, file: string, dataset: Dataset): { clips: Clip[]; annotations: Map<string, Annotation[]>; report: IngestReport }`
    - Each caption becomes `{ id: `${dataset}:${video_id}:${caption.id}`, track: "captions", level, time, text, original: caption }`. Annotations are merged per video and deduplicated by id.
    - A row whose `category` is `null` is skipped as `unlabeled` (D9).
    - Any other schema error throws `IngestFormatError` with `file:line`.
    - A row whose `source.dataset` differs from `--dataset` is an error.
    - A duplicate `clip_id` is an error.
  - `CatalogWriter.writeClipCatalog(dataset: Dataset, clips: Clip[], annotations: Map<string, Annotation[]>): void`. It replaces `catalog/<ds>/clips.jsonl` and annotations, with the same validation-before-write and stale-file removal as `writeEventCatalog`.
  - CLI: `npm run ingest -- bank --input <file.jsonl> --dataset <ds> [--data <dir>]`

- [ ] **Step 1: Write failing tests**
  - `two rows of one video share one annotation file`
  - `unlabeled row is skipped and counted (report.skipped.unlabeled === 1)`
  - `bad row reports file:line`
  - `dataset mismatch throws`
  - `re-ingest removes annotation files of videos no longer present`
  - `index.json has clips.youtube.home === 1`
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them. Expected: PASS.
- [ ] **Step 5:** Commit `feat(ingest): ingest contract clip banks`.

---

### Task 8: NextMe draft labeling

**Files:**
- Create: `packages/ingest/src/nextme/draft.ts`, `config/nextme_label_rules.json`
- Modify: `packages/ingest/src/cli.ts`. `nextme --days … --out <draft.jsonl>` now writes a draft and no longer writes `catalog/nextme800/days`. `writeBaseDay` is removed from the writer.
- Test: `packages/ingest/test/nextme-draft.test.ts`. Reuse the existing synthetic NextMe fixtures.

**Interfaces:**
- Consumes: `buildNextmeDay` (unchanged; it returns legacy blocks + annotations), `parseHkt`, `timeOfDayAt`, `keywordMatches`-style whole-word matching (move the regex helper to `packages/ingest/src/words.ts` and reuse it from `categories.ts`)
- Produces:
  - `LabelRulesSchema = { version; person_source: string; places: { words: string[]; node: string; setting: Setting }[]; activities: { words: string[]; category: Category }[]; default_category_by_node: Record<string, Category> }`
  - `nextmeDraftRows(day: string, built: ReturnType<typeof buildNextmeDay>, rules: LabelRules): DraftRow[]`
    - `DraftRow` is `ClipContractRow` with `category`, `setting` and `node` nullable.
    - One row per L4-tiled block.
    - `clip_id` = block id.
    - `activity` = block summary.
    - `day_group` = `day`.
    - `person_source` = `rules.person_source`.
    - `time_of_day = timeOfDayAt(minutes of (start_wallclock + src_start) in HKT)`.
    - `fingerprint: null`.
    - `extractor_version: "nextme-draft.v1"`.
    - `captions` = the recording's annotations overlapping the block, with `id` = the annotation id suffix.
  - Labeling: the place is the first `places` rule matching the block's L4 text, then the L5 text containing the block start. The category is the first `activities` match, else `default_category_by_node[node]`. With no place match, `category`/`setting`/`node` are `null`.

**`nextme_label_rules.json` v1:**
- `person_source: "nextme800:wearer"`
- Places:
  - home `["apartment","home","bedroom","kitchen","dorm"]` → home_indoor
  - workplace `["hospital","library","office","lab","laboratory","campus"]` → venue_indoor
  - transit `["street","streets","tram","bus","subway","train","MTR"]` → outdoor
- Activities:
  - meal `["meal","eat","eating","ate","cook","cooking","breakfast","lunch","dinner"]`
  - transit `["commute","commuting"]`
- Defaults: home → home, workplace → work, transit → transit

- [ ] **Step 1: Write failing tests**
  - A fixture block with L4 `"I conduct my meal routine at my apartment."` → `{category:"meal", node:"home", setting:"home_indoor"}`.
  - `"I study … at the library desk"` → `{category:"work", node:"workplace"}`.
  - No place word → nulls.
  - A block starting 13:23 HKT → `time_of_day:"afternoon"` under both TZs.
  - Every row has `day_group === "2026-04-15"`.
  - Feeding the draft (with nulls) to `ingestBank` skips exactly the unlabeled rows.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them, plus `npm test`. Expected: PASS.
- [ ] **Step 5:** Commit `feat(ingest): write NextMe days as labeled clip drafts`.

---

### Task 9: Ego4D / Ego-Exo4D as hobby clips; drop HD-EPIC

**Files:**
- Create: `packages/ingest/src/clipmap.ts`
- Modify: `config/categories.json`, `packages/ingest/src/categories.ts`, `packages/ingest/src/cli.ts` (`ego4d`/`egoexo` write `writeClipCatalog`)
- Delete: `packages/ingest/src/hdepic/`, `packages/ingest/test/hdepic.test.ts`, `packages/ingest/test/fixtures/hdepic/`, `scripts/convert_hdepic.py`, `writeEventCatalog`. `hdepic` also leaves `DatasetSchema` and `LICENSES`.
- Rename: `packages/ingest/src/legacy.ts` → `packages/ingest/src/blocks.ts`. It keeps `BlockCandidate` and `BaseDay`, which only the adapters use internally.
- Test: `packages/ingest/test/clipmap.test.ts`. Update the expectations in `ego4d.test.ts` and `egoexo.test.ts` from candidates to clips.

**Interfaces:**
- Produces:
  - `CategorySchema.clip?: { category: Category; setting: Setting; node: string; tags: string[] }`
  - `candidateToClip(c: BlockCandidate, clip: NonNullable<Category["clip"]>): Clip`
    - `clip_id = c.id`
    - `activity = c.summary`
    - `tags = [...clip.tags, c.category]`
    - `time_of_day: "unknown"`
    - `day_group = person_source = `${dataset}:${video_id}``
    - `fingerprint: null`
    - `extractor_version: `${dataset}-ingest.v1``
  - Categories without `clip` are counted as `skipped.no_clip_mapping`.

**Initial `clip` mapping:**
- hobby with setting outdoor: walking, hobby_tent, sport_soccer
- hobby with setting venue_indoor: sport_climbing, sport_basketball, hobby_dance, hobby_music
- transit with setting outdoor: transit (D5)
- No mapping: hobby_lego (another person's home), cooking, health, bike_repair

`node` is `"hobby"` for the hobby rows and `"transit"` for transit.

- [ ] **Step 1: Write failing tests**
  - `climbing candidate becomes hobby/venue_indoor with tags ["sport_climbing"]`
  - `cooking candidate is skipped as no_clip_mapping`
  - `ego4d ingest writes clips.jsonl and no candidates.jsonl`
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement and do the deletions.
- [ ] **Step 4:** Run `npm test && npm run typecheck`. Expected: PASS.
- [ ] **Step 5:** Commit `feat(ingest): map Ego4D and Ego-Exo4D to hobby clips; drop HD-EPIC`.

---

### Task 10: Timeline model and matching

**Files:**
- Create: `packages/core/src/match.ts`, `packages/core/src/layout.ts`
- Rewrite: `packages/core/src/timeline.ts`
- Test: `packages/core/test/timeline.test.ts`, `packages/core/test/match.test.ts`, `packages/core/test/layout.test.ts`, `packages/core/test/fixtures.ts` (rewritten: an in-memory `ClipCatalog` with synthetic clips for one person over 30 day_groups, plus a minimal world and schedule builder)

**Interfaces:**
- Consumes: `Schedule`, `ScheduleInput`, `generateDay`, `ClipCatalog`, `World`, `rngFor`, `bucketOverlaps`
- Produces:
  - `Relaxation = "ignore_tags" | "widen_window_30m" | "dropped_optional"`
  - `TimelineSchema = { schema_version: 1; id; name?; revision; world: World; schedule: Schedule; slots: Record<string /* slot_id */, { clip_ids: string[]; relaxations: Relaxation[] }> }`
  - `createTimeline(world: World, schedule: Schedule, genId: () => string): Timeline` (revision 0, every slot empty)
  - `assignClip(t: Timeline, catalog: ClipCatalog, a: { slotId: string; clipId: string; relaxations?: Relaxation[] }): Timeline`. It increments the revision and throws `SLOT_NOT_FOUND`, `CLIP_NOT_FOUND`, or `CLIP_INELIGIBLE { failed: string[] }` (the hard-filter names the clip fails at the given relaxations).
  - `unassignClip(t, a: { slotId: string; clipId?: string }): Timeline`. Without `clipId` it clears the slot.
  - `assertRevision(t, expected?: number)`
  - `bindingAccepts(world: World, nodeName: string, clip: Clip): boolean`. Every binding field that is present must match. `kitchen` delegates to its default node.
  - `hardFilterFailures(t, catalog, slotId, clip, relax: Relaxation[]): string[]`. The names are `category_setting_node | time_of_day | already_used | day_group_other_day | does_not_fit | tags`.
  - `type RankedCandidate = { clip: Clip; score: number; reasons: string[] }`
  - `candidatesFor(t, catalog, slotId, relax: Relaxation[]): RankedCandidate[]`. Sorted by score desc, then `clip_id`.
    - `score = jaccard(words(slot activity+tags), words(clip activity+tags))`
    - `+ 0.5` if the clip's `day_group` is already used today by a slot of the same node
    - `+ routine_strength × jaccard(clip, clip assigned to the same role yesterday)`
  - `interface Selector { choose(slot: Slot, candidates: RankedCandidate[] /* top 10 */, recent: { day_index: number; role: string; clip_id: string }[]): string }`
  - `rankSelector: Selector` (takes `candidates[0]`)
  - `preflightCapacity(t, catalog): void`. Throws `BANK_CAPACITY { node, needed_days, distinct_day_groups }[]` (D7).
  - `fillTimeline(t: Timeline, catalog: ClipCatalog, selector: Selector, regen: (dayIndex: number, attempt: number) => ScheduleDay): Timeline`
  - `layoutTimeline(t, catalog): PlacedClip[]` with `PlacedClip = { slot_id; day_index; clip: Clip; synthetic_start_s; synthetic_end_s }`

**Hard filters (§6.2):**
- `does_not_fit`: `Σ durations in slot + this clip ≤ window length`.
- `time_of_day`: `bucketOverlaps(clip.time_of_day, window, relax has widen ? 30 : 0)`.
- `day_group_other_day`: the clip's `day_group` is already used on another synthetic day.

**`fillTimeline` ladder (§6.4, D10, D11, D13):**
- Days run in order and slots run in order, skipping slots that already have clips.
- For each slot, try relax `[]`, then `["ignore_tags"]`, then `["ignore_tags","widen_window_30m"]`. The first non-empty candidate list feeds `selector.choose`, and the relaxations used are recorded.
- An optional slot with no candidates gets `["dropped_optional"]` and stays empty.
- A required slot with no candidates triggers regeneration:
  1. Clear all of that day's slots.
  2. Replace the day with `regen(d, attempt)`, for attempt 1..5.
  3. Refill the day.
- After 5 failed attempts, throw `UNFILLABLE { slots: [slot_id…] }` without returning a partial timeline.
- The revision increments once per `fillTimeline` call.

**`layoutTimeline` placement (§6.5):**
- `free = window_len − Σ dur`.
- Draw `k+1` non-negative gaps from sorted uniforms of `rngFor(seed,"place",slot_id)` and place the clips in assignment order.
- When every clip has a known `time_of_day`, the window is first narrowed to its overlap with the union of the clips' buckets, if that overlap still fits.
- `synthetic_start_s = (day_index−1)·86400 + start_min·60`.

- [ ] **Step 1: Write failing tests**
  - `timeline.test.ts`:
    - `assign increments revision`
    - `assign rejects clip of another node with CLIP_INELIGIBLE ["category_setting_node"]`
    - `assign rejects second use of a clip (already_used)`
    - `assign rejects a day_group used on another day`
    - `unassign frees it`
  - `match.test.ts`:
    - `RankSelector fills a 28-day fixture with every required slot filled`
    - `same-node slots of one day share a day_group when possible`
    - `tags filter drops at level 0 and passes with ignore_tags (recorded)`
    - `evening-only clip fits an afternoon window only with widen_window_30m`
    - `optional slot with no clips is dropped_optional`
    - `required slot with no clips triggers regen attempts 1..5, then UNFILLABLE lists the slot`
    - **Review Focus 2:** `regen of day 3 keeps days 1–2 identical and frees only day-3 day_groups`
    - `preflight reports BANK_CAPACITY {node:"home", needed_days:28, distinct_day_groups:10}`
    - **Review Focus 4:** `every clip longer than every window → UNFILLABLE`
  - `layout.test.ts`:
    - `clips stay inside window and do not overlap`
    - `deterministic for same seed`
    - `unknown time_of_day samples within the full window`
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them. Expected: PASS.
- [ ] **Step 5:** Commit `feat(core): add timeline model, matching ladder and clip layout`.

---

### Task 11: Rule verifier

**Files:**
- Create: `packages/core/src/verify.ts`
- Test: `packages/core/test/verify.test.ts`, with one violating fixture per rule built by mutating a valid filled timeline from Task 10's fixtures.

**Interfaces:**
- Consumes: `Timeline`, `ClipCatalog`, `layoutTimeline`, `hardFilterFailures`
- Produces:
  - `RULE_IDS = ["day_type","slot_window","wake_sleep","no_overlap","monotonic","node_binding","clip_unique","day_group_single_day","time_of_day","exceptions"] as const`
  - `type VerifierReport = { pass: boolean; rules: Record<RuleId, { pass: boolean; violations: { slot_id?: string; day_index?: number; detail: string }[] }>; flags: { kind: "swap_point"; day_index: number; slot_ids: [string, string] }[]; seams: { day_index: number; before_clip: string; after_clip: string; at_s: number }[]; missing_clips: string[] }`
  - `verifyTimeline(t: Timeline, catalog: ClipCatalog): VerifierReport`
    - `missing_clips` non-empty → every rule that needs clips fails.
    - `required` slots that are empty fail `slot_window`.
    - `time_of_day` uses the recorded relaxations for each slot.
    - `exceptions` checks that each recorded slot exception matches the slot's resolved meal place and that no day has two exceptions.
    - A seam is every point where consecutive placed clips of one day differ in `day_group` (input to Plan 3).

- [ ] **Step 1: Write failing tests:** `valid timeline passes every rule`, plus one test per rule id named `<rule> violation is caught`. Each asserts `rules[<rule>].pass === false`, the other rules still pass, and `pass === false`. Also `swap_point flag for same node, different day_group` and `seam list has the expected at_s`.
- [ ] **Step 2:** Run it. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run it. Expected: PASS.
- [ ] **Step 5:** Commit `feat(core): add timeline rule verifier`.

---

### Task 12: Timeline manifest

**Files:**
- Rewrite: `packages/core/src/manifest.ts`
- Test: `packages/core/test/manifest.test.ts` (new golden snapshot from the synthetic fixture: `packages/core/test/__snapshots__/timeline.manifest.json`)

**Interfaces:**
- Consumes: `layoutTimeline`, `verifyTimeline`, `remapAnnotations` (with `composed_start = synthetic_start_s`, `block_id = "clip:<slot_id>:<clip_id>"`), `LICENSES`
- Produces:
  - `TimelineManifestSchema` (§8):
    - `schema_version: 1`, `id`, `revision`, `created_at`
    - `versions { schema: 1; generator; template; gap_table; exception_catalog; extractor: string[] }`
    - `inputs { world; dials; seed; node_binding; distribution_source; donors }`
    - `days[] { day_index; weekday; day_type; attempt; wake; sleep; coverage_ratio; exceptions; clips[] }`
    - each clip: `{ source; video_id; source_start_s; source_end_s; synthetic_start_s; synthetic_end_s; slot_id; category; node; setting; relaxations; fingerprint; provenance { spec: ClipSpec-with-unknown; extractor_version } }`
    - `captions: ManifestCaption[]`
    - `verifier_summary { pass; failed_rules; flags; seams: number }`
    - `licenses[]`
    - `qa_tracks: {}`
  - `buildTimelineManifest(t: Timeline, catalog: ClipCatalog, now: Date): TimelineManifest`
    - Throws `MISSING_CLIPS { ids }` before anything else.
    - Throws `VERIFY_FAILED { failed_rules }` if `verifyTimeline(...).pass` is false.
    - `coverage_ratio = round3(Σ clip seconds / ((sleep − wake) · 60))`.
    - Captions are sorted with the existing level order.

- [ ] **Step 1: Write failing tests**
  - `golden manifest for fixture`
  - `caption clipped at clip edge keeps clipped flag` (reuses the remap rule)
  - **Review Focus 3:** `catalog missing two clips → MISSING_CLIPS lists both`
  - `failing verifier → VERIFY_FAILED`
  - `coverage_ratio = 0.25 for 4 h of clips over 16 h awake`
- [ ] **Step 2:** Run it. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run it. Expected: PASS.
- [ ] **Step 5:** Commit `feat(core): build timeline manifests`.

---

### Task 13: MCP timeline tools and batch `generate`

**Files:**
- Create: `packages/mcp/src/{config,fsCatalog,store,view,server,generate,main,index}.ts`, `config/worlds/example-nextme.json`
- Test: `packages/mcp/test/{helpers,tools,generate}.test.ts`. Fixtures are written into a temp data dir via `CatalogWriter.writeClipCatalog`.
- Modify: `package.json` scripts `mcp` and `"generate": "tsx packages/mcp/src/generate.ts"`

**Interfaces:**
- Consumes: everything above
- Produces:
  - `loadConfig(configDir: string, dataDir: string): Omit<ScheduleInput, "world" | "boundNodes">`. It reads `timeline/templates.v1.json`, `timeline/gaps.v1.json`, `timeline/exceptions.v1.json`, `schedule_defaults_atus2025.json` and `<dataDir>/distributions/atus2025/atus2025_stats_per_respondent.csv`, and names the missing file on error.
  - `FsClipCatalog implements ClipCatalog` (reads every `catalog/<ds>/clips.jsonl`)
  - `WorldStore(dataDir)`: `list()` and `load(personId)` from `data/worlds/<id>.json`. Throws `WORLD_NOT_FOUND`.
  - `TimelineStore(dataDir)`: `load`, `save`, `list` under `data/timelines/`, and `writeExport(m)` to `data/exports/<id>-r<rev>.manifest.json`, using atomic writes and the safe-id rule kept from the old store.
  - `renderDay(t: Timeline, catalog: ClipCatalog, day: number): string`. Prints one line per slot: `slot_id window role node clip_id(s) relaxations`.
  - Tools, all taking `expected_revision` where they edit:

    | Tool | Purpose |
    |---|---|
    | `list_worlds` | worlds under `data/worlds` |
    | `create_timeline(person_id, name?)` | runs the generator and `preflightCapacity`, saves, returns id + day 1 view |
    | `list_timelines` | saved timelines |
    | `get_timeline(timeline_id, day?)` | text day view |
    | `list_slot_candidates(timeline_id, slot_id, relax?, limit=10)` | ranked candidates (§6.3) |
    | `assign_clip(timeline_id, slot_id, clip_id, expected_revision, relax?)` | assign one clip |
    | `unassign_clip(timeline_id, slot_id, clip_id?, expected_revision)` | remove one or all clips |
    | `auto_fill(timeline_id, expected_revision)` | `fillTimeline` with `rankSelector` |
    | `verify_timeline(timeline_id)` | writes `data/timelines/<id>.verifier_report.json` |
    | `export_manifest(timeline_id)` | writes the manifest |

    Errors are returned as `isError` text `"<CODE>: <message>"` with details JSON.
  - `npm run generate -- --world <path> [--data <dir>] [--config <dir>]`:
    1. copies the world into `data/worlds`
    2. runs create → auto_fill → verify → export
    3. prints `{ timeline_id, revision, manifest, days, clips, coverage_mean }`
    4. exits 1 with the error JSON on any failure, leaving no timeline file behind

**`config/worlds/example-nextme.json`:**
- `country: "HK"` with `allow_country_fallback: true` (NextMe is HKT; spec open item 4)
- `occupation_group: "professional"`
- home/workplace bound to `{ dataset: ["nextme800"], person_source: "nextme800:wearer", node: <same> }`
- `hobby_1`: climbing, venue_indoor, weekly_freq 2, bound to `{ dataset: ["egoexo4d"], tags_any: ["sport_climbing"] }`
- no frequent places
- 28 days, routine_strength 0.7, `exception_rate {slot:0.1, day:0.05}`, seed 1234

- [ ] **Step 1: Write failing tests**
  - `tools.test.ts`:
    - `full flow on fixture: create → auto_fill → verify → export writes manifest`
    - `assign with stale expected_revision returns REVISION_MISMATCH and does not save`
    - `list_slot_candidates returns ≤ limit, sorted`
    - `create_timeline with a 10-day_group bank returns BANK_CAPACITY and saves nothing`
    - **Review Focus 3:** `export after re-ingest removed a clip returns MISSING_CLIPS listing it, no file written`
    - `missing ATUS pool file names its path`
  - `generate.test.ts`:
    - `generate CLI exits 0 and prints manifest path`
    - **Review Focus 4:** `generate with unfillable bank exits 1 and leaves data/timelines empty`
- [ ] **Step 2:** Run `npx vitest run packages/mcp`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `npm test && npm run typecheck`. Expected: PASS.
- [ ] **Step 5:** Commit `feat(mcp): timeline tools and batch generate`.

---

### Task 14: Docs and handoff

**Files:**
- Modify: `README.md` (rewrite sections 2–4 for the new flow), `CLAUDE.md` (pipeline, waiting items), the spec header (status `구현 중`), `docs/superpowers/specs/2026-11-02-video-composer-mcp-design.md` and `2026-11-07-video-render-design.md` (add one line at the top: `대체됨: 2026-11-12-synthetic-timeline-generation-design.md`)

The new README flow:
1. Sync or recompute ATUS.
2. `ingest nextme --days … --out data/raw/bank/nextme800.draft.jsonl`
3. Review the draft, then `ingest bank --dataset nextme800`.
4. `ingest egoexo` / `ingest ego4d`
5. `npm run generate -- --world config/worlds/example-nextme.json`, or register the MCP server.
6. Note the requirement of at least 28 NextMe days with both home and work clips (D7).

- [ ] **Step 1:** Write the docs in Korean. The command lines must match the CLI usage strings.
- [ ] **Step 2:** Run `npm test && npm run typecheck && python3 -m pytest scripts/`. Expected: PASS.
- [ ] **Step 3:** Commit `docs: describe synthetic timeline pipeline`.

---

## Spec Coverage

| Spec | Task |
|---|---|
| §4.1 contract | 3, 7 |
| §4.2 world | 3 |
| §4.3 schedule | 6 |
| §4.4 gaps | 5, 6 (transition gaps); 10, 12 (unrecorded gaps and coverage ratio) |
| §5.2 steps 1–3, 6, 7 | 5 |
| §5.2 steps 4, 5, 8 | 6 |
| §5.3 exceptions | 5 |
| §5.4 multi_day | schema only (3, 5) |
| §5.5 defaults | 4, 5, 6 |
| §5.6 donors | 4 |
| §6.1 binding | 3, 10 |
| §6.2–6.5 matching | 10 |
| §6.3 MCP tools | 13 |
| §7.1 rule verifier | 11 |
| §7.2 seam VLM | Plan 3 (seam list from 11) |
| §7.3 report | 11, 13 |
| §8 manifest | 12 |
| §9 rendering | Plan 2 |
| §10 errors | throughout |
| §11 tests | in each task |
| §12 same-pool routine ablation | the world dial plus a fixed seed; no extra code |
