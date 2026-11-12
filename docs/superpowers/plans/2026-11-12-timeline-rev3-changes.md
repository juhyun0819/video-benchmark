# Timeline rev3 Change Requests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply spec rev3 and change requests CR1–CR8 to the synthetic-timeline pipeline:
- the realized-vs-dial record;
- a required workday dinner with a fixed reduce order;
- the ATUS-based evening and rest-day priors;
- donor redraw instead of clamping;
- three kinds of no-video time;
- strict world validation;
- sport-time fill.

**Architecture:** The changes stay inside the existing packages. `@vb/core` (donor, schedule, manifest, world) takes most of them. `@vb/mcp` gets world parsing and a `stats` CLI. Config moves to `templates.v2.json` and the rev3 `schedule_defaults_atus2025.json`. No new dependencies.

**Tech Stack:** Node 24, TypeScript 5 (ESM, NodeNext, strict), zod 3, vitest.

**Spec:** `docs/superpowers/specs/2026-11-12-synthetic-timeline-generation-design.md`. Rev3 lives in the bundle at `/tmp/claude-3952/-NHNHOME-langlab-zzuhyun-video-benchmark/48bbfc44-42e0-4cc5-8a4f-1622eb5c7396/scratchpad/rev3_bundle/` and is copied into the repo in Task 1.

**Change requests:** `CHANGE_REQUESTS_2026-11-12.md` in the same bundle folder.

**Previous plan and rulings:**
- `docs/superpowers/plans/2026-11-12-synthetic-timeline.md`
- `docs/superpowers/plans/2026-11-12-synthetic-timeline-rulings.md` (R1–R48). The CRs revise R12, R13, R14, R15, R16, R21, R22, R24, R25, R26, R44 and R45. Every other ruling stays.

## Global Constraints

- Same as the previous plan. In short:
  - Code, messages and tests are in English; docs are in Korean.
  - `@vb/core` stays pure.
  - Byte-identical `schedule.json` for identical input, with every draw from `rngFor` keyed per day.
  - Synthetic fixtures only.
  - Never commit `data/`.
  - Commit after each task with `git -c user.name=juhyun0819 -c user.email=p3hyun2@gmail.com commit` and the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never push.
- Values from the rev3 config are read from config, never hard-coded. This covers CR3's `weekday_evening_home.slot_probability` and the `rest_day` priors.
- Failures write nothing and report what is missing (§10). World errors use `TimelineError` code `WORLD_INVALID` and surface as `WORLD_INVALID: <message>`.

## Decisions (where the CRs leave a choice)

| # | Decision | Task |
|---|---|---|
| E1 | Realized slot-exception rate:<br>• `slot_exception_rate` = days with a slot exception ÷ days eligible for one. Eligible means no day exception and at least one feasible candidate slot. This is the number that should match the dial.<br>• The spec's `slot_exception_share` = exception slots ÷ candidate slots is also recorded. | 4 |
| E2 | `composition_consistency` = three values per template, each over all pairs of days that use that template (user-approved, 2026-11-12):<br>• `optional_jaccard`: mean Jaccard of the sets of present optional roles;<br>• `order_similarity`: mean of 1 − normalized edit distance of the role sequences (edit distance ÷ max length);<br>• `start_diff_minutes`: mean absolute start-time difference of slots with the same role, in minutes.<br>Shape: `{ [template]: { optional_jaccard, order_similarity, start_diff_minutes, pairs } }`. | 4 |
| E3 | A slot exception is placed after windows and drops (§5.2 step 8):<br>• Candidates are kept slots whose role is a catalog `replaces_role` for the day type, whose place is usable, and whose day still lays out with the exception's place (re-layout feasible).<br>• Bernoulli(`exception_rate.slot`) decides; the slot is picked uniformly among the candidates.<br>• R45 is removed. | 4 |
| E4 | Reduce order is template config: `reduce_order: [{role, action: "shrink" \| "drop"}]`.<br>• Workday: hobby shrink (90 → `min_minutes` 30), hobby drop, home_evening drop.<br>• home_evening is a fill slot with `min_minutes` 30, the ATUS ≥30-min definition, so "shrink home_evening" happens on its own.<br>• Templates without `reduce_order` keep R24's rule.<br>• Required slots are never shrunk or dropped. | 3 |
| E5 | Donor acceptance (CR5 at donor level), a predicate checked on the donor's raw profile after fills:<br>• wake ≥ 04:00 and bed ≤ 27:59;<br>• breakfast ≥ wake + 10; dinner ≤ bed − 60;<br>• weekday: work_end ≥ work_start + 240; lunch within [work_start + 60, work_end − 60 − 45];<br>• weekday: the template's required slots lay out with median gaps (no jitter);<br>• weekend: sport ≤ bed − 120 − 60.<br>A rejected donor is redrawn from the same backoff level (up to 5). After 5 rejections the last pick is clamped and each clamp is recorded as `{item, from, to}`. | 2, 3 |
| E6 | Day-level jitter (CR5 at day level): the whole jitter vector is redrawn (key `"jitter", k`, k = 0..5) until the E5 bounds hold for the day's anchors. After that it is clamped, and `anchor_adjustments: [{item, from, to}]` is recorded on the ScheduleDay. | 3 |
| E7 | Sport fill (CR8): a donor without sport gets `sport` from weighted rows that have `sport_time` at the same backoff level. With fewer than 30 such rows, it uses all weekend pool rows with sport. Both cases are recorded in `filled` (`"sport_time"`) and `sport_fill: "level" \| "pool"`. The defaults `sports_start` fallback stays only when the pool has no sport rows at all. | 2 |
| E8 | World validation (CR7): every meal distribution in `meal_style` must sum to 1 ± 0.01. `rest_day` must not contain `workplace` (CR4.4). The workday dinner must keep mass after D4 dropping of unbound places, or `WORLD_INVALID` is raised at generation. Hobbies stay at 1–3. `meal_style.weekend` is renamed `rest_day`. | 1 |
| E9 | `no_video_spans` (CR6), built from the windows between wake and sleep:<br>• between `end_i` and `start_{i+1}`: `transition` for the scheduled gap minutes (clipped to the space), then `unscheduled` for the rest;<br>• inside a window: `unrecorded` around the placed clips;<br>• after the last slot, up to sleep: `unscheduled`.<br>The spans plus clips tile [wake, sleep] exactly. Times are seconds since day 1 00:00, rounded to 3 decimals. | 5 |
| E10 | The stats CLI (`npm run stats -- <manifest.json>...`) groups by `(routine_strength, exception_rate.slot, exception_rate.day)` and prints, for each `realized` field, the mean, min and max. It also prints the distribution of donor redraw counts and the share of donors and days with adjustments. The aggregation lives in core as a pure function. | 6 |
| E11 | CR9: the commits used `juhyun0819` and the email of the account running this session, which is the user's own identity. Nothing to change. | — |

## Review Focus

1. **Exception rate:** with `exception_rate.slot = 0.5` over 40 seeds, the mean realized `slot_exception_rate` is within 0.05 of 0.5. Test in Task 4.
2. **Required dinner:** no workday ever lacks a dinner slot with the real templates and pool (sweep of 200 seeds in Task 3's report, plus a unit test).
3. **Tiling:** `no_video_spans` and clip spans tile each day's [wake, sleep] with no gap and no overlap. Test in Task 5.
4. **Redraw before clamp:** a donor out of range is redrawn, not clamped. Only a fixture where all candidates are out of range gets clamped, and the clamp is recorded. Test in Task 2.
5. **Determinism:** schedule bytes stay identical across runs and TZ after all changes. Existing tests are kept green.

---

### Task 1: Rev3 inputs, rest_day priors, world validation (CR4.1–4.4, CR7)

**Files:**
- Copy from the bundle:
  - `docs/superpowers/specs/...design.md`, keeping the repo header `상태: 구현 중`
  - `config/schedule_defaults_atus2025.json`
  - `scripts/atus/atus_extra.py`
  - `data/distributions/atus2025/atus2025_extra.json` (not committed)
- Modify:
  - `packages/core/src/world.ts` (rename `rest_day`, E8 refinements, `parseWorld`)
  - `packages/core/src/errors.ts` (`WORLD_INVALID`)
  - `packages/core/src/schedule/types.ts` (`ScheduleDefaultsSchema` parses `weekday_evening_home` and `meal_location_priors.rest_day`; passthrough otherwise)
  - `packages/core/src/schedule/decisions.ts` (`effectiveMealStyle` returns `{workday, rest_day}`; R16 filtering removed, now enforced by validation)
  - `config/worlds/example-nextme.json` (rest_day values from the rev3 defaults)
  - `packages/mcp/src/store.ts` (`WorldStore.load` uses `parseWorld`)
  - every test fixture world
- Tests: `world.test.ts`, `decisions.test.ts`, `mcp/test/tools.test.ts`

**Interfaces:**
- `parseWorld(json: unknown): World` throws `TimelineError("WORLD_INVALID", "<path>: <message>; ...")`.
- `EffectiveMealStyle = { workday: {lunch, dinner}; rest_day: {breakfast, lunch, dinner} }`.

Steps (TDD):
- [ ] Failing tests:
  - a world with lunch weights summing to 0 → `WORLD_INVALID`; summing to 0.98 → `WORLD_INVALID`; summing to 0.995 → ok;
  - `rest_day.lunch.workplace = 0.1` → `WORLD_INVALID`;
  - 4 hobbies → `WORLD_INVALID`;
  - MCP `create_timeline` on a bad world returns text starting `WORLD_INVALID: `.
- [ ] Implement.
- [ ] Run tests.
- [ ] Commit `feat(core): rest-day meal priors and strict world validation (CR4, CR7)`.

### Task 2: Donor pool, redraw-not-clamp at donor level, sport fill (CR4.5, CR5, CR8)

**Files:** `packages/core/src/donor.ts`; test `donor.test.ts`.

**Interfaces:**
- `pool("weekend")` keeps only `!worked` rows.
- `selectDonors(attrs, src, { seed, allowCountryFallback, accept?: (kind: "weekday" | "weekend", profile: DayProfile) => DonorViolation[] })`, where `DonorViolation = { item: string; bound: number; value: number }`.
- Each `DonorPick` gains:
  - `redraws: number` (0–5)
  - `adjustments: { item: string; from: number; to: number }[]`
  - `sport_fill?: "level" | "pool"`
- E5's generic bounds (wake, bed, breakfast, dinner, work, lunch, sport) live in donor.ts as `donorBoundViolations(kind, profile, lunchMinutes, hobbyMinutes)`.
- The template-dependent check (required slots fit) is passed in by Task 3 through `accept`.
- Redraw draws use key `rngFor(seed, "donor", kind, "accept", k)`. The weekend early-wake redraw (R4 era) stays and runs before acceptance.

Steps (TDD):
- [ ] Failing tests:
  - weekend pool excludes worked rows;
  - a level where the first draw has bed 1700 → redraws are > 0 and the chosen donor is in range, with no adjustments;
  - a level where every row has bed 1700 → redraws = 5 and adjustments `[{item: "sleep", from: 1700, to: 1679}]`;
  - a donor without sport, at a level with ≥ 30 sport rows → `sport_fill: "level"`; with < 30 → `"pool"`; `filled` includes `"sport_time"`;
  - same seed → same picks.
- [ ] Implement.
- [ ] Run tests.
- [ ] Commit `feat(core): redraw out-of-range donors, rest-day weekend pool, sport fill (CR4.5, CR5, CR8)`.

### Task 3: Templates v2, reduce order, day-level redraw, evening probability (CR2, CR3, CR5)

**Files:**
- Create `config/timeline/templates.v2.json` and delete v1.
- Modify:
  - `packages/core/src/schedule/types.ts` (`TemplateSlotSchema` keeps fields; `TemplatesSchema` gets an optional `reduce_order` per day type; `ScheduleDay` gains `anchor_adjustments`)
  - `packages/core/src/schedule/windows.ts` (apply `reduce_order`; R24 fallback)
  - `packages/core/src/schedule/generate.ts` (E6 day-level redraw replaces the R14/R22/R25/R44 clamps; donors are selected with `accept` = E5 template check)
  - `packages/core/src/schedule/decisions.ts` (home_evening probability = `defaults.weekday_evening_home.slot_probability`; `decideDay` gets `defaults` via opts)
  - `packages/mcp/src/config.ts` (loads templates.v2.json)
- Tests: `windows.test.ts`, `generate.test.ts`, `decisions.test.ts`

**templates.v2.json:** the workday changes are:
- `dinner.required: true`
- `hobby.min_minutes: 30`
- `home_evening.min_minutes: 30`
- `reduce_order: [{role: "hobby", action: "shrink"}, {role: "hobby", action: "drop"}, {role: "home_evening", action: "drop"}]`

Weekend templates are unchanged apart from the version (`templates.v2`).

**Reduce algorithm** (windows.ts), when layout fails:
1. Apply the next `reduce_order` step whose role is still kept: shrink sets that slot's duration to its `min_minutes`; drop removes it and records `{role, reason: "no_time"}`.
2. Re-layout.
3. When the steps run out, use R24 among the remaining optional slots.
4. If still infeasible, throw `SCHEDULE_INFEASIBLE`.

Record shrinks as `dropped_slots` entries with `reason: "shrunk"` so `realized` can count them.

Steps (TDD):
- [ ] Failing tests:
  - a workday with work_end 18:30 and sleep 21:30 keeps dinner, shrinks then drops the hobby, keeps home_evening ≥ 30 min, and `dropped_slots` records the steps in order;
  - a workday where even required slots cannot fit throws `SCHEDULE_INFEASIBLE`;
  - home_evening presence frequency over 2000 draws at routine_strength 0 is within ±0.02 of the configured 0.91, and changing the config to 0.5 changes it;
  - a jitter that pushes wake below 04:00 is redrawn (no `anchor_adjustments`) while forcing σ huge yields recorded adjustments;
  - `generateSchedule` with a donor whose required workday slots cannot fit triggers a donor redraw (`donors.weekday.redraws > 0`).
- [ ] Implement.
- [ ] Run tests.
- [ ] Run a local sweep (real config and pool, 200 seeds × routine_strength {0, 0.5, 1}; not committed) and report:
  - failures;
  - workday days without dinner (must be 0);
  - dropped and shrunk counts by role;
  - donor redraw distribution;
  - adjustment shares.
- [ ] Commit `feat(core): required workday dinner, reduce order and redraw-not-clamp (CR2, CR3, CR5)`.

### Task 4: Slot exceptions after windows; realized block (CR1)

**Files:**
- `packages/core/src/schedule/decisions.ts` (stop drawing slot exceptions)
- `packages/core/src/schedule/generate.ts` (E3 placement after windows; remove R45; compute `realized`)
- `packages/core/src/schedule/types.ts` (`Schedule.realized`)
- Tests: `generate.test.ts`, `decisions.test.ts`, `verify.test.ts`

**Interfaces:**
- `Schedule.realized` contains:
  - `slot_exception_rate`
  - `slot_exception_share`
  - `slot_exception_eligible_days`
  - `day_exception_rate` (day exceptions ÷ days whose base type has a day-exception entry)
  - `optional_presence: Record<role, number>` (kept ÷ days whose template has the role)
  - `shrunk: Record<role, number>`
  - `composition_consistency` (E2)
  - `dials: { routine_strength, exception_rate }`, echoed
- `realizedOf(days, world)` is exported for Task 5.
- Slot-exception draw keys: `rngFor(seed, "day", d, attempt, "slot_exception")` for the Bernoulli and `"slot_exception_pick"` for the choice.

Steps (TDD):
- [ ] Failing tests:
  - **Review Focus 1:** a world with a bound eat_out place, `exception_rate.slot = 0.5`, 40 seeds → mean `slot_exception_rate` within 0.05 of 0.5;
  - every slot exception points at an existing slot whose node is the place node;
  - `realized.slot_exception_share` and `optional_presence` equal counts recomputed directly from `days`;
  - **E2 monotonicity:** generating the same world at routine_strength 0, 0.5 and 1 (several seeds, averaged) makes `optional_jaccard` and `order_similarity` non-decreasing and `start_diff_minutes` non-increasing for the workday template. If this fails, the metric does not reflect the dial — report it rather than tuning the test;
  - day exceptions are unchanged;
  - determinism tests still pass.
- [ ] Implement.
- [ ] Run tests.
- [ ] Commit `feat(core): place slot exceptions after reduction and record realized dials (CR1)`.

### Task 5: Manifest — no_video_spans, realized, donor_adjustments (CR1.3, CR5.2, CR6)

**Files:** `packages/core/src/manifest.ts`; test `manifest.test.ts` plus the golden snapshot (regenerate with `UPDATE_GOLDEN=1`, intentionally).

**Interfaces:**
- Each manifest day gains:
  - `no_video_spans: { start: number; end: number; kind: "transition" | "unrecorded" | "unscheduled" }[]` (E9)
  - `anchor_adjustments`
  - `shrunk`, from the schedule day
- Top level gains:
  - `realized`: Schedule.realized plus `matching_dropped_slots` (count of slots with relaxation `dropped_optional`) and `matching_dropped_by_role`
  - `donor_adjustments`: `{ weekday: {case_id, backoff_level, redraws, adjustments, filled, sport_fill?}, weekend: {...} }`

Steps (TDD):
- [ ] Failing tests:
  - **Review Focus 3:** for every day in the fixture, the sorted clip spans and no_video_spans tile [wake·60, sleep·60] exactly (offset by day), with no gaps or overlaps, and each kind appears where expected (a transition after a node change, unscheduled before an anchored dinner, unrecorded inside a slot);
  - `realized.matching_dropped_slots` equals the count of `dropped_optional` slots;
  - the golden snapshot is updated.
- [ ] Implement.
- [ ] Run tests.
- [ ] Commit `feat(core): manifest no-video spans, realized dials and donor adjustments (CR1, CR5, CR6)`.

### Task 6: Stats CLI (CR1.4, CR5.3)

**Files:**
- Create `packages/core/src/stats.ts` (pure `summarizeManifests(ms: TimelineManifest[]): StatsReport`) and `packages/mcp/src/stats.ts` (CLI that reads paths, parses with `TimelineManifestSchema`, prints JSON).
- Modify `package.json` (`"stats": "tsx packages/mcp/src/stats.ts"`).
- Tests: `core/test/stats.test.ts`, `mcp/test/stats.test.ts`

**Interfaces:**
- `StatsReport = { groups: { dials: {...}; n: number; realized: Record<field, {mean, min, max}> }[]; donors: { redraws_histogram: Record<"0".."5", number>; adjusted_share: number }; days: { anchor_adjusted_share: number } }`
- `Record<role, number>` fields are flattened as `field.role`.

Steps (TDD):
- [ ] Failing tests:
  - two synthetic manifests with different dials form two groups with exact means;
  - the histogram counts;
  - the CLI prints valid JSON and exits 1 naming an unreadable path.
- [ ] Implement.
- [ ] Run tests.
- [ ] Commit `feat: stats report over manifests (CR1, CR5)`.

### Task 7: Before/after comparison report (user request, 2026-11-12)

**Files:**
- Create `scripts/compare/rev3_compare.ts`, a committed script that takes a repo root, a data dir and seeds, and runs the pipeline. It imports core from `<root>/packages/core/src/index.ts`, so it runs against either tree.
- Create `docs/superpowers/reports/2026-11-12-rev3-comparison.md` (Korean report).

**Method:**
- The baseline is commit `ea048f5`, checked out with `git worktree add` into a scratchpad folder and removed afterwards. The new tree is HEAD.
- Both trees run on the same seeds (200 seeds × routine_strength {0, 0.5, 1}, `exception_rate {slot: 0.5, day: 0.1}`), the example world with a bound eat_out frequent_place, and the real ATUS pool and config.
- Two levels:
  - Schedule level: every seed.
  - Full generate (fill → verify → manifest): 20 seeds, on the synthetic bank from the MCP test helpers, written to a temp data dir. No real clips are needed.

**Metrics** (old vs new, with counts):
- **Dropped optional slots:** share of optional slots dropped at schedule time (previously ~17% of days), split by role, plus matching-time `dropped_optional` from the full runs.
- **Workday dinner:** workday days without a dinner slot (target 0).
- **Exception rate:** realized slot-exception rate vs dial (0.5), computed the same way for both trees (E1 definition, counted from the schedules), plus the day-exception rate.
- **Donor redraws:** redraw count distribution and adjustment shares, donor level and day level.
  - The old tree has no redraws. Report how many of its selected donors and jittered days violate the E5/E6 bounds, using the new `donorBoundViolations` applied to the old schedules' recorded donor profiles and anchors.
- **Generation failures:** count in each tree.

- [ ] Write the script.
- [ ] Run both trees.
- [ ] Write the report: method, tables, short interpretation, and anything that did not meet its target.
- [ ] Commit the script and report. Do not commit generated data.

### Task 8: Docs

**Files:** `README.md`, `CLAUDE.md`, `docs/superpowers/plans/2026-11-12-synthetic-timeline-rulings.md` (append a section "rev3 반영" listing which rulings the CRs replaced, plus E1–E11).

The rulings section must state the E4 note: because home_evening fills until sleep, the 0.91 presence probability only matters on the days the slot is dropped (when it is absent, that time becomes unscheduled time at home). Link the comparison report from Task 7.

Content:
- Korean.
- The `npm run stats` usage.
- templates.v2.
- The rest_day world field (`meal_style.rest_day`) and how validation errors look.
- The three no-video kinds.
- CR9: no change needed.

- [ ] Write.
- [ ] Run `npm test && npm run typecheck && python3 -m pytest scripts/`.
- [ ] Commit `docs: rev3 timeline changes`.
