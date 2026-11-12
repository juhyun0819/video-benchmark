# Synthetic timeline — controller rulings (SDD ledger export, 2026-11-12)


- R1 Ruling: T1 may trim or delete core/test/fixtures.ts to what remap.test needs (no legacy types in core tests) — T10 rewrites it anyway — cost if wrong: none.
- R2 Ruling: core exports `ClipCategorySchema` / type `ClipCategory` (not `CategorySchema`); wherever the plan says core `Category`, use `ClipCategory` — avoids clash with ingest's config CategorySchema — cost: rename only.
- R3 Ruling: T3 defines `WeekdaySchema = enum("Mon".."Sun")` in world.ts; T5 defines `Meal = "breakfast"|"lunch"|"dinner"` in schedule/types.ts — cost: none.
- R4 Ruling: D1 fallback not implemented; an empty weekend pool throws DONOR_POOL_EMPTY — ATUS pool has 1,313 weekend rows so the path is unreachable in v1 (YAGNI) — cost: a later source without weekend rows needs this added.
- R5 Ruling: T6 adds `ScheduleDefaultsSchema` (parses day_level_sigma, weekend.sports_start, person_level from schedule_defaults_atus2025.json; unknown keys pass through) and `defaults: ScheduleDefaults` to ScheduleInput; T13 loadConfig returns it — cost: none.
- R6 Ruling: in T10 a regen attempt whose generateDay throws SCHEDULE_INFEASIBLE counts as a failed attempt and the loop continues to the next attempt — cost: none.
- R7 Ruling: T2 overlap is half-open `[ws−pad, we+pad)` vs `[bs, be)`; replace the plan's case with `bucketOverlaps("evening",[1000,1030]) === true` (keep the other two) — plan's [1000,1019] contradicts D2 — cost: none.
- R8 Ruling: T8 also removes `base_days` from index.json (and nextme800/days reading) — cost: none.
- R9 Ruling: T5's B=2/B=3 day-type/presence tests run with routine_strength=1 — cost: none.
- R10 Ruling: work in the current checkout on a new branch from Task 0 instead of a separate git worktree — the user's pending work lives here and Task 0 commits it first; branch is not main — cost: none.
- R12 Ruling: in Task 5, a meal whose effective distribution (after D4 renormalization) has zero total mass resolves to "skip" for that meal and is recorded in meal_style_effective as {} — avoids a divide-by-zero; cost: a misconfigured world silently loses that meal instead of failing.

- R13 Ruling: Task 4 DayProfile — work_end filled like other fields if missing; meals always filled; sport never filled (absent when donor has none; Task 6 falls back to defaults sports_start p50); fill draws keyed rngFor(seed,"donor",kind,"fill",field) from the drawn backoff level's pool; unknown occupation_group falls to "all" — cost: small distribution shifts if wrong.

- R14 Ruling: Task 6 sanitizes each day's anchors after jitter and rounds them to integer minutes: breakfast = max(breakfast, wake+10); on workdays work_end = max(work_end, work_start + 240) and lunch is clamped into [work_start+60, work_end−60]; dinner = min(dinner, sleep−60). Donor fills draw meal times independently of the donor's wake/work times, and ATUS has rows with work_end−work_start < 120 (review of Task 4 on the real CSV), so without this many days would be infeasible — cost: a few donors' raw times are bent toward plausibility.

- R15 Ruling: Task 5 dispatch clarifications — exceptions use day_range [d,d] (1-based); slot exception `replaces` = replaced role, day exception `replaces` = "workday"; optional keys are "hobby" (workdays) and "home_evening"; weekend hobby presence is the hobby_day day type; Task 5 also exports effectiveMealStyle(world, boundNodes) for Task 6 to record — cost: none.

- R16 Ruling: on every non-workday day type (hobby_day, home_day, leave_day) the `workplace` meal place is dropped from the meal distribution before renormalization (and recorded in meal_style_effective for weekend) — eating at the workplace on a day with no work slot is a place contradiction, and the spec's first success criterion is minimizing contradictions; the 0.10/0.02 weekend workplace mass in §4.2/§5.5 comes from ATUS weekend workers whom our weekend template does not model — cost: weekend meal-place mix deviates from the ATUS prior.
- R17 Ruling (Task 5 implementer choices accepted): workday optional.home_evening presence p=0.5 (no source; design value); per-decision reference streams rngFor(seed,"reference",refKey,decisionName); weekend hobby node per weekday; weekly_freq 0 hobbies excluded; workday hobby occurrences capped at 5; slot exceptions may be no-ops and override "skip"; exception draws keyed with attempt — cost: small distribution choices, all recorded here.

- R18 Ruling: hobby occurrences cycle over a frequency-expanded, interleaved list (smooth weighted round-robin by weekly_freq desc, then name; e.g. A:3,B:1 → A,A,B,A… per week) instead of plain cycling — plain cycling gives each hobby equal counts and violates per-hobby weekly_freq (§4.2) — cost: different hobby order than the brief's literal text.
- R19 Ruling: decideDay applies a slot exception's place directly into `meal_place[role]` (overriding the routine draw or "skip"); ExceptionCatalogSchema refines slot `type`s to be unique — Task 6 then needs no catalog lookup — cost: none.

- R20 Ruling (Task 6 dispatch): non-workday day types use the weekend donor; leave_day uses the home_day template; chores slot has no anchor and min_minutes 60; work_am/work_pm min_minutes 60, others 10; σ = lerp(routine_strength_0, routine_strength_1, rs); gap draws rngFor(seed,"day",d,attempt,"gap",i); a required slot whose node resolves to nothing is SCHEDULE_INFEASIBLE — cost: tuning values only.

- R21 Ruling: resolveDayWindows drops optional slots that do not fit instead of failing: when the windows are infeasible, drop the last-in-template-order optional slot still present and recompute; repeat until feasible or no optional slot is left; only then throw SCHEDULE_INFEASIBLE (which triggers attempt retries). Dropped roles are recorded on ScheduleDay as `dropped_slots: [{ role, reason: "no_time" }]`. Implementer's real-ATUS sweep showed ~25% of worlds failing, mostly on the optional 90-min workday hobby — a 1-in-4 failure rate breaks the scalability criterion (§1.2-3); this extends D10's "optional slots are dropped" to scheduling — cost: some days lose an evening hobby/home_evening slot (recorded).
- R22 Ruling: the R14 lunch clamp upper bound becomes work_end − 60 − lunch duration (45), so work_pm keeps ≥ 60 min — fixes the "work_pm too short" failures — cost: none.
- R23 Ruling (Task 6 choices accepted): Transition type named SlotTransition (gaps.ts already exports Transition); no gap/transition between consecutive same-node slots; $hobby slots take the hobby node's activity and append its tags; ScheduleDefaults leaves person_level as passthrough; gap fn takes (from,to,index) — cost: none.

- R24 Ruling (amends R21 drop order): drop the failing slot if it is optional; otherwise drop the nearest optional slot before the failing slot (in template order); if none before it, the last optional slot after it — the "last optional first" order dropped home_evening/dinner to rescue work_pm on ~170 sweep days and 4 slots on 101 hobby days — cost: none.
- R25 Ruling: R14 sanitation also clamps the sport anchor to ≤ sleep − hobby duration (120) − 60 — one ATUS weekend donor reports sport at 22:00 with bed 22:50, making the required weekend hobby infeasible for every seed — cost: that donor's late sport moves earlier.

- R26 Ruling: time between consecutive slots beyond the sampled transition (start = max(anchor, prevEnd + gap)) stays outside every slot; Tasks 10/12 take timestamps from slot windows, never from prevEnd + transition minutes, and treat that idle time like unrecorded time (no video, no QA, counted in awake time for coverage_ratio) — deviates from §6.5's literal "next start = prev end + gap", but forcing contiguity would bend donor anchors (e.g. 17:00 work end vs 18:30 dinner) — cost: manifest readers see three kinds of non-video time (transition, in-slot unrecorded, idle).
- R27 Ruling: slot ids are positional among kept slots, so Task 10 regeneration replaces the whole ScheduleDay (incl. day_type/exceptions) and clears every assignment of that day; matching-time drops (D10) are recorded in the timeline as relaxation "dropped_optional", distinct from schedule-time `dropped_slots` — cost: none.

- R28 Ruling: catalog path segments for video ids allow a leading `-` or `_` (pattern `^[A-Za-z0-9_-][A-Za-z0-9._-]*$`, still rejecting `..`); paths are only used through node:fs, never passed to a CLI — real YouTube ids start with `-`/`_` ~3% of the time — cost: none.

- R29 Ruling (Task 8 dispatch): HKT minute-of-day = floor(((parseHkt(start_wallclock)+src_start+8h) mod 86400)/60), no Date getters; `nextme --days … --out <jsonl>` (default data/raw/bank/nextme800.draft.jsonl) writes drafts only — no catalog write or index rebuild; caption ids = `<level>:<id>`; captions overlap rule same as remap; L5 = interval containing block start; draft tags [] — cost: none.

- R30 Ruling: Task 8's "13:23 HKT → afternoon" test expectation was a plan error; under D2, 13:23 (803 min) is midday — assert midday for 13:23 and afternoon for 14:23 — cost: none.

- R31 Ruling (Task 9 dispatch): writeEventCatalog/writeCandidates and candidates counting in rebuildIndex are removed (nothing writes candidates.jsonl after Task 9); index `categories` is derived from clips or dropped; only videos with ≥1 mapped clip get annotation files — cost: none.

- R32 Ruling: Task 10's `already_used` hard filter also rejects a clip whose source span overlaps an already-assigned clip of the same `source.video_id` (Ego4D summaries of one video can yield overlapping walking/transit clips) — reusing the same footage twice is the reuse the spec forbids — cost: none.

- R33 Ruling (Task 10 dispatch): Timeline.slots has an entry per slot of every day; fillTimeline gets `regen` from the caller; jaccard over lowercase alnum tokens (empty∪empty → 0); Selector `recent` = previous 3 days; K=10; placement narrows to window ∩ union of clip buckets only if it still fits, gaps from rngFor(world.seed,"place",slot_id) — cost: tuning only.

- R34 Ruling: fillTimeline regeneration tries attempts day.attempt+1 … day.attempt+5 (not 1..5), so a day the generator already built at attempt k gets five genuinely new attempts — cost: none.
- R35 Ruling (Task 10 choices accepted): filters in match/filters.ts to avoid a timeline↔match import cycle; preflight counts only required-slot days per node; placement narrows to earliest..latest bucket span; R32 compares dataset:video_id; extra exports and UNFILLABLE {day, attempts} details — cost: none.

- R36 Ruling: assignClip removes "dropped_optional" from a slot's relaxations when a clip is added, and rejects "dropped_optional" as an input relaxation — otherwise a later manual/LLM assignment carries a false "dropped" flag into the manifest (Task 12 copies slot relaxations onto clips) — cost: none.

- R37 Ruling (Task 11 dispatch): when the verifier re-checks an assigned clip with hardFilterFailures, it evaluates against unassignClip(t, {slotId, clipId}) so the clip is not counted against itself (already_used / does_not_fit) — cost: none.

- R38 Ruling (Task 11 dispatch): rule definitions — day_type = role set consistent with day type; slot_window = required slots filled and clips inside windows; wake_sleep = windows within [wake, sleep]; exceptions checked by type via optional `opts.exceptions` catalog (overlap/uniqueness always); missing clips reported as violations, not thrown; seam at_s = later clip start — cost: none.

- R39 Ruling (amends R38): slot exceptions carry their place in the record — ExceptionSchema gains optional `meal_place: MealPlace` (set by decideDay for scope "slot"); verifyTimeline drops the optional catalog parameter and checks slot exceptions from the record itself (a slot exception without meal_place is a violation) — a verifier that silently skips checks when a parameter is omitted is a trap for Tasks 12/13, and the manifest gains the information QA needs — cost: one optional field in schedule/manifest exceptions.
- R40 Ruling: swap points follow §7.1 literally — consecutive placed clips of one day (layout order) with the same slot node and different day_group; slot_ids are the pair's slots — cost: none.

- R41 Ruling (Task 12 dispatch): MISSING_CLIPS (sorted ids) before verifying; day clips in layout order with slot relaxations; days carry dropped_slots; versions.extractor = sorted unique used extractor_versions; reintroduce round3/LEVEL_ORDER/caption sort; small golden snapshot — cost: none.

- R42 Ruling (Task 13 dispatch): @vb/mcp depends on @vb/ingest (workspace) to reuse isSafeVideoIdSegment; generate CLI writes nothing until every step succeeds; world copy overwritten only if identical; tool errors "<CODE>: <message>\n<details>", INTERNAL for non-TimelineError; server name stays "video-composer"; config dir via VIDEO_BENCH_CONFIG_DIR / --config — cost: none.

- R43 Ruling (final review I1): ranking adds +0.25 when the clip's day_group is already used today at any node (on top of the existing +0.5 same-node bonus); README/CLAUDE.md state that real NextMe needs comfortably more than 28 usable dates — a synthetic day otherwise retires two source dates and ~30 dates fail — cost: slightly stronger same-source-day preference.
- R44 Ruling (final review I2 + Minor 3): R14 sanitation clamps each day's anchors to wake ≥ 240 (04:00) and sleep ≤ 1679 (27:59); ClockSchema accepts exactly what parseClock accepts (HH 00–27, MM 00–59) — ATUS bed reaches 27:55 and jitter pushes past 28:00, crashing auto_fill for 1–5% of worlds — cost: a few extreme donors' times are bent.
- R45 Ruling (final review I3): when R21/R24 drops a slot that a slot exception replaces, the exception record is removed from that day (the exception did not happen) — otherwise the verifier rejects a generator-produced day at export — cost: realized exception rate slightly below the dial.
- R46 Ruling (final review I4, spec §6.3): list_slot_candidates also returns `recent` (previous 3 days' picks: day, role, clip_id, activity, day_group) and the world's routine_strength — cost: none.
- R47 Ruling (final review I5): `ingest nextme` refuses an existing --out unless --overwrite; README states that `ingest bank` replaces the dataset's whole catalog and that reviewed drafts should be kept (outside data/raw if they must sync) — cost: none.
- R48 Ruling (final review Minor 1): the manifest day gains `dropped_optional: string[]` (roles of slots left empty with relaxation "dropped_optional" at matching time) — cost: none.
- Final review: With fixes — 5 Important (I1–I5) + Minor 1/3 folded; one fix wave dispatched (opus)


## Deferred minors
- Task 0: minor (deferred): CLAUDE.md:15 and README.md:196,200 still name compositions as sync target / pull guard — Task 14 docs
- Task 0: minor (deferred): .gitignore:13 comment lists old folder names (data/ is ignored as a whole — verified)
- Task 1: minor (deferred): README.md:11,140 describe deleted MCP server/command — Task 14
- Task 1: minor (deferred): remap.ts:17 parameter still named `block` (type PlacedSpan) — polish
- Task 1: minor (deferred): legacy.ts is renamed to blocks.ts in Task 9 (owner confirmed); empty @vb/mcp workspace until Task 13
- Task 2: minor (deferred): bucketOverlaps does not normalize windows <240 (callers must use ≥1440 convention); timeOfDayAt ≥1680 branch / out-of-range silently "night"; formatClock negative input malformed
- Task 2: minor (deferred): no golden values for Rng.next(); int/pickWeighted input validation untested; rngFor key join "|" can collide ("a|b" vs "a","b")
- Task 3: minor (deferred): contract-row test asserts only captions length (intersection merge); node_binding keys naming no node accepted; coverage gaps (weekly_freq range, unknown NodeDef kind, NodeBinding, seed sign)
- Task 4: minor (deferred): conditional assert in "missing lunch_time" test; same-seed test can't detect ignored seed; untested branches (weekend no work fields, filled meal no meal_loc); ≥30 count includes w<=0 rows; partial sex column skips levels silently
- Task 4: minor (deferred): short data rows (fewer cells than header) still default silently; sport_time/outdoor required columns slightly strict
- Task 5: minor (deferred): leave_day coverage thin (beyond R16 test); determinism test title overclaims; GapTableSchema accepts self keys; per-day recomputation of hobbyPlan/effectiveMealStyle; day_range start≤end unchecked
- Task 5: minor (deferred): ExceptionCatalog slot replaces_role now MealSchema (only meals replaceable in v1)
- Task 6: minor (deferred): TZ restore sets "undefined" string in generate.test.ts:136-139; no hand-computed weekend layout test; R14 clamps/GAP_MISSING propagation/missing-anchor untested; optional slot with literal unknown node silently dropped (config typo swallowed) and duration<min not rejected; unreachable overlap check windows.ts:86
- Task 7: minor (deferred): empty annotation files for caption-less clips; duplicate caption id with differing content silently dropped; writeClipCatalog doesn't re-check dataset/clip_id uniqueness; CLI bank path untested; nextme800 clip-catalog rewrite removes legacy per-day annotation files (expected with Task 8)
- Task 7: minor (deferred): no RED evidence captured for Task 7 (tests written after code)
- Task 8: minor (deferred): "(unlabeled)" summary becomes activity when L5 matches a place; TZ-independence test only runs under NY; words.ts compiles RegExp per call; main() --out/--rules parsing untested; drafts never emit hobby nodes (Task 10 note)
- Task 8: minor (deferred): no RED evidence; report omits pasted output
- Task 9: minor (deferred): Ego4D activity text may carry "#Summary C" prefix (CLAUDE.md open item); clipmap ego4d ingest test duplicates ego4d.test; stale local data/catalog/hdepic/ and data/catalog/{ego4d,egoexo4d}/candidates.jsonl remain on disk and would be pushed by sync_data.py (catalog folder) — user must delete; surface in final message
- Task 10: minor (deferred): hardFilterFailures counts the checked clip itself (Task 11 must unassign first or exclude); Selector.choose sync vs "LLM through MCP" comment; regen result day_index unchecked; "not_a_candidate" outside HardFilter names; layout hides overruns (free=max(0,…)); per-slot full catalog scans (perf for bulk); Focus-4 test doesn't assert input unchanged
- Task 11: minor (deferred): day exceptions not tied to day_type; no_overlap only consecutive pairs; quadratic usage rebuild; tag failures not reported by design; test gaps (wrong-node exception isolation, unknown type, missing replaced slot, loose seam at_s bounds)
- Task 12: minor (deferred): golden test self-writes when snapshot missing (should fail unless UPDATE flag); coverage gaps (exceptions/relaxations/dropped_slots passthrough, clipped/point captions, multi-dataset licenses, idle-time coverage); per-day filter O(days×clips)
- Task 13: minor (deferred): timeline id collision overwrites silently; generate writes world copy before store.save re-parse; failed_rules derivation duplicated 3×; out-of-range day → INTERNAL; renderDay prints raw minutes not HH:MM; **list_slot_candidates lacks recent picks (§6.3 says LLM sees recent days' choices) — spec gap, triage in final review**; WorldStore.load doesn't check person_id vs file name; list() fails wholesale on one bad file; fsCatalog unsafe videoId returns [] silently; test temp dirs not cleaned
- Task 14: minor (deferred): CLAUDE.md:13 still lists HD-EPIC as data source and omits YouTube; sample section refers to old manifests; README ATUS paragraph lacks raw source/output filename; README Ego-Exo4D "이벤트 후보" wording; commit cbf8fdf trailer says Sonnet 5.5 (others Opus 5.5)
