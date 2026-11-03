# Video Composer MCP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an npm-workspaces TypeScript project. It ingests NextMe-800 / HD-EPIC / Ego4D annotations into a local catalog and exposes an MCP server. Through that server a user composes a base day (L4 blocks) with inserted event blocks and exports a manifest whose captions are remapped to the composed timeline.

**Architecture:** There are three packages.
- `@vb/core` holds pure functions: zod schemas, tiling, composition ops, timeline, remap and manifest. It does no I/O.
- `@vb/ingest` is a CLI that writes `data/catalog/`.
- `@vb/mcp` is a stdio MCP server. It reads the catalog, persists compositions and writes exports.

Packages import each other's TypeScript source directly (`exports` → `src/index.ts`) and run through `tsx`/`vitest`. There is no build step.

**Tech Stack:** Node 24 LTS (via nvm), TypeScript 5, npm workspaces, zod 3, `@modelcontextprotocol/sdk` 1.x, `csv-parse`, vitest, tsx. Python (`uv run --with pandas`) is used only for `scripts/convert_hdepic.py`.

**Spec:** `docs/superpowers/specs/2026-11-02-video-composer-mcp-design.md`

## Global Constraints

- Node ≥ 20 (spec). Use Node 24 LTS and pin it in `.nvmrc`. The machine currently has Node 12, so Task 2 installs nvm + Node 24 in user space. No Docker.
- ESM everywhere (`"type": "module"`). TS `module`/`moduleResolution`: `NodeNext`, `strict: true`. Relative imports use `.js` suffixes.
- `@vb/core` must not import `node:fs`, `node:path`, `fetch`, or any other I/O. Id generation and `now` are injected.
- Times are seconds (float) from source-video start (`src_t`). NextMe wall-clock strings are HKT (UTC+08:00). Parsing must not depend on the machine's `TZ`.
- All manifest time values are rounded to 3 decimals (`Math.round(x*1000)/1000`).
- Caption text and `original` records are copied verbatim, never rewritten.
- Ego4D point narrations stay `kind: "point"`. Never convert them to intervals.
- `data/` is gitignored and never committed. Test fixtures are **synthetic** and mimic real formats. Never commit rows copied from gated datasets.
- User-facing MCP error messages are in Korean, matching the spec's examples.
- Commit after each task. Messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Annotations touching block edges.** A point at exactly `src_end` is excluded and a point at `src_start` is included. A zero-length interval (`start == end`) behaves like a point. An interval ending exactly at `src_start` is excluded. Covered in Task 5 `remap.test.ts`.
2. **NextMe captions/L4 outside the recording span.** Captions with `approx_time`, or that start before or end after the recording, are clipped to the block and do not crash. L4 starts outside `(0, duration)` are ignored by tiling. Covered in Task 3 and Task 7.
3. **Stale composition after re-ingest.** A candidate referenced by a composition is no longer in the catalog. `get_composition` and `export_manifest` return an error that lists every missing id; they do not crash or skip silently. Covered in Task 5 and Task 12.
4. **Day with no L4 at all, or a recording shorter than 60 s.** Each recording becomes one `(unlabeled)` block, and a sub-60 s recording stays its own block because it cannot merge across recordings. A requested day with zero recordings makes ingest fail with a clear message. Covered in Task 3 and Task 7.
5. **Machine timezone.** `parseHkt` gives identical results under `TZ=UTC` and `TZ=America/New_York`. Covered in Task 7.

---

## File Structure

```
.nvmrc                         "24"
package.json                   workspaces + scripts (test, typecheck, ingest, mcp)
tsconfig.json                  single root config covering packages/*/src and packages/*/test
vitest.config.ts               include packages/*/test/**/*.test.ts
config/categories.json         event categories (Ego4D scenario/keyword mapping)
scripts/convert_hdepic.py      HD-EPIC pkl → JSONL (one-off)
README.md                      setup, ingest, MCP registration
packages/core/src/
  schema.ts       zod schemas + inferred types (spec §5.2, §5.3)
  catalog.ts      Catalog interface (read port)
  errors.ts       CompositionError + codes
  tiling.ts       tileRecording()
  composition.ts  createComposition / insertEvent / moveEvent / removeEvent / assertRevision
  timeline.ts     resolveTimeline()
  remap.ts        remapAnnotations()
  manifest.ts     buildManifest(), LICENSES, LEVEL_ORDER
  index.ts        re-exports
packages/core/test/  fixtures.ts + one test file per src module
packages/ingest/src/
  time.ts         parseHkt()
  jsonl.ts        readJsonl() with line-numbered IngestFormatError, IngestReport
  writer.ts       CatalogWriter + rebuildIndex()
  nextme/hf.ts         HF file fetcher (injectable fetch)
  nextme/adapter.ts    parseRecordingsCsv, buildNextmeDay
  hdepic/adapter.ts    parseHighLevel, buildHdepicCatalog
  ego4d/adapter.ts     listScenarios, buildEgo4dCatalog
  categories.ts   categories.json schema + loader
  cli.ts          parseArgs dispatcher
packages/ingest/test/  fixtures/ (synthetic files) + tests
packages/mcp/src/
  fsCatalog.ts    FsCatalog implements Catalog (+ list helpers)
  store.ts        CompositionStore (atomic JSON writes), export writing
  render.ts       renderTimetable()
  server.ts       createServer(dataDir, opts) registering all tools
  main.ts         stdio entry
packages/mcp/test/   fixture catalog builder + integration tests
```

---

### Task 1: Verify source formats (no code)

Spec §11 items 1, 2 and 4. Ego4D (item 3) is handled in Task 13.

**Files:**
- Modify: `docs/superpowers/specs/2026-11-02-video-composer-mcp-design.md` (append "확인 결과" under §11)

**Interfaces:**
- Produces: the confirmed facts later tasks read.
  - `RECORDINGS_COLUMNS`: the column names for recording id, start time and duration/end in `raw/recordings.csv`, plus their format.
  - Whether `L1..L5.jsonl` live in `captions/1_nextme/<recording_id>/`.
  - The HD-EPIC high-level file that carries segment start/end, with its columns and time format.
  - Whether pandas can unpickle `HD_EPIC_Narrations.pkl`.

- [ ] **Step 1: Check NextMe access.** Run `curl -sI -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/datasets/mmm8383/NextMe-800/resolve/main/raw/recordings.csv`. Expected: `HTTP/2 200` (or a 302 to a CDN). On 401/403, stop and ask the user for an `HF_TOKEN` with gated access approved.
- [ ] **Step 2: Inspect NextMe.** Download `raw/recordings.csv` (header + 3 rows) and one recording's `raw_caption.jsonl` and `L4.jsonl` (3 lines each) into `data/samples/`. List that recording's caption directory with the HF API (`https://huggingface.co/api/datasets/mmm8383/NextMe-800/tree/main/captions/1_nextme/<recording_id>`).
- [ ] **Step 3: Inspect HD-EPIC.** Run `git clone --depth 1 https://github.com/hd-epic/hd-epic-annotations data/raw/hd-epic-annotations`. Read the header of `high-level/P01_recipe_timestamps.csv` (or whichever high-level file has times). Then run `uv run --with pandas python -c "import pandas as pd; df=pd.read_pickle('data/raw/hd-epic-annotations/narrations-and-action-segments/HD_EPIC_Narrations.pkl'); print(df.dtypes); print(df.head(2).to_dict('records'))"`.
- [ ] **Step 4: Record findings.** Append "### 확인 결과 (2026-11-03)" to spec §11 with the exact column names and formats. **If the HD-EPIC high-level data has no start/end times, stop and ask the user** which fallback from §11.2 to use before Task 8.
- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-11-02-video-composer-mcp-design.md
git commit -m "docs: record verified source formats"
```

---

### Task 2: Workspace scaffold + core schemas

**Files:**
- Create: `.nvmrc`, `package.json`, `tsconfig.json`, `vitest.config.ts`
- Create: `packages/{core,ingest,mcp}/package.json`
- Create: `packages/core/src/{schema,catalog,errors,index}.ts`
- Test: `packages/core/test/schema.test.ts`, `packages/core/test/fixtures.ts`

**Interfaces:**
- Produces (`@vb/core`):
  - zod schemas `DatasetSchema`, `SourceRefSchema`, `TimeSpanSchema`, `AnnotationSchema`, `BlockCandidateSchema`, `BaseDaySchema`, `EventRefSchema`, `CompositionSchema`, `ManifestSchema`.
  - Inferred types with the same names minus `Schema`.
  - All fields exactly as spec §5.2/§5.3, with `Composition.schema_version: z.literal(1)` and `Manifest.schema_version: z.literal(1)`.
  - `interface Catalog { getBaseDay(day: string): BaseDay | undefined; getCandidate(id: string): BlockCandidate | undefined; getAnnotations(dataset: Dataset, videoId: string): Annotation[] }`
  - `class CompositionError extends Error { code: CompositionErrorCode; details: Record<string, unknown> }`, where `CompositionErrorCode = "BASE_DAY_NOT_FOUND" | "SLOT_OUT_OF_RANGE" | "POSITION_OUT_OF_RANGE" | "CANDIDATE_NOT_FOUND" | "NOT_EVENT" | "REF_NOT_FOUND" | "REVISION_MISMATCH" | "MISSING_CANDIDATES" | "COMPOSITION_NOT_FOUND"`
  - `Catalog.getCandidate` resolves both event candidates and base-day blocks, so `NOT_EVENT` is reachable.
  - `packages/core/test/fixtures.ts` exports `makeFixtureCatalog(): Catalog & { baseDay: BaseDay }`. It returns base day `"2026-04-15"` with 3 blocks: `[0,600]`, `[600,1500]` of `rec-A`, and `[0,300]` of `rec-B`. It also holds two event candidates:
    - `hdepic:P01-x:10.000-70.000` (category `cooking`, 60 s, interval narrations)
    - `ego4d:uid-1:0.000-120.000` (category `walking`, 120 s, point narrations + 1 summary)

    Later tasks reuse it.

- [ ] **Step 1: Install Node 24 (user space).** Run `curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash`, then `. ~/.nvm/nvm.sh && nvm install 24 && nvm use 24 && node --version`. Expected: `v24.x`. Write `24` to `.nvmrc`.
- [ ] **Step 2: Scaffold.**
  - Root `package.json`: `"private": true`, `"type": "module"`, `"workspaces": ["packages/*"]`, `"engines": {"node": ">=20"}`.
  - Root scripts: `"test": "vitest run"`, `"typecheck": "tsc --noEmit"`, `"ingest": "tsx packages/ingest/src/cli.ts"`, `"mcp": "tsx packages/mcp/src/main.ts"`.
  - Each package `package.json` has `"name": "@vb/<pkg>"`, `"type": "module"`, `"exports": "./src/index.ts"`. Entry points (`cli.ts`, `main.ts`) are never re-exported, so importing a package has no side effects.
  - `vitest.config.ts` sets `test.env = { TZ: "America/New_York" }` so timezone bugs surface.
  - Dev deps at root: `typescript@^5`, `vitest@^3`, `tsx@^4`, `@types/node@^24`.
  - Deps: core `zod@^3.25`; ingest `@vb/core`, `zod`, `csv-parse@^5`; mcp `@vb/core`, `zod`, `@modelcontextprotocol/sdk@^1`.
  - Run `npm install`.
- [ ] **Step 3: Write failing test `schema.test.ts`:**
  - `parses fixture catalog base day and candidates`: every fixture object passes its schema.
  - `rejects interval annotation without end`: `AnnotationSchema.safeParse({...valid, time: {kind: "interval", start: 1}}).success === false`.
  - `rejects composition with schema_version 2`.
- [ ] **Step 4: Run** `npm test -- schema`. Expected: FAIL (modules missing).
- [ ] **Step 5: Implement** `schema.ts`, `catalog.ts`, `errors.ts`, `index.ts` and `test/fixtures.ts` per the Interfaces above. `Annotation.level` is `z.enum(["raw","L1","L2","L3","L4","L5","narration","summary"])`.
- [ ] **Step 6: Run** `npm test && npm run typecheck`. Expected: PASS, no type errors.
- [ ] **Step 7: Commit**

```bash
git add .nvmrc package.json package-lock.json tsconfig.json vitest.config.ts packages
git commit -m "feat(core): scaffold workspace and core schemas"
```

---

### Task 3: Base-day tiling

**Files:**
- Create: `packages/core/src/tiling.ts`
- Test: `packages/core/test/tiling.test.ts`

**Interfaces:**
- Consumes: `BlockCandidate`, `SourceRef`, `Annotation` (Task 2).
- Produces: `tileRecording(input: { source: SourceRef; duration: number; l4: Annotation[] }): BlockCandidate[]`. Every block has `role: "base"`, `category: "base"` and `id = \`${dataset}:${video_id}:${start.toFixed(3)}-${end.toFixed(3)}\``. Blocks are returned in time order.

Algorithm (spec §6.2):
1. Boundaries are `{0, duration}` ∪ the L4 `time.start` values strictly inside `(0, duration)`, sorted and unique.
2. Each block spans two consecutive boundaries.
3. Repeat until stable: a block shorter than 60 s merges into the previous block. If it is the first block, it merges into the next one. A lone block is never merged.
4. The summary is the `text` of the L4 whose interval contains the block's start (`start ≤ t < end`). If no L4 contains it, the summary is `"(unlabeled)"`.

- [ ] **Step 1: Write failing tests:**
  - `tiles at L4 starts covering whole recording`: duration 3600, L4 starts at 300 and 1800 → spans `[0,300],[300,1800],[1800,3600]`.
  - `first block before any L4 is unlabeled`: the summary of `[0,300]` is `"(unlabeled)"`.
  - `merges sub-60s block into previous`: L4 starts 300, 330 → `[0,300],[300,3600]`.
  - `merges sub-60s first block into next`: L4 start 30 → `[0,3600]`.
  - `recording shorter than 60s stays one block`: duration 40 → one block `[0,40]`.
  - `no L4 → single unlabeled block`.
  - `ignores L4 starts outside (0, duration)`: starts −5 and 4000 with duration 3600 → `[0,3600]`.
- [ ] **Step 2: Run** `npm test -- tiling`. Expected: FAIL.
- [ ] **Step 3: Implement `tileRecording`** in `tiling.ts`.
- [ ] **Step 4: Run** `npm test -- tiling`. Expected: PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(core): tile recordings into base blocks at L4 boundaries"`

---

### Task 4: Composition editing ops

**Files:**
- Create: `packages/core/src/composition.ts`
- Test: `packages/core/test/composition.test.ts`

**Interfaces:**
- Consumes: `Catalog`, `Composition`, `BaseDay`, `CompositionError` (Task 2).
- Produces:
  - `createComposition(baseDay: BaseDay, opts: { genId: () => string; name?: string }): Composition`: `revision: 0`, with `baseDay.blocks.length + 1` empty slots.
  - `insertEvent(comp, catalog, args: { slot: number; candidateId: string; position?: number }, genId: () => string): Composition`: `position` defaults to the end of the slot.
  - `moveEvent(comp, args: { refId: string; toSlot: number; toPosition?: number }): Composition`
  - `removeEvent(comp, args: { refId: string }): Composition`
  - `assertRevision(comp: Composition, expected?: number): void`: throws `REVISION_MISMATCH` with `details.current`.
  - All ops are immutable: they return a new object with `revision + 1` and never mutate the input.

- [ ] **Step 1: Write failing tests** (use `makeFixtureCatalog`):
  - `creates composition with n+1 empty slots`: 3 blocks → `slots.length === 4`, `revision === 0`.
  - `inserts event at end of slot and bumps revision`.
  - `inserts at explicit position`.
  - `allows same candidate twice with distinct ref_ids`.
  - `rejects slot out of range` (slot 4) → `code === "SLOT_OUT_OF_RANGE"`, `details` deep-equals `{ slot: 4, max: 3 }`.
  - `rejects position beyond slot length` → `POSITION_OUT_OF_RANGE`.
  - `rejects unknown candidate` → `CANDIDATE_NOT_FOUND`.
  - `rejects base candidate` → `NOT_EVENT`.
  - `moves event across slots and within a slot`.
  - `move/remove unknown ref` → `REF_NOT_FOUND`.
  - `does not mutate input`: `structuredClone(before)` deep-equals the original after an op.
  - `assertRevision mismatch` → `REVISION_MISMATCH`, `details.current === comp.revision`. `assertRevision(comp, undefined)` does not throw.
- [ ] **Step 2: Run** `npm test -- composition`. Expected: FAIL.
- [ ] **Step 3: Implement** the ops in `composition.ts`.
- [ ] **Step 4: Run** `npm test -- composition`. Expected: PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(core): composition editing ops with invariants"`

---

### Task 5: Timeline + remap

**Files:**
- Create: `packages/core/src/timeline.ts`, `packages/core/src/remap.ts`
- Test: `packages/core/test/timeline.test.ts`, `packages/core/test/remap.test.ts`

**Interfaces:**
- Consumes: Tasks 2 and 4.
- Produces:
  - `type TimelineBlock = { block_id: string; role: "base" | "event"; category: string; summary: string; source: SourceRef; src_start: number; src_end: number; composed_start: number; composed_end: number; slot?: number; ref_id?: string; base_index?: number }`
  - `resolveTimeline(comp: Composition, catalog: Catalog): TimelineBlock[]`
    - Order: slot0 events → base0 → … → slotN events.
    - `block_id` is `base:<day>:<idx>` or `evt:<ref_id>`.
    - Throws `BASE_DAY_NOT_FOUND`, or `MISSING_CANDIDATES` with `details.ids: string[]` listing **all** missing ids.
  - `type ManifestCaption = Manifest["tracks"]["captions"][number]`
  - `remapAnnotations(block: TimelineBlock, annotations: Annotation[]): ManifestCaption[]`
    - Composed values are unrounded here; rounding happens in Task 6.

Remap rules (spec §7.3):
- `composed = block.composed_start + (src_t − block.src_start)`.
- An interval is kept when `start < src_end && end > src_start`. It is clipped to `[src_start, src_end]`, with `clipped.start`/`clipped.end` set on the side that was cut.
- A point is kept when `src_start ≤ t < src_end`.
- A zero-length interval (`start == end`) follows the point rule.
- `provenance.src_time` is the unclipped original. `provenance.orig_id = annotation.id`. `provenance.original = annotation.original`. When `original` has string `start`/`end` (NextMe), they are copied to `provenance.orig_time_string`.

- [ ] **Step 1: Write failing tests:**
  - timeline: `orders slot events before following base block`. Insert cooking (60 s) in slot 1 → blocks `base:2026-04-15:0 [0,600]`, `evt:<ref> [600,660]`, `base:2026-04-15:1 [660,1560]`, `base:2026-04-15:2 [1560,1860]`.
  - timeline: `empty composition equals base blocks back to back`.
  - timeline: `lists all missing candidate ids`. Two refs to ids absent from the catalog → `MISSING_CANDIDATES` with both ids.
  - remap: `shifts interval by block offset`. Block src `[600,1500]` composed `[660,1560]`; annotation `[700,710]` → `[760,770]`.
  - remap: `clips interval crossing block start and marks clipped.start`. `[590,610]` → `[660,670]`, `clipped: {start: true}`, `provenance.src_time` = `[590,610]`.
  - remap: `clips interval crossing block end`.
  - remap: `point at src_end excluded, at src_start included`.
  - remap: `interval ending exactly at src_start excluded`.
  - remap: `zero-length interval treated as point`.
  - remap: `copies text and original verbatim and NextMe time strings into provenance`.
- [ ] **Step 2: Run** `npm test -- timeline remap`. Expected: FAIL.
- [ ] **Step 3: Implement** `resolveTimeline` and `remapAnnotations`.
- [ ] **Step 4: Run** `npm test -- timeline remap`. Expected: PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(core): resolve timeline and remap annotations"`

---

### Task 6: Manifest build

**Files:**
- Create: `packages/core/src/manifest.ts`
- Test: `packages/core/test/manifest.test.ts`, `packages/core/test/__snapshots__/` (vitest file snapshot `golden.manifest.json`)

**Interfaces:**
- Consumes: Task 5.
- Produces:
  - `buildManifest(comp: Composition, catalog: Catalog, now: Date): Manifest`. The result is validated with `ManifestSchema.parse`.
  - `LEVEL_ORDER = ["L5","L4","L3","L2","L1","raw","summary","narration"] as const`
  - `LICENSES: Record<Dataset, { license: string; note?: string }>` =
    - `nextme800`: `"CC BY-NC 4.0"`, note `"Research/evaluation only; gated (mmm8383/NextMe-800)"`
    - `hdepic`: `"HD-EPIC dataset terms"`, note `"https://hd-epic.github.io/site/"`
    - `ego4d`: `"Ego4D License Agreement"`, note `"https://ego4d-data.org/"`

Rules:
- `duration` is the last block's `composed_end` (0 if there are no blocks).
- Captions are sorted by composed start (`time.start` or `time.t`), then by `LEVEL_ORDER` index, then by `id`.
- Every number under `blocks[].composed_*`, `tracks.captions[].time`, and `duration` is rounded to 3 decimals.
- `licenses` lists only the datasets used, in `Dataset` enum order.
- `created_at = now.toISOString()`.

- [ ] **Step 1: Write failing tests:**
  - `golden manifest`: fixture catalog, composition with cooking in slot 1 and walking in slot 3, `now = new Date("2026-11-03T00:00:00Z")`, and fixed genId (`"r1"`, `"r2"`, …) → `expect(JSON.stringify(m, null, 2)).toMatchFileSnapshot("__snapshots__/golden.manifest.json")`.
  - `every caption lies within its block`: for each caption, `block.composed_start ≤ start` and `end ≤ block.composed_end`.
  - `duration equals sum of block lengths`.
  - `removing all events yields base captions offset only`: a composition with no events gives the same base captions with `composed == src` for block 0.
  - `licenses only for used datasets`: a base-only composition → just `nextme800`.
  - `fails with MISSING_CANDIDATES when catalog lost a candidate`.
- [ ] **Step 2: Run** `npm test -- manifest`. Expected: FAIL.
- [ ] **Step 3: Implement** `buildManifest`. Export everything from `index.ts`.
- [ ] **Step 4: Run** `npm test -- manifest`. Expected: PASS. Open `golden.manifest.json` and check by eye that `evt:r1` sits at `[600,660]` and its narrations are shifted by +590.
- [ ] **Step 5: Commit** — `git commit -m "feat(core): build manifest with deterministic ordering"`

---

### Task 7: Ingest — NextMe-800 adapter, writer, CLI

**Files:**
- Create: `packages/ingest/src/{time,jsonl,writer,cli,index}.ts` (`index.ts` re-exports `CatalogWriter`, `rebuildIndex` and the adapters), `packages/ingest/src/nextme/{hf,adapter}.ts`
- Test: `packages/ingest/test/{time,jsonl,nextme,writer}.test.ts`, `packages/ingest/test/fixtures/nextme/` (synthetic: `recordings.csv`, `2026-04-15_0900-1000/{raw_caption,L1,L2,L3,L4,L5}.jsonl`, `2026-04-15_1100-1105/...`)

**Interfaces:**
- Consumes: `tileRecording`, schemas (core).
- Produces:
  - `parseHkt(s: string): number` returns epoch seconds for `"YYYY-MM-DD HH:MM:SS"` in UTC+08:00. Implement it as `Date.UTC(...)/1000 − 8*3600` and throw on a malformed string.
  - `readJsonl<T>(text: string, file: string, schema: z.ZodType<T>): T[]` throws `IngestFormatError` (`file`, `line`, `issue`) on invalid JSON or schema mismatch.
  - `type IngestReport = { ingested: Record<string, number>; skipped: Record<string, number> }` with skip reasons `"invalid_time"` and `"end_before_start"`.
  - `parseRecordingsCsv(text: string): RecordingInfo[]`, where `RecordingInfo = { recording_id: string; day: string; start_epoch: number; end_epoch: number; tar: string }`. Column names come from Task 1's `RECORDINGS_COLUMNS`, and `tar` comes from `tar_root_dir`.
  - `buildNextmeDay(day: string, recordings: RecordingInfo[], files: (recId: string, name: string) => string, report: IngestReport): { baseDay: BaseDay; annotations: Map<string, Annotation[]> }`.
    - Annotation ids are `nextme800:<rec>:raw:<seg_id>` and `nextme800:<rec>:<Lk>:<id>`.
    - Raw text is `action`. Lk text is `text`.
    - `original` is the whole source row.
    - `src_t = parseHkt(x) − start_epoch`.
  - `makeHfFetcher(token: string, fetchImpl = fetch): (path: string) => Promise<string>` uses `https://huggingface.co/datasets/mmm8383/NextMe-800/resolve/main/${path}` with `Authorization: Bearer`. On 401/403 it throws `"HF 접근 거부: NextMe-800 접근 승인과 HF_TOKEN을 확인하세요"`.
  - `class CatalogWriter { constructor(dataDir: string); writeBaseDay(d: BaseDay): void; writeAnnotations(dataset: Dataset, videoId: string, a: Annotation[]): void; writeCandidates(dataset: Dataset, c: BlockCandidate[]): void }`. It writes the paths in spec §6.5.
  - `rebuildIndex(dataDir: string, now: Date): void` scans `data/catalog` and writes `index.json` = `{ updated_at, base_days: string[], categories: Record<string, number> }`.
  - CLI: `npm run ingest -- nextme --days <d1,d2>`. It reads `HF_TOKEN` (missing → exit 1 with a message), fetches, builds, writes, rebuilds the index and prints the `IngestReport`.

- [ ] **Step 1: Write failing tests:**
  - time: `parses HKT independent of TZ`. `parseHkt("2026-04-15 09:00:00") === 1776214800`. The workers run with `TZ=America/New_York` (Task 2), so a local-time implementation fails here.
  - time: `throws on malformed`.
  - jsonl: `reports file and line on schema mismatch`. Line 2 is missing `start` → error message contains `L4.jsonl:2`.
  - nextme: `builds base day from fixture`. Two recordings → blocks tile each recording, and recording `1100-1105` (300 s, no L4) gives one `(unlabeled)` block. Block `source.media` is `{ kind: "frames", approx_fps: 1, tar }`.
  - nextme: `converts caption times to recording-relative seconds`.
  - nextme: `skips record with unparseable time and counts reason`.
  - nextme: `keeps caption extending past recording end` (verbatim; clipping happens at remap).
  - nextme: `fails clearly when day has no recordings`. The error message contains the day.
  - writer: `writes catalog layout and index` (temp dir; check the paths from §6.5 and the `index.json` counts).
- [ ] **Step 2: Run** `npm test -- packages/ingest`. Expected: FAIL.
- [ ] **Step 3: Implement** the modules above.
- [ ] **Step 4: Run** `npm test -- packages/ingest && npm run typecheck`. Expected: PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(ingest): NextMe-800 adapter, catalog writer and CLI"`

---

### Task 8: Ingest — HD-EPIC

**Files:**
- Create: `scripts/convert_hdepic.py`, `packages/ingest/src/hdepic/adapter.ts`
- Modify: `packages/ingest/src/cli.ts` (add the `hdepic` command)
- Test: `packages/ingest/test/hdepic.test.ts`, `packages/ingest/test/fixtures/hdepic/` (synthetic `HD_EPIC_Narrations.jsonl`, high-level CSV in the Task 1 format, `complete_recipes.json` subset)

**Interfaces:**
- Consumes: `readJsonl`, `IngestReport`, `CatalogWriter` (Task 7).
- Produces:
  - `scripts/convert_hdepic.py <in.pkl> <out.jsonl>` runs `pd.read_pickle` → `to_json(orient="records", lines=True, force_ascii=False)`. Invocation: `uv run --with pandas python scripts/convert_hdepic.py …`.
  - `parseHighLevel(csvText: string, file: string): HighLevelSegment[]`, where `HighLevelSegment = { video_id: string; start: number; end: number; label: string; recipe_id?: string }`. Columns come from Task 1.
  - `buildHdepicCatalog(segments: HighLevelSegment[], narrations: HdNarration[], recipes: Record<string, { name: string }>, report: IngestReport): { candidates: BlockCandidate[]; annotations: Map<string, Annotation[]> }`
    - `category: "cooking"`.
    - `summary = \`${participant} · ${label}\`` plus `` ` (${recipeName})` `` when there is a recipe.
    - `media: { kind: "video", fps: 30 }`.
    - Narrations become `level: "narration"` intervals with id `hdepic:<unique_narration_id>`.
  - CLI: `npm run ingest -- hdepic --annotations <repo dir> --narrations <converted.jsonl>`.

- [ ] **Step 1: Write failing tests:**
  - `builds one cooking candidate per high-level segment with recipe name in summary`.
  - `attaches narrations by video_id`. Narrations outside the segment are kept in the video's annotation file, because remap filters them later.
  - `candidate ids are deterministic across runs`.
  - `skips narration with end_before_start`.
- [ ] **Step 2: Run** `npm test -- hdepic`. Expected: FAIL.
- [ ] **Step 3: Implement** the adapter and CLI command. Write `convert_hdepic.py` (≈10 lines, argparse).
- [ ] **Step 4: Run** `npm test -- hdepic`. Expected: PASS. Manually run `uv run --with pandas python scripts/convert_hdepic.py data/raw/hd-epic-annotations/narrations-and-action-segments/HD_EPIC_Narrations.pkl data/raw/hdepic_narrations.jsonl && head -c 400 data/raw/hdepic_narrations.jsonl`. Expected: JSON with `unique_narration_id` and `start_timestamp`.
- [ ] **Step 5: Commit** — `git commit -m "feat(ingest): HD-EPIC adapter and pkl conversion script"`

---

### Task 9: Ingest — Ego4D + categories config

**Files:**
- Create: `packages/ingest/src/categories.ts`, `packages/ingest/src/ego4d/adapter.ts`, `config/categories.json`
- Modify: `packages/ingest/src/cli.ts` (add the `ego4d` command)
- Test: `packages/ingest/test/ego4d.test.ts`, `packages/ingest/test/fixtures/ego4d/{ego4d.json,narration.json}` (synthetic, 3 videos)

**Interfaces:**
- Consumes: Task 7 utilities.
- Produces:
  - `CategoriesSchema = z.object({ categories: z.array(z.object({ id: z.string(), label: z.string(), dataset: z.literal("ego4d"), scenarios: z.array(z.string()).min(1), keywords: z.array(z.string()) })) })` and `loadCategories(path: string)`.
  - `listScenarios(meta: Ego4dMeta): Array<{ scenario: string; videos: number }>` sorted by count desc.
  - `buildEgo4dCatalog(meta, narrations, categories, report): { candidates: BlockCandidate[]; annotations: Map<string, Annotation[]> }`.
    - A video matches a category when its `scenarios` intersects the category's `scenarios`.
    - A summary matches when `keywords` is empty or any keyword appears in `summary_text` (case-insensitive).
    - Candidates come from `narration_pass_1.summaries`, one per matching summary.
    - Annotations: narrations become `kind: "point"`, `level: "narration"`, id `ego4d:<video_uid>:n:<index>`. Summaries become intervals, `level: "summary"`, id `ego4d:<video_uid>:s:<index>`.
    - `media: { kind: "video", fps: 30 }`.
    - A video with no `narration_pass_1` is counted as skipped (`"no_narration"`).
  - `config/categories.json` holds the initial entries `walking`, `transit`, `hobby_lego` and `hobby_tent`. Each holds the provisional value from Step 3 of this task until Task 13 replaces it with real `--list-scenarios` output.
  - CLI: `npm run ingest -- ego4d --metadata <ego4d.json> --narrations <narration.json> [--categories config/categories.json] [--list-scenarios]`. A file over 512 MB or a JSON parse failure exits with a message naming the file.

- [ ] **Step 1: Write failing tests:**
  - `lists scenarios with counts`.
  - `creates candidates only for matching scenario and keyword`.
  - `keeps narrations as points, never intervals`.
  - `skips video without narration_pass_1 and counts reason`.
  - `rejects categories.json with empty scenarios`.
- [ ] **Step 2: Run** `npm test -- ego4d`. Expected: FAIL.
- [ ] **Step 3: Implement** `categories.ts`, the adapter and the CLI command. Write `config/categories.json` with the provisional scenarios: `walking` → `["Walking on street"]`, `transit` → `["Car - commuting, road trip"]`, `hobby_lego` → `["Playing board games"]` with keywords `["lego"]`, `hobby_tent` → `["Camping"]` with keywords `["tent"]`.
- [ ] **Step 4: Run** `npm test -- ego4d`. Expected: PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(ingest): Ego4D adapter with scenario-based categories"`

---

### Task 10: MCP — catalog, store, browse tools

**Files:**
- Create: `packages/mcp/src/{fsCatalog,store,server,main,index}.ts` (`index.ts` re-exports `createServer`)
- Test: `packages/mcp/test/helpers.ts` (writes a fixture catalog into a temp dir via `CatalogWriter` from `makeFixtureCatalog` data), `packages/mcp/test/browse.test.ts`

**Interfaces:**
- Consumes: core (all), `CatalogWriter` (Task 7; tests only, via a dev dependency on `@vb/ingest`).
- Produces:
  - `class CatalogMissingError extends Error`
  - `class FsCatalog implements Catalog { constructor(dataDir: string); listBaseDays(): Array<{ day: string; blocks: number; duration: number }>; listCategories(): Array<{ category: string; count: number }>; listCandidates(f: { category: string; query?: string; minSec?: number; maxSec?: number; limit: number; offset: number }): { total: number; items: BlockCandidate[] } }`.
    - The constructor throws `CatalogMissingError` if `data/catalog/index.json` is absent.
    - Candidates load once and annotations are cached per file.
    - `query` is a case-insensitive substring match on `summary`.
  - `class CompositionStore { constructor(dataDir: string); load(id: string): Composition; save(c: Composition): void; list(): Array<{ id: string; name?: string; base_day: string; revision: number }>; writeExport(m: Manifest): string /* path */ }`.
    - `save` writes `<file>.tmp-<pid>` and then `renameSync`.
    - An unknown id throws `CompositionError` with code `"COMPOSITION_NOT_FOUND"`.
  - `FsCatalog.getCandidate` looks up event candidates first, then base blocks across `days/*.json`.
  - `createServer(dataDir: string, opts?: { now?: () => Date; genId?: () => string }): McpServer`. It registers the tools from spec §8.1 with zod input schemas (`limit` default 20, `preview` default 10). Every tool returns `{ content: [{type:"text", text}], structuredContent }`.
  - `toToolError(e: unknown): CallToolResult` maps errors to `isError: true` with these Korean messages:
    - `SLOT_OUT_OF_RANGE` → `slot ${slot}은 범위 밖 (0..${max})`
    - `CANDIDATE_NOT_FOUND` → `candidate_id 없음 — list_event_candidates로 확인`
    - `REVISION_MISMATCH` → `revision 불일치 (현재 ${current})`
    - `MISSING_CANDIDATES` → `카탈로그에 없는 후보: ${ids.join(", ")}`
    - `CatalogMissingError` → `카탈로그가 없습니다. 먼저 npm run ingest를 실행하세요`
    - All other codes → `${code}: ${message}`
  - `main.ts` runs `createServer(process.env.VIDEO_BENCH_DATA_DIR ?? "./data")` over `StdioServerTransport`.

- [ ] **Step 1: Write failing tests** (in-memory: `InMemoryTransport.createLinkedPair()` + `Client`):
  - `list_base_days returns fixture day with 3 blocks`.
  - `list_categories returns cooking and walking with counts`.
  - `list_event_candidates filters by category and query and pages`.
  - `get_candidate returns caption preview limited to preview`.
  - `missing catalog yields Korean ingest hint`. Create the server on an empty dir and call `list_base_days` → `isError`, and the text contains `npm run ingest`.
- [ ] **Step 2: Run** `npm test -- packages/mcp`. Expected: FAIL.
- [ ] **Step 3: Implement** `fsCatalog.ts`, `store.ts`, `server.ts` (browse tools + `toToolError`) and `main.ts`. Construct `FsCatalog` lazily per call so a missing catalog becomes a tool error, not a startup crash.
- [ ] **Step 4: Run** `npm test -- packages/mcp`. Expected: PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(mcp): fs catalog, composition store and browse tools"`

---

### Task 11: MCP — edit tools + timetable

**Files:**
- Create: `packages/mcp/src/render.ts`
- Modify: `packages/mcp/src/server.ts`
- Test: `packages/mcp/test/render.test.ts`, `packages/mcp/test/edit.test.ts`

**Interfaces:**
- Consumes: Tasks 4, 5 and 10.
- Produces:
  - `renderTimetable(comp: Composition, timeline: TimelineBlock[]): string`, following spec §8.2:
    - Header: `rev ${revision} · 총 ${H:MM:SS}`.
    - Base line: `B${i}  ${HH:MM:SS}–${HH:MM:SS}  ${"base".padEnd(8)}  ${summary}`.
    - Event line: `S${slot}  ${HH:MM:SS}–${HH:MM:SS}  ${category.padEnd(8)}  evt:${ref_id}  ${summary}`. An empty slot renders as `S${slot}  (비어 있음)`.
    - Slots and base lines alternate S0, B0, S1, B1, …, SN.
  - Tools `create_composition`, `list_compositions`, `get_composition`, `insert_event`, `move_event`, `remove_event` (spec §8.2).
    - Edit tools call `assertRevision` before applying.
    - Edits and `get_composition` return `text = renderTimetable(...)` and `structuredContent = { composition, timeline }`.

- [ ] **Step 1: Write failing tests:**
  - render: `renders empty slots and base lines in order` (exact string for the fixture with one cooking event in slot 1).
  - edit: `create → insert → move → remove round trip updates revision 0→1→2→3 and persists to data/compositions/<id>.json`.
  - edit: `insert with stale expected_revision returns revision 불일치 and does not save`.
  - edit: `slot out of range returns Korean range message`.
  - edit: `get_composition on composition referencing removed candidate lists missing ids` (delete the candidate line from `candidates.jsonl` in the temp catalog).
- [ ] **Step 2: Run** `npm test -- render edit`. Expected: FAIL.
- [ ] **Step 3: Implement** `render.ts` and the edit tools.
- [ ] **Step 4: Run** `npm test -- render edit`. Expected: PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(mcp): composition edit tools with timetable rendering"`

---

### Task 12: MCP — export + end-to-end + README

**Files:**
- Modify: `packages/mcp/src/server.ts`
- Create: `README.md`
- Test: `packages/mcp/test/export.test.ts`

**Interfaces:**
- Consumes: `buildManifest` (Task 6), `CompositionStore.writeExport` (Task 10).
- Produces: tool `export_manifest({ composition_id })`. It writes `data/exports/<id>-r<revision>.manifest.json` atomically and returns `{ path, duration, blocks, captions_by_dataset: Record<Dataset, number> }`.

- [ ] **Step 1: Write failing tests:**
  - `end-to-end: browse → create → insert cooking → insert walking → export`. The manifest file exists and parses with `ManifestSchema`, `duration` equals the timetable total, and `captions_by_dataset.hdepic > 0`.
  - `export twice at same revision overwrites same path; after an edit writes r<n+1>`.
  - `export with missing candidate returns 카탈로그에 없는 후보 and writes nothing`.
- [ ] **Step 2: Run** `npm test -- export`. Expected: FAIL.
- [ ] **Step 3: Implement** the export tool. Write `README.md` covering:
  - nvm/Node 24 setup and `npm install`
  - `HF_TOKEN` and NextMe gated access
  - the HD-EPIC clone + `convert_hdepic.py`
  - Ego4D CLI download of annotations
  - every ingest command
  - registration: `claude mcp add video-composer -e VIDEO_BENCH_DATA_DIR=$PWD/data -- npx tsx $PWD/packages/mcp/src/main.ts`
  - a license note: `data/` is never committed
- [ ] **Step 4: Run** `npm test && npm run typecheck`. Expected: all PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(mcp): manifest export tool and README"`

---

### Task 13: Real-data smoke + finalize categories

Needs `HF_TOKEN` and local Ego4D annotation files. If either is unavailable, stop and ask the user.

**Files:**
- Modify: `config/categories.json`, `docs/superpowers/specs/2026-11-02-video-composer-mcp-design.md` (§11 item 3 결과)

- [ ] **Step 1: Ingest real data.** Run:
  - `npm run ingest -- nextme --days <one day the user picks>`
  - `npm run ingest -- hdepic …`
  - `npm run ingest -- ego4d --metadata … --narrations … --list-scenarios`

  Expected: the reports print non-zero `ingested` counts and the scenario list.
- [ ] **Step 2: Fill categories.** Replace the provisional `scenarios` in `config/categories.json` with the real values from the list. Show the proposed mapping to the user and confirm it before writing. Rerun the Ego4D ingest. Expected: `list_categories` shows non-zero counts for walking/transit and for any hobby category that has data.
- [ ] **Step 3: Smoke through MCP.** Register the server and, in a Claude session, run `list_base_days` → `create_composition` → `list_categories` → `list_event_candidates(cooking)` → `insert_event` ×2 → `export_manifest`. Expected: the manifest has `blocks` in timetable order, and spot-checking 3 captions per dataset gives `composed − block.composed_start == src − block.src_start`.
- [ ] **Step 4: Record results** in spec §11 item 3. Commit — `git commit -m "chore: finalize Ego4D category mapping from real scenarios"`
