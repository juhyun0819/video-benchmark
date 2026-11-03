# Video Composer MCP — 설계 스펙

- 날짜: 2026-11-02
- 상태: 리뷰 대기
- 저장소: https://github.com/juhyun0819/video-benchmark

## 1. 목적

일상 1인칭 영상(base 하루)에 사용자가 고른 이벤트(요리, 산책, 이동, 취미 등)를 블록 단위로 끼워 넣어 **새로운 합성 영상의 정의(manifest)** 를 만든다. manifest는 비디오 이해 모델을 평가하기 위한 **벤치마크 데이터**로 쓰인다. 다운스트림 파이프라인이 manifest를 읽어 실제 영상을 합성하고, manifest 안의 타임라인·캡션을 정답 메타데이터로 사용한다.

1차 결과물은 MCP 서버이며, 이후 같은 코어 위에 UI(드래그 가능한 시간표)를 붙인다.

## 2. 요구사항

### 사용자가 명시한 것

1. 런타임/의존성은 Docker가 아닌 **npm**(Node)으로 해결한다.
2. base 영상(일반적인 하루)을 선택하고, 그 위에 이벤트 블록을 시간표처럼 넣고/옮기고/뺄 수 있다.
3. 결과는 영상 원본이 아닌 **manifest**로 export한다.
4. 지금은 MCP로 구현하고, 이후 UI로 확장한다.
5. manifest는 벤치마크 데이터 용도다. 원본 캡션을 합성 영상의 **새 타임스탬프**로 변환해 연결한다.
6. 원본 영상은 직접 내려받지 않는다. 주석 정보만 수집해 재조합 가능한 **통일된 형태**로 정규화한다(데이터셋마다 fps·프레임 간격 등이 다름).
7. 데이터셋: base = **NextMe-800**, 요리 = **HD-EPIC**, 나머지(산책·이동·취미 등) = **Ego4D**에서 추출.
8. 이벤트는 **끼워넣기** 방식이다. 넣은 만큼 이후 base가 뒤로 밀리고 전체 길이가 늘어난다.
9. 합성은 **블록 나열**이다. base도 여러 블록으로 나뉘어 있고, 블록 사이 슬롯에 이벤트를 0개 이상 넣는다.
10. base 블록 단위는 **L4 활동**(평균 약 24분)이다.
11. 캡션은 **원본 그대로** 가져오고 타임스탬프만 바꾼다.
12. QA는 이번 범위에서 생성하지 않는다. 단, 나중에 추가할 때 추가 비용이 없도록 설계한다.
13. 이벤트 후보는 원본 주석에서 **자동으로** 만든다. MCP에서는 카테고리를 먼저 고르고, 그 안에서 summary를 보고 고른다.
14. HD-EPIC의 `.pkl`은 한 번만 실행하는 Python 보조 스크립트로 변환한다.

### 설계 결정(대화 중 합의)

- base 블록은 원래 시간 순서로 **고정**한다. 재정렬하거나 삭제할 수 없다.
- 구조는 npm workspaces 모노레포(`core` / `ingest` / `mcp`)로 한다.
- 시간은 모두 "원본 영상 시작 기준 초(float)"로 정규화한다.
- 합성 시각은 저장하지 않고 블록 순서로부터 계산한다.

### 범위 밖 (이번 스펙)

다음은 이번 스펙에서 다루지 않는다.
- QA 생성
- 기존 QA(HD-EPIC VQA) 가져오기
- 영상 렌더링·합성
- 미디어 다운로드
- UI
- LLM 연동
- Ego4D 주석 자동 다운로드

## 3. 원본 데이터셋 특성

| | NextMe-800 (base) | HD-EPIC (요리) | Ego4D (기타) |
|---|---|---|---|
| 미디어 | 약 1fps JPG, 녹화별 TAR `raw/YYYY-MM-DD/YYYY-MM-DD_HHMM-HHMM.tar`, 프레임 파일명 `..._HH-MM-SS_TZ__frame_NNNNN.jpg` | MP4 30fps | canonical 30fps |
| 시간 기준 | 벽시계 문자열 `YYYY-MM-DD HH:MM:SS` (HKT) | 영상 기준 초(float) | 영상 기준 초 + 프레임 번호 |
| 캡션 | `raw_caption.jsonl`(30초 창 기반 구간 캡션), `L1~L5.jsonl`(계층 활동, 구간) | `HD_EPIC_Narrations.pkl`: `narration`, `start_timestamp`, `end_timestamp` 등 | `narration.json`: `narration_pass_1.narrations[]`(시점형 `timestamp_sec`), `summaries[]`(구간형 `start_sec`/`end_sec`) |
| 상위 구조 | L1(~5s) ⊂ L2(~1m) ⊂ L3(~7m) ⊂ L4(~24m) ⊂ L5(~39m), 부모가 자식 구간을 덮음 | recipe, high-level activity (`high-level/`) | 영상 단위 `scenarios` (`ego4d.json`) |
| 접근 | HF gated (`HF_TOKEN`), CC BY-NC 4.0 | GitHub `hd-epic/hd-epic-annotations` | Ego4D 라이선스 + 공식 CLI |

## 4. 아키텍처

```
packages/
  core/     타입·스키마(zod), 블록 타일링, 편집 연산, 타임라인 계산, 재매핑, manifest 빌드
            순수 함수만 두고 파일·네트워크 I/O는 없음
  ingest/   데이터셋 어댑터 CLI. 원본 주석을 읽어 data/catalog/ 생성
  mcp/      stdio MCP 서버. 카탈로그를 읽고 core를 호출하며, 컴포지션 저장과 export 담당
  (ui/)     향후. core를 그대로 재사용
scripts/
  convert_hdepic.py   HD-EPIC pkl → JSONL (일회성)
config/
  categories.json     이벤트 카테고리 정의(Ego4D scenario·키워드 매핑)
data/                 .gitignore. 카탈로그·컴포지션·export 저장
```

의존 방향: `ingest → core`, `mcp → core`. core는 다른 패키지에 의존하지 않는다.

기술 스택:
- 언어·패키지: Node 20+, TypeScript, npm workspaces
- 라이브러리: zod, `@modelcontextprotocol/sdk`
- 테스트·개발: vitest, tsx
- Python: `scripts/convert_hdepic.py`에만 쓴다(`uvx --with pandas`로 실행).

## 5. 데이터 모델 (core)

### 5.1 시간 정규화

- `src_t`는 원본 영상(녹화) 시작 기준 초(float)다.
- NextMe의 벽시계 시각(HKT)은 `src_t = wallclock − recording_start`로 바꾼다. `recording_start`는 `raw/recordings.csv`에서 얻는다. 원래 문자열은 provenance에 보존한다.
- HD-EPIC과 Ego4D는 초 값을 그대로 쓴다.
- manifest는 fps를 강제하지 않는다. 시각은 초로만 적고, 샘플링은 다운스트림 렌더러가 미디어 설명자를 보고 결정한다.

### 5.2 타입

```ts
type Dataset = "nextme800" | "hdepic" | "ego4d";

type SourceRef = {
  dataset: Dataset;
  video_id: string;              // NextMe recording_id | HD-EPIC video_id | Ego4D video_uid
  media:
    | { kind: "frames"; approx_fps: 1; tar: string }   // NextMe
    | { kind: "video"; fps: number };                  // HD-EPIC, Ego4D
};

type TimeSpan =
  | { kind: "interval"; start: number; end: number }
  | { kind: "point"; t: number };

type Annotation = {
  id: string;                    // 원본 id 기반, 결정적
  track: "captions";
  level: "raw" | "L1" | "L2" | "L3" | "L4" | "L5" | "narration" | "summary";
  time: TimeSpan;                // src_t 기준
  text: string;                  // 원본 그대로
  original: Record<string, unknown>;  // 원본 레코드 전체
};

type BlockCandidate = {
  id: string;                    // `${dataset}:${video_id}:${src_start.toFixed(3)}-${src_end.toFixed(3)}` (결정적)
  role: "base" | "event";
  category: string;              // "base" | "cooking" | "walking" | "transit" | "hobby_lego" | ...
  source: SourceRef;
  src_start: number;
  src_end: number;
  summary: string;
};

type BaseDay = { day: string; blocks: BlockCandidate[] };   // 시간순 고정

type EventRef = { ref_id: string; candidate_id: string };

type Composition = {
  schema_version: 1;
  id: string;
  name?: string;
  base_day: string;
  revision: number;              // 변경마다 +1
  slots: EventRef[][];           // 길이 = base 블록 수 + 1
};
```

### 5.3 Manifest

```ts
type Manifest = {
  schema_version: 1;
  composition_id: string;
  composition_revision: number;
  created_at: string;            // ISO 8601
  duration: number;              // 초
  blocks: Array<{
    block_id: string;            // base:<day>:<idx> | evt:<ref_id>
    role: "base" | "event";
    category: string;
    summary: string;
    composed_start: number;
    composed_end: number;
    source: SourceRef;
    src_start: number;
    src_end: number;
  }>;
  tracks: {
    captions: Array<{
      id: string;
      block_id: string;
      level: Annotation["level"];
      time: TimeSpan;            // 합성 기준
      text: string;
      clipped?: { start?: true; end?: true };
      provenance: {
        dataset: Dataset;
        video_id: string;
        orig_id: string;
        src_time: TimeSpan;      // 잘리기 전 원본 구간
        orig_time_string?: { start?: string; end?: string };  // NextMe 벽시계 원문
        original: Record<string, unknown>;
      };
    }>;
    // 향후: qa
  };
  licenses: Array<{ dataset: Dataset; license: string; note?: string }>;
};
```

**QA를 나중에 추가하기 위한 전제:** 아래 세 가지가 지켜지면 QA는 manifest를 읽어 `tracks.qa`를 추가하는 후처리 모듈로 붙일 수 있다.
- 블록 구조(`blocks`)가 manifest에 보존된다.
- 출처(`provenance`)가 보존된다.
- `tracks` 맵과 `schema_version`으로 스키마를 확장할 수 있다.

## 6. Ingest

### 6.1 CLI

```
npm run ingest -- nextme --days 2026-04-15[,2026-04-16...]
npm run ingest -- hdepic --annotations <hd-epic-annotations 경로>
npm run ingest -- ego4d  --metadata <ego4d.json> --narrations <narration.json>
npm run ingest -- ego4d  --metadata <ego4d.json> --list-scenarios
```

### 6.2 NextMe-800 어댑터 (base)

- 입력은 HF Hub에서 받는다(`HF_TOKEN` 필요).
  - `raw/recordings.csv`
  - `captions/1_nextme/<recording_id>/`의 `raw_caption.jsonl`과 `L1.jsonl`~`L5.jsonl`
- 지정한 날짜만 받는다.
- 하루의 녹화들을 시간순으로 놓고, **녹화마다 L4 시작 지점을 경계로 빈틈없이 타일링**한다.
  - 녹화 시작부터 첫 L4 시작까지도 하나의 블록이 된다. 마지막 블록은 녹화 끝까지 간다.
  - 녹화 경계는 항상 블록 경계다. 녹화 사이 공백은 블록 나열에서 자연스럽게 빠진다.
  - 60초 미만 블록은 같은 녹화의 앞 블록에 합친다. 녹화의 첫 블록이면 다음 블록에 합친다.
  - 블록 summary는 블록 시작과 겹치는 L4의 `text`다. 해당하는 L4가 없으면 `"(unlabeled)"`로 둔다.
- 주석 레벨은 `raw`(raw_caption의 `action`을 text로 사용)와 `L1`~`L5`(`text`)다. `approx_time: true` 레코드는 `original`에 그대로 보존한다.

### 6.3 HD-EPIC 어댑터 (요리)

- `scripts/convert_hdepic.py`가 `HD_EPIC_Narrations.pkl`을 JSONL로 한 번 변환한다.
- 이벤트 후보는 `high-level/`의 high-level activity 구간이고, category는 `cooking`이다. summary는 `high_level_activity_label`에 레시피 이름이 있으면 덧붙인다.
- 후보 구간에 들어가는 narration을 `level: "narration"` 구간형 주석으로 만든다.

### 6.4 Ego4D 어댑터 (산책·이동·취미 등)

- 입력은 사용자가 공식 CLI로 받은 `ego4d.json`과 `narration.json`의 로컬 경로다.
- `config/categories.json`의 매핑(카테고리 → scenario 목록 + summary 키워드)에 맞는 영상을 고른다. 그 영상의 `narration_pass_1.summaries[]` 각 구간이 이벤트 후보이고, summary는 `summary_text`다.
- 후보 구간에 들어가는 narration을 `level: "narration"` **시점형** 주석으로 만들고, 구간화하지 않는다. summary 자체도 `level: "summary"` 구간형 주석으로 넣는다.
- `--list-scenarios`로 실제 scenario 값과 영상 수를 출력해 매핑 작성을 돕는다.

### 6.5 카탈로그 레이아웃

```
data/catalog/
  index.json                               카테고리별 후보 수, 수집 시각, 어댑터 버전
  nextme800/days/<day>.json                BaseDay
  nextme800/annotations/<recording_id>.jsonl
  hdepic/candidates.jsonl
  hdepic/annotations/<video_id>.jsonl
  ego4d/candidates.jsonl
  ego4d/annotations/<video_uid>.jsonl
```

### 6.6 공통 규칙

- 모든 출력은 zod 스키마로 검증한다. 같은 입력이면 같은 출력이 나온다(결정적, 멱등).
- 실행이 끝나면 수집 수와 건너뛴 수(이유별)를 요약해 출력한다.

## 7. 합성과 재매핑 (core)

### 7.1 편집 연산 (순수 함수)

```ts
createComposition(baseDay, opts?): Composition
insertEvent(comp, catalog, slot, candidateId, position?): Composition   // position 생략 시 슬롯 끝
moveEvent(comp, refId, toSlot, toPosition?): Composition
removeEvent(comp, refId): Composition
```

불변식은 아래와 같고, 위반하면 오류를 던진다.
- base 블록의 순서와 개수는 바뀌지 않는다.
- `slots.length === baseDay.blocks.length + 1`
- 넣는 후보는 카탈로그에 있어야 하고 `role === "event"`여야 한다.
- `ref_id`는 컴포지션 안에서 유일하다. 같은 후보를 여러 번 넣을 수 있다.
- 컴포지션 `id`와 `ref_id`는 `crypto.randomUUID()` 앞 8자리로 만든다. 순수성을 위해 id 생성기는 인자로 주입한다.
- 성공한 변경마다 `revision`이 1 오른다.

### 7.2 타임라인 계산

- `resolveTimeline(comp, catalog)`가 블록의 순서 목록을 만든다. 순서는 `slot0 이벤트들 → base0 → slot1 이벤트들 → base1 → … → base(n−1) → slotN 이벤트들`이다.
- `composed_start`는 앞선 블록 길이(`src_end − src_start`)의 누적합이다. 0에서 시작한다.

### 7.3 재매핑

블록마다 원본 영상의 주석 중 `[src_start, src_end)`와 겹치는 것을 고른다.

```
composed_t = block.composed_start + (src_t − block.src_start)
```

- **구간형:** 겹치는 부분만 남기고 블록 범위로 자른다. 잘렸으면 해당 쪽에 `clipped`를 표시한다. 잘리기 전 구간은 `provenance.src_time`에 남는다.
- **시점형:** `src_start ≤ t < src_end`일 때만 포함한다.
- 텍스트와 원본 레코드는 바꾸지 않는다.

### 7.4 Export

- `buildManifest(comp, catalog, now)`는 Manifest를 만들고 zod로 검증한다.
- 정렬은 결정적이다. blocks는 타임라인 순서, captions는 `composed start → level → id` 순서다.
- 컴포지션이 참조하는 후보가 카탈로그에 없으면 실패한다. 빠진 id 목록을 함께 알린다.
- `licenses`는 사용된 데이터셋에서 채운다.

### 7.5 저장 (mcp 패키지가 담당)

```
data/compositions/<id>.json
data/exports/<id>-r<revision>.manifest.json
```

## 8. MCP 도구

- stdio transport를 쓴다.
- 데이터 루트는 `VIDEO_BENCH_DATA_DIR` 환경변수로 받고, 기본값은 `./data`다.
- 서버는 카탈로그를 읽기 전용으로 쓴다.

### 8.1 탐색

| 도구 | 입력 | 출력 |
|---|---|---|
| `list_base_days` | — | 날짜, 블록 수, 총 길이 |
| `get_base_day` | `day` | 블록 인덱스, 길이, summary, 원본 시각 범위 |
| `list_categories` | — | 카테고리, 후보 수 |
| `list_event_candidates` | `category`, `query?`, `min_sec?`, `max_sec?`, `limit`(기본 20), `offset` | 후보 id, summary, 길이, 데이터셋, 전체 개수 |
| `get_candidate` | `candidate_id`, `preview?`(기본 10) | 상세 + caption 미리보기 |

### 8.2 편집

| 도구 | 입력 |
|---|---|
| `create_composition` | `day`, `name?` |
| `list_compositions` | — |
| `get_composition` | `composition_id` |
| `insert_event` | `composition_id`, `slot`, `candidate_id`, `position?`, `expected_revision?` |
| `move_event` | `composition_id`, `ref_id`, `to_slot`, `to_position?`, `expected_revision?` |
| `remove_event` | `composition_id`, `ref_id`, `expected_revision?` |

- 편집 도구와 `get_composition`은 갱신된 시간표를 텍스트와 `structuredContent`로 함께 돌려준다. 텍스트 예시는 아래와 같다.

```
rev 3 · 총 7:42:10
S0  (비어 있음)
B0  00:00:00–00:23:41  base     Working at desk, reviewing documents
S1  00:23:41–00:41:02  cooking  evt:a1  P01 · Making breakfast (omelette)
B1  00:41:02–01:05:19  base     Video call with colleagues
S2  (비어 있음)
```

- `expected_revision`이 현재 revision과 다르면 변경을 거부하고 현재 revision을 알린다.

### 8.3 Export

| 도구 | 입력 | 출력 |
|---|---|---|
| `export_manifest` | `composition_id` | 파일 경로, 총 길이, 블록 수, 데이터셋별 caption 수 |

## 9. 오류 처리

- **Ingest**
  - 파일 수준 형식 불일치(필수 필드 없음, 스키마 불일치)가 있으면 즉시 중단하고 파일과 줄 번호를 알린다.
  - 레코드 수준 문제(시각 파싱 실패, `end < start`)는 그 레코드만 건너뛰고 이유별로 집계한다.
- **저장:** 컴포지션과 manifest는 원자적으로 쓴다(임시 파일에 쓴 뒤 rename).
- **MCP**
  - 잘못된 입력에는 `isError`와 바로 고칠 수 있는 메시지를 돌려준다. 예: `slot 21은 범위 밖 (0..20)`, `candidate_id 없음 — list_event_candidates로 확인`.
  - 카탈로그가 없으면 ingest 실행 방법을 안내한다.

## 10. 테스트

모든 테스트는 vitest로 돌리고 네트워크를 쓰지 않는다.

| 대상 | 내용 |
|---|---|
| core 단위 | L4 타일링, 짧은 블록 병합, 누적합 타임라인, 구간·시점 재매핑, `clipped`, 불변식 위반 거부, revision 증가 |
| core 불변식 | 모든 caption이 소속 블록 범위 안에 있는지 확인한다. 블록 길이 합이 `duration`과 같은지 확인한다. 이벤트를 모두 빼면 base 원본 캡션과 오프셋만 다른지 확인한다. |
| export | 고정 fixture와 고정 `now`로 golden manifest를 비교한다. |
| ingest | 실제 형식을 흉내 낸 소형 fixture로 검증한다. NextMe는 `recordings.csv`, `raw_caption.jsonl`, `L4.jsonl` 등, HD-EPIC은 변환된 JSONL, Ego4D는 `ego4d.json`과 `narration.json` 일부를 쓴다. HKT → 초 변환과 블록 병합도 확인한다. |
| mcp 통합 | SDK in-memory transport로 탐색 → 생성 → 삽입 → 이동 → 삭제 → export 흐름과 오류 케이스를 확인한다. |
| 실데이터 스모크 | 수동으로 한다. NextMe 하루와 HD-EPIC/Ego4D 후보 몇 개로 export까지 확인한다. |

## 11. 구현 전 확인할 항목

아래는 공개 문서만으로 확정하지 못한 부분이다. 구현 첫 단계에서 실제 파일을 보고 확인한다. 확인 결과가 설계를 바꾸면 이 스펙을 갱신한다.

1. **NextMe L1~L5 파일 위치:** `captions/1_nextme/<recording_id>/` 아래에 있는지 확인한다. 또 `recordings.csv`의 시작 시각 컬럼 이름과 정밀도를 확인한다.
2. **HD-EPIC high-level activity의 구간 정보:** `PXX_recipe_timestamps.csv`에 시작·종료 시각이 있는지 확인한다. 없으면 `complete_recipes.json`의 step 구간이나 영상 전체를 후보 단위로 쓴다.
3. **Ego4D scenario 값:** 산책, 차량·대중교통, 레고, 텐트에 해당하는 실제 `scenarios` 값을 확인해 `config/categories.json`을 채운다.
4. **HD-EPIC pkl:** pandas 버전 호환성을 확인한다. 공식 CSV나 JSON 배포본이 있으면 변환 스크립트 대신 그것을 쓴다.

### 확인 결과 (2026-11-03) — HD-EPIC

`hd-epic/hd-epic-annotations`를 `--depth 1`로 clone해서 확인했다(커밋 대상 아님, `data/raw/hd-epic-annotations`). NextMe 항목(§11.1)은 `HF_TOKEN`이 준비된 뒤 따로 기록한다.

**항목 2 — high-level 구간 정보: 있다 (후보 단위는 `activities` CSV).**

- 파일은 `high-level/`이 아니라 `high-level/activities/P01_recipe_timestamps.csv` ~ `P09_recipe_timestamps.csv`(참가자 9명, 총 1490행, 149개 video)다. 문서의 경로를 `high-level/activities/`로 읽는다.
- 헤더: `video_id,recipe_id,high_level_activity_label,start_time,end_time`.
- 시간 형식은 **초 단위 float 문자열**이다(`HH:MM:SS` 아님). 소수 첫째 자리까지이고 video 시작 기준이다. 146개 video는 첫 행이 `0`에서 시작한다.
- `video_id`는 `P01-20240202-110250` 형태로 pkl의 `video_id`와 같은 키다.
- `recipe_id`는 비어 있을 수 있다(1490행 중 612행, 배경 활동). 값은 `R<n>` 하나이거나 `R<n>, R<m>`처럼 쉼표로 이어진 여러 개다. `complete_recipes.json`의 키(`P01_R01`)와 맞추려면 `participant + "_" + recipe_id`로 만들고, 쉼표 값은 분리해야 한다.
- `high_level_activity_label`은 자유 문장이다(빈 값 1행).
- **파싱 시 주의(어댑터에 반영):**
  - `end_time`이 숫자가 아니라 리터럴 `end`인 행이 148개다. 147개 video의 **마지막 행**이고, 1개는 중간 행이다. "video 끝까지"로 해석하고, 길이는 같은 video의 pkl `end_timestamp` 최댓값(147개 video에서 사용 가능)이나 영상 메타데이터로 채운다. 중간 행의 `end`는 다음 행 `start_time`으로 닫는 것을 우선 검토한다.
  - 숫자형 행 중 `start_time >= end_time`인 행이 7개, 같은 video 안에서 `start_time`이 정렬되지 않은 video가 있다. 연속한 행 1340쌍 중 1319쌍은 맞닿고, 13쌍은 겹치고, 8쌍은 사이가 벌어진다. 정렬과 검증을 하고 위반 행은 경고로 건너뛴다.
  - video 집합이 pkl과 완전히 같지는 않다. CSV 149개, pkl 156개, 교집합 148개이며 CSV에만 1개, pkl에만 8개가 있다.
- 폴백(§11.2의 step 구간, 영상 전체)은 쓰지 않아도 된다.

**`complete_recipes.json`** (`high-level/complete_recipes.json`)

- 최상위는 dict이고 69개 레시피다. 키는 `P01_R01` 형태다. 값의 필드는 `participant`, `name`, `type`, `source`, `steps`, `captures`다.
- `steps`는 `{ "P01_R01_S01": "<문자열>" }` dict이다. step 항목 자체에는 시간이 없다.
- **시간은 `captures[]` 안에 있다.** `captures`는 리스트이고 각 항목의 필드는 `videos`(list), `ingredients`(dict), `step_times`(dict), `prep_times`(dict)다.
  - `step_times`는 `{ "<step_id>": [ {"video": <video_id>, "start": <float 초>, "end": <float 초>}, ... ] }`이다. 키는 `steps`의 step id와 모두 일치하고(504개 중 불일치 0), 80개 capture 전부에 값이 있다. 같은 step이 여러 구간으로 나뉘거나 여러 video에 걸칠 수 있어 값이 리스트다.
  - `prep_times`도 같은 구조(`video`, `start`, `end`)이고 step id를 키로 쓴다.
  - `ingredients[<id>]`의 `weigh`와 `add`도 같은 `{video, start, end}` 리스트다. 그 밖의 필드는 `name`, `amount`, `amount_unit`, `calories`, `carbs`, `fat`, `protein`이고 값은 모두 문자열이다.
- 따라서 step 단위 하위 구간이 필요하면 `captures[].step_times`를 쓴다. 단위는 `activities` CSV와 같은 초 float이다.

**항목 4 — pkl과 CSV/JSON 배포본**

- pkl은 `uv run --with pandas python ...`처럼 **기본 환경으로는 읽히지 않는다.** uv가 miniconda의 python(numpy 1.x)을 골라 `ModuleNotFoundError: No module named 'numpy._core.numeric'`로 실패했다. pkl이 numpy 2.x로 만들어졌기 때문이다.
- 다음 명령은 읽힌다(`numpy 2.4.6`, `pandas 3.1.0`로 확인): `uv run --isolated --no-project --with pandas --with "numpy>=2" python ...`. 변환 스크립트도 이 호출을 쓴다.
- `pd.read_pickle`의 결과는 `DataFrame`이고 shape은 `(59454, 16)`, 인덱스는 `RangeIndex`다. 결측값 0, `unique_narration_id` 중복 0, 156개 video, 모든 행에서 `start_timestamp <= end_timestamp`이다.
- dtypes:

  | 컬럼 | dtype / 원소 타입 |
  | --- | --- |
  | `unique_narration_id`, `participant_id`, `video_id`, `narration` | object (str) |
  | `start_timestamp`, `end_timestamp`, `narration_timestamp` | float64 (초) |
  | `nouns`, `verbs`, `hands` | object (list of str) |
  | `pairs`, `main_actions`, `pair_classes`, `main_action_classes` | object (list of tuple) |
  | `verb_classes`, `noun_classes` | object (list of int) |

- 시간 값은 초 float이다(`start_timestamp`는 0.0 ~ 4480.94 범위).
- **공식 CSV/JSON 배포본은 없다.** 저장소에서 narration이 들어 있는 파일은 pkl뿐이다. `HD_EPIC_Narrations_erratum.csv`는 컬럼이 `erratum_narration_id` 하나인 정오표(56개 id, 모두 pkl에 존재)이고, `HD_EPIC_noun_classes.csv`와 `HD_EPIC_verb_classes.csv`는 클래스 사전이다. 그러므로 `scripts/convert_hdepic.py`로 pkl을 JSONL로 변환하는 설계(§6.3)를 유지하되, 위 numpy 2 호출을 쓰고 erratum id 처리 방식(제외 또는 표시)을 정한다.

## 12. 향후 확장

- **QA 트랙:** manifest를 읽어 블록 구조 기반 템플릿 QA를 만들어 `tracks.qa`에 추가한다. 이후 기존 QA(HD-EPIC VQA)를 provenance로 재매핑해 가져온다. LLM 생성은 선택 플러그인으로 둔다.
- **UI:** core를 재사용한 시간표 드래그 UI. `revision`으로 동시 편집 충돌을 감지한다.
- **카테고리·데이터셋 추가:** 어댑터와 `categories.json` 항목만 추가하면 된다.
