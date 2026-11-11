# video-benchmark

NextMe-800, Ego4D, Ego-Exo4D 같은 데이터셋과 직접 모은 YouTube 영상의 주석(caption)을 클립 bank로 정규화하고, "합성 사람 × N일" 타임라인(고정 루틴 + 변동 + 예외)을 만들어 manifest(JSON)로 내보내는 도구다. 영상 자체는 다루지 않고, 합성된 구간 구조와 시간 재매핑된 caption만 만든다. 영상 렌더링은 다음 계획(Plan 2)이다.

구성은 npm workspaces 세 패키지다.

| 패키지 | 역할 |
|---|---|
| `@vb/core` | 스키마, world/schedule 생성, 클립 매칭, 규칙 검사, manifest 생성 |
| `@vb/ingest` | 데이터셋별 어댑터와 `ingest` CLI. `data/catalog/`를 쓴다 |
| `@vb/mcp` | stdio MCP 서버 `video-composer`와 `generate` CLI |

설계 문서는 `docs/superpowers/specs/2026-11-12-synthetic-timeline-generation-design.md`에 있다. 이전 스펙(`2026-11-02-video-composer-mcp-design.md`, `2026-11-07-video-render-design.md`)은 이 문서로 대체됐다.

## 1. 준비

Node 24를 쓴다(`.nvmrc`가 24).

```bash
nvm install   # .nvmrc를 읽는다
nvm use
npm install
```

데이터 루트는 기본값이 `./data`다. `VIDEO_BENCH_DATA_DIR` 환경변수로 바꿀 수 있고, ingest CLI는 `--data <dir>` 플래그로도 지정할 수 있다(플래그가 환경변수보다 우선). 설정 폴더(`config/`)는 `VIDEO_BENCH_CONFIG_DIR`로 바꾼다.

### 이전 버전에서 올린 뒤 정리

예전 base day + 이벤트 방식에서 올리면 아래 로컬 파일이 낡은 채 남는다. 다음 `sync_data.py push` 전에 사용자가 직접 지운다(도구는 아무것도 지우지 않는다).

- `data/catalog/hdepic/`
- `data/catalog/ego4d/candidates.jsonl`, `data/catalog/egoexo4d/candidates.jsonl`
- `data/catalog/nextme800/days/`
- `data/compositions/`

## 2. 데이터 가져오기

전체 순서는 다음과 같다. 도움말은 `npm run ingest -- --help`.

1. ATUS 분포를 받거나 다시 계산한다(아래 "ATUS 분포").
2. NextMe-800 초안을 만든다(`ingest nextme`).
3. 초안을 사람이 검수한 뒤 bank에 넣는다(`ingest bank --dataset nextme800`).
4. Ego-Exo4D, Ego4D를 ingest한다.
5. YouTube 영상은 사용자가 내려받고 검수한 clips JSONL을 `ingest bank --dataset youtube`로 넣는다.
6. 타임라인을 만든다(3절).

Ego4D와 Ego-Exo4D는 실행할 때마다 그 데이터셋의 카탈로그를 통째로 바꾼다.

### ATUS 분포

`data/distributions/atus2025/`에 ATUS 2025에서 뽑은 시간 분포(`atus2025_stats.json` 등)가 있다. 이 폴더는 동기화 대상이라 `sync_data.py pull`로 받으면 된다. 다시 계산하려면 BLS 원자료(Activity, Respondent csv)를 받아 아래를 실행한다. pandas와 numpy가 필요하다.

```bash
uv run --isolated --no-project --with pandas --with numpy \
  python scripts/atus/atus_stats.py <activity.csv> <respondent.csv> <out.json>
```

출력은 `data/distributions/atus2025/`에 둔다.

### NextMe-800 (집·직장 클립)

NextMe-800은 Hugging Face의 gated 데이터셋(`mmm8383/NextMe-800`)이다. 먼저 접근 승인을 받아야 한다. 토큰은 둘 중 하나로 준다.

```bash
export HF_TOKEN=hf_xxx          # 방법 1: 환경변수
huggingface-cli login           # 방법 2: 로그인 (토큰이 ~/.cache/huggingface/token 에 저장됨)
```

`HF_TOKEN`이 없으면 CLI가 `~/.cache/huggingface/token`을 읽는다. 둘 다 없으면 오류로 끝난다.

```bash
npm run ingest -- nextme --days 2026-04-15,2026-04-16 \
  --out data/raw/bank/nextme800.draft.jsonl \
  --rules config/nextme_label_rules.json
```

`--days`는 `YYYY-MM-DD`를 쉼표로 이어서 준다. `--out`과 `--rules`는 위 값이 기본값이다. 이 명령은 카탈로그가 아니라 초안 JSONL만 만든다. `--out` 파일이 이미 있으면 검수한 라벨을 지우지 않도록 오류로 끝난다. 덮어쓰려면 `--overwrite`를 주고, 날짜를 더 받을 때는 다른 `--out`에 쓴 뒤 검수한 초안과 합친다.

초안의 장소(node)와 활동 분류는 `config/nextme_label_rules.json`의 키워드로 자동 붙인 것이라 틀릴 수 있다. 반드시 사람이 검수해 고친 뒤 bank에 넣는다.

```bash
npm run ingest -- bank --input data/raw/bank/nextme800.draft.jsonl --dataset nextme800
```

`--dataset`은 `nextme800`, `ego4d`, `egoexo4d`, `youtube` 중 하나다. 같은 형식의 JSONL(검수한 YouTube 클립 등)은 이 명령으로 넣는다.

`ingest bank`는 그 데이터셋의 카탈로그 전체를 입력 파일로 바꾼다. 그래서 NextMe 초안에는 쓰려는 NextMe 날짜가 모두 들어 있어야 하고, 이번에 받은 날짜만 넣으면 전에 넣은 날짜가 카탈로그에서 빠진다. 검수한 초안은 `data/raw`에 있는데 `data/raw`는 동기화하지 않는다. 검수 결과를 잃지 않고 다른 서버에서도 쓰려면 사본을 따로 둔다.

타임라인에는 집(home)과 직장(workplace) 클립이 모두 있는 NextMe 날짜만 쓸 수 있다. 쓸 수 있는 날짜는 출근 전 `home_morning` 창(대개 1시간 안팎)에 들어갈 만큼 짧은 아침 집 영상과, 오전·오후 직장 영상이 모두 있어야 한다. 한 원본 날짜는 합성 하루 하나에만 쓰인다(D7). 매칭은 합성 하루를 되도록 한 원본 날짜로 채우지만, 맞는 클립이 없으면 하루가 원본 날짜 두 개를 쓰고 둘 다 다른 날에 다시 쓰지 못한다. 그래서 28일 타임라인에는 쓸 수 있는 NextMe 날짜가 28개보다 넉넉히 많아야 한다. 지금은 1일만 ingest했으므로 날짜를 더 받아야 한다.

### Ego4D (산책·이동·취미 등 이벤트)

Ego4D와 Ego-Exo4D 접근은 [Lab22](https://lab22.ai)로 옮겨졌다. 라이선스 승인 후 [Lab22 CLI](https://lab22.ai/docs/cli)로 annotations를 내려받는다. Ego4D에 필요한 파일은 메타데이터(`ego4d.json`)와 narration(`narration.json`)이고, 영상은 받지 않는다.

1. Lab22의 API Keys 페이지에서 키를 만들어 저장소 루트의 `.env`에 넣는다. `.env`는 git이 무시하고, 파일 권한은 600으로 둔다. 스크립트는 이 파일에서 `LAB22_API_KEY` 줄만 읽는다. 환경 변수 `LAB22_API_KEY`가 이미 있으면 그 값을 쓰고, 다른 파일을 쓰려면 `LAB22_ENV_FILE=<path>`를 준다.

   ```bash
   LAB22_API_KEY=<API 키>
   ```

2. 다운로드 스크립트를 실행한다. CLI는 `uvx`로 그때그때 받아 실행하므로 따로 설치하지 않아도 된다.

   ```bash
   scripts/download_lab22.sh ego4d      # 기본 출력 경로: data/raw/ego4d
   ```

   Lab22는 현재 릴리스 하나(지금은 `v2_1`)만 제공하므로 버전을 지정하지 않는다. 받은 파일의 폴더 구조는 릴리스를 따라 바뀔 수 있어서, 스크립트가 끝나면서 실제 경로로 다음 명령을 출력한다. `v2_1`에서는 `ego4d.json`이 출력 폴더 바로 아래에, narration이 `v2/annotations/narration.json`에 있다.

3. narration을 줄인다. `narration.json`은 1GB가 넘어 Node가 한 번에 읽을 수 없다. 이 스크립트는 `config/categories.json`의 ego4d scenario에 해당하는 영상의 `narration_pass_1`만 남긴다(`v2_1`에서 642개 영상, 약 55MB). ego4d scenario를 바꾸면 다시 실행한다.

   ```bash
   python3 scripts/convert_ego4d_narrations.py \
     --metadata data/raw/ego4d/ego4d.json \
     --narrations data/raw/ego4d/v2/annotations/narration.json \
     --out data/raw/ego4d/narration_selected.json
   ```

4. 줄인 파일로 ingest한다.

   ```bash
   npm run ingest -- ego4d --metadata data/raw/ego4d/ego4d.json --narrations data/raw/ego4d/narration_selected.json
   ```

카테고리 정의는 `config/categories.json`이다. ego4d 카테고리의 `scenarios`는 `v2_1` 메타데이터의 실제 scenario 이름으로 확정했다(2026-11-10). 메타데이터에 있는 scenario와 video 수는 아래 명령으로 볼 수 있다.

scenario가 맞는 영상의 summary 중에서 `keywords`의 단어가 들어 있는 것만 후보가 된다. 키워드는 대소문자를 가리지 않고 단어 단위로 비교한다(`car`는 `carried`, `card`에 걸리지 않는다). 그래서 활용형은 `tent`, `tents`처럼 따로 적는다. `keywords`가 비어 있으면 모든 summary가 후보다. summary 하나가 여러 카테고리에 맞으면 파일에서 먼저 나오는 카테고리로 간다. 그래서 ego4d 카테고리는 구체적인 것부터 둔다(lego, tent, transit, walking 순). Ego4D summary는 5분 단위라 여러 행동이 섞이므로, 실내에서 걷는 장면처럼 키워드로 거르지 못하는 후보가 남는다.

```bash
npm run ingest -- ego4d --metadata <ego4d.json> --list-scenarios
```

다른 카테고리 파일을 쓰려면 `--categories <path>`를 준다(기본값은 `config/categories.json`).

### Ego-Exo4D (요리·건강·자전거 수리·취미·운동 이벤트)

Ego4D와 같은 키로 메타데이터와 annotations만 내려받은 뒤 ingest한다. take 하나가 이벤트 후보 하나가 된다.

```bash
scripts/download_lab22.sh egoexo     # 기본 출력 경로: data/raw/egoexo
```

```bash
npm run ingest -- egoexo --root data/raw/egoexo
```

블록이 될 구간이 60초보다 짧은 take는 건너뛴다. 기준을 바꾸려면 `--min-sec <초>`를 준다. 도메인과 카테고리 매핑은 `config/categories.json`의 `dataset: "egoexo4d"` 항목이고, 다른 카테고리 파일을 쓰려면 `--categories <path>`를 준다.

ingest가 끝나면 `data/catalog/` 아래에 클립 카탈로그가 만들어지고, 건너뛴 항목의 사유별 개수가 JSON 리포트로 출력된다.

## 3. 타임라인 생성

world(합성 사람 한 명) 파일 하나가 입력이다. 예시는 `config/worlds/example-nextme.json`이다. world는 노드(집, 직장, 취미), 노드와 bank 소스의 연결(`node_binding`), 식사 습관, 다이얼(`length_days`, `routine_strength`, `exception_rate`), seed를 담는다.

규칙 설정은 `config/timeline/`에 있다.

| 파일 | 내용 |
|---|---|
| `templates.v1.json` | 요일 유형별 하루 템플릿 |
| `exceptions.v1.json` | 날 단위·슬롯 단위 예외 |
| `gaps.v1.json` | 전환 사이 공백 시간 표. `reviewed: false`이므로 통근 외 전환은 사람이 검토해야 한다(스펙 13절 열린 항목 1) |

`config/categories.json`의 `clip` 항목은 Ego4D 카테고리가 클립 bank에서 어떤 category, setting, node, tags로 쓰일지 정한다.

### 한 번에 만들기 (CLI)

```bash
npm run generate -- --world config/worlds/example-nextme.json [--data <dir>] [--config <dir>]
```

world를 `data/worlds/`에 저장하고(같은 `person_id`에 내용이 다르면 거부), 타임라인을 만들고, 자동 채움과 검증을 거쳐 manifest를 쓴 뒤 `timeline_id`, `revision`, `manifest` 경로, 일수, 클립 수, 평균 coverage를 JSON으로 출력한다. 실패하면 stderr에 오류 JSON을 쓰고 종료 코드 1로 끝난다.

### MCP 서버로 고쳐 가며 만들기

```bash
claude mcp add video-composer \
  -e VIDEO_BENCH_DATA_DIR=$PWD/data -e VIDEO_BENCH_CONFIG_DIR=$PWD/config \
  -- npx tsx $PWD/packages/mcp/src/main.ts
```

저장소 루트에서 실행한다(`npm run mcp`로 직접 띄울 수도 있다). 서버는 stdio로 동작한다.

| 도구 | 설명 |
|---|---|
| `list_worlds` | `data/worlds`의 world 목록 |
| `create_timeline` | world로 타임라인(schedule과 빈 슬롯) 생성 |
| `list_timelines` | 저장된 타임라인 목록 |
| `get_timeline` | 현재 타임라인과 슬롯 상태 |
| `list_slot_candidates` | 슬롯의 hard filter를 통과한 클립을 점수순으로 |
| `assign_clip` | 슬롯에 클립 배정(완화 단계 지정 가능) |
| `unassign_clip` | 슬롯에서 클립 하나 또는 전부 제거 |
| `auto_fill` | 빈 슬롯을 자동으로 채우고, 못 채우는 날은 다시 생성 |
| `verify_timeline` | 규칙 검사 후 `data/timelines/<id>.verifier_report.json` 작성 |
| `export_manifest` | 검증된 타임라인의 manifest 저장 |

편집 도구는 `expected_revision`을 받아, 다른 곳에서 먼저 바뀐 타임라인을 덮어쓰지 않게 한다.

흐름은 `list_worlds` → `create_timeline` → `auto_fill`(또는 `list_slot_candidates`와 `assign_clip`/`unassign_clip`으로 손질) → `verify_timeline` → `export_manifest`다. 이음새 VLM 검사(스펙 7.2)는 Plan 3이다.

## 4. Manifest

`export_manifest`(와 `generate`)는 아래 경로에 파일을 쓴다.

```
data/exports/<timeline_id>-r<revision>.manifest.json
```

manifest의 구조와 필드는 스펙 8절을 따른다. 검증을 통과하지 못한 타임라인은 내보내지 않는다.

## 5. 라이선스와 데이터 취급

- NextMe-800은 CC BY-NC 4.0이다. 연구 목적으로만 쓴다.
- Ego4D, Ego-Exo4D는 각 데이터셋의 약관·라이선스 계약을 따른다. YouTube 영상은 크리에이터와 플랫폼의 조건을 따른다.
- `data/`(원본 주석, 카탈로그, 타임라인, export된 manifest)는 라이선스가 있는 데이터에서 파생된 것이므로 절대 커밋하지 않는다. `.gitignore`에 들어 있다.

## 6. 서버 간 데이터 동기화

`data/`의 `catalog`, `worlds`, `timelines`, `exports`, `renders`, `distributions`는 비공개 Hugging Face 저장소 `zzu819/videomem`의 `data/` 폴더로 동기화한다. `huggingface-cli login`이나 `HF_TOKEN`으로 이 저장소에 접근할 수 있어야 한다.

```bash
python3 scripts/sync_data.py push   # 저장소가 비공개가 아니면 올리지 않는다
python3 scripts/sync_data.py pull   # 로컬 timelines가 있으면 --overwrite가 필요하다
```

원본 미디어와 `data/raw`는 동기화하지 않는다. 각 데이터셋에서 직접 받는다.

## 7. 테스트

```bash
npm test
npm run typecheck
python3 -m pytest scripts/
```
