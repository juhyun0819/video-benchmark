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

rev3에서 쉬는 날 끼니 사전값과 평일 저녁 집 슬롯 확률은 별도 스크립트로 계산한다. 입력은 위 `atus2025_stats_per_respondent.csv`이고, 출력은 `data/distributions/atus2025/atus2025_extra.json`이다.

```bash
uv run --isolated --no-project --with pandas --with numpy \
  python scripts/atus/atus_extra.py <activity.csv> <respondent.csv> \
  data/distributions/atus2025/atus2025_stats_per_respondent.csv \
  data/distributions/atus2025/atus2025_extra.json
```

- 근무하지 않은 주말 응답자(n=910)의 끼니별 장소 비율과 시각 → `config/schedule_defaults_atus2025.json`의 `meal_location_priors.rest_day`(점심: 집 0.67, 외식 0.27, 남의 집 0.06 / 저녁: 집 0.79, 외식 0.14, 남의 집 0.07).
- 평일 근무자(n=752)가 퇴근 후 취침 전까지 집에서 30분 이상 깨어 활동한 비율 0.91과 길이 분포(p25/p50/p75 = 105/187/270분) → `weekday_evening_home.slot_probability`, `weekday_evening_home.minutes`.

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
| `templates.v2.json` | 요일 유형별 하루 템플릿과 줄이는 순서(`reduce_order`) |
| `exceptions.v1.json` | 날 단위·슬롯 단위 예외 |
| `gaps.v1.json` | 전환 사이 공백 시간 표. `reviewed: false`이므로 통근 외 전환은 사람이 검토해야 한다(스펙 13절 열린 항목 1) |

### templates.v2

`templates.v2.json`은 `day_types`(workday, hobby_day, home_day)와 `reduce_order`를 가진다. leave_day는 home_day 템플릿을 쓴다.

- 평일 저녁 식사(`dinner`)는 필수 슬롯이다. 필수 슬롯은 줄이거나 빼지 않는다.
- 평일에 시간이 모자라면 `reduce_order.workday` 순서로 줄인다: 취미 길이 줄이기(90분 → 최소 30분), 취미 빼기, `home_evening` 빼기. `home_evening`은 취침까지 채우는 슬롯이고 최소 길이가 30분(ATUS의 30분 이상 정의)이라 "길이 줄이기"는 저절로 일어난다.
- `reduce_order`가 없는 템플릿(hobby_day, home_day)은 기존 규칙을 쓴다: 실패한 슬롯이 선택 슬롯이면 그것을, 아니면 그 앞의 가장 가까운 선택 슬롯을, 없으면 뒤의 마지막 선택 슬롯을 뺀다.
- 평일 저녁 `home_evening`이 있을 확률은 `config/schedule_defaults_atus2025.json`의 `weekday_evening_home.slot_probability`(0.91)다. 이 슬롯은 취침까지 채워지므로 확률은 슬롯이 없는 날에만 영향을 준다. 그 시간은 마지막 슬롯(대개 저녁) 장소의 `unscheduled` 시간이 된다. 저녁을 밖에서 먹은 날이면 집이 아니라 그 장소다.
- 쉬는 날(hobby_day)의 hobby 슬롯은 템플릿 위치에 고정되지 않는다. sport 앵커보다 늦은 첫 끼니(아침, 점심, 저녁) 바로 앞에 두고, 그런 끼니가 없으면 마지막 끼니 뒤에 둔다. 다른 슬롯(`home_afternoon` 등)은 끼니에 대한 템플릿 순서를 지킨다. 그래서 18:30 운동이면 점심을 먼저 먹고 오후를 보낸 뒤 hobby, 저녁 순이 된다. 이 때문에 hobby_day의 슬롯 순서는 그날 운동 시각에 따라 달라진다.
- 쉬는 날의 저녁은 필수가 아니다. 위 순서 덕분에 비교 리포트 설정에서 hobby_day의 1.3%에서만 빠진다(이전 12.0%).

### world의 `meal_style`과 검증

`meal_style`은 `workday`(lunch, dinner)와 `rest_day`(breakfast, lunch, dinner)를 가진다. 이전 이름 `weekend`는 `rest_day`로 바뀌었다. `rest_day`는 주말뿐 아니라 연차·병가 같은 하루 예외에도 쓴다. 예시는 `config/worlds/example-nextme.json`이다.

world 검증은 생성 전에 다음을 확인하고, 어기면 `WORLD_INVALID: <path>: <message>` 형식 오류를 낸다. 이 문구는 MCP 도구 오류 텍스트이고, generate CLI는 stderr에 `{"error":{"code","message","details"}}`를 출력한다. 문제가 여러 개면 `; `로 이어 붙인다.

- 끼니별 장소 확률의 합은 1 ± 0.01이어야 한다. 합이 0이어도 오류다. 예: `WORLD_INVALID: meal_style.workday.dinner: meal weights must sum to 1 +/- 0.01, got 0.5`
- `rest_day`의 끼니에는 `workplace`가 있으면 안 된다(`meal_style.rest_day.lunch.workplace: rest_day meals cannot take place at the workplace`).
- 취미 노드는 1~3개여야 한다.
- 바인딩되지 않은 장소를 뺀 뒤에도 평일 저녁 확률이 남아 있어야 한다(생성 시점에 검사).
- `meal_style.weekend`가 남아 있는 이전 world는 오류 끝에 `hint: meal_style.weekend was renamed to meal_style.rest_day in rev3; rename it`이 붙는다.

`config/categories.json`의 `clip` 항목은 Ego4D 카테고리가 클립 bank에서 어떤 category, setting, node, tags로 쓰일지 정한다.

### 일정 생성 규칙 (rev3)

- **donor 다시 뽑기**: donor(ATUS 응답자)의 시각이 범위를 벗어나면 잘라내지 않고 같은 backoff 단계에서 다시 뽑는다(최대 5회). 범위는 기상 04:00 이후, 취침 27:59 이전, 저녁 ≤ 취침−60분, 평일은 퇴근 ≥ 출근+240분과 점심이 근무 안에 들어가는 것, 평일 필수 슬롯이 들어갈 시간이 있는 것, 주말은 아침 ≥ 기상+10분과 운동 ≤ 취침−180분이다. 템플릿이 쓰지 않는 앵커는 검사하지 않는다(평일 템플릿에는 아침 슬롯이 없고, 주말 템플릿에는 근무가 없다). 5회 모두 실패했을 때만 값을 범위 안으로 당기고 `{item, from, to}`로 기록한다. 기록 위치는 schedule의 `donors.<weekday|weekend>.adjustments`와 manifest의 `donor_adjustments`다. 평일 donor가 그래도 필수 슬롯 검사를 못 넘으면 `work_end`를 필수 슬롯이 들어가는 가장 늦은 분으로 당기고 같은 곳에 기록한다.
- **날 단위 jitter**: 날마다 앵커(기상, 취침, 출근, 퇴근, 끼니, 운동)에 jitter를 더한다. 그 날의 템플릿이 쓰는 앵커만 범위를 검사하고, 어긴 항목만 항목별로 다시 뽑는다(최대 5회). 그래도 안 되면 당기고 ScheduleDay의 `anchor_adjustments`에 기록한다. 총 다시 뽑은 횟수는 `jitter_redraws`다.
- **평일 필수 슬롯 검사(날 단위)**: 평일에는 퇴근과 저녁을 뽑을 때 그 날의 전환 공백 표본으로 필수 슬롯(아침 집, 오전·오후 근무, 저녁)이 들어가는지도 본다. 안 들어가면 그 항목을 다시 뽑는다(최대 5회). 그래도 안 되면 퇴근을 필수 슬롯이 들어가는 가장 늦은 분(출근+240분 이상)으로 당기고 `anchor_adjustments`에 `{item: "work_end", from, to}`로 기록한다.
- **운동 시각 채우기**: donor에게 운동 기록이 없으면 같은 backoff 단계에서 운동 기록이 있는 응답자(30명 이상)의 값으로 채운다(`sport_fill: "level"`). 30명 미만이면 주말 pool 전체에서 채운다(`"pool"`). pool에도 운동 기록이 없으면 `schedule_defaults`의 `weekend.sports_start.p50`을 쓴다(`"default"`). 셋 다 `filled`에 `sport_time`으로 남는다.
- **슬롯 예외**: 선택 슬롯을 정리한 뒤에 추첨한다. 후보는 남은 슬롯 중 예외 카탈로그의 `replaces_role`에 해당하고 장소를 쓸 수 있는 끼니이다. 예외가 슬롯 정리 때문에 지워지는 경로는 없다. 추첨된 예외의 9.6%(이 world 기준, [비교 리포트](docs/superpowers/reports/2026-11-12-rev3-comparison.md) 참고. 이전에 적었던 약 16%는 다른 테스트 설정에서 잰 값이다)는 루틴과 같은 장소라 실제로는 아무것도 바꾸지 않는다(`slot_exception_noop_share`).

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
| `list_worlds` | `data/worlds`의 world 목록. 읽지 못한 파일은 건너뛰고 `skipped`에 적는다 |
| `create_timeline` | world로 타임라인(schedule과 빈 슬롯) 생성 |
| `list_timelines` | 저장된 타임라인 목록. 읽지 못한 파일은 건너뛰고 `skipped`에 적는다 |
| `get_timeline` | 현재 타임라인과 슬롯 상태 |
| `list_slot_candidates` | 슬롯의 hard filter를 통과한 클립을 점수순으로 |
| `assign_clip` | 슬롯에 클립 배정(완화 단계 지정 가능) |
| `unassign_clip` | 슬롯에서 클립 하나 또는 전부 제거 |
| `auto_fill` | 빈 슬롯을 자동으로 채우고, 못 채우는 날은 다시 생성 |
| `verify_timeline` | 규칙 검사 후 `data/timelines/<id>.verifier_report.json` 작성 |
| `export_manifest` | 검증된 타임라인의 manifest 저장 |

편집 도구는 `expected_revision`을 받아, 다른 곳에서 먼저 바뀐 타임라인을 덮어쓰지 않게 한다.

흐름은 `list_worlds` → `create_timeline` → `auto_fill`(또는 `list_slot_candidates`와 `assign_clip`/`unassign_clip`으로 손질) → `verify_timeline` → `export_manifest`다. 이음새 VLM 검사(스펙 7.2)는 Plan 3이다.

### rev3 이전 파일 (마이그레이션)

rev3에서 schedule의 `generator_version`은 `schedule.v2`, manifest의 `schema_version`(과 `versions.schema`)은 2가 됐다. 이전 파일은 변환하지 않고 다시 만든다.

- 저장된 타임라인(`data/timelines/*.json`): 현재 schema로 읽히지 않으면 `TIMELINE_INVALID`가 난다. 메시지에 "regenerate it with create_timeline"이 붙는다. 같은 world로 `create_timeline`을 다시 실행한다. `list_timelines`는 이런 파일을 건너뛰고 `skipped`(`{file, code, message}`)와 텍스트의 `skipped <file>: ...` 줄로 알린다.
- world(`data/worlds/*.json`): `meal_style.weekend`를 `meal_style.rest_day`로 바꾼다. 바꾸기 전에는 `WORLD_INVALID`와 이름 변경 안내가 나고, `list_worlds`는 그 파일을 건너뛰고 `skipped`에 적는다.
- manifest(`data/exports/*.manifest.json`): `schema_version` 1인 타임라인 manifest는 `npm run stats`가 "re-export the timeline" 오류로 멈춘다. 타임라인을 다시 만들고 내보낸다. 합성 영상용 composition manifest(`composition_id`가 있는 파일)는 stats가 stderr 경고와 함께 건너뛴다.

## 4. Manifest

`export_manifest`(와 `generate`)는 아래 경로에 파일을 쓴다.

```
data/exports/<timeline_id>-r<revision>.manifest.json
```

manifest의 구조와 필드는 스펙 8절을 따른다. 검증을 통과하지 못한 타임라인은 내보내지 않는다.

### 영상이 없는 구간 (`days[].no_video_spans`)

날마다 `no_video_spans: [{start, end, kind}]`가 있다. 시각은 1일차 00:00부터의 초(소수 셋째 자리)이고, 클립과 이 구간이 [기상, 취침]을 빈틈과 겹침 없이 채운다.

| kind | 뜻 |
|---|---|
| `transition` | 슬롯 사이 전환 공백(통근 등). 일정에 잡힌 길이만큼, 다음 슬롯 바로 앞에 둔다 |
| `unrecorded` | 슬롯 안에서 클립이 덮지 못한 시간. 매칭 단계에서 비워진 슬롯은 전체가 `unrecorded`다 |
| `unscheduled` | 슬롯 사이에서 전환을 빼고 남는 시간(전환보다 앞에 둔다)과, 마지막 슬롯부터 취침까지의 시간. 그 사람은 직전 슬롯의 장소에 있는 것으로 본다. 마지막 슬롯이 집 밖(예: 외식 저녁)이면 취침까지의 시간도 그 장소다 |

### `realized`: 다이얼이 실제로 만든 값

`schedule.realized`와 manifest의 `realized`는 다이얼(`routine_strength`, `exception_rate`)이 생성 결과에 실제로 얼마나 반영됐는지를 날들에서 세어 기록한다. manifest의 값은 매칭이 끝난 최종 날들로 다시 계산한 값이고, `schedule.realized`는 매칭 전 값이다. manifest에는 `matching_dropped_slots`, `matching_dropped_by_role`(매칭 단계에서 비워진 슬롯)이 더 붙는다. 분모가 0인 비율은 `null`이다.

- `optional_presence`와 `composition_consistency`는 매칭에서 비워진 슬롯도 남은 슬롯으로 센다. 일정이 만든 구성을 재는 값이고, 매칭 단계의 손실은 `matching_dropped_*`로 따로 본다.
- manifest의 각 날에는 `slot_exception_eligible`, `slot_exception_candidates`, `slots: [{slot_id, role, required, window}]`가 있다. 그래서 manifest만으로 `realized`를 다시 셀 수 있다(core의 `realizedOf(manifest.days, manifest.inputs.world, templates)`).

| 필드 | 뜻 |
|---|---|
| `slot_exception_rate` | 슬롯 예외가 있는 날 ÷ 예외가 가능한 날(하루 예외가 없고 후보 슬롯이 하나 이상). `exception_rate.slot`과 맞아야 하는 값 |
| `slot_exception_share` | 슬롯 예외 수 ÷ 후보 슬롯 수 |
| `slot_exception_noop_share` | 슬롯 예외 중 루틴과 같은 장소인 비율(분석에서 제외할 때 쓴다) |
| `slot_exception_eligible_days` | 예외가 가능했던 날 수 |
| `day_exception_rate` | 하루 예외 ÷ 하루 예외 대상 날(평일) |
| `optional_presence` | 선택 슬롯별 등장률: 그 슬롯을 선택으로 둔 템플릿의 날 중 남은 날의 비율 |
| `shrunk` | 줄이기 단계가 있는 역할(예: hobby)이 줄어든 채 남은 날 수. 줄었다가 빠진 슬롯은 빠진 것으로 센다 |
| `composition_consistency` | 템플릿별로 날 쌍들의 구성 일치도(아래) |
| `dials` | 이 값이 나온 다이얼 |

`composition_consistency.<template>`는 `{optional_jaccard, order_similarity, start_diff_minutes, pairs, same_reference}`다.

- `optional_jaccard`: 선택 슬롯 역할 집합의 평균 Jaccard
- `order_similarity`: 역할 순서의 평균 `1 − 편집거리 ÷ 최대 길이`
- `start_diff_minutes`: 같은 역할 슬롯 시작 시각 차이의 평균(분)
- `pairs`: 그 템플릿을 쓴 날의 모든 쌍 수
- 위 세 값은 모든 쌍(all-pairs)에 대한 평균이다. 평일 쌍의 Jaccard와 순서 일치도는 `routine_strength`를 바꿔도 거의 움직이지 않는다(비교 리포트 8절의 예시 world 기준 0.866/0.874/0.887). 루틴 기준이 요일마다 따로 있어서 다른 요일끼리는 원래 다르기 때문이다.
- `same_reference`: 같은 세 값을 루틴 기준을 공유하는 쌍(평일은 같은 요일, 나머지는 같은 날 유형)에만 적용한 것이다. 쌍이 없으면 `null`. 다이얼 효과는 이 값에서 읽는다(같은 리포트 기준 0.866 → 0.894 → 0.967). 값은 world와 seed에 따라 달라진다. all-pairs 값은 사용자가 승인한 정의이고 `same_reference`는 거기에 더한 것이라, 다른 정의를 원하면 바꿀 수 있다.

### 통계 보기

```bash
npm run stats -- <manifest.json 또는 폴더> [...]
```

인자는 manifest 파일이나 `*.manifest.json`이 있는 폴더이고 여러 개를 줄 수 있다. 다이얼 조합(`routine_strength`, `exception_rate.slot`, `exception_rate.day`)별로 묶어 `realized`의 각 필드(점 경로, 예: `composition_consistency.workday.same_reference.optional_jaccard`)의 평균, 최솟값, 최댓값, manifest 수 n을 JSON으로 stdout에 낸다. 값이 없거나 `null`인 manifest는 그 필드에서 빠지므로 n이 manifest 수보다 작을 수 있다. 또 donor 다시 뽑기 횟수 분포와 당김이 있는 donor·날의 비율을 낸다. 폴더에 composition manifest(`composition_id`가 있는 파일)가 섞여 있으면 stderr에 `skipping <file>: ...`을 쓰고 건너뛴다.

### rev3 전후 비교

rev3 변경(기준 ea048f5)이 분포를 어떻게 바꿨는지는 [비교 리포트](docs/superpowers/reports/2026-11-12-rev3-comparison.md)에 있다. 다시 돌리는 명령은 다음과 같다. `$BASE`(기준 worktree)와 `$OUT`(출력 디렉터리)은 저장소 밖에 둔다. 기본값은 `--seeds 200 --full-seeds 20 --rs 0,0.5,1 --slot 0.5 --day 0.1`이다.

```bash
BASE=<저장소 밖 경로>/rev3-baseline
OUT=<저장소 밖 경로>/rev3cmp
git worktree add "$BASE" ea048f5
ln -s "$PWD/node_modules" "$BASE/node_modules"
npx tsx scripts/compare/rev3_compare.ts run --root "$BASE" --label old --data data --out "$OUT"
npx tsx scripts/compare/rev3_compare.ts run --root . --label new --data data --out "$OUT"
npx tsx scripts/compare/rev3_compare.ts report --old "$OUT/old.json" --new "$OUT/new.json"
rm "$BASE/node_modules" && git worktree remove "$BASE"
```

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
