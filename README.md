# video-benchmark

NextMe-800, HD-EPIC, Ego4D의 주석(caption)을 정규화한 카탈로그로 만들고, Claude Code에서 MCP 도구로 "하루짜리 영상 타임라인"을 합성한 뒤 manifest(JSON)로 내보내는 도구다. 영상 자체는 다루지 않고, 합성된 구간 구조와 시간 재매핑된 caption만 만든다.

구성은 npm workspaces 세 패키지다.

| 패키지 | 역할 |
|---|---|
| `@vb/core` | 스키마, 타일링, 편집 연산, 타임라인 계산, caption 재매핑, manifest 생성 |
| `@vb/ingest` | 데이터셋별 어댑터와 `ingest` CLI. `data/catalog/`를 쓴다 |
| `@vb/mcp` | stdio MCP 서버 `video-composer`. 카탈로그를 읽기 전용으로 쓴다 |

설계 문서는 `docs/superpowers/specs/2026-11-02-video-composer-mcp-design.md`에 있다.

## 1. 준비

Node 24를 쓴다(`.nvmrc`가 24).

```bash
nvm install   # .nvmrc를 읽는다
nvm use
npm install
```

데이터 루트는 기본값이 `./data`다. `VIDEO_BENCH_DATA_DIR` 환경변수로 바꿀 수 있고, ingest CLI는 `--data <dir>` 플래그로도 지정할 수 있다(플래그가 환경변수보다 우선).

## 2. 데이터 가져오기 (ingest)

모든 명령은 `npm run ingest -- <데이터셋> ...` 형태다. 도움말은 `npm run ingest -- --help`.

### NextMe-800 (base day)

NextMe-800은 Hugging Face의 gated 데이터셋(`mmm8383/NextMe-800`)이다. 먼저 접근 승인을 받아야 한다. 토큰은 둘 중 하나로 준다.

```bash
export HF_TOKEN=hf_xxx          # 방법 1: 환경변수
huggingface-cli login           # 방법 2: 로그인 (토큰이 ~/.cache/huggingface/token 에 저장됨)
```

`HF_TOKEN`이 없으면 CLI가 `~/.cache/huggingface/token`을 읽는다. 둘 다 없으면 오류로 끝난다.

```bash
npm run ingest -- nextme --days 2026-04-15
npm run ingest -- nextme --days 2026-04-15,2026-04-16
```

`--days`는 `YYYY-MM-DD`를 쉼표로 이어서 준다. 한 날짜가 하나의 base day가 된다.

### HD-EPIC (요리 이벤트)

주석 저장소를 clone한다(`data/`는 커밋 대상이 아니다).

```bash
git clone --depth 1 https://github.com/hd-epic/hd-epic-annotations data/raw/hd-epic-annotations
```

narration은 pkl로만 배포되고 numpy 2가 필요하다. 한 번만 JSONL로 변환한다. pkl은 clone한 저장소의 `narrations-and-action-segments/HD_EPIC_Narrations.pkl`이다.

```bash
uv run --isolated --no-project --with pandas --with "numpy>=2" \
  python scripts/convert_hdepic.py \
    data/raw/hd-epic-annotations/narrations-and-action-segments/HD_EPIC_Narrations.pkl \
    data/raw/hdepic_narrations.jsonl
```

그 다음 ingest한다.

```bash
npm run ingest -- hdepic \
  --annotations data/raw/hd-epic-annotations \
  --narrations data/raw/hdepic_narrations.jsonl
```

`--annotations`는 clone한 저장소의 루트 디렉터리다.

### Ego4D (산책·이동·취미 등 이벤트)

Ego4D와 Ego-Exo4D 접근은 [Lab22](https://lab22.ai)로 옮겨졌다. 라이선스 승인 후 [Lab22 CLI](https://lab22.ai/docs/cli)로 annotations를 내려받는다. Ego4D에 필요한 파일은 메타데이터(`ego4d.json`)와 narration(`narration.json`)이고, 영상은 받지 않는다.

1. Lab22의 API Keys 페이지에서 키를 만들어 `~/.config/lab22/env`에 넣는다. 파일 권한은 600으로 두고, 저장소에는 넣지 않는다. 환경 변수 `LAB22_API_KEY`가 이미 있으면 그 값을 쓴다.

   ```bash
   export LAB22_API_KEY='<API 키>'
   ```

2. 다운로드 스크립트를 실행한다. CLI는 `uvx`로 그때그때 받아 실행하므로 따로 설치하지 않아도 된다.

   ```bash
   scripts/download_lab22.sh ego4d      # 기본 출력 경로: data/raw/ego4d
   ```

3. 받은 파일로 ingest한다.

   ```bash
   npm run ingest -- ego4d --metadata data/raw/ego4d/v2/ego4d.json --narrations data/raw/ego4d/v2/annotations/narration.json
   ```

카테고리 정의는 `config/categories.json`이다. 이 파일의 `scenarios` 값은 임시값이다. 실제 메타데이터에 있는 scenario 이름과 맞는지 먼저 확인하고, 다르면 파일을 고친다. 메타데이터에 있는 scenario와 video 수는 아래 명령으로 볼 수 있다.

```bash
npm run ingest -- ego4d --metadata <ego4d.json> --list-scenarios
```

다른 카테고리 파일을 쓰려면 `--categories <path>`를 준다(기본값은 `config/categories.json`).

### Ego-Exo4D (다운로드만)

Ego-Exo4D는 같은 키로 메타데이터와 annotations만 받을 수 있다. 아직 ingest 어댑터는 없다.

```bash
scripts/download_lab22.sh egoexo     # 기본 출력 경로: data/raw/egoexo
```

ingest가 끝나면 `data/catalog/` 아래에 카탈로그와 인덱스가 만들어지고, 건너뛴 항목의 사유별 개수가 JSON 리포트로 출력된다.

## 3. MCP 서버 등록

```bash
claude mcp add video-composer -e VIDEO_BENCH_DATA_DIR=$PWD/data -- npx tsx $PWD/packages/mcp/src/main.ts
```

저장소 루트에서 실행한다. 서버는 stdio로 동작하고, 데이터 루트를 `VIDEO_BENCH_DATA_DIR`(기본 `./data`)에서 읽는다.

### 도구

| 구분 | 도구 | 설명 |
|---|---|---|
| 탐색 | `list_base_days` | ingest된 base day와 블록 수, 총 길이 |
| | `get_base_day` | base day의 블록 목록 |
| | `list_categories` | 이벤트 카테고리와 후보 수 |
| | `list_event_candidates` | 카테고리별 후보 (`query`, `min_sec`, `max_sec`, `limit`, `offset`) |
| | `get_candidate` | 후보 상세와 caption 미리보기 |
| 편집 | `create_composition` | base day 위에 빈 컴포지션 생성 |
| | `list_compositions` | 저장된 컴포지션 목록 |
| | `get_composition` | 현재 타임테이블 |
| | `insert_event` | 슬롯에 이벤트 삽입 |
| | `move_event` | 이벤트를 다른 슬롯/위치로 이동 |
| | `remove_event` | 이벤트 제거 |
| 내보내기 | `export_manifest` | manifest 파일 저장 |

편집 도구는 `expected_revision`을 받아, 다른 곳에서 먼저 바뀐 컴포지션을 덮어쓰지 않게 한다.

### 사용 흐름

1. `list_base_days`로 쓸 날짜를 고른다.
2. `create_composition`으로 그 날짜의 컴포지션을 만든다.
3. `list_categories`, `list_event_candidates`, `get_candidate`로 넣을 이벤트를 찾는다.
4. `insert_event`로 슬롯에 넣는다. 슬롯 `i`는 base 블록 `i` 앞이다. 필요하면 `move_event`, `remove_event`로 고친다. 편집할 때마다 타임테이블이 돌아온다.
5. `export_manifest`로 내보낸다.

## 4. Manifest

`export_manifest`는 아래 경로에 파일을 쓴다(원자적 쓰기, 같은 revision을 다시 내보내면 같은 파일을 덮어쓴다).

```
data/exports/<composition_id>-r<revision>.manifest.json
```

반환값은 파일 경로, 총 길이(`H:MM:SS`), 블록 수, 데이터셋별 caption 수다. 참조한 후보가 카탈로그에서 사라졌으면 `카탈로그에 없는 후보: …`로 실패하고 아무것도 쓰지 않는다.

manifest에는 다음이 들어 있다.

- `blocks`: 합성 타임라인의 블록(base/event), 합성 시간 구간, 원본 영상과 원본 구간
- `tracks.captions`: 합성 시간으로 재매핑된 caption. 원본 구간과 원문은 `provenance`에 보존된다
- `licenses`: 사용된 데이터셋의 라이선스

## 5. 라이선스와 데이터 취급

- NextMe-800은 CC BY-NC 4.0이다. 연구 목적으로만 쓴다.
- HD-EPIC과 Ego4D는 각 데이터셋의 약관·라이선스 계약을 따른다.
- `data/`(원본 주석, 카탈로그, 컴포지션, export된 manifest)는 라이선스가 있는 데이터에서 파생된 것이므로 절대 커밋하지 않는다. `.gitignore`에 들어 있다.

## 6. 테스트

```bash
npm test
npm run typecheck
```
