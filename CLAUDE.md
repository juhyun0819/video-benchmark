# video-benchmark 작업 안내

서버를 옮겨도 이어서 작업할 수 있도록 합의 사항을 적어 둔다. 구조와 사용법은 `README.md`, 설계는 `docs/superpowers/specs/`에 있다.

## 작업 규칙

- 코드, 설정, 메시지, 스크립트, 테스트는 영어로 쓴다. 대화와 README, 스펙 본문은 한국어로 쓴다.
- 코드를 만들거나 git 작업을 할 때는 superpowers 스킬을 쓴다. 설계를 먼저 승인받고 구현한다.
- 커밋과 push는 요청받았을 때만 한다.

## 데이터 규칙

- `data/`는 NextMe-800, HD-EPIC, Ego-Exo4D, Ego4D에서 파생된 라이선스 데이터다. 커밋하거나 공개하지 않는다.
- 예외: `data/distributions/`(ATUS 산출물)는 미국 정부 공개 자료라 git에 커밋한다. 새 서버에서 clone만 하면 donor pool을 쓸 수 있게 하기 위해서다. 다른 `data/` 폴더는 계속 커밋하지 않는다.
- 공개 저장소에 올리지 않는다. 서버 간 동기화는 비공개 저장소 `zzu819/videomem`과 `scripts/sync_data.py`로 한다. 이 스크립트는 저장소가 비공개가 아니면 올리지 않는다.
- 동기화 대상은 `catalog`, `worlds`, `timelines`, `exports`, `renders`, `distributions`뿐이다. 원본 미디어와 `raw`는 올리지 않고 각 출처에서 받는다.
- 테스트 fixture는 합성 데이터만 쓴다.
- 토큰과 키 값은 출력하거나 기록하지 않는다. 있는지만 확인한다.
- Lab22 CLI(`LAB22_API_KEY`)는 사용자가 직접 실행한다.

## 파이프라인

1. 합성 타임라인 manifest 생성: 합성 사람 × N일 타임라인 파이프라인 (구현 중. Plan 1과 rev3 변경 완료, main에 병합. 스펙: `2026-11-12-synthetic-timeline-generation-design.md`. 이전 base day 방식과 두 옛 스펙은 대체됐다)
2. manifest로 영상 렌더링 (다음 계획 Plan 2). 이음새 VLM 검사는 Plan 3
3. 영상을 보고 샘플을 수정한 뒤 5개를 고정한다
4. 합성으로 생기는 QA를 자동으로 만들기 위한 QA 카테고리를 정한다
5. 샘플 QA를 생성하고 사용자가 직접 검증한다
6. 시나리오를 짜고 대량 생성한다

벤치마크 영상은 전 구간 1 fps이고 오디오와 화면 글자가 없다. 영상에 오디오가 없으므로, QA에서는 음성에만 근거한 caption을 걸러야 한다.

## 샘플

샘플 manifest는 `363c4d53-r6`, `db9af59e-r7`, `883d71cd-r3`, `e1b14c2f-r4`, `8459376a-r3`다.

영상을 본 뒤 고칠 후보는 다음과 같다.

- `883d71cd`는 이벤트가 3분 42초뿐이다
- 같은 자전거 체인 클립이 `883d71cd`와 `8459376a`에 모두 있다
- `8459376a`의 클라이밍 블록처럼 내부 task 이름이 그대로 요약인 경우가 있다
- Ego4D summary 원문에는 `#Summary C ...`처럼 표기 접두어가 붙어 있다. 블록 요약과 QA에 쓸 때 다듬을지 정한다
- 샘플마다 합성 QA 유형을 정해 두면 검증이 쉽다

## 원본 미디어

`data/media/`에 샘플에 쓰는 파일만 둔다.

| 데이터셋 | 받는 방법 |
|---|---|
| NextMe-800 | `hf download mmm8383/NextMe-800 raw/2026-04-15/2026-04-15_1323-1709.tar --repo-type dataset --local-dir data/media/nextme` (`huggingface_hub` 1.x부터 CLI 이름이 `hf`다) |
| Ego-Exo4D | `lab22 egoexo -o data/media/egoexo --parts takes --views ego --uids <16개 take uid>` |

Ego-Exo4D take uid는 샘플 manifest에서 `source.dataset`이 `egoexo4d`인 블록의 `video_id`다.

## 진행 중: NextMe 28일 manifest (2026-11-13, 블록 방식으로 완성)

목표는 NextMe 착용자 한 사람의 28일 manifest 하나를 완성하는 것이다. 슬롯마다 실제 어느 영상이 들어가는지 확정한다. 로직은 따르고, 이 영상을 위한 결정은 world와 입력으로 한다.

- 브랜치: `feat/nextme-28day`
- 설계: `docs/superpowers/specs/2026-11-12-nextme-28day-design.md`(늦은 출근, 여행, 라벨)와 `2026-11-13-nextme-28day-blocks-design.md`(블록 방식, B1~B17)
- 계획: 같은 이름의 `docs/superpowers/plans/`
- 결과와 재현 명령: `docs/superpowers/reports/2026-11-13-nextme-28day.md`
- 블록 방식:
  - 시각표는 ATUS donor가 정한다.
  - 각 블록에는 합성 하루마다 고른 실제 NextMe 날짜의 영상을 통째로 넣는다.
  - 빈 집 블록과 덜 찬 근무·집 블록은 남는 날짜의 영상으로 채운다.
  - 실행: `npm run compose-blocks -- --world config/worlds/nextme-p00.json --fit-roles hobby --pins config/worlds/nextme-p00.pins.json`
- 데이터(커밋하지 않음):
  - `data/raw/nextme-hf`(caption)
  - `data/raw/egoexo`(메타데이터)
  - `data/raw/bank/{nextme800.overrides.json, nextme800.l3.draft.jsonl, youtube.lego.jsonl}`
  - `data/catalog`
  - `data/exports`
- 남은 일:
  - 라벨 검수(D9)
  - 렌더링(Plan 2)
  - 최종 리뷰에서 나온 정리 항목

## 대기 중인 일

- NextMe-800 ingest는 위의 "진행 중" 절을 따른다.
- `config/timeline/gaps.v1.json`은 `reviewed: false`다. 통근 외 전환의 공백 시간 표를 사람이 검토한다(스펙 13절 열린 항목 1).
- YouTube 영상 10+10개(집 10, 직장 10)를 사용자가 내려받고 검수한 뒤 `ingest bank --dataset youtube`로 넣는다.
- 예전 로컬 파일(`data/catalog/hdepic/`, `data/catalog/{ego4d,egoexo4d}/candidates.jsonl`, `data/catalog/nextme800/days/`, `data/compositions/`)은 사용자가 지운 뒤 push한다.
- 새 CLI로 `ingest ego4d`와 `ingest egoexo`를 다시 실행해 클립 카탈로그를 만든다. 지금 로컬에는 예전 `annotations/`와 `candidates.jsonl`만 있고 클립 카탈로그가 없어서 bank가 비어 있다. 순서는 README 2절의 Ego4D, Ego-Exo4D 절을 따른다.
- 이력(클립 bank 이전): Ego4D는 2026-11-10에 주석을 받아 ingest했고, 2026-11-11에 walking, transit 키워드를 넣고 단어 단위 비교로 바꿔 다시 ingest했다(walking 1811, transit 542, hobby_tent 37, hobby_lego 29). 이 결과는 예전 `candidates.jsonl`이며 새 bank에는 없다. 주석 다운로드 절차는 README의 Ego4D 절이고, Lab22 키는 저장소 `.env`의 `LAB22_API_KEY`다.

## 결정 대기 (타임라인 rev3)

- 쉬는 날(hobby_day) 저녁 drop은 G20(hobby 슬롯을 운동 앵커 순서에 맞춰 끼니 사이에 둠) 뒤 hobby day의 1.3%다(rev3 전 4.5%, G20 전 12.0%). 남은 drop은 16시 이후 시작하는 hobby에 몰려 있다(그 시각대의 5.0%). 이 정도면 그대로 둘지, 쉬는 날 저녁을 필수로 하거나 hobby_day에 `reduce_order`를 줄지 사용자가 정한다. G20의 대가로 hobby_day 슬롯 순서가 그날 운동 시각에 따라 달라진다(order_similarity가 조금 낮아짐). 수치는 비교 리포트(`docs/superpowers/reports/2026-11-12-rev3-comparison.md`) 4절과 8절에 있다.
- G22(donor 경계는 템플릿이 쓰는 앵커만)는 사용자가 승인한 E5의 문구를 바꾼다. 평일 donor는 아침을 검사하지 않는다. 승인이 필요하다.
- seed 149 건은 해결됐다. 첫 donor가 아침 위반 하나로 거절되고 다시 뽑은 donor가 모두 필수 슬롯 검사를 못 넘던 경우였다. G22 뒤에는 첫 donor가 그대로 수락된다(redraw 0, 위반 없음). seed 0–999 × rs {0, 0.5, 0.7, 1} sweep의 생성 실패는 0/4000이다(이전 7/4000, seed 242와 747). 최종 재검토 결과 이 0은 G22만으로 나온다. G21은 자기 테스트 fixture로 검증된다.

## 서버 환경

- 지금 서버(convei-rack3)의 Node 24는 `/convei_nas2/zzuhyun/.node/current/bin`에 있다(NAS라 서버를 옮겨도 남는다). 시스템 node는 18이라 쓸 수 없다. 쓰기 전에 `export PATH="/convei_nas2/zzuhyun/.node/current/bin:$PATH"`를 실행한다. 이전 서버는 `/NHNHOME/langlab/zzuhyun/.node/current/bin`이었다.
- Lab22 다운로드 스크립트는 `uvx`가 필요하다. `pip install --user uv`로 `~/.local/bin`에 설치하고 PATH에 넣는다.
