# video-benchmark 작업 안내

서버를 옮겨도 이어서 작업할 수 있도록 합의 사항을 적어 둔다. 구조와 사용법은 `README.md`, 설계는 `docs/superpowers/specs/`에 있다.

## 작업 규칙

- 코드, 설정, 메시지, 스크립트, 테스트는 영어로 쓴다. 대화와 README, 스펙 본문은 한국어로 쓴다.
- 코드를 만들거나 git 작업을 할 때는 superpowers 스킬을 쓴다. 설계를 먼저 승인받고 구현한다.
- 커밋과 push는 요청받았을 때만 한다.

## 데이터 규칙

- `data/`는 NextMe-800, HD-EPIC, Ego-Exo4D, Ego4D에서 파생된 라이선스 데이터다. 커밋하거나 공개하지 않는다.
- 공개 저장소에 올리지 않는다. Hugging Face `zzu819/videomem`은 공개 저장소다.
- 테스트 fixture는 합성 데이터만 쓴다.
- 토큰과 키 값은 출력하거나 기록하지 않는다. 있는지만 확인한다.
- Lab22 CLI(`LAB22_API_KEY`)는 사용자가 직접 실행한다.

## 파이프라인

1. manifest 생성 (완료)
2. manifest로 영상 합성 (진행 중, 스펙: `2026-11-07-video-render-design.md`, 승인 대기)
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
- 샘플마다 합성 QA 유형을 정해 두면 검증이 쉽다

## 원본 미디어

`data/media/`에 샘플에 쓰는 파일만 둔다.

| 데이터셋 | 받는 방법 |
|---|---|
| NextMe-800 | `huggingface-cli download mmm8383/NextMe-800 raw/2026-04-15/2026-04-15_1323-1709.tar --repo-type dataset --local-dir data/media/nextme` |
| Ego-Exo4D | `lab22 egoexo -o data/media/egoexo --parts takes --views ego --uids <16개 take uid>` |
| HD-EPIC | `hd-epic-downloader.py <data/media/hdepic 절대 경로> --videos --video-id P03-20240217-192543,P09-20240623-120359` |

Ego-Exo4D take uid는 샘플 manifest에서 `source.dataset`이 `egoexo4d`인 블록의 `video_id`다.

## 대기 중인 일

- Ego4D는 라이선스 승인을 기다린다. 승인되면 `scripts/download_lab22.sh ego4d`로 받고 `--list-scenarios`로 `config/categories.json`의 매핑을 확정한다.
