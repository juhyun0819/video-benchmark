# 시간표 기준 영상 타임스탬프

이미 만들어진 **1fps 영상**의 왼쪽 아래에 합성 날짜와 시각을 표시한다. 원본 타임스탬프 영역은 불투명한 검정색으로 가리고, 새 타임스탬프는 흰색 굵은 고정폭 글꼴로 쓴다. 입력 영상은 그대로 두고 클립별 출력 파일을 별도 폴더에 만든다.

영상 안에서는 재생 1초마다 표시 시각도 1초 증가한다. 다음 클립으로 넘어가면 그 클립의 시간표상 시작 시각으로 바뀐다. 예를 들어 아침 식사 영상이 `07:35:00`에 시작해 10분 동안 재생되고 다음 영상이 점심이라면, 점심 영상의 첫 표시는 `12:30:00`이다. 그 사이의 빈 시간을 영상으로 채우거나 재생 속도를 바꾸지 않는다.

## 준비

- Linux, Python 3.10 이상
- `ffmpeg`, `ffprobe`: `drawtext` 필터와 글꼴 렌더링 지원 필요
- DejaVu Sans Mono Bold 시스템 글꼴. 다른 글꼴은 `--font /path/to/font.ttf`로 지정
- 다운로드 및 1fps 변환이 완료된 입력 파일

별도 Python 패키지를 설치할 필요는 없다. 이 도구는 영상 다운로드나 원본의 1fps 변환을 수행하지 않으므로 데이터셋 인증 키도 요구하지 않는다.

```bash
python3 --version
ffmpeg -hide_banner -filters
ffprobe -version
python3 scripts/timestamp_timeline.py --help
```

아래 명령은 저장소 루트에서 실행한다. 예시의 `/data`는 실행할 서버의 데이터 경로로 바꾼다. 입력 영상, 실행 계획, 출력 영상은 Git 저장소 밖에 두고 코드와 설정 예시만 공유할 수 있다.

## 현재 시간표에서 실행하기

### 1. manifest에서 실행 계획 생성

```bash
python3 scripts/timestamp_timeline.py plan \
  --manifest /data/carpenter_profile/caption-pipeline/manifest.json \
  --base-date 2025-07-07 \
  --timezone Asia/Seoul \
  --out /data/timestamp-overlay/plan.json
```

`2025-07-07`이 Day1 월요일이고 Day28은 `2025-08-03` 일요일이다. 배치가 활성화된 슬롯만 실행 계획에 포함하며, 제외된 일정에 새 영상을 만들지 않는다. 이 계획 생성은 시간표를 다시 편성하거나 원본 구간을 다시 선택하는 작업이 아니다.

### 2. 클립별 타임스탬프 출력

```bash
python3 scripts/timestamp_timeline.py render \
  --plan /data/timestamp-overlay/plan.json \
  --output /data/timestamp-overlay/videos \
  --workers 2 \
  --threads 2
```

`--workers`는 동시에 처리할 클립 수, `--threads`는 각 작업의 스레드 설정이다. 우선 `2 × 2`로 시작하고 같은 서버의 다운로드·변환 작업과 자원을 나누어 쓴다. 워커 수를 늘린다고 항상 빨라지는 것은 아니다.

인코딩 기본값은 `--preset veryfast --crf 18`이다. 이 옵션을 바꾸어 속도와 압축 품질을 조절할 수 있지만, 프레임 선택이나 재생시간을 바꾸는 옵션은 아니다.

일부 클립만 확인할 때는 계획의 `id`를 지정한다. `--only`는 여러 번 쓸 수 있다.

```bash
python3 scripts/timestamp_timeline.py render \
  --plan /data/timestamp-overlay/plan.json \
  --output /data/timestamp-overlay/preview \
  --workers 1 \
  --threads 2 \
  --only day01-breakfast \
  --font /usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf
```

`day01-breakfast`는 아래 일반 계획 예시의 ID다. manifest에서 만든 계획에는 그 계획에 실제로 기록된 ID를 사용한다.

### 3. 다른 서버의 NAS 경로가 다를 때

```bash
python3 scripts/timestamp_timeline.py render \
  --plan /data/timestamp-overlay/plan.json \
  --output /data/timestamp-overlay/videos \
  --workers 2 \
  --threads 2 \
  --path-map /mnt/source=/mnt/shared
```

`--path-map OLD=NEW`는 입력 경로의 접두사를 현재 서버의 마운트 경로로 치환한다. 여러 경로가 다르면 옵션을 반복한다. 파일을 복사하거나 NAS를 마운트하는 기능은 아니므로, 바뀐 경로에서 입력 파일에 접근할 수 있어야 한다.

### 4. 1fps 변환이 끝나는 대로 처리

```bash
python3 scripts/timestamp_timeline.py render \
  --plan /data/timestamp-overlay/plan.json \
  --output /data/timestamp-overlay/videos \
  --workers 2 \
  --threads 2 \
  --watch \
  --poll-seconds 30
```

`--watch`는 **계획에 이미 적힌 입력 경로**를 주기적으로 확인한다. 변환 작업은 임시 파일에 쓰고 완료 후 최종 이름으로 바꾸는 방식이어야 한다. 최종 이름이 생겨도 실제로는 작성 중인 파일을 완료된 입력처럼 전달하면 안 된다.

새 영상이 배정되거나, 경로가 없던 슬롯의 입력 경로가 나중에 정해지거나, 일정이 바뀌면 `plan`을 다시 생성해야 한다. 실행 중인 `--watch`는 다음 주기에 계획 JSON을 다시 읽는다. 원본 manifest의 변경을 자동으로 반영하는 것은 아니다. 기존 다운로드·1fps 변환 작업을 재시작할 필요는 없다.

## 출력과 재실행

출력 폴더에는 다음 파일들이 생긴다.

```text
videos/
├── day01-day01-breakfast-<설정해시16자리>.mp4
├── day01-day01-breakfast-<설정해시16자리>.json
├── .day01-day01-breakfast-<설정해시16자리>.lock
└── run-<서버이름>-<PID>.json
```

파일 이름은 `day{day:02}-{id}-{설정해시 앞 16자리}` 형식이다. 클립별 JSON에는 입력 파일 정보, 시각·스타일·인코딩 설정, 검증 결과가 들어간다. `run-…json`은 해당 실행에서 처리한 클립들의 상태를 기록한다. 입력 파일과 설정, 기존 출력 검증이 모두 일치하면 그 결과를 재사용한다. 날짜·시각·스타일·입력 또는 처리 설정이 달라지면 다른 이름의 결과를 생성한다.

동일한 결과를 두 워커가 동시에 만들지 않도록 파일 잠금을 사용하며, 다른 워커가 처리 중이면 `owned_by_other_worker`로 기록한다. 실행 중 임시 파일은 `.part.mp4`이고 완료 후 최종 이름으로 바뀐다. 서로 다른 서버가 같은 NAS 출력 폴더를 쓸 경우 해당 NAS가 프로세스 간 파일 잠금을 지원해야 한다.

| 종료 코드 | 의미 |
|---|---|
| `0` | 선택한 모든 클립이 생성되었거나 기존 결과로 확인됨 |
| `1` | 계획·입력·처리 오류. 실행 리포트와 오류 메시지 확인 필요 |
| `3` | 한 번 확인한 결과 일부 입력이 없거나 다른 워커가 처리 중. `--watch` 없이 실행한 경우 |

`--watch`는 정상적인 입력 대기를 반복하지만, 실제 처리 오류가 있으면 코드 `1`로 종료한다. 잘못된 입력이나 설정을 무한히 재시도하지 않는다. `pending_input`은 경로는 있으나 파일이 아직 없는 상태이고, `missing_source_path`는 계획에 입력 경로 자체가 없는 상태다. 후자는 경로를 넣어 계획을 다시 생성해야 해결된다.

## 다른 프로젝트에서 쓰는 계획 형식

manifest 없이도 다음 구조의 JSON을 직접 만들어 `render --plan`에 전달할 수 있다. [설정 예시](../config/timestamp-overlay.example.json)도 참고한다.

```json
{
  "version": 1,
  "base_date": "2025-07-07",
  "timezone": "Asia/Seoul",
  "style": {
    "reference_height_px": 1024,
    "original_mask_xyxy": [0, 995, 200, 1024],
    "text_height_px": 20,
    "text_x_px": 10,
    "bottom_margin_px": 10,
    "box_padding_px": 7
  },
  "clips": [
    {
      "id": "day01-breakfast",
      "day": 1,
      "start_seconds": 27300,
      "input": "media/breakfast-1fps.mp4",
      "expected_duration_s": 600
    },
    {
      "id": "day01-lunch",
      "day": 1,
      "start_seconds": 45000,
      "input": "media/lunch-1fps.mp4",
      "expected_duration_s": 840
    }
  ]
}
```

| 필드 | 의미 |
|---|---|
| `version` | 계획 형식 버전. 현재 `1` |
| `base_date` | Day1의 합성 날짜, `YYYY-MM-DD` |
| `timezone` | 합성 시각의 IANA 시간대 표기. 예: `Asia/Seoul` |
| `clips[].id` | 클립의 고유 ID. `--only`로 선택할 때 사용 |
| `clips[].day` | 1부터 시작하는 합성 Day |
| `clips[].start_seconds` | 해당 Day의 자정부터 시작 시각까지의 초 |
| `clips[].input` | 완료된 1fps 파일. 상대 경로는 **계획 JSON의 폴더 기준** |
| `clips[].expected_duration_s` | 예정된 클립 재생시간. 시간표의 활동 창 길이와 구별 |
| `clips[].source` | 선택 사항. 원본 식별자·원본 구간 등 출처 추적용 정보 |

`expected_duration_s`는 검증용 길이이며 이 길이에 맞춰 입력을 임의로 자르거나 늘리는 지시가 아니다. 전체 원본을 쓰는 배치에는 전체 길이, 검토 후 특정 구간만 살린 배치에는 그 구간을 담은 1fps 파일과 구간 길이를 넣는다.

## 시각이 계산되는 기준

```text
표시 시각 = Day1 날짜
          + (day - 1)일
          + start_seconds
          + 클립 내부 재생 경과시간
```

현재 caption-pipeline manifest의 `slot.day`와 `proposed_observation.synthetic_start_s`가 각각 `day`와 `start_seconds`에 대응한다. 시작 초는 28일 전체의 누적 초가 아니라 **그날의 자정 기준 초**다. 슬롯 ID 문자열에서 Day를 추정하지 않는다.

입력의 PTS를 0부터 시작하도록 정규화한 뒤 재생 경과시간을 사용한다. 이미 원본의 일부를 추출한 파일에도 원본 `source.start_s`를 다시 더하지 않는다. 원본 촬영 시각이나 서버의 현재 시각 역시 표시 시각에 사용하지 않는다.

`timezone`은 합성 시각의 기준을 기록하기 위한 값이다. 이 계획은 일정표상의 벽시계 시각을 표현하며, 원본 촬영 시각을 UTC로 변환하는 작업이 아니다. 일광절약시간 전환을 포함하는 계획은 별도의 시각 정책을 확인해야 한다. 위의 서울 여름 일정에는 해당 전환이 없다.

## 검정 마스크와 글자 크기

| 설정 | 적용 방식 |
|---|---|
| 원본 글자 마스크 | 높이 1024px 기준 XYXY `(0, 995, 200, 1024)`를 불투명 검정색으로 가림 |
| 마스크 크기 보정 | 입력 높이가 `H`이면 기준 좌표에 `H / 1024`를 적용. 왼쪽 아래 위치 유지 |
| 표시 형식 | `2025-07-07 07:48:12` |
| 글자 | 흰색, 굵은 고정폭 글꼴 |
| 글자 높이 | 화면에 실제 보이는 글자의 높이 **20px**. FFmpeg의 명목 `fontsize=20`과 다름 |
| 글자 위치 | 왼쪽 10px, 글자 아래쪽에서 영상 바닥까지 10px |
| 새 글자 배경 | 불투명 검정색, 여백 7px. `max(3, 20/3)`을 정수 픽셀로 올림 |

원본 마스크 좌표만 영상 높이에 맞춰 보정한다. 새 글자의 20px 높이·10px 위치·7px 여백은 출력에서 고정된 픽셀 값이다. 글꼴의 실제 렌더링 높이를 측정해 폰트 크기를 정하므로 다른 글꼴로 바꾸어도 명목 폰트 크기만 20으로 지정하지 않는다.

지정한 마스크는 그 영역만 가린다. 원본 타임스탬프가 다른 위치에 있거나 더 넓으면 해당 영상에 맞는 스타일로 별도 계획을 만들어 확인한다.

## 결과 확인과 범위

- 결과는 타임스탬프가 들어간 **클립별 1fps 영상**이다. 하루 전체를 연결한 최종 영상은 아니다.
- 입력의 재생시간과 해상도를 유지한다. 1fps 프레임 경계 때문에 소수점 길이에는 반올림 차이가 있을 수 있다.
- fps·길이 검증은 처리 형식에 대한 검사다. 활동 내용, 같은 집 여부, 원본 타임스탬프가 모두 가려졌는지까지 시각적으로 인증하지는 않는다. 대표 입력의 처음·중간·끝과 다음 클립 전환을 확인한다.
- 입력 해상도가 서로 다르면 클립들을 그대로 `concat -c copy`로 합치지 않는다. 최종 영상의 크기를 통일할 계획이라면 **크기를 먼저 맞추고 타임스탬프를 넣는다**. 글자를 넣은 후 영상을 확대·축소하면 실제 글자 높이 20px이 바뀌므로 다시 생성해야 한다.
- 시간표의 공백은 관측 영상이 없는 시간으로 남는다. 합성 시각이 다음 클립에서 건너뛰는 것은 의도한 동작이다.
