# Long-term Egocentric Video Memory Benchmark — MVP

로컬 영상에서 사람이 구간을 확인하고, 가상 Persona의 반복 일상을 구성하여 MP4·Manifest·QA를 생성하는 Python 3.10+ 프로젝트입니다. 원본 영상은 수정하지 않으며 데이터셋 다운로드나 모델/API 호출을 하지 않습니다.

## 실행

```bash
cd benchmark
# macOS: brew install ffmpeg
# Linux: ffmpeg와 ffprobe를 PATH에 설치
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

1. **Clip Browser**에서 데이터셋과 로컬 폴더를 선택하고 Scan을 누릅니다. 파일 정보와 재생 화면을 확인하여 초 단위 시작·끝을 지정합니다.
2. 활동, 실제 참가자, 장소, 원본 촬영 시각(알 때만), 검증된 속성을 입력합니다. 확인 체크와 검증자/annotation 참조를 함께 기록하면 QA에 포함됩니다. 등록 시 중간 프레임 썸네일을 생성합니다. JSON·CSV 다운로드와 선택 순서 재생을 지원합니다.
3. **Persona & Routine**에서 Template 행의 활동·가상 시각을 수정/추가/삭제하고 D·N·seed를 지정합니다. 기본은 7일 × 5개 활동이며 N은 각 날짜의 각 슬롯을 반복하는 횟수입니다. 날짜별 이벤트와 클립을 확인하고 변경하거나 제거합니다.
4. **Generate & Validate**에서 Manifest 전용 또는 MP4 렌더링을 선택합니다. 새 출력 디렉터리에 생성하며 이전 결과를 덮어쓰지 않습니다.

```text
outputs/<persona_id>/<run_id>/
  day_01.mp4 ... day_07.mp4  # 렌더링 모드
  full_stream.mp4
  parts/                   # 정규화된 이벤트별 영상
  manifest.json
  qa.json
  validation.json
```

렌더링 오류는 UI에 표시됩니다. 부분 산출물은 진단을 위해 남으며 완성 Manifest는 최종 성공 후에만 저장합니다. 큰 영상은 메모리에 로드하지 않고 FFmpeg로 순차 처리하지만 전체 렌더링에는 임시 parts만큼 추가 디스크 공간이 필요합니다.

## 데이터 없는 재현 테스트

```bash
python demo.py --render
python demo.py --output outputs/test_manifest_only
python -m pytest -q
```

`datasets/TEST_ONLY`에 FFmpeg test pattern 10개를 생성합니다. **사람/행동을 촬영한 데이터가 아니며 실제 Egocentric Dataset이나 연구 평가 데이터로 사용하면 안 됩니다.** 테스트에서 활동 이름은 파이프라인 검증을 위한 스크립트 레이블일 뿐입니다. 모든 클립에 `is_test_data=true`와 TEST ONLY 출처를 기록합니다. 고정 seed로 35개 이벤트, 7일, 총 35초의 압축 스트림을 생성합니다. 동일 output으로 재실행하려면 다른 `--output`을 지정하세요.

## 모듈과 저장 형식

- `dataset_adapters`: EgoLife, HD-EPIC, EPIC-KITCHENS-100, Ego4D, Ego-Exo4D용 로컬 재귀 검색 fallback. 다양한 폴더 구조를 탐색하며, 참가자/촬영 시각을 파일명에서 추측하지 않습니다. 릴리스별 annotation 파서는 아직 없으며 `LocalAdapter.metadata/discover` 재정의 후 `register`로 확장합니다.
- `clip_manager`: JSON ClipStore, CSV export, 구간 검증, OpenCV 썸네일. `metadata/clips.json`에 원본 절대 경로·구간·출처·속성·검증 근거를 저장합니다.
- `persona_builder`, `routine_generator`: 항상 Synthetic Persona로 표시. 독립 `build`/`change` API, 고정 seed, Stable Routine/Episodic Deviation/Activity Replacement 지원. UI 이벤트 제거와 템플릿 편집 지원.
- `stream_generator`: `manifest`/`generate` API. 해상도·FPS·SAR 통일, 오디오 제거 또는 AAC stereo 통일(없는 경우 무음), 날짜별 및 전체 concat. FPS에 맞춰 이벤트 길이를 프레임 수로 양자화하며 전체 렌더링 길이를 재검증합니다.
- `qa_generator`: 검증된 활동 레이블 기반 Episodic Recall, Frequency, Temporal Order, Cross-day Comparison. 근거는 **event ID와 clip ID 둘 다**, 압축 스트림 시각 구간으로 연결합니다. 같은 원본 재사용도 서로 다른 이벤트로 구분합니다. `QAGenerator` Protocol이 향후 LLM 플러그인 경계입니다.
- `validation`: 파일/구간/가상 시각 순서/중복/참가자 혼합/QA 일치 검사. 자동 판단 불가 항목은 `manual_review_required`에 기록합니다.

`metadata/plan.json`은 일별 순서와 clip ID를 참조하고, `manifest.json`은 각 이벤트에 클립 메타데이터 전체와 `stream_start/end`, `day_start/end`, 프레임 수를 복사합니다. `source_recorded_at`은 실제 원본 시각, `synthetic_day/synthetic_time`은 가상의 시각입니다. `08:30`은 활동 시작 레이블이며 실제 스트림의 8시간 30분을 의미하지 않습니다. 일별 빈 시간이나 수면 시간은 영상으로 채우지 않습니다. N 반복은 같은 가상 슬롯에 속하며 연속된 실제 촬영이라고 주장하지 않습니다.

## Ground Truth 경계와 미구현 사항

사람의 확인 또는 annotation provenance가 없는 레이블은 QA에서 제외합니다. 빈도는 전체 이벤트가 확인된 경우에만 생성하며 **선택된 영상의 발생 횟수**를 묻습니다. Cross-day Comparison은 확인된 활동 레이블만 비교하며 동일 객체나 실제 동일인임을 추론하지 않습니다. 검증 체크의 진실성 자체와 참가자/공간/시점/실제 촬영 순서의 연속성은 사람이 검토해야 합니다.

Activity Omission, Temporal Order Change, Cross-day State Change, Gradual Drift는 이름/확장 지점만 예약되어 있습니다. `change`로 요청하면 명시적으로 미지원 오류를 냅니다. 원본 annotation 자동 ingest, 객체 동일성 검증, 자동 행동 인식, V-JEPA 2, LLM QA, 다운로드, 다중 사용자 동시 편집, 렌더 취소/재개는 미구현입니다. JSON 저장소는 단일 사용자 로컬 작업용입니다.

실제 영상에서 활동당 여러 클립을 등록해야 다른 날의 다양한 장면을 얻습니다. 한 클립뿐이면 재사용 가능하며 validator가 경고합니다. Manifest 전용 모드도 FFprobe로 로컬 파일을 검증하므로 원본 접근이 필요합니다. 로컬 경로를 공유 환경에 배포할 때는 인증과 파일 접근 정책을 별도로 구성하세요.

## 이번 구현의 검증 결과

2026-10-26 Linux / Python 3.11 환경에서 `pytest`: **6 passed**. 7일 MP4 렌더와 길이, JSON/CSV/썸네일, 손상 파일, 잘못된 구간/경로/레이블, 미검증 QA 제외, QA 변조 탐지, 오디오 없는 소스의 무음 트랙 생성, Streamlit AppTest 초기 실행을 확인했습니다. 별도 데모로 35개 이벤트·7개 일별 MP4·전체 35초 MP4·98개 QA 및 Manifest 전용 출력을 생성했습니다. 실제 데이터셋 영상과 macOS에서의 실행, 브라우저의 실제 영상 재생은 이 환경에서 검증하지 않았습니다. `requirements-lock.txt`는 이 검증 환경의 패키지 버전 기록이며 macOS에서는 기본 `requirements.txt`를 사용하세요.

## 실제 TeleEgo P4 Day 1

`teleego.py`는 공개 `H-oliday/TeleEgo-Source`의 고정 revision을 `config/teleego_p4.json`에 보존합니다. 선택 영상 9개만 다운로드하며 공식 narration SRT를 별도 세부 사건으로 저장합니다. 합계 원본 용량은 72,316,650,245 bytes(약 72.3GB)입니다.

```bash
python teleego.py --download annotations  # 작은 SRT와 추정 Manifest만
python teleego.py --download all --render # 원본 9개, SHA256 검증, 길이 측정, Day 1 렌더
```

다운로드는 `.partial`에 기록하고 재실행 시 이어받습니다. 완성된 원본은 파일 크기로 건너뛰며 최초 다운로드는 저장소 LFS SHA256도 검증합니다. 영상 측정 전에는 `media_verified=false`, `duration_status=provisional SRT estimate`로 표시합니다. 원본 다운로드 후 FFprobe 길이로 구간을 다시 계산하며 자막 구간의 영상 범위 초과를 보고합니다. 모든 원본 날짜와 narration 텍스트를 보존하고 파일명에서 전체 영상의 활동 정답을 추론하지 않습니다.

결과는 `outputs/teleego_p4_day1`의 `source_manifest.json`, `annotations.json`, `validation.json`, `annotation_qa.json`, `day_01.mp4`입니다. QA는 원본 중국어 narration을 그대로 답으로 사용하는 추출형이며 독립 시각 검토 전 상태를 명시합니다. 영상 미확보 상태에서는 QA를 생성하지 않습니다. Streamlit 하단에서 소스별 주석과 원본 구간을 확인할 수 있습니다.

전체 렌더는 640×360/24fps/무음으로 정규화하고 원본 길이와 최종 길이를 검사합니다. 원본 오디오는 원본 MP4에 보존됩니다. 자막의 실제 시각과 합성 시간은 충돌할 수 있으므로 실제 08:00부터 녹화된 하루로 취급하지 않습니다. 동일 revision/설정의 완료된 정규화 파일은 재실행 시 재사용합니다. 설정을 바꿀 때는 별도 출력 경로를 사용해야 합니다.
