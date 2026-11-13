# Routine-Aware Online Memory: 실험용 구현

이미 추출한 visual feature를 시간순으로 읽어 최근 관측, 반복 패턴,
개별 증거를 보존하는 학습 없는 메모리 구현이다. 검증된 의미 기반 활동
인식기나 확정된 QA baseline 성능으로 해석하면 안 된다.

총 보존 예산은 기본 **16,000,000,000 bytes**다. Short-term, Routine,
Event 메모리와 후보 prototype, 인덱스, 제어 상태를 모두 합친 상한이며
각 메모리에 16GB를 별도로 배정하지 않는다. 실제 보존 상태는 CPU RAM에서
증가한다. 인코더·LLM 가중치, 입력 shard, 임시 계산 버퍼와 프로세스 RSS는
보존 예산과 별도로 보고한다.

이 디렉터리에는 코드, 설정과 합성 테스트만 포함된다. 실제 영상, QA,
추출 feature, 체크포인트, 예측 결과는 사용자가 별도로 준비한다.

## 환경과 합성 예제

아래 명령은 이 저장소의 `integrations/videomem`에서 실행한다.
Python 환경은 [feature-memory 안내](../feature_memory/README.md)를 따른다.
추가 학습 모델은 필요하지 않으며 출력 경로는 새 디렉터리여야 한다.

```bash
cd integrations/videomem
PY=/path/to/feature-env/bin/python

CUDA_VISIBLE_DEVICES='' "$PY" -m baselines.routine_memory synthetic \
  --output /path/to/new/routine-synthetic
```

합성 예제는 중간에 다른 패턴이 나타나는 세 번의 독립 방문으로 routine
승격을 확인한다. 한 활동에서 연속으로 들어온 동일한 프레임만으로 독립
발생 횟수를 늘리지 않는다. 합성 정답 라벨은 matcher에 전달되지 않는다.

## 입력 형식과 호환 범위

| 경로 | 현재 허용하는 입력 |
| --- | --- |
| Python `RoutineAwareMemory.update` | 한 스트림 안에서 고정된 `[tokens, hidden]` CPU fp16/bf16/fp32 tensor, 증가하는 frame ID와 timestamp |
| `python -m baselines.routine_memory build` | 기존 단일 Qwen2.5-VL archive의 `[676,3584]` bf16 feature 및 지정된 모델·plan·inventory provenance |
| `python -m baselines.routine_memory.online_qa` | 별도 JAKE archive의 `[2500,3584]` bf16, `50×50` grid와 canonical absolute-time QA 계약 |

새 [영상별 토큰 추출 도구](../../tools/feature_memory/VISUAL_TOKENS.md)는
해상도마다 token 수가 다른 영상별 archive를 만든다. 그 최상위 디렉터리를
`build` 또는 `online_qa`에 바로 넣는 연결은 아직 구현하지 않았다. 동일한
bf16 feature라는 이유만으로 archive header, 시간축, grid를 호환된 것으로
취급하면 안 된다. 영상별 feature를 사용하는 새 실험은 입력 adapter에서
원본 ID·시간축·grid를 명시적으로 연결해야 한다.

## 호환되는 기존 archive에서 구축

```bash
ARCHIVE=/path/to/compatible-qwen25-feature-archive
MODEL=/path/to/Qwen2.5-VL-7B-Instruct

CUDA_VISIBLE_DEVICES='' "$PY" -m baselines.routine_memory build \
  --archive "$ARCHIVE" --model "$MODEL" \
  --config baselines/routine_memory/configs/default_16gb.json \
  --max-frames 300 --snapshot-seconds 60 300 \
  --output /path/to/new/routine-smoke
```

기본 실행 범위는 300프레임 또는 더 짧은 archive 길이다. 전체 입력은
명시적인 `--full`이 필요하다. feature를 다시 인코딩하거나 영상을 다시
디코딩하지 않는다. `--model`은 가중치를 로드하는 대신 로컬 설정·processor
파일과 archive provenance를 교차 확인한다. shard와 프레임 체크섬은
기존 archive reader가 검증한다.

입력은 프레임마다 독립 이미지로 인코딩한 1fps post-merger feature다.
두 프레임을 묶는 native video 입력이나 decoder KV와 같지 않다.
입력 버퍼에 나중 프레임이 있어도 현재 순서의 update 이전에는 메모리에
넣지 않으며 snapshot cutoff 이후의 관측은 해당 snapshot에 포함하지 않는다.

## 메모리와 판정

| 구성 | 보존하는 내용 |
| --- | --- |
| Short-term | 최근 최대 16프레임의 선택 원본 token, token index, 시간, 공간·전역 특징 |
| Routine | 공간·시간 prototype, 독립 발생 횟수, 제한된 발생 구간과 대표 증거 |
| Event | 개별 증거, 발생 구간, 관련 routine/candidate ID, 선택적인 짧은 설명 |
| Candidate | Event 메모리 내부의 반복 후보와 통계; 별도의 무제한 bank가 아님 |

기본 선택 증거는 프레임당 64개 원본 token이다. 공간 prototype은 token
순서를 16개 구간으로 나눠 평균한 값이며 의미 기반 segmentation이 아니다.
시간 특징은 최근 관측을 순서가 있는 4개 구간으로 표현한다. prototype은
fp32, 선택 원본 feature는 입력 dtype으로 보존한다.

완전 일치(A)는 routine을 갱신하고, 부분 일치(B)는 공유 부분을 갱신하면서
차이가 나는 증거를 남긴다. 충분히 일치하는 routine이 없으면(C) event와
candidate를 관리한다. 승격에는 독립 발생 3회와 평균 안정도 0.88 이상이
필요하다. 독립 발생에는 다른 활동으로의 전환과 최소 60초의 간격이
필요하므로 오래 지속된 한 방문을 여러 방문으로 계산하지 않는다.

현재 C 후보에는 **고정된 event 증거와의 원본 token 비교**가 추가되어 있다.
후보의 이동평균 prototype과 별개로, 기존 증거의 원래 token index에서
현재 feature를 비교한다. 기존 pooled 안정성 조건 아래 3회 연속으로
증거 유사도가 `event_anchor_similarity=.80`보다 낮으면 새 증거를 저장한다.
이 추가 저장은 독립 발생 횟수를 늘리지 않는다. 증거는 승격 후에도 관련
routine과 함께 보존되며, 후보 판정·승격 threshold는 변경하지 않았다.

이 기준은 의미적인 event 경계로 보정되지 않았다. 카메라 움직임만으로도
거의 3프레임마다 추가 증거가 생길 수 있다. 추가된 event 수를 서로 다른
활동 수로 해석하면 안 되며, 장시간 처리 전에 저장 증가율과 처리 비용을
측정해야 한다. A/B 경로의 추가 coverage 정책은 이 수정에 포함되지 않는다.

전체 설정은 [config.py](config.py)의 `RoutineConfig`에 있고, JSON에서
생략한 항목은 코드 기본값을 쓴다. `--budget-bytes`는 고정 byte 상한을
설정한다. 예전 실험의 `--budget-ratio 0.05/0.10/0.25/0.50`은 **전체 archive
footprint**에서 절대 byte 상한을 한 번 계산하며, snapshot별 비율이 아니다.
큰 비율 예산을 설정한다고 RAM이 자동으로 확보되지는 않는다.

## 예산·체크포인트·검색

보존 byte 계산에는 tensor backing storage, 공유 storage 중복 제거,
metadata/index 직렬화와 보수적인 Python 객체 reserve가 포함된다.
입력 배치와 임시 복사 때문에 RSS는 보존 예산보다 클 수 있다.
`effective_retention_ratio`는 누적 입력 feature bytes 대비 보존 bytes이고,
`budget_utilization`은 고정 예산 대비 보존 bytes다.

예산 압력이 생기면 routine 병합, 중복 증거 감소, 낮은 우선순위 event
제거 등을 결정적인 순서로 수행하고 각 동작을 기록한다. snapshot과
audit log를 추가 검색 메모리로 읽지는 않는다. 체크포인트는 raw tensor와
JSON·SHA256 manifest를 사용하며 pickle을 사용하지 않는다.

```bash
CUDA_VISIBLE_DEVICES='' "$PY" -m baselines.routine_memory build \
  --archive "$ARCHIVE" --max-frames 600 \
  --resume-checkpoint /path/to/routine-smoke/checkpoint_final \
  --output /path/to/new/routine-resumed

CUDA_VISIBLE_DEVICES='' "$PY" -m baselines.routine_memory query \
  --checkpoint /path/to/routine-smoke/checkpoint_final \
  --archive "$ARCHIVE" --frame-id 0 --top-k 5 \
  --output /path/to/new/routine-query
```

재개 시 설정과 입력 identity를 검증하고 이미 처리한 프레임은 다시
update하지 않는다. 읽기 과정에서는 prefix shard를 다시 검사할 수 있다.
이전 checkpoint에 coverage 필드가 없으면 기본 필드를 추가하고 byte
상한을 다시 확인한다. 과거에 버린 증거를 복원하지는 않는다.

검색은 보존된 visual evidence의 pooled cosine으로 순위를 매긴다.
질문에 대한 정답을 생성하는 기능은 아니다. `--features`로 별도의
safetensors query를 줄 수도 있다. snapshot 생성보다 이른 `--as-of`는
거부하며, 과거 질문에는 그 시점의 snapshot을 사용해야 한다.

## 별도 온라인 QA 시험

아래 경로는 위 표의 JAKE 전용 입력 계약을 만족할 때만 사용한다. GPU는
사용 가능한 번호를 **명시**한다. 이 handoff의 기본값은 GPU 0 한 개다.
개발 서버의 GPU 2–7 기본 목록만 변경했으며 메모리 알고리즘은 유지했다.

```bash
"$PY" -m baselines.routine_memory.online_qa \
  --archive /path/to/compatible-jake-archive \
  --qa /path/to/canonical-visual-qa.json \
  --model "$MODEL" --output /path/to/new/online-smoke \
  --gpus 0 --max-questions 3 --budget-bytes 16000000000 \
  --qa-read-tokens 8192 --archive-readers 4 --archive-buffer-shards 6 \
  --cpu-threads 8
```

canonical QA의 `query_time`은 이미 DAY offset을 포함한 절대 시간이다.
질문 시점까지의 메모리만 고정해 readout을 만들고, 여러 GPU는 그 readout에
대해 독립적인 frozen Qwen 모델로 답한다. 메모리는 하나이며 날짜마다
초기화하지 않는다. 각 GPU의 LLM 가중치와 진행 중 QA 버퍼는 보존 예산과
별도인 실행 비용이다.

readout은 원본 token의 sparse 위치를 Qwen mRoPE에 연결하는 실험용
adapter다. native 전체 영상 입력과의 동등성이나 학습된 semantic retrieval을
주장하지 않는다. 정답은 평가기에만 있고 QA worker에는 질문·선택지·시각만
전달된다. audio가 필요한 질문은 제외·집계하며 생성 실패도 분모에 남긴다.

## 출력과 검증

구축 실행은 `run_config.json`, `decisions.jsonl`, `metrics.jsonl`,
`status.json`, `summary.json`, snapshot과 `checkpoint_final/`을 만든다.
별도 QA 실행은 문항별 prediction·ground truth·정답 여부, 시점별 routine,
event, 승격 횟수와 메모리별 bytes를 기록한다. 이 결과물은 Git에 넣지 않는다.

```bash
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  "$PY" -m pytest -q baselines/routine_memory/tests
```

합성 테스트는 독립 반복의 승격, 서로 다른 관측의 증거 보존, 작은 편차,
짧은 잡음, causal 순서, 고정 byte 상한, 체크포인트 재개와 손상 검출,
원본 token 소유권, QA에 정답이 전달되지 않는 계약을 확인한다. 이전 구현의
개발용 결과 표는 이 공개 handoff에 포함하지 않았다. 테스트 통과는 장시간
QA 정확도나 개선을 입증하는 결과가 아니다.

이 handoff의 `integrations/videomem` 경로에서 위 테스트 63개가 통과했다.
