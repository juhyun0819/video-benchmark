# Reservoir sampling + Qwen3-VL

지금까지 관측한 영상 특징에서 균등 무작위 표본을 보관하는 별도 베이스라인입니다.
Qwen3-VL-8B-Instruct를 사용하며, 샘플링 단위는 **원시 샘플 프레임 2장으로
만든 완전한 temporal grid**입니다. TAS/SDC 압축, 중요도 선택, 별도 학습 및
proactive 응답은 사용하지 않습니다.

`reservoir_videomem/memory.py`와 실행기는 독립적입니다. 영상 입력 검증, 샘플러,
질문 스케줄러, native Qwen3 인코딩/응답 함수는 옆의 `fluxmem_qwen3vl`에서
재사용합니다. 저장소 전체 구조를 유지하고, 각 baseline은 별도 Python
프로세스에서 실행하세요. 기존 FIFO/FluxMem의 메모리 코드를 가져오지 않습니다.

## 선택 규칙

Algorithm R을 사용합니다. 보관 가능한 grid 수를 K, 지금 도착한 grid까지의
누적 관측 수를 n이라고 하면 다음과 같이 동작합니다.

1. 첫 K개는 모두 저장합니다.
2. 이후에는 정수 `j`를 `[0, n)`에서 균등하게 뽑습니다.
3. `j < K`이면 해당 저장 슬롯을 새 grid로 교체하고, 아니면 새 grid를 버립니다.

n ≥ K일 때 각 관측 grid가 남아 있을 확률은 K/n입니다. 이는 **샘플링되어
완전한 쌍을 이룬 grid**에 대한 확률이며, 원본 영상의 모든 프레임이나 사건을
균등하게 뽑는다는 의미는 아닙니다. 최신 grid가 버려질 수도 있습니다.

내부 슬롯 순서는 보존하고, 답변에 사용할 snapshot만 시간순으로 정렬합니다.
Main embedding, DeepStack 3개, 좌표와 metadata는 함께 선택/제거됩니다.
관측 누적 수와 마지막 관측 시점은 새 grid를 버린 경우에도 증가합니다.
선택은 질문 응답 **전에** 끝나므로 답변 시점에도 용량을 넘지 않습니다.

## 용량과 seed는 나중에 지정

**메모리 용량과 seed의 기본값은 없습니다.** 예제 config는 둘 다 미설정이며,
실행할 때 명시적으로 정해야 합니다.

| 옵션 | 의미 |
| --- | --- |
| `--memory-frames N` | 샘플 프레임 N장에 해당하는 특징을 보관. 2 이상의 짝수, K=N/2 |
| `--memory-tokens N` | visual token 행 예산. 한 grid당 T행이면 K=floor(N/T) |
| `--seed S` | 0 이상의 정수. reservoir 전용 난수 발생기에 사용 |

프레임/토큰 예산은 정확히 하나만 지정합니다. 토큰 예산이 grid 하나보다 작으면
오류로 처리하며, grid 일부를 잘라 넣지 않습니다. 한 stream의 공간 grid 크기,
특징 차원, dtype/device는 고정되어야 하며 변경 시 거부합니다. 이 조건으로
토큰 모드에서도 K가 일정하여 Algorithm R의 균등성을 유지합니다.

전용 `random.Random(seed)`를 사용하므로 전역 Python/Torch 난수나 질문 횟수가
선택에 영향을 주지 않습니다. 같은 입력 순서·seed·런타임에서는 같은 선택을
재현합니다. 실험 provenance에 seed와 Python 버전을 기록합니다. 여러 seed의
정답률 변동까지 비교하는 실험은 이후에 별도로 구성하세요.

`memory-tokens`는 main visual token 행 수를 세며, DeepStack을 포함한 실제
저장량은 `retained_tensor_bytes`로 따로 기록합니다. 모델 가중치, 영상 디코딩,
snapshot 복사, 생성 KV cache와 누적 출력은 이 제한에 포함되지 않습니다.

## 준비 및 실행

별도 Python 3.11 환경을 권장합니다. 설치 버전은 기존 Qwen3 비교 환경과
같으며, 실제 실행에는 호환 CUDA/Torch와 FFmpeg/FFprobe가 필요합니다.

```bash
cd baselines/reservoir_qwen3vl
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.inference.txt
```

모델은 로컬 `Qwen/Qwen3-VL-8B-Instruct`, revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`를 사용합니다. 기존 Qwen3 실행기와
같은 `verification.json`의 파일별 해시 검증을 적용합니다. 검증 기록에는
`repo_id`, `revision`, `verified_files`의 `path`, `bytes`, `hash`가 필요합니다.
모델/데이터는 자동 다운로드하지 않습니다.

`configs/sources.example.json`을 복사해 실제 영상의 절대경로와 Day 순서를
기입하고 manifest를 만듭니다. 원본 파일을 수정하지 않고 길이/해시를 확인합니다.
FIFO용으로 이미 준비한 같은 manifest와 질문 파일을 사용할 수도 있습니다.

```bash
python -m reservoir_videomem.prepare \
  --sources /absolute/path/sources.json \
  --output /absolute/path/manifest.json
```

질문 형식은 `configs/questions.example.json`을 참고하세요. 각 질문에는
`question_id`, `question`, `choices`(자유 응답은 `[]`), 검토된 `query_time`이
필요합니다. 예제는 스키마 설명용입니다. 정답과 근거 등 추가 필드는 모델
질문에서 제거하고, 관측 종료 시점을 정답 시각에서 추정하지 않습니다.

`configs/reservoir.example.json`을 복사해 입력/출력/모델 경로를 채우세요.
`gpu`, `seed`와 한 가지 메모리 예산을 실험에 맞게 정한 뒤 실행합니다.

```bash
python -m reservoir_videomem.run \
  --config /absolute/path/reservoir.local.json \
  --gpu "${RESERVOIR_GPU:?set the physical GPU index}" \
  --seed "${RESERVOIR_SEED:?set a nonnegative integer seed}" \
  --memory-frames "${RESERVOIR_MEMORY_FRAMES:?set an even sampled-frame budget}"
```

토큰 모드에서는 마지막 옵션을
`--memory-tokens "${RESERVOIR_MEMORY_TOKENS:?set the visual-token budget}"`로
바꾸세요. CLI는 같은 config 필드보다 우선합니다. 예산 종류를 바꿀 때는
config에서 사용하지 않을 예산을 `null`로 설정해야 합니다.

출력 폴더는 새 경로여야 합니다. 지정한 physical GPU를 UUID로 확인하고, 이미
1 GiB 초과 메모리를 사용 중이면 실행을 거부합니다. 과거 호스트 GPU 4/5 제한은
상속하지 않으며 기존 프로세스를 중단하지 않습니다.

관측 범위는 `[start, cutoff)`이고 두 번째 프레임이 cutoff와 같은 grid는 해당
질문에서 제외합니다. 단독 프레임과 파일 경계를 걸치는 쌍은 사용하지 않습니다.
마지막 질문의 관측 시점에서 처리를 마치며, 같은 시점의 여러 질문은 동일한
reservoir를 사용합니다. 답변에는 재인코딩이나 재샘플링을 하지 않습니다.

## 결과와 checkpoint

| 결과 파일 | 내용 |
| --- | --- |
| `provenance.json` | Algorithm R, 용량/seed/모델/GPU, 코드와 입력 해시 |
| `manifest.json`, `public_questions.json` | 검증한 영상 매핑과 공개 질문 필드 |
| `events.jsonl` | 선택/답변 이벤트, 보관/교체/신규 거절량 |
| `predictions.json` | 예측, 선택된 메모리 ID, 답변 상태 |
| `status.json` | 실행 상태, 메모리 통계와 완료/오류 요약 |
| `memory.pt` | 최종 reservoir bank, RNG 상태와 provenance |

`evicted_grids`는 기존 저장 grid의 교체 횟수, `rejected_grids`는 새 관측을
저장하지 않은 횟수입니다. `discarded_grids`는 둘의 합입니다. 둘 중 하나라도
발생하면 snapshot의 `memory_truncated`가 참이어서 일부 관측이 빠졌음을
응답 기록에 남깁니다.

`state_dict()`/`from_state_dict()`는 내부 슬롯 순서, 누적 관측 수, 마지막
관측 시점, RNG 상태를 보존합니다. 마지막 입력을 거절한 경우에도 이후 선택을
정확히 이어갈 수 있습니다. 신뢰하는 로컬 checkpoint만 로드하세요. CLI에는
영상 커서/남은 질문까지 복원하는 resume 기능이 없습니다.

EOS로 끝난 비어 있지 않은 답변만 완료입니다. 미완료 답변이 있으면 종료 코드
2이며, `status.json`과 함께 확인하세요. 구현 테스트는 사전학습 모델의 정답률
검증과 구분됩니다.

## 테스트

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 USE_FLAX=0 \
  PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q -p no:cacheprovider tests
```

작은 경우의 모든 난수 선택을 열거해 균등성을 검증하고, seed 재현성,
거절/교체 분기, 엄격한 용량 제한, RNG checkpoint 연속성, 미래 프레임 제외,
DeepStack 정렬 및 작은 무작위 Qwen3 연결을 검사합니다. 실제 영상/QA나
사전학습 모델 가중치는 테스트에 필요하지 않습니다.

2026-11-10 검증: **118 passed**, skip 없음. 별도 CPU 테스트 환경에서
Python 3.12, Torch 2.6.0, Torchvision 0.21.0, Transformers 4.57.3을 사용했습니다.
Algorithm R의 N=4, K=1/2/3 경우에 가능한 모든 선택 경로를 열거해 각 부분집합의
빈도가 같은지 검사했습니다. 실제 FFmpeg 테스트 영상 준비/PyAV 디코딩과 작은
무작위 native Qwen3의 인코딩 → reservoir 선택 → 답변 생성도 포함합니다.
사전학습 8B 모델의 GPU 실행이나 실제 데이터 정답률은 아직 검증하지 않았습니다.

FIFO와 비교할 때 같은 모델·FPS·해상도·질문·관측 시점·메모리 예산을 사용하세요.
압축하는 FluxMem과의 비교에서는 frame 수뿐 아니라 보관 token 수와 byte 수도
함께 기록해야 합니다. 기존 FluxMem의 답변 전 일시적 추가 grid 허용과 달리
이 baseline은 답변 시점에도 엄격한 예산을 지킵니다.
