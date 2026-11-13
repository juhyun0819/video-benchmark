# FIFO + Qwen3-VL

영상에서 최근 관측만 보존하는 독립 FIFO 베이스라인입니다. 별도 학습 없이
Qwen3-VL-8B-Instruct의 영상 특징을 저장하고, 용량을 넘으면 가장 오래된
**완전한 temporal grid**부터 삭제합니다. TAS/SDC 압축, 중요도 선택, proactive
trigger를 사용하지 않습니다.

메모리 정책과 실행기는 이 폴더에 있습니다. 입력 검증, 영상 샘플러, 질문
스케줄러와 native Qwen3 인코딩/응답 함수는 옆의 `fluxmem_qwen3vl`에서
재사용합니다. 그 베이스라인의 메모리나 실행 설정은 수정하지 않습니다.
이 디렉터리만 따로 복사하지 말고 저장소 전체 구조를 유지하세요. 한 Python
프로세스에 여러 baseline의 `streaming` 패키지를 함께 로드하지 마세요.

## 메모리 크기는 실행 시 결정

**기본 메모리 용량은 없습니다.** 설정 예제의 두 값은 `null`이며 실행할 때
아래 중 하나를 지정해야 합니다.

| 옵션 | 의미 |
| --- | --- |
| `--memory-frames N` | 샘플링된 원시 프레임 N장에 해당하는 최신 특징 유지. N은 2 이상의 짝수 |
| `--memory-tokens N` | 보관하는 visual token 행 수를 N 이하로 제한. 가장 오래된 grid 전체 삭제 |

Qwen3에서는 샘플링된 프레임 2장이 하나의 temporal grid입니다. 원본 영상의
전체 프레임 수와 `memory-frames`는 다릅니다. 토큰 예산이 grid 하나보다 작으면
명시적으로 오류를 내며, 남은 공간에 grid 일부를 잘라 넣지 않습니다.
두 옵션을 동시에 지정하면 오류입니다. JSON config의 예산 종류를 바꿀 때는
사용하지 않을 값을 `null`로 바꾸세요. 같은 설정 항목에 대해서는 CLI 값이
config보다 우선합니다.

용량 제한은 **ingest 중, 질문에 답하기 전** 적용됩니다. Qwen3 main embedding,
DeepStack 3개, 공간 좌표와 metadata를 한 묶음으로 보관하고 함께 제거합니다.
압축이나 특징 값 변경이 없고, 매 질문마다 영상을 다시 인코딩하지 않습니다.
순서, 크기, dtype/device가 일관된 stream을 받으며 중간 해상도/grid 변경은
명시적으로 거부합니다.

`memory-tokens`는 visual token 행 수이며 3개 DeepStack에 대해 네 배로 세는
단위가 아닙니다. 실제 특징 메모리 크기에는 네 tensor와 위치 좌표가 포함됩니다.
로그는 `total_tokens`와 `retained_tensor_bytes`를 둘 다 기록합니다. 이는 모델
가중치, 영상 디코딩, 임시 snapshot, 생성 KV cache, 누적 결과까지 포함한
프로세스 전체 메모리 제한은 아닙니다.

## 설치 및 입력 준비

Python 3.11의 별도 환경을 권장합니다. `requirements.inference.txt`는 비교 대상인
Qwen3 베이스라인과 같은 버전을 참조합니다. 실제 추론에는 호환 CUDA/Torch,
FFmpeg/FFprobe, 검증된 로컬 모델 가중치가 필요합니다.

```bash
cd baselines/fifo_qwen3vl
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.inference.txt
```

모델은 `Qwen/Qwen3-VL-8B-Instruct`, revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`를 사용합니다. 기존 Qwen3 실행기와 같은
`verification.json`을 요구합니다. 파일별 `path`, `bytes`, `hash`를 포함한
`verified_files`, `repo_id`, `revision`이 있어야 하며 실행기가 모델 파일을
검증합니다. 모델/데이터 자동 다운로드는 하지 않습니다.

`configs/sources.example.json`을 복사해 실제 영상의 절대경로와 Day 순서를
기입하고, 새 출력 경로로 manifest를 만드세요. 이 명령은 원본을 수정하지 않고
FFprobe와 SHA256으로 길이와 파일 상태를 확인합니다.

```bash
python -m fifo_videomem.prepare \
  --sources /absolute/path/sources.json \
  --output /absolute/path/manifest.json
```

질문 형식은 `configs/questions.example.json`을 참고하세요. 예제 질문은 스키마
설명용이며 평가용 정답이 아닙니다. 각 질문에는 `question_id`, `question`,
`choices`(자유 응답은 `[]`), 검토된 `query_time`이 필요합니다. 모델에 전달하는
질문에서 정답·근거 등 추가 필드는 제거됩니다. 정답 시각을 관측 종료 시점으로
자동 변환하지 않습니다.

## 실행

`configs/fifo.example.json`을 별도의 설정 파일로 복사해 입력/출력/모델 경로를
채우세요. `gpu`와 메모리 예산은 아직 지정되어 있지 않습니다. 예를 들어
셸 변수 `FIFO_GPU`와 `FIFO_MEMORY_FRAMES`를 실험에 맞게 정한 다음 실행합니다.

```bash
python -m fifo_videomem.run \
  --config /absolute/path/fifo.local.json \
  --gpu "${FIFO_GPU:?set the physical GPU index}" \
  --memory-frames "${FIFO_MEMORY_FRAMES:?set an even sampled-frame budget}"
```

토큰 기준 실험에서는 마지막 옵션 대신 다음을 사용합니다.

```bash
--memory-tokens "${FIFO_MEMORY_TOKENS:?set the visual-token budget}"
```

출력 폴더는 새 경로여야 하며 기존 실행을 덮어쓰지 않습니다. GPU는 지정한
physical index를 UUID로 확인해 선택하며, 이미 1 GiB 초과 메모리를 사용 중이면
실행을 거부합니다. 기존 작업을 중단하지 않습니다. 이 실행기는 과거 호스트의
GPU 4/5 제한을 상속하지 않으며 선택한 장치 한 개만 사용합니다.

질문의 관측 범위는 `[start, cutoff)`입니다. 두 번째 프레임이 cutoff와 같은
묶음은 해당 질문에 사용하지 않습니다. 단독으로 남은 프레임은 버리고, 파일
경계를 넘어서 두 프레임을 묶지 않습니다. Day/file 시각은 감사용으로 보존하고
모델 timestamp는 기존 Qwen3 bridge의 압축된 순차 타임라인을 사용합니다.
마지막 질문의 관측 시점에서 처리를 종료합니다. proactive 응답은 비활성입니다.

| 결과 파일 | 내용 |
| --- | --- |
| `provenance.json` | FIFO 정책, 선택한 예산/GPU/모델, 입력 및 코드 해시 |
| `manifest.json`, `public_questions.json` | 실행 입력과 정답 필드를 제거한 질문 |
| `events.jsonl` | 입력, 답변, 보관량과 삭제량 |
| `predictions.json` | 예측, 사용한 메모리, 답변 상태 |
| `status.json` | 진행/완료/오류 상태 및 메모리 통계 |
| `memory.pt` | 최종 FIFO bank와 provenance; 신뢰하는 파일만 로드할 것 |

EOS로 끝난 비어 있지 않은 답변만 완료로 표시합니다. 미완료 응답이 있으면
종료 코드는 2이며 정상 실행/오류 여부는 `status.json`과 함께 확인하세요.
Python API의 `state_dict()`와 `from_state_dict()`는 bank를 복원합니다.
CLI의 stream 위치·질문 상태까지 복원하는 resume 기능은 제공하지 않습니다.

## 검증 및 비교 범위

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 USE_FLAX=0 \
  PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q -p no:cacheprovider tests
```

CPU 테스트는 FIFO 순서, 답변 전 용량 제한, DeepStack 정렬, 입력/출력 복사,
checkpoint, 질문 cutoff, 공개 질문 필터링, CLI 설정 및 작은 무작위 native
Qwen3 모델 연결을 검증합니다. 실제 사전학습 모델의 정답률 검증과 구분하세요.

2026-11-10 검증: **83 passed**, skip 없음. 별도 테스트 환경에서 Python 3.12,
Torch 2.6.0, Torchvision 0.21.0, Transformers 4.57.3을 사용했습니다. 실제
FFmpeg 테스트 영상의 manifest 생성/PyAV 디코딩과 작은 무작위 Qwen3의
인코딩 → FIFO 삭제 → 답변 생성도 포함합니다. 이는 위에 기록한 실제 추론용
Torch 2.9 환경이나 사전학습 8B 모델의 GPU 실행을 검증했다는 의미는 아닙니다.

FluxMem과 비교할 때 모델, FPS, 해상도, 질문과 관측 시점을 맞추고
`total_tokens`/`retained_tensor_bytes`를 함께 보고하세요. FluxMem은 압축량이
달라 같은 frame 수가 같은 token/byte 예산을 보장하지 않습니다. 기존 FluxMem
session은 cascade 전에 새 grid 하나를 추가로 볼 수 있지만 이 FIFO는 답변
시점에도 엄격한 예산을 지킵니다. 비교에서 이 차이를 명시해야 합니다.

영상/QA, 모델 가중치와 실제 벤치마크 결과는 이 구현에 포함되지 않습니다.
