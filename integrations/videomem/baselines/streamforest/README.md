# StreamForest — local causal video-memory adapter

공식 StreamForest의 Qwen2-7B 모델, SigLIP vision tower, projector와 memory manager를
사용하는 별도 베이스라인입니다. 영상 프레임을 순차적으로 인코딩하고, 검토된 관측
종료 시점까지의 메모리로 질문에 답합니다. 메모리 용량과 GPU는 실행할 때 지정합니다.

## 소스와 모델

| 항목 | 고정 식별자 |
| --- | --- |
| 공식 소스 | [MCG-NJU/StreamForest](https://github.com/MCG-NJU/StreamForest), commit `f9d14670ff451ff5865638e17c9958e99bf2fec5` |
| 공식 모델 | [MCG-NJU/StreamForest-Qwen2-7B](https://huggingface.co/MCG-NJU/StreamForest-Qwen2-7B), revision `6840f91e17346b4c406ba03b57d1a63582765c91` |
| 소스 검증 목록 | `vendor/streamforest/UPSTREAM.json` |
| upstream 라이선스 | `vendor/streamforest/LICENSE` |

원본 README의 데모 이미지는 [고정 revision의 공식 README](https://github.com/MCG-NJU/StreamForest/blob/f9d14670ff451ff5865638e17c9958e99bf2fec5/README.md)에서 볼 수 있습니다.

`vendor/streamforest`의 보존된 upstream 파일은 수정하지 않습니다. 로컬 어댑터는
학습용 package initializer와 사용하지 않는 vision encoder import를 건너뛰고,
SigLIP 경로만 연결합니다. 이 inference 경로에 FlashAttention, TensorFlow 또는
학습 도구 설치는 필요하지 않습니다.

공식 checkpoint에는 vision tower의 가중치 tensor 항목 421개도 포함되어 있습니다.
별도의 SigLIP 가중치를 다운로드하지 않고 이 checkpoint의 vision 가중치를 사용합니다.
실제 모델 실행에는 전체 공식 checkpoint와 tokenizer/config 파일이 로컬에 있어야 합니다.
실행기는 모델이나 영상을 자동 다운로드하지 않습니다.

## 메모리 업데이트와 실험 범위

프레임 하나에서 추출한 **raw vision features**를 공식 online 경로처럼 처리합니다.
각 프레임 도착마다 `MemoryManager.update()`와 `get_memory_tokens()`를 실행하여
now → short → long tier의 전이와 event 병합을 완료합니다. 메모리는 질문 사이에도
유지됩니다. 질문을 받을 때는 저장된 tensor를 읽기만 하며, 질문 횟수가 메모리
업데이트나 압축 횟수를 바꾸지 않습니다.

일반 batch projector 경로는 위치 정보를 더하고 전체 입력 뒤에 메모리를 한 번
가져오는 등 실행 순서가 다릅니다. 이 어댑터는 single-frame online 경로를 기준으로
구현했으며, batch projector를 호출한 결과와 같다고 가정하지 않습니다.

질문은 `qwen_1_5` 대화 템플릿의 image + question 형식을 사용합니다. 압축된 event에
복원할 수 없는 개별 프레임 시각이나 가짜 per-token 시간 레이블을 붙이지 않습니다.
원본 파일 PTS와 마지막 관측 시점은 모델 프롬프트와 별도의 audit metadata로 보존합니다.

이 코드는 서로 다른 질문을 지정된 시점에 처리하는 로컬 streaming adapter입니다.
공식 `short_online_v2` 평가 프로토콜의 정확한 재현이나 논문 정답률을 주장하지 않습니다.
별도 학습, proactive 응답, 음성 이해 및 benchmark 채점은 포함하지 않습니다.

## 메모리 크기는 나중에 지정

**기본 메모리 설정은 없습니다.** 실행 config의 `memory_config`는 `null`이며,
실행 전에 JSON 경로를 채우거나 `--memory-config`로 명시해야 합니다.
다음 두 파일은 upstream 설정을 기록한 선택용 예제입니다. 자동 적용되지 않습니다.

| 예제 | now/short windows | now/short tokens | event window | long tokens/frame | long quota |
| --- | --- | --- | --- | --- | --- |
| `configs/memory.official_manager.example.json` | 1 / 18 | 729 / 128 | 8 | 64 | 5,120 |
| `configs/memory.official_online.example.json` | 1 / 6 | 64 / 16 | 4 | 16 | 800 |

각 파일에는 `sim_weight_g=0.4`, `time_weight_a=0.2`, `merge_weight_b=0.4`도 명시되어
있습니다. 첫 파일은 공식 manager의 기본 인자, 두 번째는 공식 online 실행 경로의
인자를 기록합니다. 실험에 맞게 별도 파일로 복사하고 값을 정하면 됩니다.

`long_memory_tokens_quota`는 **long tier만의 제한**입니다. 전체 visual token 수에는
now tier, short tier와 event 분할 대기분도 포함됩니다. 완료된 업데이트 이후의
보수적인 visual token 상한은 다음과 같습니다.

```text
now_window × now_tokens
+ (short_window + event_split_window - 1) × short_tokens
+ long_memory_tokens_quota
```

실행 결과의 `total_tokens`, `tier_tokens`, `visual_token_upper_bound`와 실제 tensor
저장량 `retained_tensor_bytes`를 함께 보세요. GPU 가중치, 디코더, snapshot 복사,
언어 모델의 KV cache와 누적 답변은 이 visual token 예산에 포함되지 않습니다.
같은 숫자의 long quota와 FIFO/Reservoir의 전체 token 예산을 직접 같다고 보지 마세요.

## 환경

Qwen3 기반 FIFO/Reservoir/FluxMem과 분리된 Python 3.10 또는 3.11 환경을 사용하세요.
`requirements.txt`는 native 소스에 맞춘 Transformers 4.46.0, Torch 2.1.2,
Torchvision 0.16.2와 최소 inference 의존성을 지정합니다. 실제 GPU 환경에 맞는
Torch/CUDA wheel이 필요하며, 영상 준비에는 FFmpeg와 FFprobe가 필요합니다.

```bash
cd baselines/streamforest
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

개발 중 CPU 검증에는 더 새로운 Torch 2.6 환경도 사용했습니다. 이는 모든
Torch/Transformers 조합의 호환성이나 GPU에서의 실제 모델 실행을 보장하지 않습니다.
Upstream import에서 `Please install pyav`가 출력될 수 있습니다. 원본 코드가
선택 의존성인 Decord 누락도 같은 문구로 표시하기 때문입니다. 이 어댑터의 영상
입력은 설치된 PyAV를 직접 사용하며 Decord 경로를 호출하지 않습니다.

## 입력 준비

`configs/sources.example.json`을 복사해 실제 영상의 절대경로, Day 순서와 해당 Day의
재생 시작 offset을 채우세요. 원본을 다시 인코딩하지 않고 길이와 SHA256을 확인하여
기존 베이스라인과 같은 manifest를 만듭니다.

```bash
python -m streamforest_videomem.prepare \
  --sources /absolute/path/sources.json \
  --output /absolute/path/manifest.json
```

FIFO/Reservoir용으로 준비한 같은 manifest와 질문 파일을 사용할 수 있습니다.
각 질문에는 `question_id`, `question`, `choices`(자유 응답이면 `[]`) 및
`query_time: {day_id, day_order, seconds}`가 필요합니다.
`configs/questions.example.json`은 스키마 설명용이며 실제 평가 질문이 아닙니다.
정답, 근거 구간과 추가 비공개 필드는 모델 프롬프트에서 제외합니다. 관측 종료 시점은
검토해 명시해야 하며 정답의 발생 시각에서 추정하지 않습니다.

관측 구간은 `[start, cutoff)`입니다. cutoff와 같은 실제 PTS의 프레임은 해당 질문에
들어가지 않습니다. single-frame 입력이므로 프레임 두 장을 맞추거나 단독 프레임을
버리지 않습니다. 질문 직후에 해당하는 프레임을 CPU에서 미리 디코딩할 수 있지만,
먼저 질문에 답한 뒤 인코딩하므로 미래 특징이 메모리에 들어가지 않습니다.

기본 `max_side=null`은 원본 해상도를 유지하며, native SigLIP 전처리에서 크기를
변환합니다. `max_side`를 지정하면 디코딩 단계에서 추가 thumbnail 처리를 합니다.
FPS와 이 전처리 선택도 비교 실험의 설정으로 기록하세요.

## 로컬 checkpoint 확인 및 실행

모델 파일을 준비한 뒤 로컬 inventory를 만듭니다.
처음 준비한다면 고정 revision을 명시하여 로컬 디렉터리에 다운로드할 수 있습니다.
아래 다운로드 명령은 대용량 모델을 실제로 저장하므로 모델 경로를 정한 뒤 실행하세요.

```bash
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="MCG-NJU/StreamForest-Qwen2-7B", revision="6840f91e17346b4c406ba03b57d1a63582765c91", local_dir="/absolute/path/StreamForest-Qwen2-7B")'
```

외부 파일로 향하는 symlink가 없는 로컬 디렉터리를 사용합니다. 다운로드 후:

```bash
python -m streamforest_videomem.checkpoint \
  --model-path /absolute/path/StreamForest-Qwen2-7B \
  --output /absolute/path/verification.json
```

이 파일은 **현재 로컬 파일의 크기와 해시 기록**입니다. Hugging Face에서 서명하거나
원격 원본과 대조한 attestation이 아닙니다. 최초 모델의 출처와 고정 revision을
확인한 상태에서 생성해야 하며, 실행기는 이후 파일 변경 여부를 이 기록으로 확인합니다.

`configs/streamforest.example.json`을 복사해 입력/출력/모델/verification 경로와
`memory_config`, `gpu`를 채운 뒤 실행하세요.

```bash
python -m streamforest_videomem.run \
  --config /absolute/path/streamforest.local.json
```

메모리 파일을 실행 시점에 지정하는 형식은 다음과 같습니다.

```bash
python -m streamforest_videomem.run \
  --config /absolute/path/streamforest.local.json \
  --memory-config /absolute/path/chosen_memory.json
```

CLI 값은 같은 config 필드보다 우선합니다. `gpu`는 명시적인 physical GPU index이며,
출력 디렉터리는 기존 결과를 덮어쓰지 않는 새 경로여야 합니다. 구체적인 GPU나
메모리 크기를 정하기 전에도 코드와 CPU 테스트를 확인할 수 있습니다.
상대경로는 현재 작업 디렉터리를 기준으로 해석합니다.

출력에는 `predictions.json`, `events.jsonl`, `status.json`, `provenance.json`,
검토된 입력 설정 및 `memory.pt`가 저장됩니다. `memory.pt`는 메모리 bank만의
checkpoint이며 영상 커서까지 포함한 전체 실행 재개 기능은 아닙니다. EOS 없이
잘린 답변이나 관측 프레임이 없는 질문은 미완료로 기록하고 종료 코드 2를 반환합니다.

## 검증과 다음 실험

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 USE_FLAX=0 \
  PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q -p no:cacheprovider tests
```

테스트는 작은 합성 영상과 tensor를 사용해 PTS 경계, source 변경 감지, 이미지 수명,
질문 간 메모리 유지, 질문 횟수와 압축 결과의 독립성을 검사합니다. 공식 manager의
동작과 native 모델 연결 검증은 사전학습 모델의 정답률 평가와 별개입니다.
실제 데이터셋 영상이나 모델 가중치를 이 구현 검증 때문에 다운로드하지 않습니다.

실제 비교 실험을 시작하려면 사용할 메모리 JSON, GPU, 로컬 모델/영상 경로 및 검토된
질문·관측 종료 시점을 정하면 됩니다. 정확도 비교에는 별도 정답 파일도 필요합니다.
동일 모델·입력 해상도·FPS·질문·관측 범위를 맞추고 전체 token 수와 byte 수를 기록하세요.
여기서는 Qwen2 기반 공식 StreamForest를 사용하므로 Qwen3 기반 FIFO/Reservoir와의
차이를 메모리 정책만의 효과로 해석해서는 안 됩니다.
