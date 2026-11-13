# Qwen3-VL offline causal-prefix baseline

Query마다 과거 원본 영상에 다시 접근하고, 공식 HF Qwen3-VL sampling 및
raw-video inference를 실행합니다. Feature archive나 persistent KV를 만들지
않습니다. 계산 재사용보다 공식 경로의 fidelity를 우선합니다.

## 입력과 시간 정책

- `sources.json`: day 순서의 절대 MP4 경로 JSON 배열. 현재 day1~day7처럼
  동일 해상도·동일 FPS의 CFR 영상만 지원합니다.
- 영상 stream의 정확한 길이로 day를 이어 하나의 virtual video를 만듭니다.
  container/audio 길이는 사용하지 않습니다. 경계에 gap을 추가하지 않습니다.
- query에 사용할 원본 프레임은 **global PTS ≤ query_time**입니다.
  기존 streaming baseline의 `< t` 정책과 경계 포함 여부가 다릅니다.
- 모든 day를 합친 prefix에서 공식 sampler를 한 번 호출합니다. Day별 sampling,
  1fps archive, 사전 thumbnail/resize는 사용하지 않습니다. Day 경계를 넘는
  temporal pair도 공식 전체 video sequence의 일부입니다.
- 원본 CFR FPS와 선택된 원본 frame indices를 processor metadata로 전달하여
  timestamp를 유지합니다. `do_sample_frames=False`는 이미 공식 sampling을
  적용한 sequence를 다시 sampling하지 않기 위한 것입니다.
- 원본은 keyframe seek 후 선택 PTS를 검증해 RGB로 읽습니다. Codec preroll과
  reference decoding은 내부 작업이며, 미래 프레임은 모델 입력에 넣지 않습니다.
  전체 파일의 모든 PTS를 전수 검사하지는 않으며, metadata와 실제 접근한 PTS가
  CFR grid와 다르면 중단합니다. VFR, 서로 다른 FPS/해상도는 지원하지 않습니다.
- raw `frame_id`는 전체 원본의 ordinal입니다. 기존 1fps archive의
  `global_frame_id`와 같은 숫자 체계가 아니며, exact global/source timestamp로
  기존 타임라인과 대응합니다.

질문 JSON은 다음과 같습니다. `query_time`은 사람이 검토한 관측 종료 시각이어야
합니다. 정답·근거 시각으로 자동 생성하지 않습니다. 추가 정답 필드는 모델에서
제외합니다. 현재 저장소 QA에는 이 cutoff가 없으므로 그대로 평가할 수 없습니다.

```json
[
  {
    "question_id": "example-only",
    "question": "What happened?",
    "choices": [],
    "query_time": {"day_id": "Day1", "day_order": 1, "seconds": 30}
  }
]
```

## 경로와 버전

공식 모델: `Qwen/Qwen3-VL-8B-Instruct`, revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
HF `transformers==4.57.3` 경로를 고정합니다. qwen-vl-utils 경로와 혼용하지 않습니다.
기본 official sampler는 2fps, 최소 4/최대 768 선택 프레임 정책입니다. 실제 짧은
prefix는 원본 프레임 수로 제한됩니다. 2프레임 미만 prefix는 오류로 처리합니다.
Sampler의 `linspace(...).round()`와 processor의 전체 video 길이 기반 resize,
2-frame temporal patch, final merger, DeepStack, timestamp/RoPE 및 생성 경로를
그대로 사용합니다. 임의 `max_pixels`/frame budget override를 제공하지 않습니다.

추론 환경은 BF16 / SDPA / greedy(`do_sample=False`, `num_beams=1`)입니다.
현재 검증 환경: Python 3.9, torch 2.8.0+cu128, torchvision 0.23.0+cu128,
transformers 4.57.3, accelerate 1.10.1, av 14.2.0. 독립 환경에서 설치하고,
다른 실행 환경의 수치 결과가 동일하다고 가정하지 마세요. FFmpeg/FFprobe도 필요합니다.

로컬 모델은 기존 `streaming.checkpoint.verify_weights`로 매 run 전체 해시를
검증합니다. `verification.json`은 pinned official HF 파일 metadata와 대조한
`repo_id`, `revision`, `verified_files:[{path, bytes, hash}]`를 포함해야 합니다.
`plan`은 processor만 읽으며 가중치 검증이나 GPU 추론을 수행하지 않습니다.
따라서 plan provenance에는 `checkpoint_verified=false`가 기록됩니다.

## 실행

저장소 루트에서 실행합니다. 각 출력은 새 경로여야 하며 자동 덮어쓰지 않습니다.

```bash
python -m baselines.qwen3_offline prepare \
  --sources /absolute/sources.json --output /absolute/inventory.json

python -m baselines.qwen3_offline plan \
  --inventory /absolute/inventory.json --questions /absolute/reviewed-questions.json \
  --model /absolute/Qwen3-VL-8B-Instruct --output /absolute/plan-output

python -m baselines.qwen3_offline run \
  --inventory /absolute/inventory.json --questions /absolute/reviewed-questions.json \
  --model /absolute/Qwen3-VL-8B-Instruct --verification /absolute/verification.json \
  --gpu 0 --max-new-tokens 192 --output /absolute/run-output
```

`--gpu`는 physical index이며 UUID로 선택합니다. 선택 GPU가 1GiB 초과 사용 중이면
기존 작업을 중단하지 않고 오류로 종료합니다. 한 프로세스/한 GPU 순차 실행입니다.

- `plan.json`: 공개 질문, prefix raw frame 수, sampled indices와 원본 시간 대응.
- `provenance.json`: 모델 검증, runtime/processor/generation 설정과 입력·코드 해시.
- `predictions.json`: 답변, 생성 token IDs, 선택 프레임, grid, 실행 시간.
- `status.json`: planned/loading/running/complete/incomplete/error.

EOS로 끝난 비어 있지 않은 답변만 complete입니다. 길이 제한으로 잘리면
incomplete와 종료 코드 2를 반환합니다. 실패 시 이미 완료된 답변은 남지만,
resume 기능은 제공하지 않습니다. 입력 inventory는 source stat 변화에도 중단합니다.

## 검증과 cache 결정

```bash
USE_TF=0 python -m pytest baselines/qwen3_offline/tests -q

# 로컬 official processor가 있으면, lossless prefix 파일을 읽는 공식 PyAV
# 경로와 새 seek 경로의 sampled pixels 및 모든 processor tensor를 비교합니다.
USE_TF=0 QWEN3_MODEL=/absolute/Qwen3-VL-8B-Instruct \
  python -m pytest baselines/qwen3_offline/tests -q
```

2026-11-11 day1의 고정 4/8 sampled frames 실험에서 전체 sequence의 final+
DeepStack 저장/복원은 입력·KV `torch.equal=True`였습니다. 반면 별도 pair
인코딩 조합은 8프레임에서 final embedding 최대 절대 차이 0.921875, KV 최대
차이 36.0으로 exact equality에 실패했습니다. 이 제한된 관측을 모든 backend의
불가능성 증명으로 해석하지 않습니다. 현재 baseline은 이 최적화를 채택하지 않습니다.

원래 1fps archive는 공식 sampling 시점과 resize 조건도 포괄하지 못합니다.
GPU smoke는 합성 영상의 실행 검사이며 benchmark QA 성능 측정이 아닙니다.
