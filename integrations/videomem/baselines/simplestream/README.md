# SimpleStream — Qwen2.5-VL-7B native-mRoPE adaptation

최근 N개 관측 프레임만 보관하는 학습 없는 baseline입니다. 관측 때는 Qwen의
post-merger visual feature를 보관하고, 질문마다 그 창을 이용해 새로 생성합니다.
질문별 검색, 압축, 과거 LLM KV 누적, 학습을 하지 않습니다.

논문의 **SimpleStream-8B/4B는 Qwen3-VL의 모델 크기**입니다. 이 구현은 기존
비교군과 같은 **Qwen2.5-VL-7B-Instruct**를 사용하며 8B/4B로 표시하지 않습니다.
최근 프레임 수는 `recent_frames` 또는 `--recent-frames`로 독립적으로 지정합니다.
표에는 `SimpleStream (Qwen2.5-VL-7B, 4f)`처럼 모델과 창 크기를 함께 적으세요.

공식 소스는 `EvolvingLMMs-Lab/SimpleStream`의
`be227ee8ce90a258b286a34161688c2a59eeeadb`를 확인했습니다. 공식 cached QA 코드의
위치 정보 처리 문제를 그대로 재현하지 않고 Qwen 기본 multi-image mRoPE와
생성 경로를 보존합니다. 따라서 **최근-N 정책의 명시적 변형**이며 원본 코드의
bitwise 재현이나 논문 정확도 재현을 주장하지 않습니다. 구체적인 차이와 근거는
[UPSTREAM_REVIEW.md](UPSTREAM_REVIEW.md)를 참고하세요.

## 입력과 메모리

- 모델: `Qwen/Qwen2.5-VL-7B-Instruct`
- 고정 revision: `cc594898137f460bfe9f0759e9844b3ce807cfb5`
- dtype: BF16, 학습 비활성, deterministic greedy decoding
- 기존 generation config의 repetition penalty `1.05`와 EOS 설정을 보존하고 해시를 기록
- 기존 영상 프로토콜: source-local 1 FPS, 각 프레임을 독립 image로 인코딩
- 기존 cache: frame당 `[676,3584]`, image grid `[1,52,52]`
- 창은 Day 경계를 지나서도 유지하며 가장 오래된 프레임부터 제거
- 질문 cutoff는 `timestamp <= cutoff`; 미래 프레임은 선택하지 않음
- 질문마다 decoder KV와 mRoPE 내부 상태를 새로 시작하며 observation은 변경하지 않음

기존 고정 해상도 cache 기준으로 유지하는 feature payload는 다음과 같습니다.
grid와 metadata의 작은 추가 비용은 실제 메모리 계수에 포함됩니다. 모델 가중치,
일시적인 QA KV와 직렬화 복사본은 observation memory와 구분합니다.

| 최근 프레임 | Visual tokens | Feature payload |
| --- | ---: | ---: |
| 4 | 2,704 | 19,382,272 bytes |
| 8 | 5,408 | 38,764,544 bytes |
| 16 | 10,816 | 77,529,088 bytes |

SimpleStream을 위해 전체 영상의 LLM KV archive를 만들 필요는 없습니다.
기존 visual archive는 공통 인코딩 계산을 재사용하는 입력 cache입니다. 질문은
해당 시점 이전의 가장 최근 N개 feature만 읽으며 모든 과거 feature를 검색하지
않습니다. 저장소 전체를 함께 사용하세요. 검증된 cache/checkpoint 읽기는
`baselines.rekv_qwen25.rekv_qwen25.inputs`의 변경 없는 입력 유틸리티를 재사용하며
ReKV attention이나 실행 상태는 사용하지 않습니다.

## 기존 cache로 바로 사용

저장소 루트에서 실행합니다. NAS의 `rekv-qwen25` 환경을 **읽기 전용으로 재사용**하며
실행 중인 ReKV 프로세스나 설치 패키지를 변경하지 않습니다. 모델, cache와 ledger는
자동 다운로드하지 않습니다.

```bash
export USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SIMPLESTREAM_PY=/convei_nas2/zzuhyun/hyunjun/envs/rekv-qwen25/bin/python
SIMPLESTREAM_ARCHIVE=/convei_nas2/zzuhyun/hyunjun/feature_archives/qwen25_day1_7_v1

# 20시간 시점에 허용되는 최근 4개 프레임과 메모리 양 확인. GPU 불필요.
"$SIMPLESTREAM_PY" -m baselines.simplestream inspect \
  --archive "$SIMPLESTREAM_ARCHIVE" --cutoff-seconds 72000 --recent-frames 4

# 기능 사용 예시이며 평가용 정답/질문 집합이 아닙니다.
"$SIMPLESTREAM_PY" -m baselines.simplestream answer \
  --archive "$SIMPLESTREAM_ARCHIVE" --cutoff-seconds 72000 --recent-frames 4 \
  --model /convei_nas2/zzuhyun/hyunjun/models/Qwen2.5-VL-7B-Instruct \
  --ledger /convei_nas2/zzuhyun/hyunjun/models/.downloads/Qwen2.5-VL-7B-Instruct/verification.json \
  --device cuda:2 --question 'Describe what is currently visible.' \
  --output /tmp/simplestream-example.json
```

`--output`은 새 파일이어야 합니다. 불완전한 입력 shard, 모델 불일치, checksum
오류를 다른 frame으로 대체하지 않습니다. 기존 archive의 일부만 준비된 경우
요청한 창이 모두 commit되어 있어야 합니다. 오래된 질문을 disk cache에서
재구성하는 것은 가능하지만, 이미 eviction된 live memory만으로 과거 시점의
정확한 창을 되살리지는 않습니다.

## 공통 continuous benchmark 연결

`python -m baselines.streambench list-methods`에 `simplestream`이 등록됩니다.
[설정 예제](../streambench/configs/simplestream.example.json)의 모델/ledger 경로와
GPU를 지정한 뒤 공통 `observe` / `evaluate` 명령을 사용합니다.

이 경로는 기존 공통 manifest의 PIL 프레임을 인코딩하여 최근 창을 유지합니다.
위의 CLI는 이미 만들어진 visual cache를 읽는 별도 경로입니다. 공통 evaluator의
입력 형식을 묵시적으로 cache 형식으로 바꾸지 않습니다. 실제 persistent memory가
byte budget을 넘으면 unsupported로 처리하며 프레임 창을 몰래 축소하지 않습니다.
checkpoint에는 선택된 features/grids, 관측 cursor, 모델/설정 identity가 들어갑니다.

## 검증

```bash
USE_TF=0 USE_FLAX=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /convei_nas2/zzuhyun/hyunjun/envs/rekv-qwen25/bin/python -m pytest -q \
  baselines/simplestream/tests \
  baselines/streambench/tests/test_streambench_simplestream.py
```

GPU smoke는 실제 Day1 영상 및 기존 cache로 native processor 입력, image features,
mRoPE, KV, logits, 생성 token과 질문 간 상태 독립성을 비교합니다. 결과 report의
status와 각 비교를 확인하세요. CPU 테스트 통과와 pretrained 정확도 평가는 별개이며,
이 구현 작업은 미완성 QA 집합의 전체 평가나 논문 점수 재현을 실행하지 않습니다.

2026-11-12 실행 결과: 새 CPU 테스트 **62개**, 기존 CLI/core 회귀 테스트 **9개**가
통과했습니다. GPU 2의 실제 pretrained 모델에서 **1/4/8/16프레임 모두** 원본 영상과
cache의 입력 IDs·mask·grid·features·mRoPE·전체 28-layer pre/post-RoPE KV·logits·
생성 token이 비트 단위로 일치했습니다. A→B→A 질문 독립성도 통과했습니다.
최대 `torch.cuda.max_memory_allocated`는 **17.76 GiB**였으며 allocator 예약량이나
프로세스 전체 GPU 점유량과는 구분합니다. 기능/fidelity 시험이며 정확도 측정이 아닙니다.

실행 보고서:
`/home/zzuhyun/현준/simplestream_work/gpu2_native_fidelity_v1/report.json`.
재실행은 비어 있는 GPU와 새로운 출력 폴더를 지정하세요.

```bash
"$SIMPLESTREAM_PY" -m baselines.simplestream.smoke \
  --gpu 2 --output /absolute/path/new-fidelity-run
```
