# 영상 메모리 베이스라인과 visual token 추출

`j-beast123/videomem-baselines`의 베이스라인 구현과
`feat/cook-visual-token-extraction`의 추출 도구를 옮긴 소스 패키지다.
실험 중인 Routine-Aware Memory 구현도 별도로 포함한다. 가져온 커밋과
파일별 해시는 [SOURCE_SNAPSHOT.json](SOURCE_SNAPSHOT.json)에 기록한다.

이 디렉터리를 Python 실행 기준 경로로 사용한다. 저장소 루트의 npm 타임라인
생성·렌더링과는 별도 실행 경로이며, 합성 manifest가 자동으로 베이스라인의
영상·QA 입력으로 변환되지는 않는다.

```bash
cd integrations/videomem
```

## 우선 실행할 visual token 추출

Qwen2.5-VL-7B의 vision encoder만 사용해 각 원본 영상 전체를 1fps로 처리한다.
모델의 spatial merger 직후 특징을 bf16 `[tokens, 3584]`로 저장한다.
언어 모델 추론, QA 풀이, KV 생성, bounded-memory 구축은 실행하지 않는다.

영상 해상도마다 토큰 수가 다를 수 있어서 영상별 archive를 만든다. `clips.csv`와
타임라인은 대응 정보로 보존하며, 이를 근거로 영상 구간을 자르지 않는다.

설치, 5분 테스트, 전체 실행, 저장 형식과 실패 조건은
[visual token 실행 가이드](tools/feature_memory/VISUAL_TOKENS.md)에 있다.

```bash
PY=/path/to/visual-token-env/bin/python
DATASET=/path/to/cook
MODEL=/path/to/Qwen2.5-VL-7B-Instruct

# 실제 파일 경로는 DATASET/videos/ 아래에서 고른다.
"$PY" tools/feature_memory/extract_visual_tokens.py \
  --dataset-root "$DATASET" --model "$MODEL" \
  --output /path/to/new/token-smoke --devices 0 \
  --source-relative videos/work/example.mp4 --max-frames-per-source 300

# 전체 원본을 GPU 6개에서 병렬 처리하는 예시. 사용 가능한 GPU만 지정한다.
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
"$PY" tools/feature_memory/extract_visual_tokens.py \
  --dataset-root "$DATASET" --model "$MODEL" \
  --output /path/to/new/all-visual-tokens --devices 0 1 2 3 4 5
```

지정한 폴더가 없으면 실패한다. 다른 폴더나 다른 인코딩의 영상으로 자동 대체하지
않는다. 영상 끝의 PTS 때문에 예상 1fps 샘플이 부족해도 완료로 표시하지 않는다.
완료 영상의 재사용은 가능하지만, 미완성 영상 내부부터 이어 쓰는 기능은 없다.

## 포함된 구현

| 분류 | 코드와 실행 안내 |
| --- | --- |
| Qwen3-VL / LLaVA-OneVision 오프라인 입력 | [ours_offline](baselines/ours_offline/README.md), [qwen3_offline](baselines/qwen3_offline/README.md) |
| GT evidence reference | [ours_evidence](baselines/ours_evidence/README.md) |
| AKS / VideoTree 선택 프레임 | [ours_selection](baselines/ours_selection/README.md) |
| EgoGPT 캡션 / EgoRAG | [egogpt_caption](baselines/egogpt_caption/README.md), [egorag](baselines/egorag/README.md) |
| ReKV Qwen2.5-VL | [rekv_qwen25](baselines/rekv_qwen25/README.md) |
| FIFO / Uniform / Reservoir / FluxMem 특징 메모리 | [feature_memory](baselines/feature_memory/README.md) |
| Qwen3-VL FIFO / Reservoir | [fifo_qwen3vl](baselines/fifo_qwen3vl/README.md), [reservoir_qwen3vl](baselines/reservoir_qwen3vl/README.md) |
| FluxMem 원본·스트리밍·Qwen3 변형 | [fluxmem_reference](baselines/fluxmem_reference/README.md), [fluxmem_streaming](baselines/fluxmem_streaming/README.md), [fluxmem_qwen3vl](baselines/fluxmem_qwen3vl/README.md) |
| StreamForest / SimpleStream | [streamforest](baselines/streamforest/README.md), [simplestream](baselines/simplestream/README.md) |
| Cambrian-S 실행 어댑터 | [ours_cambrian](baselines/ours_cambrian/README.md) |
| JAKE 온라인 / StreamTTT 실행 어댑터 | [jake_online](baselines/jake_online/README.md) |
| 실험용 Routine-Aware Memory | [routine_memory](baselines/routine_memory/README.md) |
| 공통 causal stream 평가 인터페이스 | [streambench](baselines/streambench/README.md) |

각 방법은 입력 형식, 모델, 환경 의존성이 다르다. 해당 README와 requirements를
따라 개별 환경을 준비한다. 일부 어댑터는 외부 모델 구현·체크포인트를 별도로
요구하며, 코드가 포함됐다는 사실만으로 논문 재현이나 QA 정확도를 보장하지 않는다.
레거시 실행 예시의 로컬 경로와 GPU 설정은 사용할 서버에 맞게 바꿔야 한다.

토큰 추출의 영상별 archive를 모든 메모리·QA 실행기에 그대로 연결할 수 있는 것은
아니다. 특히 고정 토큰 geometry와 단일 stream 타임라인을 요구하는 실행기는 입력
계약을 먼저 확인해야 한다. 공용 `iter_archive`로는 각 영상의 저장 특징을 읽을 수 있다.

## 입력 데이터와 출처

이 패키지는 코드, 설정 예시, 합성 테스트, 문서, upstream 라이선스·고지 파일을
포함한다. 원본 저장소의 `data/` 주석·QA 패키지, 영상, 모델 가중치, 특징 archive,
실행 결과 및 운영 환경 설정은 배포하지 않는다. 모델·영상·QA 경로는 실행할 때
명시하고, 접근 자격은 각 사용자의 환경에서 설정한다.

원본의 공개된 구현 및 어댑터 구분과 라이선스는 각 방법의 README와
[출처 안내](docs/egorag_export.md), [FluxMem 출처](docs/fluxmem_export.md)를 따른다.
포함된 라이선스를 다른 라이선스로 대체하거나 전체 코드에 새 라이선스를 부여하지
않는다. 원래 저장소의 내부 설계 이력과 실제 실험 결과는 이 배포에서 제외했다.

## 검증

아래 테스트는 CPU의 구조·저장·인과성 검증이다. 모델 다운로드나 전체 데이터셋의
QA 평가를 실행하지 않는다.

```bash
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
"$PY" -m pytest -q \
  tools/feature_memory/tests/test_extract_visual_tokens.py \
  baselines/feature_memory/tests/test_archive.py \
  baselines/feature_memory/tests/test_video.py \
  baselines/routine_memory/tests
```

실제 GPU 실행은 작은 prefix로 shape, dtype, bytes, PTS와 출력 receipt를 확인한
뒤 시작한다. 이번 소스 이관은 모델별 전체 평가를 다시 수행한 결과가 아니다.
