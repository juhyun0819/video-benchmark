# ReKV continual adapter

`baselines.streambench.adapters.rekv` uses the [official ReKV implementation](https://github.com/Becomebright/ReKV) from *Streaming Video Question-Answering with In-context Video KV-Cache Retrieval*. The inference subset is pinned to commit [`5a17285d5e1a7488228eb0098b5e586d72a73e3a`](https://github.com/Becomebright/ReKV/commit/5a17285d5e1a7488228eb0098b5e586d72a73e3a) in `baselines/streambench/vendor/rekv`; provenance records every upstream Git blob. Source verification runs before use. The only normalization is one final newline in the attention package initializer. The upstream project metadata declares the Apache Software License; upstream does not include a separate LICENSE file.

The supported frozen backbone is the official repository's [LLaVA-OneVision Qwen2 7B checkpoint](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf). The model keeps its vision tower, projector, pooling, decoder width/layers and native 196 visual positions per frame. It does not retrain. ReKV's [installation script](https://github.com/Becomebright/ReKV/blob/5a17285d5e1a7488228eb0098b5e586d72a73e3a/prepare.sh) pins Transformers commit `66bc4def9505fa7c7fe4aa7a248c34a026bb552b`; use that compatible environment rather than assuming newer Transformers is compatible with its attention patch.

## Native policy and byte accounting

The unchanged [ContextManager](https://github.com/Becomebright/ReKV/blob/5a17285d5e1a7488228eb0098b5e586d72a73e3a/model/attention/kv_cache_manager.py) maintains:

- Per-layer initial and recent KV, pending global-remainder KV and indexing boundaries.
- Complete older KV blocks in CPU `MemoryUnit` archives. These remain retrievable and are charged to B.
- GPU representative-key `VectorTensor` allocations, including their unused allocated capacity.
- A fixed GPU `CudaCache` pool, its free slot set, active block views, CPU copies and GPU copies.
- Global retrieval buffers, initialization ranges, LRU/load counters, retrieval indices/modes, similarity arrays and total lengths.
- RoPE inverse frequencies, cosine/sine tables and cached-length bookkeeping.

The CPU archive grows without a bounded eviction rule. This adapter preserves it. `n_local` limits the sliding attention window; `topk` and `retrieval_chunk_size` control native retrieval; `max_cached_block` limits the GPU block pool. None of these deletes the CPU archive. If total state exceeds B, observation rolls back and reports `unsupported_under_budget`. It never substitutes FIFO, reservoir sampling, or feature compression.

Default native settings are `n_local=15000`, `topk=64`, `retrieval_chunk_size=1`, `max_cached_block=128`, 196 positions per block, float16 and pinned CPU memory. For the 28-layer, 4-KV-head, 128-head-dimension geometry, the fixed GPU block pool and retrieval buffer alone cost at least **2,157,969,408 bytes**, before prompt KV, local/remainder KV, RoPE, indices or CPU archives. Thus 64–1024 MiB pilots with these settings are correctly unsupported. The adapter calculates this conservative mandatory lower bound from local `config.json` before allocating model weights. Optional smaller native window/read/cache settings are explicit experiment settings; the benchmark never shrinks hidden dimensions or pretrained layers.

All subsequent decisions use actual live-state accounting. CPU/GPU copies count separately; shared cache views count once. The runner also charges its stream cursor/RNG metadata. Archived benchmark checkpoints are not reachable by the adapter and do not serve as extra retrieval storage.

## Observation, restore and QA

The common sampler supplies chronological RGB frames. The adapter calls the original `encode_video` path one frame at a time, keeping the native preprocessing and sliding attention. This fixes the encoder chunking across common input chunks and day boundaries. All per-layer native contexts persist across Day1–Day7; only a new experiment's `reset()` reinitializes them.

The checkpoint graph preserves complete native object identities and tensor storage aliases, including CPU archive pointers to the shared GPU pool. Tensors carry their original device placement so a checkpoint loaded on CPU can rebuild CPU archives and GPU state correctly. CUDA work is synchronized before export; completed scheduling events and the global stream are reconstructed on restore. They contain no additional semantic memory. Unknown persistent object types fail explicitly instead of being omitted.

The original QA method changes retrieval/LRU buffers. The adapter snapshots the full graph and restores it in `finally`, including decoding failures. A temporary instrument around the original `get_retrieved_kv` measures actual per-layer KV payload sizes and retrieved block indices; it is removed before restore. Question text never enters observation state. Aggregate `query_read_bytes` conservatively reports all live native KV/cache/RoPE tensor storage with shared allocations deduplicated and `query_read_bytes_is_upper_bound=true`. `query_selected_kv_bytes` separately reports native selected KV payload; `query_representative_scan_bytes` reports the ranking branch's key scan. The latter includes raw global-remainder video keys before offloading begins. Item counts describe layer/block uses, not unique source-video frames.

Every checkpoint records the pinned source revision, method configuration, stream progress and a SHA256 fingerprint of the actual local model/tokenizer/configuration artifacts. Model loading is local-only; a path reused for different weight bytes is rejected on restore. Hashing adds sequential model-load I/O without retaining another copy of the model.

## Configuration and commands

Example native configuration, with an illustrative 4 GiB B that still requires measured profiling:

```json
{
  "method": "rekv",
  "memory_budget_bytes": 4294967296,
  "model_id": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
  "model_path": "/absolute/local/llava-onevision-qwen2-7b-ov-hf",
  "device": "cuda:0",
  "dtype": "float16",
  "n_local": 15000,
  "topk": 64,
  "max_cached_block": 128,
  "retrieval_chunk_size": 1,
  "max_new_tokens": 128
}
```

```bash
python -m baselines.streambench observe --config rekv.json --manifest frames.jsonl \
  --output runs/rekv-smoke --stop-after-day 2
python -m baselines.streambench evaluate --config rekv.json \
  --checkpoint runs/rekv-smoke/checkpoint_D1_D2 \
  --questions questions_day1.json --output runs/rekv-day1-qa.json
python -m baselines.streambench sweep --config rekv.json --manifest frames.jsonl \
  --output runs/rekv-budget-pilot --budgets-mib 64 128 256 512 1024 --stop-after-day 2
```

Production initialization requires CUDA and the compatible native dependencies. Missing weights or dependencies produce an explicit failure; there is no replacement model or CPU surrogate. No full seven-day run has been performed.

## Validation evidence and limits

`baselines/streambench/tests/test_streambench_rekv.py` exercises the unchanged native CPU offload and dot-product retrieval code with synthetic layer KV supplied at the encoder boundary. It tests complete graph serialization, shared CPU/cache aliases, continued representative-key growth, fresh-process resume, independent QA, restoration after failed QA, budget overflow rollback and the mandatory native pool floor. These fixtures explicitly bypass CUDA allocation and LLM encoding; they are structural validation, not a pretrained model-quality or CUDA-attention result.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -q -p no:cacheprovider baselines/streambench/tests/test_streambench_rekv.py
```

The current workspace lacks local ReKV model weights and a validated GPU runtime for its pinned Transformers version. Native model loading, CUDA attention, event reconstruction and real-video answer quality therefore remain unvalidated here, while the actual source/API connection and CPU policy/state paths are implemented. The isolated test environment supports Qwen model imports, but that does not validate ReKV's separate pinned attention patch or pretrained execution.
