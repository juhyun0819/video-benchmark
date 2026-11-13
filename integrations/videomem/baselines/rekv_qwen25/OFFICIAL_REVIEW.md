# Official ReKV implementation review and Qwen2.5-VL adaptation boundaries

Reviewed on 2026-11-11. This document distinguishes the released ReKV algorithm
from proposed changes for **ReKV (Qwen2.5-VL-7B adaptation)**. It does not claim
that the adaptation or its full archive has passed execution tests.

## Sources and provenance

- Paper: [Streaming Video Question-Answering with In-context Video KV-Cache
  Retrieval, ICLR 2025](https://arxiv.org/abs/2503.00540).
- Official repository: [Becomebright/ReKV](https://github.com/Becomebright/ReKV).
- Reviewed commit: **`5a17285d5e1a7488228eb0098b5e586d72a73e3a`**. The official
  repository's main branch resolved to this commit at review time.
- Existing source subset:
  `../streambench/vendor/rekv/`, with `provenance.json`. All 12 recorded Git blob
  hashes were verified, accounting for the explicitly recorded added trailing
  newline in `model/attention/__init__.py`. Its revision matches the official
  checkout. The complete read-only review checkout is
  `/tmp/rekv_official_review`.
- Upstream `pyproject.toml` declares the Apache Software License classifier;
  the reviewed repository has no standalone LICENSE file. Preserve the recorded
  provenance rather than inventing an upstream license file.

The code is the authority for the implementation details below. For example,
the paper describes similarity in cosine terms, but the released internal
retrieval implementation uses an unnormalized float32 dot product.

## 1. Original supported backbones

The evaluation registry supports LLaVA-OneVision with Qwen2 0.5B, 7B and 72B,
Video-LLaVA-7B, and LongVA-7B. A separate Flash-VStream adapter file exists but
is not in that registry. The attention patch dispatches on Llama, Mistral,
Qwen2, and a MiniCPM class-name case. **Qwen2.5-VL is not a released supported
adapter**; replacing a checkpoint name is insufficient.

References: [model registry](https://github.com/Becomebright/ReKV/blob/5a17285d5e1a7488228eb0098b5e586d72a73e3a/video_qa/base.py#L23),
[attention/model class dispatch](https://github.com/Becomebright/ReKV/blob/5a17285d5e1a7488228eb0098b5e586d72a73e3a/model/patch.py#L136).

## 2. KV extraction layer and timing

Every decoder attention layer projects Q, K and V from its input hidden state.
Projected K/V are reshaped to `[batch, num_key_value_heads, tokens, head_dim]`
and passed to the layer's `ContextManager`. **These cached keys are pre-RoPE**.
RoPE is applied to temporary Q/K tensors when calculating attention, not stored
permanently in the offloaded key payload. V is the corresponding projection.

This distinction matters because Hugging Face's normal Qwen2.5-VL cache stores
keys after mRoPE. A dump of that cache is useful as a reference, but is not the
same storage representation as native ReKV.

References: [projection and reshape](https://github.com/Becomebright/ReKV/blob/5a17285d5e1a7488228eb0098b5e586d72a73e3a/model/attention/rekv_attention.py#L27),
[encoding branch](https://github.com/Becomebright/ReKV/blob/5a17285d5e1a7488228eb0098b5e586d72a73e3a/model/attention/rekv_attention.py#L138),
[temporary rotary application](https://github.com/Becomebright/ReKV/blob/5a17285d5e1a7488228eb0098b5e586d72a73e3a/model/attention/kv_cache_manager.py#L534).

## 3. Sampling and chunk units

The paper's experimental settings use **0.5 FPS**, FP16, and a local attention
window of 15,000 tokens. The released CLI defaults to **1 FPS**, so FPS is a
configurable protocol choice, not an architectural constant. The raw-video
loader samples at a stride based on rounded source FPS. The benchmark's existing
1 FPS, exact-source-PTS timeline should be explicitly documented as the
controlled comparison's sampling protocol.

Three different quantities must not be confused:

| Quantity | LLaVA-OneVision implementation |
| --- | --- |
| Video encoding batch | `encode_chunk_size=64` frames, with a smaller tail |
| Cache/retrieval block | `block_size=196` visual tokens, one frame |
| Internal attention execution block | `exc_block_size=196` tokens |
| Retrieval grouping | `chunk_size=1` frame by default; can group consecutive frame scores |

The encoder asserts that a video's input chunk fits inside `n_local`. The native
64-frame choice fits because `64 * 196 = 12,544 <= 15,000`. With 676 Qwen visual
tokens/frame, the same 64-frame choice would require 43,264 tokens and violate
that constraint. A pure-visual chunk can contain at most 22 frames at 15K;
special/text tokens must also be counted. The requested 1/4/8/16-frame smoke
sizes fit this token bound, subject to their actual sequence construction.

References: [paper implementation settings](https://arxiv.org/html/2503.00540v1#S4.SS2),
[CLI defaults](https://github.com/Becomebright/ReKV/blob/5a17285d5e1a7488228eb0098b5e586d72a73e3a/video_qa/base.py#L179),
[raw-video sampling](https://github.com/Becomebright/ReKV/blob/5a17285d5e1a7488228eb0098b5e586d72a73e3a/video_qa/rekv_stream_vqa.py#L9),
[encoding chunks and assertion](https://github.com/Becomebright/ReKV/blob/5a17285d5e1a7488228eb0098b5e586d72a73e3a/model/abstract_rekv.py#L24),
[native frame/block settings](https://github.com/Becomebright/ReKV/blob/5a17285d5e1a7488228eb0098b5e586d72a73e3a/model/llava_onevision_rekv.py#L109).

## 4. Sliding-window attention and context state

At each layer, new raw K/V are appended to a persistent local cache. Encoding
attention combines causal local attention with attention to the initial prompt
tokens. The local mask accepts distances `0 <= query_index-key_index < n_local`.
After a chunk, the layer retains the last `n_local` raw K/V tokens. Initial
prompt/sink K/V remain separately available. The multi-stage attention combines
the allowed logits under one softmax, avoiding two independently normalized
attention distributions.

The video encoder feeds `self.kv_cache` from the previous chunk into the next
language-model call. Consequently, higher-layer K/V contain preceding temporal
context. A storage shard is not an independent model context.

References: [persistent chunk cache](https://github.com/Becomebright/ReKV/blob/5a17285d5e1a7488228eb0098b5e586d72a73e3a/model/abstract_rekv.py#L28),
[append and retain local K/V](https://github.com/Becomebright/ReKV/blob/5a17285d5e1a7488228eb0098b5e586d72a73e3a/model/attention/kv_cache_manager.py#L616),
[local/sink attention](https://github.com/Becomebright/ReKV/blob/5a17285d5e1a7488228eb0098b5e586d72a73e3a/model/attention/kv_cache_manager.py#L518),
[joint softmax](https://github.com/Becomebright/ReKV/blob/5a17285d5e1a7488228eb0098b5e586d72a73e3a/model/attention/dot_production_attention/torch_impl.py#L16),
[exact local mask](https://github.com/Becomebright/ReKV/blob/5a17285d5e1a7488228eb0098b5e586d72a73e3a/model/attention/dot_production_attention/torch_impl.py#L60).

## 5. KV offload representation

Each `MemoryUnit` owns a pair of contiguous CPU K/V tensors, optionally pinned,
with per-batch-element shape `[kv_heads, block_tokens, head_dim]`. It may also
own a slot in a fixed GPU block cache and a CUDA completion event. Global frame
blocks and their representative keys are tracked separately from the local
attention cache.

The active released manager implements CPU/GPU transfer, but this review found
**no disk serializer, persistent archive manifest, or memmap-based disk store**
in the active ReKV code. Disk offload is discussed in the paper/README. The
requested hash-checked, atomic NAS block format must therefore be identified as
an adaptation infrastructure extension, not a native ReKV file format.

References: [MemoryUnit construction](https://github.com/Becomebright/ReKV/blob/5a17285d5e1a7488228eb0098b5e586d72a73e3a/model/attention/kv_cache_manager.py#L31),
[frame-block offload](https://github.com/Becomebright/ReKV/blob/5a17285d5e1a7488228eb0098b5e586d72a73e3a/model/attention/kv_cache_manager.py#L569).

## 6. Query-time retrieval

Internal retrieval operates independently per decoder layer. It averages raw
frame keys across the frame's tokens, expands grouped KV heads to query-head
layout, and flattens heads. Question Q projections are averaged across question
tokens and flattened similarly. The released implementation scores their
**unnormalized dot product in float32**. If multiple frames form a retrieval
group, their scores are averaged; top groups are selected and sorted into
temporal order. External frame indices are also supported.

The retrieval-only forward does not append question KV to the persistent video
manager. A subsequent answer forward uses a separate tuple of retrieved KV.
The present archive task must not invoke either retrieval or answering.

References: [query and top-k scoring](https://github.com/Becomebright/ReKV/blob/5a17285d5e1a7488228eb0098b5e586d72a73e3a/model/attention/kv_cache_manager.py#L420),
[float32 dot product](https://github.com/Becomebright/ReKV/blob/5a17285d5e1a7488228eb0098b5e586d72a73e3a/model/attention/kv_cache_manager.py#L172),
[representative key construction](https://github.com/Becomebright/ReKV/blob/5a17285d5e1a7488228eb0098b5e586d72a73e3a/model/attention/kv_cache_manager.py#L600),
[retrieval and answer separation](https://github.com/Becomebright/ReKV/blob/5a17285d5e1a7488228eb0098b5e586d72a73e3a/model/llava_onevision_rekv.py#L36).

## 7. Position and RoPE handling

Native ReKV uses scalar `RotaryEmbeddingESM`. Within a local window, Q/K are
rotated using consecutive relative sequence positions. Distant initial tokens
use the LM-Infinite distance ceiling: the global query uses
`apply_rotary_pos_emb_one_angle(q, n_local)`, whose implementation selects rotary
table index `n_local - 1`. Retrieved raw keys are treated as a newly assembled
consecutive context, rather than retaining original distant positions.

The native patch routes decoder layers through `self.position_bias`; it does
not preserve an arbitrary backbone's native multidimensional position IDs.
Blindly applying it to Qwen2.5-VL would discard mRoPE semantics.

References: [model position dispatch](https://github.com/Becomebright/ReKV/blob/5a17285d5e1a7488228eb0098b5e586d72a73e3a/model/patch.py#L97),
[scalar rotary replacement](https://github.com/Becomebright/ReKV/blob/5a17285d5e1a7488228eb0098b5e586d72a73e3a/model/patch.py#L151),
[rotary equations and fixed-angle index](https://github.com/Becomebright/ReKV/blob/5a17285d5e1a7488228eb0098b5e586d72a73e3a/model/attention/rope.py#L89),
[paper position policy](https://arxiv.org/html/2503.00540v1#S3).

## 8. KV reload

Selected frame blocks are copied from CPU into a reusable GPU context buffer,
preceded by initial prompt K/V. The manager evicts cached GPU blocks using LRU
bookkeeping and synchronizes asynchronous copy events/streams. Retrieval returns
raw K/V; the answer attention path applies rotary positions to the assembled
context. A disk archive adapter must preserve dtype, head grouping, token order,
frame boundaries, and raw-key convention when reconstructing this input.

References: [MemoryUnit reload](https://github.com/Becomebright/ReKV/blob/5a17285d5e1a7488228eb0098b5e586d72a73e3a/model/attention/kv_cache_manager.py#L67),
[context assembly and synchronization](https://github.com/Becomebright/ReKV/blob/5a17285d5e1a7488228eb0098b5e586d72a73e3a/model/attention/kv_cache_manager.py#L343),
[answer attention/cache path](https://github.com/Becomebright/ReKV/blob/5a17285d5e1a7488228eb0098b5e586d72a73e3a/model/attention/rekv_attention.py#L64).

## 9. Training requirement

The paper describes ReKV as training-free. The released adapters load existing
pretrained model weights, patch attention/cache behavior, and switch to eval
mode. There is no new ReKV checkpoint training step in the evaluated mechanism.
Bundled third-party training utilities are not evidence that ReKV needs training.
The controlled adaptation must leave Qwen2.5-VL pretrained parameters unchanged.

References: [paper abstract](https://arxiv.org/html/2503.00540v1#abstract),
[pretrained model load, patch and eval](https://github.com/Becomebright/ReKV/blob/5a17285d5e1a7488228eb0098b5e586d72a73e3a/model/llava_onevision_rekv.py#L128).

## 10. Backbone-dependent components and proposed adaptation boundary

| Component | Native ReKV behavior | Qwen2.5-VL adaptation obligation |
| --- | --- | --- |
| Visual path | Backbone-specific encoder, projector, pooling, tokens/frame | Preserve the existing Qwen checkpoint, processor and post-merger tensor semantics |
| Sequence | Initial chat prefix followed by projected video features | Specify exact vision delimiter/text placement and cached embedding insertion |
| Position | Scalar rotary, local renumbering and sink ceiling | Preserve native mRoPE for the reference; define and test any ReKV-specific position policy explicitly |
| KV | Raw layer K/V, grouped KV-head layout | Capture pre-mRoPE K and V; do not label post-mRoPE DynamicCache dumps native raw KV |
| Attention | Local causal window + initial prompt | Explicitly implement/test the corresponding mask and sink behavior |
| Cache object | Tuple of per-layer `ContextManager` objects | Bridge the installed Transformers cache API without silently changing attention |
| Frame grouping | Fixed token block per native frame | Distinguish visual tokens from image delimiters/text and retain exact block mapping |
| Storage | CPU tensor blocks | Add atomic NAS payload/index/receipt, hashes and continuation state |

The local Qwen checkpoint config was inspected at
`/convei_nas2/zzuhyun/hyunjun/models/Qwen2.5-VL-7B-Instruct/config.json`:
28 decoder layers, hidden size 3584, 28 query heads, 4 KV heads, head dimension
128, `mrope_section=[16,24,24]`, `rope_theta=1000000`, checkpoint dtype bf16,
and default `use_sliding_window=false`. Its configured `sliding_window=32768`
does not enable windowing by itself.

Two fidelity claims must remain separate. Passive KV capture must preserve the
standard Qwen forward, including embeddings, attention mask, multidimensional
positions, KV and logits. ReKV windowing intentionally changes attention once
the local window is exceeded; it cannot be claimed equal to unrestricted Qwen
attention in that regime. Test the windowed adaptation against its explicit
reference, and test untruncated short inputs against standard Qwen.

The existing visual archive is frame-level. Reusing it can be validated against
the corresponding raw **image/frame** path; equality with a temporal video
encoder path cannot be assumed, especially where temporal patching differs.
Preserve and test processor/grid metadata, embeddings and mRoPE positions.

## Temporal parallelism and restart caveat

Native ReKV carries layer KV across chunks. **Resetting cache at arbitrary
temporal shard boundaries changes higher-layer KV**, even when global frame
IDs and timestamps are preserved. A one-window overlap generally does not fix
this, because layerwise attention propagates dependencies through multiple
windows. Independent GPU workers need an explicit correctness strategy before
their output can form one continuous-stream archive:

1. Exact continuation-state handoff or prefix replay; or
2. A proved and tested finite-receptive-field warmup overlap, preserving the
   initial prompt, global multimodal positions and native execution alignment.

The second option is an inference from the causal local attention graph, **not
an official ReKV implementation feature**. A conservative starting bound is
`num_layers * n_local` sequence tokens. With 28 layers, 15K and 676 visual tokens
per frame, this is about 622 frames of warmup, before accounting for special
tokens and execution-block alignment. The final bound and numerical agreement
must be verified against a continuous reference at shard boundaries. Long
balanced shards can amortize overlap cost, but a claim of exact equivalence
requires evidence rather than the bound alone.

Atomic output receipts also do not by themselves make contextual extraction
resumable. Restart must restore the exact per-layer local/sink state and position
cursor, or reproduce it using the verified replay/warmup strategy. A restart
from the first uncommitted frame with an empty context is invalid.

Future causal retrieval must filter candidate blocks by query cutoff **before
top-k selection and payload loading**. Keep global frame IDs, source PTS and
exact global timestamps separate from the positional scheme used for attention.

## Storage calculation before any full run

For the existing 676-token/frame Qwen archive and a 2-byte KV dtype:

```text
28 layers * 2 (K and V) * 4 KV heads * 676 tokens * 128 head_dim * 2 bytes
  = 38,764,544 bytes/frame

38,764,544 * 221,728 frames
  = 8,595,184,812,032 bytes
  = 8.595 TB (decimal), approximately 7.817 TiB
```

This is a **visual-token KV payload lower bound**, not a measured archive size.
Image delimiters/text KV, headers, indices, receipts, prompt/continuation state
and retrieval representatives add storage. FP16 and BF16 have the same byte
count; dtype selection must follow the controlled comparison and fidelity tests.

The native representative-key layout expands GQA to query heads. Retaining
that representation for every layer/frame would add 44,501,696,512 bytes. Compact
four-KV-head means would require 6,357,385,216 bytes and can potentially be expanded
logically at retrieval, but exact scoring equivalence must be tested before using
that representation. No retrieval code or QA evaluation is part of the present
archive-generation stage.
