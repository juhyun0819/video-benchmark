# SimpleStream upstream review and adaptation boundary

Reviewed 2026-11-12. This document separates the paper's memory policy, the
released implementation, and the corrected Qwen2.5-VL implementation used in
this benchmark. It does not claim reproduced paper scores. Executed GPU fidelity
results and their artifact path are recorded in [README.md](README.md).

## Source provenance

- Paper: [A Simple Baseline for Streaming Video Understanding, arXiv:2604.02317v1](https://arxiv.org/html/2604.02317v1).
- Official repository: [EvolvingLMMs-Lab/SimpleStream](https://github.com/EvolvingLMMs-Lab/SimpleStream).
- Reviewed commit: [`be227ee8ce90a258b286a34161688c2a59eeeadb`](https://github.com/EvolvingLMMs-Lab/SimpleStream/commit/be227ee8ce90a258b286a34161688c2a59eeeadb),
  dated 2026-05-01, `chore: align release evaluation setup`.
- Full, unmodified shallow checkout used for inspection:
  `/tmp/simplestream-upstream`. Its working tree was clean after inspection.
- Upstream specifies [Transformers 4.52.4 for Qwen2.5](https://github.com/EvolvingLMMs-Lab/SimpleStream/blob/be227ee8ce90a258b286a34161688c2a59eeeadb/requirements-qwen25.txt#L1-L5),
  [4.57.6 for Qwen3](https://github.com/EvolvingLMMs-Lab/SimpleStream/blob/be227ee8ce90a258b286a34161688c2a59eeeadb/requirements-qwen3.txt#L1-L5),
  and [qwen-vl-utils 0.0.14](https://github.com/EvolvingLMMs-Lab/SimpleStream/blob/be227ee8ce90a258b286a34161688c2a59eeeadb/requirements.txt#L17-L18).
- No repository-wide LICENSE/COPYING file or README license declaration was
  present in the reviewed checkout. This review links upstream; its source must
  not be vendored as though a repository-wide redistribution license had been
  established. No upstream source was copied into this baseline directory.

## Meaning of 8B, 4B, and frame windows

SimpleStream applies a recent-N-observed-frames policy to an existing VLM. It adds
no learned memory module or extra training. Frames outside the window are
discarded. The paper uses 1 FPS, with windows 2/4/8 in its main comparison and
2/4/8/16 in its scale ablation. Its Qwen3-VL-8B row peaks at four frames; the
Qwen3-VL-4B row peaks at sixteen. These are benchmark observations, not universal
optimal windows. [Paper, Sections 3.2–4.3 and Table 2](https://arxiv.org/html/2604.02317v1#S3.SS2)

| Requested/native label | Underlying published checkpoint | Window is a separate choice |
| --- | --- | --- |
| SimpleStream-8B | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | Paper's best reported OVO mean: 4 frames |
| SimpleStream-4B | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | Paper's best reported OVO mean: 16 frames |
| Controlled Qwen2.5 baseline | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | Name the actual N explicitly |

`B` denotes model size; `f` denotes frame count. Running Qwen2.5-VL-7B with four
or eight frames does not reproduce a Qwen3-VL-4B or Qwen3-VL-8B checkpoint.
Upstream explicitly lists Qwen3-VL-8B as primary and Qwen2.5-VL-7B for
cross-validation, and loads these checkpoints directly through Transformers.
There is no additional SimpleStream fine-tuned checkpoint in this released
loading path. [Model list](https://github.com/EvolvingLMMs-Lab/SimpleStream/blob/be227ee8ce90a258b286a34161688c2a59eeeadb/README.md#L73-L77),
[Qwen2.5 loading](https://github.com/EvolvingLMMs-Lab/SimpleStream/blob/be227ee8ce90a258b286a34161688c2a59eeeadb/lib/recent_window_eval.py#L78-L119),
[Qwen3 loading](https://github.com/EvolvingLMMs-Lab/SimpleStream/blob/be227ee8ce90a258b286a34161688c2a59eeeadb/lib/recent_window_eval_qwen3.py#L32-L63).

## Released observation and inference path

The Qwen2.5 OVO wrapper preprocesses frames as independent images and obtains
image features. It retains a bounded deque of encoded chunks, concatenates the
retained embeddings and grids, and starts a new answer generation. The retained
objects are vision features, not a persistent historical language-model KV
state. No question-conditioned retrieval is used by this path.
[Encoding](https://github.com/EvolvingLMMs-Lab/SimpleStream/blob/be227ee8ce90a258b286a34161688c2a59eeeadb/lib/recent_window_eval_qwen25.py#L41-L67),
[window and query](https://github.com/EvolvingLMMs-Lab/SimpleStream/blob/be227ee8ce90a258b286a34161688c2a59eeeadb/lib/recent_window_eval_qwen25.py#L148-L202).

The released StreamingBench script rejects nonzero retrieval `top_k`.
[Source](https://github.com/EvolvingLMMs-Lab/SimpleStream/blob/be227ee8ce90a258b286a34161688c2a59eeeadb/main_experiments/eval_streamingbench.py#L136-L150).

Sampling fidelity needs separate attention. The exact recent decoder constructs
the full Qwen sampling indices with `linspace(...).round()` and takes their
tail. The general decoder groups decoded timestamps into duration buckets. Its
`recent_frames_only` limit is operationally a chunk limit; one chunk need not
always contain exactly one frame. Reusing this benchmark's fixed global 1 FPS
archive therefore preserves a declared causal recent-frame policy but does not
by itself prove identical frame selection to every upstream video decoder run.
[Exact sampling](https://github.com/EvolvingLMMs-Lab/SimpleStream/blob/be227ee8ce90a258b286a34161688c2a59eeeadb/lib/qwen_exact_recent_decoder.py#L66-L83),
[chunk construction](https://github.com/EvolvingLMMs-Lab/SimpleStream/blob/be227ee8ce90a258b286a34161688c2a59eeeadb/lib/recent_window_eval.py#L453-L533).

## Released cached-prefix fidelity findings

### Qwen2.5 multiple-image boundary mismatch

The subclass constructs one vision-start/end pair around all frames' image
tokens, then passes all per-image grid rows to `get_rope_index`. The base-class
docstring claiming per-frame blocks disagrees with its implementation.
[Actual subclass input construction](https://github.com/EvolvingLMMs-Lab/SimpleStream/blob/be227ee8ce90a258b286a34161688c2a59eeeadb/lib/recent_window_eval_qwen25.py#L82-L128),
[base docstring](https://github.com/EvolvingLMMs-Lab/SimpleStream/blob/be227ee8ce90a258b286a34161688c2a59eeeadb/lib/recent_window_eval.py#L64-L69).

The pinned Transformers implementation counts images from vision-start markers.
With one such marker, it consumes only the first image grid and assigns ordinary
one-dimensional positions to the remaining tokens. Unconsumed grid rows are not
rejected. [Transformers 4.52.4 position calculation](https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1484-L1564).

A CPU-only test executed that exact upstream function without loading model
weights. With four image grids `[1,52,52]` (676 merged tokens each), all 2,028
visual tokens after the first image had different position columns from the
per-frame-delimited construction. In the test's short synthetic text prefix,
the first question token had scalar position 2,059 versus 115. At eight frames,
4,732 visual tokens differed. These are position-construction results, not
measured accuracy changes.

### Explicit positions are discarded by the pinned generation entry point

There is an additional issue before treating those explicit positions as the
effective generation positions. The released wrapper passes only
`inputs_embeds`, `attention_mask`, and `position_ids` to `generate`; it does not
forward the original token IDs or grids. Transformers 4.52.4 then unconditionally
replaces prepared `position_ids` with `None`. Its prefill recomputes positions;
without IDs/grids, the all-ones mask yields ordinary sequential positions for
the entire prompt. Thus the source-level effective-path conclusion is stronger
than the first-image-only error in the intermediate calculation. No full-model
run was performed in this review. [Wrapper call](https://github.com/EvolvingLMMs-Lab/SimpleStream/blob/be227ee8ce90a258b286a34161688c2a59eeeadb/lib/recent_window_eval.py#L206-L216),
[discarded positions](https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1946-L1987),
[prefill recomputation](https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1697-L1712).

The Qwen3 wrapper correctly emits separate per-frame vision boundaries, but its
pinned 4.57.6 generation method also discards explicit positions. The same
missing-ID/grid cached call therefore has a corresponding position issue.
Additionally, Qwen3's image extractor returns DeepStack features alongside the
merged image features. The released flattening helper retains only the merged
features; its cached generation does not inject the DeepStack features used by
the native pixel-input path. These findings concern the released wrapper, not
the underlying pretrained checkpoint.
[Qwen3 boundary construction and generation call](https://github.com/EvolvingLMMs-Lab/SimpleStream/blob/be227ee8ce90a258b286a34161688c2a59eeeadb/lib/recent_window_eval_qwen3.py#L206-L251),
[flattening helper](https://github.com/EvolvingLMMs-Lab/SimpleStream/blob/be227ee8ce90a258b286a34161688c2a59eeeadb/lib/recent_window_eval.py#L171-L182),
[Qwen3 DeepStack return](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L1050-L1064),
[native DeepStack input handling](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L1134-L1175),
[Qwen3 generation preparation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L1390-L1408).

The separate upstream efficiency script follows another path: processor-built
inputs, explicit prefill, and explicit `rope_deltas` before autoregressive
decoding. Its mechanics should not be conflated with the cached OVO wrapper.
[Efficiency path](https://github.com/EvolvingLMMs-Lab/SimpleStream/blob/be227ee8ce90a258b286a34161688c2a59eeeadb/efficiency/eval_efficiency.py#L273-L328).

## Controlled Qwen2.5 adaptation decision

The benchmark implementation should preserve causal recent-N selection while
using native multi-image mRoPE and ordinary Hugging Face generation. This is a
**corrected Qwen2.5-VL-7B adaptation of the paper's policy**, not a literal
reproduction of the released cached-prefix behavior, and not a Qwen3 8B/4B run.

The proposed cache reuse boundary is the post-merger image embedding. It does
not reuse ReKV's contextualized keys/values. For each question:

1. Select only observed frames at or before the query cutoff; retain the newest
   N in chronological order. Record their exact timestamps and source IDs.
2. Build native per-image vision blocks with matching image grids and the
   declared chat template. Pass `input_ids`, `attention_mask`, and
   `image_grid_thw` to a fresh `generate` call.
3. A scoped prefill hook on `model.model` substitutes the cached image features
   into the ordinary token embeddings. Retain input IDs and native position
   metadata; pass no dummy pixels. Leave subsequent decode calls unchanged.
4. Use a new KV cache for each question. Remove the hook and clear `rope_deltas`
   on success or failure; do not retain prior answer state.

This hook boundary is compatible with the inspected Transformers 4.57.3
Qwen2.5 model: it accepts token IDs together with supplied embeddings; it only
embeds IDs when embeddings are absent, and skips visual extraction when pixels
are absent. Native generation computes position metadata from the retained IDs
and grids before entering that model. The language model itself receives only
embeddings. [Multimodal forward](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1253-L1316),
[native generation positions](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1534-L1577).

The initial implementation should restrict batch size, beams, and return
sequences to one; require caching; and assert exactly one prefill substitution
at cache position zero. Check placeholder count, grid-derived token counts,
feature shape/dtype, and finite values before generation. Do not infer prefill
only from the presence of image-token IDs, because generated tokens can contain
special IDs. An exception-safe hook scope is essential because hooks and
`rope_deltas` are mutable model state; concurrent answers on one model instance
must be serialized.

## Fidelity evidence required from implementation

Raw-pixel native generation and cached-feature generation must use the same
checkpoint revision, decoded RGB pixels, resize/normalization rules, grid,
tokenizer/template, attention implementation, generation settings, and selected
frames. Merely sharing hidden dimension or model family does not establish
compatibility. The existing Qwen2.5 archive contains frame-as-image features;
it is not interchangeable with native temporally paired video features or any
Qwen3 features.

Separate two tests: first, cache features from the same visual forward as the
raw reference to isolate the substitution boundary; second, compare actual
archive features and end-to-end generation against the declared image path.
Different vision batching/kernel shapes can change BF16 rounding, even when
images are mathematically independent. Report exact equality and numerical
error separately. Compare prompt IDs, all position axes, prefill logits,
generated IDs, and a subsequent independent question. Include different window
sizes, an early query with fewer than N frames, and a failure/cleanup case.

This document records source inspection and the CPU position test only.
Implementation fidelity reports must supply their own measured results. No
GPU workload, model download, dependency change, or modification to the running
ReKV extraction was performed for this review.
