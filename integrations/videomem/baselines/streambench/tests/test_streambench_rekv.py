"""Real pinned ReKV retrieval/offload policy; synthetic CPU KV from the encoder boundary."""
from copy import deepcopy
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch


def native_context():
    r = importlib.import_module("baselines.streambench.adapters.rekv")
    policy = r.load_policy()
    rope = policy.RotaryEmbeddingESM.__new__(policy.RotaryEmbeddingESM)
    torch.nn.Module.__init__(rope)
    rope.register_buffer("inv_freq", torch.ones(1), persistent=False)
    rope.base, rope.distance_scale = 10000, 1
    rope._seq_len_cached, rope._cos_cached, rope._sin_cached = 4, torch.ones(4, 2), torch.zeros(4, 2)
    c = policy.ContextManager(rope, 1, 4, 2, 2, 1, 1, 2, fattn=False,
                              async_global_stream=False, pin_memory=False)
    c.initialized = True
    c.batch_size = c.num_units = c.num_heads = c.num_heads_kv = c.unit_size = c.unit_size_kv = 1
    c.dim_head = 2
    c.dtype = torch.float32
    c.global_blocks, c.cached_blocks, c.block_k = [[]], [{}], [policy.VectorTensor(2, torch.float32, "cpu")]
    c.num_global_block = 0
    c.local_k = torch.zeros(1, 1, 0, 2)
    c.local_v = torch.zeros(1, 1, 0, 2)
    c.global_remainder = (torch.zeros(1, 1, 0, 2), torch.zeros(1, 1, 0, 2))
    c.init_k = torch.ones(1, 1, 1, 2)
    c.init_v = torch.ones(1, 1, 1, 2)
    c.init_exc = True
    c.global_buffer = torch.zeros(2, 1, 1, 3, 2)
    c.global_buffer_init_st, c.global_buffer_init_ed = 0, 1
    c._global_remainder_st = c._global_remainder_ed = 0
    cache = policy.CudaCache.__new__(policy.CudaCache)
    cache.num_units, cache.unit_size, cache.dtype = 2, 8, torch.float32
    cache.data, cache.idle_set = torch.zeros(2, 8), {0, 1}
    c.cuda_cache = cache
    return c


def append_native(c, value):
    key = torch.tensor([value, 1.0, value, 1.0]).reshape(1, 1, 2, 2)
    c.global_remainder = (key, key + 100)
    c._global_remainder_st, c._global_remainder_ed = 0, 2
    c._append_global()  # Unchanged upstream offloading and representative keys.
    c.local_k = torch.cat((c.local_k, key), dim=-2)[:, :, -4:]
    c.local_v = torch.cat((c.local_v, key + 100), dim=-2)[:, :, -4:]
    c.global_remainder = tuple(t[:, :, 2:] for t in c.global_remainder)
    c.length += 2


def test_pinned_policy_verifies_and_retrieval_survives_complete_graph_roundtrip():
    r = importlib.import_module("baselines.streambench.adapters.rekv")
    c = native_context()
    for value in (8., 2., 4.):
        append_native(c, value)
    query = torch.tensor([[[[1., 0.]]]])
    assert c._calc_block_topk(query) == [[0]]
    # A loaded block aliases a slot of the preallocated native GPU cache.
    block = c.global_blocks[0][0]
    block.gpu_data = c.cuda_cache.data[0].view(2, 1, 2, 2)
    block.gpu_data_id = 0
    c.cuda_cache.idle_set = {1}
    state = r.encode_cache_graph((c, c.position_embedding))
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    other, rope = r.decode_cache_graph(torch.load(buffer, weights_only=False), device="cpu")
    assert other.position_embedding is rope
    assert other.global_blocks[0][0].cache is other.cuda_cache
    assert other.global_blocks[0][0].gpu_data.untyped_storage().data_ptr() == other.cuda_cache.data.untyped_storage().data_ptr()
    assert other._calc_block_topk(query) == [[0]]
    append_native(c, 12.)
    append_native(other, 12.)
    assert c._calc_block_topk(query) == other._calc_block_topk(query) == [[3]]
    assert c.num_global_block == other.num_global_block == 4
    torch.testing.assert_close(c.block_k[0].get_data(), other.block_k[0].get_data(), rtol=0, atol=0)


def test_graph_codec_rejects_unknown_persistent_objects():
    r = importlib.import_module("baselines.streambench.adapters.rekv")
    c = native_context()
    c.unsupported_future_state = object()
    with pytest.raises(TypeError, match="persistent"):
        r.encode_cache_graph((c,))


class EncoderBoundary:
    def __init__(self):
        self.kv_cache = None
        self.init_prompt_ids = torch.tensor([[1]])

    def encode_init_prompt(self):
        self.kv_cache = (native_context(),)

    def encode_video(self, video, encode_chunk_size):
        for frame in video:
            append_native(self.kv_cache[0], float(frame[0, 0, 0]))

    def get_prompt(self, question, mc=False):
        return question

    def question_answering(self, input_text, max_new_tokens=128):
        c = self.kv_cache[0]
        chosen = c._calc_block_topk(torch.tensor([[[[1., 0.]]]]))
        c.load_count += 1
        c.set_retrieved_block_indices(chosen)
        c.global_buffer.fill_(-90)
        c.to_retrieve = True
        return str(chosen)


def method(budget=100_000):
    r = importlib.import_module("baselines.streambench.adapters.rekv")
    result = r.ReKVMemory({"method": "rekv", "memory_budget_bytes": budget,
                          "synthetic_test_mode": True, "encoder_id": "cpu-kv-boundary"},
                         model=EncoderBoundary())
    result.initialize()
    return result


def inputs(start, count, day=1):
    frames = [np.full((2, 2, 3), index, dtype=np.uint8) for index in range(start, start + count)]
    records = [{"stream_id": "subject", "day": day, "global_frame_id": i,
                "source_timestamp": float(i), "global_timestamp": float(i),
                "source_id": "clip", "source_frame_index": i} for i in range(start, start + count)]
    return frames, records


def test_rekv_checkpoint_continuation_and_query_restoration():
    m = method()
    m.observe(*inputs(0, 3))
    original = deepcopy(m.state_dict())
    old_context = m._model.kv_cache[0]
    answer1 = m.answer("question")
    answer2 = m.answer("question")
    assert answer1["answer"] == answer2["answer"] == "[[2]]"
    assert answer1["query_read_bytes"] > 0
    assert m._model.kv_cache[0].load_count == 0
    assert not m._model.kv_cache[0].to_retrieve
    assert not m._model.kv_cache[0].global_buffer.any()
    assert "get_retrieved_kv" not in vars(old_context)
    resumed = method()
    resumed.load_state_dict(original)
    for item in (m, resumed):
        item.observe(*inputs(3, 4, day=2))
    assert m.answer("later")["answer"] == resumed.answer("later")["answer"] == "[[6]]"
    assert m.memory_bytes() == resumed.memory_bytes()
    assert m._model.kv_cache[0].num_global_block == 7


def test_rekv_counts_cpu_archive_and_fixed_cache_and_rejects_overflow():
    from baselines.streambench.method import UnsupportedBudget
    m = method()
    initial = m.memory_bytes()["kv_memory_bytes"]
    m.observe(*inputs(0, 3))
    assert m.memory_bytes()["kv_memory_bytes"] > initial
    m.memory_budget_bytes = m.memory_bytes()["persistent_total_bytes"]
    count = m._model.kv_cache[0].num_global_block
    with pytest.raises(UnsupportedBudget):
        m.observe(*inputs(3, 1))
    assert m._model.kv_cache[0].num_global_block == count


def test_real_rekv_requires_local_pinned_weights_and_does_not_fall_back():
    r = importlib.import_module("baselines.streambench.adapters.rekv")
    m = r.create_method({"method": "rekv", "memory_budget_bytes": 1_000_000})
    with pytest.raises((ValueError, FileNotFoundError), match="model_path"):
        m.initialize()


def test_native_fixed_pool_floor_rejects_small_budget_before_loading_weights(tmp_path):
    from baselines.streambench.method import UnsupportedBudget
    r = importlib.import_module("baselines.streambench.adapters.rekv")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llava_onevision", "text_config": {
        "model_type": "qwen2", "hidden_size": 3584, "num_attention_heads": 28,
        "num_key_value_heads": 4, "num_hidden_layers": 28}}))
    m = r.create_method({"method": "rekv", "memory_budget_bytes": 1024**3,
                         "model_path": str(tmp_path)})
    with pytest.raises(UnsupportedBudget) as caught:
        m.initialize()
    assert caught.value.minimum_memory_bytes == 2_157_969_408


def test_rekv_query_exception_restores_native_lru_and_modes():
    m = method()
    m.observe(*inputs(0, 2))
    def fail(*args, **kwargs):
        m._model.kv_cache[0].to_retrieve = True
        m._model.kv_cache[0].load_count = 900
        raise RuntimeError("decode failure")
    m._model.question_answering = fail
    with pytest.raises(RuntimeError, match="decode failure"):
        m.answer("question")
    assert m._model.kv_cache[0].load_count == 0
    assert not m._model.kv_cache[0].to_retrieve


def test_rekv_fresh_process_resume_preserves_native_archive_and_retrieval(tmp_path):
    m = method()
    m.observe(*inputs(0, 3))
    torch.save(m.state_dict(), tmp_path / "input.pt")
    script = """
import runpy,sys,torch
t=runpy.run_path('baselines/streambench/tests/test_streambench_rekv.py')
m=t['method']()
m.load_state_dict(torch.load(sys.argv[1],weights_only=False))
m.observe(*t['inputs'](3,4,day=2))
torch.save({'state':m.state_dict(),'answer':m.answer('question')},sys.argv[2])
"""
    subprocess.run([sys.executable, "-c", script, str(tmp_path / "input.pt"), str(tmp_path / "output.pt")],
                   check=True, cwd=Path(__file__).resolve().parents[3])
    m.observe(*inputs(3, 4, day=2))
    resumed = torch.load(tmp_path / "output.pt", weights_only=False)
    assert resumed["answer"] == m.answer("question")
    other = method()
    other.load_state_dict(resumed["state"])
    assert m.memory_bytes() == other.memory_bytes()
    assert m._model.kv_cache[0].length == other._model.kv_cache[0].length


def test_rekv_preoffload_retrieval_reports_full_live_kv_upper_bound():
    m = method()
    m.observe(*inputs(0, 3))
    c = m._model.kv_cache[0]
    context_keys = torch.cat([block.cpu_data[0] for block in c.global_blocks[0]], dim=1).unsqueeze(0)
    context_values = context_keys + 100
    c.global_remainder = (torch.cat((c.init_k, context_keys), dim=2),
                          torch.cat((c.init_v, context_values), dim=2))
    c.global_blocks, c.num_global_block, c.init_exc = [[]], 0, False
    c.block_k[0].length = 0
    before = m.memory_bytes()["kv_memory_bytes"]
    result = m.answer("question")
    assert result["answer"] == "[[2]]"
    assert result["query_read_bytes_is_upper_bound"] is True
    assert result["query_read_bytes"] == before
    assert result["query_selected_kv_bytes"] > 0
    assert result["query_representative_scan_bytes"] >= context_keys.numel() * context_keys.element_size()
