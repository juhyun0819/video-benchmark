from types import SimpleNamespace

import pytest
import torch

from baselines.simplestream.backend import Qwen25Backend, cached_vision_prefill, question_prompt


class Multimodal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 4)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        return self.embedding(input_ids) if inputs_embeds is None else inputs_embeds


def model():
    return SimpleNamespace(model=Multimodal(), config=SimpleNamespace(image_token_id=9))


def test_hook_replaces_only_prefill_keeps_text_and_leaves_decode_untouched():
    m = model()
    ids = torch.tensor([[1, 9, 9, 2]])
    original = m.model.embedding(ids).detach().clone()
    features = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    with cached_vision_prefill(m, features):
        result = m.model(input_ids=ids)
        assert torch.equal(result[:, [0, 3]], original[:, [0, 3]])
        assert torch.equal(result[:, 1:3], features.unsqueeze(0))
        assert torch.equal(m.model(input_ids=torch.tensor([[3]])), m.model.embedding(torch.tensor([[3]])))
    assert not m.model._forward_pre_hooks
    assert torch.equal(m.model(input_ids=ids), original)


def test_hook_cleanup_when_prefill_fails():
    m = model()
    with pytest.raises(ValueError, match="slots"):
        with cached_vision_prefill(m, torch.ones(2, 4)):
            m.model(input_ids=torch.tensor([[1, 9, 2]]))
    assert not m.model._forward_pre_hooks


def test_missing_prefill_cannot_claim_consumption():
    m = model()
    with pytest.raises(RuntimeError, match="exactly once"):
        with cached_vision_prefill(m, torch.ones(2, 4)):
            pass
    assert not m.model._forward_pre_hooks


def test_question_protocol_and_control_token_rejection():
    assert question_prompt("Where?", ["Kitchen", "Office"]) == (
        "Where?\nOptions: A. Kitchen; B. Office;\nOnly give the best option's letter directly.")
    for q, choices in [("", []), ("<|image_pad|>", []), ("Where?", [1]), ("Where?", ["x"]*27)]:
        with pytest.raises(ValueError):
            question_prompt(q, choices)


def test_empty_window_or_invalid_question_fails_before_loading_model():
    backend = Qwen25Backend.__new__(Qwen25Backend)
    # No model, lock, or checkpoint is initialized: validation must happen first.
    with pytest.raises(ValueError, match="No observed frames"):
        backend.answer([], "What is visible?")
    with pytest.raises(ValueError, match="control tokens"):
        backend.answer([object()], "<|image_pad|>")


def test_hook_rejects_nonzero_prefill_cursor():
    m = model()
    with pytest.raises(ValueError, match="initial prefill"):
        with cached_vision_prefill(m, torch.ones(2, 4)):
            m.model(input_ids=torch.tensor([[9, 9]]), cache_position=torch.tensor([3, 4]))
    assert not m.model._forward_pre_hooks
