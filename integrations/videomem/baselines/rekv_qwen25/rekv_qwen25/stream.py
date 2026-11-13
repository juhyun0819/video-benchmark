"""Fixed frame-as-image stream protocol with absolute Qwen mRoPE positions."""
from __future__ import annotations

import torch

from .smoke import PREFIX


class StreamingInputs:
    """Reuse native Qwen position construction, without retaining a long prompt.

    Frames contain 676 spatially merged image tokens and two image boundaries.
    Each image is independent, matching the existing baseline visual archive.
    The fixed open observation prefix occurs once, at global frame zero.
    """

    def __init__(self, model, processor):
        self.model = model
        self.prefix = processor.tokenizer(PREFIX, add_special_tokens=False).input_ids
        self.n_init = len(self.prefix)
        self.block = [model.config.vision_start_token_id] + [model.config.image_token_id] * 676 + [model.config.vision_end_token_id]
        ids = torch.tensor([self.prefix + self.block * 2], device=model.device)
        grid = torch.tensor([[1, 52, 52]] * 2, device=model.device)
        positions, _ = model.model.get_rope_index(ids, image_grid_thw=grid, attention_mask=torch.ones_like(ids))
        self.prefix_positions = positions[:, :, :self.n_init].clone()
        self.frame_positions = positions[:, :, self.n_init:self.n_init + 678].clone()
        difference = positions[:, :, self.n_init + 678:] - self.frame_positions
        if not torch.all(difference == difference.flatten()[0]):
            raise ValueError('Native mRoPE image blocks do not have a common scalar stride')
        self.stride = int(difference.flatten()[0])
        if self.stride != 28:
            raise ValueError('Unexpected native mRoPE stride for the fixed 26x26 image grid')

    def prefix_inputs(self):
        ids = torch.tensor([self.prefix], device=self.model.device)
        return dict(input_ids=ids, position_ids=self.prefix_positions,
                    inputs_embeds=self.model.get_input_embeddings()(ids))

    def build(self, features, start_frame, *, include_prefix=False):
        if type(start_frame) is not int or start_frame < 0:
            raise ValueError('start_frame must be a nonnegative global frame ID')
        if features.ndim != 3 or tuple(features.shape[1:]) != (676, 3584) or features.shape[0] < 1:
            raise ValueError('Expected [frames,676,3584] visual features')
        if features.dtype != torch.bfloat16:
            raise ValueError('Visual archive protocol requires bfloat16')
        count = features.shape[0]
        if include_prefix and start_frame != 0:
            raise ValueError('The fixed prefix can occur only at frame zero')
        prefix = self.prefix if include_prefix else []
        ids = torch.tensor([prefix + self.block * count], device=self.model.device)
        positions = [self.frame_positions + (start_frame + i) * self.stride for i in range(count)]
        if include_prefix:
            positions.insert(0, self.prefix_positions)
        positions = torch.cat(positions, dim=-1)
        embeddings = self.model.get_input_embeddings()(ids)
        embeddings = embeddings.masked_scatter(
            (ids == self.model.config.image_token_id).unsqueeze(-1),
            features.to(self.model.device).reshape(-1),
        )
        return dict(input_ids=ids, position_ids=positions, inputs_embeds=embeddings)


def observe(model, session, inputs):
    """One state-preserving observation call; no question or retrieval."""
    session.set_positions(inputs['position_ids'])
    result = model.model.language_model(
        inputs_embeds=inputs['inputs_embeds'], position_ids=inputs['position_ids'],
        attention_mask={'full_attention': None}, use_cache=False, return_dict=True,
    )
    return model.lm_head(result.last_hidden_state[:, -1:])
