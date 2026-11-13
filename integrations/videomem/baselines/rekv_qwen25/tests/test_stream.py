from types import SimpleNamespace

import torch
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

from rekv_qwen25.stream import StreamingInputs


class TinyInputModel:
    def __init__(self):
        self.device = torch.device('cpu')
        self.config = SimpleNamespace(image_token_id=19, video_token_id=20,
            vision_start_token_id=18, vision_end_token_id=17,
            vision_config=SimpleNamespace(spatial_merge_size=2,tokens_per_second=2))
        self.model = self
        self.embedding = torch.nn.Embedding(21, 3584).to(torch.bfloat16)

    get_rope_index = Qwen2_5_VLModel.get_rope_index

    def get_input_embeddings(self):
        return self.embedding


def test_absolute_stream_positions_match_native_across_many_chunks():
    model = TinyInputModel()
    processor = SimpleNamespace(tokenizer=lambda *a, **kw: SimpleNamespace(input_ids=[1] * 17))
    stream = StreamingInputs(model, processor)
    features = torch.arange(30 * 676 * 3584, dtype=torch.float32).reshape(30, 676, 3584).to(torch.bfloat16)
    full = stream.build(features, 0, include_prefix=True)
    ids = full['input_ids']
    native, _ = model.get_rope_index(ids, image_grid_thw=torch.tensor([[1,52,52]] * 30),
                                    attention_mask=torch.ones_like(ids))
    assert torch.equal(full['position_ids'], native)
    parts = [stream.prefix_inputs()] + [stream.build(features[start:stop], start) for start,stop in [(0,1),(1,5),(5,13),(13,29),(29,30)]]
    for key in ['input_ids', 'position_ids']:
        assert torch.equal(torch.cat([p[key] for p in parts], dim=-1), full[key])
    assert torch.equal(torch.cat([p['inputs_embeds'] for p in parts], dim=1), full['inputs_embeds'])
    assert int(stream.build(features[:1], 200000)['position_ids'].min()) == 17 + 28 * 200000
