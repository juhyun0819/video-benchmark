import copy

import torch
from PIL import Image
from day1_compare.run import public_prompt, assemble_inputs, honor_full_attention
from qwen2_5_vl_fluxmem import Qwen2_5_VLProcessor


def test_prompt_whitelist_and_nested_choice():
    q = {'question_id':'x','question':'What happened?',
         'choices':['a','b',['nested','option'],'d','e'],
         'input_aliases':[{'alias':'video 1','global_start_seconds':1,
                           'global_end_seconds':2,'source_path':'PRIVATE_FILE'}],
         'correct_answer':'PRIVATE_GOLD', 'reason':'PRIVATE_REASON'}
    text = public_prompt(q,duration=100,frames=4)
    assert 'PRIVATE_' not in text
    assert '["nested", "option"]' in text
    assert 'global_start_seconds' in text
    clean = copy.deepcopy(q)
    clean.pop('correct_answer')
    clean.pop('reason')
    assert public_prompt(clean,duration=100,frames=4)==text


def test_cached_visual_assembly_equals_native_processor():
    processor = Qwen2_5_VLProcessor.from_pretrained(
        '/home/kai/models/Qwen2.5-VL-7B-Instruct',local_files_only=True)
    frames = [Image.new('RGB',(224,224),(i*20,40,80)) for i in range(4)]
    prompt = 'Which option is correct? A. one B. two'
    messages = [{'role':'system','content':'You are a helpful assistant.'},
                {'role':'user','content':[{'type':'text','text':prompt},{'type':'video'}]}]
    text = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    native = processor(text=[text],videos=[frames],fps=0.25,padding=True,
                       return_tensors='pt',min_pixels=224*224,max_pixels=224*224)
    visual = processor.image_processor(images=None,videos=[frames],return_tensors='pt',
                                       min_pixels=224*224,max_pixels=224*224)
    cached = assemble_inputs(processor,prompt,visual,0.25)
    assert set(native)==set(cached)
    for key in native:
        if isinstance(native[key],torch.Tensor):
            assert torch.equal(native[key],cached[key]), key
        else:
            assert native[key]==cached[key], key
    for frame in frames: frame.close()


def test_full_attention_config_overrides_upstream_unintended_window():
    from types import SimpleNamespace
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter
    config = SimpleNamespace(use_sliding_window=False,sliding_window=32768)
    model = SimpleNamespace(config=config,model=SimpleNamespace(config=config))
    correction = honor_full_attention(model)
    assert correction['checkpoint_sliding_window']==32768
    assert model.model.config.sliding_window is None
    # This is the exact decision helper used by released model SDPA; checking
    # shape-only inputs avoids constructing a 32769-square attention mask.
    embeds = torch.empty(1,32769,1)
    assert AttentionMaskConverter._ignore_causal_mask_sdpa(
        attention_mask=None,inputs_embeds=embeds,past_key_values_length=0,
        sliding_window=model.config.sliding_window,is_training=False)
    assert not AttentionMaskConverter._ignore_causal_mask_sdpa(
        attention_mask=None,inputs_embeds=embeds,past_key_values_length=0,
        sliding_window=32768,is_training=False)
