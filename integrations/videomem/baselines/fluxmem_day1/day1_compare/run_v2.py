"""Separate offline-QA protocol arm: official video-then-question ordering.

Version 1 used the README caption example's text-then-video layout and is kept
unchanged. This arm matches evaluation/lmms-eval/lmms_eval/models/
qwen2_5_vl_fluxmem.py:218. Frames, questions, FPS, decoding, and memory sizes are
unchanged; neither labels nor earlier predictions are read.
"""
from day1_compare import run


def assemble_inputs(processor, prompt, visual, fps):
    messages = [{'role':'system','content':'You are a helpful assistant.'},
                {'role':'user','content':[{'type':'video'},{'type':'text','text':prompt}]}]
    text = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    if text.count(processor.video_token)!=1 or tuple(visual['video_grid_thw'].shape)!=(1,3):
        raise ValueError('Expected exactly one chronological video')
    count = int(visual['video_grid_thw'][0].prod()) // processor.image_processor.merge_size**2
    text = text.replace(processor.video_token,processor.video_token*count,1)
    inputs = dict(processor.tokenizer([text],padding=True,return_tensors='pt'))
    inputs.update(visual)
    inputs['second_per_grid_ts']=[processor.image_processor.temporal_patch_size/fps]
    return inputs


if __name__=='__main__':
    run.assemble_inputs = assemble_inputs
    run.main()
