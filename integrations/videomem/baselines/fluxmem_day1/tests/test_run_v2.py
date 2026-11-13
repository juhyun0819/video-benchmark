import torch
from PIL import Image
from day1_compare.run_v2 import assemble_inputs
from qwen2_5_vl_fluxmem import Qwen2_5_VLProcessor


def test_offline_question_follows_video_and_matches_native_processor():
    p=Qwen2_5_VLProcessor.from_pretrained('/home/kai/models/Qwen2.5-VL-7B-Instruct',local_files_only=True)
    images=[Image.new('RGB',(224,224),(i*20,50,90)) for i in range(4)]
    question='Which option is correct? A. one B. two'
    messages=[{'role':'system','content':'You are a helpful assistant.'},
              {'role':'user','content':[{'type':'video'},{'type':'text','text':question}]}]
    text=p.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    assert text.index(p.video_token)<text.index(question)
    native=p(text=[text],videos=[images],fps=.25,padding=True,return_tensors='pt',
             min_pixels=224*224,max_pixels=224*224)
    visual=p.image_processor(images=None,videos=[images],return_tensors='pt',
                             min_pixels=224*224,max_pixels=224*224)
    ours=assemble_inputs(p,question,visual,.25)
    assert set(native)==set(ours)
    for key in ours:
        if isinstance(ours[key],torch.Tensor): assert torch.equal(ours[key],native[key])
        else: assert ours[key]==native[key]
    for image in images: image.close()
