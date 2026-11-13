"""Exact shipped EgoRAG caption call, with external aligned media transport."""
import copy
import re

CAPTION_PROMPT = ('Imagine you are the character in the video, describing from a first-person '
                  'perspective what you saw, and everything that happened over time. Use I as the subject.')


def prompt_ids(tokenizer,template,image_token,speech_token):
    conv=copy.deepcopy(template)
    conv.append_message(conv.roles[0],'<image>\n<speech>\n\n'+CAPTION_PROMPT)
    conv.append_message(conv.roles[1],None)
    prompt=conv.get_prompt()
    parts=re.split('(<image>|<speech>)',prompt)
    ids=[]
    for part in parts:
        if not part: continue
        if part=='<image>': ids.append(image_token)
        elif part=='<speech>': ids.append(speech_token)
        else: ids.extend(tokenizer(part).input_ids)
    return ids,prompt


def caption(tokenizer,model,video,audio):
    import numpy as np
    import torch
    import whisper
    from egogpt.constants import IMAGE_TOKEN_INDEX,SPEECH_TOKEN_INDEX
    from egogpt.conversation import conv_templates
    speech=whisper.pad_or_trim(audio.astype(np.float32))
    speech=whisper.log_mel_spectrogram(speech,n_mels=128).permute(1,0)
    speech_lengths=torch.LongTensor([speech.shape[0]])
    speech=torch.stack([speech]).to('cuda').half()
    ids,prompt=prompt_ids(tokenizer,conv_templates['qwen_1_5'],IMAGE_TOKEN_INDEX,SPEECH_TOKEN_INDEX)
    input_ids=torch.tensor(ids,dtype=torch.long).unsqueeze(0).to('cuda')
    processor=model.get_vision_tower().image_processor
    pixels=processor.preprocess(video,return_tensors='pt')['pixel_values']
    # The upstream generate method already applies its own torch.no_grad().
    cont=model.generate(input_ids,images=[pixels.half()],image_sizes=[video[0].size],
                        speech=speech,speech_lengths=speech_lengths,
                        do_sample=False,temperature=0.5,max_new_tokens=4096,
                        modalities=['video'],eos_token_id=tokenizer.eos_token_id)
    tokens=cont[0].tolist()
    return {'caption':tokenizer.batch_decode(cont,skip_special_tokens=True)[0],
            'generated_token_ids':tokens,'eos_token_id':tokenizer.eos_token_id,
            'prompt':prompt,'input_token_ids':ids,'pixel_shape':list(pixels.shape),
            'speech_shape':list(speech.shape),'speech_lengths':speech_lengths.tolist(),
            'model_dtype':str(next(model.parameters()).dtype),'speech_dtype':str(speech.dtype)}
