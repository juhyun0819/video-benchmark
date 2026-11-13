from types import SimpleNamespace
from baselines.egogpt_caption.native import prompt_ids,CAPTION_PROMPT


def test_caption_prompt_uses_original_chat_system_and_two_modalities():
    class Template:
        roles=['user','assistant']
        def __init__(self): self.rows=[]
        def append_message(self,role,text): self.rows.append((role,text))
        def get_prompt(self): return 'You are a helpful assistant.\n'+self.rows[0][1]
    template=Template()
    def tokenizer(text): return SimpleNamespace(input_ids=list(text.encode()))
    ids,prompt=prompt_ids(tokenizer,template,-200,-300)
    assert ids.count(-200)==ids.count(-300)==1
    assert prompt=='You are a helpful assistant.\n<image>\n<speech>\n\n'+CAPTION_PROMPT
    assert not template.rows
