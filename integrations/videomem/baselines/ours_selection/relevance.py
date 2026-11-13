"""Explicit local text model for VideoTree's 1/2/3 caption relevance calls."""
import json


class LocalRelevanceScorer:
    def __init__(self, config):
        from baselines.streambench.adapters.videotree import LocalArtifacts
        self.config=dict(config)
        self.artifacts=LocalArtifacts(config['model_path'],config['model_revision'],required=('config.json',),weights=True)
        self.identity={'implementation':'local_text_causal_lm_json_relevance','config':self.config,'artifacts':self.artifacts.identity}
        self.model=self.tokenizer=None
        self.calls=[]

    def initialize(self):
        """Load local weights before collecting per-query latency."""
        self._load()

    def _load(self):
        self.artifacts.assert_unchanged()
        if self.model is None:
            import torch
            from transformers import AutoTokenizer,AutoModelForCausalLM
            c=self.config;dtype=c.get('dtype','bfloat16')
            if dtype not in ('float16','bfloat16','float32'):
                raise ValueError('Unsupported relevance model dtype')
            kw={'revision':c['model_revision'],'local_files_only':True}
            self.tokenizer=AutoTokenizer.from_pretrained(c['model_path'],**kw)
            self.model=AutoModelForCausalLM.from_pretrained(c['model_path'],torch_dtype=getattr(torch,dtype),
                    device_map={'':c.get('device','cuda:0')},**kw).eval()

    def score(self, question, choices, captions):
        import torch
        self._load()
        prompt=json.dumps({'question':question,'choices':choices,'captions':captions},ensure_ascii=False)
        instruction="Score each caption's relevance to answering the question: 1 low, 2 medium, 3 high. Return only a JSON list of integers in caption order."
        inputs=self.tokenizer.apply_chat_template([{'role':'system','content':instruction},{'role':'user','content':prompt}],
                    add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt',enable_thinking=False)
        inputs={k:v.to(self.model.device) for k,v in inputs.items()}
        with torch.inference_mode():
            generated=self.model.generate(**inputs,max_new_tokens=int(self.config.get('max_new_tokens',512)),do_sample=False,num_beams=1)
        tail=generated[0,inputs['input_ids'].shape[1]:]
        raw=self.tokenizer.decode(tail,skip_special_tokens=True)
        self.calls.append({'question':question,'captions':list(captions),'raw_output':raw})
        values=json.loads(raw)
        if not isinstance(values,list) or len(values)!=len(captions) or any(type(x) is not int or x not in (1,2,3) for x in values):
            raise ValueError('Relevance model must return one JSON integer 1..3 per caption')
        return values
