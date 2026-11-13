"""Explicit protocol-test fixture; never registered as a scientific baseline."""
import torch
from baselines.streambench.method import StreamingMemoryMethod

class CounterMemory(StreamingMemoryMethod):
    def initialize(self):
        self.values = []
        self.calls = 0
    def observe(self, frames, timestamps):
        self.values.extend((t['global_frame_id'], int(torch.as_tensor(f).sum())) for f,t in zip(frames,timestamps))
    def persistent_state(self):
        return {'metadata_bytes': {'values': self.values, 'calls': self.calls}}
    def state_dict(self):
        return {'values': self.values, 'calls': self.calls}
    def load_state_dict(self,state):
        self.values = state['values']
        self.calls = state['calls']
    def answer(self, question, choices=None):
        self.calls += 1
        self.values.append((-1, 123))
        if question == 'fail':
            raise RuntimeError('intentional QA failure')
        return {'answer': str(sum(v for _,v in self.values)), 'query_read_bytes': 8*len(self.values),
                'num_retrieved_items': len(self.values)}
