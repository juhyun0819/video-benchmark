"""Reject state that plain PyTorch serialization cannot restore faithfully."""
from collections import deque
from dataclasses import fields, is_dataclass

from .accounting import DiskState


def validate_checkpoint_state(value):
    import numpy as np
    import torch
    seen=set()

    def visit(item):
        if id(item) in seen:return
        seen.add(id(item))
        if isinstance(item,DiskState):
            raise TypeError('DiskState accounts external files only; state_dict must embed their complete contents/index state before checkpointing')
        if isinstance(item,np.ndarray):
            if item.base is not None or not item.flags.owndata:
                raise TypeError('Non-owning NumPy state cannot preserve aliases through pickle; expose full backing storage as torch tensors/views')
            if item.dtype.hasobject:
                raise TypeError('Object NumPy arrays must expose explicit checkpoint contents')
            return
        if isinstance(item,torch.Tensor):return
        if isinstance(item,dict):
            for key,entry in item.items():visit(key);visit(entry)
        elif isinstance(item,(list,tuple,deque,set,frozenset)):
            for entry in item:visit(entry)
        elif is_dataclass(item) and not isinstance(item,type):
            for field in fields(item):visit(getattr(item,field.name))
    visit(value)
