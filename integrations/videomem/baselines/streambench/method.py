"""Shared lifecycle; adapters expose native state rather than common tokens."""
from copy import deepcopy
from pathlib import Path
from typing import Any

from .accounting import measure_state


class UnsupportedBudget(RuntimeError):
    def __init__(self, message='Persistent state exceeds requested budget', *,
                 memory_budget_bytes=None, persistent_memory_bytes=None,
                 minimum_memory_bytes=None):
        super().__init__(message)
        self.memory_budget_bytes = memory_budget_bytes
        self.persistent_memory_bytes = persistent_memory_bytes
        self.minimum_memory_bytes = minimum_memory_bytes

    def as_dict(self):
        return {'status': 'unsupported_under_budget', 'reason': str(self),
                'memory_budget_bytes': self.memory_budget_bytes,
                'persistent_memory_bytes': self.persistent_memory_bytes,
                'minimum_memory_bytes': self.minimum_memory_bytes}


class StreamingMemoryMethod:
    """One adapter instance is one stream. Only reset starts an independent one.

    state_dict must include *all* continuation state, including pending frame
    groups and native RNGs, but no frozen pretrained parameters or dependencies.
    Return tensors without individual cloning: deepcopy/torch serialization
    preserve shared storage and views. persistent_state exposes the live state,
    before any serialization copies. QA returns an answer/read-metrics mapping.
    """
    def __init__(self, memory_budget_bytes: int):
        if type(memory_budget_bytes) is not int or memory_budget_bytes <= 0:
            raise ValueError('memory_budget_bytes must be a positive integer')
        self.memory_budget_bytes = memory_budget_bytes
        self.protocol_state: dict[str, Any] = {}
        self.minimum_memory_bytes = None

    def initialize(self):
        raise NotImplementedError

    def observe(self, frames, timestamps):
        raise NotImplementedError

    def finalize_chunk(self):
        pass

    def persistent_state(self):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state):
        raise NotImplementedError

    def memory_bytes(self):
        components = dict(self.persistent_state())
        components['metadata_bytes'] = [components.get('metadata_bytes'), self.protocol_state]
        return measure_state(components)

    def enforce_budget(self):
        report = self.memory_bytes()
        actual = report['persistent_total_bytes']
        if actual > self.memory_budget_bytes:
            raise UnsupportedBudget(memory_budget_bytes=self.memory_budget_bytes,
                                    persistent_memory_bytes=actual,
                                    minimum_memory_bytes=self.minimum_memory_bytes)
        return report

    def snapshot(self):
        from .state import validate_checkpoint_state
        state = {'method': self.state_dict(), 'protocol': self.protocol_state}
        validate_checkpoint_state(state)
        return deepcopy(state)

    def restore(self, state):
        owned = deepcopy(state)
        self.load_state_dict(owned['method'])
        self.protocol_state = owned['protocol']

    def save_checkpoint(self, path):
        from .checkpoint import write_checkpoint
        self.enforce_budget()
        return write_checkpoint(path, {'schema_version': 1, 'adapter': type(self).__module__ + '.' + type(self).__qualname__,
                                      'memory_budget_bytes': self.memory_budget_bytes,
                                      'snapshot': self.snapshot()})

    def load_checkpoint(self, path):
        from .checkpoint import read_checkpoint
        state = read_checkpoint(path)
        if (state['adapter'] != type(self).__module__ + '.' + type(self).__qualname__
                or state['memory_budget_bytes'] != self.memory_budget_bytes):
            raise ValueError('Checkpoint method/budget does not match adapter')
        self.restore(state['snapshot'])
        self.enforce_budget()

    def answer(self, question, choices=None):
        raise NotImplementedError

    def reset(self):
        self.protocol_state = {}
        self.initialize()
