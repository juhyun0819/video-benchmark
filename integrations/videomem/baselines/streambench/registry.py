"""Lazy adapters; unavailable baselines are explicit, never replacements."""
from importlib import import_module

ADAPTERS = {'fifo':'native', 'reservoir':'native', 'streamforest':'native',
            'fluxmem_streaming':'native', 'fluxmem_qwen3vl':'native',
            'egorag':'egorag', 'videotree':'videotree', 'aks':'aks',
            'streamttt':'streamttt', 'cambrian_s':'cambrian_s', 'rekv':'rekv',
            'simplestream':'simplestream'}
MISSING = {'hermes':'No HERMES implementation is present in this checkout.',
           'fluxmem_reference':'The released reference re-prefills the whole video; use the explicitly documented streaming adapter.'}


def available_methods():
    return {**{name:{'status':'adapter_available','runtime_validation':'requires native dependencies/checkpoint'} for name in ADAPTERS},
            **{name:{'status':'not_in_checkout' if name!='fluxmem_reference' else 'not_continual','reason':reason} for name,reason in MISSING.items()}}


def create_method(config):
    if not isinstance(config,dict):raise ValueError('Method config must be a JSON object')
    name=config.get('method')
    if name in MISSING:raise NotImplementedError(MISSING[name])
    if name not in ADAPTERS:raise ValueError(f'Unknown method {name!r}; use list-methods')
    return import_module('baselines.streambench.adapters.'+ADAPTERS[name]).create_method(config)
