from types import SimpleNamespace

import pytest
import torch

from baselines.egogpt_caption.memory import offload_idle_encoders


class Encoder:
    def __init__(self,name,events):
        self.name=name;self.events=events;self.device='cpu'
    def to(self,*,device):
        self.device=str(device);self.events.append((self.name,self.device));return self


class Model:
    def __init__(self):
        self.events=[]
        self.speech=Encoder('speech',self.events)
        self.vision=Encoder('vision',self.events)
        self.prepared=(None,object(),object(),None,object(),None)
    def get_speech_encoder(self):return self.speech
    def get_vision_tower(self):return self.vision
    def get_input_embeddings(self):return SimpleNamespace(weight=SimpleNamespace(device='cuda:0'))
    def prepare_inputs_labels_for_speech_and_text(self,mode):
        if mode=='prefill':
            assert self.speech.device==self.vision.device=='cuda:0'
        self.events.append(('prepare',mode))
        if mode=='early_decode':return (object(),None,None,object(),None,None)
        return self.prepared


def test_offload_preserves_prepared_tuple_and_occurs_only_after_prefill(monkeypatch):
    model=Model()
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:model.events.append(('cache','empty')))
    with offload_idle_encoders(model):
        result=model.prepare_inputs_labels_for_speech_and_text('prefill')
        assert result is model.prepared
        assert model.speech.device==model.vision.device=='cpu'
        checkpoint=list(model.events)
        model.prepare_inputs_labels_for_speech_and_text('decode')
        assert model.events==checkpoint+[('prepare','decode')]
    assert 'prepare_inputs_labels_for_speech_and_text' not in model.__dict__
    prefix=model.events.index(('prepare','prefill'))
    assert model.events.index(('speech','cpu'))>prefix
    assert model.events.index(('vision','cpu'))>prefix


def test_failed_generation_restores_method_and_next_caption_restores_encoders(monkeypatch):
    model=Model()
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    with pytest.raises(RuntimeError,match='generation failed'):
        with offload_idle_encoders(model):
            model.prepare_inputs_labels_for_speech_and_text('prefill')
            raise RuntimeError('generation failed')
    assert 'prepare_inputs_labels_for_speech_and_text' not in model.__dict__
    with offload_idle_encoders(model):
        assert model.speech.device==model.vision.device=='cuda:0'
        model.prepare_inputs_labels_for_speech_and_text('prefill')


def test_existing_instance_method_is_restored(monkeypatch):
    model=Model();original=model.prepare_inputs_labels_for_speech_and_text
    model.prepare_inputs_labels_for_speech_and_text=original
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    with offload_idle_encoders(model):model.prepare_inputs_labels_for_speech_and_text('prefill')
    assert model.__dict__['prepare_inputs_labels_for_speech_and_text'] is original


def test_no_offload_until_native_multimodal_embeddings_exist(monkeypatch):
    model=Model()
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    with offload_idle_encoders(model):
        result=model.prepare_inputs_labels_for_speech_and_text('early_decode')
        assert result[4] is None
        assert model.speech.device==model.vision.device=='cuda:0'
        model.prepare_inputs_labels_for_speech_and_text('prefill')
        assert model.speech.device==model.vision.device=='cpu'
