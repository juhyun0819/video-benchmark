"""Release idle encoder weights after native multimodal input construction.

Encoder arithmetic always runs on the original CUDA device and dtype. Only
weights and buffers that the language decoder no longer uses move to CPU;
the native prepared tuple and language-model computation are left untouched.
"""
from contextlib import contextmanager


@contextmanager
def offload_idle_encoders(model):
    import torch
    name='prepare_inputs_labels_for_speech_and_text'
    original=getattr(model,name)
    had_override=name in model.__dict__
    prior_override=model.__dict__.get(name)
    encoders=(model.get_speech_encoder(),model.get_vision_tower())
    if any(encoder is None for encoder in encoders):
        raise ValueError('Audiovisual captions require both native encoders')
    device=model.get_input_embeddings().weight.device
    # The previous caption may have failed with live allocations until its
    # exception handler finished. Release unused allocator blocks before reload.
    torch.cuda.empty_cache()
    for encoder in encoders:encoder.to(device=device)
    offloaded=False

    def prepare(*args,**kwargs):
        nonlocal offloaded
        result=original(*args,**kwargs)
        if not offloaded and result[4] is not None:
            for encoder in encoders:encoder.to(device='cpu')
            torch.cuda.empty_cache()
            offloaded=True
        return result

    setattr(model,name,prepare)
    try:
        yield
    finally:
        if had_override:setattr(model,name,prior_override)
        else:delattr(model,name)
        # Keep inactive encoders on CPU between captions. The next entry reloads
        # them before the unchanged native preparation method executes.
