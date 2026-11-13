"""Optional native decoder placement; no temporal or independent-state shards."""
import types
import torch
from baselines.streambench.adapters.streamttt import move_tree


def install_two_gpu_layers(method, second_device='cuda:1', split=18):
    model=method._model
    layers=model.model.language_model.layers
    device=torch.device(second_device)
    first=torch.device(method.config.get('device','cuda:0'))
    assert 3 <= split < len(layers)
    original_load=method.load_state_dict

    def relocate():
        if method._states is not None:
            method._states=[move_tree(s,device if i>=split else first) for i,s in enumerate(method._states)]
        if method._cache is not None:
            for i,layer in enumerate(method._cache.layers):
                target=device if i>=split else first
                for key in ('keys','values'):
                    value=getattr(layer,key,None)
                    if value is not None:setattr(layer,key,value.to(target))
                if hasattr(layer,'device'):layer.device=target

    def load(self,state):
        original_load(state)
        relocate()
    method.load_state_dict=types.MethodType(load,method)
    handles=[]
    for i in range(split,len(layers)):
        layers[i].to(device)
        def before(module,args,kwargs,index=i):
            # Native Triton kernels launch on the current CUDA device.
            torch.cuda.set_device(device)
            states=kwargs.get('states')
            cache=kwargs.get('past_key_values')
            ordinary={k:v for k,v in kwargs.items() if k not in ('states','past_key_values')}
            ordinary=move_tree(ordinary,device)
            if states is not None:
                local=[None]*len(layers);local[index]=move_tree(states[index],device)
                ordinary['states']=local
            if cache is not None:ordinary['past_key_values']=cache
            return move_tree(args,device),ordinary
        handles.append(layers[i].register_forward_pre_hook(before,with_kwargs=True))
    def after(module,args,result):
        result=move_tree(result,first)
        torch.cuda.set_device(first)
        return result
    handles.append(layers[-1].register_forward_hook(after))
    relocate()
    return handles


def placement_neutral_state(value):
    """Only physical device labels are excluded; all tensor bytes stay audited."""
    if isinstance(value,torch.device):return '<physical-device>'
    if isinstance(value,dict):return {k:placement_neutral_state(v) for k,v in value.items()}
    if isinstance(value,list):return [placement_neutral_state(v) for v in value]
    if isinstance(value,tuple):return tuple(placement_neutral_state(v) for v in value)
    return value
