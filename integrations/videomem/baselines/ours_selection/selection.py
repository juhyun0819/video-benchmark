"""Question-adaptive selection over a strictly causal archive prefix."""
import math
import time
from baselines.ours_offline.inputs import decode_sample
from baselines.streambench.scoring import score_answer


def select_ids(archive, question, *, method, scorer, parameters, captions=None, tree_builder=None):
    import numpy as np
    import torch
    cutoff = question['query_time']
    if cutoff is None or not math.isfinite(cutoff):
        raise ValueError('Reviewed numeric query_time is required')
    candidates = archive.eligible(cutoff)
    if method == 'videotree':
        if captions is None:
            raise ValueError('VideoTree requires explicit aligned captions')
        candidates = captions.eligible(candidates,cutoff)
    if not candidates:
        raise ValueError('No causally available candidate evidence')
    features = torch.from_numpy(np.array(archive.features[candidates],copy=True))
    if method == 'aks':
        from baselines.streambench.adapters.aks import adaptive_select
        # Only the public question enters CLIP relevance scoring, not options/GT.
        scores = scorer.score(features,question['question'])
        if len(scores) != len(candidates):
            raise ValueError('Scorer must score every eligible candidate')
        local_ids = adaptive_select(scores,**parameters)
        details = {'selected_scores':[float(scores[i]) for i in local_ids],
                   'variant':'official_clip_scoring_native_target_vlm'}
    elif method == 'videotree':
        if tree_builder is None:
            from baselines.streambench.adapters.videotree import build_tree
            tree_builder = build_tree
        texts = captions.texts(candidates)
        local_ids, details = tree_builder(features,texts,
            lambda group:scorer.score(question['question'],list(question['choices']),group),**parameters)
        details = dict(details,variant='videotree_external_captions_selected_raw_frames_target_vlm')
    else:
        raise ValueError('Unsupported selection method')
    if not local_ids or len(local_ids)!=len(set(local_ids)) or any(type(i) is not int or not 0<=i<len(candidates) for i in local_ids):
        raise ValueError('Selector returned invalid/duplicate candidate IDs')
    selected = sorted(candidates[i] for i in local_ids)
    if any(archive.times[i]>cutoff for i in selected):
        raise ValueError('Future candidate selected')
    return selected,dict(details,candidate_count=len(candidates),selected_count=len(selected))


def evaluate_question(root, archive, question, *, method, scorer, parameters, backend,
                      max_frames, captions=None, decoder=decode_sample, tree_builder=None):
    start=time.perf_counter()
    record={'method':method+'+'+backend.backbone,'qa_id':question['qa_id'],
            'query_time':question['query_time'],'actual_frame_timestamps':[],
            'prediction':'','raw_output':'','GT':question['answer'],'correct':False,
            'answer_latency':None,'stream_update_time':0.,'status':'error','error':None}
    try:
        tick=time.perf_counter()
        ids,selection=select_ids(archive,question,method=method,scorer=scorer,parameters=parameters,
                                 captions=captions,tree_builder=tree_builder)
        record.update(selection=selection,selected_frame_ids=ids,selection_seconds=time.perf_counter()-tick)
        if len(ids)>max_frames:
            raise ValueError(f'Selected {len(ids)} frames exceeds explicit answer budget {max_frames}; no silent truncation')
        if len(ids)<2:
            raise ValueError('Native video answer requires at least two selected frames; no duplicate padding')
        archive.verify_selected_sources(root,ids)
        tick=time.perf_counter();frames=[];actual=[]
        for i in ids:
            frame,stamp=decoder(root,archive.records[i],question['query_time'])
            if abs(float(stamp['actual_global_timestamp'])-archive.times[i])>1e-8:
                raise ValueError('Selected raw frame no longer matches encoded candidate timestamp')
            if stamp['actual_global_timestamp']>question['query_time']+1e-8:
                raise ValueError('Future raw frame rejected')
            frames.append(frame);actual.append(stamp)
        record.update(actual_frame_timestamps=actual,decode_seconds=time.perf_counter()-tick)
        tick=time.perf_counter()
        response=backend.answer(frames,actual,question['question'],list(question['choices']))
        record['answer_latency']=time.perf_counter()-tick
        prediction=response['prediction']
        complete=bool(response.get('complete',False)) and bool(prediction.strip())
        correct,kind=score_answer(prediction,question)
        record.update(prediction=prediction,raw_output=response['raw_output'],
                      correct=bool(correct and complete),score_kind=kind,
                      status='ok' if complete else 'incomplete',
                      error=None if complete else 'Native answer generation was incomplete or returned an empty prediction',
                      generated_token_ids=response.get('generated_token_ids'),
                      model_inputs_metadata=response.get('model_inputs_metadata'),
                      input_tokens=response.get('input_tokens'))
    except Exception as exc:
        record['error']=f'{type(exc).__name__}: {exc}'
    record['total_query_seconds']=time.perf_counter()-start
    return record
