"""Evidence-side routing is isolated from the model's question/choice interface."""
import math
import time
from baselines.streambench.scoring import score_answer
from .inputs import reference_samples


def plan_question(question, annotation, sources, max_frames):
    record = {'method':'GT-Evidence-Clip', 'qa_id':question['qa_id'],
        'query_time':question['query_time'], 'original_query_time':question.get('original_query_time'),
        'GT':question['answer'], 'prediction':'', 'raw_output':'', 'correct':False,
        'status':'error', 'error':None, 'actual_frame_timestamps':[], 'answer_latency':None,
        'stream_update_time':0.0, 'oracle_evidence_access':True,
        'evidence_policy':'declared points, source clips or half-open intervals; causal uniform sampling per clip',
        'need_audio':question.get('need_audio'), 'need_name':question.get('need_name')}
    try:
        clips, samples = reference_samples(sources, annotation, question['query_time'], max_frames)
        record.update(status='planned', evidence_reference=annotation,
            evidence_sources=[clip['path'] for clip in clips], selected_frames=samples)
    except (KeyError, ValueError, TypeError) as error:
        record['error'] = f'{type(error).__name__}: {error}'
    return record


def answer_question(question, record, root, backend, decoder=None):
    """Only raw frames, actual times, question and choices cross the model boundary."""
    if record['status'] != 'planned':return record
    if decoder is None:
        from baselines.ours_offline.inputs import decode_samples
        decoder=decode_samples
    try:
        start=time.perf_counter()
        frames, actual=decoder(root,record['selected_frames'],question['query_time'])
        if len(frames)!=len(actual) or not frames:
            raise ValueError('Decoder frame/timestamp mismatch')
        times=[item['actual_global_timestamp'] for item in actual]
        if any(t>question['query_time']+1e-8 for t in times):
            raise ValueError('Future evidence frame rejected before model call')
        if any(not math.isfinite(t) or t<0 for t in times) or any(a>=b for a,b in zip(times,times[1:])):
            raise ValueError('Evidence frame times must be finite and strictly increasing')
        record['decode_seconds']=time.perf_counter()-start
        record['actual_frame_timestamps']=actual
        start=time.perf_counter()
        result=backend.answer(frames,actual,question['question'],question['choices'])
        record['answer_latency']=time.perf_counter()-start
        prediction=result['prediction']
        correct,kind=score_answer(prediction,question)
        record.update(prediction=prediction,raw_output=result['raw_output'],correct=correct if result.get('complete',False) else False,
            score_kind=kind,generated_token_ids=result.get('generated_token_ids'),
            input_tokens=result.get('input_tokens'),model_inputs_metadata=result.get('model_inputs_metadata'),
            status='ok' if result.get('complete',False) else 'incomplete',
            error=None if result.get('complete',False) else 'Generation did not complete with EOS')
    except Exception as error:
        record.update(status='error',error=f'{type(error).__name__}: {error}')
    return record
