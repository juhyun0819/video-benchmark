"""Bounded real-media/local-model integration check, not benchmark accuracy.

Runs all stages in one process so the two role-specific model replicas are
reused. Uses a short MP4 as its own explicit local playback timeline. No wall
clock or final Day-concatenation mapping is inferred.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from videomem_adapter.common import read, write
from videomem_adapter.media import prepare_manifest
from videomem_adapter.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--work', required=True)
    parser.add_argument('--cutoff', type=float, default=60)
    args = parser.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    config = read(args.config)
    if (config.get('test_only') or config['embedding']['backend'] != 'openclip'
            or config.get('index_engine') != 'chroma'):
        raise ValueError('Real smoke requires OpenCLIP embeddings, Chroma, and real models')
    if any(r['backend'] != 'hf_local' for r in config['models'].values()):
        raise ValueError('This smoke is local-HF only; no external API is called')
    manifest = prepare_manifest([{
        'day_id': 'SmokeClip', 'day_order': 1,
        'path': str(Path(args.source).resolve()),
        'timeline_status': 'direct_mp4',
    }], clip_seconds=30, fingerprint_cache_dir=work/'fingerprints')
    if len(manifest['clips']) > 4 or not 30 <= args.cutoff <= 90:
        raise ValueError('Smoke input must be at most four 30-second clips; cutoff30..90s')
    q = {
        'question_id': 'real-local-video-smoke-not-benchmark',
        'question': 'What setting and activity are visible in the observed video?',
        'choices': [],
        'query_time': {'day_id': 'SmokeClip', 'day_order': 1, 'seconds': args.cutoff},
        'answer': 'FORBIDDEN_GOLD_MARKER', 'reason': 'FORBIDDEN_REASON_MARKER',
        'target_time': 'FORBIDDEN_TARGET_MARKER', 'keywords': 'FORBIDDEN_KEYWORD_MARKER',
    }
    write(work/'config.json', config)
    write(work/'manifest.json', manifest)
    write(work/'smoke_question_private.json', q)
    started = time.monotonic()
    stage = 'initializing'
    report = {'scope': 'real local inference smoke; no benchmark accuracy',
              'real_llm_inference': False, 'status': 'running'}
    try:
        pipe = Pipeline(config, manifest, work/'run')
        stage = 'create'
        created = pipe.create(cutoff=q['query_time'])
        write(work/'create.json', created)
        if created['errors']:
            raise RuntimeError('Preprocessing failed; inspect create.json; no semantic retries')
        stage = 'query'
        query = pipe.query(q)
        write(work/'query.json', query)
        stage = 'answer'
        answer = pipe.answer(query)
        write(work/'answer.json', answer)
        if answer['status'] != 'complete' or not answer['used_video_segments']:
            raise RuntimeError('No completed visual answer; result preserved')
        assert all(h['clip']['day_end_seconds'] <= args.cutoff for h in query['retrieved'])
        assert all(f['day_seconds'] < args.cutoff
                   for seg in answer['used_video_segments'] for f in seg['frames'])
        calls = pipe.models.calls
        embedding_calls = pipe.embeddings.calls
        stage = 'resume'
        resumed = pipe.create(cutoff=q['query_time'])
        assert not resumed['errors']
        assert pipe.models.calls == calls and pipe.embeddings.calls == embedding_calls
        trace = (work/'run/model_calls.jsonl').read_text()
        assert 'FORBIDDEN_' not in trace
        records = [json.loads(line) for line in trace.splitlines()]
        assert any(r['role']=='answerer' and r['images'] and r['status']=='ok' for r in records)
        assert all(r['backend']=='hf_local' for r in records)
        report.update(status='passed', real_llm_inference=True,
                      model_calls=calls, embeddings_calls=embedding_calls,
                      caption_resume=resumed, new_calls_on_caption_resume=0,
                      prediction=answer['prediction'], storage=pipe.storage(),
                      checks=['future_intervals_excluded', 'gold_fields_hidden',
                              'real_frames_in_final_answer', 'real_openclip_and_chroma',
                              'caption_and_embedding_resume'])
    except Exception as exc:
        report.update(status='execution_error', failed_stage=stage,
                      error_type=type(exc).__name__, error=str(exc)[:1000])
        raise
    finally:
        report['wall_seconds'] = time.monotonic()-started
        trace_path = work/'run/model_calls.jsonl'
        if trace_path.exists():
            traces = [json.loads(line) for line in trace_path.read_text().splitlines()]
            report['real_llm_inference'] = any(
                r['backend']=='hf_local' and r['status']=='ok' for r in traces)
            report['recorded_calls'] = len(traces)
            report['recorded_execution_errors'] = sum(r['status']!='ok' for r in traces)
        write(work/'result.json', report)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
