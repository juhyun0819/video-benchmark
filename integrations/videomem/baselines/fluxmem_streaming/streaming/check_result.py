"""Read-only audit of a completed local streaming pilot (no model inference)."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streaming.video import frame_key, time_key


def check_result(directory):
    path = Path(directory)
    status = json.loads((path/'status.json').read_text())
    provenance = json.loads((path/'provenance.json').read_text())
    events = [json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
    arrivals = [r for r in events if r['event'] == 'arrival']
    answers = [r for r in events if r['event'] == 'answer']
    explicit = [r for r in answers if r['kind'] == 'explicit']
    automatic = [r for r in answers if r['kind'] == 'proactive']
    assert status['phase'] == 'finished'
    assert len(arrivals) == status['encoder_calls'] == status['decode']['pairs']
    assert len({r['pair_index'] for r in arrivals}) == len(arrivals)
    assert len(explicit) == status['explicit_questions']
    assert len(automatic) == status['proactive_answers']
    assert all(r['status'] == 'complete' for r in answers)
    assert status['incomplete_answers'] == 0
    assert status['decode']['max_live_sampled_frames'] <= 2
    assert status['training'] is False
    assert all(not p.get('training', False) for p in [provenance])
    for row in answers:
        # Exact leak sentinels belong only to the deliberately private input.
        assert 'FORBIDDEN_' not in row['prompt']
        assert set(row['question']) <= {'question_id','question','choices','query_time'}
        if row['kind'] == 'explicit':
            for record in row['memory_records']:
                assert all(frame_key(f) < time_key(row['question'])
                           for f in record['metadata']['frame_metadata'])
        assert row['memory_tokens'] == row['memory_stats']['total_tokens']
    settings = provenance['settings']
    capacities = {'short':settings['short_frames']//2, 'medium':settings['medium_frames']//2}
    if settings['long_frames'] is not None:
        capacities['long'] = settings['long_frames']//2
    for row in arrivals:
        assert all(row['after_advance']['tier_grids'][key] <= limit for key,limit in capacities.items())
        if row['proactive_answered']:
            assert row['observation']['has_prior_response']
            assert row['observation']['backward_score'] > settings['trigger_threshold']
            assert not row['explicit_answered']
    repeated = []
    for previous, current in zip(explicit,explicit[1:]):
        if previous['question']['query_time'] == current['question']['query_time'] and previous['prompt'] == current['prompt']:
            assert previous['encoder_calls'] == current['encoder_calls']
            assert previous['memory_records'] == current['memory_records']
            assert previous['token_ids'] == current['token_ids']
            repeated.append([previous['question']['question_id'],current['question']['question_id']])
    return {'checked':True,'arrival_count':len(arrivals),'explicit_answers':len(explicit),
            'automatic_answers':len(automatic),'repeated_query_no_reencoding':repeated,
            'future_frame_checks':'passed','private_input_marker_checks':'passed',
            'capacity_checks':'passed','trigger_checks':'passed',
            'benchmark_accuracy':None, 'semantic_answer_quality':'not_scored'}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory')
    args=p.parse_args()
    print(json.dumps(check_result(args.directory),ensure_ascii=False,indent=2))
