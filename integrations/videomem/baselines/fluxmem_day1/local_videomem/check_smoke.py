"""Inspect saved real-video outputs without running inference or changing them."""
import argparse
import json
from pathlib import Path


def inspect(work):
    results = {}
    for label in ('fluxmem', 'uncompressed_control'):
        directory = work / label
        rows = json.loads((directory / 'predictions.json').read_text())
        provenance = json.loads((directory / 'provenance.json').read_text())
        status = json.loads((directory / 'status.json').read_text())
        if len(rows) != 1 or status['completed'] != 1:
            raise ValueError(f'{label}: the one-question smoke did not complete')
        row = rows[0]
        wire = json.loads((directory / 'wire_0000.json').read_text())
        if row['status'] != 'complete' or not row['prediction'].strip():
            raise ValueError(f'{label}: incomplete or empty prediction')
        # These assertions concern the actual text given to the processor. Full
        # paths remain in the external audit trail, not the model prompt.
        for forbidden in ('FORBIDDEN_', '/home/', '.mp4', 'target_time', 'gt_answer'):
            if forbidden in wire['prompt']:
                raise ValueError(f'{label}: private metadata in the prompt: {forbidden}')
        shapes = wire['tensor_shapes']
        if not shapes.get('pixel_values_videos') or not shapes.get('video_grid_thw'):
            raise ValueError(f'{label}: no native video tensor was passed')
        cutoff = row['question']['query_time']['seconds']
        if len(wire['frames']) != 90 or any(f['day_seconds'] >= cutoff for f in wire['frames']):
            raise ValueError(f'{label}: unexpected frame count or future frame')
        events = row['memory_events']
        if label == 'fluxmem':
            if len(events) != 1 or events[0]['temporal_calls'] <= 0 or events[0]['spatial_calls'] <= 0:
                raise ValueError('The real-video run did not exercise both memory operations')
            if events[0]['dropped_tokens'] <= 0:
                raise ValueError('The real-video run did not compress tokens')
        elif events:
            raise ValueError('Disabled control executed compression')
        if provenance['training'] is not False or provenance['gpu']['physical_index'] != (4 if label == 'fluxmem' else 5):
            raise ValueError('Unexpected training or GPU configuration')
        results[label] = {'prediction':row['prediction'], 'frames':row['frames'],
                         'generation_seconds':row['generation_seconds'],
                         'peak_gpu_bytes':row['peak_gpu_bytes'],
                         'input_tokens':row['input_tokens'], 'output_tokens':row['output_tokens'],
                         'memory_events':events, 'gpu':provenance['gpu']}
    on = json.loads((work/'fluxmem/wire_0000.json').read_text())
    off = json.loads((work/'uncompressed_control/wire_0000.json').read_text())
    if on != off:
        raise ValueError('Paired runs did not receive identical video and question inputs')
    return {'execution_contract':'passed', 'benchmark_accuracy':None,
            'scope':'One synthetic question about a real 90-second local video; no fine-tuning.',
            'results':results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=Path(__file__).resolve().parents[1]/'local_runs/smoke_v1')
    args = parser.parse_args()
    print(json.dumps(inspect(args.work), ensure_ascii=False, indent=2))
