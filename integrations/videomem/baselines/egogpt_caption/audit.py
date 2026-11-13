"""Count verified outputs and gate expansion from real-media smoke inference."""
from fractions import Fraction
import json
from pathlib import Path

from .archive import load_complete, object_hash


def inspect(root, clip_ids=None, smoke=False):
    root = Path(root)
    job = json.loads((root/'job.json').read_text())
    if object_hash({k:v for k,v in job.items() if k!='job_hash'}) != job['job_hash']:
        raise ValueError('Job hash mismatch')
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['job_hash'] != job['job_hash']:
        raise ValueError('Manifest job mismatch')
    from .media import build_clips
    if manifest['clips'] != build_clips(job['inventory']):
        raise ValueError('Manifest does not cover the exact source inventory')
    clips = manifest['clips']
    if clip_ids is not None:
        wanted = set(clip_ids)
        if not wanted <= {c['clip_id'] for c in clips}:
            raise ValueError('Unknown requested clip')
        clips = [c for c in clips if c['clip_id'] in wanted]
    complete = []
    missing = []
    finish_reason_counts = {'eos':0,'length':0}
    length_stopped = []
    for clip in clips:
        record = load_complete(root, clip['clip_id'], job['job_hash'])
        if record is None:
            missing.append(clip['clip_id'])
            continue
        if record['clip'] != clip:
            raise ValueError('Caption clip metadata mismatch')
        provenance = json.loads((root/'provenance'/f"{record['provenance_hash']}.json").read_text())
        if object_hash(provenance) != record['provenance_hash']:
            raise ValueError('Caption provenance hash mismatch')
        if smoke:
            media = record['media']
            if (record['pixel_shape'] != [len(clip['frame_indices']),3,384,384]
                    or record['speech_shape'] != [1,3000,128]
                    or record['speech_lengths'] != [3000]
                    or record['model_dtype'] != 'torch.float16'
                    or record['speech_dtype'] != 'torch.float16'
                    or len(media['frames']) != len(clip['frame_indices'])
                    or media['audio']['sample_rate'] != 16000
                    or media['audio']['sample_count'] <= 0
                    or record['strict_causal'] is not False
                    or Fraction(record['conservative_available_at_global_seconds_fraction'])
                       < Fraction(clip['global_end_fraction'])):
                raise ValueError(f"Smoke protocol mismatch: {clip['clip_id']}")
        complete.append(clip['clip_id'])
        reason = record.get('finish_reason','eos')
        finish_reason_counts[reason] += 1
        if reason == 'length':
            length_stopped.append(clip['clip_id'])
    return {'job_hash':job['job_hash'],'expected':len(clips),'completed':len(complete),
            'missing_or_incomplete':len(missing),'missing_clip_ids':missing,
            'finish_reason_counts':finish_reason_counts,'length_stopped_clip_ids':length_stopped,
            'status':'complete' if clips and not missing else 'incomplete'}


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--clip-ids',nargs='+')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    result=inspect(args.output,args.clip_ids,args.smoke)
    print(json.dumps(result,indent=2))
    return 0 if result['status']=='complete' else 2


if __name__=='__main__':
    raise SystemExit(main())
