"""Rescore completed QA and report explicit audio-requirement subsets."""
import argparse
import json
from pathlib import Path
from baselines.streambench.scoring import score_answer
from .inputs import sha256


def performance_breakdown(questions, rows):
    canonical = {q['qa_id']:q for q in questions}
    if len(canonical) != len(questions):raise ValueError('Duplicate canonical QA ID')
    observed = {}
    for row in rows:
        key = row['qa_id']
        if key not in canonical or key in observed:raise ValueError('Unknown or duplicate result QA ID')
        if row.get('supervisor_generated'):continue
        q = canonical[key]
        index = q.get('correct_idx')
        if type(index) is not int:
            answer=q.get('answer')
            index=(ord(answer)-65 if isinstance(answer,str) and len(answer)==1 and 'A'<=answer<='Z'
                   else q['choices'].index(answer))
        gold=chr(65+index)
        if row.get('GT') != gold or row.get('query_time') != q['query_time']:
            raise ValueError('Result GT or query time differs from canonical QA')
        if row.get('status') not in ('ok','error','incomplete'):raise ValueError('Unexpected result status')
        correct,_ = score_answer(row['prediction'],q)
        observed[key] = (bool(correct and row['status']=='ok'),row['status'])
    groups = {
        'all':list(canonical),
        'no_audio_required':[k for k,q in canonical.items() if q.get('need_audio') is False],
        'audio_required':[k for k,q in canonical.items() if q.get('need_audio') is True],
        'audio_requirement_unknown':[k for k,q in canonical.items() if type(q.get('need_audio')) is not bool],
    }
    result = {}
    for name,ids in groups.items():
        values = [observed[k] for k in ids if k in observed]
        correct = sum(c for c,_ in values)
        result[name] = {'expected':len(ids),'completed':len(values),'correct':correct,
            'errors_or_incomplete':sum(status!='ok' for _,status in values),
            'accuracy_over_completed':correct/len(values) if values else None,
            'accuracy_over_expected':correct/len(ids) if ids else None,
            'coverage_complete':len(values)==len(ids)}
    return result


def read_complete_rows(path):
    if not Path(path).exists():return []
    return [json.loads(line) for line in Path(path).read_bytes().splitlines(keepends=True) if line.endswith(b'\n')]


def report_run(folder):
    folder = Path(folder)
    plan = json.loads((folder/'plan.json').read_text())
    path = Path(plan['config']['canonical'])
    if sha256(path) != plan['canonical_sha256']:raise ValueError('Canonical manifest changed')
    questions = json.loads(path.read_text())['questions']
    output = {'run':str(folder),'max_frames':plan['config']['max_frames'],
        'sampling_fps':plan['config'].get('sampling_fps'),
        'subset_definition':'need_audio is exactly false; need_name questions remain included',
        'models':{}}
    for model in plan['config']['models']:
        rows = []
        for worker in plan['workers']:
            if worker['model'] == model['name']:
                rows.extend(read_complete_rows(Path(worker['output'])/'predictions.jsonl'))
        output['models'][model['name']] = performance_breakdown(questions,rows)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=report_run(args.run)
    text=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if args.output:
        temporary=args.output.with_name(args.output.name+'.tmp')
        temporary.write_text(text);temporary.replace(args.output)
    print(text)


if __name__=='__main__':main()
