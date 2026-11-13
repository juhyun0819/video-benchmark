"""Strict alignment and availability checks for externally generated captions."""
import json
import math
from pathlib import Path
from baselines.ours_offline.inputs import sha256


class CaptionManifest:
    def __init__(self, path, archive):
        self.identity = {'path':str(Path(path).resolve()),'sha256':sha256(path)}
        data = json.loads(Path(path).read_text())
        if data.get('schema') != 'ours_videotree_captions_v1' or data.get('archive_sha256') != archive.identity:
            raise ValueError('Captions must identify the exact feature archive')
        producer = data.get('producer',{})
        if producer.get('question_conditioned') is not False or not producer.get('model'):
            raise ValueError('Caption producer must identify its model and declare question-independent generation')
        self.identity['producer'] = producer
        self.rows = {}
        for row in data['captions']:
            i = row.get('global_frame_id')
            if type(i) is not int or not 0 <= i < len(archive.records) or i in self.rows:
                raise ValueError('Duplicate or unknown caption frame ID')
            stamp = float(row['global_timestamp'])
            until = float(row['observed_until'])
            available = float(row['available_at'])
            if not all(math.isfinite(x) for x in (stamp,until,available)) or abs(stamp-archive.times[i])>1e-8:
                raise ValueError('Caption timestamp is not aligned to the decoded frame')
            if not stamp <= until <= available:
                raise ValueError('Caption cannot be available before all its observed evidence')
            if not isinstance(row.get('text'),str) or not row['text'].strip():
                raise ValueError('Empty caption')
            self.rows[i] = dict(row,global_timestamp=stamp,observed_until=until,available_at=available)
        if set(self.rows) != set(range(len(archive.records))):
            raise ValueError('Caption manifest must cover every feature row; missing captions are not silently dropped')

    def eligible(self, candidate_ids, cutoff):
        return [i for i in candidate_ids if self.rows[i]['available_at'] <= cutoff]

    def texts(self, ids):
        return [self.rows[i]['text'] for i in ids]
