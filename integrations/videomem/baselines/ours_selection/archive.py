"""Verified, query-independent frame features used only for offline selection."""
from __future__ import annotations
import json
import math
from pathlib import Path
import time
from fractions import Fraction
from baselines.ours_offline.inputs import sha256, decode_sample, validate_sources

SCHEMA = 'ours_selection_archive_v1'


def candidate_plan(sources, fps=1.):
    """Fixed-rate source-local grid on the supplied timeline, with no day clock assumptions."""
    sources=sorted(sources,key=lambda s:s['start_time'])
    rate=Fraction(str(fps))
    if rate<=0:raise ValueError('Positive candidate fps required')
    result=[]
    for n,source in enumerate(sources):
        start=Fraction(str(source.get('start_time_fraction',source['start_time'])))
        duration=Fraction(str(source.get('duration_fraction',source['duration'])))
        native_fps=Fraction(str(source.get('fps_fraction',source['fps'])))
        duration=min(duration,Fraction(source['frames'],1)/native_fps)
        if n+1<len(sources):
            following=sources[n+1]
            next_start=Fraction(str(following.get('start_time_fraction',following['start_time'])))
            duration=min(duration,next_start-start)
        if duration<=0:raise ValueError('Distinct source start times required')
        for i in range(math.ceil(duration*rate)):
            seconds=Fraction(i,1)/rate
            result.append({'source':source['path'],'source_start_time':source['start_time'],
                'day':source.get('day'),'requested_source_seconds':float(seconds),
                'requested_global_timestamp':float(start+seconds),'global_frame_id':len(result)})
    return result


def verify_sources(root, sources):
    root = Path(root).resolve()
    for source in sources:
        path = (root/source['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Source escapes dataset root')
        stat = path.stat()
        state = [stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        if source.get('file_state') != state:
            raise ValueError(f'Source changed since verified index: {source["path"]}')


def prepare_archive(root, sources_path, output, method, encoder, *, fps=1., batch_size=8):
    import numpy as np
    if method not in ('aks', 'videotree') or not math.isfinite(fps) or fps <= 0 or batch_size < 1:
        raise ValueError('Invalid feature extraction configuration')
    root, sources_path, output = Path(root), Path(sources_path), Path(output)
    sources = validate_sources(root,json.loads(sources_path.read_text())['sources'])
    if not sources:
        raise ValueError('Empty source index')
    if any(fps > float(s['fps']) for s in sources):
        raise ValueError('Candidate fps must not exceed native fps')
    verify_sources(root, sources)
    plan = candidate_plan(sources, fps=fps)
    output.mkdir(parents=True, exist_ok=False)
    status_path = output/'status.json'
    status_path.write_text(json.dumps({'status':'building','frames':0}))
    array = None
    encoder_dtype = None
    started = time.perf_counter()
    try:
        with (output/'frames.jsonl').open('w') as records:
            previous_time = -math.inf
            for start in range(0, len(plan), batch_size):
                images, stamps = [], []
                for sample in plan[start:start+batch_size]:
                    image, actual = decode_sample(root, sample, sample['requested_global_timestamp'])
                    if actual['actual_global_timestamp'] <= previous_time:
                        raise ValueError('Actual candidate timestamps are not strictly increasing')
                    previous_time = actual['actual_global_timestamp']
                    images.append(image);stamps.append(actual)
                features = encoder.encode_images(images) if method == 'aks' else encoder(images)
                actual_dtype=str(features.dtype)
                if encoder_dtype is None:encoder_dtype=actual_dtype
                if encoder_dtype!=actual_dtype:raise ValueError('Encoder dtype changed')
                if hasattr(features,'detach'):
                    features=features.detach().cpu()
                    # NumPy has no BF16 dtype. FP32 represents every BF16 value exactly.
                    if str(features.dtype)=='torch.bfloat16':features=features.float()
                    features=features.numpy()
                else:features=np.asarray(features)
                if features.ndim != 2 or len(features) != len(stamps) or features.shape[1]<1 or features.dtype.kind!='f' or not np.isfinite(features).all():
                    raise ValueError('Encoder returned invalid feature rows')
                if array is None:
                    array = np.lib.format.open_memmap(output/'features.npy', mode='w+', dtype=features.dtype,
                                                    shape=(len(plan), features.shape[1]))
                if features.shape[1] != array.shape[1] or features.dtype != array.dtype:
                    raise ValueError('Encoder feature schema changed')
                array[start:start+len(stamps)] = features
                for stamp in stamps:
                    records.write(json.dumps(stamp,ensure_ascii=False)+'\n')
                records.flush();array.flush()
                status_path.write_text(json.dumps({'status':'building','frames':start+len(stamps)}))
        verify_sources(root, sources)
        metadata = {'schema':SCHEMA,'method':method,'status':'complete','candidate_fps':fps,
                    'source_index_sha256':sha256(sources_path),'source_index':sources,
                    'encoder_identity':encoder.identity,'feature_shape':list(array.shape),
                    'feature_dtype':str(array.dtype),'frames_sha256':sha256(output/'frames.jsonl'),
                    'encoder_feature_dtype':encoder_dtype,
                    'features_sha256':sha256(output/'features.npy'),
                    'query_conditioned':False,'answer_visual_feature_cache':False,
                    'storage_policy':'unbounded offline feature index; answer model rereads selected raw frames'}
        (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        status_path.write_text(json.dumps({'status':'complete','frames':len(plan),'seconds':time.perf_counter()-started}))
        return metadata
    except Exception as exc:
        status_path.write_text(json.dumps({'status':'failed','error':f'{type(exc).__name__}: {exc}'}))
        raise


class SelectionArchive:
    def __init__(self, path, root, sources_path):
        import numpy as np
        self.path = Path(path)
        self.identity = sha256(self.path/'metadata.json')
        self.metadata = json.loads((self.path/'metadata.json').read_text())
        self._anchors={name:self._stat(self.path/name) for name in ('metadata.json','frames.jsonl','features.npy')}
        meta = self.metadata
        if meta.get('schema') != SCHEMA or meta.get('status') != 'complete' or meta.get('query_conditioned') is not False:
            raise ValueError('A complete question-independent selection archive is required')
        if sha256(sources_path) != meta['source_index_sha256']:
            raise ValueError('Archive does not match requested source index')
        verify_sources(root, meta['source_index'])
        for filename,key in [('frames.jsonl','frames_sha256'),('features.npy','features_sha256')]:
            if sha256(self.path/filename) != meta[key]:
                raise ValueError(f'Archive integrity mismatch: {filename}')
        self.records = [json.loads(line) for line in (self.path/'frames.jsonl').read_text().splitlines()]
        self.features = np.load(self.path/'features.npy',mmap_mode='r',allow_pickle=False)
        if list(self.features.shape) != meta['feature_shape'] or str(self.features.dtype) != meta['feature_dtype'] or len(self.records) != len(self.features):
            raise ValueError('Archive feature/record shape mismatch')
        if not np.isfinite(self.features).all():
            raise ValueError('Nonfinite archive features')
        self.times = [float(r['actual_global_timestamp']) for r in self.records]
        if any(not math.isfinite(t) for t in self.times) or any(a>=b for a,b in zip(self.times,self.times[1:])):
            raise ValueError('Invalid candidate chronology')
        if [r['global_frame_id'] for r in self.records] != list(range(len(self.records))):
            raise ValueError('Candidate IDs must match archive row order')
        self._sources={s['path']:s for s in meta['source_index']}
        self.assert_unchanged()

    @staticmethod
    def _stat(path):
        s=path.stat()
        return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns

    def assert_unchanged(self):
        if any(self._stat(self.path/name)!=anchor for name,anchor in self._anchors.items()):
            raise ValueError('Selection archive changed during evaluation')

    def verify_selected_sources(self, root, ids):
        names={self.records[i]['source'] for i in ids}
        verify_sources(root,[self._sources[name] for name in names])

    def eligible(self, cutoff):
        import bisect
        self.assert_unchanged()
        if not math.isfinite(cutoff):
            raise ValueError('Invalid QA cutoff')
        return list(range(bisect.bisect_right(self.times,cutoff)))
