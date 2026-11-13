"""Query-free, lossless cache of the released online MemoryManager input.

No questions, memory settings, language model inference, or forest construction.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

from .upstream import UPSTREAM_COMMIT, VENDOR_ROOT, _namespace, verify_source

SHAPE = (729, 1152)
FRAME_BYTES = math.prod(SHAPE) * 2
REPO = 'MCG-NJU/StreamForest-Qwen2-7B'
REVISION = '6840f91e17346b4c406ba03b57d1a63582765c91'
SCHEMA = 'streamforest_native_archive_v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def sync_dir(path):
    fd = os.open(str(path), os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def ensure_dir(path):
    path = Path(path)
    missing=[]
    current=path
    while not current.exists():
        missing.append(current)
        current=current.parent
    path.mkdir(parents=True,exist_ok=True)
    for created in reversed(missing):
        sync_dir(created.parent)


def atomic(path, raw):
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    sync_dir(path.parent)


def write_json(path, value):
    atomic(path, canonical(value))


def source_state(path):
    s = Path(path).stat()
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns, inode=s.st_ino)


def validate_inventory(inventory):
    total = 0
    offset = Fraction(0)
    for source in inventory['sources']:
        if source_state(source['path']) != source['verified_state']:
            raise ValueError(f"Source changed: {source['path']}")
        if Fraction(source['global_start_fraction']) != offset:
            raise ValueError('Inventory global timeline has a gap')
        duration = Fraction(source['duration_fraction'])
        if math.ceil(duration) != source['expected_1fps_frames']:
            raise ValueError('Unexpected source frame count')
        total += source['expected_1fps_frames']
        offset += duration
    if total != inventory['total_frames']:
        raise ValueError('Inventory count mismatch')


def native_module():
    verify_source()
    for name in ('llava', 'llava.model', 'llava.model.multimodal_encoder'):
        _namespace(name)
    return importlib.import_module('llava.model.multimodal_encoder.siglip_encoder')


def load_tower(model_path, gpu):
    """Load *embedded trained* vision weights, using unmodified native classes."""
    from accelerate import init_empty_weights
    from safetensors import safe_open
    mod = native_module()
    model_path = Path(model_path)
    receipt = json.loads((model_path / 'native-vision-download.json').read_text())
    if (receipt['repo_id'], receipt['revision']) != (REPO, REVISION):
        raise ValueError('Wrong official checkpoint identity')
    config = json.loads((model_path / 'config.json').read_text())
    if config['mm_hidden_size'] != 1152 or config['mm_projector_type'] != 'tome729_fstw_pemf':
        raise ValueError('Wrong official vision/projector config')
    tower = mod.SigLipVisionTower(str(model_path), SimpleNamespace(mm_tunable_parts=''), delay_load=True)
    with init_empty_weights():
        tower.vision_tower = mod.SigLipVisionModel(tower.config)
        del tower.vision_tower.vision_model.encoder.layers[-1:]
        tower.vision_tower.vision_model.head = torch.nn.Identity()
    index = json.loads((model_path / 'model.safetensors.index.json').read_text())['weight_map']
    prefix = 'model.vision_tower.vision_tower.'
    selected = {k: v for k, v in index.items() if k.startswith(prefix)}
    if len(selected) != 421:
        raise ValueError('Incomplete official 26-layer vision weights')
    weights = {}
    for shard in sorted(set(selected.values())):
        with safe_open(model_path / shard, framework='pt', device='cpu') as stream:
            for key, filename in selected.items():
                if filename == shard:
                    weights[key[len(prefix):]] = stream.get_tensor(key)
    tower.vision_tower.load_state_dict(weights, strict=True, assign=True)
    del weights
    tower.is_loaded = True
    tower.eval().requires_grad_(False).to(device=f'cuda:{gpu}', dtype=torch.float16)
    return tower


def frames_for_range(inventory, start, end):
    """Decode exact source PTS on the same source-local 1fps grid as Qwen."""
    import av
    base = 0
    for source in inventory['sources']:
        count = source['expected_1fps_frames']
        lo, hi = max(0, start - base), min(count, end - base)
        if lo < hi:
            if source_state(source['path']) != source['verified_state']:
                raise ValueError('Source changed before decode')
            with av.open(source['path']) as container:
                stream = container.streams.video[0]
                stream.thread_type = 'SLICE'
                stream.codec_context.thread_count = 1
                tb = Fraction(stream.time_base)
                origin = Fraction(stream.start_time or 0) * tb
                if lo:
                    container.seek(int((origin + max(0, lo - 1)) / tb), stream=stream, backward=True)
                target = lo
                decode_timer = time.perf_counter()
                for decoded in container.decode(stream):
                    if decoded.pts is None:
                        raise ValueError('Missing video PTS')
                    actual = Fraction(decoded.pts) * Fraction(decoded.time_base) - origin
                    if actual < target:
                        continue
                    if actual >= Fraction(source['duration_fraction']) or target >= hi:
                        break
                    # These verified inputs are 25fps CFR. Refuse missing targets.
                    if actual >= target + 1:
                        raise ValueError('Source is missing an expected sampling target')
                    image = decoded.to_image().convert('RGB')
                    global_time = Fraction(source['global_start_fraction']) + actual
                    metadata = dict(
                        global_frame_id=base + target, day_id=source['day_id'],
                        day_order=source['day_order'], source_id=source['source_id'],
                        timestamp_seconds=float(global_time), timestamp_fraction=str(global_time),
                        source_timestamp=float(actual), source_timestamp_fraction=str(actual),
                        sampling_target_seconds=target, source_pts=decoded.pts,
                        source_time_base=str(decoded.time_base), raw_pts_fraction=str(actual + origin),
                        source_sha256=source['source_fingerprint'],
                        source_frame_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                        source_resolution=[image.height, image.width], preprocessing_resolution=[384, 384],
                        dtype='float16', tensor_shape=list(SHAPE), tokens=729, hidden_dimension=1152,
                        spatial_grid=[27, 27], temporal_position_applied=False,
                        frame_position_id=base + target,
                        optional_batch_projector_temporal_index=(base + target) % 512,
                    )
                    metadata['decode_seconds'] = time.perf_counter()-decode_timer
                    yield image, metadata
                    decode_timer = time.perf_counter()
                    target += 1
                    if target >= hi:
                        break
                if target != hi:
                    raise ValueError(f'Decode count mismatch: expected target {hi}, got {target}')
            if source_state(source['path']) != source['verified_state']:
                raise ValueError('Source changed during decode')
        base += count


def prefetched_batches(inventory, start, end, processor, batch_size, metrics):
    """Two prepared batches maximum; decoder and GPU compute overlap."""
    pending = queue.Queue(maxsize=2)
    stop = threading.Event()

    def put(item):
        while not stop.is_set():
            try:
                pending.put(item, timeout=0.2)
                return
            except queue.Full:
                pass

    def producer():
        try:
            images, rows = [], []
            begin = time.perf_counter()
            metrics['preprocess_compute_seconds'] = 0.0
            for image, row in frames_for_range(inventory, start, end):
                images.append(image)
                rows.append(row)
                if len(images) == batch_size:
                    preprocess_start = time.perf_counter()
                    pixels = processor.preprocess(images, return_tensors='pt')['pixel_values']
                    metrics['preprocess_compute_seconds'] += time.perf_counter()-preprocess_start
                    for im in images:
                        im.close()
                    put((pixels, rows))
                    images, rows = [], []
            if images:
                preprocess_start = time.perf_counter()
                pixels = processor.preprocess(images, return_tensors='pt')['pixel_values']
                metrics['preprocess_compute_seconds'] += time.perf_counter()-preprocess_start
                for im in images:
                    im.close()
                put((pixels, rows))
            metrics['decode_preprocess_wall_seconds'] = time.perf_counter() - begin
            put(None)
        except BaseException:
            put(RuntimeError(traceback.format_exc()))

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    try:
        while True:
            item = pending.get()
            if isinstance(item, BaseException):
                raise item
            if item is None:
                break
            metrics['max_input_queue_backlog'] = max(metrics.get('max_input_queue_backlog', 0), pending.qsize())
            yield item
    finally:
        stop.set()
        thread.join(timeout=10)


@torch.inference_mode()
def encode_range(tower, inventory, start, end, batch_size):
    metrics = dict(frames=end-start, encode_seconds=0.0, h2d_seconds=0.0,
                   d2h_seconds=0.0, batch_latencies_seconds=[])
    chunks, rows = [], []
    begin = time.perf_counter()
    for pixels, metadata in prefetched_batches(inventory, start, end, tower.image_processor, batch_size, metrics):
        if tuple(pixels.shape[1:]) != (3, 384, 384):
            raise ValueError('Unexpected official preprocessing shape')
        for value, row in zip(pixels, metadata):
            row['preprocessed_float32_sha256'] = hashlib.sha256(value.numpy().tobytes()).hexdigest()
        torch.cuda.synchronize(tower.device)
        t = time.perf_counter()
        device_pixels = pixels.to(device=tower.device, dtype=tower.dtype)
        torch.cuda.synchronize(tower.device)
        metrics['h2d_seconds'] += time.perf_counter()-t
        t = time.perf_counter()
        output = tower(device_pixels)
        torch.cuda.synchronize(tower.device)
        latency = time.perf_counter()-t
        metrics['encode_seconds'] += latency
        metrics['batch_latencies_seconds'].append(latency)
        if tuple(output.shape) != (len(metadata), *SHAPE) or output.dtype != torch.float16:
            raise ValueError('Unexpected native memory input dtype/shape')
        if not bool(torch.isfinite(output).all()):
            raise ValueError('Non-finite native features')
        t = time.perf_counter()
        raw = output.cpu().contiguous().view(torch.int16).numpy().copy()
        torch.cuda.synchronize(tower.device)
        metrics['d2h_seconds'] += time.perf_counter()-t
        for value, row in zip(raw, metadata):
            row['feature_sha256'] = hashlib.sha256(value.tobytes()).hexdigest()
        chunks.append(raw)
        rows.extend(metadata)
    metrics['wall_seconds'] = time.perf_counter()-begin
    metrics['frames_per_second'] = len(rows)/metrics['wall_seconds']
    metrics['decode_compute_seconds'] = sum(row['decode_seconds'] for row in rows)
    metrics['decode_frames_per_second'] = len(rows)/metrics['decode_compute_seconds']
    metrics['encoding_frames_per_second'] = len(rows)/metrics['encode_seconds']
    metrics['peak_vram_bytes'] = torch.cuda.max_memory_allocated(tower.device)
    return np.concatenate(chunks), rows, metrics


def worker(gpu, model_path, inventory, batch_size, tasks, results):
    try:
        torch.set_num_threads(1)
        torch.cuda.set_device(gpu)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        tower = load_tower(model_path, gpu)
        results.put(('ready', gpu))
        while True:
            task = tasks.get()
            if task is None:
                return
            start, end = task
            payload, rows, metrics = encode_range(tower, inventory, start, end, batch_size)
            results.put(('result', gpu, start, payload, rows, metrics))
    except BaseException:
        results.put(('error', gpu, traceback.format_exc()))


def make_config(inventory, model_path, frames_per_shard):
    model_path = Path(model_path)
    receipt = json.loads((model_path/'native-vision-download.json').read_text())
    for item in receipt['files']:
        if digest(model_path/item['file']) != item['sha256']:
            raise ValueError('Checkpoint payload changed')
    return dict(schema=SCHEMA, checkpoint=REPO, checkpoint_revision=REVISION,
                code_revision=UPSTREAM_COMMIT, native_source_manifest_sha256=digest(VENDOR_ROOT/'UPSTREAM.json'),
                extractor_source_sha256=digest(__file__),
                checkpoint_receipt=receipt, checkpoint_config_sha256=digest(model_path/'config.json'),
                vision_encoder='checkpoint-embedded SigLIP-so400m-patch14-384; native 26-layer tower hidden_states[-1]',
                projector='tome729_fstw_pemf; MLP applied AFTER memory; excluded from encoder cache',
                cache_boundary='SigLipVisionTower.forward -> MemoryManager.update(image_features[frame_idx])',
                dtype='float16', shape=list(SHAPE), tokens_per_frame=729, bytes_per_frame=FRAME_BYTES,
                frames_per_shard=frames_per_shard, fps=1, byte_order='little',
                preprocessing=dict(resolution=[384,384],resize='bicubic',rescale=1/255,mean=[.5]*3,std=[.5]*3),
                temporal_position_applied=False, learned_spatial_position='native vision patch embedding',
                query_independent=True, memory_constructed=False, inventory=inventory)


class Archive:
    """Validate committed shards, then yield exact tensors in causal order."""

    def __init__(self, root, *, expected_config_id=None, require_complete=True):
        self.root = Path(root)
        self.config = json.loads((self.root/'config.json').read_text())
        self.config_id = hashlib.sha256(canonical(self.config)).hexdigest()
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        if self.manifest['config_id'] != self.config_id:
            raise ValueError('Archive config identifier mismatch')
        if expected_config_id is not None and expected_config_id != self.config_id:
            raise ValueError('Unexpected model/config identifier')
        if self.config['schema'] != SCHEMA or self.config['dtype'] != 'float16' or self.config['shape'] != list(SHAPE):
            raise ValueError('Unsupported archive tensor contract')
        if (self.config.get('checkpoint'),self.config.get('checkpoint_revision')) != (REPO,REVISION):
            raise ValueError('Unexpected native model checkpoint identifier')
        if require_complete and self.manifest['status'] != 'complete':
            raise ValueError('Archive is incomplete')

    def shard(self, shard_id):
        receipt_path = self.root/'receipts'/f'{shard_id:06d}.ready.json'
        receipt = json.loads(receipt_path.read_text())
        if receipt['config_id'] != self.config_id or receipt['shard_id'] != shard_id:
            raise ValueError('Shard model/config identifier mismatch')
        folder = self.root/'shards'/f'shard_{shard_id:06d}'
        if digest(folder/'tokens.bin') != receipt['payload_sha256']:
            raise ValueError('Payload hash mismatch')
        if digest(folder/'index.json') != receipt['index_sha256']:
            raise ValueError('Index hash mismatch')
        index = json.loads((folder/'index.json').read_text())
        rows = index['frames']
        if index['config_id'] != self.config_id or len(rows) != receipt['frame_count']:
            raise ValueError('Shard index identity/count mismatch')
        expected_count = min(self.config['frames_per_shard'],
                             self.manifest['requested_frames'] - shard_id*self.config['frames_per_shard'])
        if len(rows) != expected_count or expected_count < 1:
            raise ValueError('Committed shard has an unexpected frame count')
        if (folder/'tokens.bin').stat().st_size != len(rows)*FRAME_BYTES:
            raise ValueError('Payload byte count mismatch')
        start = shard_id*self.config['frames_per_shard']
        if [r['global_frame_id'] for r in rows] != list(range(start,start+len(rows))):
            raise ValueError('Frame id continuity failure')
        for row in rows:
            if row['tensor_shape'] != list(SHAPE) or row['dtype'] != 'float16':
                raise ValueError('Frame dtype/shape mismatch')
            expected = frame_envelope(self.config['inventory'], row['global_frame_id'])
            for key, value in expected.items():
                if row[key] != value:
                    raise ValueError(f'Frame timestamp/source mismatch: {key}')
        if receipt['first_frame_id'] != start or receipt['last_frame_id'] != start+len(rows)-1:
            raise ValueError('Receipt frame range mismatch')
        if receipt['first_timestamp'] != rows[0]['timestamp_seconds'] or receipt['last_timestamp'] != rows[-1]['timestamp_seconds']:
            raise ValueError('Receipt timestamp mismatch')
        raw = np.memmap(folder/'tokens.bin',mode='r',dtype='<i2',shape=(len(rows),*SHAPE))
        return rows, raw

    def __iter__(self):
        expected_count = self.manifest['requested_frames']
        seen, previous = 0, None
        for shard_id in range(math.ceil(expected_count/self.config['frames_per_shard'])):
            rows, raw = self.shard(shard_id)
            for row, value in zip(rows, raw):
                if row['global_frame_id'] != seen:
                    raise ValueError('Global frame continuity failure')
                timestamp = Fraction(row['timestamp_fraction'])
                if previous is not None and timestamp <= previous:
                    raise ValueError('Timestamp ordering failure')
                if hashlib.sha256(value.tobytes()).hexdigest() != row['feature_sha256']:
                    raise ValueError('Per-frame payload hash mismatch')
                tensor = torch.from_numpy(value.copy()).view(torch.float16)
                yield SimpleNamespace(visual_tokens=tensor, timestamp=row['timestamp_seconds'],
                                      global_frame_id=seen, metadata=row)
                previous = timestamp
                seen += 1
        if seen != expected_count:
            raise ValueError('Archive frame count mismatch')


def frame_envelope(inventory, frame_id):
    base = 0
    for source in inventory['sources']:
        count = source['expected_1fps_frames']
        if frame_id < base+count:
            local = frame_id-base
            # All sources were freshly verified 25fps CFR; sampling at integer PTS.
            t = Fraction(source['global_start_fraction']) + local
            return dict(timestamp_seconds=float(t), timestamp_fraction=str(t),
                        source_timestamp=float(local), source_timestamp_fraction=str(local),
                        day_id=source['day_id'], source_id=source['source_id'])
        base += count
    raise ValueError('Frame outside inventory')


def commit_shard(root, config_id, shard_id, rows, arrays):
    folder = root/'shards'/f'shard_{shard_id:06d}'
    started = time.perf_counter()
    payload = b''.join(a.tobytes(order='C') for a in arrays)
    serialization_seconds = time.perf_counter()-started
    t = time.perf_counter()
    atomic(folder/'tokens.bin',payload)
    write_seconds = time.perf_counter()-t
    index_raw = canonical(dict(config_id=config_id,frames=rows))
    atomic(folder/'index.json',index_raw)
    receipt = dict(config_id=config_id,shard_id=shard_id,frame_count=len(rows),
        first_frame_id=rows[0]['global_frame_id'],last_frame_id=rows[-1]['global_frame_id'],
        first_timestamp=rows[0]['timestamp_seconds'],last_timestamp=rows[-1]['timestamp_seconds'],
        payload_sha256=hashlib.sha256(payload).hexdigest(),index_sha256=hashlib.sha256(index_raw).hexdigest(),
        payload_bytes=len(payload),serialization_seconds=serialization_seconds,
        nas_payload_write_seconds=write_seconds,nas_write_MB_per_second=len(payload)/write_seconds/1e6)
    # Ready receipt is the commit point, after both durable atomic payload writes.
    write_json(root/'receipts'/f'{shard_id:06d}.ready.json',receipt)
    return receipt


@contextmanager
def run_lock(root):
    import fcntl
    ensure_dir(root)
    with (root/'.writer.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def extract(config, model_path, root, *, frames, gpus, batch_size=1, task_frames=25):
    root = Path(root)
    validate_inventory(config['inventory'])
    config_id = hashlib.sha256(canonical(config)).hexdigest()
    with run_lock(root):
        if (root/'config.json').exists():
            if (root/'config.json').read_bytes() != canonical(config):
                raise ValueError('Refusing resume with different config')
        else:
            write_json(root/'config.json',config)
        shard_size = config['frames_per_shard']
        if shard_size % task_frames:
            raise ValueError('Task frames must divide storage shard size')
        manifest = dict(schema=SCHEMA,config_id=config_id,requested_frames=frames,status='running',
                        archive_kind='encoder_cache_only',memory_constructed=False,
                        batch_size=batch_size,task_frames=task_frames,gpus=gpus)
        if (root/'manifest.json').exists():
            old=json.loads((root/'manifest.json').read_text())
            if old['requested_frames']!=frames:
                raise ValueError('Refusing resume with different requested count')
            if old['batch_size']!=batch_size or old['task_frames']!=task_frames:
                raise ValueError('Refusing resume with different batching or task boundaries')
        write_json(root/'manifest.json',manifest)
        existing=Archive(root,require_complete=False)
        resume=0
        missing_receipt_seen=False
        for sid in range(math.ceil(frames/shard_size)):
            ready=root/'receipts'/f'{sid:06d}.ready.json'
            if ready.exists():
                if missing_receipt_seen:
                    raise ValueError('Noncontiguous ready shards; refusing to re-encode committed frames')
                rows,_=existing.shard(sid)
                resume += len(rows)
            else:
                missing_receipt_seen=True
        if resume == frames:
            manifest['status']='complete'
            write_json(root/'manifest.json',manifest)
            return manifest
        (root/'logs').mkdir(exist_ok=True)
        ctx=mp.get_context('spawn')
        result_queue=ctx.Queue(maxsize=len(gpus)*2)
        task_queues=[ctx.Queue(maxsize=2) for _ in gpus]
        workers=[ctx.Process(target=worker,args=(gpu,str(model_path),config['inventory'],batch_size,tq,result_queue))
                 for gpu,tq in zip(gpus,task_queues)]
        for proc in workers:proc.start()
        start_time=time.perf_counter()
        metrics=[]
        telemetry=[]
        stop_monitor=threading.Event()
        def monitor():
            while not stop_monitor.is_set():
                try:
                    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used',
                                                    '--format=csv,noheader,nounits'],text=True,timeout=10)
                    sample=dict(elapsed_seconds=time.perf_counter()-start_time,csv=output.strip())
                    telemetry.append(sample)
                    with (root/'logs'/'gpu-utilization.jsonl').open('ab') as stream:
                        stream.write(canonical(sample))
                except Exception as exc:
                    telemetry.append({'monitor_error':str(exc)})
                stop_monitor.wait(1)
        monitor_thread=threading.Thread(target=monitor,daemon=True)
        monitor_thread.start()
        try:
            ready=set()
            while len(ready)<len(gpus):
                message=result_queue.get(timeout=600)
                if message[0]=='error':raise RuntimeError(message)
                ready.add(message[1])
            warm_time=time.perf_counter()
            tasks=[(s,min(s+task_frames,frames)) for s in range(resume,frames,task_frames)]
            submitted=0
            pending={}
            cursor=resume
            rows_buffer,arrays_buffer=[],[]
            receipts=[]
            max_pending=0
            # Bound the entire in-flight reorder window, including queued results.
            def dispatch():
                nonlocal submitted
                while submitted<len(tasks) and tasks[submitted][0]<cursor+len(gpus)*2*task_frames:
                    task_queues[submitted % len(gpus)].put(tasks[submitted])
                    submitted+=1
            dispatch()
            while cursor<frames:
                try:message=result_queue.get(timeout=60)
                except queue.Empty:
                    if any(not p.is_alive() for p in workers):
                        raise RuntimeError('Encoder worker died')
                    continue
                if message[0]=='error':raise RuntimeError(message)
                _,gpu,start,payload,rows,metric=message
                if start in pending or start<cursor:
                    raise ValueError('Duplicate encoder task')
                pending[start]=(gpu,payload,rows,metric)
                max_pending=max(max_pending,len(pending))
                while cursor in pending:
                    gpu,payload,rows,metric=pending.pop(cursor)
                    count=len(rows)
                    if [r['global_frame_id'] for r in rows]!=list(range(cursor,cursor+count)):
                        raise ValueError('Worker frame reorder/count failure')
                    for row in rows:row['encoder_gpu']=gpu
                    rows_buffer.extend(rows)
                    arrays_buffer.append(payload)
                    metrics.append(dict(gpu=gpu,start_frame=cursor,**metric))
                    cursor+=count
                    if len(rows_buffer)==shard_size or cursor==frames:
                        sid=rows_buffer[0]['global_frame_id']//shard_size
                        receipts.append(commit_shard(root,config_id,sid,rows_buffer,arrays_buffer))
                        rows_buffer,arrays_buffer=[],[]
                        write_json(root/'logs'/'progress.json',dict(committed_frames=cursor,requested_frames=frames,
                            elapsed_seconds=time.perf_counter()-start_time,max_reorder_tasks=max_pending))
                    dispatch()
            elapsed=time.perf_counter()-warm_time
            summary=dict(frames_encoded=frames-resume,resumed_frames=resume,requested_frames=frames,
                steady_wall_seconds=elapsed,total_wall_seconds=time.perf_counter()-start_time,
                steady_frames_per_second=(frames-resume)/elapsed,
                projected_full_run_seconds=config['inventory']['total_frames']/((frames-resume)/elapsed),
                max_reorder_tasks=max_pending,worker_tasks=metrics,shard_receipts=receipts,
                batch_size=batch_size,gpus=gpus,task_frames=task_frames,
                torch_version=torch.__version__)
            write_json(root/'logs'/'extraction-summary.json',summary)
            manifest['status']='complete'
            manifest['encoded_frames']=frames
            write_json(root/'manifest.json',manifest)
        finally:
            stop_monitor.set()
            monitor_thread.join(timeout=15)
            for proc in workers:
                if proc.is_alive():proc.terminate()
            for proc in workers:proc.join(timeout=15)
            for q in task_queues+[result_queue]:
                q.cancel_join_thread()
                q.close()
        return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['extract','validate'])
    parser.add_argument('--output',required=True)
    parser.add_argument('--inventory')
    parser.add_argument('--model-path')
    parser.add_argument('--frames',type=int)
    parser.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    parser.add_argument('--batch-size',type=int,default=1)
    args=parser.parse_args()
    if args.command=='validate':
        archive=Archive(args.output)
        count=sum(1 for _ in archive)
        print(json.dumps(dict(validated_frames=count,config_id=archive.config_id)))
        return
    inventory=json.loads(Path(args.inventory).read_text())
    frames=args.frames or inventory['total_frames']
    if frames<1 or frames>inventory['total_frames']:
        raise ValueError('Invalid requested frame count')
    config=make_config(inventory,args.model_path,100)
    summary=extract(config,args.model_path,args.output,frames=frames,
                    gpus=[int(v) for v in args.gpus.split(',')],batch_size=args.batch_size)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
