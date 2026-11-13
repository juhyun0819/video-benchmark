"""Disk-backed, byte-capped visual memories; no full-history feature archive.

Records contain bf16/fp16/fp32 feature rows and fixed-size provenance. Only
retained slots exist on disk: evicted rows are overwritten, and uniform
compaction shrinks the files. The live bank has no per-token Python index.

The byte cap charges logical shard file lengths, the actual control-file length,
and a conservative canonical representation of resident control state. Python
allocator/FD headers and bounded read/write batches are working RAM, reported
separately by the runner. Immutable snapshot copies are experiment artifacts,
not additional retrievable memory available to an individual policy.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import errno
from functools import lru_cache
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import re

import numpy as np
import torch


DTYPES = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}
BASE_FIELDS = [('global_token_id', '<i8'), ('frame_id', '<i8'),
               ('timestamp', '<f8'), ('token_index', '<i8')]
MAX_BATCH_TOKENS = 1024
_COPY_FALLBACK_ERRNOS = {errno.EXDEV, errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP}


@lru_cache(maxsize=1)
def _libc_copy_file_range():
    # Some Conda Python builds omit os.copy_file_range even on Linux/NFS4.2.
    # ctypes reaches the same kernel operation without loading a new extension.
    import ctypes
    try:
        function = ctypes.CDLL(None, use_errno=True).copy_file_range
    except AttributeError:
        return None
    function.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                         ctypes.c_size_t, ctypes.c_uint)
    function.restype = ctypes.c_ssize_t
    return function


def _copy_file_range(source_fd, destination_fd, count):
    function = getattr(os, 'copy_file_range', None)
    if function is not None:
        return function(source_fd, destination_fd, count)
    function = _libc_copy_file_range()
    if function is None:
        raise OSError(errno.ENOSYS, 'copy_file_range is unavailable')
    result = function(source_fd, None, destination_fd, None, count, 0)
    if result < 0:
        import ctypes
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result


def _int(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _write_all(fd, data, offset):
    data = memoryview(data).cast('B')
    written = 0
    while written < len(data):
        n = os.pwrite(fd, data[written:], offset + written)
        if n <= 0:
            raise IOError('Short retained-record write')
        written += n


def _read_all(fd, size, offset):
    result = bytearray(size)
    view = memoryview(result)
    done = 0
    while done < size:
        part = os.pread(fd, size - done, offset + done)
        if not part:
            raise IOError('Short retained-record read')
        view[done:done+len(part)] = part
        done += len(part)
    return result


class DiskTokenStore:
    """A dense array of fixed-size retained records split across bounded shards.

    ``append`` grows only filled slots. ``write_slots`` overwrites existing
    slots. ``compact_stride`` copies survivors forward in-place then truncates;
    there is never an extra complete bank during compaction. Read/write batch
    sizes are bounded, and no mmap or unbounded userspace cache is used.
    """
    FORMAT = 'videomem_byte_bounded_records_v1'

    def __init__(self, directory, hard_budget_bytes, hidden_dim=3584,
                 dtype='bfloat16', extra_fields=None, control_reserve=4096,
                 shard_bytes=64 * 1024 * 1024, max_open_shards=8):
        self.directory = Path(directory)
        self.hard_budget_bytes = _int(hard_budget_bytes, 'hard_budget_bytes', 1)
        self.hidden_dim = _int(hidden_dim, 'hidden_dim', 1)
        self.control_reserve = _int(control_reserve, 'control_reserve', 512)
        self.max_open_shards = _int(max_open_shards, 'max_open_shards', 1)
        if dtype not in DTYPES:
            raise ValueError('unsupported feature dtype')
        self.dtype = dtype
        self.element_bytes = torch.empty((), dtype=DTYPES[dtype]).element_size()
        self.extra_fields = {}
        for name, field_dtype in sorted((extra_fields or {}).items()):
            if (not re.fullmatch(r'[a-z][a-z0-9_]*', name)
                    or name in {'features', *(n for n, _ in BASE_FIELDS)}):
                raise ValueError('Invalid or duplicate provenance field')
            field_dtype = np.dtype(field_dtype)
            if field_dtype.kind not in 'iu f'.replace(' ', '') or field_dtype.itemsize not in (4, 8):
                raise ValueError('Extra provenance must be scalar 32/64-bit integer/float')
            self.extra_fields[name] = field_dtype.newbyteorder('<').str
        bits = '<u2' if self.element_bytes == 2 else '<u4'
        self.record_dtype = np.dtype([('features', bits, (self.hidden_dim,)), *BASE_FIELDS,
                                      *self.extra_fields.items()], align=False)
        self.record_bytes = self.record_dtype.itemsize
        self.capacity = (self.hard_budget_bytes - self.control_reserve) // self.record_bytes
        if self.capacity < 1:
            raise ValueError('hard budget cannot fit control reserve and one feature record')
        self.shard_records = max(1, _int(shard_bytes, 'shard_bytes', 1) // self.record_bytes)
        self.count = 0
        self._fds = OrderedDict()
        self._external_control = {}
        self._control_file_bytes = 0
        self._closed = False
        self.directory.mkdir(parents=True, exist_ok=False)
        self._persist_control()

    def _path(self, shard):
        return self.directory / f'records_{shard:08d}.bin'

    @property
    def control(self):
        """Owner counters/provenance saved with the retained records."""
        return deepcopy(self._external_control)

    def _fd(self, shard, create=False):
        if self._closed:
            raise RuntimeError('Store is closed')
        if shard in self._fds:
            fd = self._fds.pop(shard)
            self._fds[shard] = fd
            return fd
        fd = os.open(self._path(shard), os.O_RDWR | (os.O_CREAT if create else 0), 0o600)
        self._fds[shard] = fd
        while len(self._fds) > self.max_open_shards:
            _, old = self._fds.popitem(last=False)
            os.close(old)
        return fd

    def _control(self):
        return {'format': self.FORMAT, 'hard_budget_bytes': self.hard_budget_bytes,
                'hidden_dim': self.hidden_dim, 'dtype': self.dtype,
                'extra_fields': self.extra_fields, 'record_bytes': self.record_bytes,
                'control_reserve': self.control_reserve, 'capacity': self.capacity,
                'shard_records': self.shard_records, 'count': self.count,
                'max_open_shards': self.max_open_shards, 'owner': self._external_control}

    def set_control(self, control):
        previous = self._external_control
        self._external_control = deepcopy(control)
        try:
            self._check_control()
        except BaseException:
            self._external_control = previous
            raise

    def _check_control(self):
        size = len(_json(self._control()))
        # Covers both the owner counters/RNG and the store's canonical copy.
        # The separately measured on-disk control file is charged as well.
        reserved = 2 * size + max(size, self._control_file_bytes)
        if reserved > self.control_reserve:
            raise ValueError(f'Control payload requires {reserved} bytes, reserve is {self.control_reserve}')
        return size

    def _persist_control(self):
        self._check_control()
        data = _json(self._control())
        # A live control file is not an immutable checkpoint. Write in-place so
        # atomic replacement cannot briefly double its persistent disk charge.
        with (self.directory / 'control.json').open('wb', buffering=0) as handle:
            if handle.write(data) != len(data):
                raise IOError('Short control-file write')
            os.fsync(handle.fileno())
        self._control_file_bytes = len(data)

    def stats(self):
        current_control_bytes = self._check_control()
        record_bytes = self.count * self.record_bytes
        disk_bytes = record_bytes + self._control_file_bytes
        resident_control_bytes = 2 * current_control_bytes
        actual = disk_bytes + resident_control_bytes
        if actual > self.hard_budget_bytes:
            raise AssertionError(f'Persistent byte cap exceeded by {actual-self.hard_budget_bytes} bytes')
        return {'capacity': self.capacity, 'count': self.count,
                'retained_visual_tokens': self.count,
                'feature_payload_bytes': self.count * self.hidden_dim * self.element_bytes,
                'payload_bytes': self.count * self.hidden_dim * self.element_bytes,
                'record_metadata_bytes': self.count * (self.record_bytes - self.hidden_dim*self.element_bytes),
                'persistent_disk_bytes': disk_bytes,
                'control_file_bytes': self._control_file_bytes,
                'resident_control_payload_bytes': resident_control_bytes,
                'actual_memory_bytes': actual, 'hard_budget_bytes': self.hard_budget_bytes,
                'control_reserve_bytes': self.control_reserve, 'record_bytes': self.record_bytes,
                'dtype': self.dtype, 'hidden_dimension': self.hidden_dim,
                'tensor_shape': [self.count, self.hidden_dim],
                'backend': 'disk', 'byte_order': 'little',
                'peak_batch_tokens_limit': MAX_BATCH_TOKENS,
                'accounting': 'retained shard lengths + actual control file + conservative canonical resident controls; Python headers and bounded batch buffers are working RAM'}

    def _records(self, features, metadata):
        if (not isinstance(features, torch.Tensor) or features.device.type != 'cpu'
                or features.ndim != 2 or features.shape[1] != self.hidden_dim
                or features.dtype != DTYPES[self.dtype]):
            raise ValueError('Feature rows must have the configured CPU dtype and hidden dimension')
        n = len(features)
        expected = {name for name, _ in BASE_FIELDS} | set(self.extra_fields)
        if set(metadata) != expected:
            raise ValueError(f'Metadata fields must be {sorted(expected)}')
        records = np.empty(n, dtype=self.record_dtype)
        bits_dtype = torch.uint16 if self.element_bytes == 2 else torch.uint32
        records['features'] = features.detach().contiguous().view(bits_dtype).numpy()
        for key in expected:
            value = metadata[key]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            value = np.asarray(value)
            if value.shape != (n,):
                raise ValueError('Every provenance column must have one scalar per feature token')
            if value.dtype.kind == 'f' and not np.isfinite(value).all():
                raise ValueError('Provenance floats must be finite')
            records[key] = value
        return records

    def _write_contiguous(self, start, records):
        offset = 0
        while offset < len(records):
            slot = start + offset
            shard, within = divmod(slot, self.shard_records)
            count = min(len(records)-offset, self.shard_records-within)
            _write_all(self._fd(shard, create=True), records[offset:offset+count], within*self.record_bytes)
            offset += count

    def append(self, features, metadata):
        n = len(features)
        if self.count + n > self.capacity:
            raise ValueError('append would exceed retained record capacity')
        start = self.count
        for lo in range(0, n, MAX_BATCH_TOKENS):
            hi = min(n, lo+MAX_BATCH_TOKENS)
            part = {k:v[lo:hi] for k,v in metadata.items()}
            records = self._records(features[lo:hi], part)
            self._write_contiguous(self.count, records)
            self.count += len(records)
        self.stats()
        return start

    def write_slots(self, slots, features, metadata):
        slots = np.asarray(slots, dtype=np.int64)
        if slots.shape != (len(features),) or (len(slots) and (slots.min()<0 or slots.max()>=self.count)):
            raise ValueError('write_slots may only overwrite existing retained slots')
        if len(np.unique(slots)) != len(slots):
            raise ValueError('write_slots requires unique destination slots')
        order = np.argsort(slots)
        for lo in range(0, len(order), MAX_BATCH_TOKENS):
            chosen = order[lo:lo+MAX_BATCH_TOKENS]
            tensor_indices = torch.from_numpy(chosen)
            part = {k:(v[tensor_indices] if isinstance(v,torch.Tensor) else np.asarray(v)[chosen])
                    for k,v in metadata.items()}
            records = self._records(features.index_select(0,tensor_indices), part)
            selected = slots[chosen]
            start = 0
            for end in range(1,len(selected)+1):
                if end == len(selected) or selected[end] != selected[end-1]+1:
                    self._write_contiguous(int(selected[start]), records[start:end])
                    start = end
        self.stats()

    def _read_records(self, slots):
        slots = np.asarray(slots,dtype=np.int64)
        if slots.ndim != 1 or len(slots)>MAX_BATCH_TOKENS:
            raise ValueError(f'Read at most {MAX_BATCH_TOKENS} slots per bounded batch')
        if len(slots) and (slots.min()<0 or slots.max()>=self.count):
            raise ValueError('Read slot outside retained bank')
        records = np.empty(len(slots),dtype=self.record_dtype)
        order = np.argsort(slots)
        selected = slots[order]
        start = 0
        for end in range(1,len(selected)+1):
            if (end == len(selected) or selected[end] != selected[end-1]+1
                    or selected[end]//self.shard_records != selected[start]//self.shard_records):
                shard,within = divmod(int(selected[start]),self.shard_records)
                raw = _read_all(self._fd(shard), (end-start)*self.record_bytes, within*self.record_bytes)
                records[order[start:end]] = np.frombuffer(raw,dtype=self.record_dtype)
                start = end
        return records

    def read_slots(self, slots):
        records = self._read_records(slots)
        feature_bits = torch.from_numpy(records['features'].copy())
        features = feature_bits.view(DTYPES[self.dtype])
        metadata = {key:torch.from_numpy(records[key].copy()) for key in records.dtype.names if key!='features'}
        return {'features':features, 'metadata':metadata}

    def iter_batches(self, max_tokens=MAX_BATCH_TOKENS):
        size = min(_int(max_tokens,'max_tokens',1),MAX_BATCH_TOKENS)
        for start in range(0,self.count,size):
            yield self.read_slots(np.arange(start,min(start+size,self.count),dtype=np.int64))

    def truncate(self, count):
        count = _int(count,'count')
        if count>self.count:
            raise ValueError('truncate cannot grow retained bank')
        previous_shards = (self.count+self.shard_records-1)//self.shard_records
        remaining_shards = (count+self.shard_records-1)//self.shard_records
        for shard in range(remaining_shards,previous_shards):
            fd = self._fds.pop(shard,None)
            if fd is not None:
                os.close(fd)
            self._path(shard).unlink()
        if count and count%self.shard_records:
            os.ftruncate(self._fd(remaining_shards-1), (count%self.shard_records)*self.record_bytes)
        self.count = count
        self.stats()

    def compact_stride(self, factor):
        factor = _int(factor,'factor',2)
        previous = self.count
        destination = 0
        # Ascending source slots always follow destination slots. Every batch is
        # read before overwrite, so no survivor can be clobbered prematurely.
        for start in range(0,previous,MAX_BATCH_TOKENS):
            end = min(previous,start+MAX_BATCH_TOKENS)
            if ((start+factor-1)//factor)*factor >= end:
                continue
            # Reading each surviving strided token separately would cause
            # millions of tiny NAS reads at a full-stream budget. Read bounded
            # contiguous blocks, then select their surviving rows in RAM.
            slots = np.arange(start,end,dtype=np.int64)
            records = self._read_records(slots)
            survivors = records[slots % factor == 0]
            self._write_contiguous(destination,survivors)
            destination += len(survivors)
        self.truncate(destination)

    def flush(self):
        for fd in self._fds.values():
            os.fsync(fd)
        self._persist_control()
        self.stats()

    def close(self):
        if not self._closed:
            self.flush()
            for fd in self._fds.values():
                os.close(fd)
            self._fds.clear()
            self._closed = True

    @staticmethod
    def _copy_hashed(source,destination):
        """Copy a stable shard, preferring server-side NFS COPY when supported.

        A successful kernel copy is followed by one source hash pass. The
        existing snapshot verifier independently hashes the destination. If a
        filesystem rejects COPY, restart the original streaming copy from byte
        zero, including when rejection follows a partially completed copy.
        """
        digest = hashlib.sha256()
        total = 0
        source, destination = Path(source), Path(destination)
        created = False
        identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        try:
            with source.open('rb',buffering=0) as inp, destination.open('xb',buffering=0) as out:
                created = True
                initial = os.fstat(inp.fileno())
                expected = initial.st_size
                copied = 0
                kernel_copy = True
                while copied < expected:
                    try:
                        count = min(expected-copied, 64*1024*1024)
                        written = _copy_file_range(inp.fileno(), out.fileno(), count)
                    except OSError as exc:
                        if exc.errno == errno.EINTR:
                            continue
                        if exc.errno not in _COPY_FALLBACK_ERRNOS:
                            raise
                        kernel_copy = False
                        out.seek(0)
                        out.truncate(0)
                        break
                    if written <= 0 or written > count:
                        raise IOError('Short immutable snapshot kernel copy')
                    copied += written
                inp.seek(0)
                while True:
                    chunk = inp.read(8*1024*1024)
                    if not chunk:
                        break
                    if not kernel_copy:
                        offset = 0
                        while offset < len(chunk):
                            written = out.write(memoryview(chunk)[offset:])
                            if not written:
                                raise IOError('Short immutable snapshot write')
                            offset += written
                    digest.update(chunk)
                    total += len(chunk)
                os.fsync(out.fileno())
                if (total != expected
                        or identity(os.fstat(inp.fileno())) != identity(initial)
                        or identity(source.stat()) != identity(initial)):
                    raise IOError('Immutable snapshot source changed during copy')
            # NFS COPY may keep destination attributes cached at size zero even
            # after fsync; closing the writer revalidates the server result.
            if destination.stat().st_size != expected:
                raise IOError('Immutable snapshot destination size mismatch')
        except BaseException:
            if created:
                destination.unlink(missing_ok=True)
            raise
        return total,digest.hexdigest()

    def snapshot(self,directory,control=None):
        if control is not None:
            self.set_control(control)
        self.flush()
        directory = Path(directory)
        directory.mkdir(parents=True,exist_ok=False)
        files = []
        paths = [self._path(i) for i in range((self.count+self.shard_records-1)//self.shard_records)]
        paths.append(self.directory/'control.json')
        for source in paths:
            size,digest = self._copy_hashed(source,directory/source.name)
            files.append({'name':source.name,'bytes':size,'sha256':digest})
        manifest = {'format':self.FORMAT,'live_control':self._control(),
                    'live_stats':self.stats(),'files':files,
                    'feature_and_provenance_storage':'fixed binary records; iterate with DiskTokenStore.load(...).iter_batches()',
                    'snapshot_copy_is_outside_live_policy_budget':True}
        (directory/'snapshot.json').write_bytes(_json(manifest))
        total = sum(f['bytes'] for f in files)+(directory/'snapshot.json').stat().st_size
        return {'path':str(directory),'snapshot_serialized_bytes':total,
                'snapshot_shards':len(paths)-1,**self.stats()}

    @classmethod
    def load(cls,snapshot_directory,directory):
        snapshot_directory = Path(snapshot_directory)
        manifest = json.loads((snapshot_directory/'snapshot.json').read_text())
        if manifest.get('format')!=cls.FORMAT:
            raise ValueError('Unknown retained-state snapshot format')
        info = manifest['live_control']
        result = cls(directory,info['hard_budget_bytes'],hidden_dim=info['hidden_dim'],
                     dtype=info['dtype'],extra_fields=info['extra_fields'],
                     control_reserve=info['control_reserve'],
                     shard_bytes=info['shard_records']*info['record_bytes'],
                     max_open_shards=info['max_open_shards'])
        try:
            expected_names = {f'records_{i:08d}.bin' for i in range((info['count']+info['shard_records']-1)//info['shard_records'])}
            expected_names.add('control.json')
            if {f['name'] for f in manifest['files']} != expected_names or len(manifest['files'])!=len(expected_names):
                raise ValueError('Snapshot shard inventory mismatch')
            for row in manifest['files']:
                name = row['name']
                if name=='control.json':
                    data = (snapshot_directory/name).read_bytes()
                    if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
                        raise ValueError('Snapshot control checksum mismatch')
                    if json.loads(data)!=info:
                        raise ValueError('Snapshot control differs from manifest')
                    continue
                if not re.fullmatch(r'records_[0-9]{8}\.bin',name):
                    raise ValueError('Unsafe snapshot filename')
                size,digest = cls._copy_hashed(snapshot_directory/name,result.directory/name)
                if size!=row['bytes'] or digest!=row['sha256']:
                    raise ValueError('Snapshot shard checksum mismatch')
            result.count = _int(info['count'],'count')
            if result.count>result.capacity or info['capacity']!=result.capacity:
                raise ValueError('Snapshot exceeds declared token capacity')
            expected_bytes = result.count*result.record_bytes
            if sum(f['bytes'] for f in manifest['files'] if f['name']!='control.json')!=expected_bytes:
                raise ValueError('Snapshot record byte length mismatch')
            result.set_control(info['owner'])
            result.flush()
            return result
        except BaseException:
            for fd in result._fds.values():
                os.close(fd)
            result._fds.clear()
            result._closed=True
            raise


class SnapshotReader(DiskTokenStore):
    """Read immutable snapshot batches without copying a large bank to RAM/disk.

    ``verify()`` hashes every shard in bounded buffers; normal iteration reads
    only requested batches. This is a reader for the retained-state snapshot,
    not a full-history feature archive. Extra method-specific files (e.g. native
    FluxMem tier state) are verified by that method's own loader.
    """
    def __init__(self, directory):
        self.directory=Path(directory)
        self.manifest=json.loads((self.directory/'snapshot.json').read_text())
        if self.manifest.get('format')!=self.FORMAT:
            raise ValueError('Unknown retained-state snapshot format')
        info=self.manifest['live_control']
        self.hard_budget_bytes=_int(info['hard_budget_bytes'],'hard_budget_bytes',1)
        self.hidden_dim=_int(info['hidden_dim'],'hidden_dim',1)
        self.control_reserve=_int(info['control_reserve'],'control_reserve',512)
        self.max_open_shards=_int(info['max_open_shards'],'max_open_shards',1)
        self.dtype=info['dtype']
        if self.dtype not in DTYPES:
            raise ValueError('Unsupported snapshot feature dtype')
        self.element_bytes=torch.empty((),dtype=DTYPES[self.dtype]).element_size()
        self.extra_fields=dict(sorted(info['extra_fields'].items()))
        bits='<u2' if self.element_bytes==2 else '<u4'
        self.record_dtype=np.dtype([('features',bits,(self.hidden_dim,)),*BASE_FIELDS,*self.extra_fields.items()],align=False)
        self.record_bytes=self.record_dtype.itemsize
        self.capacity=(self.hard_budget_bytes-self.control_reserve)//self.record_bytes
        self.count=_int(info['count'],'count')
        self.shard_records=_int(info['shard_records'],'shard_records',1)
        if info['record_bytes']!=self.record_bytes or info['capacity']!=self.capacity or self.count>self.capacity:
            raise ValueError('Snapshot record layout/capacity mismatch')
        self._external_control=deepcopy(info['owner'])
        self._control_file_bytes=(self.directory/'control.json').stat().st_size
        self._fds=OrderedDict()
        self._closed=False
        names={f'records_{i:08d}.bin' for i in range((self.count+self.shard_records-1)//self.shard_records)}|{'control.json'}
        rows=self.manifest['files']
        if {r['name'] for r in rows}!=names or len(rows)!=len(names):
            raise ValueError('Snapshot shard inventory mismatch')
        self.stats()

    def _fd(self,shard,create=False):
        if create or self._closed:
            raise RuntimeError('Immutable snapshot reader cannot create files or read after close')
        if shard in self._fds:
            fd=self._fds.pop(shard)
            self._fds[shard]=fd
            return fd
        fd=os.open(self._path(shard),os.O_RDONLY)
        self._fds[shard]=fd
        while len(self._fds)>self.max_open_shards:
            _,old=self._fds.popitem(last=False)
            os.close(old)
        return fd

    def verify(self):
        verified_bytes=0
        for row in self.manifest['files']:
            name=row['name']
            path=self.directory/name
            before=path.stat()
            expected=row['bytes']
            if name!='control.json':
                shard=int(name[8:16])
                expected_records=min(self.shard_records,self.count-shard*self.shard_records)
                if expected!=expected_records*self.record_bytes:
                    raise ValueError('Snapshot shard record length mismatch')
            if before.st_size!=expected:
                raise ValueError('Snapshot actual file length differs from manifest')
            digest=hashlib.sha256()
            with path.open('rb',buffering=0) as handle:
                while True:
                    chunk=handle.read(8*1024*1024)
                    if not chunk:
                        break
                    digest.update(chunk)
            after=path.stat()
            if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
                raise ValueError('Snapshot changed during verification')
            if digest.hexdigest()!=row['sha256']:
                raise ValueError('Snapshot checksum mismatch')
            verified_bytes+=before.st_size
        if json.loads((self.directory/'control.json').read_text())!=self.manifest['live_control']:
            raise ValueError('Snapshot control differs from manifest')
        return {'passed':True,'verified_files':len(self.manifest['files']),
                'verified_bytes':verified_bytes,'retained_visual_tokens':self.count,
                'verification':'SHA256 of every record shard and control file; no copied working bank',
                'method_specific_extra_files_verified':False}

    def _immutable(self,*args,**kwargs):
        raise PermissionError('SnapshotReader is read-only')

    append=write_slots=truncate=compact_stride=set_control=snapshot=_immutable

    def flush(self):
        if self._closed:
            raise RuntimeError('Snapshot reader is closed')

    def close(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
        self._closed=True

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()


class ByteBudgetMemory:
    """FIFO, dyadic uniform and Algorithm-R memories over one shared feature stream."""
    def __init__(self,method,hard_budget_bytes,retention_ratio,hidden_dim=3584,
                 tokens_per_frame=676,seed=0,directory=None,backend='disk',
                 dtype='bfloat16',**store_options):
        if method not in ('fifo','uniform','reservoir'):
            raise ValueError('Method must be fifo, uniform or reservoir')
        if backend!='disk' or directory is None:
            raise ValueError('A disk directory is required for this bounded backend')
        if not isinstance(retention_ratio,Real) or not 0<float(retention_ratio)<=1:
            raise ValueError('retention_ratio must be in (0,1]')
        self.method=method
        self.retention_ratio=float(retention_ratio)
        self.tokens_per_frame=_int(tokens_per_frame,'tokens_per_frame',1)
        self.seed=_int(seed,'seed')
        self.store=DiskTokenStore(directory,hard_budget_bytes,hidden_dim=hidden_dim,dtype=dtype,**store_options)
        self._rng=np.random.Generator(np.random.PCG64(self.seed)) if method=='reservoir' else None
        self._frames=0
        self._tokens=0
        self._bytes=0
        self._cursor=0
        self._stride=1
        self._last_timestamp=None
        self._sync_control()

    def _control(self):
        return {'method':self.method,'retention_ratio':self.retention_ratio,
                'tokens_per_frame':self.tokens_per_frame,'seed':self.seed,
                'total_frames_seen':self._frames,'total_visual_tokens_seen':self._tokens,
                'cumulative_feature_bytes':self._bytes,'fifo_cursor':self._cursor,
                'uniform_stride':self._stride,'last_timestamp':self._last_timestamp,
                'rng_state':deepcopy(self._rng.bit_generator.state) if self._rng is not None else None}

    def _sync_control(self):
        self.store.set_control(self._control())

    def _metadata(self,indices,frame_id,timestamp):
        return {'global_token_id':self._tokens+indices,
                'frame_id':np.full(len(indices),frame_id,dtype=np.int64),
                'timestamp':np.full(len(indices),timestamp,dtype=np.float64),
                'token_index':indices}

    @torch.no_grad()
    def update(self,features,frame_id,timestamp):
        frame_id=_int(frame_id,'frame_id')
        if frame_id!=self._frames:
            raise ValueError('Byte memory requires consecutive frame IDs beginning at zero')
        if (not isinstance(timestamp,Real) or isinstance(timestamp,bool) or not math.isfinite(timestamp)
                or timestamp<0 or (self._last_timestamp is not None and timestamp<=self._last_timestamp)):
            raise ValueError('Timestamp must strictly increase and remain finite/nonnegative')
        if (not isinstance(features,torch.Tensor) or tuple(features.shape)!=(self.tokens_per_frame,self.store.hidden_dim)
                or features.dtype!=DTYPES[self.store.dtype] or features.device.type!='cpu'):
            raise ValueError('Feature shape/dtype must match the fixed shared representation')
        timestamp=float(timestamp)
        n=len(features)
        cap=self.store.capacity
        if self.method=='fifo':
            if n>=cap:
                indices=np.arange(n-cap,n,dtype=np.int64)
                selected=features.index_select(0,torch.from_numpy(indices))
                metadata=self._metadata(indices,frame_id,timestamp)
                if self.store.count<cap:
                    self.store.truncate(0)
                    self.store.append(selected,metadata)
                else:
                    self.store.write_slots(np.arange(cap),selected,metadata)
                self._cursor=0
            else:
                fill=min(n,cap-self.store.count)
                if fill:
                    indices=np.arange(fill,dtype=np.int64)
                    self.store.append(features[:fill],self._metadata(indices,frame_id,timestamp))
                if fill<n:
                    indices=np.arange(fill,n,dtype=np.int64)
                    slots=(np.arange(n-fill,dtype=np.int64)+self._cursor)%cap
                    self.store.write_slots(slots,features[fill:],self._metadata(indices,frame_id,timestamp))
                    self._cursor=(self._cursor+n-fill)%cap
        elif self.method=='uniform':
            stride=self._stride
            end=self._tokens+n
            while (end-1)//stride+1>cap:
                stride*=2
            if stride!=self._stride:
                self.store.compact_stride(stride//self._stride)
                self._stride=stride
            first=((self._tokens+stride-1)//stride)*stride
            indices=np.arange(first,end,stride,dtype=np.int64)-self._tokens
            if len(indices):
                self.store.append(features.index_select(0,torch.from_numpy(indices)),self._metadata(indices,frame_id,timestamp))
        else:
            fill=min(n,cap-self.store.count)
            if fill:
                indices=np.arange(fill,dtype=np.int64)
                self.store.append(features[:fill],self._metadata(indices,frame_id,timestamp))
            indices=np.arange(fill,n,dtype=np.int64)
            if len(indices):
                draws=self._rng.integers(0,self._tokens+indices+1,dtype=np.int64)
                keep=draws<cap
                slots,chosen=draws[keep],indices[keep]
                if len(slots):
                    _,rev=np.unique(slots[::-1],return_index=True)
                    last=len(slots)-1-rev
                    selected=chosen[last]
                    self.store.write_slots(slots[last],features.index_select(0,torch.from_numpy(selected)),self._metadata(selected,frame_id,timestamp))
        self._frames+=1
        self._tokens+=n
        self._bytes+=features.numel()*features.element_size()
        self._last_timestamp=timestamp
        self._sync_control()
        self.stats()  # O(1), fail-fast byte cap on every frame.

    def stats(self):
        value=self.store.stats()
        count=self.store.count
        if not count:
            items=0
        elif self.method=='fifo':
            items=(self._tokens-1)//self.tokens_per_frame-(self._tokens-count)//self.tokens_per_frame+1
        elif self.method=='uniform':
            items=min(count,((count-1)*self._stride)//self.tokens_per_frame+1)
        else:
            items=None  # Exact per-frame indices are in records, never an unbounded Python set.
        return {**value,'method':self.method,'retention_ratio':self.retention_ratio,
                'total_frames_seen':self._frames,'total_visual_tokens_seen':self._tokens,
                'total_tokens_seen':self._tokens,'cumulative_feature_bytes':self._bytes,
                'retained_items':items,'retained_items_note':'reservoir distinct-frame count not maintained; stream metadata for exact membership',
                'approximate_frame_equivalent':count/self.tokens_per_frame,
                'target_visual_tokens':self.store.capacity,
                'compression_ratio':self._bytes/value['actual_memory_bytes'],
                'effective_retention_ratio':value['actual_memory_bytes']/self._bytes if self._bytes else 0.0,
                'payload_retention_ratio':value['feature_payload_bytes']/self._bytes if self._bytes else 0.0,
                'budget_utilization':value['actual_memory_bytes']/value['hard_budget_bytes'],
                'last_timestamp':self._last_timestamp,
                'uniform_stride':self._stride if self.method=='uniform' else None,
                'retained_frame_indices':'embedded frame_id column in retained record shards',
                'retained_timestamps':'embedded timestamp column in retained record shards'}

    def iter_batches(self,max_tokens=MAX_BATCH_TOKENS):
        if self.method=='fifo' and self.store.count==self.store.capacity and self._cursor:
            size=min(_int(max_tokens,'max_tokens',1),MAX_BATCH_TOKENS)
            for start,end in [(self._cursor,self.store.count),(0,self._cursor)]:
                for low in range(start,end,size):
                    yield self.store.read_slots(np.arange(low,min(low+size,end),dtype=np.int64))
        else:
            yield from self.store.iter_batches(max_tokens)

    def snapshot(self,directory,timestamp):
        if (not isinstance(timestamp,Real) or not math.isfinite(timestamp)
                or self._last_timestamp is None or timestamp<=self._last_timestamp):
            raise ValueError('Snapshot cutoff must be strictly later than observed frames')
        self._sync_control()
        copied=self.store.snapshot(directory)
        metadata={**self.stats(),'timestamp':float(timestamp),'timestamp_hours':float(timestamp)/3600,
                  'snapshot_serialized_bytes':copied['snapshot_serialized_bytes'],
                  'snapshot_shards':copied['snapshot_shards'],'path':str(directory)}
        path=Path(directory)/'metadata.json'
        path.write_bytes(_json(metadata))
        metadata['snapshot_serialized_bytes']+=path.stat().st_size
        return metadata

    @classmethod
    def load(cls,snapshot_directory,directory):
        store=DiskTokenStore.load(snapshot_directory,directory)
        owner=store._external_control
        if owner.get('method') not in ('fifo','uniform','reservoir'):
            store.close()
            raise ValueError('Snapshot owner is not a baseline byte memory')
        result=cls.__new__(cls)
        result.store=store
        result.method=owner['method']
        result.retention_ratio=owner['retention_ratio']
        result.tokens_per_frame=owner['tokens_per_frame']
        result.seed=owner['seed']
        result._frames=owner['total_frames_seen']
        result._tokens=owner['total_visual_tokens_seen']
        result._bytes=owner['cumulative_feature_bytes']
        result._cursor=owner['fifo_cursor']
        result._stride=owner['uniform_stride']
        result._last_timestamp=owner['last_timestamp']
        result._rng=np.random.Generator(np.random.PCG64(result.seed)) if result.method=='reservoir' else None
        if result._rng is not None:
            result._rng.bit_generator.state=owner['rng_state']
        result.stats()
        return result

    def flush(self):
        self._sync_control()
        self.store.flush()

    def close(self):
        if not self.store._closed:
            self.flush()
            self.store.close()
