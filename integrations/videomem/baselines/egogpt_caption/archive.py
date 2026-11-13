"""Atomic caption records, verifiable receipts and strict resume identity."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

OFFICIAL_MAX_NEW_TOKENS = 4096


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+'\n').encode()


def object_hash(value):
    return hashlib.sha256(encode(value)).hexdigest()


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.tmp',delete=False) as f:
            temporary=Path(f.name);f.write(encode(value));f.flush();os.fsync(f.fileno())
        os.replace(temporary,path)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def finish_reason(tokens,eos,*,max_new_tokens=None,generation_returned_normally=False):
    """Verify a native batch-one EOS stop or the official bounded normal return.

    EgoRAG returns decoded text at its 4096-token limit even without EOS. A
    length stop is valid only with explicit evidence that this exact call
    returned normally; short non-EOS outputs remain incomplete.
    """
    eos=eos if isinstance(eos,list) else [eos]
    if (not isinstance(tokens,list) or not tokens or not eos
            or any(type(token) is not int or token<0 for token in tokens+eos)
            or len(tokens)>OFFICIAL_MAX_NEW_TOKENS
            or any(token in eos for token in tokens[:-1])):
        return 'incomplete'
    if max_new_tokens is not None and (type(max_new_tokens) is not int
                                      or max_new_tokens!=OFFICIAL_MAX_NEW_TOKENS):
        return 'incomplete'
    if tokens[-1] in eos:
        return 'eos'
    if (generation_returned_normally is True
            and max_new_tokens==OFFICIAL_MAX_NEW_TOKENS
            and len(tokens)==OFFICIAL_MAX_NEW_TOKENS):
        return 'length'
    return 'incomplete'


def is_complete(text,tokens,eos,*,max_new_tokens=None,generation_returned_normally=False):
    return bool(isinstance(text,str) and text.strip()
                and finish_reason(tokens,eos,max_new_tokens=max_new_tokens,
                                  generation_returned_normally=generation_returned_normally)
                in ('eos','length'))


def save_record(root,record):
    root=Path(root);clip_id=record['clip_id']
    if Path(clip_id).name != clip_id: raise ValueError('Invalid clip ID')
    # A retry must never leave an older receipt pointing at a newer payload.
    # Missing receipt means uncommitted and is safely retried after interruption.
    (root/'receipts'/f'{clip_id}.json').unlink(missing_ok=True)
    atomic_json(root/'captions'/f'{clip_id}.json',record)
    atomic_json(root/'receipts'/f'{clip_id}.json',{'clip_id':clip_id,'job_hash':record['job_hash'],
                'sha256':object_hash(record),'status':record['status']})


def load_complete(root,clip_id,job_hash):
    root=Path(root)
    record_path=root/'captions'/f'{clip_id}.json'
    receipt_path=root/'receipts'/f'{clip_id}.json'
    if not receipt_path.exists() or not record_path.exists(): return None
    receipt=json.loads(receipt_path.read_text())
    data=record_path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=receipt['sha256']: raise ValueError('Caption hash mismatch')
    record=json.loads(data)
    if record['job_hash']!=job_hash or receipt['job_hash']!=job_hash: raise ValueError('Caption job mismatch')
    if record['clip_id']!=clip_id or receipt['clip_id']!=clip_id: raise ValueError('Caption clip mismatch')
    reason=finish_reason(record.get('generated_token_ids',[]),record.get('eos_token_id'),
                         max_new_tokens=record.get('max_new_tokens'),
                         generation_returned_normally=record.get('generation_returned_normally',False))
    # Legacy EOS records remain readable. A length stop requires all explicit
    # evidence and reporting fields; contradictory metadata never counts as done.
    metadata_matches=(record.get('finish_reason','eos')==reason
                      and record.get('truncated',False) is (reason=='length')
                      and ('generation_returned_normally' not in record
                           or record['generation_returned_normally'] is True)
                      and ('max_new_tokens' not in record
                           or (type(record['max_new_tokens']) is int
                               and record['max_new_tokens']==OFFICIAL_MAX_NEW_TOKENS)))
    text=record.get('caption','')
    done=(record['status']=='complete' and receipt['status']=='complete'
          and isinstance(text,str) and bool(text.strip())
          and reason in ('eos','length') and metadata_matches)
    return record if done else None
