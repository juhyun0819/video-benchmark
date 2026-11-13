"""Small query interface over retained visual evidence, separate from building.

Scores use normalized pooled visual features. A query must use the same encoder
feature space. This is an evidence-retrieval MVP, not a text embedding model or
a QA generator. Historical queries require an appropriately early checkpoint.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _owned(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _owned(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_owned(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_owned(item) for item in value)
    return value


def _cutoff(memory, as_of):
    last = memory.state_dict()['control'].get('last_timestamp')
    if as_of is not None:
        if isinstance(as_of, bool) or not isinstance(as_of, (int, float)) or not math.isfinite(as_of):
            raise ValueError('as_of must be a finite timestamp')
        if last is not None and as_of < last:
            raise ValueError('Historical as_of precedes checkpoint build time; load an earlier snapshot')
    return last


def _query_vector(features):
    if (not isinstance(features, torch.Tensor) or features.device.type != 'cpu'
            or features.ndim not in (1, 2) or not features.is_floating_point()
            or features.numel() == 0 or not torch.isfinite(features).all()):
        raise ValueError('Query must be finite nonempty CPU visual features [D] or [tokens,D]')
    value = features.float().mean(dim=0) if features.ndim == 2 else features.float()
    return F.normalize(value, dim=0)


def retrieve(memory, query_features, k=5, as_of=None):
    """Return owned selected features, scores and original temporal provenance.

    Ranking is deterministic: cosine descending, timestamp ascending, evidence
    order as the final tie-break. Build state is not changed or supplemented by
    query features. The whole checkpoint must precede as_of, including learned
    routines; filtering timestamps alone would leak later consolidation.
    """
    if type(k) is not int or k < 0:
        raise ValueError('k must be a nonnegative integer')
    last = _cutoff(memory, as_of)
    query = _query_vector(query_features)
    if k == 0:
        return []
    scored = []
    for order, record in enumerate(memory.retained_evidence()):
        features = record['features']
        if (features.ndim != 2 or features.shape[1] != query.shape[0]
                or not features.is_floating_point() or not torch.isfinite(features).all()):
            raise ValueError('Retained evidence differs from the query feature space')
        if not len(features):
            continue
        stamp = float(record['timestamp'])
        end = float(record.get('end_timestamp', stamp))
        if last is not None and (stamp > last or end > last):
            raise ValueError('Retained evidence refers to a future observation')
        score = float(torch.dot(query, F.normalize(features.float().mean(dim=0), dim=0)))
        scored.append((score, stamp, order, record))
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    result = []
    for score, _, _, record in scored[:k]:
        # Query outputs are bounded transient results. Do not expose aliases
        # that could overwrite the retained memory through caller mutation.
        row = _owned(record)
        row.update(score=score, score_kind='cosine_of_pooled_visual_features',
                   checkpoint_timestamp=last)
        result.append(row)
    return result


def iter_evidence_batches(memory, max_tokens=1024):
    """Yield small owned batches with source record metadata and token positions."""
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError('max_tokens must be a positive integer')
    _cutoff(memory, None)
    for record in memory.retained_evidence():
        features = record['features']
        for lo in range(0, len(features), max_tokens):
            hi = min(len(features), lo + max_tokens)
            metadata = {key: _owned(value) for key, value in record.items()
                        if key not in ('features', 'token_indices')}
            result = {'features': features[lo:hi].clone(), 'metadata': metadata,
                      'record_token_offset': lo, 'record_tokens': len(features)}
            if 'token_indices' in record:
                result['token_indices'] = record['token_indices'][lo:hi].clone()
            yield result
