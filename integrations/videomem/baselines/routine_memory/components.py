"""Interchangeable stateless components; all learned state lives in plain banks."""
import math

import torch
import torch.nn.functional as F


def unit(value):
    return F.normalize(value.float(), dim=-1, eps=1e-12).contiguous()


def cosine(left, right):
    return (unit(left) * unit(right)).sum(-1).clamp(-1, 1)


def plain_copy(value, *, clone_tensors=False):
    """Copy containers only for transactions; persistent tensors are immutable.

    Deserialization uses clone_tensors=True to detach compact owned allocations.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().clone().contiguous() if clone_tensors else value
    if isinstance(value, dict):
        return {key: plain_copy(item, clone_tensors=clone_tensors) for key, item in value.items()}
    if isinstance(value, list):
        return [plain_copy(item, clone_tensors=clone_tensors) for item in value]
    if isinstance(value, tuple):
        return tuple(plain_copy(item, clone_tensors=clone_tensors) for item in value)
    return value


def bounded_metadata(value, limit=512, depth=0):
    if depth > 4:
        raise ValueError('Metadata nesting is too deep')
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, str) and len(value) <= limit:
        return value
    if isinstance(value, dict) and len(value) <= 32:
        forbidden = {'question', 'answer', 'choices', 'correct_idx', 'ground_truth', 'gt_evidence'}
        if any(not isinstance(key, str) or len(key) > 128 or key.lower() in forbidden for key in value):
            raise ValueError('Observation metadata must not contain QA or ground-truth fields')
        return {key: bounded_metadata(item, limit, depth+1) for key, item in value.items()}
    if isinstance(value, (list, tuple)) and len(value) <= 32:
        return [bounded_metadata(item, limit, depth+1) for item in value]
    raise ValueError('Metadata must be bounded plain finite values')


class FeatureAggregator:
    def spatial(self, features, config):
        count = min(config.spatial_bins, len(features))
        bins = torch.div(torch.arange(len(features)) * count, len(features), rounding_mode='floor')
        rows = torch.stack([features[bins == index].float().mean(0) for index in range(count)])
        return unit(rows)

    def temporal(self, frames, config, shared_mask=None):
        rows = []
        for frame in frames:
            rows.append(unit(frame['spatial'][shared_mask].mean(0)) if shared_mask is not None else frame['coarse'])
        values = torch.stack(rows)
        # Ordered phase bins, with nearest-phase repetition for short prefixes.
        output = []
        for phase in range(config.temporal_bins):
            begin = phase * len(values) // config.temporal_bins
            end = (phase+1) * len(values) // config.temporal_bins
            output.append(values[begin:max(begin+1, end)].mean(0))
        return unit(torch.stack(output))

    def frame(self, features, frame_id, timestamp, metadata, config, previous=None):
        spatial = self.spatial(features, config)
        coarse = unit(features.float().mean(0))
        count = min(config.evidence_tokens, len(features))
        indices = torch.div(torch.arange(count) * len(features), count, rounding_mode='floor')
        return {'frame_id': frame_id, 'timestamp': timestamp,
                'features': features.index_select(0, indices).detach().clone().contiguous(),
                'token_indices': indices.clone(), 'spatial': spatial, 'coarse': coarse,
                'temporal_change': 0.0 if previous is None else float(1-cosine(previous['coarse'], coarse)),
                'metadata': metadata}

    def describe(self, frames, config):
        latest = frames[-1]
        return {'spatial': latest['spatial'], 'coarse': latest['coarse'],
                'temporal': self.temporal(frames, config)}


class RoutineMatcher:
    def compare(self, descriptor, prototype, config):
        bins = cosine(descriptor['spatial'], prototype['spatial'])
        spatial = float(bins.clamp(min=0).mean())
        coarse = max(0.0, float(cosine(descriptor['coarse'], prototype['coarse'])))
        temporal = float(cosine(descriptor['temporal'], prototype['temporal']).clamp(min=0).mean())
        score = (1-config.temporal_weight) * (.75*spatial + .25*coarse) + config.temporal_weight*temporal
        return {'score': score, 'spatial_similarity': spatial, 'temporal_similarity': temporal,
                'shared_mask': bins >= config.residual_similarity,
                'residual_scores': (1-bins).clamp(0, 2)/2}

    def best(self, descriptor, records, config):
        if not records:
            return None, None
        scored = [(record, self.compare(descriptor, record, config)) for record in records]
        return max(scored, key=lambda pair: (pair[1]['score'], -pair[0]['id']))

    def route(self, descriptor, records, config):
        record, match = self.best(descriptor, records, config)
        if record is None:
            return 'C', None, None
        shared = float(match['shared_mask'].float().mean())
        if match['score'] >= config.routine_similarity and bool(match['shared_mask'].all()):
            return 'A', record, match
        if match['score'] >= config.partial_similarity and shared >= config.min_shared_fraction:
            return 'B', record, match
        return 'C', None, match


class RoutineUpdater:
    def update(self, record, descriptor, similarity, config, shared_mask=None):
        rate = config.prototype_update_rate
        updated = unit((1-rate)*record['spatial'] + rate*descriptor['spatial'])
        if shared_mask is not None:
            updated = torch.where(shared_mask[:, None], updated, record['spatial']).contiguous()
        record['spatial'] = updated
        record['coarse'] = unit(updated.mean(0))
        record['temporal'] = unit((1-rate)*record['temporal'] + rate*descriptor['temporal'])
        record['update_count'] += 1
        record['similarity_sum'] += float(similarity)
        record['similarity_count'] += 1
        record['minimum_similarity'] = min(record['minimum_similarity'], float(similarity))

    def occurrence(self, record, pending, frame, active, kind, config):
        same = active is not None and active['kind'] == kind and active['id'] == record['id']
        interval = {'start_frame_id': pending['start_frame_id'], 'end_frame_id': frame['frame_id'],
                    'start_timestamp': pending['start_timestamp'], 'end_timestamp': frame['timestamp']}
        new = False
        if same and record['intervals']:
            record['intervals'][-1]['end_frame_id'] = frame['frame_id']
            record['intervals'][-1]['end_timestamp'] = frame['timestamp']
        else:
            prior = pending.get('prior_last_seen')
            new = record['occurrence_count'] == 0 or (prior is not None and pending['start_timestamp']-prior >= config.occurrence_gap_seconds)
            if new:
                record['occurrence_count'] += 1
            interval['independent'] = new
            record['intervals'].append(interval)
            record['intervals'] = record['intervals'][-config.max_occurrence_intervals:]
        record['last_seen'] = frame['timestamp']
        record['last_frame_id'] = frame['frame_id']
        return {'kind': kind, 'id': record['id']}, new


class EventSelector:
    def evidence(self, features, frame, config, residual_mask=None, residual_scores=None):
        if residual_mask is None:
            indices = frame['token_indices']
        else:
            bins = torch.div(torch.arange(len(features))*len(residual_mask), len(features), rounding_mode='floor')
            indices = torch.nonzero(residual_mask[bins], as_tuple=False).flatten()
            if residual_scores is not None and len(indices):
                rank = torch.argsort(residual_scores[bins[indices]], descending=True, stable=True)
                indices = indices[rank]
            indices = indices[:config.evidence_tokens].sort().values
        return {'features': features.index_select(0, indices).detach().clone().contiguous(),
                'token_indices': indices.clone(), 'frame_id': frame['frame_id'], 'timestamp': frame['timestamp'],
                'metadata': plain_copy(frame['metadata'])}

    def priority(self, event):
        return (1 if event['distinctive'] else 0, event['novelty_score'], event['end_timestamp'], event['id'])


class CandidateManager:
    def create(self, identifier, descriptor, timestamp, frame_id):
        return {'id': identifier, 'spatial': descriptor['spatial'].clone(),
                'coarse': descriptor['coarse'].clone(), 'temporal': descriptor['temporal'].clone(),
                'occurrence_count': 0, 'intervals': [], 'last_seen': timestamp,
                'last_frame_id': frame_id, 'created_timestamp': timestamp,
                'update_count': 0, 'similarity_sum': 0.0, 'similarity_count': 0,
                'minimum_similarity': 1.0}

    def eligible(self, candidate, config):
        average = candidate['similarity_sum'] / max(1, candidate['similarity_count'])
        return candidate['occurrence_count'] >= config.promotion_occurrences and average >= config.promotion_stability


class MemoryBudgetManager:
    """Deterministic pressure policy; helper owns exact persistent byte charging."""
    def enforce(self, state, config, audit, matcher, selector):
        from .budget import measure_state_bytes
        def measured():
            return measure_state_bytes(state)
        report = measured()
        while report['actual_memory_bytes'] > config.budget_bytes:
            routines = state['routine_memory']['routines']
            events = state['event_memory']['events']
            candidates = state['event_memory']['candidates']
            frames = state['short_term']['frames']
            merged = False
            scan = routines[-config.merge_scan_limit:]
            for position, left in enumerate(scan):
                for right in scan[position+1:]:
                    match = matcher.compare(left, right, config)
                    if match['score'] < config.merge_similarity or not bool(match['shared_mask'].all()):
                        continue
                    # Preserve the older shared prototype and bounded provenance;
                    # no retrospective reclassification or feature synthesis.
                    left['intervals'] = sorted(left['intervals']+right['intervals'], key=lambda x:x['start_timestamp'])[-config.max_occurrence_intervals:]
                    left['occurrence_count'] += right['occurrence_count']
                    left['last_seen'] = max(left['last_seen'], right['last_seen'])
                    left['representatives'] = (left['representatives']+right['representatives'])[-config.max_routine_evidence:]
                    for event in events:
                        if event['routine_id'] == right['id']:
                            event['routine_id'] = left['id']
                    active = state['control']['active']
                    if active and active['kind'] == 'routine' and active['id'] == right['id']:
                        active['id'] = left['id']
                    pending = state['control']['pending']
                    if pending and pending.get('routine_id') == right['id']:
                        pending['routine_id'] = left['id']
                    routines.remove(right)
                    state['control']['routine_merges'] += 1
                    audit.append({'operation':'merge_routines', 'kept_routine_id':left['id'], 'removed_routine_id':right['id']})
                    merged = True
                    break
                if merged:
                    break
            if not merged:
                # Old STM rows duplicate information already routed to the banks.
                if len(frames) > 1:
                    removed = frames.pop(0)
                    audit.append({'operation':'evict_short_term', 'frame_id':removed['frame_id']})
                elif any(len(r['representatives']) > 1 for r in routines):
                    routine = min((r for r in routines if len(r['representatives']) > 1), key=lambda r:(r['last_seen'],r['id']))
                    evidence = routine['representatives'].pop(0)
                    audit.append({'operation':'drop_redundant_routine_evidence','routine_id':routine['id'],'frame_id':evidence['frame_id']})
                elif events:
                    event = min(events, key=selector.priority)
                    events.remove(event)
                    state['control']['event_evictions'] += 1
                    audit.append({'operation':'evict_event','event_id':event['id'],'candidate_id':event['candidate_id'],'distinctive':event['distinctive']})
                elif any(len(record['intervals']) > 1 for record in routines+candidates):
                    record = min((r for r in routines+candidates if len(r['intervals'])>1),key=lambda r:(r['last_seen'],r['id']))
                    interval = record['intervals'].pop(0)
                    audit.append({'operation':'prune_occurrence_interval','record_id':record['id'],**interval})
                elif candidates:
                    candidate = min(candidates,key=lambda r:(r['occurrence_count'],r['last_seen'],r['id']))
                    candidates.remove(candidate)
                    audit.append({'operation':'evict_candidate','candidate_id':candidate['id']})
                elif routines:
                    routine = min(routines,key=lambda r:(r['occurrence_count'],r['last_seen'],r['id']))
                    routines.remove(routine)
                    audit.append({'operation':'evict_routine','routine_id':routine['id']})
                elif frames and len(frames[0]['features']) > 1:
                    frame = frames[0]
                    old_count = len(frame['features'])
                    frame['features'] = frame['features'][::2].clone().contiguous()
                    frame['token_indices'] = frame['token_indices'][::2].clone().contiguous()
                    audit.append({'operation':'reduce_current_evidence','from_tokens':old_count,'to_tokens':len(frame['features'])})
                else:
                    raise ValueError('Budget cannot hold the current compact observation and control state')
            self.clean_references(state)
            if len(audit) > config.max_audit_entries:
                raise ValueError('Budget update would exceed the bounded audit record limit')
            report = measured()
        return report

    def clean_references(self, state):
        routine_ids = {r['id'] for r in state['routine_memory']['routines']}
        candidate_ids = {r['id'] for r in state['event_memory']['candidates']}
        event_ids = {r['id'] for r in state['event_memory']['events']}
        for event in state['event_memory']['events']:
            if event['routine_id'] not in routine_ids:
                event['routine_id'] = None
            if event['candidate_id'] not in candidate_ids:
                event['candidate_id'] = None
        active = state['control']['active']
        if active and active['id'] not in (routine_ids if active['kind']=='routine' else candidate_ids):
            state['control']['active'] = None
        pending = state['control']['pending']
        if pending:
            if pending.get('event_id') not in event_ids:
                pending['event_id'] = None
            if pending.get('candidate_id') not in candidate_ids:
                pending['candidate_id'] = None
            if pending.get('routine_id') not in routine_ids:
                pending['routine_id'] = None
