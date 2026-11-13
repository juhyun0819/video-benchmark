"""Causal routine discovery over compact owned visual evidence, without QA."""
from dataclasses import asdict
import math
import time

import torch

from .budget import measure_state_bytes
from .config import RoutineConfig
from .components import (FeatureAggregator, RoutineMatcher, RoutineUpdater, EventSelector,
                         CandidateManager, MemoryBudgetManager, plain_copy, bounded_metadata)


class RoutineAwareMemory:
    """All continuation data is in state_dict; helpers hold no learned state.

    ``caption_callback`` is an optional per-update callable, not a retained model
    or cache. It receives a compact copied evidence record and returns str/None.
    The default stores no invented description. Callback failures leave memory
    unchanged (external callback side effects are the caller's responsibility).
    """
    def __init__(self, config=None):
        config = RoutineConfig() if config is None else config
        if not isinstance(config, RoutineConfig):
            raise TypeError('config must be RoutineConfig')
        self._state = {
            'short_term': {'frames': []},
            'routine_memory': {'routines': []},
            'event_memory': {'events': [], 'candidates': []},
            'control': {'schema_version': 1, 'config': asdict(config), 'shape': None, 'dtype': None,
                        'total_frames_seen': 0, 'total_visual_tokens_seen': 0, 'cumulative_feature_bytes': 0,
                        'last_frame_id': None, 'last_timestamp': None,
                        'next_event_id': 1, 'next_candidate_id': 1, 'next_routine_id': 1,
                        'routing_counts': {'A': 0, 'B': 0, 'C': 0}, 'promotions': 0,
                        'routine_merges': 0, 'event_evictions': 0, 'pending_observations': 0,
                        'active': None, 'pending': None, 'runner_source': None}}
        if measure_state_bytes(self._state)['actual_memory_bytes'] > config.budget_bytes:
            raise ValueError('Budget cannot hold the minimal configuration and control state')

    @property
    def config(self):
        return RoutineConfig(**self.control['config'])

    @property
    def control(self):
        return self._state['control']

    def state_dict(self):
        """Borrowed read-only state references; accounting does not clone a bank."""
        return self._state

    def set_source_context(self, context):
        context = bounded_metadata(context)
        draft = plain_copy(self._state)
        if draft['control']['runner_source'] not in (None, context):
            raise ValueError('Cannot change the source identity of an existing memory')
        draft['control']['runner_source'] = context
        self._enforce(draft, [])
        self._state = draft

    @classmethod
    def from_state_dict(cls, state):
        # The accounting validator rejects opaque objects, nonfinite metadata,
        # unsupported tensors and cycles before any feature allocation/copy.
        report = measure_state_bytes(state)
        if set(state) != {'short_term','routine_memory','event_memory','control'}:
            raise ValueError('Unexpected routine memory state components')
        control = state['control']
        if control.get('schema_version') != 1:
            raise ValueError('Unsupported routine memory schema')
        config = RoutineConfig(**control['config'])
        if report['actual_memory_bytes'] > config.budget_bytes:
            raise ValueError('Checkpoint exceeds its fixed memory budget')
        result = cls(config)
        result._state = plain_copy(state, clone_tensors=True)
        # Legacy checkpoints did not track raw event coverage. Materialize the
        # default and bounded counters so migration is explicitly byte-charged.
        result.control['config'] = asdict(config)
        pending = result.control['pending']
        if pending is not None:
            for key, value in cls._coverage_defaults().items():
                pending.setdefault(key, value)
        result._validate_state(result._state)
        if measure_state_bytes(result._state)['actual_memory_bytes'] > config.budget_bytes:
            raise ValueError('Owned restored state exceeds its fixed budget')
        return result

    def _validate_state(self, state):
        control = state['control']
        frames = state['short_term']['frames']
        routines = state['routine_memory']['routines']
        events = state['event_memory']['events']
        candidates = state['event_memory']['candidates']
        if any(type(control[key]) is not int or control[key] < 0 for key in (
                'total_frames_seen','total_visual_tokens_seen','cumulative_feature_bytes','promotions',
                'routine_merges','event_evictions','pending_observations')):
            raise ValueError('Invalid stream counters')
        if not control['total_frames_seen']:
            if frames or routines or events or candidates or control['last_timestamp'] is not None:
                raise ValueError('Nonempty evidence with an empty stream cursor')
            return
        if (type(control['last_frame_id']) is not int or control['last_frame_id'] < 0
                or not isinstance(control['last_timestamp'], (int,float))
                or not math.isfinite(control['last_timestamp'])):
            raise ValueError('Invalid stream cursor')
        shape = control['shape']
        dtype = getattr(torch, control['dtype'], None)
        if (not isinstance(shape, list) or len(shape) != 2 or any(type(x) is not int or x<1 for x in shape)
                or dtype not in (torch.float16,torch.bfloat16,torch.float32)):
            raise ValueError('Invalid recorded feature geometry')
        last = -1
        for frame in frames:
            if frame['frame_id'] <= last:
                raise ValueError('Short-term frame order is invalid')
            last = frame['frame_id']
        if len(frames)>self.config.short_term_frames:
            raise ValueError('Short-term window exceeds configuration')
        for records in (routines,events,candidates):
            ids = [r['id'] for r in records]
            if len(ids) != len(set(ids)) or any(type(i) is not int or i<1 for i in ids):
                raise ValueError('Duplicate/invalid memory record IDs')
        routine_ids, candidate_ids = {r['id'] for r in routines}, {r['id'] for r in candidates}
        for event in events:
            if event['routine_id'] is not None and event['routine_id'] not in routine_ids:
                raise ValueError('Dangling routine association')
            if event['candidate_id'] is not None and event['candidate_id'] not in candidate_ids:
                raise ValueError('Dangling candidate association')
        pending = control['pending']
        if pending is not None:
            count = pending.get('anchor_change_count', 0)
            if type(count) is not int or not 0 <= count < self.config.stability_frames:
                raise ValueError('Invalid event coverage confirmation count')
            anchor_id = pending.get('anchor_event_id')
            if anchor_id is not None and not any(event['id']==anchor_id and
                    event['candidate_id']==pending['candidate_id'] for event in events):
                raise ValueError('Dangling candidate coverage anchor')
            start_id = pending.get('anchor_change_start_frame_id')
            start_time = pending.get('anchor_change_start_timestamp')
            if count and (type(start_id) is not int or not 0 <= start_id <= control['last_frame_id']
                    or not isinstance(start_time, (int,float)) or not math.isfinite(start_time)
                    or not 0 <= start_time <= control['last_timestamp']):
                raise ValueError('Invalid event coverage confirmation interval')
        for record in routines+candidates:
            if (record['spatial'].shape != (min(shape[0],self.config.spatial_bins),shape[1])
                    or record['coarse'].shape != (shape[1],)
                    or record['temporal'].shape != (self.config.temporal_bins,shape[1])):
                raise ValueError('Prototype geometry differs from the shared representation')
            if len(record['intervals']) > self.config.max_occurrence_intervals:
                raise ValueError('Unbounded occurrence history')
            if record['last_seen'] > control['last_timestamp']:
                raise ValueError('Prototype includes a future observation')
        for evidence in self._evidence(state):
            tensor, indices = evidence['features'], evidence['token_indices']
            if (tensor.ndim != 2 or tensor.shape[1] != shape[1] or tensor.dtype != dtype
                    or indices.shape != (len(tensor),) or indices.dtype != torch.int64
                    or len(tensor)>self.config.evidence_tokens or len(tensor)<1
                    or int(indices.min())<0 or int(indices.max())>=shape[0]
                    or evidence['timestamp'] > control['last_timestamp']
                    or evidence['frame_id'] > control['last_frame_id']):
                raise ValueError('Malformed or noncausal retained evidence')

    def _enforce(self, state, audit):
        report = MemoryBudgetManager().enforce(state, RoutineConfig(**state['control']['config']),
                                               audit, RoutineMatcher(), EventSelector())
        pending = state['control']['pending']
        if pending is not None and pending.get('anchor_event_id') is not None:
            if not any(event['id']==pending['anchor_event_id'] and
                       event['candidate_id']==pending['candidate_id']
                       for event in state['event_memory']['events']):
                self._reset_coverage_pending(pending)
                report = measure_state_bytes(state)
        return report

    @staticmethod
    def _coverage_defaults():
        return {'anchor_event_id':None, 'anchor_change_count':0,
                'anchor_change_start_frame_id':None, 'anchor_change_start_timestamp':None}

    @classmethod
    def _reset_coverage_pending(cls, pending):
        pending.update(cls._coverage_defaults())

    @staticmethod
    def _evidence(state):
        for frame in state['short_term']['frames']:
            yield {'kind':'short_term', **frame}
        for routine in state['routine_memory']['routines']:
            for evidence in routine['representatives']:
                yield {'kind':'routine', 'routine_id':routine['id'], **evidence}
        for event in state['event_memory']['events']:
            yield {'kind':'event', 'event_id':event['id'], 'routine_id':event['routine_id'],
                   'candidate_id':event['candidate_id'], 'description':event['description'],
                   **{key:event[key] for key in ('start_frame_id','end_frame_id','start_timestamp','end_timestamp')},
                   **event['evidence']}

    def retained_evidence(self):
        """Borrowed compact evidence plus exact original token/frame provenance."""
        return self._evidence(self._state)

    def stats(self):
        control = self.control
        report = measure_state_bytes(self._state)
        denominator = control['cumulative_feature_bytes']
        report.update(total_frames_seen=control['total_frames_seen'],
            total_visual_tokens_seen=control['total_visual_tokens_seen'],
            cumulative_feature_bytes=denominator, last_frame_id=control['last_frame_id'],
            last_timestamp=control['last_timestamp'], budget_bytes=self.config.budget_bytes,
            effective_retention_ratio=report['actual_memory_bytes']/denominator if denominator else 0.0,
            budget_utilization=report['actual_memory_bytes']/self.config.budget_bytes,
            short_term_frames=len(self._state['short_term']['frames']),
            routines=len(self._state['routine_memory']['routines']),
            events=len(self._state['event_memory']['events']),
            candidates=len(self._state['event_memory']['candidates']),
            promotions=control['promotions'], event_to_routine_promotions=control['promotions'],
            routine_merges=control['routine_merges'], event_evictions=control['event_evictions'],
            matched_observations=control['routing_counts']['A'],
            partially_matched_observations=control['routing_counts']['B'],
            unmatched_observations=control['routing_counts']['C'],
            pending_observations=control['pending_observations'],
            retained_visual_tokens=sum(len(item['features']) for item in self.retained_evidence()))
        return report

    def record_costs(self):
        """Inspect standalone per-record costs without adding cached state.

        Shared tensors and JSON offset/header sizes mean these estimates must
        not be summed as a replacement for the full-state hard-cap accountant.
        """
        empty = {'short_term':{'frames':[]},'routine_memory':{'routines':[]},
                 'event_memory':{'events':[],'candidates':[]},'control':{}}
        baseline = measure_state_bytes(empty)
        rows = []
        for kind, component, key in [('short_term','short_term','frames'),
                                      ('routine','routine_memory','routines'),
                                      ('event','event_memory','events'),
                                      ('candidate','event_memory','candidates')]:
            for record in self._state[component][key]:
                scoped = plain_copy(empty)
                scoped[component][key] = [record]
                value = measure_state_bytes(scoped)
                rows.append({'kind':kind,'id':record.get('id',record.get('frame_id')),
                    'tensor_storage_bytes':value['tensor_storage_bytes'],
                    'estimated_memory_bytes':value['actual_memory_bytes']-baseline['actual_memory_bytes'],
                    'serialized_bytes_increment':value['serialized_bytes']-baseline['serialized_bytes']})
        return {'records':rows,'actual_memory_bytes':measure_state_bytes(self._state)['actual_memory_bytes'],
                'cost_basis':'Standalone record increment including metadata, object reserves and index; do not sum when storage is shared'}

    def _pending(self, draft, case, routine, candidate, match, frame, previous, config):
        control = draft['control']
        pending = control['pending']
        routine_id = routine['id'] if routine else None
        residual = (torch.nonzero(~match['shared_mask'],as_tuple=False).flatten().tolist()
                    if case in ('B','C') and match is not None else [])
        local_similarity = 1.0
        if previous is not None:
            from .components import cosine
            local_similarity = float(cosine(frame['spatial'],previous['spatial']).clamp(min=0).mean())
        same = (pending is not None and pending['case']==case and pending['routine_id']==routine_id
                and (case!='C' or pending['candidate_id']==(candidate['id'] if candidate else None))
                and pending['residual_bins']==residual and local_similarity>=config.stability_similarity)
        reset_reason = None
        if same:
            pending['count'] += 1
        else:
            if pending is None:
                reset_reason = 'initial'
            elif pending['case'] != case:
                reset_reason = 'case_changed'
            elif pending['routine_id'] != routine_id:
                reset_reason = 'routine_changed'
            elif case=='C' and pending['candidate_id'] != (candidate['id'] if candidate else None):
                reset_reason = 'candidate_changed'
            elif pending['residual_bins'] != residual:
                reset_reason = 'residual_bins_changed'
            else:
                reset_reason = 'pooled_instability'
            reference = routine or candidate
            pending = {'case':case,'routine_id':routine_id,
                'candidate_id':candidate['id'] if candidate else None,'event_id':None,
                'count':1,'start_frame_id':frame['frame_id'],'start_timestamp':frame['timestamp'],
                'prior_last_seen':reference['last_seen'] if reference else None,'residual_bins':residual,
                **self._coverage_defaults()}
            control['pending'] = pending
        return pending, pending['count']>=config.stability_frames, reset_reason

    def _candidate_coverage(self, draft, candidate, features, frame, descriptor, pending,
                            config, audit, caption_callback):
        """Preserve changed raw evidence without creating a new occurrence.

        Confirmation uses the existing pooled stability gate. Raw aligned
        features can change with camera motion, so this is evidence coverage,
        not semantic event segmentation or an activity-boundary detector.
        """
        from .components import cosine
        events = draft['event_memory']['events']
        anchor = next((event for event in events if event['id']==pending.get('anchor_event_id')
                       and event['candidate_id']==candidate['id']), None)
        if anchor is None:
            # A pending reset must not reopen the original common evidence as
            # the coverage anchor after a newer coverage event was retained.
            anchor = next((event for event in reversed(events)
                           if event['candidate_id']==candidate['id'] and
                           event.get('retention_reason')=='anchor_change'), None)
            if anchor is None:
                anchor = next((event for event in events if event['id']==pending['event_id']
                               and event['candidate_id']==candidate['id']), None)
            self._reset_coverage_pending(pending)
            if anchor is None:
                return {'anchor_similarity':None, 'coverage_action':'no_anchor'}
            pending['anchor_event_id'] = anchor['id']
        evidence = anchor['evidence']
        current = features.index_select(0, evidence['token_indices'])
        similarity = float(cosine(current, evidence['features']).mean())
        if similarity >= config.event_anchor_similarity:
            self._reset_coverage_pending(pending)
            pending['anchor_event_id'] = anchor['id']
            return {'anchor_similarity':similarity, 'coverage_action':'covered'}
        if not pending['anchor_change_count']:
            pending['anchor_change_start_frame_id'] = frame['frame_id']
            pending['anchor_change_start_timestamp'] = frame['timestamp']
        pending['anchor_change_count'] += 1
        if pending['anchor_change_count'] < config.stability_frames:
            return {'anchor_similarity':similarity, 'coverage_action':'pending'}
        coverage_pending = dict(pending)
        coverage_pending['start_frame_id'] = pending['anchor_change_start_frame_id']
        coverage_pending['start_timestamp'] = pending['anchor_change_start_timestamp']
        coverage_pending['count'] = pending['anchor_change_count']
        # _new_event writes its pending pointer. A private scalar dict prevents
        # the primary common/residual path from seeing this distinctive event.
        event = self._new_event(draft, features, frame, descriptor, coverage_pending, config,
                               candidate_id=candidate['id'], distinctive=True,
                               caption_callback=caption_callback)
        event['retention_reason'] = 'anchor_change'
        event['novelty_score'] = 1.0-similarity
        self._reset_coverage_pending(pending)
        pending['anchor_event_id'] = event['id']
        audit.append({'operation':'retain_candidate_coverage', 'candidate_id':candidate['id'],
                      'event_id':event['id'], 'previous_anchor_event_id':anchor['id'],
                      'anchor_similarity':similarity, 'confirmation_frames':config.stability_frames,
                      'evidence_frame_id':frame['frame_id']})
        return {'anchor_similarity':similarity, 'coverage_action':'retained'}

    def _new_event(self, draft, features, frame, descriptor, pending, config, *, candidate_id=None,
                   routine_id=None, match=None, distinctive=False, caption_callback=None):
        selector = EventSelector()
        residual = ~match['shared_mask'] if distinctive and match is not None and not bool(match['shared_mask'].all()) else None
        evidence = selector.evidence(features,frame,config,residual,
                                     match['residual_scores'] if match is not None else None)
        description = None
        if caption_callback is not None:
            description = caption_callback(plain_copy(evidence,clone_tensors=True))
            if description is not None and (not isinstance(description,str) or len(description)>config.max_description_chars):
                raise ValueError('Caption callback must return bounded text or None')
        event = {'id':draft['control']['next_event_id'], 'candidate_id':candidate_id,'routine_id':routine_id,
            'description':description,'distinctive':bool(distinctive),
            'novelty_score':1-float(match['score']) if match is not None else 1.0,
            'start_frame_id':pending['start_frame_id'],'end_frame_id':frame['frame_id'],
            'start_timestamp':pending['start_timestamp'],'end_timestamp':frame['timestamp'],
            'frames_seen':pending['count'], 'evidence':evidence,
            **{key:descriptor[key].clone() for key in ('spatial','coarse','temporal')}}
        draft['control']['next_event_id'] += 1
        draft['event_memory']['events'].append(event)
        pending['event_id'] = event['id']
        return event

    def _promote(self, draft, candidate, config, audit):
        events = draft['event_memory']['events']
        common = [event for event in events if event['candidate_id']==candidate['id'] and not event['distinctive']]
        routine = plain_copy(candidate)
        identifier = draft['control']['next_routine_id']
        draft['control']['next_routine_id'] += 1
        routine.update(id=identifier,source_candidate_id=candidate['id'],
                       representatives=[event['evidence'] for event in common[-config.max_routine_evidence:]])
        for event in events:
            if event['candidate_id']==candidate['id'] and event['distinctive']:
                event['candidate_id'] = None
                event['routine_id'] = identifier
        draft['event_memory']['events'] = [event for event in events if event not in common]
        draft['event_memory']['candidates'].remove(candidate)
        draft['routine_memory']['routines'].append(routine)
        draft['control']['promotions'] += 1
        draft['control']['active'] = {'kind':'routine','id':identifier}
        draft['control']['pending']['candidate_id'] = None
        draft['control']['pending']['event_id'] = None
        self._reset_coverage_pending(draft['control']['pending'])
        audit.append({'operation':'promote_candidate','candidate_id':candidate['id'],'routine_id':identifier,
                      'independent_occurrences':routine['occurrence_count'],
                      'common_events_released':len(common),
                      'representative_frame_ids':[item['frame_id'] for item in routine['representatives']],
                      'distinctive_events_preserved':sum(event['routine_id']==identifier for event in draft['event_memory']['events'])})
        return identifier

    @torch.no_grad()
    def update(self, features, frame_id, timestamp, metadata=None, *, caption_callback=None):
        started = time.perf_counter()
        config = self.config
        if (not isinstance(features,torch.Tensor) or features.device.type!='cpu' or features.ndim!=2
                or features.dtype not in (torch.float16,torch.bfloat16,torch.float32)
                or any(size<1 for size in features.shape) or not bool(torch.isfinite(features).all())):
            raise ValueError('Expected finite nonempty CPU fp16/bf16/fp32 [tokens,hidden] features')
        if (type(frame_id) is not int or frame_id<0 or isinstance(timestamp,bool)
                or not isinstance(timestamp,(int,float)) or not math.isfinite(timestamp) or timestamp<0):
            raise ValueError('Invalid frame identity or timestamp')
        timestamp = float(timestamp)
        if self.control['last_frame_id'] is not None and (frame_id<=self.control['last_frame_id'] or timestamp<=self.control['last_timestamp']):
            raise ValueError('Frame IDs and timestamps must strictly increase')
        if self.control['shape'] is not None and (list(features.shape)!=self.control['shape'] or str(features.dtype).removeprefix('torch.')!=self.control['dtype']):
            raise ValueError('Feature shape and dtype must remain fixed')
        metadata = bounded_metadata({} if metadata is None else metadata, config.max_description_chars)
        if caption_callback is not None and not callable(caption_callback):
            raise TypeError('caption_callback must be callable or None')
        # Containers are transaction-local. Existing tensors are only read;
        # prototype updates/selection allocate compact replacement tensors.
        draft = plain_copy(self._state)
        control = draft['control']
        aggregator,matcher,updater = FeatureAggregator(),RoutineMatcher(),RoutineUpdater()
        previous = draft['short_term']['frames'][-1] if draft['short_term']['frames'] else None
        frame = aggregator.frame(features,frame_id,timestamp,metadata,config,previous)
        draft['short_term']['frames'].append(frame)
        dropped = draft['short_term']['frames'][:-config.short_term_frames]
        draft['short_term']['frames'] = draft['short_term']['frames'][-config.short_term_frames:]
        descriptor = aggregator.describe(draft['short_term']['frames'],config)
        case,routine,match = matcher.route(descriptor,draft['routine_memory']['routines'],config)
        candidate, candidate_match = matcher.best(descriptor,draft['event_memory']['candidates'],config)
        candidate_similarity = candidate_match['score'] if candidate_match is not None else None
        if candidate_match is None or candidate_match['score']<config.candidate_similarity:
            candidate,candidate_match = None,None
        pending,stable,pending_reset_reason = self._pending(draft,case,routine,candidate,
                                      candidate_match if case=='C' else match,frame,previous,config)
        coverage = {'anchor_similarity':None, 'coverage_action':'not_applicable'}
        audit = [{'operation':'expire_short_term','frame_id':old['frame_id']} for old in dropped]
        action = 'pending'
        selected_routine_id = routine['id'] if routine else None
        if case in ('A','B'):
            shared = match['shared_mask']
            update_descriptor = descriptor
            if case=='B':
                update_descriptor = {**descriptor,'temporal':aggregator.temporal(draft['short_term']['frames'],config,shared)}
            updater.update(routine,update_descriptor,match['score'],config,shared)
            routine['last_seen'],routine['last_frame_id'] = timestamp,frame_id
            action = 'update_routine' if case=='A' else 'update_shared_routine'
            if stable:
                control['active'],new_occurrence = updater.occurrence(routine,pending,frame,control['active'],'routine',config)
                if new_occurrence:
                    audit.append({'operation':'independent_routine_occurrence','routine_id':routine['id'],'count':routine['occurrence_count']})
                if case=='B':
                    event = next((event for event in draft['event_memory']['events'] if event['id']==pending['event_id']),None)
                    if event is None:
                        event = self._new_event(draft,features,frame,descriptor,pending,config,
                            routine_id=routine['id'],match=match,distinctive=True,caption_callback=caption_callback)
                        audit.append({'operation':'retain_residual_event','event_id':event['id'],'routine_id':routine['id'],
                                      'token_indices':event['evidence']['token_indices'].tolist()})
                    else:
                        event.update(end_frame_id=frame_id,end_timestamp=timestamp,frames_seen=event['frames_seen']+1)
                    action = 'retain_residual'
        elif stable:
            if candidate is None:
                identifier = control['next_candidate_id']
                control['next_candidate_id'] += 1
                candidate = CandidateManager().create(identifier,descriptor,timestamp,frame_id)
                draft['event_memory']['candidates'].append(candidate)
                candidate_match = matcher.compare(descriptor,candidate,config)
                pending['prior_last_seen'] = None
                audit.append({'operation':'create_candidate','candidate_id':identifier})
            pending['candidate_id'] = candidate['id']
            same_activity = (control['active'] is not None and control['active']['kind']=='candidate'
                             and control['active']['id']==candidate['id'])
            control['active'],new_occurrence = updater.occurrence(candidate,pending,frame,control['active'],'candidate',config)
            updater.update(candidate,descriptor,candidate_match['score'],config,candidate_match['shared_mask'])
            distinctive = not bool(candidate_match['shared_mask'].all())
            event = next((event for event in draft['event_memory']['events'] if event['id']==pending['event_id']),None)
            if event is None and same_activity and not distinctive:
                event = next((old for old in reversed(draft['event_memory']['events'])
                              if old['candidate_id']==candidate['id'] and not old['distinctive']),None)
                if event is not None:
                    pending['event_id'] = event['id']
                    audit.append({'operation':'reuse_common_candidate_evidence','candidate_id':candidate['id'],'event_id':event['id']})
            if event is None or event['candidate_id']!=candidate['id'] or event['distinctive']!=distinctive:
                event = self._new_event(draft,features,frame,descriptor,pending,config,
                    candidate_id=candidate['id'],match=candidate_match,distinctive=distinctive,caption_callback=caption_callback)
                audit.append({'operation':'retain_candidate_event','event_id':event['id'],'candidate_id':candidate['id'],'distinctive':distinctive})
            else:
                event.update(end_frame_id=frame_id,end_timestamp=timestamp,frames_seen=event['frames_seen']+1)
            if new_occurrence:
                audit.append({'operation':'independent_candidate_occurrence','candidate_id':candidate['id'],'count':candidate['occurrence_count']})
            action = 'update_candidate'
            coverage = self._candidate_coverage(draft,candidate,features,frame,descriptor,pending,
                                                config,audit,caption_callback)
            if CandidateManager().eligible(candidate,config):
                selected_routine_id = self._promote(draft,candidate,config,audit)
                action = 'promote_candidate'
        if not stable:
            control['pending_observations'] += 1
        control['shape'],control['dtype'] = list(features.shape),str(features.dtype).removeprefix('torch.')
        control['total_frames_seen'] += 1
        control['total_visual_tokens_seen'] += len(features)
        control['cumulative_feature_bytes'] += features.numel()*features.element_size()
        control['last_frame_id'],control['last_timestamp'] = frame_id,timestamp
        control['routing_counts'][case] += 1
        report = self._enforce(draft,audit)
        if len(audit)>config.max_audit_entries:
            raise ValueError('Update audit exceeds configured bound')
        self._state = draft
        return {'case':case,'stable':stable,'action':action,'frame_id':frame_id,'timestamp':timestamp,
                'routine_id':selected_routine_id,'candidate_id':control['pending']['candidate_id'],
                'similarity':match['score'] if match is not None else None,'audit':audit,
                'candidate_similarity':candidate_similarity, 'pending_reset_reason':pending_reset_reason,
                **coverage,
                'actual_memory_bytes':report['actual_memory_bytes'],
                'processing_seconds':time.perf_counter()-started}
