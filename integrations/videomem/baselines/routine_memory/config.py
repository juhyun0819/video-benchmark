"""Recorded, training-free MVP settings; the byte cap covers all three banks."""
from dataclasses import dataclass, fields
import math


@dataclass(frozen=True)
class RoutineConfig:
    budget_bytes: int = 16_000_000_000
    short_term_frames: int = 16
    spatial_bins: int = 16
    temporal_bins: int = 4
    evidence_tokens: int = 64
    stability_frames: int = 3
    stability_similarity: float = .85
    routine_similarity: float = .88
    partial_similarity: float = .65
    candidate_similarity: float = .88
    residual_similarity: float = .80
    # Raw evidence coverage only; never an independent-occurrence threshold.
    event_anchor_similarity: float = .80
    min_shared_fraction: float = .50
    temporal_weight: float = .10
    prototype_update_rate: float = .10
    promotion_occurrences: int = 3
    promotion_stability: float = .88
    occurrence_gap_seconds: float = 60.0
    max_occurrence_intervals: int = 32
    max_routine_evidence: int = 3
    merge_similarity: float = .985
    max_description_chars: int = 512
    max_audit_entries: int = 128
    merge_scan_limit: int = 64

    def __post_init__(self):
        integer_names = {'budget_bytes', 'short_term_frames', 'spatial_bins', 'temporal_bins',
                         'evidence_tokens', 'stability_frames', 'promotion_occurrences',
                         'max_occurrence_intervals', 'max_routine_evidence', 'max_description_chars',
                         'max_audit_entries', 'merge_scan_limit'}
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in integer_names:
                if type(value) is not int or value < 1:
                    raise ValueError(field.name + ' must be a positive integer')
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(field.name + ' must be finite')
            elif field.name == 'occurrence_gap_seconds':
                if value <= 0:
                    raise ValueError('occurrence_gap_seconds must be positive')
            elif not 0 <= value <= 1:
                raise ValueError(field.name + ' must lie in [0,1]')
        if self.partial_similarity > self.routine_similarity:
            raise ValueError('partial_similarity cannot exceed routine_similarity')
        if self.prototype_update_rate <= 0 or self.min_shared_fraction <= 0:
            raise ValueError('update rate and minimum shared fraction must be positive')
        if self.temporal_weight >= 1:
            raise ValueError('temporal_weight must be below one')
        if self.max_audit_entries < 8:
            raise ValueError('max_audit_entries must be at least 8')
