"""EgoLife-only test fixture conversion; never an ours input requirement."""
import re
from baselines.ours_evidence.inputs import EvidenceError

def point_seconds(day, code):
    """Official EgoLife frame-offset arithmetic, including FF >= 20."""
    match = re.fullmatch(r'DAY([1-7])', str(day), re.IGNORECASE)
    if not match or not re.fullmatch(r'\d{8}', str(code)):
        raise EvidenceError(f'Invalid evidence day/time: {day!r}, {code!r}')
    hour, minute, second, frame = [int(str(code)[i:i+2]) for i in range(0, 8, 2)]
    if hour >= 24 or minute >= 60 or second >= 60:
        raise EvidenceError(f'Invalid evidence clock: {code!r}')
    return (int(match[1])-1)*86400 + hour*3600 + minute*60 + second + frame/20


def evidence_points(target):
    """time_list and concatenated DAYn points are discrete events, not ranges."""
    if not isinstance(target, dict) or 'date' not in target:
        raise EvidenceError('target_time must contain a date')
    has_time, has_list = 'time' in target, 'time_list' in target
    if has_time == has_list:
        raise EvidenceError('Specify exactly one of target_time.time or time_list')
    values = target['time_list'] if has_list else [target['time']]
    if not isinstance(values, list) or not values:
        raise EvidenceError('Evidence time_list must be a nonempty list')
    result = []
    for value in values:
        if not isinstance(value, str) or not re.fullmatch(r'\d{8}(?:DAY[1-7]_\d{8})*', value, re.IGNORECASE):
            raise EvidenceError(f'Unsupported evidence point encoding: {value!r}')
        result.append(point_seconds(target['date'], value[:8]))
        for day, clock in re.findall(r'(DAY[1-7])_(\d{8})', value[8:], re.IGNORECASE):
            result.append(point_seconds(day, clock))
    return sorted(set(result))
