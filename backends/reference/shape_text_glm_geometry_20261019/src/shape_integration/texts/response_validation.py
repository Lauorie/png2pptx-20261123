"""Lossless wire-format normalization and per-target response isolation."""
from copy import deepcopy
import math
import re

_NUMBER = re.compile(r'-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?\Z')


def numeric_box(value):
    """Accept JSON numbers or their exact string spellings, never repair geometry."""
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError('Expected four bbox coordinates')
    result = []
    for coordinate in value:
        if isinstance(coordinate, str) and _NUMBER.fullmatch(coordinate.strip()):
            coordinate = float(coordinate)
        if type(coordinate) not in (int, float) or not math.isfinite(coordinate):
            raise ValueError('Coordinate must be a finite JSON number')
        result.append(coordinate)
    return result


def isolated(raw, ids, validator):
    """Keep valid targets even when a neighbor is missing, duplicated or invalid."""
    if isinstance(raw, list):
        raw = {'items': raw}
    if not isinstance(raw, dict) or not isinstance(raw.get('items'), list):
        raise ValueError('Expected items array')
    indexed = {}
    warnings = []
    for row in raw['items']:
        if not isinstance(row, dict) or not isinstance(row.get('id'),str) or row.get('id') not in ids:
            warnings.append('Ignored unassociated response item')
            continue
        indexed.setdefault(row['id'], []).append(row)
    checked = []
    for ident in sorted(ids):
        candidates = indexed.get(ident, [])
        try:
            if len(candidates) != 1:
                raise ValueError('Missing or duplicate target ID')
            if candidates[0].get('status') == 'invalid':
                raise ValueError(candidates[0].get('reason', 'Invalid cached item'))
            checked.append(validator(deepcopy(candidates[0])))
        except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError) as exc:
            failed={'id': ident, 'status': 'invalid', 'units': [], 'reason': str(exc)}
            if len(candidates)==1 and candidates[0].get('status')=='invalid':
                count=candidates[0].get('visual_line_count',1)
                if type(count) is int and 2<=count<=64:failed['visual_line_count']=count
            if len(candidates)==1 and candidates[0].get('status')=='resolved':
                units=candidates[0].get('units')
                if isinstance(units,list) and units:
                    count=units[0].get('visual_line_count',1) if len(units)==1 and isinstance(units[0],dict) else len(units)
                    if type(count) is int and 2<=count<=64:
                        failed['visual_line_count']=count
            checked.append(failed)
    return {'items': checked, 'warnings': warnings}


def retry_invalid(result, prompt, call):
    """One bounded repair request for invalid IDs only; never retry good targets."""
    pending = {r['id'] for r in result['items'] if r['status'] == 'invalid'}
    if not pending:
        return result
    retry = deepcopy(prompt)
    retry['targets'] = [r for r in prompt['targets'] if r['id'] in pending]
    # The stage-2 protocol calls its target array 'targets' as well.
    retry['repair_instruction'] = 'Return ONLY the listed invalid/missing target IDs. Do not return neighboring IDs.'
    try:
        repaired = call(retry, pending)
    except Exception as exc:
        return {**result, 'retry_error': str(exc)}
    replacements = {r['id']: r for r in repaired['items']}
    return {**result, 'items': [replacements.get(r['id'], r) if r['id'] in pending else r
                              for r in result['items']],
            'warnings': result.get('warnings', []) + repaired.get('warnings', []),
            'retried_ids': sorted(pending)}
