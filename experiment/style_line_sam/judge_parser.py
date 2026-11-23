"""Extract a final, schema-valid verdict without interpreting model reasoning."""
import json


def validate_judge(raw):
    if not isinstance(raw, dict) or raw.get('preferred') not in ('candidate_1', 'candidate_2', 'tie'):
        raise ValueError('preferred must be candidate_1, candidate_2 or tie')
    if not isinstance(raw.get('evidence'), str):
        raise ValueError('evidence must be a string')
    return raw


def parse_judge(text):
    decoder = json.JSONDecoder()
    cursor = 0
    valid = []
    while True:
        start = text.find('{', cursor)
        if start < 0:
            break
        try:
            raw, end = decoder.raw_decode(text[start:])
        except ValueError:
            cursor = start + 1
            continue
        cursor = start + end
        try:
            valid.append(validate_judge(raw))
        except ValueError:
            continue
    if not valid:
        raise ValueError('No schema-valid final judge object; reasoning is not a verdict')
    if len({v['preferred'] for v in valid}) != 1:
        raise ValueError('Conflicting final judge objects')
    return valid[-1]
