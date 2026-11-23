"""One observed object, a shared point table and primitive/path parts.

Structural errors invalidate the observation as a whole. Numerical degeneracy
is handled per part by the adapter so other valid parts can be inspected.
"""
import copy
import math

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

VERSION = 'shape_integration.fitting.1'
MODEL_VERSION = 'shape_integration.local_observation.7'
POINT_COUNTS = {'rectangle': 4, 'ellipse': 8, 'rounded_rectangle': 16}


class ObservationError(ValueError):
    pass


def protocol_error(exc):
    """Field-specific feedback without echoing malformed coordinate tables."""
    path = '.'.join(str(p) for p in exc.absolute_path) or 'response'
    if exc.validator == 'type':
        detail = 'expected '+str(exc.validator_value)+', received '+type(exc.instance).__name__
    elif exc.validator == 'required':
        detail = 'missing required fields: '+', '.join(k for k in exc.validator_value if k not in exc.instance)
    elif exc.validator in ('minItems', 'maxItems'):
        detail = f'{exc.validator}={exc.validator_value}; use the specified array structure'
    elif exc.validator in ('minimum', 'maximum'):
        detail = 'coordinate outside the supplied local image range'
    elif exc.validator == 'additionalProperties':
        allowed = sorted(exc.schema.get('properties', {}))
        extra = sorted(set(exc.instance) - set(allowed))
        detail = 'unexpected fields: '+', '.join(extra)+'; allowed fields: '+', '.join(allowed)
    elif exc.validator == 'enum':
        detail = 'received '+repr(exc.instance)+'; expected one of '+', '.join(map(str, exc.validator_value))
    else:
        detail = 'violates '+str(exc.validator)+'; follow the supplied schema exactly'
    return path+': '+detail


def _object(properties):
    return {'type': 'object', 'properties': properties,
            'required': list(properties), 'additionalProperties': False}


def input_dimensions(size):
    if (len(size) != 2 or any(type(n) is not int or n < 2 for n in size)):
        raise ObservationError('Input size must contain two integer dimensions >= 2')
    return tuple(size)


def observation_schema(input_size):
    w, h = input_dimensions(input_size)
    ident = {'type': 'string', 'minLength': 1}

    def ids(count=None, minimum=2):
        value = {'type': 'array', 'items': ident, 'minItems': count or minimum}
        if count is not None:
            value['maxItems'] = count
        return value

    parts = [_object({'id': ident, 'kind': {'const': kind}, 'point_ids': ids(n)})
             for kind, n in POINT_COUNTS.items()]
    segment = {'oneOf': [
        _object({'kind': {'const': 'line'}, 'point_ids': ids(2)}),
        _object({'kind': {'const': 'curve'}, 'point_ids': ids(minimum=3)})]}
    parts.append(_object({'id': ident, 'kind': {'const': 'path'},
                          'closed': {'type': 'boolean'},
                          'segments': {'type': 'array', 'items': segment, 'minItems': 1}}))
    point = _object({'id': ident, 'xy': {'type': 'array', 'minItems': 2, 'maxItems': 2,
        'prefixItems': [{'type': 'number', 'minimum': 0, 'maximum': w-1},
                        {'type': 'number', 'minimum': 0, 'maximum': h-1}], 'items': False}})
    return _object({'object_id': ident,
                    'status': {'enum': ['ok', 'partial', 'needs_context', 'unsupported']},
                    'points': {'type': 'array', 'items': point},
                    'parts': {'type': 'array', 'items': {'oneOf': parts}},
                    'reason': {'type': 'string'},
                    'uncertain': {'type': 'array', 'items': {'type': 'string'}}})


def part_refs(part):
    if part['kind'] == 'path':
        return [pid for segment in part['segments'] for pid in segment['point_ids']]
    return list(part['point_ids'])


def model_observation_schema(input_size):
    """Small wire contract; backend identity and legacy defaults stay internal."""
    schema = observation_schema(input_size)
    for field in ('object_id', 'uncertain'):
        del schema['properties'][field]
    schema['required'] = ['status', 'points', 'parts']
    return schema


def normalize_observation(raw, object_id, input_size, *, audit=None):
    """Accept the compact protocol and historical records without changing geometry.

    Only the known top-level version metadata is removed. Unknown fields, wrong
    legacy identity, invalid references and unsupported geometry still fail.
    """
    if not isinstance(raw, dict):
        raise ObservationError('response: expected object, received '+type(raw).__name__)
    value = copy.deepcopy(raw)
    metadata = {}
    if 'version' in value:
        if not isinstance(value['version'], str):
            raise ObservationError('version: expected string metadata')
        metadata['version'] = value.pop('version')
    defaults = {'object_id': object_id, 'reason': '', 'uncertain': []}
    added = {key: val for key, val in defaults.items() if key not in value}
    value.update(added)
    validate_observation(value, object_id, input_size)
    if audit is not None:
        audit.update(protocol=MODEL_VERSION, backend_added=added,
                     removed_metadata=metadata, legacy_fields_present=[
                         key for key in ('object_id', 'uncertain') if key in raw],
                     geometry_changed=False)
    return value


def validate_observation(raw, object_id, input_size):
    """Validate without mutating, rounding, merging or inventing points."""
    try:
        Draft202012Validator(observation_schema(input_size)).validate(raw)
    except ValidationError as exc:
        raise ObservationError(protocol_error(exc)) from exc
    if raw['object_id'] != object_id:
        raise ObservationError('Wrong object ID')
    points = {p['id']: p['xy'] for p in raw['points']}
    if len(points) != len(raw['points']):
        raise ObservationError('Duplicate point IDs')
    for xy in points.values():
        if any(not math.isfinite(x) for x in xy):
            raise ObservationError('Coordinates must be finite')
    if raw['status'] in ('unsupported', 'needs_context') and (points or raw['parts']):
        raise ObservationError('Unsupported/needs_context must have empty geometry; use partial to retain parts')
    if raw['status'] == 'ok' and (not points or not raw['parts']):
        raise ObservationError('Empty ok response')
    if raw['status'] != 'ok' and not raw['reason'].strip():
        raise ObservationError('Incomplete observations require a reason')
    seen = set()
    used = set()
    for part in raw['parts']:
        if part['id'] in seen:
            raise ObservationError('Duplicate part IDs')
        seen.add(part['id'])
        refs = part_refs(part)
        if any(pid not in points for pid in refs):
            raise ObservationError('Missing point reference')
        used.update(refs)
        if part['kind'] != 'path':
            if len(set(refs)) != len(refs):
                raise ObservationError('Repeated primitive point ID')
        else:
            segments = part['segments']
            if any(a['point_ids'][-1] != b['point_ids'][0]
                   for a, b in zip(segments, segments[1:])):
                raise ObservationError('Disconnected path')
            if part['closed'] and segments[-1]['point_ids'][-1] != segments[0]['point_ids'][0]:
                raise ObservationError('Unclosed path')
    if used != set(points):
        raise ObservationError('Unused points')
    return raw
