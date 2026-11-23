"""Compile the current observation's declared branch roles before geometry.

Part roles come from the same CLEAN observation as the geometry. A headless
route inside a registered compound is shape-owned; a directed route stays
arrow-owned whole.
"""
from copy import deepcopy

ROLES = ('body', 'contour', 'divider', 'intrinsic_stroke', 'connection', 'external_connection',
         'directional_connection', 'directional_head', 'lettering', 'image_content', 'uncertain')
PROTOCOL = 'declared_source_part_ownership.5'


def retained_receipt(current, base, part_ids):
    """Preserved geometry keeps its roles when an edit only returns replacements."""
    if current is None:
        return None
    result=deepcopy(current)
    inherited=(base or {}).get('retained_roles',{})
    roles={**inherited,**current.get('retained_roles',{})}
    result['retained_roles']={ident:roles.get(ident,'uncertain')for ident in part_ids}
    delegated={p['id']:deepcopy(p) for receipt in (base or {},current)
               for p in receipt.get('delegated_parts',[]) if p['id'] not in part_ids}
    if delegated:result['delegated_parts']=list(delegated.values())
    result['uncertain_parts']=[ident for ident,role in result['retained_roles'].items()if role=='uncertain']
    return result


def schema():
    return {'type': 'object', 'additionalProperties': False,
        'required': ['unit', 'unit_evidence', 'parts'], 'properties': {
            'unit': {'enum': ['neural_network', 'other']},
            'unit_evidence': {'type':'string','minLength':1},
            'parts': {'type': 'array', 'items': {'type': 'object',
                'additionalProperties': False, 'required': ['id', 'role'],
                'properties': {'id': {'type': 'string', 'minLength': 1},
                               'role': {'enum': list(ROLES)}}}}}}


def compile(raw, *, required=False, allow_compound_connections=True):
    """Strip wire metadata and return (geometry, receipt), never mutate input."""
    from .contracts import ObservationError, part_refs
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import ValidationError
    value = deepcopy(raw)
    ownership = value.pop('ownership', None)
    if ownership is None:
        if required:
            raise ObservationError('ownership: declare the source unit and one role per returned part')
        return value, None
    try:
        Draft202012Validator(schema()).validate(ownership)
    except ValidationError as exc:
        raise ObservationError('ownership: invalid source role declaration') from exc
    parts = value.get('parts')
    if not isinstance(parts, list) or not all(isinstance(p, dict) for p in parts):
        raise ObservationError('parts: expected geometry parts')
    ids = [p.get('id') for p in parts]
    role_ids = [p['id'] for p in ownership['parts']]
    if (any(not isinstance(ident,str) or not ident for ident in ids) or len(set(ids)) != len(ids)
            or sorted(ids) != sorted(role_ids) or len(set(role_ids)) != len(role_ids)):
        raise ObservationError('ownership: each returned part ID requires exactly one role')
    roles = {p['id']: p['role'] for p in ownership['parts']}
    head_points={pid for part in parts if roles[part['id']]=='directional_head'
                 for pid in part_refs(part)}
    retired, uncertain = [], []
    for part in parts:
        role = roles[part['id']]
        connection = role == 'connection'
        attached_directional=False
        if connection and head_points and part.get('kind')=='path' and not part.get('closed'):
            strokes=[]
            for segment in part.get('segments',[]):
                ids=segment['point_ids']
                if not strokes or strokes[-1][-1]!=ids[0]:strokes.append(list(ids))
                else:strokes[-1].extend(ids[1:])
            headed=[bool({stroke[0],stroke[-1]}&head_points)for stroke in strokes]
            if any(headed) and not all(headed):
                raise ObservationError('ownership: split headed and headless connections into separate parts')
            attached_directional=bool(headed)and all(headed)
        keep = (role not in ('external_connection', 'directional_connection', 'directional_head', 'lettering', 'image_content')
                and not attached_directional
                and (not connection or allow_compound_connections))
        if connection and keep and (part.get('kind') != 'path' or part.get('closed') is not False):
            raise ObservationError('ownership: headless compound connections must be open line paths; visibly directed connections belong whole to arrows')
        if not keep:
            retired.append({'id': part['id'], 'role': 'directional_connection'if attached_directional else role,
                            'owner': 'text' if role == 'lettering' else 'image' if role == 'image_content' else 'arrow'})
        elif role == 'uncertain':
            uncertain.append(part['id'])
    dropped = {p['id'] for p in retired}
    value['parts'] = [p for p in parts if p['id'] not in dropped]
    if dropped:
        used = {pid for part in value['parts'] for pid in part_refs(part)}
        value['points'] = [p for p in value.get('points', []) if p.get('id') in used]
    if not value['parts'] and value.get('status') == 'ok':
        value.update(status='unsupported', reason='The observed source content belongs to another branch.')
    elif uncertain and value.get('status') == 'ok':
        value.update(status='partial', reason='Source branch role remains uncertain for: '+', '.join(uncertain))
    receipt = {'protocol': PROTOCOL, 'unit': ownership['unit'],
               'compound_connection_scope_eligible': allow_compound_connections,
               'retained_roles': {p['id']: roles[p['id']] for p in value['parts']},
               'unit_evidence':ownership['unit_evidence'],
               'delegated_parts': retired, 'uncertain_parts': uncertain,
               'classification_and_geometry_call_shared': True}
    return value, receipt
