"""Versioned target -> paths -> parts protocol. No image-dependent routing rules.

Only parts within a path must be continuous. Different paths owned by the same
target may branch, cross or be disconnected. This module never invents joins.
"""
import copy
import hashlib
import json
import math
import re

HEADS = {'none', 'triangle', 'stealth', 'arrow', 'diamond', 'oval', 'unknown'}
LINE_STYLES = {'solid', 'dashed', 'dotted'}


def revision(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def number(x):
    return type(x) in (int, float) and math.isfinite(x)


def identifier(x):
    if not isinstance(x, str) or not x or '/' in x:
        raise ValueError('IDs must be nonempty strings without /')
    return x


def check_points(points, size, minimum):
    if not isinstance(points, list) or len(points) < minimum:
        raise ValueError('Insufficient points')
    for p in points:
        if not isinstance(p, list) or len(p) != 2 or not all(number(x) for x in p):
            raise ValueError('Points are finite [x,y] pairs')
        if not (0 <= p[0] < size[0] and 0 <= p[1] < size[1]):
            raise ValueError('New points must lie inside CLEAN; request context for unseen geometry')
    if any(a == b for a, b in zip(points, points[1:])):
        raise ValueError('Consecutive points must differ')


def check_style(style, block=False):
    if not isinstance(style, dict) or not isinstance(style.get('color'), str) or not re.fullmatch(r'#[0-9a-fA-F]{6}', style['color']):
        raise ValueError('Style requires #rrggbb color')
    if not block and (not number(style.get('width_px')) or style['width_px'] <= 0):
        raise ValueError('Positive finite CLEAN-pixel width required')
    if style.get('line_style') not in LINE_STYLES:
        raise ValueError('line_style is solid, dashed or dotted')
    heads = style.get('heads')
    if not isinstance(heads, list) or len(heads) != 2 or any(h not in HEADS for h in heads):
        raise ValueError('heads are [first,last] head enums')
    if block and heads != ['none', 'none']:
        raise ValueError('Filled contour already includes its head')
    if 'dash_params' in style:
        d = style['dash_params']
        if not isinstance(d, dict) or not all(number(d.get(k)) and d[k] > 0 for k in ('mark_px', 'gap_px')):
            raise ValueError('dash_params require positive mark_px and gap_px')
        if 'phase_px' in d and not number(d['phase_px']):
            raise ValueError('Finite dash phase required')


def check_path(path, size):
    if not isinstance(path, dict):
        raise ValueError('Path object required')
    identifier(path.get('id'))
    kind = path.get('kind')
    if kind not in ('stroke', 'block_arrow'):
        raise ValueError('Path kind is stroke or block_arrow')
    check_style(path.get('style'), block=kind == 'block_arrow')
    if kind == 'block_arrow':
        check_points(path.get('contour'), size, 3)
        if len({tuple(p) for p in path['contour']}) < 3 or path.get('parts'):
            raise ValueError('A filled contour needs three distinct vertices and no shaft parts')
        return
    if path.get('contour'):
        raise ValueError('A stroke uses parts, not a filled contour')
    parts = path.get('parts')
    if not isinstance(parts, list) or not parts:
        raise ValueError('Stroke requires ordered parts')
    previous = None
    for part in parts:
        if not isinstance(part, dict) or part.get('kind') not in ('line', 'curve'):
            raise ValueError('Part kind is line or curve')
        points = part.get('points')
        check_points(points, size, 2 if part['kind'] == 'line' else 3)
        if part['kind'] == 'line' and len(points) != 2:
            raise ValueError('A line has exactly two endpoints; use multiple line parts for corners')
        if previous is not None and previous != points[0]:
            raise ValueError('Parts within one path must meet; use separate paths for branches')
        previous = points[-1]


def validate(document):
    d = copy.deepcopy(document)
    if not isinstance(d, dict) or d.get('version') != 1 or d.get('frame') != 'CLEAN':
        raise ValueError('Protocol requires version=1 and frame=CLEAN')
    size = d.get('size')
    if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
        raise ValueError('size is [CLEAN width,height] in pixels')
    if not isinstance(d.get('targets'), list):
        raise ValueError('targets list required')
    seen = set()
    for target in d['targets']:
        name = identifier(target.get('id'))
        if name in seen or not isinstance(target.get('paths'), list):
            raise ValueError('Unique target IDs and paths arrays required')
        seen.add(name); paths = set()
        for p in target['paths']:
            check_path(p, size)
            if p['id'] in paths:
                raise ValueError('Path IDs must be unique within their target')
            paths.add(p['id'])
    return d


def path_index(document):
    return {t['id']+'/'+p['id']: p for t in document['targets'] for p in t['paths']}


def apply_patch(document, answer, expected_revision, writable_targets=None, context_size=None):
    """Validate atomically; failed/stale/unresolved edits leave the source intact."""
    if revision(document) != expected_revision:
        raise ValueError('Stale base revision; rebase the affected targets before editing')
    original = validate(document); d = copy.deepcopy(original)
    if not isinstance(answer, dict) or answer.get('status') not in ('pass', 'patch', 'need_context', 'unresolved'):
        raise ValueError('status is pass, patch, need_context or unresolved')
    edits = answer.get('edits')
    if not isinstance(edits, list):
        raise ValueError('edits array required')
    if answer['status'] != 'patch':
        if edits:
            raise ValueError('Only status=patch may contain edits')
        if answer['status'] == 'need_context':
            import contracts
            if context_size is None:
                raise ValueError('CONTEXT frame unavailable')
            contracts.bbox(answer.get('context_bbox'), context_size)
        elif answer.get('context_bbox') is not None:
            raise ValueError('context_bbox belongs only to need_context')
        return original
    if not edits or answer.get('context_bbox') is not None:
        raise ValueError('Patch requires edits and no expansion bbox')
    targets = {t['id']: t for t in d['targets']}
    writable = set(targets) if writable_targets is None else set(writable_targets)
    touched = set()
    for edit in edits:
        if not isinstance(edit, dict):raise ValueError('Edit object required')
        tid, pid, op = edit.get('target'), edit.get('path'), edit.get('op')
        if tid not in targets or tid not in writable:
            raise ValueError('Edit targets must belong to this request; neighbors are read-only')
        identifier(pid); key = (tid, pid)
        if key in touched:
            raise ValueError('Each path may be edited once per atomic patch')
        touched.add(key); paths = targets[tid]['paths']; positions = {p['id']: i for i, p in enumerate(paths)}
        if op == 'add':
            if pid in positions:raise ValueError('add requires a new path ID')
            value = copy.deepcopy(edit.get('value'))
            if not isinstance(value, dict):raise ValueError('add requires a complete path value')
            if 'id' in value and value['id'] != pid:raise ValueError('Conflicting path ID')
            value['id'] = pid; paths.append(value)
        else:
            if pid not in positions:raise ValueError('Unknown existing path ID')
            i = positions[pid]
            if op == 'delete':paths.pop(i)
            elif op == 'replace':
                value = copy.deepcopy(edit.get('value'))
                if not isinstance(value, dict):raise ValueError('replace requires a complete path value')
                if 'id' in value and value['id'] != pid:raise ValueError('Conflicting path ID')
                value['id'] = pid; paths[i] = value
            elif op == 'update_style':
                value = edit.get('style')
                if not isinstance(value, dict) or not value or set(value)-{'color','width_px','line_style','heads','dash_params'}:
                    raise ValueError('update_style accepts explicit style fields only')
                # A changed dash category cannot inherit a measured old cycle.
                if ('line_style' in value and value['line_style'] != paths[i]['style']['line_style'] and 'dash_params' not in value):
                    paths[i]['style'].pop('dash_params', None)
                paths[i]['style'].update(copy.deepcopy(value))
            else:raise ValueError('op is add, replace, delete or update_style')
    return validate(d)


def render_style(style):
    head = lambda h: None if h in ('none','unknown') else {'type':h,'width':'med','length':'med'}
    dash = dict(type=style['line_style'], preset={'solid':'solid','dashed':'dash','dotted':'sysDot'}[style['line_style']])
    dash.update(style.get('dash_params', {}))
    return dict(color=style['color'],width_px=style.get('width_px',1),dash=dash,
                start=head(style['heads'][0]),end=head(style['heads'][1]))


def rows(document):
    """Compile each path separately. No accidental bridge between branch paths."""
    import segmented_geometry as sg
    d=validate(document); output=[]
    for t in d['targets']:
        for path in t['paths']:
            rid=t['id']+'/'+path['id'];style=render_style(path['style'])
            baseline=dict(id=rid,parent_id=t['id'],label=rid,heads=path['style']['heads'],style=style,
                          coordinate_source='glm_protocol_paths')
            if path['kind']=='block_arrow':
                p=copy.deepcopy(path['contour'])
                output.append(dict(baseline,kind='block_arrow',contours=[p],start=p[0],via=p[1:-1],end=p[-1],closed=True,beziers=[]))
                continue
            nodes=[];parts=[];last=None;last_id=None
            for part in path['parts']:
                ids=[]
                for i,p in enumerate(part['points']):
                    if i==0 and last==p:nid=last_id
                    else:nid='n'+str(len(nodes)+1);nodes.append([nid,*p])
                    ids.append(nid)
                parts.append(dict(kind=part['kind'],nodes=ids));last=part['points'][-1];last_id=ids[-1]
            answer={'nodes':nodes,'routes':[dict(id=rid,segments=parts)]}
            sg.validate(answer,{rid:baseline},d['size'])
            output.extend(sg.geometry(answer,{rid:baseline}))
    return output


def from_legacy(answer,size,styles):
    """Import explicit legacy primitives, splitting ONLY at reference breaks.

    Every primitive and point is retained in original order. No proximity-based
    joining, geometry correction, branch assignment or head inference occurs.
    Branch import requires an explicit style without inferred endpoint heads.
    """
    table={n:[x,y] for n,x,y in answer['nodes']};targets=[];audit=[]
    for route in answer['routes']:
        paths=[];previous=None
        for seg in route['segments']:
            ids=seg['nodes']
            if not ids or any(n not in table for n in ids):raise ValueError('Undefined legacy node')
            if not paths or previous!=ids[0]:
                paths.append(dict(id='P'+str(len(paths)+1),kind='stroke',parts=[],style=copy.deepcopy(styles[route['id']])))
            paths[-1]['parts'].append(dict(kind=seg['kind'],points=[copy.deepcopy(table[n]) for n in ids]));previous=ids[-1]
        if len(paths)>1 and styles[route['id']]['heads']!=['none','none']:
            raise ValueError('Branched legacy heads need per-path classification; do not duplicate old route heads')
        targets.append(dict(id=route['id'],paths=paths))
        audit.append(dict(target=route['id'],paths=len(paths),parts=len(route['segments']),coordinates_changed=False))
    return validate(dict(version=1,frame='CLEAN',size=list(size),targets=targets)),audit
