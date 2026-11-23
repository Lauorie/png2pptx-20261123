"""Small shared utilities; import the unchanged reference geometry implementation."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
REFERENCE = Path(os.environ.get('SHAPE_REFERENCE_ROOT', '/mnt/shared-storage-user/intern7shared/yinjinhui/scifig_png2ppt/ref/png2pptx_geometry/experiments/shape_geometry27_20261009'))
CODE_ROOT = Path(os.environ['SHAPE_REFERENCE_ROOT']).resolve() if os.environ.get('SHAPE_REFERENCE_ROOT') else (
    HERE / 'reference' if (HERE / 'reference/src').exists() else HERE.parent.parent)
sys.path[:0] = [str(CODE_ROOT / 'src'), str(CODE_ROOT / 'vendor')]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def box_iou(a, b):
    area = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return float(area / max(1e-9, (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area))


def union_box(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def clip_box(box, size):
    return [max(0., min(size[0]-2., box[0])), max(0., min(size[1]-2., box[1])),
            max(1., min(size[0]-1., box[2])), max(1., min(size[1]-1., box[3]))]


def public_proposal(p):
    return {k: v for k, v in p.items() if k not in {'small_mask', 'contours', 'contour'}}


def update_counts(scene):
    scene['counts'] = {**scene.get('counts', {}), 'candidates': len(scene['objects']),
        'terminal_objects': len(scene['objects']),
        'successful_parts': sum(p['status'] == 'ok' for p in scene['parts'].values()),
        'failed_parts': sum(p['status'] != 'ok' for p in scene['parts'].values()),
        'accepted_by_model': sum(o['state'] == 'accepted_by_model' for o in scene['objects'].values()),
        'unresolved_objects': len(scene.get('unresolved', []))}


def assert_preserved(baseline, scene, changed_keys):
    """A patch must not delete any pre-existing object/part or alter its siblings."""
    if not set(baseline['objects']) <= set(scene['objects']):
        raise ValueError('A baseline object was lost')
    if not set(baseline['parts']) <= set(scene['parts']):
        raise ValueError('A baseline part was lost')
    for key, part in baseline['parts'].items():
        if key not in changed_keys and scene['parts'][key] != part:
            raise ValueError('Unaffected baseline part changed: ' + key)


def apply_fit(scene, fit, candidate, target_key=None):
    """Replace exactly one part, or append a new object. Keep all successful siblings."""
    from shape_integration.fitting.scene import native_parameters
    parts = [copy.deepcopy(p) for p in fit['parts'] if p['status'] == 'ok']
    if not parts:
        raise ValueError('No usable parts')
    if target_key is not None and len(parts) != 1:
        raise ValueError('A part replacement must contain exactly one part; do not drop siblings')
    out = copy.deepcopy(scene)
    oid = out['parts'][target_key]['object_id'] if target_key else candidate['id']
    if not target_key:
        if oid in out['objects']:
            raise ValueError('Addition would overwrite an existing object')
        out['objects'][oid] = {'id': oid, 'candidate': candidate, 'state': 'accepted_by_model',
            'visual_status': 'accepted_by_incremental_review', 'part_keys': [], 'point_keys': [],
            'sam_refine': True, 'frame': fit['frame']}
    obj = out['objects'][oid]
    point_remap = {}
    point_id_remap = {}
    point_namespace = out['parts'][target_key]['key'][0] if target_key else None
    point_tag = hashlib.sha256(json.dumps(fit['key']).encode()).hexdigest()[:12]
    for point in fit['points']:
        p = copy.deepcopy(point)
        pid = point['id']
        if target_key:
            p['id'] = 'SR_'+point_tag+'_'+pid
            p['key'] = [point_namespace, oid, p['id']]
        else:
            p['key'][1] = oid
        key = json.dumps(p['key'], ensure_ascii=False)
        point_remap[pid] = key
        point_id_remap[pid] = p['id']
        out['points'][key] = p
        if key not in obj['point_keys']:
            obj['point_keys'].append(key)
    for p in parts:
        if target_key:
            old = out['parts'][target_key]
            p['key'], p['id'] = old['key'], old['id']
            p['sam_previous_geometry'] = {'commands': old['commands'], 'bbox_2d': old['bbox_2d']}
        else:
            p['key'][1] = oid
        key = target_key or json.dumps(p['key'], ensure_ascii=False)
        p['object_id'] = oid
        p['point_keys'] = [point_remap[pid] for pid in p['observed_point_ids']]
        p['observed_point_ids'] = [point_id_remap[pid] for pid in p['observed_point_ids']]
        observed = p['observed_part']
        observed['id'] = p['id']
        if 'point_ids' in observed:
            observed['point_ids'] = [point_id_remap[pid] for pid in observed['point_ids']]
        for segment in observed.get('segments', []):
            segment['point_ids'] = [point_id_remap[pid] for pid in segment['point_ids']]
        p['native_parameters_original'] = native_parameters(p, fit['frame'])
        p['sam_refine_frame'] = fit['frame']
        out['parts'][key] = p
        out['paths'][key] = {'key': p['key'], 'commands': p['commands'], 'object_id': oid}
        if key not in obj['part_keys']:
            obj['part_keys'].append(key)
    update_counts(out)
    return out
