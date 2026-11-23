"""Object-level fitting with preserved observations and per-part failures."""
import copy
import math
from collections import defaultdict

import numpy as np

from .contracts import VERSION, part_refs, validate_observation
from .frames import FitFrame
from .paths import (box_commands, curve_commands, ellipse_commands, fit_rectangle,
                    rectangle_corners, sample_commands)
from .primitives import FitError, fit_boundary


def bounds(commands):
    points = np.asarray(sample_commands(commands, 256))
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise FitError('Invalid or empty fitted path')
    return [*points.min(axis=0).tolist(), *points.max(axis=0).tolist()]


def union_bounds(parts):
    boxes = [p['bbox_2d'] for p in parts if p['status'] == 'ok']
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def fit_part(part, table):
    kind = part['kind']
    params = {}
    if kind == 'cylinder':
        from . import cylinder
        params, diagnostics = cylinder.fit([table[pid] for pid in part['point_ids']])
        commands = cylinder.commands(params)
    elif kind in ('ellipse', 'rounded_rectangle'):
        fitted = fit_boundary(kind, [table[pid] for pid in part['point_ids']])
        params = {k: v for k, v in fitted.items() if k != 'point_fit_diagnostics'}
        diagnostics = fitted['point_fit_diagnostics']
        a, c, w, h = (params[k] for k in ('angle_radians', 'center', 'width', 'height'))
        if kind == 'ellipse':
            commands = ellipse_commands(c, [math.cos(a)*w/2, math.sin(a)*w/2],
                                        [-math.sin(a)*h/2, math.cos(a)*h/2])
        else:
            commands = box_commands([*c, w, h, a], params['radius'])
            diagnostics['radius_policy'] = 'One radius within this part; no cross-part tying.'
    elif kind == 'rectangle':
        par, _, diagnostics = fit_rectangle([table[pid] for pid in part['point_ids']])
        params = dict(center=par[:2], width=par[2], height=par[3], angle_radians=par[4])
        commands = box_commands(par)
    elif kind == 'path':
        commands = []
        previous_end = None
        for segment in part['segments']:
            xy = [table[pid] for pid in segment['point_ids']]
            if any(math.dist(a, b) < 1e-9 for a, b in zip(xy, xy[1:])):
                raise FitError('Zero-length path segment or duplicate consecutive curve knots')
            if segment['kind'] == 'line':
                cmd = [['M', *xy[0]], ['L', *xy[1]]]
            else:
                periodic = (part['closed'] and len(part['segments']) == 1
                            and segment['point_ids'][0] == segment['point_ids'][-1])
                cmd = curve_commands(xy, periodic=periodic)
            # Open paths can contain independent/branching strokes. Keep their
            # explicit starts; never invent a connector to the previous stroke.
            commands.extend(cmd[1:] if previous_end == segment['point_ids'][0] else cmd)
            previous_end = segment['point_ids'][-1]
        if part['closed']:
            commands.append(['Z'])
        diagnostics = {'interpolates_observed_points': True,
                       'open_curve_policy': 'coordinate-wise shape-preserving cubic interpolation',
                       'closed_curve_policy': 'periodic cubic interpolation',
                       'disconnected_open_segments': 'separate pen strokes'}
    else:
        raise FitError('Unsupported part kind: ' + str(kind))
    return {'parameters': params, 'commands': commands, 'bbox_2d': bounds(commands),
            'diagnostics': diagnostics}


def shared_point_diagnostics(parts, table):
    """Report disagreement; do not weld independent fits or rewrite points."""
    predictions = defaultdict(list)
    for part in parts:
        if part['status'] != 'ok':
            continue
        observed = part['observed_part']
        refs = list(dict.fromkeys(part_refs(observed)))
        if part['kind'] == 'rectangle':
            p = part['parameters']
            xy = rectangle_corners([*p['center'], p['width'], p['height'], p['angle_radians']])
            method = 'fitted_corners'
        elif part['kind'] == 'path':
            xy = np.asarray([table[pid] for pid in refs])
            method = 'interpolated_knots'
        else:
            samples = np.asarray(sample_commands(part['commands'], 256))
            xy = [samples[np.argmin(np.linalg.norm(samples-table[pid], axis=1))] for pid in refs]
            method = 'nearest_sampled_contour_point'
        for pid, q in zip(refs, xy):
            predictions[pid].append({'part_id': part['id'], 'xy': np.asarray(q).tolist(),
                                     'method': method})
    records = []
    for pid, items in predictions.items():
        if len(items) < 2:
            continue
        separation = max(math.dist(a['xy'], b['xy']) for a in items for b in items)
        records.append({'point_id': pid, 'predictions': items,
                        'max_separation_px': separation,
                        'approximate': any(p['method'] == 'nearest_sampled_contour_point' for p in items)})
    return {'coordinate_frame': 'crop_local_pixel_centers', 'shared_points': records,
            'joint_constraints_applied': False}


def fit_object(raw, frame, *, object_id, namespace, provenance=None):
    """Fit all parts without network calls. Invalid shared contracts raise.

    namespace identifies the case/revision; local point and part IDs remain
    intact. A list-valued key avoids delimiter collisions in caller IDs.
    """
    if not isinstance(frame, FitFrame):
        frame = FitFrame.from_metadata(frame)
    if not isinstance(namespace, str) or not namespace:
        raise ValueError('A nonempty case/revision namespace is required')
    validate_observation(raw, object_id, frame.input_size)
    table = {p['id']: p['xy'] for p in raw['points']}
    parts = []
    for observed in raw['parts']:
        part = {'id': observed['id'], 'kind': observed['kind'], 'object_id': object_id,
                'key': [namespace, object_id, observed['id']],
                'observed_point_ids': list(dict.fromkeys(part_refs(observed))),
                'observed_part': copy.deepcopy(observed)}
        try:
            part.update(fit_part(observed, table), status='ok')
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            part.update(status='failed', error=str(exc),
                        diagnostics=copy.deepcopy(getattr(exc, 'diagnostics', {})),
                        commands=[], parameters={}, bbox_2d=None)
        parts.append(part)
    for part in parts:
        if part['kind']=='cylinder' and part['status']=='ok':
            cap=next((p for p in parts if p['kind']=='ellipse' and p['status']=='ok' and
                      p['observed_part']['point_ids']==part['observed_part']['point_ids'][:8]),None)
            if cap:part['native_top_face_id']=cap['id']
    succeeded = sum(p['status'] == 'ok' for p in parts)
    failed = len(parts)-succeeded
    status = ('not_run' if not parts else 'failed' if not succeeded else
              'partial' if failed or raw['status'] != 'ok' else 'ok')
    return {'schema_version': VERSION, 'object_id': object_id, 'key': [namespace, object_id],
            'coordinate_frame': 'crop_local_pixel_centers', 'frame': frame.metadata(),
            'observation': copy.deepcopy(raw), 'observation_coordinate_frame': 'crop_local_pixel_centers',
            'observation_status': raw['status'], 'fit_status': status,
            'completeness_status': 'unreviewed' if status == 'ok' else 'incomplete',
            'provenance': copy.deepcopy(provenance or {}),
            'parts': parts, 'part_counts': {'observed': len(parts), 'succeeded': succeeded, 'failed': failed},
            'bbox_2d': union_bounds(parts), 'bbox_scope': 'successful_parts_only',
            'bbox_method': 'path samples, 256 samples per cubic',
            'points': [{'id': p['id'], 'key': [namespace, object_id, p['id']],
                        'xy_local': list(p['xy']), 'xy_global': frame.to_global(p['xy']),
                        'xy_original': frame.to_original(p['xy'])} for p in raw['points']],
            'shared_point_diagnostics': shared_point_diagnostics(parts, table)}


def transform_fit(result):
    """Map every endpoint/control point; local primitive parameters stay local."""
    if result['coordinate_frame'] != 'crop_local_pixel_centers':
        raise ValueError('Expected a local fit, refusing a second transform')
    frame = FitFrame.from_metadata(result['frame'])
    out = copy.deepcopy(result)
    out['coordinate_frame'] = 'original_pixel_centers'
    for part in out['parts']:
        part['parameters_local'] = part.pop('parameters')
        part['diagnostics_local'] = part.pop('diagnostics')
        if part['status'] != 'ok':
            continue
        if part.get('_preserved_original_commands') is not None:
            part['commands']=part.pop('_preserved_original_commands')
            part['bbox_2d']=bounds(part['commands'])
            continue
        part['commands'] = [[c[0], *[v for i in range(1, len(c), 2)
                                      for v in frame.to_original(c[i:i+2])]] for c in part['commands']]
        part['bbox_2d'] = bounds(part['commands'])
    out['bbox_2d'] = union_bounds(out['parts'])
    return out
