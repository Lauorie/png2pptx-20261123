"""SAM-supported primitive fits and shared-face geometry for the existing PPT exporter."""
from __future__ import annotations

import copy
import json
import math
import re

import cv2
import numpy as np
from PIL import Image

from shape_integration.fitting.paths import box_commands, ellipse_commands, fit_rectangle
from shape_integration.fitting.primitives import fit_boundary, rr_signed_distance
from scipy.optimize import least_squares
from shape_integration.styling.geometry import bounds


def color(value):
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r'#[0-9a-fA-F]{6}', value):
        raise ValueError('Colors must be #RRGGBB or null')
    return value.upper()


def style(raw):
    fill = raw.get('fill')
    if isinstance(fill, dict):
        angle = float(fill['angle_deg'])
        start, end = color(fill['start_color']), color(fill['end_color'])
        if not math.isfinite(angle) or start is None or end is None:
            raise ValueError('A gradient needs two colors and a finite angle')
        fill = {'start_color': start, 'end_color': end, 'angle_deg': angle % 360}
    else:
        fill = color(fill)
    width = float(raw.get('width_px', 1.0))
    if not math.isfinite(width) or not 0 <= width <= 30:
        raise ValueError('Invalid stroke width')
    if raw.get('stroke') is not None and width == 0:
        raise ValueError('A visible stroke requires positive width')
    dash = raw.get('dash', 'solid')
    if dash not in {'solid', 'dash', 'dot', 'dash_dot'}:
        raise ValueError('Invalid stroke dash')
    return {'fill': fill, 'stroke': color(raw.get('stroke')), 'width_px': width, 'dash': dash}


def primitive_commands(parameters):
    if parameters['kind'] == 'ellipse':
        angle = parameters['angle_radians']
        u = [math.cos(angle)*parameters['width']/2, math.sin(angle)*parameters['width']/2]
        v = [-math.sin(angle)*parameters['height']/2, math.cos(angle)*parameters['height']/2]
        return ellipse_commands(parameters['center'], u, v)
    return box_commands([*parameters['center'], parameters['width'], parameters['height'],
                         parameters['angle_radians']], parameters.get('radius', 0))



def validate_fit_support(parameters, points):
    """Convergence is insufficient when a primitive leaves its observed support."""
    points = np.asarray(points, dtype=float)
    lo, hi = points.min(axis=0), points.max(axis=0)
    span = hi-lo
    diagonal = max(1., float(np.linalg.norm(span)))
    center = np.asarray(parameters['center'])
    margin = np.maximum(2., span*.2)
    if (np.any(center < lo-margin) or np.any(center > hi+margin)
            or max(parameters['width'], parameters['height']) > 1.6*diagonal):
        raise ValueError('Primitive fit escapes the observed contour. Reobserve outer boundary points; do not include the center or interior points.')
    diagnostic = parameters.get('point_fit_diagnostics', {})
    rms = diagnostic.get('rms_px', parameters.get('mask_fit_rms_px'))
    if rms is not None and rms > max(2., diagonal*.05):
        raise ValueError(f'Contour fit residual is too large ({rms:.2f}px). Reobserve the complete outer boundary or use a path for a non-primitive shape.')
    parameters['support_check'] = {'observed_bbox': [*lo.tolist(), *hi.tolist()],
                                   'rms_px': rms, 'diagonal_px': diagonal}


def sam_geometry(prediction, kind):
    contours = []
    if prediction.get('mask_path'):
        mask = (np.asarray(Image.open(prediction['mask_path'])) > 0).astype(np.uint8)
        mode = cv2.RETR_CCOMP if kind == 'path' else cv2.RETR_EXTERNAL
        contours, _ = cv2.findContours(mask, mode, cv2.CHAIN_APPROX_SIMPLE)
        contours = [item.reshape(-1, 2) for item in contours if len(item) >= 3]
    if not contours:
        contours = [np.asarray(prediction['contour'], dtype=np.float32)]
    # Dashed frames contain many disconnected strokes. Fit their joint support,
    # never just the largest dash; preserve all components for custom paths.
    contour = np.concatenate(contours).astype(np.float32)
    if prediction.get('mask_path') and kind != 'path':
        # Weight actual edge length, not the number of CHAIN_APPROX_SIMPLE corners.
        # Otherwise a tiny attached spur can outweigh a long straight box edge.
        perimeter = sum(cv2.arcLength(item.astype(np.float32), True) for item in contours)
        spacing = max(1., perimeter/1024)
        samples = []
        for component in contours:
            for start, finish in zip(component, np.roll(component, -1, axis=0)):
                length = float(np.linalg.norm(finish-start))
                if length > 0:
                    fractions = np.arange(0, length, spacing)/length
                    samples.extend(start+fractions[:, None]*(finish-start))
        if samples:
            contour = np.asarray(samples, dtype=np.float32)
    if len(contour) > 1024:
        contour = contour[np.linspace(0, len(contour)-1, 1024).astype(int)]
    if len(contour) < 3:
        raise ValueError('SAM mask has insufficient boundary points')
    parameters = None
    if kind in {'rectangle', 'rounded_rectangle'}:
        center, size, cv_angle = cv2.minAreaRect(contour)
        angle = math.radians(cv_angle)
        # Long straight edges are more reliable than one convex-hull outlier.
        # The fourth harmonic treats opposite/orthogonal rectangle edges alike.
        harmonic = 0j
        weight_sum = 0.
        for component in contours:
            perimeter = cv2.arcLength(component.astype(np.float32), True)
            polygon = cv2.approxPolyDP(component.astype(np.float32), max(.7, perimeter*.002), True).reshape(-1, 2)
            edges = np.roll(polygon, -1, axis=0)-polygon
            for dx, dy in edges:
                weight = float(dx*dx+dy*dy)
                if weight > 9:
                    theta = math.atan2(float(dy), float(dx))
                    harmonic += weight*complex(math.cos(4*theta), math.sin(4*theta))
                    weight_sum += weight
        coherence = abs(harmonic)/max(1., weight_sum)
        locked = coherence > .65
        if locked:
            angle = math.atan2(harmonic.imag, harmonic.real)/4
            pixel_angle = math.atan2(1.5, max(np.ptp(contour, axis=0)))
            if abs(angle) < pixel_angle:
                angle = 0.
        rotation = np.asarray([[math.cos(angle), -math.sin(angle)],
                               [math.sin(angle), math.cos(angle)]])
        local = contour @ rotation
        lo, hi = local.min(axis=0), local.max(axis=0)
        origin = ((lo+hi)/2) @ rotation.T
        size = np.maximum(hi-lo, 1.)
        scale = float(max(size))
        points = (contour-origin)/scale
        initial = [0., 0., size[0]/scale, size[1]/scale]
        lower, upper = [-1, -1, .001, .001], [1, 1, 3, 3]
        if not locked:
            initial.append(angle);lower.append(-2*math.pi);upper.append(2*math.pi)
        if kind == 'rounded_rectangle':
            initial.append(.12);lower.append(0);upper.append(.5)
        def unpack(value):
            theta = angle if locked else value[4]
            radius = value[-1] if kind == 'rounded_rectangle' else 0.
            return [*value[:4], theta, radius]
        result = least_squares(lambda value: rr_signed_distance(points, unpack(value)), initial,
            bounds=(lower, upper), loss='soft_l1', f_scale=1.5/scale, max_nfev=200)
        if not result.success or not np.isfinite(result.x).all():
            raise ValueError('SAM box fit did not converge')
        cx, cy, width, height, angle, radius = unpack(result.x)
        parameters = {'kind': kind, 'center': (origin+np.asarray([cx, cy])*scale).tolist(),
            'width': float(width*scale), 'height': float(height*scale), 'angle_radians': float(angle),
            'radius': float(radius*min(width, height)*scale),
            'edge_orientation_coherence': float(coherence),
            'mask_fit_rms_px': float(np.sqrt(np.mean(rr_signed_distance(points,unpack(result.x))**2))*scale)}
    elif kind == 'ellipse' and len(contour) >= 5:
        center, size, angle = cv2.fitEllipse(contour)
        parameters = {'kind': kind, 'center': list(center), 'width': max(1., size[0]),
                      'height': max(1., size[1]), 'angle_radians': math.radians(angle)}
    if parameters is not None:
        validate_fit_support(parameters, contour)
        commands = primitive_commands(parameters)
    else:
        commands = []
        for component in contours:
            polygon = cv2.approxPolyDP(component.astype(np.float32), .7, True).reshape(-1, 2)
            if len(polygon) >= 3:
                commands.extend([['M', *map(float, polygon[0])],
                                 *[['L', *map(float, point)] for point in polygon[1:]], ['Z']])
    return commands, parameters


def fit_cuboid(vertices, projection):
    """Fit x=o+a*u+b*v+c*d for an affine cuboid; retain observations if unsupported."""
    if projection != 'affine':
        return vertices, {'applied': False, 'reason': 'perspective or unspecified projection'}
    names = [key for key in vertices if re.fullmatch('[01]{3}', key)]
    if len(names) < 4:
        return vertices, {'applied': False, 'reason': 'fewer than four labeled cuboid vertices'}
    design = np.asarray([[1, *map(int, key)] for key in names], dtype=float)
    if np.linalg.matrix_rank(design) < 4:
        return vertices, {'applied': False, 'reason': 'projected basis is underdetermined'}
    observed = np.asarray([vertices[key] for key in names], dtype=float)
    basis, *_ = np.linalg.lstsq(design, observed, rcond=None)
    fitted = design @ basis
    rms = float(np.sqrt(np.mean(np.sum((observed-fitted)**2, axis=1))))
    limit = max(2., 0.015*float(np.linalg.norm(np.ptp(observed, axis=0))))
    if rms > limit:
        return vertices, {'applied': False, 'reason': 'observations violate affine projection', 'rms_px': rms}
    out = dict(vertices)
    out.update({key: point.tolist() for key, point in zip(names, fitted)})
    return out, {'applied': True, 'basis_origin_u_v_d': basis.tolist(), 'rms_px': rms}


def transport(commands, parameters, frame):
    x0, y0, x1, y1 = frame['box']
    width, height = frame['size']
    sx, sy = (x1-x0)/width, (y1-y0)/height
    offset = np.asarray([x0+(sx-1)/2, y0+(sy-1)/2])
    out = [[command[0], *[float(value * (sx if i % 2 else sy) + offset[(i-1) % 2])
                          for i, value in enumerate(command[1:], 1)]] for command in commands]
    native = None
    if parameters:
        angle = parameters['angle_radians']
        axes = np.diag([sx, sy]) @ np.asarray([[math.cos(angle), -math.sin(angle)],
                                             [math.sin(angle), math.cos(angle)]])
        axes = axes @ np.diag([parameters['width'], parameters['height']])
        center = np.asarray(parameters['center']) * [sx, sy] + offset
        if parameters['kind'] == 'ellipse':
            rotation, lengths, _ = np.linalg.svd(axes)
            native = dict(center=center.tolist(), width=float(lengths[0]), height=float(lengths[1]),
                          angle_radians=math.atan2(rotation[1, 0], rotation[0, 0]))
        elif abs(float(axes[:, 0] @ axes[:, 1])) < 1e-9*np.prod(np.linalg.norm(axes, axis=0)):
            if parameters['kind'] != 'rounded_rectangle' or math.isclose(sx, sy, rel_tol=1e-9):
                lengths = np.linalg.norm(axes, axis=0)
                native = dict(center=center.tolist(), width=float(lengths[0]), height=float(lengths[1]),
                    angle_radians=math.atan2(axes[1, 0], axes[0, 0]), radius=parameters.get('radius', 0)*sx)
    return out, native


def _decode_objects(raw, predictions, frame, namespace, order):
    if not isinstance(raw, dict) or not isinstance(raw.get('objects'), list):
        raise ValueError('Expected objects array')
    known = {pred['id']: pred for pred in predictions}
    objects = []
    for index, description in enumerate(raw['objects']):
        obj = copy.deepcopy(description)
        obj['id'] = f'{namespace}_O{index+1:03d}'
        obj['z_order'] = [*order, float(description.get('z_order', index))]
        obj['frame'] = frame
        obj['parts_decoded'] = []
        vertices = dict(description.get('vertices', {}))
        for part in description['parts']:
            for key, point in part.get('vertices', {}).items():
                if key in vertices and vertices[key] != point:
                    raise ValueError('Conflicting shared vertex: ' + key)
                vertices[key] = point
        if any(len(point) != 2 or not np.isfinite(point).all() for point in vertices.values()):
            raise ValueError('Non-finite shared vertices')
        if description.get('solid_type') == 'cuboid':
            vertices, obj['cuboid_fit'] = fit_cuboid(vertices, description.get('projection'))
        else:
            obj['cuboid_fit'] = {'applied': False}
        obj['vertices'] = vertices
        for part_index, part in enumerate(description['parts']):
            kind = part['kind']
            parameters = None
            use_sam = part.get('geometry_source', 'sam' if part.get('sam_id') else 'observed') == 'sam'
            if use_sam and part.get('sam_id'):
                if part['sam_id'] not in known:
                    raise ValueError('Unknown SAM instance')
                commands, parameters = sam_geometry(known[part['sam_id']], kind)
            elif kind in {'rectangle', 'ellipse', 'rounded_rectangle'}:
                points = [vertices[key] for key in part['point_ids']]
                if kind == 'rectangle':
                    p, _, diagnostic = fit_rectangle(points)
                    parameters = {'kind': kind, 'center': p[:2], 'width': p[2], 'height': p[3],
                                  'angle_radians': p[4], 'point_fit_diagnostics': diagnostic}
                else:
                    expected = 8 if kind == 'ellipse' else 16
                    if len(points) == expected and len(np.unique(points, axis=0)) == expected:
                        parameters = fit_boundary(kind, points)
                    else:
                        minimum = 5 if kind == 'ellipse' else 8
                        observed = np.asarray(points, dtype=float)
                        _, indices = np.unique(observed, axis=0, return_index=True)
                        unique = observed[np.sort(indices)]
                        if len(unique) < minimum or np.linalg.matrix_rank(unique-unique.mean(axis=0)) < 2:
                            raise ValueError(f'{kind} needs at least {minimum} distinct 2D contour observations; four sharp corners cannot determine a radius')
                        _, parameters = sam_geometry({'contour': unique.tolist()}, kind)
                        parameters['observation_count'] = len(unique)
                validate_fit_support(parameters, points)
                commands = primitive_commands(parameters)
            elif kind == 'path':
                commands = []
                for command in part['path']:
                    operator = command[0]
                    if operator not in {'M', 'L', 'C', 'Z'} or len(command)-1 != {'M': 1, 'L': 1, 'C': 3, 'Z': 0}[operator]:
                        raise ValueError('Invalid shared-vertex path command')
                    commands.append([operator, *[v for key in command[1:] for v in vertices[key]]])
                if not commands or commands[0][0] != 'M':
                    raise ValueError('A shape path must begin with M')
                if part.get('fill') is not None and commands[-1][0] != 'Z':
                    raise ValueError('A filled shape path must be closed')
            else:
                raise ValueError('Unsupported shape primitive')
            commands, native = transport(commands, parameters, frame)
            if not commands or not all(np.isfinite(c[1:]).all() for c in commands):
                raise ValueError('Non-finite shape path')
            key = [obj['id'], f'part{part_index+1}']
            decoded = {'key': key, 'id': key[1], 'object_id': obj['id'], 'kind': kind,
                       'status': 'ok', 'commands': commands, 'native_parameters_original': native,
                       'fit_support': parameters.get('support_check') if parameters else None,
                       'style': style(part), 'source': {'sam_id': part.get('sam_id'),
                       'point_ids': part.get('point_ids'), 'path': part.get('path')}}
            scale = math.sqrt((frame['box'][2]-frame['box'][0])*(frame['box'][3]-frame['box'][1]) /
                              (frame['size'][0]*frame['size'][1]))
            decoded['style']['width_px'] *= scale
            obj['parts_decoded'].append(decoded)
        if obj['parts_decoded']:
            boxes = np.asarray([bounds(part) for part in obj['parts_decoded']])
            obj['bbox'] = [*boxes[:, :2].min(axis=0).tolist(), *boxes[:, 2:].max(axis=0).tolist()]
            obj['state'] = 'reconstructed_unreviewed'
            if obj.get('solid_type') in {'cuboid', 'cylinder'} and len(obj['parts_decoded']) < 2:
                obj['state'] = 'unresolved_3d_faces'
            objects.append(obj)
    return objects



def decode_objects(raw, predictions, frame, namespace, order, *, collect_errors=False):
    if not collect_errors:
        return _decode_objects(raw, predictions, frame, namespace, order)
    if not isinstance(raw, dict) or not isinstance(raw.get('objects'), list):
        raise ValueError('Expected objects array')
    objects, errors = [], []
    for index, description in enumerate(raw['objects']):
        try:
            objects.extend(_decode_objects({'objects': [description]}, predictions, frame,
                                          f'{namespace}_{index:03d}', order))
        except (ValueError, KeyError, TypeError, IndexError) as error:
            errors.append({'input_index': index, 'label': description.get('label', ''),
                           'error': str(error), 'object': description})
    return objects, errors


def scene_from_objects(objects, size):
    scene = {'schema_version': 'shape_integration.scene.1', 'source_size': list(size),
             'objects': {}, 'parts': {}, 'points': {}, 'paths': {}, 'unresolved': []}
    styles = {}
    for obj in sorted(objects, key=lambda item: item['z_order']):
        obj = copy.deepcopy(obj)
        parts = obj.pop('parts_decoded')
        obj['part_keys'] = []
        for part in parts:
            key = json.dumps(part['key'], ensure_ascii=False)
            styles[key] = part.pop('style')
            scene['parts'][key] = part
            obj['part_keys'].append(key)
        scene['objects'][obj['id']] = obj
        if obj['state'] != 'reconstructed_unreviewed':
            scene['unresolved'].append({'object_id': obj['id'], 'state': obj['state']})
    relations = {'object_order': list(scene['objects']),
                 'part_orders': {key: obj['part_keys'] for key, obj in scene['objects'].items()},
                 'flat_order': [key for obj in scene['objects'].values() for key in obj['part_keys']]}
    return scene, styles, relations
