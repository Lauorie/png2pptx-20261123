"""SAM candidate accounting, coordinate transport, and source-image geometry gates."""
import copy
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates
from core import box_iou, clip_box, public_proposal, read, save, union_box
from shape_integration.fitting.paths import sample_commands
from legacy_geometry import sam_geometry, transport, primitive_commands

cv2.setNumThreads(1)


def path_points(commands, spacing=2., limit=3000):
    """Approximately uniform arc length; long straight edges must receive samples."""
    # Sampling each subpath separately avoids invented edges between disconnected parts.
    out, chunk = [], []
    chunks = []
    for cmd in commands:
        if cmd[0] == 'M' and chunk:
            chunks.append(chunk); chunk = []
        chunk.append(cmd)
    if chunk:
        chunks.append(chunk)
    for chunk in chunks:
        pts = np.asarray(sample_commands(chunk, 48), dtype=float)
        if len(pts) < 2:
            continue
        length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        valid = np.r_[True, np.diff(length) > 1e-8]
        length, pts = length[valid], pts[valid]
        if len(length) < 2:
            continue
        at = np.linspace(0, length[-1], min(limit, max(4, int(length[-1]/spacing)+1)))
        out.extend(np.column_stack([np.interp(at, length, pts[:, i]) for i in (0, 1)]))
    pts = np.asarray(out, dtype=float)
    return pts if len(pts) <= limit else pts[np.linspace(0, len(pts)-1, limit).astype(int)]


def edge_distance(source):
    rgb = np.asarray(source.convert('RGB'))
    edges = np.zeros(rgb.shape[:2], np.uint8)
    for channel in range(3):
        edges |= cv2.Canny(rgb[:, :, channel], 40, 110)
    return cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 5)


def score(commands, distance):
    pts = path_points(commands)
    if len(pts) < 4:
        raise ValueError('Insufficient boundary')
    d = map_coordinates(distance, [pts[:, 1], pts[:, 0]], order=1, mode='constant', cval=30.)
    return {'mean_clipped_px': float(np.minimum(d, 12).mean()), 'p90_px': float(np.percentile(d, 90)),
            'support_2px': float((d <= 2).mean()), 'sample_count': len(d)}


def improves(before, after):
    return (after['mean_clipped_px'] <= before['mean_clipped_px']-max(.20, .12*before['mean_clipped_px'])
            and after['p90_px'] <= before['p90_px']+.75
            and after['support_2px'] >= before['support_2px']-.03)


def corner_score(part, distance):
    """Measure rounded corners separately so long straight edges cannot hide them."""
    if part['kind'] != 'rounded_rectangle':
        return None
    points=[];pen=None
    for cmd in part['commands']:
        if cmd[0]=='C' and pen is not None:
            points.extend(sample_commands([['M',*pen],cmd],48))
        if cmd[0] in {'M','L','C'}:pen=cmd[-2:]
    if not points:return None
    pts=np.asarray(points,float)
    values=map_coordinates(distance,[pts[:,1],pts[:,0]],order=1,mode='constant',cval=30.)
    return {'mean_clipped_px':float(np.minimum(values,12).mean()),'p90_px':float(np.percentile(values,90)),
            'support_2px':float((values<=2).mean())}


def boundary_improves(before_part, after_part, distance):
    before,after=score(before_part['commands'],distance),score(after_part['commands'],distance)
    bc,ac=corner_score(before_part,distance),corner_score(after_part,distance)
    if bc and ac and (ac['mean_clipped_px']>bc['mean_clipped_px']+.25 or ac['p90_px']>bc['p90_px']+.75):
        return False  # Long edges must never hide a degraded rounded corner.
    if improves(before,after):return True
    return bool(bc and ac and ac['mean_clipped_px']<=bc['mean_clipped_px']-max(.4,.15*bc['mean_clipped_px'])
        and ac['p90_px']<=bc['p90_px']+.5 and after['mean_clipped_px']<=before['mean_clipped_px']+.1
        and after['p90_px']<=before['p90_px']+.5)


def needs_numeric_attention(part, distance):
    stats=score(part['commands'],distance);corners=corner_score(part,distance)
    # Avoid spending a model call to chase subpixel offsets on already aligned strokes.
    return stats['p90_px']>2.5 or stats['support_2px']<.9 or bool(corners and corners['p90_px']>3.)


def frame_crop(source, box, folder, long_edge=2048):
    folder.mkdir(parents=True, exist_ok=True)
    w, h = source.size
    pad = .05*max(box[2]-box[0], box[3]-box[1])
    rect = [max(0, math.floor(box[0]-pad)), max(0, math.floor(box[1]-pad)),
            min(w, math.ceil(box[2]+pad)+1), min(h, math.ceil(box[3]+pad)+1)]
    native = source.crop(rect)
    scale = long_edge/max(native.size)
    size = tuple(max(2, round(n*scale)) for n in native.size)
    image = native.resize(size, Image.Resampling.LANCZOS)
    image.save(folder/'clean.png')
    frame = {'box': rect, 'size': list(size)}
    save(folder/'frame.json', frame)
    return folder/'clean.png', frame


def normalize_predictions(result, frame, prefix, source_size):
    x0, y0, x1, y1 = frame['box']; w, h = frame['size']
    sx, sy = (x1-x0)/w, (y1-y0)/h
    offset = np.asarray([x0+(sx-1)/2, y0+(sy-1)/2])
    small_size = tuple(max(2, round(n*512/max(source_size))) for n in source_size)
    output = []
    for raw in result['predictions']:
        p = copy.deepcopy(raw)
        p['id'] = prefix+'-'+raw['id']; p['frame'] = frame
        p['bbox_local'] = p['bbox']
        # Detector boxes describe outer edges; geometry uses pixel centers.
        p['bbox'] = clip_box([x0+raw['bbox'][0]*sx-.5, y0+raw['bbox'][1]*sy-.5,
                              x0+raw['bbox'][2]*sx-.5, y0+raw['bbox'][3]*sy-.5], source_size)
        mask = np.asarray(Image.open(p['mask_path'])) > 0
        yy, xx = np.nonzero(mask)
        if not len(xx):
            continue
        actual = [float(xx.min()), float(yy.min()), float(xx.max()), float(yy.max())]
        p['mask_bbox'] = [actual[0]*sx+offset[0], actual[1]*sy+offset[1],
                          actual[2]*sx+offset[0], actual[3]*sy+offset[1]]
        span = np.maximum(1., np.asarray(p['bbox'][2:])-p['bbox'][:2])
        p['mask_extent_ratio'] = ((np.asarray(p['mask_bbox'][2:])-p['mask_bbox'][:2])/span).tolist()
        p['mask_bbox_iou'] = box_iou(p['bbox'], p['mask_bbox'])
        p['mask_quality_ok'] = bool(min(p['mask_extent_ratio']) >= .7 and p['mask_bbox_iou'] >= .6)
        small = Image.new('1', small_size)
        rect = [round(x0/source_size[0]*small_size[0]), round(y0/source_size[1]*small_size[1]),
                round(x1/source_size[0]*small_size[0]), round(y1/source_size[1]*small_size[1])]
        if rect[2] > rect[0] and rect[3] > rect[1]:
            small.paste(Image.fromarray(mask).resize((rect[2]-rect[0], rect[3]-rect[1]), Image.Resampling.NEAREST), rect[:2])
        p['small_mask'] = np.asarray(small)
        output.append(p)
    return output


def group_duplicates(predictions):
    """Both bbox and mask IoU required. Containment alone never merges objects."""
    groups = []
    for p in sorted(predictions, key=lambda p: (p['mask_quality_ok'], p['score']), reverse=True):
        chosen = None
        for group in groups:
            q = group['representative']
            if box_iou(p['bbox'], q['bbox']) < .82:
                continue
            intersection = np.count_nonzero(p['small_mask'] & q['small_mask'])
            union = np.count_nonzero(p['small_mask'] | q['small_mask'])
            if union and intersection/union >= .82:
                chosen = group; break
        if chosen:
            chosen['members'].append(p)
        else:
            groups.append({'representative': p, 'members': [p]})
    for i, g in enumerate(groups, 1):
        g['id'] = f'S{i:04d}'
    return groups


def group_hint(group):
    p = group['representative']
    return {'id': group['id'], 'category': p['category'], 'score': round(p['score'], 3),
            'bbox': [round(v, 1) for v in p['bbox']], 'mask_quality_ok': p['mask_quality_ok'],
            'mask_bbox_iou': round(p['mask_bbox_iou'], 3), 'duplicate_count': len(group['members'])}


def full_mask(p, source_size):
    x0, y0, x1, y1 = p['frame']['box']
    mask = Image.open(p['mask_path']).convert('L').resize((x1-x0, y1-y0), Image.Resampling.NEAREST)
    canvas = Image.new('L', source_size)
    canvas.paste(mask, (x0, y0))
    return np.asarray(canvas) > 0


def view(source, parts=(), groups=(), crop=None, long_edge=1600, show_boxes=True):
    image = source.copy()
    draw = ImageDraw.Draw(image)
    for label, part in parts:
        # Draw independent subpaths without joining their endpoints.
        chunk = []
        for cmd in [*part['commands'], ['M']]:
            if cmd[0] == 'M' and chunk:
                pts = [tuple(p) for p in sample_commands(chunk, 48)]
                if len(pts) > 1:
                    draw.line(pts, fill='#006bea', width=2)
                chunk = []
            if len(cmd) > 1 or cmd[0] == 'Z':
                chunk.append(cmd)
        box = part['bbox_2d']
        if label:
            draw.text((max(0, box[0]), max(0, box[1])), label, fill='#006bea', stroke_width=1, stroke_fill='white')
    for group in groups:
        p = group['representative']; mask = full_mask(p, source.size).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            pts = [tuple(map(int, q)) for q in contour.reshape(-1, 2)]
            if len(pts) > 1:
                draw.line(pts+[pts[0]], fill='#d00090', width=2)
        box = p['bbox']
        if show_boxes:
            draw.rectangle(box, outline='#ef8c00', width=1)
        draw.text((max(0, box[0]), max(0, box[1])), group['id'], fill='#d00090', stroke_width=1, stroke_fill='white')
    if crop:
        image = image.crop(crop)
    scale = long_edge/max(image.size)
    return image.resize(tuple(max(2, round(n*scale)) for n in image.size), Image.Resampling.LANCZOS)


def numeric_candidates(group, kind, source_size, folder):
    """Return alternatives, not an unconditional geometry replacement."""
    candidates = []
    p = group['representative']
    if not p['mask_quality_ok'] or kind not in {'rectangle', 'rounded_rectangle', 'ellipse'}:
        return candidates
    alternatives = [('representative', p)]
    if len(group['members']) > 1:
        # Only highly overlapping duplicate masks are eligible; compare union separately.
        mask = np.zeros((source_size[1], source_size[0]), bool)
        for member in group['members']:
            mask |= full_mask(member, source_size)
        path = folder/'duplicate_union.png'; Image.fromarray(mask.astype(np.uint8)*255).save(path)
        alternatives.append(('duplicate_union', {**p, 'mask_path': str(path),
            'frame': {'box': [0, 0, *source_size], 'size': list(source_size)}}))
    for name, pred in alternatives:
        try:
            cmds, params = sam_geometry(pred, kind)
            if params is None:
                continue  # A fragmented polygon is not a fitted ellipse/rounded rectangle.
            commands, native = transport(cmds, params, pred['frame'])
            pts = path_points(commands)
            box = [*pts.min(axis=0).tolist(), *pts.max(axis=0).tolist()]
            if box_iou(box, p['bbox']) < .65:
                continue
            candidates.append({'name': name, 'commands': commands, 'native_parameters_original': native,
                               'kind': kind, 'bbox_2d': box, 'status': 'ok'})
        except (ValueError, RuntimeError) as exc:
            save(folder/(name+'_failure.json'), {'error': str(exc)})
    return candidates


def snap_primitive(part, distance):
    """Small bounded source-edge adjustment, independently checked after fitting."""
    native = part.get('native_parameters_original')
    if not native or part['kind'] not in {'ellipse', 'rectangle', 'rounded_rectangle'}:
        return None
    kind = part['kind']; p = {**native, 'kind': kind}
    w, h = p['width'], p['height']; limit = max(2., min(w, h)*.06)
    initial = np.asarray([*p['center'], w, h, p['angle_radians'], p.get('radius', 0.)], float)
    lower = initial-np.asarray([limit, limit, 2*limit, 2*limit, .025, limit])
    upper = initial+np.asarray([limit, limit, 2*limit, 2*limit, .025, limit])
    lower[2:4] = np.maximum(1., lower[2:4]); lower[5] = 0.
    upper[5] = max(.001, min(w, h)/2)
    if kind != 'rounded_rectangle':
        initial, lower, upper = initial[:5], lower[:5], upper[:5]
    def unpack(x):
        return {'kind': kind, 'center': x[:2].tolist(), 'width': float(x[2]), 'height': float(x[3]),
                'angle_radians': float(x[4]), 'radius': float(min(x[5], min(x[2:4])/2)) if len(x)>5 else 0.}
    def objective(x):
        # Fixed-size samples give a stable residual dimension during optimization.
        pts = path_points(primitive_commands(unpack(x)), limit=400)
        pts = pts[np.linspace(0, len(pts)-1, 256).astype(int)]
        ds = map_coordinates(distance, [pts[:, 1], pts[:, 0]], order=1, mode='constant', cval=20.)
        return np.r_[np.minimum(ds, 8), .1*(x-initial)/np.maximum(upper-lower, 1e-6)]
    fit = least_squares(objective, np.clip(initial, lower+1e-8, upper-1e-8), bounds=(lower, upper),
                        diff_step=1e-4, max_nfev=70, loss='soft_l1', f_scale=1.)
    if not fit.success:
        return None
    params = unpack(fit.x); cmds = primitive_commands(params); pts = path_points(cmds)
    return {**part, 'name': part.get('name', 'baseline')+'_source_edges', 'commands': cmds,
            'native_parameters_original': params, 'bbox_2d': [*pts.min(axis=0).tolist(), *pts.max(axis=0).tolist()]}
