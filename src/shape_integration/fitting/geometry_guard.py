"""Local, conservative revision checks against cached SAM and source pixels.

SAM supplies ownership hints, not exact centerlines or a filled-shape IoU target.
Only source-supported spans of a previously fitted simple contour can veto an
edit. Missing/contradictory masks abstain; dashed gaps never become observations.
No model calls, target IDs, labels, or reference shapes are used here.
"""
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from ..sam_evidence import iou, project_mask
from .paths import command_strokes

VERSION = 'shape_integration.geometry_guard.3'


def straight_endpoints(commands):
    """Recognize one monotonic straight stroke, including segmented polylines."""
    if (len(commands) < 2 or commands[0][0] != 'M' or
            any(c[0] != 'L' for c in commands[1:]) or
            any(len(c) != 3 for c in commands)):
        return None
    points = np.asarray([c[1:] for c in commands], float)
    if not np.isfinite(points).all():
        return None
    delta = points[-1] - points[0]
    length = float(np.linalg.norm(delta))
    if length <= 1e-9:
        return None
    offset = points - points[0]
    along = offset @ (delta / length)
    perpendicular = np.abs(offset[:, 0]*delta[1] - offset[:, 1]*delta[0])/length
    # Only absorb coordinate serialization noise, not bent or retraced routes.
    if perpendicular.max() > .01 or np.any(np.diff(along) < -1e-9):
        return None
    return points[[0, -1]]


def unsupported_grid_additions(before,after,source):
    """Veto an invented divider only with visible parallel source witnesses.

    This does not infer a row/column count. Sparse dashes, occlusion and grids
    without strongly supported existing dividers leave the decision to review.
    """
    from shapely.geometry import Polygon,LineString,Point
    old={p['id']:p for p in before['parts'] if p['status']=='ok'}
    def straight(p):
        return straight_endpoints(p.get('commands',[]))
    existing=[(p,straight(p)) for p in old.values() if straight(p) is not None]
    added=[(p,straight(p)) for p in after['parts'] if p['status']=='ok' and p['id'] not in old and straight(p) is not None]
    contours=[Polygon(stroke) for p in old.values() if p.get('commands',[])[-1:]==[['Z']]
              and all(c[0] in ('M','L','Z') for c in p['commands'])
              for stroke in command_strokes(p['commands']) if len(stroke)>=4]
    contours=[p for p in contours if p.is_valid and p.area>0]
    if len(existing)<2 or not added or not contours:return None
    if any(f.get('coordinate_frame')!='original_pixel_centers' for f in (before,after)):return None
    bounds=[*before['bbox_2d'],*after['bbox_2d']]
    try:
        with Image.open(source) as opened:
            b=[max(0,math.floor(min(bounds[0],bounds[4]))-4),max(0,math.floor(min(bounds[1],bounds[5]))-4),
               min(opened.width,math.ceil(max(bounds[2],bounds[6]))+5),min(opened.height,math.ceil(max(bounds[3],bounds[7]))+5)]
            if b[0]>=b[2] or b[1]>=b[3]:return None
            crop=opened.convert('RGB').crop(b);scale=min(1.,1600/max(crop.size))
            size=tuple(max(2,round(v*scale)) for v in crop.size);rgb=np.asarray(crop.resize(size))
    except (OSError,ValueError):return None
    edges=np.maximum.reduce([cv2.Canny(rgb[:,:,i].copy(),20,60) for i in range(3)])
    distance=cv2.distanceTransform((edges==0).astype('uint8'),cv2.DIST_L2,5)
    def support(points):
        length=float(np.linalg.norm(points[1]-points[0]));count=max(2,math.ceil(length)+1)
        xy=(points[0]+np.linspace(0,1,count)[:,None]*(points[1]-points[0])-b[:2])*scale
        xy=np.rint(xy).astype(int);xy[:,0]=np.clip(xy[:,0],0,size[0]-1);xy[:,1]=np.clip(xy[:,1],0,size[1]-1)
        hit=distance[xy[:,1],xy[:,0]]<=max(1.,2*scale)
        gaps=np.flatnonzero(np.diff(np.r_[False,~hit,False].astype(int))).reshape(-1,2)
        longest=max((hi-lo for lo,hi in gaps),default=0)*length/count
        return float(hit.mean()),longest
    checks=[]
    def divider_in(poly,points):
        return (poly.buffer(2).covers(LineString(points)) and
                all(poly.boundary.distance(Point(p))<=3 for p in points) and
                poly.boundary.distance(Point(points.mean(axis=0)))>3)
    for part,points in added:
        length=float(np.linalg.norm(points[1]-points[0]))
        enclosing=[poly for poly in contours if divider_in(poly,points)]
        if length<40 or not enclosing:continue
        # Renaming an existing divider is not a geometry addition.
        if any(min(np.linalg.norm(points-line,axis=1).max(),
                   np.linalg.norm(points-line[::-1],axis=1).max())<=2 for _,line in existing):continue
        direction=(points[1]-points[0])/length;reference=[]
        for _,line in existing:
            delta=line[1]-line[0];norm=float(np.linalg.norm(delta))
            if (norm>=.6*length and abs(float(direction@delta/norm))>.995 and
                    any(divider_in(poly,line) for poly in enclosing)):
                reference.append(support(line)[0])
        if len(reference)<2 or float(np.median(reference))<.75:continue
        fraction,gap=support(points)
        checks.append({'part':part['id'],'source_supported_fraction':fraction,'longest_unsupported_span_px':gap,
                       'parallel_source_witnesses':len(reference),'reference_supported_fraction':float(np.median(reference))})
        if fraction<.15 and gap>max(40.,.1*length):
            return {'decision':'rollback','reason':'new_grid_divider_has_no_source_support','checks':checks}
    return {'decision':'abstain','reason':'no_clear_unsupported_grid_addition','checks':checks}


def geometry_signature(fit):
    """Ignore IDs, prose and namespaces; preserve actual contour topology."""
    parts = []
    for part in fit['parts']:
        if part['status'] == 'ok':
            parts.append([part['kind'], [[c[0], *[round(float(v), 6) for v in c[1:]]]
                                        for c in part['commands']]])
    value = [fit['fit_status'], fit['observation_status'], sorted(parts, key=repr)]
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def contour_samples(fit, spacing=1., limit=8000):
    """Arc-length samples; never join distinct moveto strokes."""
    chunks = []
    for part in fit['parts']:
        if part['status'] != 'ok':
            continue
        for stroke in command_strokes(part['commands'], 96):
            xy = np.asarray(stroke, float)
            if len(xy) < 2:
                continue
            length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
            unique = np.r_[True, np.diff(length) > 1e-9]
            xy, length = xy[unique], length[unique]
            if length[-1] <= 0:
                continue
            ts = np.linspace(0, length[-1], min(limit, max(2, math.ceil(length[-1]/spacing)+1)))
            chunks.append(np.column_stack([np.interp(ts, length, xy[:, k]) for k in (0, 1)]))
    return np.concatenate(chunks) if chunks else np.empty((0, 2))


def compare_revision(before, after, source, evidence, candidate):
    """Return a veto only for a clear loss of independently supported spans."""
    result = {'version': VERSION, 'decision': 'abstain', 'reason': 'no_usable_sam_support'}
    addition=unsupported_grid_additions(before,after,source)
    if addition and addition['decision']=='rollback':return {**result,**addition}
    if addition:result['grid_addition_check']=addition
    if (all(fit['fit_status'] == 'ok' and fit['observation_status'] == 'ok' for fit in (before, after))
            and geometry_signature(before) == geometry_signature(after)):
        return {**result, 'decision': 'stop', 'reason': 'repeated_geometry'}
    previous = [p for p in before['parts'] if p['status'] == 'ok']
    current = [p for p in after['parts'] if p['status'] == 'ok']
    # A projected solid's supported exterior may veto a collapsed revision;
    # it still does not certify internal faces or graph completeness.
    simple=(len(previous)==1 and len(current)==1 and
            previous[0]['kind'] in {'rectangle','rounded_rectangle','ellipse','cylinder'})
    projected=candidate.get('kind')=='projected_3d' and bool(previous) and bool(current)
    if before['fit_status'] != 'ok' or not (simple or projected):
        return {**result, 'reason': 'no_single_complete_primitive'}
    if not evidence or not hasattr(evidence, 'support'):
        return result
    select = getattr(evidence, 'geometry_support', evidence.support)
    group = select(candidate)
    if not group or not evidence.mask_usable(group, candidate):
        return result
    prediction = group['representative']
    result['sam_id'] = group['id']
    result['sam_score'] = prediction['score']
    if iou(before['bbox_2d'], prediction['bbox']) < .5:
        return {**result, 'reason': 'sam_and_previous_extent_disagree'}
    for fit in (before, after):
        if fit['coordinate_frame'] != 'original_pixel_centers':
            raise ValueError('Geometry guard requires original pixel coordinates')
    try:
        with Image.open(source) as im:
            bounds = [before['bbox_2d'], after['bbox_2d'], prediction['bbox']]
            crop = [max(0, math.floor(min(b[0] for b in bounds))-4),
                    max(0, math.floor(min(b[1] for b in bounds))-4),
                    min(im.width, math.ceil(max(b[2] for b in bounds))+5),
                    min(im.height, math.ceil(max(b[3] for b in bounds))+5)]
            if crop[0] >= crop[2] or crop[1] >= crop[3]:
                return {**result, 'reason': 'empty_source_roi'}
            native = im.convert('RGB').crop(crop)
        # Bound CPU/memory while retaining explicit original-pixel units.
        scale = min(1., 1600/max(native.size))
        size = tuple(max(2, round(v*scale)) for v in native.size)
        native = native.resize(size, Image.Resampling.BILINEAR)
        mask = np.asarray(project_mask(prediction, crop, size)) > 0
    except (OSError, ValueError) as exc:
        return {**result, 'reason': 'sam_or_source_unavailable', 'error': str(exc)}
    yy, xx = np.nonzero(mask)
    if len(xx) < 16:
        return {**result, 'reason': 'insufficient_mask_pixels'}
    density = len(xx)/max(1, (xx.max()-xx.min()+1)*(yy.max()-yy.min()+1))
    # Filled shape: use its exterior, ignoring text holes. Outline/dashes: keep
    # actual foreground fragments, without creating an enclosing polygon.
    support = mask.astype('uint8')
    if density > .4:
        contours, _ = cv2.findContours(support, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        support = np.zeros_like(support)
        cv2.drawContours(support, contours, -1, 1, 1)
    rgb = np.asarray(native)
    edges = np.maximum.reduce([cv2.Canny(rgb[:, :, k].copy(), 40, 120) for k in range(3)]) > 0
    if not edges.any():
        return {**result, 'reason': 'no_source_edges'}
    # Uncertainty is based on the model's *real* 1008 network frame, not on
    # the dimensions of an upsampled PNG. It also covers stroke half-width.
    frame = prediction['frame']['box']
    resolution = getattr(evidence, 'processor_resolution', 1008)
    tolerance = max(1.5, 1.5*max(frame[2]-frame[0], frame[3]-frame[1])/resolution)
    result.update(source_roi_half_open=crop, sam_bbox_original=prediction['bbox'],
                  sam_mask_path=str(prediction['mask_path']), sam_network_resolution=resolution,
                  before_bbox_original=before['bbox_2d'], after_bbox_original=after['bbox_2d'])
    step = np.array([(crop[2]-crop[0])/size[0], (crop[3]-crop[1])/size[1]])
    to_mask = cv2.distanceTransform((support == 0).astype('uint8'), cv2.DIST_L2, 5)
    edges &= to_mask*max(step) <= tolerance
    y, x = np.nonzero(edges)
    anchors = np.column_stack([x, y])*step + np.array(crop[:2]) + .5*step - .5
    old_points, new_points = contour_samples(before), contour_samples(after)
    if len(anchors) < 16 or min(len(old_points), len(new_points)) < 16:
        return {**result, 'reason': 'insufficient_source_supported_samples'}
    old_tree, new_tree = cKDTree(old_points), cKDTree(new_points)
    distance, indices = old_tree.query(anchors)
    # Trust only spans where the previous contour, source and SAM agree.
    keep = distance <= 2*tolerance
    anchors, distance, indices = anchors[keep], distance[keep], indices[keep]
    # Adjacent PPT frames can overlap geometrically while using different
    # stroke colors. A mask mixing these contours must not turn the neighbor's
    # colored side into evidence against a valid correction of this target.
    # Nearby channel minima suppress white antialiasing at a thin stroke edge.
    ink = cv2.erode(rgb, np.ones((3, 3), 'uint8'))[y[keep], x[keep]].astype(float)
    if len(ink):
        bins = (ink//32).astype(int)
        keys, counts = np.unique(bins, axis=0, return_counts=True)
        dominant = keys[np.argmax(counts)]*32+16
        coherent = np.max(np.abs(ink-dominant), axis=1) <= 48
        result['coherent_stroke_fraction'] = float(np.mean(coherent))
        if np.mean(coherent) < .5:
            return {**result, 'reason': 'ambiguous_source_stroke_ownership'}
        anchors, distance, indices = anchors[coherent], distance[coherent], indices[coherent]
    sectors = np.minimum(15, 16*indices//len(old_points))
    count = len(set(sectors.tolist()))
    if len(anchors) < 16 or count < 8:
        return {**result, 'reason': 'insufficient_distributed_agreement', 'supported_sectors': count}
    if len(anchors) > 4000:
        selection = np.linspace(0, len(anchors)-1, 4000).astype(int)
        anchors, distance = anchors[selection], distance[selection]
    changed = new_tree.query(anchors)[0]
    p90_before, p90_after = [float(np.percentile(v, 90)) for v in (distance, changed)]
    lost = float(np.mean(changed > 2*tolerance))
    result.update(decision='allow', reason='no_clear_supported_contour_regression',
                  tolerance_original_px=tolerance, mask_representation='exterior' if density > .4 else 'foreground_fragments',
                  supported_sectors=count, source_sam_anchor_count=len(anchors),
                  before_p90_px=p90_before, after_p90_px=p90_after, lost_supported_fraction=lost)
    # This is a regression veto, not a general visual-accuracy certificate.
    if p90_after > p90_before + 2*tolerance and lost >= .1:
        result.update(decision='rollback', reason='source_and_sam_supported_spans_lost')
    return result
