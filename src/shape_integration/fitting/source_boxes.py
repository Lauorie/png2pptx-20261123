"""Register repeated primitives to independent, isolated SOURCE fill components.

Components nominate identity only. A one-to-one, order-preserving assignment
and four actual source stroke sides are required before any geometry changes.
No detector or segmentation bbox becomes the exported geometry.
"""
import copy
import math

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def register_compact(local, job, clean):
    """Measure small repeated colored rectangles in separate source columns.

    This handles compound scenes whose other primitive family defeats a global
    count. Components nominate a column; four source stroke sides supply every
    returned contour. No detector box, missing-member guess or grid is exported.
    """
    from .frame_edges import measure
    from .adapter import bounds,union_bounds
    from .paths import box_commands
    f=job['frame'];crop=f['source_crop_half_open'];size=f['actual_crop_input_size_px']
    scale=max(size[i]/(crop[i+2]-crop[i]) for i in (0,1));rgb=np.asarray(clean.convert('RGB'))
    eligible=[p for p in local['parts'] if p.get('status')=='ok' and p['kind']=='rectangle'
              and abs(p['parameters']['angle_radians'])<math.radians(.5)
              and 8*scale<=min(np.asarray(p['bbox_2d'])[2:]-np.asarray(p['bbox_2d'])[:2])<24*scale]
    families=[]
    for part in sorted(eligible,key=lambda p:np.prod(np.asarray(p['bbox_2d'])[2:]-np.asarray(p['bbox_2d'])[:2])):
        b=np.asarray(part['bbox_2d']);span=b[2:]-b[:2]
        for family in families:
            if all(np.max(np.maximum(span/(np.asarray(p['bbox_2d'])[2:]-np.asarray(p['bbox_2d'])[:2]),
                         (np.asarray(p['bbox_2d'])[2:]-np.asarray(p['bbox_2d'])[:2])/span))<1.5 for p in family):
                family.append(part);break
        else:families.append([part])
    audits=[]
    for family in families:
        typical=np.median([np.asarray(p['bbox_2d'])[2:]-np.asarray(p['bbox_2d'])[:2] for p in family],axis=0)
        groups=[]
        for part in sorted(family,key=lambda p:p['bbox_2d'][0]):
            if groups and part['bbox_2d'][0]-groups[-1][-1]['bbox_2d'][0]<2*typical[0]:groups[-1].append(part)
            else:groups.append([part])
        for group in groups:
            if len(group)<3:continue
            boxes=np.asarray([p['bbox_2d'] for p in group]);lo=np.maximum(0,np.floor(boxes[:,:2].min(0)-2*typical)).astype(int)
            hi=np.minimum(clean.size,np.ceil(boxes[:,2:].max(0)+2*typical)).astype(int)
            if np.any(hi<=lo):continue
            tile=rgb[lo[1]:hi[1],lo[0]:hi[0]];pixels=tile.reshape(-1,3)
            colors,counts=np.unique(pixels//8,axis=0,return_counts=True);sets=[]
            for index in np.argsort(counts)[-20:][::-1]:
                pigment=np.median(pixels[(pixels//8==colors[index]).all(1)],axis=0)
                if np.ptp(pigment)<25:continue
                _,_,stats,_=cv2.connectedComponentsWithStats((np.linalg.norm(tile.astype(float)-pigment,axis=2)<16).astype('uint8'),8)
                comps=[np.array([x,y,x+w,y+h],float)+np.tile(lo,2)-.5 for x,y,w,h,area in stats[1:]
                       if .5*typical[0]<=w<=1.5*typical[0] and .5*typical[1]<=h<=1.5*typical[1] and area>=.6*w*h]
                if len(comps)<3:continue
                if sets and any(len(old[0])==len(comps) and
                     np.max(np.abs(np.sort(np.array(comps),axis=0)-np.sort(np.array(old[0]),axis=0)))<2*scale for old in sets):continue
                sets.append((comps,pigment))
            if len(sets)!=1:
                if sets:audits.append({'applied':False,'reason':'compact_source_pigment_ambiguous','parts':[p['id'] for p in group]})
                continue
            comps,pigment=sets[0];source=np.array(comps);targets=(source[:,:2]+source[:,2:])/2;centers=(boxes[:,:2]+boxes[:,2:])/2
            cost=np.linalg.norm((centers[:,None]-targets[None])/typical,axis=2);ii,jj=linear_sum_assignment(cost)
            if len(ii)!=len(group) or len(comps)!=len(group):
                # A count conflict needs the existing geometry observation;
                # measuring existing identities still may fix their position.
                audits.append({'applied':False,'reason':'compact_source_count_conflict','parts':[p['id'] for p in group],
                    'observed_parts':len(group),'source_components':len(comps),'source_boxes':[b.tolist() for b in comps]})
            if len(ii)!=len(group):continue
            assigned=dict(zip(ii,jj));ordered=np.argsort(centers[:,1])
            if np.any(np.diff(targets[[assigned[i] for i in ordered],1])<=0):continue
            # Source gaps define identity uncertainty; the correction may not
            # jump over a different source occurrence along the column.
            gaps=np.diff(np.sort(targets[:,1]));reach=max(typical[1],float(gaps.min())/2) if len(gaps) else typical[1]
            if any(abs(centers[i,0]-targets[j,0])>2*typical[0] or abs(centers[i,1]-targets[j,1])>reach for i,j in assigned.items()):continue
            for i,j in assigned.items():
                part=group[i]
                b,audit=measure(part['parameters'],clean,source[j],source_scale=scale,
                    source_identity=True,fill_pigment=pigment,identity_displacement_px=2*max(typical))
                audit.update(part_id=part['id'],identity='isolated_compact_source_column',source_component=source[j].tolist(),source_pigment=pigment.tolist())
                audits.append(audit)
                if b is None:continue
                p=part['parameters'];p.update(center=[(b[0]+b[2])/2,(b[1]+b[3])/2],width=b[2]-b[0],height=b[3]-b[1],angle_radians=0.)
                part['commands']=box_commands([*p['center'],p['width'],p['height'],0.]);part['bbox_2d']=bounds(part['commands'])
                part['diagnostics']['source_box_registration']=copy.deepcopy(audit)
    if any(r.get('applied') for r in audits):local['bbox_2d']=union_bounds(local['parts'])
    return audits


def register(local, job, clean, *, owned_regions=()):
    from .adapter import bounds, union_bounds
    from .frame_edges import measure
    from .paths import box_commands
    parts = [p for p in local['parts'] if p.get('status') == 'ok' and p['kind'] in ('rectangle', 'rounded_rectangle')
             and abs(p['parameters']['angle_radians']) <= math.radians(.5)]
    from .source_primitives import register as register_primitives
    compact_audit=register_primitives(local,job,clean,owned_regions=owned_regions)
    if len(parts) < 3:
        return compact_audit
    f = job['frame']; crop = f['source_crop_half_open']; size = f['actual_crop_input_size_px']
    scale = max(size[i] / (crop[i+2] - crop[i]) for i in (0, 1))
    boxes = np.asarray([p['bbox_2d'] for p in parts], float)
    spans = boxes[:, 2:] - boxes[:, :2]; typical = np.median(spans, axis=0)
    lo = np.maximum(0, np.floor(boxes[:, :2].min(0) - [.2 * typical[0], typical[1]])).astype(int)
    hi = np.minimum(clean.size, np.ceil(boxes[:, 2:].max(0) + [.2 * typical[0], typical[1]])).astype(int)
    if np.any(hi <= lo):
        return compact_audit
    rgb = np.asarray(clean.convert('RGB'))[lo[1]:hi[1], lo[0]:hi[0]]
    pixels = rgb.reshape(-1, 3)
    quantized = (pixels // 12).astype(np.int32)
    bins = (quantized[:, 0] * 22 + quantized[:, 1]) * 22 + quantized[:, 2]
    counts = np.bincount(bins, minlength=22**3)
    packed = pixels.astype(np.uint32)
    packed = (packed[:, 0] << 16) | (packed[:, 1] << 8) | packed[:, 2]
    rgb_float = rgb.astype(np.float32)
    components = []
    for index in np.argsort(counts)[::-1][:20]:
        if counts[index] == 0:
            break
        values, masses = np.unique(packed[bins == index], return_counts=True)
        value = int(values[int(masses.argmax())])
        pigment = np.asarray([value >> 16, (value >> 8) & 255, value & 255], np.float32)
        if pigment.mean() < 100 or pigment.min() > 245 or counts[index] < max(30, .08 * np.prod(typical)):
            continue
        mask = (((rgb_float - pigment)**2).sum(axis=2) < 12**2).astype('uint8')
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for x, y, w, h, area in stats[1:]:
            if (not .4 * typical[0] <= w <= 2 * typical[0] or not .3 * typical[1] <= h <= 2 * typical[1]
                    or area < .6 * w * h):
                continue
            b = np.array([x, y, x+w, y+h], float) + np.tile(lo, 2) - .5
            if any(np.max(np.abs(b - old['box'])) < 2 * scale for old in components):
                continue
            components.append({'box': b, 'pigment': pigment.tolist(), 'area': int(area)})
    if len(components) != len(parts):
        return compact_audit+[{'applied': False, 'reason': 'source_primitive_count_ambiguous',
                 'observed_parts': len(parts), 'source_components': len(components)}]
    source = np.asarray([c['box'] for c in components]); centers = (boxes[:, :2]+boxes[:, 2:])/2
    targets = (source[:, :2]+source[:, 2:])/2
    costs = (((centers[:, None]-targets[None])/typical)**2).sum(2)
    ii, jj = linear_sum_assignment(costs); assignment = dict(zip(ii, jj))
    for i, j in assignment.items():
        displacement = abs(centers[i]-targets[j])
        ratio = (source[j, 2:]-source[j, :2])/spans[i]
        if displacement[0] > .3 * spans[i, 0] or displacement[1] > 1.25 * spans[i, 1] or np.any(ratio < .65) or np.any(ratio > 1.45):
            return compact_audit+[{'applied': False, 'reason': 'source_assignment_not_local', 'part_id': parts[i]['id'],
                     'displacement_px': displacement.tolist(), 'size_ratio': ratio.tolist()}]
    for i in range(len(parts)):
        for j in range(i):
            if abs(centers[i, 0]-centers[j, 0]) < .2 * typical[0] and abs(centers[i, 1]-centers[j, 1]) > .5 * typical[1]:
                if (centers[i, 1]-centers[j, 1]) * (targets[assignment[i], 1]-targets[assignment[j], 1]) <= 0:
                    return compact_audit+[{'applied': False, 'reason': 'source_assignment_changes_order'}]
    audits = []
    for i, part in enumerate(parts):
        component = components[assignment[i]]
        b, audit = measure(part['parameters'], clean, component['box'], source_scale=scale,
                           translation_hint=True, source_identity=True, fill_pigment=component['pigment'])
        audit.update(part_id=part['id'], identity='one_to_one_source_components',
                     source_component=component['box'].tolist(), source_pigment=component['pigment'])
        audits.append(audit)
        if b is None:
            continue
        p = part['parameters']; width, height = b[2]-b[0], b[3]-b[1]
        radius = min(p.get('radius', 0), min(width, height)/2)
        p.update(center=[(b[0]+b[2])/2, (b[1]+b[3])/2], width=width, height=height, angle_radians=0.)
        if part['kind'] == 'rounded_rectangle':
            p.update(radius=radius, pptx_adjust=radius/min(width, height))
        part['commands'] = box_commands([*p['center'], width, height, 0.], radius)
        part['bbox_2d'] = bounds(part['commands'])
        part['diagnostics']['source_box_registration'] = copy.deepcopy(audit)
    local['bbox_2d'] = union_bounds(local['parts'])
    return compact_audit+audits
