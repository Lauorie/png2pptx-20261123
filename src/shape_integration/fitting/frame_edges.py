"""Source stroke measurements inside the existing rectangular numerical fit.

The observed contour anchors the search window. Source support is required on all sides;
an ambiguous parallel side retains its observation while other sides can fit.
Masks and detection boxes never become geometry.
"""
import math

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter1d


def measure(parameters, clean, reference, *, source_scale=1., translation_hint=False,
            source_identity=False, fill_pigment=None, side_limits=None,identity_displacement_px=None,allow_aligned=False):
    audit = {'applied': False, 'reference_local': list(reference)}
    if abs(parameters['angle_radians']) > math.radians(.5):
        return None, {**audit, 'reason': 'rotated_frame'}
    rgb = np.asarray(clean.convert('RGB'), dtype=float)
    gray = rgb.mean(axis=2)
    b = np.asarray(reference, float)
    span = b[2:] - b[:2]
    if min(span) < (8 if source_identity else 24) * source_scale:
        return None, {**audit, 'reason': 'small_frame'}
    search = max(8 * source_scale, .03 * max(span))
    band = np.arange(-search, search + .2 * source_scale, .25 * source_scale)
    old = [parameters['center'][i] + sign * parameters[dim] / 2
           for sign in (-1, 1) for i, dim in enumerate(('width', 'height'))]
    measurements = []; pigment = None
    for side in range(4):
        axis = side % 2
        tangent = 1 - axis
        along = np.linspace(b[tangent] + .12 * span[tangent],
                            b[tangent + 2] - .12 * span[tangent], 160)
        normal = b[side] + band
        if side_limits is not None:
            lo,hi=side_limits[side]
            normal=normal[(normal>lo)&(normal<hi)]
            if not len(normal):
                return None,{**audit,'reason':'search_belongs_to_another_observed_frame','side':side}
        xy = np.zeros((len(along), len(normal), 2))
        xy[:, :, axis] = normal[None, :]
        xy[:, :, tangent] = along[:, None]
        def pixels(offset):
            q = xy.copy(); q[:, :, axis] += offset * source_scale
            return map_coordinates(gray, [q[:, :, 1], q[:, :, 0]], order=1,
                                   mode='constant', cval=255.)
        core = pixels(0)
        # A stroke must be darker than BOTH sides, unlike an arbitrary fill step.
        contrast = np.maximum(0, np.minimum(pixels(-3), pixels(3)) - core)
        source_colours = np.stack([map_coordinates(rgb[:, :, c], [xy[:, :, 1], xy[:, :, 0]],
                             order=1, mode='constant', cval=255.) for c in range(3)], axis=-1)
        visible = np.ones(len(along), bool)
        if source_identity and fill_pigment is not None:
            # Independent source-fill identity also tells us where an opaque
            # foreground symbol hides this frame. Unknown spans do not become
            # false missing sides, nor do they supply a fabricated edge.
            inside = xy[:, int(np.argmin(abs(normal-b[side])))].copy()
            inside[:, axis] = b[side] + (1 if side < 2 else -1) * 4 * source_scale
            colours = np.stack([map_coordinates(rgb[:, :, c], [inside[:, 1], inside[:, 0]],
                               order=1, mode='constant', cval=255.) for c in range(3)], axis=-1)
            visible = np.linalg.norm(colours - np.asarray(fill_pigment), axis=-1) < 20
            if visible.mean() < .3:
                return None, {**audit, 'reason': 'source_side_mostly_occluded', 'sides': measurements}
        if pigment is not None:
            contrast *= np.linalg.norm(source_colours - pigment, axis=-1) < 85
        coverage = (contrast[visible] > 12).mean(axis=0)
        strength = np.minimum(contrast[visible], 100).mean(axis=0)
        profile = gaussian_filter1d(strength, .45)
        best = int(np.argmax(profile))
        far = np.abs(normal - normal[best]) > 4 * source_scale
        second = float(profile[far].max()) if far.any() else 0.
        anchored = False
        if second > .85 * profile[best]:
            # A nearby parallel frame must not veto corrections of every
            # other side. Resolve the tie only when the existing contour and
            # CLEAN independently agree within source-pixel uncertainty.
            near = np.flatnonzero(np.abs(normal-old[side]) <= 1.5 * source_scale)
            if len(near):
                supported = int(near[np.argmax(profile[near])])
                if profile[supported] >= .85 * profile[best]:
                    best = supported; anchored = True
        good = contrast[:, best] > 12
        if pigment is None and (good & visible).any():
            pigment = np.median(source_colours[good & visible, best], axis=0)
        groups = [float(g[v].mean()) for g, v in zip(np.array_split(good, 4), np.array_split(visible, 4)) if v.sum() >= 5]
        changes = np.flatnonzero(np.diff(np.r_[True, good | ~visible, True].astype(int))).reshape(-1, 2)
        max_gap = max((hi - lo for lo, hi in changes), default=0) / len(good)
        row = {'side': side, 'position': float(normal[best]),
               'coverage': float(coverage[best]), 'quarter_coverage': groups,
               'longest_gap_fraction': max_gap, 'strength': float(profile[best]),
               'source_visible_fill_fraction': float(visible.mean()),
               'competing_strength': second, 'source_pigment': pigment.tolist() if pigment is not None else None}
        if anchored:row['ambiguity_resolved_by_source_supported_prior'] = float(old[side])
        measurements.append(row)
        if (coverage[best] < .55 or len(groups) < 2 or min(groups) < .25 or max_gap > .13
                or profile[best] < 10
                or best in (0, len(normal) - 1)):
            return None, {**audit, 'reason': 'weak_ambiguous_or_open_side', 'sides': measurements}
        if second > .85 * profile[best] and not anchored:
            row['retained_observed_position'] = float(old[side])
    result = [row.get('retained_observed_position', row['position']) for row in measurements]
    if result[2] <= result[0] or result[3] <= result[1]:
        return None, {**audit, 'reason': 'invalid_extent', 'sides': measurements}
    delta = max(abs(a - z) for a, z in zip(old, result))
    # A matched local proposal can locate the search around a translated
    # primitive. It never licenses a large resize or supplies the final edges.
    limit=1.25 if source_identity else .5 if translation_hint else .25
    resized=max(abs((result[i+2]-result[i])-(old[i+2]-old[i])) for i in (0,1))
    size_changed = (max(abs((result[i+2]-result[i])/(old[i+2]-old[i])-1) for i in (0,1)) > .35
                    if source_identity else translation_hint and resized > .25 * min(span))
    allowed=identity_displacement_px if source_identity and identity_displacement_px is not None else limit*min(span)
    if delta > allowed or size_changed or delta < .35 * source_scale and not allow_aligned:
        return None, {**audit, 'reason': 'outside_bounded_repair_or_already_aligned', 'sides': measurements}
    reason=('source_stroke_centerlines_with_preserved_ambiguous_sides'
            if any('retained_observed_position' in row for row in measurements) else 'four_source_stroke_centerlines')
    return result, {**audit, 'applied': True, 'reason': reason,
                    'before': old, 'after': result, 'sides': measurements}


def measure_fill(parameters, clean, reference, *, source_scale=1.):
    """Measure an isolated flat coloured rectangle, never a detector box."""
    import cv2
    rgb=np.asarray(clean.convert('RGB'));b=np.asarray(reference,float);span=b[2:]-b[:2]
    audit={'applied':False,'reason':'no_unambiguous_flat_fill'}
    if min(span)<8*source_scale or abs(parameters['angle_radians'])>math.radians(.5):return None,audit
    lo=np.maximum(0,np.floor(b[:2]+.2*span)).astype(int)
    hi=np.minimum(rgb.shape[1::-1],np.ceil(b[2:]-.2*span)).astype(int)
    pixels=rgb[lo[1]:hi[1],lo[0]:hi[0]].reshape(-1,3)
    if not len(pixels):return None,audit
    bins,counts=np.unique(pixels//8,axis=0,return_counts=True);index=int(counts.argmax())
    if counts[index]<.35*len(pixels):return None,audit
    pigment=np.median(pixels[np.all(pixels//8==bins[index],axis=1)],axis=0)
    if np.ptp(pigment)<20:return None,audit
    # Fill is evidence for the observed contour only within the same local
    # search band used for strokes. A central colour patch can belong to a
    # different, contained shape; it cannot resize its enclosing outline.
    margin=max(8*source_scale,.03*max(span))
    lo=np.maximum(0,np.floor(b[:2]-margin)).astype(int)
    hi=np.minimum(rgb.shape[1::-1],np.ceil(b[2:]+margin)).astype(int)
    mask=(np.linalg.norm(rgb[lo[1]:hi[1],lo[0]:hi[0]].astype(float)-pigment,axis=2)<18).astype('uint8')
    _,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    if len(stats)<=1:return None,audit
    x,y,w,h,area=stats[1+int(stats[1:,4].argmax())]
    if area<.65*w*h or x==0 or y==0 or x+w>=mask.shape[1] or y+h>=mask.shape[0]:return None,audit
    result=np.array([x,y,x+w,y+h],float)+np.tile(lo,2)-.5
    if np.max(np.abs(result-b))>margin or np.max(np.abs(result-b))<.35*source_scale:return None,audit
    # All four central side spans must border the same fill. This excludes
    # tapers, merged neighbours and nonrectangular colour patches.
    inside=mask[y:y+h,x:x+w]
    sides=[inside[0,int(.2*w):int(.8*w)],inside[-1,int(.2*w):int(.8*w)],
           inside[int(.2*h):int(.8*h),0],inside[int(.2*h):int(.8*h),-1]]
    if any(not len(s) or s.mean()<.85 for s in sides):return None,audit
    return result.tolist(),{'applied':True,'reason':'four_sides_of_isolated_source_fill',
                           'source_pigment':pigment.tolist(),'before':b.tolist(),'after':result.tolist()}


def competing_frame_sides(local):
    """Keep different observed faces from snapping to one neighbour's edges.

    Only partially overlapping frames compete. Adjacent grid cells and inner
    details of a containing frame keep their existing numerical fitting path.
    Observed positions are frozen before any part is refined.
    """
    points={p['id']:p['xy'] for p in local['observation']['points']};boxes={}
    for part in local['parts']:
        if (part['status']!='ok' or part['kind'] not in ('rectangle','rounded_rectangle')
                or abs(part.get('parameters',{}).get('angle_radians',0))>math.radians(.5)):
            continue
        observed=np.asarray([points[k] for k in part.get('observed_point_ids',[])],float)
        if len(observed)>=4:boxes[part['id']]=np.r_[observed.min(0),observed.max(0)]
    limits={}
    for ident,a in boxes.items():
        peers=[];lo=np.full(4,-np.inf);hi=np.full(4,np.inf)
        for other,b in boxes.items():
            if other==ident or np.any(np.minimum(a[2:],b[2:])<=np.maximum(a[:2],b[:2])):continue
            if (np.all(a[:2]<=b[:2]) and np.all(a[2:]>=b[2:]) or
                    np.all(b[:2]<=a[:2]) and np.all(b[2:]>=a[2:])):continue
            peers.append(other);mid=(a+b)/2
            lo=np.where(a>b,np.maximum(lo,mid),lo)
            hi=np.where(a<b,np.minimum(hi,mid),hi)
        if peers:limits[ident]={'sides':list(zip(lo.tolist(),hi.tolist())),'peers':peers}
    return limits


def refine(local, job, clean, evidence):
    from .adapter import bounds, union_bounds
    from .paths import box_commands
    from .primitives import rr_signed_distance
    f = job['frame']; crop = f['source_crop_half_open']; size = f['actual_crop_input_size_px']
    scale = max(size[i] / (crop[i + 2] - crop[i]) for i in (0, 1))
    protected = set((job.get('revision_context') or {}).get('preserved_ids', []))
    audits = []
    nested_peers = [row['id'] for row in job.get('input_context', {}).get('registered_peer_shapes', [])
                    if row.get('relation_hint') == 'inside_target']
    shared={pid for part in local['parts'] for pid in part.get('observed_point_ids',[])
            if sum(pid in other.get('observed_point_ids',[]) for other in local['parts'])>1}
    ownership=competing_frame_sides(local)
    for part in local['parts']:
        if (part['kind'] not in ('rectangle', 'rounded_rectangle') or part['status'] != 'ok'
                or part['id'] in protected or shared.intersection(part.get('observed_point_ids',[]))):
            continue
        if nested_peers:
            # This local numerical search has no pixel ownership for separate
            # enclosed objects. Their stronger edges can erase the outer gap.
            # Keep the observation for the existing source critic to inspect.
            audits.append({'applied': False, 'part_id': part['id'],
                           'reason': 'enclosed_peer_edges_have_no_owner_specific_evidence',
                           'enclosed_peers': nested_peers})
            continue
        # A segmentation may cover only a face below a header divider. Its
        # extent does not identify the contour the model actually observed.
        reference=part['bbox_2d']
        owned=ownership.get(part['id'])
        box, audit = measure(part['parameters'], clean, reference, source_scale=scale,
                             side_limits=owned['sides'] if owned else None)
        if box is None and not owned:
            box,fill_audit=measure_fill(part['parameters'],clean,part['bbox_2d'],source_scale=scale)
            if box is not None:audit=fill_audit
        if owned:
            audit['competing_observed_frames']=owned['peers']
            audit['shared_fill_not_isolated']=True
        audit.update(part_id=part['id']); audits.append(audit)
        if box is None:
            continue
        p = part['parameters']; w, h = box[2] - box[0], box[3] - box[1]
        radius = min(p.get('radius', 0), min(w, h) / 2)
        p.update(center=[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                 width=w, height=h, angle_radians=0.)
        if part['kind'] == 'rounded_rectangle':
            p.update(radius=radius, pptx_adjust=radius / min(w, h))
        part['commands'] = box_commands([*p['center'], w, h, 0.], radius)
        part['bbox_2d'] = bounds(part['commands'])
        part['diagnostics']['frame_edge_refinement'] = audit
        table = {q['id']: q['xy'] for q in local['observation']['points']}
        points = np.array([table[k] for k in part['observed_point_ids']])
        residual = np.abs(rr_signed_distance(points, [*p['center'], w, h, 0., radius / min(w, h)]))
        part['diagnostics'].update(rms_px=float(np.sqrt(np.mean(residual ** 2))),
                                   p90_px=float(np.percentile(residual, 90)), max_px=float(residual.max()))
    local['bbox_2d'] = union_bounds(local['parts'])
    return audits


def refine_outline(local, job, clean, evidence=None):
    """Fit source lines of an already observed polygonal callout, keeping topology.

This cannot turn a rectangle into a callout. Two adjoining oblique pointer
sides and four axis-aligned body sides must already have been observed.
"""
    from .adapter import bounds, union_bounds
    f=job['frame'];crop=f['source_crop_half_open'];size=f['actual_crop_input_size_px']
    scale=max(size[i]/(crop[i+2]-crop[i]) for i in (0,1))
    gray=np.asarray(clean.convert('RGB'),float).mean(axis=2);audits=[]
    reference=None
    if evidence is not None and hasattr(evidence,'geometry_support'):
        from ..sam_evidence import iou,local_box
        group=evidence.geometry_support(job['candidate'])
        if group and iou(group['representative']['bbox'],job['candidate']['bbox_2d'])>=.65:
            reference=local_box(group['representative']['bbox'],crop,size)
    protected=set((job.get('revision_context') or {}).get('preserved_ids',[]))
    shared={pid for part in local['parts'] for pid in part.get('observed_point_ids',[])
            if sum(pid in other.get('observed_point_ids',[]) for other in local['parts'])>1}
    for part in local['parts']:
        cs=part.get('commands',[])
        if (part['id'] in protected or shared.intersection(part.get('observed_point_ids',[]))
                or part['kind']!='path' or part['status']!='ok'
                or not cs or cs[-1][0]!='Z' or any(c[0] not in ('M','L','Z') for c in cs)):
            continue
        vertices=np.asarray([c[1:3] for c in cs if c[0] in ('M','L')],float)
        if np.linalg.norm(vertices[0]-vertices[-1])<1e-5:vertices=vertices[:-1]
        if not 6<=len(vertices)<=12:continue
        delta=np.roll(vertices,-1,axis=0)-vertices
        angles=np.arctan2(delta[:,1],delta[:,0]);axis=np.abs(np.sin(2*angles))<.015
        oblique=np.flatnonzero(~axis)
        if len(oblique)!=2 or axis.sum()<4 or (oblique[1]-oblique[0]) not in (1,len(vertices)-1):continue
        lines=[];measurements=[];failed=False
        for a,b,theta,is_axis in zip(vertices,np.roll(vertices,-1,axis=0),angles,axis):
            length=np.linalg.norm(b-a)
            if length<14*scale:failed=True;break
            midpoint=(a+b)/2;along=np.linspace(-.37*length,.37*length,120)
            if is_axis and reference is not None:
                normal_axis=0 if abs(b[0]-a[0])<abs(b[1]-a[1]) else 1
                choices=[reference[normal_axis],reference[normal_axis+2]]
                target=min(choices,key=lambda v:abs(v-midpoint[normal_axis]))
                if abs(target-midpoint[normal_axis])<64*scale:midpoint[normal_axis]=target
            offsets=np.arange(-24*scale,24.01*scale,.5*scale)
            theta_grid=[round(theta/(math.pi/2))*(math.pi/2)] if is_axis else theta+np.radians(np.arange(-20,20.01,.5))
            best=None
            for angle in theta_grid:
                direction=np.array([math.cos(angle),math.sin(angle)]);normal=np.array([-direction[1],direction[0]])
                xy=midpoint+along[:,None,None]*direction+offsets[None,:,None]*normal
                def sample(d):
                    q=xy+d*scale*normal
                    return map_coordinates(gray,[q[:,:,1],q[:,:,0]],order=1,mode='constant',cval=255.)
                contrast=np.maximum(0,np.minimum(sample(-3),sample(3))-sample(0))
                values=np.minimum(contrast,100).mean(axis=0)
                coverage_by_offset=(contrast>12).mean(axis=0)
                quarters_by_offset=np.stack([chunk.mean(axis=0) for chunk in np.array_split(contrast>12,4)])
                eligible=(coverage_by_offset>=.25)
                if is_axis:
                    eligible &= ((quarters_by_offset>=.12).sum(axis=0)>=2)&(quarters_by_offset[0]>=.12)&(quarters_by_offset[-1]>=.12)
                else:eligible &= (quarters_by_offset>=.12).all(axis=0)
                if not eligible.any():continue
                k=int(np.where(eligible,values,-np.inf).argmax());good=contrast[:,k]>12
                coverage=float(good.mean());quarters=[float(x.mean()) for x in np.array_split(good,4)]
                distributed=(sum(q>=.12 for q in quarters)>=2 and quarters[0]>=.12 and quarters[-1]>=.12) if is_axis else min(quarters)>=.12
                if coverage<.25 or not distributed:continue
                # Penalise negligible score differences, preserving the observed direction.
                score=float(values[k])-.025*abs(offsets[k]/scale)-.025*abs(math.degrees(angle-theta))
                if best is None or score>best[0]:best=(score,normal,midpoint+offsets[k]*normal,coverage,quarters,float(angle))
            if best is None or best[0]<12:
                measurements.append({'failed_edge':len(lines),'a':a.tolist(),'b':b.tolist(),
                                     'best_strength':None if best is None else best[0]})
                failed=True;break
            score,normal,point,coverage,quarters,angle=best
            lines.append((normal,float(normal@point)))
            measurements.append({'coverage':coverage,'quarter_coverage':quarters,'angle_radians':angle,'strength':score})
        audit={'part_id':part['id'],'applied':False,'sides':measurements}
        if failed:
            audits.append({**audit,'reason':'insufficient_distributed_source_support'});continue
        fitted=[]
        for i in range(len(lines)):
            normals=np.asarray([lines[i-1][0],lines[i][0]])
            if abs(np.linalg.det(normals))<.15:failed=True;break
            fitted.append(np.linalg.solve(normals,[lines[i-1][1],lines[i][1]]))
        if failed:
            audits.append({**audit,'reason':'unstable_intersection'});continue
        fitted=np.asarray(fitted)
        from shapely.geometry import Polygon
        poly=Polygon(fitted)
        displacement=np.linalg.norm(fitted-vertices,axis=1)
        bound=max(32*scale,min(64*scale,.16*np.ptp(vertices,axis=0).min()))
        if (not poly.is_valid or poly.area<=0 or displacement.max()>bound
                or np.any(fitted<0) or np.any(fitted>np.asarray(size)-1)):
            audits.append({**audit,'reason':'bounded_topology_guard'});continue
        if displacement.max()<.35*scale:continue
        part['commands']=[['M',*fitted[0]],*[['L',*p] for p in fitted[1:]],['Z']]
        part['bbox_2d']=bounds(part['commands'])
        audit.update(applied=True,reason='same_topology_source_line_intersections',
                     before_vertices=vertices.tolist(),after_vertices=fitted.tolist(),
                     original_observation_preserved=True)
        part['diagnostics']['outline_edge_refinement']=audit;audits.append(audit)
    local['bbox_2d']=union_bounds(local['parts'])
    return audits
