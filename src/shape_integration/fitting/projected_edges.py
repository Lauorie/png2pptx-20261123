"""Conservative RGB direction fit for one three-face affine solid.

The native rectangular face and every part/point ID remain fixed. Only its
shared extrusion vector may rotate onto a repeated, source-visible direction.
This runs inside numerical fitting; no new crop, model call or revision loop.
"""
import copy
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from .adapter import fit_part, union_bounds, shared_point_diagnostics
from .paths import command_strokes


def angular_distance(a,b):
    return abs((np.asarray(a)-b+90)%180-90)


def face_conflicts(local,job):
    """Flag folded face interiors; source review decides the correction."""
    if job.get('proposed_type',job.get('candidate',{}).get('kind'))!='projected_3d':return []
    if len(local.get('observation',{}).get('points',[]))!=7 or len(local['parts'])!=3:return []
    from shapely.geometry import Polygon
    polygons=[]
    for part in local['parts']:
        cs=part.get('commands',[])
        if part['status']!='ok' or not cs or cs[-1][0]!='Z' or any(c[0] not in ('M','L','Z') for c in cs):return []
        polygon=Polygon([c[1:3] for c in cs if c[0] in ('M','L')])
        if not polygon.is_valid or polygon.area<=0:return []
        polygons.append(polygon)
    conflicts=[]
    for i,a in enumerate(polygons):
        for j,b in enumerate(polygons[:i]):
            fraction=a.intersection(b).area/min(a.area,b.area)
            if fraction>.02:conflicts.append({'parts':[local['parts'][j]['id'],local['parts'][i]['id']],
                'interior_overlap_fraction_of_smaller_face':round(fraction,4)})
    return conflicts


def refine(local,job,clean):
    audit={'protocol':'source_projected_direction.1','applied':False}
    raw=local['observation'];parts=raw['parts']
    if (job['candidate'].get('kind')!='projected_3d' or job.get('revision_context') or
            local['fit_status']!='ok' or len(parts)!=3 or len(raw['points'])!=7):
        return {**audit,'reason':'not_one_complete_three_face_solid'}
    rectangles=[p for p in parts if p['kind']=='rectangle']
    if len(rectangles)!=1:return {**audit,'reason':'no_unique_rectangular_anchor_face'}
    table={p['id']:np.asarray(p['xy'],float) for p in raw['points']};adj={k:set() for k in table};owners={k:0 for k in table}
    for p in parts:
        if p['kind']=='rectangle':
            ids=p['point_ids'];edges=list(zip(ids,ids[1:]+ids[:1]))
        elif p['kind']=='path' and p['closed'] and len(p['segments'])==4 and all(s['kind']=='line' for s in p['segments']):
            edges=[s['point_ids'] for s in p['segments']]
        else:return {**audit,'reason':'not_three_quadrilateral_faces'}
        for a,b in edges:adj[a].add(b);adj[b].add(a)
        for k in {k for edge in edges for k in edge}:owners[k]+=1
    roots=[k for k,n in owners.items() if n==3]
    if len(roots)!=1:return {**audit,'reason':'ambiguous_shared_vertex'}
    root=roots[0];neighbors=sorted(adj[root]);origin=table[root]
    depth=[k for k in neighbors if k not in rectangles[0]['point_ids']]
    if len(neighbors)!=3 or len(depth)!=1:return {**audit,'reason':'ambiguous_extrusion_axis'}
    vectors={k:table[k]-origin for k in neighbors};vector=vectors[depth[0]];length=np.linalg.norm(vector)
    if length<8:return {**audit,'reason':'short_extrusion'}
    affected={depth[0]}
    for k in table.keys()-set(neighbors)-{root}:
        if any(np.linalg.norm(table[k]-(origin+vector+vectors[n]))<=max(2.,.03*length)
               for n in neighbors if n!=depth[0]):affected.add(k)
    if len(affected)!=3:return {**audit,'reason':'non_affine_face_topology'}
    rgb=np.asarray(clean.convert('RGB'));edge=np.maximum.reduce([cv2.Canny(rgb[:,:,i].copy(),60,140) for i in range(3)])
    scale=max(job['frame']['actual_crop_input_size_px'][i]/(job['frame']['source_crop_half_open'][i+2]-job['frame']['source_crop_half_open'][i]) for i in (0,1))
    detected=cv2.HoughLinesP(edge,1,np.pi/720,threshold=35,minLineLength=max(30,int(min(rgb.shape[:2])*.08)),maxLineGap=6)
    if detected is None:return {**audit,'reason':'no_source_line_family'}
    q=detected[:,0];delta=q[:,2:]-q[:,:2];weights=np.linalg.norm(delta,axis=1);theta=np.degrees(np.arctan2(delta[:,1],delta[:,0]))%180
    angle=float(np.degrees(np.arctan2(vector[1],vector[0]))%180);keep=angular_distance(theta,angle)<=30
    for k,v in vectors.items():
        if k!=depth[0]:keep &= angular_distance(theta,np.degrees(np.arctan2(v[1],v[0]))%180)>8
    if keep.sum()<3:return {**audit,'reason':'insufficient_independent_source_segments'}
    theta,weights=theta[keep],weights[keep]
    scores=np.array([weights[angular_distance(theta,a)<2].sum() for a in range(180)])
    dominant=int(scores.argmax());support=angular_distance(theta,dominant)<2
    if support.sum()<3 or scores[dominant]<.6*weights.sum() or not 4<angular_distance(angle,dominant)<28:
        return {**audit,'reason':'direction_supported_or_ambiguous'}
    distance=distance_transform_edt(edge==0)
    def score(fits):
        chunks=[]
        for part in fits:
            for stroke in command_strokes(part['commands']):
                for a,b in zip(stroke,stroke[1:]):
                    a,b=np.asarray(a),np.asarray(b);t=np.linspace(0,1,max(3,int(np.linalg.norm(b-a)/2)))
                    chunks.append(a[None]+t[:,None]*(b-a))
        xy=np.rint(np.concatenate(chunks)).astype(int)
        if np.any(xy<0) or np.any(xy[:,0]>=rgb.shape[1]) or np.any(xy[:,1]>=rgb.shape[0]):return float('inf')
        return float(np.minimum(distance[xy[:,1],xy[:,0]],8*scale).mean()/scale)
    before=score(local['parts'])
    if not np.isfinite(before):return {**audit,'reason':'fitted_contour_exits_source_crop'}
    best=before;chosen=None
    for fraction in (.5,1.):
        turn=((dominant-angle+90)%180-90)*fraction;rad=np.radians(turn)
        shift=np.array([[np.cos(rad),-np.sin(rad)],[np.sin(rad),np.cos(rad)]])@vector-vector
        adjusted={k:(v+shift if k in affected else v) for k,v in table.items()}
        fits=[{**p,**fit_part(p['observed_part'],adjusted)} for p in local['parts']]
        value=score(fits)
        if value<best:best=value;chosen=(fits,adjusted,turn)
    audit.update(source_direction_degrees=dominant,model_direction_degrees=angle,
                 source_segments=int(support.sum()),loss_before_source_px=before,loss_after_source_px=best)
    if chosen is None or before-best<.25 or best>.85*before:
        return {**audit,'reason':'no_clear_source_edge_improvement'}
    fits,adjusted,turn=chosen
    audit.update(applied=True,reason='shared_extrusion_aligned_to_source_edges',rotation_degrees=float(turn),
                 changed_point_ids=sorted(affected),original_observation_preserved=True)
    local['parts']=fits;local['bbox_2d']=union_bounds(fits)
    local['shared_point_diagnostics']=shared_point_diagnostics(fits,adjusted)
    local['shared_point_diagnostics'].update(joint_constraints_applied=True,numerical_point_source='source_projected_direction')
    for p in fits:p['diagnostics']['source_projected_direction']=copy.deepcopy(audit)
    return audit
