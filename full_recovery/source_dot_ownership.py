"""Bind compact native copies to independently measured original dot groups.

Fitted displacement nominates a neighborhood. Original component count and
the complete native source-dot replacement establish ownership, independently
of a part's name or semantic class.
"""
from itertools import combinations

import cv2
import numpy as np
from PIL import Image


def source_unique_pattern(centers, extents, xy, diam, rgb, pigment):
    _,_,axes=np.linalg.svd(xy-xy.mean(0),full_matrices=False)
    source_span=float(np.ptp(xy@axes[0]));span=float(np.ptp(centers@axes[0]))
    # The physical source pattern defines its search scale. Inter-dot spacing
    # is not a bound on the displacement of a complete reconstructed group.
    radius=float(np.linalg.norm(np.ptp(xy,axis=0)+diam.max())/2)
    if (np.linalg.norm(centers.mean(0)-xy.mean(0))>radius or
            not .5*source_span<=span<=2*source_span or
            np.ptp(centers@axes[1])>.6*diam.max() or
            np.any(extents.max(1)<.3*diam.min()) or np.any(extents.max(1)>2*diam.max())):
        return False
    all_points=np.vstack((xy,centers))
    lo=np.maximum(0,np.floor(all_points.min(0)-diam.max())).astype(int)
    hi=np.minimum(rgb.shape[1::-1],np.ceil(all_points.max(0)+diam.max()+1)).astype(int)
    ink=(np.linalg.norm(rgb[lo[1]:hi[1],lo[0]:hi[0]].astype(float)-pigment,axis=2)<70).astype('uint8')
    count,_,stats,_=cv2.connectedComponentsWithStats(ink,8)
    # Extra source ink, including an external shaft entering the neighborhood,
    # defeats this broader nomination. Never discard it as irrelevant context.
    from .source_ellipsis import compact_component
    return count==4 and all(compact_component(w,h,area)
                           for x,y,w,h,area in stats[1:])


def resolve(scene,styles,rows,source,arrows=()):
    from .handoff import strokes
    from .source_ellipsis import observe
    with Image.open(source) as opened:
        rgb=np.asarray(opened.convert('RGB'));confirmed=observe(opened,rows,arrows)
    units=[]
    for key,part in scene['parts'].items():
        if part.get('status')!='ok':continue
        paths=strokes(part.get('commands',[]))
        if len(paths) not in (1,3):continue
        if len(paths)==1 and not (part.get('commands',[])[-1:]==[['Z']] or
                                  np.linalg.norm(paths[0][0]-paths[0][-1])<1e-6):continue
        extents=np.array([np.ptp(path,axis=0) for path in paths])
        if np.any(extents.min(1)<=0) or np.any(extents.max(1)>1.8*extents.min(1)):continue
        centers=np.array([(path.min(0)+path.max(0))/2 for path in paths])
        units.append((key,centers,extents,styles.get(key,{})))
    removed=set();audit=[]
    for mark in confirmed:
        xy=np.array([d['center'] for d in mark['dots']]);diam=np.array([d['diameter'] for d in mark['dots']])
        pigment=np.array([int(mark['color'][i:i+2],16) for i in (1,3,5)])
        radius=float(np.linalg.norm(np.ptp(xy,axis=0)+diam.max())/2)
        single=[];matched=set()
        for key,centers,extents,style in units:
            if not any(isinstance(p,str) and len(p)==7 and
                       np.linalg.norm(np.array([int(p[i:i+2],16) for i in (1,3,5)])-pigment)<=80
                       for p in (style.get('fill'),style.get('stroke'))):continue
            if np.any(extents.max(1)<.3*diam.min()) or np.any(extents.max(1)>2*diam.max()):continue
            if len(centers)==3:
                if source_unique_pattern(centers,extents,xy,diam,rgb,pigment):matched.add(key)
                continue
            center=centers[0];nearest=int(np.linalg.norm(xy-center,axis=1).argmin())
            if np.linalg.norm(center-xy[nearest])<=max(2.,.8*diam[nearest]):matched.add(key)
            if np.linalg.norm(center-xy.mean(0))<=2*radius:single.append((key,center,extents[0]))
        # Separate native parts can represent one displaced group too. Keep
        # the work bounded, and require the same original uniqueness proof.
        if 3<=len(single)<=12:
            for group in combinations(single,3):
                centers=np.array([r[1] for r in group]);extents=np.array([r[2] for r in group])
                if min(np.linalg.norm(centers[i]-centers[j]) for i,j in ((0,1),(0,2),(1,2)))<.8*diam.min():continue
                if source_unique_pattern(centers,extents,xy,diam,rgb,pigment):matched.update(r[0] for r in group)
        if matched:
            removed.update(matched)
            audit.append({'text_id':mark['text_id'],'text':mark['text'],'drop_shape_keys':sorted(matched),
                'source_dot_centers':xy.tolist(),'source_physical_component_count':3,
                'replacement':'source_measured_native_dots',
                'reason':'standalone_three_dot_source_occurrence_owned_once_by_text'})
    return sorted(removed),audit
