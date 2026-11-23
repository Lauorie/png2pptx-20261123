"""Nominate omitted straight implementations by their original source ridge.

This is an evidence locator, not a deletion certificate. Full native routes and
attached heads remain the comparison unit. Ambiguous ridges and distinct source
parallels do not acquire a new competition relation.
"""
import numpy as np


def straight(row):
    if row.get('closed') or 'segment' in row or row.get('filled_face') or row.get('filled_route'):return False
    p=np.asarray(row['points'],float);v=p[-1]-p[0];length=np.linalg.norm(v)
    return bool(length>=16 and np.quantile(np.abs(np.cross(v,p-p[0]))/length,.95)<1.)


def shared_ridge(a,b,rgb,*,comparable=False,headless_pair=False):
    """Both native corridors must locate the SAME unique persistent source band."""
    from .line_identity import pattern
    if not all(straight(r) and pattern(r.get('style',{}))=='solid' for r in (a,b)):return None
    if not headless_pair and not any(h not in (None,'none') for r in (a,b)for h in r.get('heads',())):return None
    short,long=sorted((a,b),key=lambda r:r['length'])
    ratio=short['length']/long['length']
    # Shortened spans and comparable omitted shape recoveries use distinct
    # eligibility checks at the caller; source identity remains required here.
    if not (.8<=ratio<=1. if comparable else .5<=ratio<.8):return None
    x=np.asarray(short['points'])[[0,-1]];y=np.asarray(long['points'])[[0,-1]]
    axis=x[1]-x[0];length=np.linalg.norm(axis);axis/=length;normal=np.array([-axis[1],axis[0]])
    other=y[1]-y[0];other/=np.linalg.norm(other)
    if abs(float(axis@other))<.995:return None
    if axis@other<0:y=y[::-1]
    along=(y-x[0])@axis;across=(y-x[0])@normal
    widths=[float(r.get('style',{}).get('width_px')or 1.)for r in (a,b)]
    guard=max(2.,sum(widths));displacement=float(np.mean(across))
    if np.ptp(across)>guard or abs(displacement)>.5*length:return None
    lo=max(0.,along[0]);hi=min(length,along[1])
    if hi-lo<.7*length or lo>.2*length or length-hi>.2*length:return None
    # Inspect surrounding source ridges too: a biased fit between two real
    # parallels must not simply select whichever one falls in a narrow crop.
    radius=max(6.,.5*length) if comparable else min(24.,max(6.,.5*length))
    offsets=np.arange(min(0.,displacement)-radius,max(0.,displacement)+radius+.25,.5)
    t=np.linspace(lo+.2*(hi-lo),hi-.2*(hi-lo),72)
    q=x[0]+t[:,None]*axis
    xy=np.rint(q[:,None]+normal[None,None]*offsets[None,:,None]).astype(int)
    valid=((xy[:,:,0]>=0)&(xy[:,:,0]<rgb.shape[1])&(xy[:,:,1]>=0)&(xy[:,:,1]<rgb.shape[0])).all(1)
    if valid.mean()<.9:return None
    xy[:,:,0]=np.clip(xy[:,:,0],0,rgb.shape[1]-1);xy[:,:,1]=np.clip(xy[:,:,1],0,rgb.shape[0]-1)
    pixels=rgb[xy[:,:,1],xy[:,:,0]].astype(float);colors=[]
    for r in (a,b):
        pigment=r.get('style',{}).get('stroke')or r.get('style',{}).get('color')
        if not isinstance(pigment,str)or len(pigment)!=7:return None
        colors.append(np.array([int(pigment[i:i+2],16)for i in (1,3,5)]))
    distance=np.minimum(*(np.linalg.norm(pixels-c,axis=2)for c in colors))
    background=np.median(np.concatenate([pixels[:,:4],pixels[:,-4:]],axis=1),axis=1)
    ink=(distance<70)&(np.linalg.norm(pixels-background[:,None],axis=2)>20)
    if headless_pair:
        # A one-pixel source line is an alpha mixture of its pigment and local
        # background. Absolute pigment distance misses a continuous pale ridge.
        # Use the same pigment direction and require visible contrast; no hue
        # or source occurrence is inferred from mere proximity.
        signal=background[:,None]-pixels
        for pigment in colors:
            direction=background-pigment
            norm=np.linalg.norm(direction,axis=1)
            projection=(signal*direction[:,None]).sum(2)/np.maximum(norm[:,None]**2,1e-9)
            predicted=projection[:,:,None]*direction[:,None]
            residual=np.linalg.norm(signal-predicted,axis=2)
            contrast=np.linalg.norm(signal,axis=2)
            ink|=(projection>.04)&(projection<=1.15)&(contrast>=12)&(
                residual<=.20*contrast)&(norm[:,None]>40)
    persistent=ink[valid].mean(0)>=.65
    for chunk in np.array_split(ink[valid],3):persistent&=chunk.mean(0)>=.5
    bands=np.flatnonzero(np.diff(np.r_[False,persistent,False].astype(int))).reshape(-1,2)
    bands=[(first,last)for first,last in bands if first>0 and last<len(offsets)
           and (last-first)*.5<=max(6.,2*max(widths))]
    if not bands:return None
    centers=np.array([np.mean(offsets[first:last])for first,last in bands])
    matched=[]
    for center in (0.,displacement):
        distances=np.abs(centers-center);order=np.argsort(distances);best=int(order[0])
        width=(bands[best][1]-bands[best][0])*.5
        if len(order)>1 and distances[order[1]]-distances[best]<=max(1.5,width):return None
        if distances[best]>.5*length:return None
        matched.append(best)
    if matched[0]!=matched[1]:return None
    index=matched[0];first,last=bands[index];ridge=centers[index]
    return {'source_band_count':len(bands),'source_ridge_width_px':(last-first)*.5,
            'shared_source_span_px':float(hi-lo),'shorter_visible_span_fraction':float((hi-lo)/length),
            'native_normal_displacement_px':abs(displacement),
            'source_ridge_points':(q[[0,-1]]+ridge*normal).round(3).tolist(),
            'source_relation':'nomination_only_complete_native_review_required'}


def nominate(rows,rgb,existing=()):
    seen={frozenset((p['a'],p['b']))for p in existing};out=[]
    eligible=[r for r in rows if straight(r)]
    from collections import Counter
    endpoint_counts=Counter((tuple(r['endpoint_nodes']),r['layer'])for r in rows
        if r.get('endpoint_nodes')and not r.get('closed')and 'segment'not in r
        and not r.get('filled_face')and not r.get('filled_route'))
    for index,a in enumerate(eligible):
        for b in eligible[index+1:]:
            if a['layer']==b['layer'] or frozenset((a['id'],b['id']))in seen:continue
            evidence=shared_ridge(a,b,rgb)
            # Headless internal neural lines retain their branch eligibility.
            # Unique endpoint identity plus the same SOURCE ridge can still
            # nominate their two recoveries when a coarse spatial gate missed
            # a shortened shaft. Multiple parallel ports never use this hint.
            endpoint=a.get('endpoint_nodes')
            headless=not any(h not in(None,'none')for r in(a,b)for h in r.get('heads',()))
            if evidence is None and headless and endpoint and tuple(endpoint)==tuple(b.get('endpoint_nodes',())):
                key=tuple(endpoint)
                if all(endpoint_counts[(key,layer)]==1 for layer in('shape','arrows')):
                    evidence=shared_ridge(a,b,rgb,headless_pair=True)
            if evidence is None:continue
            out.append({'a':a['id'],'b':b['id'],'nomination':'shared_original_straight_ridge',
                'p90_px':evidence['native_normal_displacement_px'],
                **({'scope':'unique_endpoint_headless_source_route'}if headless else{}),**evidence})
    # Previously matched arrows still need to compete with an omitted shape
    # recovery of the same source route. Only unreviewed shape paths enter
    # this fallback; a unique common source ridge establishes the nomination.
    involved={p[k]for p in [*existing,*out]for k in ('a','b')}
    shapes=[r for r in eligible if r['layer']=='shape'and r['id']not in involved]
    arrows=[r for r in eligible if r['layer']=='arrows']
    matches=[]
    for a in shapes:
        for b in arrows:
            evidence=shared_ridge(a,b,rgb,comparable=True)
            if evidence is not None:matches.append((a,b,evidence))
    # Several source ridges or equally plausible implementations must not
    # acquire deletion authority through a nearest-neighbor guess.
    for a in shapes:
        candidates=[m for m in matches if m[0]['id']==a['id']]
        if len(candidates)!=1:continue
        _,b,evidence=candidates[0]
        out.append({'a':a['id'],'b':b['id'],'nomination':'shared_original_straight_ridge',
            'p90_px':evidence['native_normal_displacement_px'],
            'scope':'unreviewed_comparable_shape_route',**evidence})
    return out
