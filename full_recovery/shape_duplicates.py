"""Collapse coincident shape recoveries only with a single source occurrence."""
import numpy as np
from .handoff import distances,strokes,resample
from .line_identity import single_source_stroke, separate_parallel_sources, compatible


def duplicates(rows, rgb):
    candidates=[r for r in rows if r['layer']=='shape' and not r.get('closed')
                and 'segment' not in r and not r.get('filled_face')]
    removed=set();audit=[]
    for i,a in enumerate(candidates):
        if a['id'] in removed:continue
        for b in candidates[i+1:]:
            if b['id'] in removed:continue
            # Head/dot attachments are independent semantic objects. A rule
            # does not discard them or replace unlike directional identities.
            ap,bp=a['points'],b['points']
            if min(a['length'],b['length'])/max(a['length'],b['length'])<.90:continue
            straight=all(np.quantile(distances(p,[p[[0,-1]]]),.95)<=.5 for p in (ap,bp))
            sa,sb=a['style'],b['style']
            if not compatible(sa,{**sb,'color':sb.get('stroke')}):continue
            width=max(sa.get('width_px') or 1,sb.get('width_px') or 1)
            tolerance=max(1.5,min(4.,width+1))
            reverse=np.linalg.norm(ap[0]-bp[-1])+np.linalg.norm(ap[-1]-bp[0])<np.linalg.norm(ap[0]-bp[0])+np.linalg.norm(ap[-1]-bp[-1])
            if a.get('heads')!=(b.get('heads',[])[::-1] if reverse else b.get('heads')):continue
            if reverse:bp=bp[::-1]
            if max(np.linalg.norm(ap[j]-bp[j]) for j in (0,-1))>tolerance:continue
            if max(np.quantile(distances(ap,[bp]),.95),np.quantile(distances(bp,[ap]),.95))>tolerance:continue
            pigment=sa.get('stroke')
            if not isinstance(pigment,str):continue
            if straight:
                if separate_parallel_sources(rgb,ap[[0,-1]],bp[[0,-1]],pigment,tolerance,sb.get('stroke')):continue
                fraction=single_source_stroke(rgb,ap[[0,-1]],bp[[0,-1]],pigment,tolerance,width)
            else:
                points=(ap+bp)/2;tangent=np.gradient(points,axis=0)
                normal=np.c_[-tangent[:,1],tangent[:,0]]/np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-9)
                offsets=np.arange(-int(np.ceil(tolerance+width+2)),int(np.ceil(tolerance+width+2))+1)
                xy=np.rint(points[:,None]+normal[:,None]*offsets[None,:,None]).astype(int)
                valid=((xy[:,:,0]>=0)&(xy[:,:,0]<rgb.shape[1])&(xy[:,:,1]>=0)&(xy[:,:,1]<rgb.shape[0])).all(1)
                xy[:,:,0]=np.clip(xy[:,:,0],0,rgb.shape[1]-1);xy[:,:,1]=np.clip(xy[:,:,1],0,rgb.shape[0]-1)
                color=np.array([int(pigment[j:j+2],16) for j in (1,3,5)])
                ink=np.linalg.norm(rgb[xy[:,:,1],xy[:,:,0]].astype(float)-color,axis=2)<70
                bands=np.sum(np.diff(np.pad(ink.astype(int),((0,0),(1,1))),axis=1)==1,axis=1)
                valid[:5]=False;valid[-5:]=False
                fraction=float(np.mean(bands[valid]==1)) if valid.sum()>=30 else 0.
            if fraction<.9:continue
            def loss(row):
                xy=np.rint(row['points']).astype(int)
                xy[:,0]=np.clip(xy[:,0],0,rgb.shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,rgb.shape[0]-1)
                color=np.array([int(row['style']['stroke'][j:j+2],16) for j in (1,3,5)])
                return float(np.median(np.linalg.norm(rgb[xy[:,1],xy[:,0]].astype(float)-color,axis=1)))
            keep,drop=(a,b) if loss(a)<=loss(b) else (b,a)
            removed.add(drop['id']);audit.append({'remove':drop['id'],'keep':keep['id'],
                'reason':'coincident_complete_route_one_source_band','source_fraction':fraction})
            if drop is a:break
    return removed,audit


def native_paint_order(pptx):
    """Read native leaf order, including leaves inside transformed groups."""
    from zipfile import ZipFile
    from lxml import etree
    from .compose import P, SLIDE
    with ZipFile(pptx) as archive:
        slide=etree.fromstring(archive.read(SLIDE))
    return {node.get('name','').split(' [',1)[0]:i
            for i,node in enumerate(slide.findall('.//{'+P+'}sp/{'+P+'}nvSpPr/{'+P+'}cNvPr'))}


def source_outline(a,b,rgb,pigment,width):
    """Require one source stroke around the whole pair, preserving double rings."""
    ap=resample(strokes(a.get('commands',[])),256)[:-1]
    bp=resample(strokes(b.get('commands',[])),256)[:-1]
    if len(ap)<60 or len(bp)<60:return 0.
    delta=ap[:,None]-bp[None,:]
    distance=np.linalg.norm(delta,axis=2)
    if max(distance.min(0).max(),distance.min(1).max())>4:return 0.
    points=(ap+bp[distance.argmin(1)])/2
    tangent=np.roll(points,-1,axis=0)-np.roll(points,1,axis=0)
    normal=np.c_[-tangent[:,1],tangent[:,0]]/np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-9)
    pad=int(np.ceil(4+width+2));offsets=np.arange(-pad,pad+.25,.5)
    xy=np.rint(points[:,None]+normal[:,None]*offsets[None,:,None]).astype(int)
    valid=((xy[:,:,0]>=0)&(xy[:,:,0]<rgb.shape[1])&(xy[:,:,1]>=0)&(xy[:,:,1]<rgb.shape[0])).all(1)
    xy[:,:,0]=np.clip(xy[:,:,0],0,rgb.shape[1]-1);xy[:,:,1]=np.clip(xy[:,:,1],0,rgb.shape[0]-1)
    color=np.array([int(pigment[j:j+2],16) for j in (1,3,5)])
    ink=np.linalg.norm(rgb[xy[:,:,1],xy[:,:,0]].astype(float)-color,axis=2)<70
    bands=np.sum(np.diff(np.pad(ink.astype(int),((0,0),(1,1))),axis=1)==1,axis=1)
    return float(np.mean(bands[valid]==1)) if valid.sum()>=max(60,.8*len(points)) else 0.


def visible_payload_between(scene,styles,rgb,a,b,paint_order):
    """A coincident front frame must not hide source-visible native contents."""
    if not paint_order:return []
    from .texture_fallback import polygon
    from shapely import contains_xy
    surfaces=[polygon(row[1]) for row in (a,b)]
    if any(p is None for p in surfaces) or surfaces[0].hausdorff_distance(surfaces[1])>.1:return []
    position=lambda p:paint_order.get(p['object_id']+'/'+p['id'],-1)
    lo,hi=sorted(position(row[1]) for row in (a,b))
    if lo<0 or lo==hi:return []
    result=[]
    for key,part in scene['parts'].items():
        if not lo<position(part)<hi:continue
        style=styles.get(key,{})
        fill=style.get('fill')
        if part.get('status')!='ok' or not isinstance(fill,str) or style.get('fill_opacity',1)!=1:continue
        color=np.array([int(fill[j:j+2],16) for j in (1,3,5)])
        if np.linalg.norm(color-a[4])<=8:continue
        body=polygon(part)
        if body is None or not all(surface.covers(body) for surface in surfaces):continue
        x0,y0=np.maximum(0,np.floor(body.bounds[:2])).astype(int)
        x1,y1=np.minimum(rgb.shape[1::-1],np.ceil(body.bounds[2:])).astype(int)
        if x1<=x0 or y1<=y0:continue
        inside=contains_xy(body,(np.arange(x0,x1)+.5)[None,:],(np.arange(y0,y1)+.5)[:,None])
        pixels=rgb[y0:y1,x0:x1].astype(float)
        foreground_distance=np.linalg.norm(pixels-color,axis=2)
        background_distance=np.linalg.norm(pixels-a[4],axis=2)
        visible=(foreground_distance<45)&(foreground_distance+8<background_distance)&inside
        if inside.any() and visible.sum()>.5*inside.sum():result.append(key)
    return result


def filled_duplicates(scene,styles,rgb,*,paint_order=None):
    """Coincident primitives must match one source outline or one filled blob.

    Source stroke bands distinguish a duplicate white node/frame from genuine
    double outlines. Numerical width differences do not establish identity.
    """
    import cv2
    candidates=[]
    for key,p in scene['parts'].items():
        s=styles.get(key,{})
        if p.get('status')!='ok' or p.get('kind') not in ('ellipse','rectangle','rounded_rectangle'):continue
        fill=s.get('fill')
        if not isinstance(fill,str) or s.get('fill_opacity',1)!=1:continue
        color=np.array([int(fill[j:j+2],16) for j in (1,3,5)])
        b=np.asarray(p['bbox_2d'],float);wh=b[2:]-b[:2]
        if min(wh)<=0:continue
        if not s.get('stroke') and (color.min()>235 or max(wh)>.06*max(rgb.shape[:2])):continue
        candidates.append((key,p,b,wh,color))
    removed=set();audit=[]
    for i,a in enumerate(candidates):
        if a[0] in removed:continue
        for b in candidates[i+1:]:
            if b[0] in removed or a[1]['kind']!=b[1]['kind']:continue
            if np.linalg.norm(a[4]-b[4])>35:continue
            low=np.maximum(a[2][:2],b[2][:2]);high=np.minimum(a[2][2:],b[2][2:])
            if np.prod(np.maximum(0,high-low))<.85*max(np.prod(a[3]),np.prod(b[3])):continue
            ca=(a[2][:2]+a[2][2:])/2;cb=(b[2][:2]+b[2][2:])/2
            if np.linalg.norm(ca-cb)>.15*min(*a[3],*b[3]):continue
            payload=visible_payload_between(scene,styles,rgb,a,b,paint_order)
            # A coincident fill is not a replacement for its border. Models
            # can split those paints across two parts which both carry fill.
            sa,sb=styles.get(a[0],{}),styles.get(b[0],{})
            stroke_a,stroke_b=sa.get('stroke'),sb.get('stroke')
            if not payload and stroke_a!=stroke_b:continue
            if not payload and stroke_a and any(sa.get(k,default)!=sb.get(k,default) for k,default in
                               [('dash','solid'),('stroke_opacity',1)]):continue
            # Fitter noise below a tenth of a source pixel is not unique paint.
            # Keep materially different borders, including split fill/outline work.
            if not payload and stroke_a and abs((sa.get('width_px') or 1)-(sb.get('width_px') or 1))>max(
                    .1,.05*max(sa.get('width_px') or 1,sb.get('width_px') or 1)):continue
            outline_fraction=0.
            if stroke_a:
                outline_fraction=source_outline(a[1],b[1],rgb,stroke_a,
                    max(sa.get('width_px') or 1,sb.get('width_px') or 1))
                if outline_fraction<.9 and not payload:continue
            if stroke_a:
                observed=(a[2]+b[2])/2
            else:
                lo=np.maximum(0,np.floor(np.minimum(a[2][:2],b[2][:2])-3)).astype(int)
                hi=np.minimum(rgb.shape[1::-1],np.ceil(np.maximum(a[2][2:],b[2][2:])+3)).astype(int)
                if np.any(hi<=lo):continue
                ink=(np.linalg.norm(rgb[lo[1]:hi[1],lo[0]:hi[0]].astype(float)-(a[4]+b[4])/2,axis=-1)<45).astype('uint8')
                count,labels,stats,centers=cv2.connectedComponentsWithStats(ink,8)
                cs=np.rint(np.array([ca,cb])-lo).astype(int)
                if np.any(cs<0) or np.any(cs>=hi-lo):continue
                same=labels[cs[:,1],cs[:,0]]
                if not same[0] or same[0]!=same[1]:continue
                x,y,w,h,area=stats[same[0]]
                if not .35*min(np.prod(a[3]),np.prod(b[3]))<=area<=1.25*max(np.prod(a[3]),np.prod(b[3])):continue
                observed=np.array([x,y,x+w,y+h])+np.tile(lo,2)
            # The visible copy may be in front of a panel while the slightly
            # closer geometric copy is behind it. Removing the front copy can
            # erase the source occurrence. Native order wins within an already
            # established duplicate pair; never move or reshape either part.
            def rank(row):
                part=row[1];name=part['object_id']+'/'+part['id']
                return ((paint_order or {}).get(name,-1),-float(np.abs(row[2]-observed).sum()))
            keep,drop=sorted((a,b),key=rank,reverse=not bool(payload))
            removed.add(drop[0]);audit.append({'remove':drop[0],'keep':keep[0],
                'reason':'coincident_frame_covers_source_payload' if payload else
                    'coincident_primitive_single_source_outline' if stroke_a else 'coincident_filled_primitive_single_source_component',
                'single_source_outline_fraction':outline_fraction if stroke_a else None,
                'selection':'source_visible_payload_between_frames' if payload else
                    'native_paint_order_then_geometry' if paint_order else 'geometry',
                **({'preserved_payload_keys':payload} if payload else {})})
            if drop is a:break
    return removed,audit
