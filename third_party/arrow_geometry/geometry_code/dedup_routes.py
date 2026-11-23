"""Conservative whole-path reconciliation; never subtract or rewrite geometry."""
import copy
import numpy as np


def owner(row):
    return row.get('source_discovery_id',row.get('geometry_owner_id',row.get('parent_id',row['id'])))


def category(head):
    return 'headed' if head in ('triangle','arrow','stealth') else head


def polyline(row):
    if row.get('beziers'):
        pieces=[]
        for raw in row['beziers']:
            p=np.asarray(raw,float)
            if p.shape!=(4,2):raise ValueError('Invalid cubic controls')
            n=max(12,min(512,int(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()*2)+1))
            t=np.linspace(0,1,n)[:,None]
            q=(1-t)**3*p[0]+3*(1-t)**2*t*p[1]+3*(1-t)*t*t*p[2]+t**3*p[3]
            pieces.extend(q if not pieces else q[1:])
        points=np.asarray(pieces)
    else:points=np.asarray([row['start'],*row.get('via',[]),row['end']],float)
    if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all():raise ValueError('Invalid path coordinates')
    return points[np.r_[True,np.linalg.norm(np.diff(points,axis=0),axis=1)>1e-9]]


def sampled(row,n=129):
    points=polyline(row)
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    if len(points)<2 or arc[-1]<1e-8:raise ValueError('Degenerate path')
    s=np.linspace(0,arc[-1],n)
    return np.column_stack([np.interp(s,arc,points[:,k]) for k in range(2)]),float(arc[-1])


def connected_endpoints(row,rows):
    """Prefer the existing junction-aligned representation when choices agree."""
    def distance(point,path):
        a=path[:-1];v=np.diff(path,axis=0);den=(v*v).sum(axis=1)
        t=np.clip(((np.asarray(point)-a)*v).sum(axis=1)/np.maximum(den,1e-20),0,1)
        return float(np.linalg.norm(a+t[:,None]*v-np.asarray(point),axis=1).min())
    own=polyline(row);junctions=set()
    for other in rows:
        if other['id']==row['id'] or owner(other)!=owner(row):continue
        other_path=polyline(other)
        for end in (row['start'],row['end']):
            if distance(end,other_path)<1e-5:junctions.add(tuple(round(v,5) for v in end))
        for end in (other['start'],other['end']):
            if distance(end,own)<1e-5:junctions.add(tuple(round(v,5) for v in end))
    return len(junctions)


def separate_source_strokes(image,a,b,tolerance):
    """Veto repeated two-ridge evidence; absence of a veto is not proof of identity."""
    import cv2
    rgb=np.asarray(image.convert('RGB'),dtype=np.float32)
    votes=0;usable=0;details=[]
    for i in np.linspace(0.2*(len(a)-1),0.8*(len(a)-1),11).astype(int):
        tangent=a[min(i+2,len(a)-1)]-a[max(0,i-2)]
        norm=np.linalg.norm(tangent)
        if norm<1e-6:continue
        normal=np.array([-tangent[1],tangent[0]])/norm
        center=(a[i]+b[i])/2;offsets=np.array([np.dot(a[i]-center,normal),np.dot(b[i]-center,normal)])
        span=max(3.,float(np.abs(offsets).max())+max(2.,tolerance))
        offsets_grid=np.linspace(-span,span,max(33,int(span*8)+1))
        xy=center+offsets_grid[:,None]*normal
        if (xy[:,0]<0).any() or (xy[:,0]>rgb.shape[1]-1).any() or (xy[:,1]<0).any() or (xy[:,1]>rgb.shape[0]-1).any():continue
        colors=cv2.remap(rgb,xy[:,0].astype('float32')[None,:],xy[:,1].astype('float32')[None,:],cv2.INTER_LINEAR)[0]
        edge=max(2,len(colors)//6);background=np.median(np.r_[colors[:edge],colors[-edge:]],axis=0)
        contrast=np.linalg.norm(colors-background,axis=1)
        if contrast.max()<20:continue
        active=contrast>max(15.,float(contrast.max())*.45)
        starts=np.flatnonzero(active&~np.r_[False,active[:-1]])
        ends=np.flatnonzero(active&~np.r_[active[1:],False])
        usable+=1
        if len(starts)<2:continue
        centers=np.array([np.average(offsets_grid[s:e+1],weights=contrast[s:e+1]) for s,e in zip(starts,ends)])
        nearest=np.argmin(np.abs(centers[:,None]-offsets),axis=0)
        distances=np.abs(centers[nearest]-offsets)
        distinct=nearest[0]!=nearest[1] and float(np.abs(centers[nearest[0]]-centers[nearest[1]]))>=.75
        if distinct and float(distances.max())<=max(.6,tolerance*.55):
            votes+=1;details.append(dict(index=int(i),ridge_separation=float(abs(centers[nearest[0]]-centers[nearest[1]]))))
    return dict(separate_strokes=usable>=5 and votes>=max(4,int(np.ceil(usable*.6))),votes=votes,usable=usable,examples=details[:3])


def source_alignment(row,image,tolerance):
    """Choose among already-matched alternatives; never change merge eligibility."""
    import cv2
    rgb=np.asarray(image.convert('RGB'),dtype=np.float32);path,_=sampled(row,129)
    errors=[];foreground=[];backgrounds=[]
    for i in np.linspace(25,103,11).astype(int):
        tangent=path[i+2]-path[i-2];length=np.linalg.norm(tangent)
        if length<1e-6:continue
        normal=np.array([-tangent[1],tangent[0]])/length;span=max(3.,tolerance*1.5)
        offsets=np.linspace(-span,span,max(33,int(span*8)+1));xy=path[i]+offsets[:,None]*normal
        if (xy[:,0]<0).any() or (xy[:,0]>rgb.shape[1]-1).any() or (xy[:,1]<0).any() or (xy[:,1]>rgb.shape[0]-1).any():continue
        colors=cv2.remap(rgb,xy[:,0].astype('float32')[None,:],xy[:,1].astype('float32')[None,:],cv2.INTER_LINEAR)[0]
        edge=max(2,len(colors)//6);bg=np.median(np.r_[colors[:edge],colors[-edge:]],axis=0);contrast=np.linalg.norm(colors-bg,axis=1)
        if contrast.max()<20:continue
        active=contrast>max(15.,float(contrast.max())*.45)
        starts=np.flatnonzero(active&~np.r_[False,active[:-1]]);ends=np.flatnonzero(active&~np.r_[active[1:],False])
        if not len(starts):continue
        centers=[float(np.average(offsets[s:e+1],weights=contrast[s:e+1])) for s,e in zip(starts,ends)]
        k=int(np.argmin(np.abs(centers)));s,e=starts[k],ends[k]
        errors.append(abs(centers[k]));foreground.append(colors[s+int(np.argmax(contrast[s:e+1]))]);backgrounds.append(bg)
    if len(errors)<5:return dict(usable=len(errors),score=None)
    fg=np.median(foreground,axis=0);bg=np.median(backgrounds,axis=0);penalty=[]
    for endpoint in (path[0],path[-1]):
        xy=endpoint+np.array([[x,y] for x in (-.75,0,.75) for y in (-.75,0,.75)])
        xy[:,0]=np.clip(xy[:,0],0,rgb.shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,rgb.shape[0]-1)
        colors=cv2.remap(rgb,xy[:,0].astype('float32')[None,:],xy[:,1].astype('float32')[None,:],cv2.INTER_LINEAR)[0]
        penalty.append(float(np.clip(np.linalg.norm(colors-fg,axis=1).min()/max(20.,float(np.linalg.norm(fg-bg))),0,1)))
    center_error=float(np.median(errors));tip_penalty=float(np.mean(penalty))
    return dict(usable=len(errors),center_error_px=center_error,endpoint_ink_penalty=tip_penalty,
                score=center_error+.25*tolerance*tip_penalty)


def compatible(a,b,sa,sb,la,lb,tolerance,config,image):
    direct=max(np.linalg.norm(sa[0]-sb[0]),np.linalg.norm(sa[-1]-sb[-1]))
    reverse=max(np.linalg.norm(sa[0]-sb[-1]),np.linalg.norm(sa[-1]-sb[0]))
    flip=reverse<direct;endpoint=float(min(direct,reverse))
    if endpoint>tolerance:return None
    sb=sb[::-1] if flip else sb
    if max(la,lb)/min(la,lb)>config['max_length_ratio']:return None
    ordered=float(np.linalg.norm(sa-sb,axis=1).max())
    if ordered>tolerance or ordered/min(la,lb)>config['max_relative_distance']:return None
    ha=[category(x) for x in a.get('heads',['unknown','unknown'])]
    hb=[category(x) for x in b.get('heads',['unknown','unknown'])]
    if flip:hb=hb[::-1]
    record=dict(paths=[a['id'],b['id']],owners=[owner(a),owner(b)],endpoint_error_px=endpoint,
                ordered_distance_px=ordered,length_ratio=max(la,lb)/min(la,lb),reverse_orientation=bool(flip),heads_aligned=[ha,hb])
    if ha!=hb or 'unknown' in ha:
        return dict(**record,decision='review',reason='endpoint_type_or_direction_conflict')
    # Parameterized shape similarity alone must not suppress visible parallel ink.
    evidence=separate_source_strokes(image,sa,sb,tolerance) if config.get('source_parallel_check') else {'separate_strokes':False}
    if evidence['separate_strokes']:return dict(**record,decision='keep_both',reason='distinct_source_strokes',source_evidence=evidence)
    return dict(**record,decision='duplicate_candidate',reason='same_full_path_and_endpoints',source_evidence=evidence)


def reconcile(rows,image,tolerance_reference_px,config):
    rows=copy.deepcopy(rows);scale=max(image.size)/config['reference_long_side'];tolerance=tolerance_reference_px*scale
    ids=[r['id'] for r in rows]
    if len(set(ids))!=len(ids):raise ValueError('Input path IDs must be unique')
    samples={};unsupported=[]
    for row in rows:
        if row.get('kind')=='block_arrow':unsupported.append(row['id']);continue
        try:samples[row['id']]=sampled(row,config['sample_count'])
        except (ValueError,KeyError):unsupported.append(row['id'])
    pairs=[];edges={};byid={r['id']:r for r in rows}
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            if owner(a)==owner(b) or a['id'] not in samples or b['id'] not in samples:continue
            sa,la=samples[a['id']];sb,lb=samples[b['id']]
            result=compatible(a,b,sa,sb,la,lb,tolerance,config,image)
            if result is None:continue
            pairs.append(result)
            if result['decision']=='duplicate_candidate':edges[frozenset(result['paths'])]=result['ordered_distance_px']
    # Complete-link clusters: A~B and B~C alone cannot merge A and C.
    clusters=[[i] for i in ids]
    for pair,distance in sorted(edges.items(),key=lambda e:(e[1],sorted(e[0]))):
        first,second=sorted(pair);a=next(c for c in clusters if first in c);b=next(c for c in clusters if second in c)
        if a is b:continue
        if all(frozenset((x,y)) in edges for x in a for y in b):
            a.extend(b);clusters.remove(b)
    removed=set();decisions=[]
    for cluster in clusters:
        if len(cluster)<2:continue
        quality={i:source_alignment(byid[i],image,tolerance) for i in cluster} if config.get('keeper_selection')=='junction_then_source_alignment_then_medoid' else {}
        comparable=bool(quality) and all(q['score'] is not None for q in quality.values())
        def rank(i):
            return (-connected_endpoints(byid[i],rows),quality[i]['score'] if comparable else 0.,sum(edges.get(frozenset((i,j)),0.) for j in cluster if j!=i),i)
        keep=min(cluster,key=rank);drop=[i for i in cluster if i!=keep];removed.update(drop)
        decisions.append(dict(keep=keep,remove=drop,source_candidates=sorted({str(owner(byid[i])) for i in cluster}),
            reason='complete_link_whole_path_match',kept_shared_endpoint_count=connected_endpoints(byid[keep],rows),keeper_quality=quality))
    kept=[r for r in rows if r['id'] not in removed]
    return kept,dict(input_paths=len(rows),output_paths=len(kept),removed_paths=len(removed),
        tolerance_reference_px=tolerance_reference_px,tolerance_native_px=tolerance,
        decisions=decisions,pair_audit=pairs,unsupported_retained=unsupported,
        partial_overlap_policy='retain; no clipping, union, subtraction or coordinate changes',model_calls=0)
