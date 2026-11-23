"""Measure a confirmed curve from shaft cross-sections, without skeletons.

The input route supplies ordering, not exact coordinates. Broad endpoint heads
are style evidence, not shaft observations. No identities or head types change.
"""
import numpy as np
from scipy.interpolate import BSpline
from scipy.ndimage import map_coordinates, label, find_objects, distance_transform_edt
from scipy.spatial import cKDTree
from lib import fit, style


def _sample_curves(curves, count=256):
    parts=[]
    for c in curves:
        c=np.asarray(c);t=np.linspace(0,1,max(8,count//len(curves)))[:,None]
        parts.append((1-t)**3*c[0]+3*(1-t)**2*t*c[1]+3*(1-t)*t**2*c[2]+t**3*c[3])
    return np.concatenate(parts)


def _mark_observations(mask,row):
    """Keep within-mark direction as well as mark centers; never new identities."""
    labels,n=label(mask,np.ones((3,3)))
    if n<4:return None
    seed=fit.points(row);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(seed,axis=0),axis=1))]
    tree=cKDTree(seed);nearest=distance_transform_edt(labels==0,return_distances=False,return_indices=True)
    excluded=set()
    for index,head in [(0,row['heads'][0]),(-1,row['heads'][1])]:
        if head in ('none','unknown'):continue
        x,y=np.rint(seed[index]).astype(int);x=np.clip(x,0,mask.shape[1]-1);y=np.clip(y,0,mask.shape[0]-1)
        excluded.add(int(labels[tuple(nearest[:,y,x])]))
    obs=[]
    for k,sl in enumerate(find_objects(labels),1):
        if sl is None or k in excluded:continue
        yy,xx=np.nonzero(labels[sl]==k);xy=np.c_[xx+sl[1].start,yy+sl[0].start].astype(float)
        if len(xy)<2:continue
        center=np.mean(xy,axis=0);distance,i=tree.query(center)
        direction=seed[min(len(seed)-1,i+2)]-seed[max(0,i-2)]
        direction/=max(np.linalg.norm(direction),1e-9)
        values,vectors=np.linalg.eigh(np.cov(xy.T));axis=vectors[:,-1]
        if axis@direction<0:axis=-axis
        anisotropy=float((values[-1]-values[0])/max(values.sum(),1e-9))
        axis=anisotropy*axis+(1-anisotropy)*direction;axis/=max(np.linalg.norm(axis),1e-9)
        projected=(xy-center)@axis;span=np.quantile(projected,[.15,.5,.85])
        # Multiple measurements retain the direction inside a mark, which a
        # single centroid loses on a wave. Remote specks receive tiny weight.
        weight=np.sqrt(len(xy))*np.exp(-distance/max(4.,row.get('width_estimate') or 8))
        for v in span:
            point=center+axis*v;t=float(arc[i]+v*(axis@direction))
            if 0<t<arc[-1]:obs.append((t,point,weight))
    if len(obs)<9:return None
    obs.sort(key=lambda item:item[0]);times=np.array([x[0] for x in obs]);keep=np.r_[True,np.diff(times)>1e-6]
    return [(t,p,w) for (t,p,w),yes in zip(obs,keep) if yes],excluded


def _dashed(mask,row):
    measured=_mark_observations(mask,row)
    if measured is None:return None
    observations,excluded=measured;seed=fit.points(row)
    length=np.linalg.norm(np.diff(seed,axis=0),axis=1).sum()
    times=np.asarray([o[0] for o in observations]);q=np.asarray([o[1] for o in observations])
    weights=np.asarray([o[2] for o in observations])
    width=row.get('width_estimate') or float(2*np.median(distance_transform_edt(mask)[mask]))
    # Fit observations with their actual confidence instead of interpolating
    # every fragment. A far-away mark may belong to another physical curve.
    curves,residual=_joint_curve(q,times,seed[[0,-1]],max(1.,width),length,weights)
    p=_sample_curves(curves)
    return dict(kind='curve',start=p[0].tolist(),via=p[1:-1].tolist(),end=p[-1].tolist(),beziers=curves.tolist(),
        width_estimate=row.get('width_estimate'),geometry_method='weighted_mask_mark_spline',
        fit_evidence=dict(observations=len(observations),excluded_head_components=sorted(excluded),
            endpoints='model-confirmed endpoints preserved',within_mark_direction=True,
            measurement_weights_used=True,weight_range=[float(weights.min()),float(weights.max())],
            residual_median=float(np.median(residual))))


def _joint_curve(observations, times, endpoints, width, length, evidence_weights=None):
    """Small robust penalized B-spline solve with exact endpoint constraints."""
    u=np.asarray(times)/length
    nc=max(6,min(24,len(u)//2+2))
    knots=np.r_[np.zeros(4),np.linspace(0,1,nc-2)[1:-1],np.ones(4)]
    basis=BSpline(knots,np.eye(nc),3)
    b=basis(u);grid=np.linspace(0,1,128)
    penalty=basis(grid,nu=2)/np.sqrt(len(grid))
    # Scale smoothing to both thickness and route length. A broad head/short
    # shaft must not give the regularizer a scale comparable to the whole route.
    smooth_length=min(3*width,.08*length)
    lam=len(u)*(smooth_length/length)**4
    fixed=np.zeros((nc,2));fixed[[0,-1]]=endpoints
    target=observations-b@fixed
    prior=np.ones(len(u)) if evidence_weights is None else np.asarray(evidence_weights,float)
    if not np.isfinite(prior).all() or (prior<0).any() or not np.any(prior>0):raise ValueError('No finite positive curve evidence weights')
    prior=prior/prior.max();control=fixed.copy();weights=prior.copy()
    for _ in range(4):
        a=np.vstack([b[:,1:-1]*np.sqrt(weights[:,None]),np.sqrt(lam)*penalty[:,1:-1]])
        rhs=np.vstack([target*np.sqrt(weights[:,None]),-np.sqrt(lam)*penalty@fixed])
        control[1:-1]=np.linalg.lstsq(a,rhs,rcond=None)[0]
        residual=np.linalg.norm(b@control-observations,axis=1)
        weights=prior*np.minimum(1,max(1.,width*.5)/np.maximum(residual,1e-9))
    spline=BSpline(knots,control,3);segments=[]
    for a,z in zip(np.unique(knots)[:-1],np.unique(knots)[1:]):
        p0,p3=spline(a),spline(z);d0,d1=spline(a,nu=1),spline(z,nu=1)
        segments.append([p0,p0+d0*(z-a)/3,p3-d1*(z-a)/3,p3])
    return np.asarray(segments),residual


def refine(mask,row):
    """Measure a confirmed curve without freezing the initial guide normals.

    Physical endpoints are exact constraints. Shaft samples, including its last
    sample, are fallible observations with robust weights. A single smooth spline
    reaches both tips; there is no separately patched head attachment.
    """
    if row.get('line_style') in ('dashed','dotted'):
        measured=_dashed(mask,row)
        if measured is not None:return measured
    seed=fit.points(row);endpoints=seed[[0,-1]].copy();history=[]
    headed={end+'_head':head not in ('none','unknown') for end,head in zip(['start','end'],row['heads'])}
    for iteration in range(4):
        distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(seed,axis=0),axis=1))];length=distance[-1]
        if length<2:raise ValueError('Curve is too short')
        s=np.linspace(0,length,max(32,min(512,round(length))))
        q=np.c_[np.interp(s,distance,seed[:,0]),np.interp(s,distance,seed[:,1])]
        tangent=np.gradient(q,axis=0);tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-9)
        normal=np.c_[-tangent[:,1],tangent[:,0]]
        radius=max(12.,min(128.,length*.3),2*(row.get('width_estimate') or 2))
        offsets=np.arange(-radius,radius+.25,.5)
        locations=q[:,None,:]+normal[:,None,:]*offsets[None,:,None]
        occupied=map_coordinates(mask.astype(float),[locations[...,1],locations[...,0]],order=0,mode='constant')>0
        centers=q.copy();spans=np.zeros(len(q));valid=np.zeros(len(q),bool)
        for i,hit in enumerate(occupied):
            intervals=[(a,b) for a,b in style.runs(hit) if a>0 and b<len(offsets)]
            if not intervals:continue
            a,b=min(intervals,key=lambda ab:abs((offsets[ab[0]]+offsets[ab[1]-1])/2))
            centers[i]+=normal[i]*(offsets[a]+offsets[b-1])/2
            spans[i]=offsets[b-1]-offsets[a]+.5;valid[i]=True
        if valid.sum()<8:raise ValueError('Too few visible sections')
        width=float(np.median(spans[valid]))
        body,regions=style.head_profile_body(s,spans,width,headed)
        use=body&valid
        if use.sum()<8:raise ValueError('Too few shaft observations')
        width=float(np.median(spans[use]));ids=np.flatnonzero(use)
        count=max(12,min(64,round((s[ids[-1]]-s[ids[0]])/max(2*width,1))))
        edges=np.linspace(s[ids[0]],s[ids[-1]]+1e-6,count+1)
        times=[];obs=[]
        for lo,hi in zip(edges,edges[1:]):
            part=use&(s>=lo)&(s<hi)
            if part.any():times.append(float(np.median(s[part])));obs.append(np.median(centers[part],axis=0))
        if len(obs)<4:raise ValueError('Insufficient independent shaft samples')
        curves,error=_joint_curve(np.asarray(obs),times,endpoints,width,length)
        if not np.isfinite(curves).all():raise ValueError('Nonfinite joint curve fit')
        seed=_sample_curves(curves);seed[0]=endpoints[0];seed[-1]=endpoints[1]
        history.append(dict(iteration=iteration,shaft_samples=len(obs),width=width,
                            head_regions=regions,residual_median=float(np.median(error)),
                            last_observation=np.asarray(obs)[-1].tolist()))
    return dict(kind='curve',start=endpoints[0].tolist(),via=seed[1:-1].tolist(),end=endpoints[-1].tolist(),
                beziers=curves.tolist(),width_estimate=width,geometry_method='joint_shaft_tips_curve',
                fit_evidence=dict(iterations=history,endpoints='fixed',head_attachment='joint spline, no added segment'))
