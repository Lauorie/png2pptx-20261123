"""Fit pixels along model-owned routes. No identity, merge or deletion decisions."""
import copy
import numpy as np
from scipy import ndimage
from scipy.interpolate import splprep, splev, PchipInterpolator
from skimage.measure import find_contours


def points(row):
    return np.asarray(row['points'] if 'points' in row else [row['start'],*row['via'],row['end']],float)


def robust_axis(xy,a,b,evidence_weight=None):
    """Ordered anchors identify the route; robust TLS estimates its visible axis."""
    direction=b-a;length=np.linalg.norm(direction)
    if length<1e-6: raise ValueError('Coincident anchors')
    direction/=length
    t=(xy-a)@direction
    distance=np.linalg.norm(xy-(a+np.clip(t,0,length)[:,None]*direction),axis=1)
    bandwidth=max(3.,min(length*.15,30.))
    prior=np.exp(-.5*(distance/bandwidth)**2)
    if evidence_weight is not None:prior*=evidence_weight
    weights=prior.copy();center=(a+b)/2
    for _ in range(6):
        if weights.sum()<1e-8: raise ValueError('No nearby mask evidence')
        center=(xy*weights[:,None]).sum(axis=0)/weights.sum()
        cov=(xy-center).T@((xy-center)*weights[:,None])
        _,v=np.linalg.eigh(cov);direction=v[:,-1]
        if direction@(b-a)<0:direction=-direction
        normal=np.array([-direction[1],direction[0]])
        residual=(xy-center)@normal
        scale=max(.75,float(np.median(abs(residual[prior>.5])))*1.4826) if np.any(prior>.5) else 1.
        weights=prior/(1+(residual/(2*scale))**2)
    project=lambda p:center+((p-center)@direction)*direction
    return center,direction,project(a),project(b),weights


def tip(xy,axis,anchor,at_start):
    center,direction,_,_,weights=axis
    # Refine the longitudinal extent near the model's endpoint. A remote speck
    # cannot stretch the entire line; raw evidence is retained for review.
    projection=(xy-center)@direction;reference=(anchor-center)@direction
    length=np.linalg.norm(axis[3]-axis[2]);reach=max(4.,.12*length)
    use=(weights>.2)&(abs(projection-reference)<=reach)
    if not use.any():return center+reference*direction
    value=projection[use].min() if at_start else projection[use].max()
    return center+value*direction


def smooth(path,smoothing=0):
    path=np.asarray(path,float)
    path=path[np.r_[True,np.linalg.norm(np.diff(path,axis=0),axis=1)>1e-6]]
    if len(path)<2:raise ValueError('Not enough distinct curve points')
    tck,_=splprep(path.T,s=smoothing,k=min(3,len(path)-1))
    knots=np.unique(tck[0]);knots=knots[(knots>=0)&(knots<=1)]
    curves=[]
    for a,b in zip(knots,knots[1:]):
        p0=np.asarray(splev(a,tck));p3=np.asarray(splev(b,tck))
        d0=np.asarray(splev(a,tck,der=1));d1=np.asarray(splev(b,tck,der=1))
        curves.append([p0.tolist(),(p0+d0*(b-a)/3).tolist(),(p3-d1*(b-a)/3).tolist(),p3.tolist()])
    sampled=np.column_stack(splev(np.linspace(0,1,max(32,min(256,len(path)*4))),tck))
    return sampled,curves


def interpolate_points(path):
    """Local shape-preserving interpolation of model anchors, without overshoot.

    Equal x/y on adjacent anchors remains equal throughout that span. Unlike a
    single global interpolating cubic, a distant corner cannot bow a straight leg.
    """
    p=np.asarray(path,float);p=p[np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-6]]
    if len(p)<2:raise ValueError('Not enough distinct curve points')
    t=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    f=PchipInterpolator(t,p,axis=0);d=f.derivative()(t);curves=[]
    for i,h in enumerate(np.diff(t)):
        curves.append([p[i].tolist(),(p[i]+d[i]*h/3).tolist(),(p[i+1]-d[i+1]*h/3).tolist(),p[i+1].tolist()])
    sampled=f(np.unique(np.r_[t,np.linspace(0,t[-1],max(32,min(256,len(p)*8)))]))
    return sampled,curves


def outline(mask,anchors):
    """Refine model-ordered polygon edges on ALL associated mask boundaries.

    No convex hull: concave shoulders and detached head/body evidence remain.
    Anchors supply topology; this routine only estimates the boundary lines.
    """
    curves=find_contours(np.pad(mask.astype(float),1),.5)
    paths=[c[:,::-1]-1 for c in curves]
    xy=np.concatenate(paths)
    tangents=np.concatenate([np.roll(c,-3,axis=0)-np.roll(c,3,axis=0) for c in paths])
    tangents/=np.maximum(np.linalg.norm(tangents,axis=1,keepdims=True),1e-9)
    axes=[];support=[]
    for a,b in zip(anchors,np.roll(anchors,-1,axis=0)):
        direction=(b-a)/np.linalg.norm(b-a)
        # Tangent evidence prevents a nearby perpendicular shaft edge from
        # pulling a concave head shoulder into a horizontal/vertical notch.
        alignment=np.abs(tangents@direction)**4
        support.append(int((alignment>.5).sum()))
        if len(xy)>=2:ax=robust_axis(xy,a,b,alignment)
        else:
            direction=(b-a)/np.linalg.norm(b-a)
            ax=((a+b)/2,direction,a,b,None)
        axes.append(ax)
    vertices=[]
    for i,right in enumerate(axes):
        left=axes[i-1]
        normals=np.array([[-left[1][1],left[1][0]],[-right[1][1],right[1][0]]])
        offsets=np.array([normals[0]@left[0],normals[1]@right[0]])
        vertices.append(np.linalg.lstsq(np.vstack([normals,np.eye(2)*.01]),
                                       np.r_[offsets,anchors[i]*.01],rcond=None)[0])
    return np.asarray(vertices),support


def estimate(mask,row,mode='mask'):
    anchors=points(row);fitted=anchors.copy();beziers=[];details=dict(mode=mode)
    if row['kind']=='block_arrow':
        # The vertices already encode the whole arrowhead; there is no shaft
        # width or synthetic head marker in this representation.
        if len(anchors)>1 and np.linalg.norm(anchors[-1]-anchors[0])<1e-6:anchors=anchors[:-1]
        fitted=anchors.copy();width=None
        if mode=='mask' and mask is not None and mask.any():
            fitted,support=outline(mask,anchors)
            details.update(method='Closed contour: model edge ownership and robust mask-boundary fitting',edge_samples=support)
        else:details['method']='Model closed contour; no mask boundary evidence'
    elif mode=='points' or mask is None or not mask.any():
        if row['kind']=='curve':fitted,beziers=interpolate_points(anchors)
        details['method']='Model points; local shape-preserving interpolation for curves'
        width=None
    else:
        # The current mask fitter uses cross-sections and robust axes. Legacy
        # skeleton/path-search implementations survive only in frozen run code.
        from mask_vectors import fit_route
        measured=fit_route(mask,row['kind'],anchors,row['heads'],'mask')
        fitted=points(measured);beziers=measured.get('beziers',[])
        width=measured.get('width_estimate')
        details.update(method=measured['geometry_method'])
    if not np.isfinite(fitted).all():raise ValueError('Nonfinite fitted geometry')
    if np.linalg.norm(np.diff(fitted,axis=0),axis=1).sum()<1e-6:raise ValueError('Degenerate fitted route')
    result=copy.deepcopy(row)
    result.update(start=fitted[0].tolist(),via=fitted[1:-1].tolist(),end=fitted[-1].tolist(),
                  anchors=anchors.tolist(),beziers=beziers,width_estimate=width,fit=details)
    if row['kind']=='block_arrow':
        result.update(closed=True,heads=['none','none'],start_head=False,end_head=False,
                      start_head_type=None,end_head_type=None)
    result.pop('points',None)
    return result


def transform(row,view,to_global=True):
    result=copy.deepcopy(row);key='view_to_global' if to_global else 'global_to_view';m=view[key]
    def p(v):return [v[0]*m[0][0]+m[0][2],v[1]*m[1][1]+m[1][2]]
    for k in ['start','end']:result[k]=p(row[k])
    result['via']=[p(v) for v in row['via']]
    result['anchors']=[p(v) for v in row.get('anchors',[])]
    result['beziers']=[[p(v) for v in c] for c in row.get('beziers',[])]
    if 'contours' in row:result['contours']=[[p(v) for v in c] for c in row['contours']]
    for field in ('shape_parts','mask_shape_parts'):
        for part in result.get(field,[]):
            part['commands']=[[c[0],*[v for i in range(1,len(c),2) for v in p(c[i:i+2])]] for c in part['commands']]
            if 'bbox_2d' in part:part['bbox_2d']=[*p(part['bbox_2d'][:2]),*p(part['bbox_2d'][2:])]
    if isinstance(row.get('shaft_sample'),list):result['shaft_sample']=[p(v) for v in row['shaft_sample']]
    scale=float(np.sqrt(abs(m[0][0]*m[1][1])))
    if row.get('width_estimate') is not None:result['width_estimate']*=scale
    if row.get('style'):
        if row['style'].get('width_px') is not None:result['style']['width_px']*=scale
        if to_global and row['style'].get('width_px') is not None:
            result['style'].setdefault('evidence',{})['width_conversion']=dict(
                local_width_px=row['style']['width_px'],local_image_size=view.get('size'),
                crop_bbox_global=view.get('bbox_global'),local_to_global_scale=scale,
                global_width_px=result['style']['width_px'])
        if row['style'].get('border_width_px') is not None:result['style']['border_width_px']*=scale
        for part in result['style'].get('shape_parts',{}).values():
            part['width_px']*=scale
            if part.get('dash_pattern_px'):part['dash_pattern_px']=[x*scale for x in part['dash_pattern_px']]
        if row['style'].get('dash',{}).get('pattern_px'):
            result['style']['dash']['pattern_px']=[x*scale for x in row['style']['dash']['pattern_px']]
        for k in ['mark_px','gap_px','period_px','phase_px']:
            if row['style'].get('dash',{}).get(k) is not None:result['style']['dash'][k]*=scale
    return result
