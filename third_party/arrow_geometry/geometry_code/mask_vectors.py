"""Mask measurements without skeletons, endpoint marching or gap graphs.

Every result is a geometric proposal. Identity, representation, mask ownership
and shared-node semantics belong to the model, not these numerical estimators.
"""
import copy
import numpy as np
from scipy.ndimage import median_filter
import straight_mask
from lib import fit


def contour(mask):
    from skimage.measure import find_contours,approximate_polygon
    contours=[]
    for c in find_contours(np.pad(mask.astype(float),1),.5):
        c=approximate_polygon(c[:,::-1]-1,tolerance=.75)
        if len(c)>1 and np.allclose(c[0],c[-1]):c=c[:-1]
        if len(c)>=3:contours.append(c.tolist())
    if not contours:raise ValueError('No closed contour in the selected mask')
    p=max(contours,key=len)
    return dict(kind='block_arrow',start=p[0],via=p[1:-1],end=p[-1],contours=contours,
                closed=True,beziers=[],width_estimate=None,geometry_method='model_selected_mask_contour')


def axis_end(ink,axis,start):
    """Use the associated whole shaft's projected extent, not an anchor window.

    The robust axis defines the normal corridor; trimmed projections prevent a
    single remote pixel from determining the end. This does not choose identity.
    """
    center,direction=axis[:2];normal=np.array([-direction[1],direction[0]])
    distance=np.abs((ink-center)@normal)
    corridor=distance<=max(2.,float(np.percentile(distance,60))*2)
    projection=(ink[corridor]-center)@direction
    if not len(projection):raise ValueError('No mask evidence on the confirmed axis')
    extent=float(np.quantile(projection,.002 if start else .998))
    return center+extent*direction


def curve_proposal(mask):
    y,x=np.nonzero(mask)
    if len(x)<12:return None
    xy=np.c_[x,y].astype(float);origin=np.median(xy,axis=0)
    _,v=np.linalg.eigh(np.cov(xy.T));axis=v[:,-1]
    if axis[np.argmax(abs(axis))]<0:axis=-axis
    normal=np.array([-axis[1],axis[0]])
    u=(xy-origin)@axis;z=(xy-origin)@normal
    lo,hi=np.quantile(u,[.002,.998]);step=max(1.,(hi-lo)/160)
    bins=np.floor((u-lo)/step).astype(int);centers=[];widths=[]
    for k in range(max(1,int((hi-lo)/step)+1)):
        inside=bins==k
        if inside.sum()<2:continue
        centers.append(origin+np.median(u[inside])*axis+np.median(z[inside])*normal)
        widths.append(float(np.ptp(z[inside])+1))
    if len(centers)<6:return None
    centers=np.asarray(centers)
    # Cross-section medians are observations, not a connected pixel graph.
    p,curves=fit.smooth(centers,max(.5,float(np.median(widths))*.08)**2*len(centers))
    return dict(kind='curve',start=p[0].tolist(),via=p[1:-1].tolist(),end=p[-1].tolist(),
                beziers=curves,width_estimate=float(np.percentile(widths,25)),
                geometry_method='mask_cross_section_spline',
                fit_evidence=dict(sections=len(centers),proposal_only=True,
                    limitation='PCA cross-sections may be ambiguous on loops, branches or mixed masks'))


def propose(mask):
    line=straight_mask.estimate(mask)
    if line is not None:
        line.pop('route_path',None)
        return line
    return curve_proposal(mask)


def fit_route(mask,kind,anchors,heads,geometry='mask',line_style='solid'):
    """Refine the model's geometric family; never decide connectivity here."""
    anchors=np.asarray(anchors,float)
    row=dict(id='route',kind=kind,points=anchors.tolist(),heads=heads)
    if kind=='block_arrow' and geometry=='mask' and mask is not None and mask.any():return contour(mask)
    if geometry=='points' or mask is None or not mask.any() or kind=='block_arrow':
        result=fit.estimate(None,row,mode='points')
        result['geometry_method']='model_keypoints'
        return result
    y,x=np.nonzero(mask);xy=np.c_[x,y].astype(float)
    if kind=='straight':
        axis=fit.robust_axis(xy,anchors[0],anchors[-1])
        # Explicit anchors describe the complete target. A supporting mask may
        # contain only its head or span more than one physical target. Preserve
        # model-owned extent; a complete D proposal keeps its own measured ends.
        pts=np.array([axis[2],axis[3]])
        curves=[];method='mask_axis_model_extent'
    elif kind=='polyline':
        distances=[]
        for a,b in zip(anchors,anchors[1:]):
            d=b-a;t=np.clip((xy-a)@d/max(d@d,1e-9),0,1)
            distances.append(np.linalg.norm(xy-(a+t[:,None]*d),axis=1))
        owners=np.argmin(distances,axis=0)
        axes=[fit.robust_axis(xy[owners==i],a,b) for i,(a,b) in enumerate(zip(anchors,anchors[1:]))]
        pts=[axis_end(xy[owners==0],axes[0],True)]
        for i,(left,right) in enumerate(zip(axes,axes[1:]),1):
            normals=np.array([[-left[1][1],left[1][0]],[-right[1][1],right[1][0]]])
            offsets=np.array([normals[0]@left[0],normals[1]@right[0]])
            # Anchored least squares stays finite for nearly parallel legs.
            pts.append(anchors[i]+np.linalg.lstsq(np.vstack([normals,.05*np.eye(2)]),
                np.r_[offsets-normals@anchors[i],[0.,0.]],rcond=None)[0])
        pts=np.array([*pts,axis_end(xy[owners==len(axes)-1],axes[-1],False)])
        curves=[];method='model_bends_mask_axes'
    else:
        # The model has selected the route and endpoints; masks fit its interior.
        import curve_sampling
        if len(anchors)>2:
            seed,beziers=fit.interpolate_points(anchors)
        else:
            proposal=curve_proposal(mask)
            if proposal is None:raise ValueError('Insufficient curve evidence')
            seed=fit.points(proposal)
            if np.linalg.norm(seed[-1]-anchors[0])<np.linalg.norm(seed[0]-anchors[0]):seed=seed[::-1].copy()
            seed[0]=anchors[0];seed[-1]=anchors[-1];beziers=[]
        seedrow=dict(kind='curve',start=seed[0].tolist(),via=seed[1:-1].tolist(),end=seed[-1].tolist(),
            heads=heads,line_style=line_style,width_estimate=None)
        measured=curve_sampling.refine(mask,seedrow)
        measured['anchors']=anchors.tolist()
        return measured
    if not np.isfinite(pts).all():raise ValueError('Nonfinite measured geometry')
    return dict(kind=kind,start=pts[0].tolist(),via=pts[1:-1].tolist(),end=pts[-1].tolist(),
                beziers=curves,heads=heads,width_estimate=None,anchors=anchors.tolist(),geometry_method=method)


def physical_rows(answer,proposals,candidates,gid,defer_errors=False):
    """Explicit model graph -> independent routes/shared networks, no snapping."""
    nodes={n:np.asarray([x,y],float) for n,x,y in answer['nodes']}
    index={r['id']:r for r in candidates};node_use={}
    for route in answer['routes']:
        for n in route['nodes']:node_use[n]=node_use.get(n,0)+1
    rows=[];supports={};issues=[]
    for i,route in enumerate(answer['routes'],1):
        if route['geometry']=='glm':continue
        mids=route['masks'];mask=np.logical_or.reduce([index[m]['mask'] for m in mids]) if mids else None
        anchors=np.array([nodes[n] for n in route['nodes']]);shared=[n for n in route['nodes'] if node_use[n]>1]
        mode=route['geometry']
        if shared or route['kind']=='polyline':
            # Shared coordinates are explicitly authored by GLM; all incident
            # branches preserve the same graph rather than independently moving it.
            mode='points'
        try:
            if route.get('proposal') and not shared and route['kind']!='polyline':
                row=copy.deepcopy(proposals[route['proposal']])
            else:row=fit_route(mask,route['kind'],anchors,route['heads'],mode,route.get('line_style','solid'))
        except (ValueError,np.linalg.LinAlgError,TypeError) as exc:
            if defer_errors:
                issues.append(dict(route=route['id'],stage='mask_measurement',error=str(exc),fallback='deferred_glm'))
                continue
            row=fit_route(None,route['kind'],anchors,route['heads'],'points')
            issues.append(dict(route=route['id'],stage='mask_measurement',error=str(exc),fallback='preserved model anchors'))
        row.update(id=gid+':'+route['id'],heads=route['heads'],line_style=route.get('line_style','solid'),
                   source_masks=mids,source_mask_paths=[index[m]['path'] for m in mids],globals=route['sources'],
                   node_ids=[gid+':'+n for n in route['nodes']],shared_nodes=shared,identity_index=i,
                   coordinate_source='glm_keypoints' if mode=='points' else 'mask_no_skeleton',
                   label=route['id'])
        if 'semantic_heads' in route:row['semantic_heads']=route['semantic_heads']
        if route.get('source_linking'):row['discovery_source_linking']=route['source_linking']
        for end,h in zip(('start','end'),route['heads']):
            row[end+'_head']=h not in ('none','unknown');row[end+'_head_type']=h if h not in ('none','unknown') else None
        rows.append(row);supports[row['id']]=mask
    assign_parents(rows,gid)
    return rows,supports,issues


def assign_parents(rows,gid):
    # A model target may own multiple paths without requiring a shared vertex.
    # Otherwise preserve the existing explicit-node ownership, never proximity.
    pending=list(rows);n=0
    while pending:
        component=[pending.pop(0)];used=set(component[0]['node_ids']);changed=True
        owners={component[0]['physical_target_id']} if component[0].get('physical_target_id') else set()
        while changed:
            changed=False
            for row in list(pending):
                if used.intersection(row['node_ids']) or (row.get('physical_target_id') and row['physical_target_id'] in owners):
                    component.append(row);used.update(row['node_ids']);pending.remove(row);changed=True
                    if row.get('physical_target_id'):owners.add(row['physical_target_id'])
        n+=1
        for row in component:row['parent_id']=gid+f'.T{n}'
