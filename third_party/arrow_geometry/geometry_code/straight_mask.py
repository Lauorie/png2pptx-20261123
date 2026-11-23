"""A single straight centerline measured from ink, without joining dash pieces.

This supplies a numerical proposal only. Curves, branches and multiple targets
retain the graph path; GLM still decides identity and endpoint head types.
"""
import copy
import cv2
import numpy as np


def estimate(mask):
    y,x=np.nonzero(mask)
    if len(x)<12:return None
    ink=np.column_stack([x,y]).astype(float)
    v=cv2.fitLine(ink.astype('float32'),cv2.DIST_HUBER,0,.01,.01).ravel()
    direction=v[:2].astype(float);origin=v[2:].astype(float)
    # Equal-weight cross-section centers prevent a wide head from pulling the
    # fitted shaft sideways. Missing dashes simply supply no observations.
    for _ in range(2):
        normal=np.array([-direction[1],direction[0]])
        u=(ink-origin)@direction;v=(ink-origin)@normal
        bins=np.floor(u/2).astype(int);order=np.argsort(bins);cuts=np.flatnonzero(np.diff(bins[order]))+1
        groups=np.split(order,cuts)
        sections=np.array([[np.median(u[g]),np.median(v[g]),np.ptp(v[g])+1] for g in groups if len(g)>=2])
        if len(sections)<6:return None
        centers=origin+sections[:,0,None]*direction+sections[:,1,None]*normal
        line=cv2.fitLine(centers.astype('float32'),cv2.DIST_HUBER,0,.001,.001).ravel()
        direction=line[:2].astype(float);origin=line[2:].astype(float)
    if direction[np.argmax(abs(direction))]<0:direction=-direction
    normal=np.array([-direction[1],direction[0]])
    u=(ink-origin)@direction;v=(ink-origin)@normal
    lo,hi=np.quantile(u,[.001,.999]);length=hi-lo
    width=float(np.median(sections[:,2]))
    residual=abs((centers-origin)@normal)
    error=float(np.quantile(residual,.98));spread=float(np.quantile(abs(v),.98))
    # These are line-fit validity checks, not semantic merge decisions. If a
    # single line cannot describe the ink, keep the existing graph proposal.
    if length<6*width or error>max(1.5,.3*width) or spread>2.5*width:return None
    if np.mean(abs(v)<max(1.,width*.35))<.25:return None  # two parallel shafts
    start=origin+lo*direction;end=origin+hi*direction
    route=np.linspace(start,end,max(2,int(length)+1))
    return dict(id='E1',kind='straight',start=start.tolist(),via=[],end=end.tolist(),
        beziers=[],route_path=route.tolist(),width_estimate=width,heads=['none','none'],
        start_head=False,end_head=False,start_head_type=None,end_head_type=None,
        coordinate_source='local_sam_mask',geometry_method='direct_straight_mask',
        fit_evidence=dict(center_residual_p98_px=error,mask_transverse_p98_px=spread,
                          mask_pixels=len(ink),native_mask_unchanged=True))


def refit_confirmed(rows,candidates,proposal=None):
    """Reuse a saved semantic decision without reusing stale E/N references.

    Only a single confirmed, unbranched straight target backed by one local
    mask is eligible. Head enums are carried by their spatial endpoints.
    """
    if len(candidates)!=1 or not rows:return rows
    if proposal is not None:
        selected={r['id'].split(':')[-1] for r in rows}
        if selected!={r['id'] for r in proposal['paths']}:return rows
    if len({r.get('identity_index') for r in rows})!=1:return rows
    if len({r.get('line_style') for r in rows})>1:return rows  # preserve explicit style boundaries
    if any(r['kind']!='straight' or r.get('shared_nodes') or r.get('also_owned_by')
           or not r.get('coordinate_source','').startswith('local_sam') for r in rows):return rows
    fitted=estimate(candidates[0]['mask'])
    if fitted is None:return rows
    ends=[(np.asarray(r[e]),r.get('heads',['unknown','unknown'])[i],
           r.get('nodes',[None,None])[i],r.get('node_ids',[None,None])[i])
          for r in rows for i,e in enumerate(('start','end'))]
    a,b=np.asarray(fitted['start']),np.asarray(fitted['end']);direction=(b-a)/np.linalg.norm(b-a)
    order=sorted(range(len(ends)),key=lambda i:float((ends[i][0]-a)@direction))
    outer=[order[0],order[-1]]
    # A head on an interior endpoint is a different semantic structure.
    if any(h not in ('none','unknown') for i,(_,h,_,_) in enumerate(ends) if i not in outer):return rows
    result=copy.deepcopy(rows[0]);result.update(fitted)
    result['id']=rows[0]['id'];result['heads']=[ends[i][1] for i in outer]
    result['nodes']=[ends[i][2] for i in outer];result['node_ids']=[ends[i][3] for i in outer]
    result['refit_from_paths']=[r['id'] for r in rows];result['source_masks']=[candidates[0]['id']]
    for end,h in zip(('start','end'),result['heads']):
        result[end+'_head']=h not in ('none','unknown')
        result[end+'_head_type']=h if h not in ('none','unknown') else None
    result['gaps']=[];result.pop('style',None)
    return [result]
