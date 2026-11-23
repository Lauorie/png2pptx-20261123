"""Suggest local review scopes; never delete or change a source path."""
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from dedup_routes import sampled, owner

def longest_run(mask,step):
    best=now=0
    for value in mask:
        now=now+1 if value else 0;best=max(best,now)
    return max(0,best-1)*step

def sample_info(row,scale,step):
    _,length=sampled(row,2)
    n=min(4097,max(5,int(np.ceil(length*scale/step))+1))
    points,_=sampled(row,n);points=points*scale
    tangent=np.gradient(points,axis=0)
    tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-10)
    return dict(points=points,tangent=tangent,tree=cKDTree(points),length=length*scale,step=length*scale/(n-1))

def near_length(a,b,tolerance,cosine):
    distance,indices=b['tree'].query(a['points'],k=min(4,len(b['points'])))
    if distance.ndim==1:distance=distance[:,None];indices=indices[:,None]
    aligned=np.abs((a['tangent'][:,None,:]*b['tangent'][indices]).sum(axis=2))>=cosine
    good=((distance<=tolerance)&aligned).any(axis=1)
    return longest_run(good,a['step'])

def components(ids,edges):
    neighbors={i:set() for i in ids}
    for a,b in edges:neighbors[a].add(b);neighbors[b].add(a)
    out=[];seen=set()
    for i in ids:
        if i in seen or not neighbors[i]:continue
        todo=[i];group=[]
        while todo:
            j=todo.pop()
            if j in seen:continue
            seen.add(j);group.append(j);todo.extend(neighbors[j]-seen)
        out.append(sorted(group,key=ids.index))
    return out

def propose(rows,size,cfg):
    ids=[r['id'] for r in rows];scale=2048/max(size)
    tolerance=cfg['scope_distance_at_2048'];step=cfg['scope_sample_step_at_2048']
    info={r['id']:sample_info(r,scale,step) for r in rows}
    pairs=[];checks=0
    for i,a_id in enumerate(ids):
        a=info[a_id];alo=a['points'].min(axis=0);ahi=a['points'].max(axis=0)
        for b_id in ids[i+1:]:
            checks+=1;b=info[b_id];blo=b['points'].min(axis=0);bhi=b['points'].max(axis=0)
            if np.any(alo>bhi+tolerance) or np.any(blo>ahi+tolerance):continue
            required=max(5.,min(2.2*tolerance,.55*min(a['length'],b['length'])))
            ab=near_length(a,b,tolerance,cfg['scope_tangent_cosine'])
            ba=near_length(b,a,tolerance,cfg['scope_tangent_cosine'])
            if min(ab,ba)+1e-7>=required:
                pairs.append(dict(a=a_id,b=b_id,span_a=ab,span_b=ba,required=required))
    # Keep attached siblings together so concurrent patches cannot move opposite
    # sides of a shared junction independently. This adds context, not deletion.
    sibling_edges=[]
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            if owner(a)!=owner(b):continue
            ai=info[a['id']];bi=info[b['id']]
            ends_a=np.asarray([a['start'],a['end']])*scale
            ends_b=np.asarray([b['start'],b['end']])*scale
            distance=min(float(bi['tree'].query(ends_a)[0].min()),float(ai['tree'].query(ends_b)[0].min()))
            if distance<=2.5:sibling_edges.append((a['id'],b['id']))
    candidate_ids={i for x in pairs for i in (x['a'],x['b'])}
    groups=[g for g in components(ids,[(x['a'],x['b']) for x in pairs]+sibling_edges) if candidate_ids.intersection(g)]
    selected={i for g in groups for i in g}
    return dict(total_paths=len(rows),pair_checks=checks,candidate_pairs=pairs,sibling_context_edges=sibling_edges,groups=groups,
                untouched_ids=[i for i in ids if i not in selected],
                note='Geometric proximity is only a review trigger; parallel/crossing/serial strokes remain distinct unless source-backed model review decides otherwise.')

