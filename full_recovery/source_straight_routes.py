"""Certify individual straight source strokes before semantic bundle review.

Candidates only locate normal strips in the ORIGINAL. Persistent pigment ridges
define physical identity. No geometry or paint is generated or adjusted. Heads,
curves, dash patterns, ambiguous parallel ridges and partial routes abstain.
"""
import numpy as np


def eligible(row):
    from .line_identity import pattern
    if (row.get('closed') or 'segment' in row or row.get('attachment_refs') or
            row.get('compound_stroked_route') or row.get('attached_stroked_head') or
            any(h not in (None,'none') for h in row.get('heads',[])) or
            pattern(row.get('style',{}))!='solid'):
        return False
    p=np.asarray(row['points'],float);axis=p[-1]-p[0];length=np.linalg.norm(axis)
    return bool(length>=16 and np.max(np.abs(np.cross(axis,p-p[0])))/length<.6)


def observe(row,rgb):
    if not eligible(row):return None
    pigment=row['style'].get('stroke') or row['style'].get('color')
    if not isinstance(pigment,str) or len(pigment)!=7:return None
    colour=np.array([int(pigment[i:i+2],16) for i in (1,3,5)])
    p=np.asarray(row['points'],float);ends=p[[0,-1]];axis=ends[1]-ends[0]
    length=np.linalg.norm(axis);normal=np.array([-axis[1],axis[0]])/length
    radius=max(4.,min(12.,.06*length));step=.5
    offsets=np.arange(-radius,radius+step,step);t=np.linspace(.2,.8,96)
    xy=np.rint(ends[0]+t[:,None,None]*axis+offsets[None,:,None]*normal).astype(int)
    valid=((xy[:,:,0]>=0)&(xy[:,:,0]<rgb.shape[1])&(xy[:,:,1]>=0)&(xy[:,:,1]<rgb.shape[0])).all(1)
    if valid.mean()<.9:return None
    xy[:,:,0]=np.clip(xy[:,:,0],0,rgb.shape[1]-1);xy[:,:,1]=np.clip(xy[:,:,1],0,rgb.shape[0]-1)
    pixels=rgb[xy[:,:,1],xy[:,:,0]].astype(float)
    background=np.median(np.concatenate([pixels[:,:3],pixels[:,-3:]],axis=1),axis=1)
    ink=(np.linalg.norm(pixels-colour,axis=2)<85)&(np.linalg.norm(pixels-background[:,None],axis=2)>25)
    centres=np.arange(-radius+1,radius,step);slopes=np.arange(-radius*.65,radius*.65+step,step)
    grid=np.array([(c,s) for c in centres for s in slopes]);u=(t-.5)/.3
    indices=np.rint((grid[:,0,None]+grid[:,1,None]*u+radius)/step).astype(int)
    inside=(indices>=1)&(indices<len(offsets)-1)
    indices=np.clip(indices,0,len(offsets)-1)
    supported=ink[np.arange(len(t))[None],indices]&inside&valid
    scores=supported.mean(1);best=int(np.argmax(scores));centre,slope=grid[best]
    support=supported[best];minimum=min(float(a.mean()) for a in np.array_split(support,3))
    result={'candidate_id':row['id'],'support':float(scores[best]),'minimum_third_support':minimum,
            'certified':False,'search_radius_px':radius}
    if scores[best]<.78 or minimum<.67:return result
    positions=centre+slope*u;bands=[];source_colours=[]
    for i in np.flatnonzero(support):
        intervals=np.flatnonzero(np.diff(np.r_[False,ink[i],False].astype(int))).reshape(-1,2)
        index=indices[best,i]
        band=next(((a,b) for a,b in intervals if a<=index<b),None)
        if band is None:continue
        a,b=band
        if a==0 or b==len(offsets):continue
        bands.append((b-a)*step);source_colours.append(pixels[i,index])
    if len(bands)<48:return result
    width=float(np.median(bands))
    if width>6:return result
    # Distinct persistent normal ridges are separate physical source paths.
    separate=np.abs(grid[:,0]-centre)>max(2.,1.5*width)
    parallel=separate&(np.abs(grid[:,1]-slope)<max(1.,.04*length))&(scores>=.67)
    if np.any(parallel):result['ambiguous_parallel_ridges']=True;return result
    source_ends=ends+normal[:,None].T*(centre+slope*np.array([-.5,.5])[:,None]/.3)
    error=float(np.mean(np.abs(positions)))
    colour_error=float(np.linalg.norm(np.median(source_colours,axis=0)-colour))
    native_width=float(row['style'].get('width_px') or 1.)
    result.update(certified=True,source_ends=source_ends.tolist(),source_width_px=width,
        source_visible_points=xy[np.flatnonzero(support),indices[best,support]].tolist(),
        endpoint_nodes=list(row.get('endpoint_nodes',[])),
        native_position_error_px=error,native_colour_error=colour_error,
        fidelity_cost=error+abs(native_width-width)*.5+colour_error/85,
        native_width_px=native_width)
    return result


def same_source(a,b):
    x=np.asarray(a['source_ends']);y=np.asarray(b['source_ends'])
    if np.linalg.norm(x[0]-y[0])+np.linalg.norm(x[1]-y[1])>np.linalg.norm(x[0]-y[1])+np.linalg.norm(x[1]-y[0]):y=y[::-1]
    v=x[1]-x[0];length=np.linalg.norm(v);unit=v/length;other=y[1]-y[0]
    if float(unit@other)/max(np.linalg.norm(other),1e-8)<.997:return False
    common_pair=(len(a['endpoint_nodes'])==len(b['endpoint_nodes'])==2 and set(a['endpoint_nodes'])==set(b['endpoint_nodes']))
    across=np.abs(np.cross(unit,y-x))
    if across.max()>max(1.,min(a['source_width_px'],b['source_width_px'])/2):return False
    along=np.abs((y-x)@unit)
    if common_pair:
        return bool(min(length,np.linalg.norm(other))/max(length,np.linalg.norm(other))>=.55)
    return bool(along.max()<=min(10.,.08*length+2) and
                min(length,np.linalg.norm(other))/max(length,np.linalg.norm(other))>=.9)


def resolve(rows,source):
    from collections import defaultdict
    from .line_identity import distinct_source_routes
    rgb=np.asarray(source.convert('RGB'));table={r['id']:r for r in rows}
    observations={};choices=[];receipts=[];claimed=set();groups=[]
    for row in rows:
        obs=observe(row,rgb)
        if obs is not None:observations[row['id']]=obs
        if obs and obs['certified']:
            group=next((g for g in groups if all(same_source(obs,observations[k]) for k in g)),None)
            if group is None:groups.append([row['id']])
            else:group.append(row['id'])
    # A shape may incorrectly join node centres while the source joins their
    # ports. Such a fit need not itself land on the source ridge. A unique
    # two-branch endpoint pair can nominate it to the measured source, but
    # never when its own original pixels support a second physical route.
    pairs=defaultdict(list)
    for row in rows:
        pair=row.get('endpoint_nodes',[])
        if len(pair)==2 and row['id'] in observations and eligible(row):pairs[tuple(sorted(pair))].append(row['id'])
    grouped={k:g for g in groups for k in g}
    for pair,ids in pairs.items():
        if len(ids)!=2 or {table[k]['layer'] for k in ids}!={'shape','arrows'}:continue
        certified=[k for k in ids if observations[k]['certified']]
        if len(certified)!=1:continue
        best=certified[0];other=next(k for k in ids if k!=best)
        if (observations[other]['support']>=.35 or len(grouped[best])!=1 or
                distinct_source_routes(rgb,table[best],table[other])):continue
        grouped[best].append(other)
        observations[other]['source_pair_nomination']=list(pair)
    identities=[]
    for group in groups:
        if len(group)<2:continue
        ordered=sorted(group,key=lambda k:(observations[k].get('fidelity_cost',float('inf')),k));best=ordered[0];obs=observations[best]
        if obs['native_position_error_px']>3 or obs['native_colour_error']>85:continue
        # Leave a substantive tie to the source/native review instead of
        # imposing a default winning branch.
        if observations[ordered[1]].get('fidelity_cost',float('inf'))-obs['fidelity_cost']<.15:continue
        identity='source_straight_%04d'%len(identities)
        identities.append({'id':identity,'kind':'line','visible_count':1,'count_status':'exact',**obs,
            'candidate_ids':group,'keep_ids':[best]})
        choices.append({'source_object_id':identity,'source_kind':'line','candidate_ids':group,'keep_ids':[best],
            'source_ridge_certificate':obs,'reason':'One unambiguous persistent original ridge; best complete existing native stroke'})
        claimed.update(group)
        receipts.extend({'candidate_id':k,'classification':'source_certified_straight','source_object_id':identity,
            'removed':k!=best,'observation':observations[k]} for k in group)
    return {'choices':choices,'receipts':receipts,'claimed_ids':claimed,'source_objects':identities,
            'observations':list(observations.values()),'protocol':'source_straight_ridges.2','arrow_geometry_changed':False}
