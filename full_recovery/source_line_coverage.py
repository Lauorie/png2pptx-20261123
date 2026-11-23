"""Deletion guards based on the original visible portion of each native route.

The guard may veto a deletion. It never generates paths or drops an unmatched
arrow because another branch happens to contain more geometry.
"""
import numpy as np
from .handoff import distances


def visible_points(row, rgb):
    p=np.asarray(row['points'],float)
    if len(p)<4:return np.empty((0,2))
    tangent=np.gradient(p,axis=0);normal=np.c_[-tangent[:,1],tangent[:,0]]
    normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-8)
    pigment=row.get('style',{}).get('stroke') or row.get('style',{}).get('color')
    if not isinstance(pigment,str):return np.empty((0,2))
    color=np.array([int(pigment[i:i+2],16) for i in (1,3,5)])
    width=row.get('style',{}).get('width_px') or 1.
    radius=max(2.,min(5.,width+1));offsets=np.arange(-radius,radius+.5,.5)
    xy=np.rint(p[:,None]+normal[:,None]*offsets[None,:,None]).astype(int)
    valid=((xy[:,:,0]>=0)&(xy[:,:,0]<rgb.shape[1])&(xy[:,:,1]>=0)&(xy[:,:,1]<rgb.shape[0])).all(1)
    xy[:,:,0]=np.clip(xy[:,:,0],0,rgb.shape[1]-1);xy[:,:,1]=np.clip(xy[:,:,1],0,rgb.shape[0]-1)
    colors=rgb[xy[:,:,1],xy[:,:,0]].astype(float)
    luminance=colors.mean(2)
    contrast=luminance.max(1)-luminance.min(1)
    bright=color.mean()>235
    ink=(luminance>=luminance.min(1)[:,None]+20 if bright
         else luminance<=luminance.max(1)[:,None]-20)
    index=np.zeros(len(p),int);supported=np.zeros(len(p),bool)
    for i in np.flatnonzero(valid&(contrast>20)):
        bands=np.flatnonzero(np.diff(np.r_[False,ink[i],False].astype(int))).reshape(-1,2)
        # Two visible ridges are ambiguous physical identities. Never snap a
        # deletion proof to an adjacent parallel source connection.
        if len(bands)!=1:continue
        first,last=bands[0]
        if first==0 or last==len(offsets):continue
        index[i]=(first+last-1)//2;supported[i]=True
    supported[:max(1,len(p)//12)]=False;supported[-max(1,len(p)//12):]=False
    return xy[np.arange(len(p)),index][supported].astype(float)


def coverage(points, survivor, rgb, native=None, normal_reference=None):
    if not len(points):return np.zeros(0,bool)
    paint=survivor.get('style',{}).get('stroke') or survivor.get('style',{}).get('color')
    width=survivor.get('style',{}).get('width_px') or 0.
    if not isinstance(paint,str) or width<=0:return np.zeros(len(points),bool)
    color=np.array([int(paint[i:i+2],16) for i in (1,3,5)])
    xy=np.rint(points).astype(int)
    colour_supported=np.linalg.norm(rgb[xy[:,1],xy[:,0]].astype(float)-color,axis=1)<85
    tolerance=max(1.5,min(3.,width/2+1))
    result=colour_supported&(distances(points,survivor.get('paths',[survivor['points']]))<=tolerance)
    if native is not None:
        ref=normal_reference or survivor
        route=np.asarray(ref['points'],float)
        nearest=np.linalg.norm(points[:,None]-route[None],axis=2).argmin(1)
        tangent=np.gradient(route,axis=0)[nearest]
        normal=np.c_[-tangent[:,1],tangent[:,0]]/np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-8)
        sample=np.rint(points[:,None]+normal[:,None]*np.array([-1.,0.,1.])[None,:,None]).astype(int)
        sample[:,:,0]=np.clip(sample[:,:,0],0,native.shape[1]-1);sample[:,:,1]=np.clip(sample[:,:,1],0,native.shape[0]-1)
        visible=(np.linalg.norm(native[sample[:,:,1],sample[:,:,0]].astype(float)-color,axis=2)<85).any(1)
        result&=visible
    return result


def guard(rows, choices, source, native_views=None):
    rgb=np.asarray(source.convert('RGB'));table={r['id']:r for r in rows};cache={};audit=[]
    for choice in choices:
        kept=set(choice.get('keep_ids',[]));survivors=[table[k] for k in kept]
        witnesses={w['candidate_id']:w for w in choice.get('source_replacement_witnesses',[])
                   if isinstance(w,dict) and isinstance(w.get('survivor_ids'),list)
                   and w['survivor_ids'] and all(isinstance(k,str) for k in w['survivor_ids'])
                   and set(w['survivor_ids'])<=kept
                   and w.get('source_object_id')==choice.get('source_object_id')
                   and w.get('candidate_id') in choice['candidate_ids']
                   and w['candidate_id'] not in kept}
        selection=choice.get('source_object_selection',{})
        single_source=(isinstance(choice.get('source_object_id'),str) and bool(choice['source_object_id'])
            and selection.get('source_object_id')==choice['source_object_id']
            and selection.get('mapping_status')=='resolved' and selection.get('count_status')=='exact'
            and type(selection.get('physical_source_count')) is int and selection['physical_source_count']==1
            and isinstance(selection.get('survivor_ids'),list) and bool(kept)
            and all(isinstance(k,str) for k in selection['survivor_ids'])
            and set(selection['survivor_ids'])==kept)
        certificate=choice.get('source_ridge_certificate',{})
        from .source_straight_routes import eligible
        if (certificate.get('certified') and len(kept)==1 and certificate.get('candidate_id') in kept
                and all(eligible(table[key]) for key in choice['candidate_ids'])):
            # Source pixels have already established ONE physical connection.
            # Fit error can change its winning representation, never its count.
            points=np.asarray(certificate['source_visible_points'],float)
            scores={key:float(coverage(points,table[key],rgb,
                native=(native_views or {}).get(table[key]['layer'])).mean())
                for key in choice['candidate_ids']}
            proposed=next(iter(kept))
            best=max(choice['candidate_ids'],key=lambda key:(scores[key],key==proposed))
            choice['keep_ids']=[best]
            choice['source_native_arbitration']={'proposed_keep_ids':[proposed],'effective_keep_ids':[best],
                'source_visible_coverage':scores,'physical_source_count':1}
            if scores[best]<.9:
                audit.append({'candidate_id':best,'source_object_id':choice.get('source_object_id'),
                    'source_visible_samples':len(points),'replacement_coverage':scores[best],
                    'decision':'retain_single_source_representation','deletion_veto':False,
                    'reason':'Physical source identity is unique; remaining native displacement is unresolved'})
            continue
        for ident in choice['candidate_ids']:
            row=table[ident]
            if ident in kept:continue
            if ident not in cache:cache[ident]=visible_points(row,rgb)
            points=cache[ident]
            if choice.get('source_ridge_certificate'):
                # The measured source ridge identifies a displaced fit. Check
                # replacement coverage there, rather than at its wrong pixels.
                points=np.asarray(choice['source_ridge_certificate']['source_visible_points'],float)
            if not survivors:
                if choice.get('source_kind')=='image_content':continue
                if len(points)<max(8,.35*len(row['points'])):continue
                kept.add(ident)
                audit.append({'candidate_id':ident,'source_visible_samples':len(points),
                    'replacement_coverage':0.,'head_content_covered':False,
                    'reason':'source_visible_native_content_requires_a_surviving_representation'})
                continue
            if len(points)<8:
                audit.append({'candidate_id':ident,'source_visible_samples':len(points),
                    'decision':'abstain','deletion_veto':False,
                    'reason':'Insufficient source evidence cannot establish zero replacement coverage'})
                continue
            covered=np.zeros(len(points),bool)
            head_points=[row['points'][0 if i==0 else -1] for i,h in enumerate(row.get('heads',[])) if h not in (None,'none')]
            retained_heads=[s['points'][0 if i==0 else -1] for s in survivors
                            for i,h in enumerate(s.get('heads',[])) if h not in (None,'none')]
            has_heads=all(any(np.linalg.norm(p-q)<=max(4.,.04*row['length']) for q in retained_heads) for p in head_points)
            for survivor in survivors:
                if len(points):covered|=coverage(points,survivor,rgb,
                    native=(native_views or {}).get(survivor['layer']),normal_reference=row)
            fraction=float(covered.mean()) if len(points)>=8 else 0.
            if fraction<.9 or not has_heads:
                if ident in witnesses or single_source:
                    # Original-source mapping already names a replacement of
                    # this SAME physical object. Fit errors remain unresolved;
                    # reinstating its rejected complete copy creates a duplicate.
                    audit.append({'candidate_id':ident,
                        'source_object_id':choice.get('source_object_id'),
                        'source_visible_samples':len(points),
                        'replacement_coverage':fraction,'head_content_covered':has_heads,
                        'decision':'retain_selected_source_representation','deletion_veto':False,
                        'replacement_witness':witnesses.get(ident,selection),
                        'reason':'Same-source replacement is explicit; remaining native fit is unresolved'})
                    continue
                kept.add(ident)
                audit.append({'candidate_id':ident,'source_visible_samples':len(points),
                    'replacement_coverage':fraction,'head_content_covered':has_heads,
                    'reason':'replacement_does_not_cover_original_visible_native_content'})
        choice['keep_ids']=[k for k in choice['candidate_ids'] if k in kept]
    # Shared native candidates may occur in adjacent source scopes. A deletion
    # must be agreed by every user; retaining the whole path takes precedence.
    live={k for c in choices for k in c['keep_ids']}
    for choice in choices:
        choice['keep_ids']=[k for k in choice['candidate_ids'] if k in live]
    return choices,audit
