"""Source-backed repair of small repeated primitives; no diagram templates.

Visible fill components establish identity, not geometry. Complete perimeter
measurements supply all exported coordinates. Extra occurrences need a unique
existing family and owner; labels, dots, connectors and hidden shapes abstain.
"""
import copy
import hashlib
import json
import math
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .paths import box_commands,ellipse_commands
from .adapter import bounds,union_bounds


def groups(local,scale):
    eligible=[p for p in local['parts']if p.get('status')=='ok'
        and (p['kind']=='ellipse' or p['kind']=='rectangle' and abs(p['parameters']['angle_radians'])<math.radians(.5)
             and min(np.asarray(p['bbox_2d'])[2:]-np.asarray(p['bbox_2d'])[:2])<24*scale)
        and min(np.asarray(p['bbox_2d'])[2:]-np.asarray(p['bbox_2d'])[:2])>=8*scale]
    families=[]
    for p in sorted(eligible,key=lambda p:np.prod(np.asarray(p['bbox_2d'])[2:]-np.asarray(p['bbox_2d'])[:2])):
        span=np.asarray(p['bbox_2d'])[2:]-np.asarray(p['bbox_2d'])[:2]
        for family in families:
            old=np.median([np.asarray(q['bbox_2d'])[2:]-np.asarray(q['bbox_2d'])[:2]for q in family],axis=0)
            if family[0]['kind']==p['kind'] and np.max(np.maximum(span/old,old/span))<1.5:
                family.append(p);break
        else:families.append([p])
    for family in families:
        typical=np.median([np.asarray(p['bbox_2d'])[2:]-np.asarray(p['bbox_2d'])[:2]for p in family],axis=0)
        columns=[]
        for p in sorted(family,key=lambda p:p['bbox_2d'][0]):
            if columns and p['bbox_2d'][0]-columns[-1][-1]['bbox_2d'][0]<2*typical[0]:columns[-1].append(p)
            else:columns.append([p])
        for column in columns:
            if len(column)>=3:yield column,typical


def register(local,job,clean,*,owned_regions=()):
    from .frame_edges import measure as rectangle
    from .source_ellipses import measure as ellipse
    frame=job['frame'];crop=frame['source_crop_half_open'];size=frame['actual_crop_input_size_px']
    scale=max(size[i]/(crop[i+2]-crop[i])for i in (0,1));rgb=np.asarray(clean.convert('RGB'))
    audits=[];additions=[]
    for group,typical in groups(local,scale):
        declared=local.get('declared_ownership')
        if declared is not None:
            roles=declared.setdefault('retained_roles',{})
            known={roles[p['id']]for p in group if p['id']in roles}
            missing_roles=[p for p in group if p['id']not in roles]
            if (len(known)==1 and known<= {'body','contour','intrinsic_stroke'} and
                    all(p.get('diagnostics',{}).get('source_primitive_registration',{}).get('added')
                        for p in missing_roles)):
                for part in group:
                    measured=part.get('diagnostics',{}).get('source_primitive_registration',{})
                    if part['id']not in roles and measured.get('added'):
                        roles[part['id']]=next(iter(known))
                        if 'uncertain_parts'in declared:
                            declared['uncertain_parts']=[ident for ident in declared['uncertain_parts']
                                if ident!=part['id']]
                        audits.append({'applied':False,'metadata_only':True,
                            'reason':'source_measured_family_role_restored','part_id':part['id'],
                            'inherited_role':roles[part['id']]})
        boxes=np.array([p['bbox_2d']for p in group]);kind=group[0]['kind']
        lo=np.maximum(0,np.floor(boxes[:,:2].min(0)-2*typical)).astype(int)
        hi=np.minimum(clean.size,np.ceil(boxes[:,2:].max(0)+2*typical)).astype(int)
        if np.any(hi<=lo):continue
        tile=rgb[lo[1]:hi[1],lo[0]:hi[0]];pixels=tile.reshape(-1,3)
        colors,counts=np.unique(pixels//8,axis=0,return_counts=True);sets=[]
        for index in np.argsort(counts)[-20:][::-1]:
            pigment=np.median(pixels[(pixels//8==colors[index]).all(1)],axis=0)
            if pigment.min()>245 or pigment.max()<80:continue
            mask=(np.linalg.norm(tile.astype(float)-pigment,axis=2)<12).astype('uint8')
            if kind=='ellipse':
                # Remove thin same-pigment connectors from a filled node's
                # identity locator. Actual perimeter still comes from CLEAN.
                diameter=max(3,int(round(3*scale))|1)
                mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(diameter,diameter)))
            _,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
            comps=[]
            for label,(x,y,w,h,area)in enumerate(stats[1:],1):
                if (.5*typical[0]<=w<=1.5*typical[0] and .5*typical[1]<=h<=1.5*typical[1]
                    and area>=(.65 if kind=='ellipse' else .6)*w*h
                    and x>0 and y>0 and x+w<tile.shape[1] and y+h<tile.shape[0]):
                    comps.append({'box':np.array([x,y,x+w,y+h],float)+np.tile(lo,2)-.5,'label':label})
            if len(comps)<3 or not .75<=len(group)/len(comps)<=4/3:continue
            if any(len(old['components'])==len(comps) and np.max(abs(np.sort([c['box']for c in comps],axis=0)-
                np.sort([c['box']for c in old['components']],axis=0)))<2*scale for old in sets):continue
            sets.append({'components':comps,'pigment':pigment,'labels':labels})
        if len(sets)!=1:
            if sets:audits.append({'applied':False,'reason':'source_primitive_pigment_ambiguous','parts':[p['id']for p in group]})
            continue
        found=sets[0];comps=found['components'];source=np.array([c['box']for c in comps]);pigment=found['pigment']
        centers=(boxes[:,:2]+boxes[:,2:])/2;targets=(source[:,:2]+source[:,2:])/2
        ii,jj=linear_sum_assignment(np.linalg.norm((centers[:,None]-targets[None])/typical,axis=2))
        assigned=dict(zip(ii,jj));ordered=sorted(assigned,key=lambda i:centers[i,1])
        # Omission gaps can separate an end member from the dense part of the
        # same column. Bound identity by that member's nearest source neighbor,
        # rather than letting the smallest gap elsewhere veto the whole family.
        def reach(j):
            gaps=np.abs(targets[:,1]-targets[j,1]);gaps=gaps[gaps>0]
            return max(typical[1],float(gaps.min())/2)if len(gaps)else typical[1]
        if np.any(np.diff(targets[[assigned[i]for i in ordered],1])<=0) or any(
            abs(centers[i,0]-targets[j,0])>2*typical[0] or abs(centers[i,1]-targets[j,1])>reach(j) for i,j in assigned.items()):continue
        unmatched=[i for i in range(len(group))if i not in assigned]
        if unmatched:
            # More predictions than source members is a coverage conflict,
            # not permission to invent members or silently skip registration.
            audits.append({'applied':False,'reason':'source_primitive_count_conflict','kind':kind,
                'parts':[p['id']for p in group],'observed_parts':len(group),'source_components':len(comps),
                'source_boxes':[c['box'].tolist()for c in comps],
                'unmatched_parts':[group[i]['id']for i in unmatched],
                'missing_members_recovered':False})
        measured={}
        for j,c in enumerate(comps):
            if kind=='rectangle':
                b=c['box'];params={'center':((b[:2]+b[2:])/2).tolist(),'width':b[2]-b[0],'height':b[3]-b[1],'angle_radians':0.}
                # The fill locator is independent of the stroke centerlines.
                # Do not skip a complete measurement because this locator is aligned.
                b,a=rectangle(params,clean,c['box'],source_scale=scale,source_identity=True,
                    fill_pigment=pigment,identity_displacement_px=2*max(typical),allow_aligned=True)
                if b is None:continue
                measured[j]={'parameters':{'center':[(b[0]+b[2])/2,(b[1]+b[3])/2],'width':b[2]-b[0],
                    'height':b[3]-b[1],'angle_radians':0.},'source_fill':pigment.tolist(),
                    'source_stroke':a['sides'][0]['source_pigment'],'source_width_px':None,'measurement':a}
            else:
                b=np.array(c['box']);a0=np.maximum(0,np.floor(b[:2]-9*scale)).astype(int)
                a1=np.minimum(clean.size,np.ceil(b[2:]+9*scale)).astype(int)
                component=(found['labels']==c['label']).astype('uint8')
                # A complete isolated fill is measured in a small local tile;
                # labels and connected neighbouring pigments cannot enter it.
                local_mask=np.zeros((a1[1]-a0[1],a1[0]-a0[0]),'uint8')
                x0,y0=np.maximum(a0,lo);x1,y1=np.minimum(a1,hi)
                local_mask[y0-a0[1]:y1-a0[1],x0-a0[0]:x1-a0[0]]=component[y0-lo[1]:y1-lo[1],x0-lo[0]:x1-lo[0]]
                m=ellipse(clean.crop((*a0,*a1)),local_mask,pigment,scale)
                if m is None:continue
                m['parameters']['center']=(np.array(m['parameters']['center'])+a0).tolist();measured[j]=m
        # A missing member is added only after this entire visible family has
        # complete source contours. A partial source fragment stays a conflict.
        complete=len(measured)==len(comps)
        missing=[j for j in range(len(comps))if j not in jj]
        if missing:audits.append({'applied':False,'reason':'source_primitive_count_conflict','kind':kind,
            'parts':[p['id']for p in group],'observed_parts':len(group),'source_components':len(comps),
            'source_boxes':[c['box'].tolist()for c in comps],'complete_perimeters':complete,'missing_members_recovered':False})
        conflict=audits[-1]if missing else None;added_count=0
        for i,j in assigned.items():
            if j not in measured:continue
            part=group[i];m=measured[j];before=copy.deepcopy(part['parameters']);params=m['parameters']
            if max(np.linalg.norm(np.array(before['center'])-params['center']),
                   abs(before['width']-params['width']),abs(before['height']-params['height']))<.35*scale:continue
            part['parameters']=copy.deepcopy(params);update(part)
            a={'applied':True,'reason':'complete_source_primitive_perimeter','part_id':part['id'],'kind':kind,
               'before':before,'after':params,**{k:v for k,v in m.items()if k!='parameters'}}
            part['diagnostics']['source_primitive_registration']=copy.deepcopy(a);audits.append(a)
        if not complete or not missing:continue
        identity=job.get('identity_bbox_local')
        # A fallible target hint can cut a family whose existing members have
        # already been measured outside it. Those owned, complete perimeters
        # establish the family's extent without adopting unrelated source parts.
        matched=source[list(assigned.values())]
        family_extent=np.r_[matched[:,:2].min(0),matched[:,2:].max(0)]
        effective_extent=(np.r_[np.minimum(identity[:2],family_extent[:2]),
            np.maximum(identity[2:],family_extent[2:])]if identity else family_extent)
        for j in missing:
            b=comps[j]['box'];m=measured[j]
            if np.any(b[:2]<effective_extent[:2]-typical)or np.any(b[2:]>effective_extent[2:]+typical):
                audits.append({'applied':False,'reason':'missing_member_outside_measured_family_extent',
                    'source_box':b.tolist(),'family_extent':family_extent.tolist()})
                continue
            if any(np.all(b[:2]>=np.asarray(region)[:2])and np.all(b[2:]<=np.asarray(region)[2:])for region in owned_regions):continue
            # Another registered part cannot be copied as this family's child.
            center=np.array(m['parameters']['center'])
            if any(p['kind']==kind and np.linalg.norm(center-np.array(p['parameters']['center']))<.5*min(typical)
                   for p in [*local['parts'],*additions]if p.get('status')=='ok'):continue
            template=group[int(np.argmin(np.linalg.norm(centers-center,axis=1)))]
            declared=local.get('declared_ownership')
            role=(declared or {}).get('retained_roles',{}).get(template['id'])
            if declared is not None and (role not in ('body','contour','intrinsic_stroke')or
                    any(declared.get('retained_roles',{}).get(p['id'])!=role for p in group)):
                audits.append({'applied':False,'reason':'source_member_family_role_unresolved',
                    'family_part':template['id'],'source_box':b.tolist()})
                continue
            part=copy.deepcopy(template);ident='source_primitive_'+hashlib.sha256(json.dumps([kind,np.round(b/scale,2).tolist()],separators=(',',':')).encode()).hexdigest()[:12]
            if any(p['id']==ident for p in [*local['parts'],*additions]):
                audits.append({'applied':False,'reason':'source_member_identity_conflict',
                    'part_id':ident,'source_box':b.tolist()})
                continue
            part.update(id=ident,key=[*template['key'][:-1],ident],parameters=copy.deepcopy(m['parameters']))
            part['observed_point_ids']=[];part['observed_part']={'id':ident,'kind':kind,'source_measured':True}
            part['diagnostics']={'source_primitive_registration':{'added':True,
                'source_family_part_id':template['id'],
                **({'source_family_role':role}if role is not None else{}),**copy.deepcopy(m)}};update(part)
            additions.append(part);audits.append({'applied':True,'added':True,'reason':'complete_unrepresented_source_primitive',
                'part_id':ident,'kind':kind,'family_parts':[p['id']for p in group],
                'measured_owned_family_extent':family_extent.tolist(),**copy.deepcopy(m)})
            if declared is not None:
                declared.setdefault('retained_roles',{})[ident]=role
            added_count+=1
        if conflict is not None:conflict['missing_members_recovered']=added_count==len(missing)
    for part in additions:
        # Added source measurements use the same valid point/part contract as
        # ordinary observations so a future bounded edit can reuse them.
        p=part['parameters'];theta=np.linspace(0,2*np.pi,8,endpoint=False);a=p['angle_radians']
        rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        if part['kind']=='ellipse':xy=(np.c_[np.cos(theta),np.sin(theta)]*[p['width']/2,p['height']/2])@rot.T+p['center']
        else:
            from .paths import rectangle_corners
            xy=rectangle_corners([*p['center'],p['width'],p['height'],a])
        ids=[part['id']+'_p'+str(i)for i in range(len(xy))]
        observed={'id':part['id'],'kind':part['kind'],'point_ids':ids}
        part.update(observed_point_ids=ids,observed_part=observed)
        if 'observation'in local:
            local['observation']['parts'].append(copy.deepcopy(observed))
            local['observation']['points'].extend({'id':pid,'xy':point.tolist()}for pid,point in zip(ids,xy))
        if 'points'in local:
            from .frames import FitFrame
            frame=FitFrame.from_metadata(local['frame'])
            local['points'].extend({'id':pid,'key':[*part['key'][:-1],pid],'xy_local':point.tolist(),
                'xy_global':frame.to_global(point),'xy_original':frame.to_original(point)}for pid,point in zip(ids,xy))
    local['parts'].extend(additions)
    if any(a.get('applied')for a in audits):
        local['bbox_2d']=union_bounds(local['parts'])
        if 'part_counts'in local:
            local['part_counts'].update(observed=len(local['parts']),succeeded=sum(p['status']=='ok'for p in local['parts']))
    return audits


def update(part):
    # A measured contour must not export stale geometry from a reused seed.
    for name in ('_preserved_original_commands','_preserved_native_parameters_original'):
        part.pop(name,None)
    p=part['parameters'];a=p['angle_radians']
    part['commands']=(box_commands([*p['center'],p['width'],p['height'],a])if part['kind']=='rectangle'else
        ellipse_commands(p['center'],[np.cos(a)*p['width']/2,np.sin(a)*p['width']/2],
                         [-np.sin(a)*p['height']/2,np.cos(a)*p['height']/2]))
    part['bbox_2d']=bounds(part['commands'])
