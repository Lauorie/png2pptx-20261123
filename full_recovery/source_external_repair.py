"""Reinspect external strokes wrongly assigned to an image-owned source.

The new source observation sees only an original-image crop. Existing native
paths are compared afterwards; no geometry, paint or image pixels are edited.
"""
import copy
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, box as footprint


def outside(row,asset):
    points=np.concatenate([row['points'],*[np.asarray(p) for p in row.get('attachment_bounds',[])]])
    low,high=points.min(0),points.max(0);box=np.asarray(asset['source_box_half_open'],float)
    return bool(np.any(low<box[:2]-1) or np.any(high>box[2:]+1))


def incident(row,asset):
    b=np.asarray(asset['source_box_half_open'],float)
    tolerance=max(4.,.05*float(min(b[2:]-b[:2])))
    return LineString(row['points']).distance(footprint(*b))<=tolerance


def repair(inventory,mapping,aliases,assets,images,source,box,scale,folder,runtime,save,*,maximum=2):
    from .source_ownership import validate_inventory,select_mapping,summary,bounds
    table={o['id']:o for o in inventory['objects']};asset_table={a['id']:a for a in assets}
    groups={};protected=set()
    for item in mapping['objects']:
        obj=table[item['id']]
        if obj['kind'] in ('line','arrow','bundle') and item['status']!='uncertain':
            protected.update(item['candidate_ids'])
        if obj['kind']!='image_content' or obj.get('asset_id') not in asset_table:continue
        for ident in item['candidate_ids']:
            row=aliases[ident]
            asset=asset_table[obj['asset_id']]
            if (row['layer']!='shape' or row.get('closed') or not outside(row,asset) or not incident(row,asset)):continue
            key=(obj['asset_id'],row.get('owner'),row.get('parent_key',row['key']))
            groups.setdefault(key,set()).add(ident)
    result=copy.deepcopy(mapping);observations=copy.deepcopy(inventory);audits=[];calls=0
    folder=Path(folder);original=np.asarray(source.convert('RGB'))
    def source_gap_score(group):
        # Intrinsic edge fits may protrude a pixel beyond an image. Actual
        # inter-image connections traverse uncovered source space; prioritize
        # that evidence rather than the arbitrary order of native IDs.
        values=[]
        regions=[np.asarray(a['source_box_half_open'],float) for a in assets]
        for ident in group[1]:
            points=aliases[ident]['points']
            distances=[np.linalg.norm(np.maximum(np.maximum(b[:2]-points,points-b[2:]),0),axis=1) for b in regions]
            values.append(float(np.mean(np.min(distances,axis=0))))
        return max(values,default=0.)
    ordered=sorted(groups.items(),key=lambda group:(-source_gap_score(group),str(group[0])))
    for index,((asset_id,owner,parent),seeds) in enumerate(ordered):
        seeds-=protected
        if not seeds:continue
        if len(audits)>=maximum:
            audits.append({'status':'uncertain','reason':'External source-completion budget exhausted','candidate_ids':sorted(seeds),
                'asset_id':asset_id,'source_gap_score':source_gap_score(((asset_id,owner,parent),seeds))})
            continue
        boxes=np.asarray([bounds(aliases[k]) for k in seeds]);lo=boxes[:,:2].min(0);hi=boxes[:,2:].max(0)
        low=(lo-np.asarray(box[:2])+.5)*scale-.5;high=(hi-np.asarray(box[:2])+.5)*scale-.5
        pad=max(40.,.35*max(high-low))
        directory=folder/('external_%03d'%index);directory.mkdir(parents=True,exist_ok=True)
        with Image.open(images[0][1]) as opened:
            clean=opened.convert('RGB')
        focus=np.r_[np.maximum(0,np.floor(low-pad)),np.minimum(clean.size,np.ceil(high+pad+1))].astype(int)
        if np.any(focus[2:]<=focus[:2]):
            audits.append({'status':'uncertain','candidate_ids':sorted(seeds),'reason':'No visible source inspection region'})
            continue
        factor=min(3.,1200/max(focus[2:]-focus[:2]));size=tuple(max(1,round(v*factor)) for v in focus[2:]-focus[:2])
        detail_scale=np.asarray(size)/(focus[2:]-focus[:2]);source_view=directory/'source.png'
        clean.crop(tuple(focus)).resize(size,Image.Resampling.LANCZOS).save(source_view)
        local_box=np.r_[np.asarray(box[:2])+focus[:2]/scale,np.asarray(box[:2])+focus[2:]/scale]
        local_scale=np.asarray(scale)*detail_scale
        save(directory/'frame.json',{'parent_clean':str(images[0][1]),'parent_clean_size':list(clean.size),
            'focus_in_parent_clean_half_open':focus.tolist(),'detail_size':list(size),
            'detail_scale':detail_scale.tolist(),'original_source_box':local_box.tolist(),
            'original_to_detail_scale':local_scale.tolist(),'new_geometry_generated':False})
        context=[]
        for asset in assets:
            b=(np.asarray(asset['source_box_half_open'])-np.tile(box[:2],2))*np.tile(scale,2)-np.tile(focus[:2],2)
            b*=np.tile(detail_scale,2)
            context.append({'id':asset['id'],'bbox':b.tolist()})
        prompt={'protocol':'source_external_connection_completion.1',
            'task':'Identify actual external SOURCE connections in this original-only local crop.',
            'coordinate_frame':{'size':list(size),'convention':'inclusive local pixel centers'},
            'image_owned_regions':context,
            'already_observed_sources':[{'id':obj['id'],'kind':obj['kind'],'description':obj['description'],
                'bbox':((np.asarray(obj['bbox'])-np.tile(focus[:2],2))*np.tile(detail_scale,2)).tolist()}
                for obj in inventory['objects'] if obj['kind'] in ('line','arrow','bundle') and
                    not (np.any(np.asarray(obj['bbox'])[:2]>focus[2:]) or np.any(np.asarray(obj['bbox'])[2:]<focus[:2]))],
            'rules':[
                'This is an inspection region, not proof that a new path exists. Read original pixels only; no recovered geometry is shown.',
                'Image interiors are context. Connections that extend OUTSIDE image-owned regions are separate physical sources, even when attached to an image boundary.',
                'Already observed source paths are context too. Define only missing source identities, not a second identity for an existing physical path.',
                'Return each visible external physical connection as its own line/arrow ID with visible_count=1. Distinguish curved and straight routes and their actual source attachments.',
                'Do not classify the whole crop as image content, duplicate intrinsic image strokes, infer symmetry, invent an arrow, or infer extra connections behind an opaque object.',
                'If the original does not establish an external path, return no objects and explain uncertainty.'],
            'output':{'objects':[{'id':'S1','kind':'line|arrow|uncertain','bbox':[0,0,size[0]-1,size[1]-1],
                'visible_count':1,'count_status':'exact','description':'actual visible original external path and attachments'}],
                'uncertain':[]}}
        def validate(value):
            value=validate_inventory(value,size,set(),allow_partial=True)
            if any(o['kind'] not in ('line','arrow','uncertain') for o in value['objects']):
                raise ValueError('External completion names individual original paths, not whole image regions or inferred networks')
            return value
        observer=runtime.for_stage(reasoning_effort='low') if hasattr(runtime,'for_stage') else runtime
        try:
            calls+=1
            extra=observer.call(directory/'source_calls','source_completion',
                'Observe SOURCE before recovery; return only JSON.',prompt,[('CLEAN original only',source_view)],validate,8192)
            if not extra['objects']:
                audits.append({'status':'uncertain','candidate_ids':sorted(seeds),'reason':'No independently observed external source identity'})
                continue
            eligible=set(seeds)
            unresolved={u['candidate_id'] for u in result['unassigned'] if u['classification']=='uncertain'}
            for item in result['objects']:
                obj=table.get(item['id'],{})
                if obj.get('kind')=='image_content' and obj.get('asset_id') in asset_table:
                    unresolved.update(k for k in item['candidate_ids'] if outside(aliases[k],asset_table[obj['asset_id']]))
            for ident in unresolved-protected:
                b=(bounds(aliases[ident])-np.tile(box[:2],2))*np.tile(scale,2)
                if not (np.any(b[:2]>focus[2:]) or np.any(b[2:]<focus[:2])):eligible.add(ident)
            # A local observation cannot authorize deletion of an entire
            # native object whose unseen extent leaves this inspection view.
            def wholly_visible(ident):
                b=(bounds(aliases[ident])-np.tile(box[:2],2))*np.tile(scale,2)
                return bool(np.all(b[:2]>=focus[:2]-1) and np.all(b[2:]<=focus[2:]+1))
            local_aliases={k:aliases[k] for k in aliases if k in eligible and wholly_visible(k)}
            if not local_aliases:
                audits.append({'status':'uncertain','candidate_ids':sorted(seeds),
                    'reason':'No whole native recovery is visible in this source inspection window'})
                continue
            detail_images=[('Frozen CLEAN original source',source_view)]
            for label,path in images[1:]:
                if Path(path).stem.endswith('_ids'):continue
                target=directory/(Path(path).stem+'.png')
                with Image.open(path) as opened:
                    native=opened.convert('RGB').crop(tuple(focus)).resize(size,Image.Resampling.LANCZOS)
                    native.save(target)
                detail_images.append((label,target))
                located=native.copy();draw=ImageDraw.Draw(located)
                branch='shape' if 'SHAPE' in label.upper() else 'arrows'
                for ident,row in local_aliases.items():
                    if row['layer']!=branch:continue
                    point=(row['points'][len(row['points'])//2]-local_box[:2]+.5)*local_scale-.5
                    draw.text(tuple(point),ident,fill='#A00090',stroke_width=1,stroke_fill='white')
                located_path=directory/(Path(path).stem+'_local_ids.png');located.save(located_path)
                detail_images.append((label+'; only locally eligible candidate ID labels',located_path))
            candidates=[summary(row,key,local_box,local_scale) for key,row in local_aliases.items()]
            for candidate in candidates:
                row=local_aliases[candidate['id']];colour=row['style'].get('stroke',row['style'].get('color'))
                if not isinstance(colour,str) or len(colour)!=7:continue
                pigment=np.array([int(colour[i:i+2],16) for i in (1,3,5)])
                supported=[]
                for point in row['points']:
                    if any(np.all(point>=np.asarray(a['source_box_half_open'])[:2]) and
                           np.all(point<=np.asarray(a['source_box_half_open'])[2:]) for a in assets):continue
                    x,y=np.rint(point).astype(int)
                    if not (0<=x<original.shape[1] and 0<=y<original.shape[0]):continue
                    tile=original[max(0,y-2):y+3,max(0,x-2):x+3]
                    if tile.size:supported.append(bool(np.any(np.linalg.norm(tile.astype(float)-pigment,axis=2)<55)))
                candidate['original_external_stroke_support']={'samples':len(supported),
                    'near_colour_fraction':float(np.mean(supported)) if supported else None,
                    'note':'Quality hint only; occlusion and source identity still need visual inspection'}
            matching={'protocol':'source_external_connection_completion.1','task':'Select existing recoveries of these newly frozen external SOURCE paths.',
                'source_objects':extra['objects'],'candidates':candidates,
                'rules':['These source identities were observed from ORIGINAL before native paths were shown.',
                    'Compare actual original curvature, endpoints and visible complete extent. Keep one recovery of a physical shaft; complementary head/shaft pieces may coexist.',
                    'External paths cannot be consumed by an image. Do not edit geometry, paint, heads, image pixels, or unrelated routes.',
                    'Only listed native IDs may be assigned. Preserve uncertain content. If no recovery is usable, record incomplete rather than inventing one.'],
                'output':{'objects':[{'id':'S1','candidate_ids':list(local_aliases)[:2],
                    'keep_ids':list(local_aliases)[:1],'status':'resolved|incomplete|uncertain','reason':'original path and actual native quality'}],
                    'unassigned':[{'candidate_id':next(iter(local_aliases)),'classification':'context|unsupported|uncertain','reason':'source evidence'}]}}
            selected,n=select_mapping(observer,directory,matching,detail_images,extra['objects'],local_aliases,{},local_box,local_scale)
            calls+=n
            kind={o['id']:o['kind'] for o in extra['objects']}
            accepted=[row for row in selected['objects'] if row['status']!='uncertain' and row['keep_ids'] and kind[row['id']] in ('line','arrow')]
            reassigned={k for row in accepted for k in row['candidate_ids']}
            protected.update(reassigned)
            for old in result['objects']:
                old['candidate_ids']=[k for k in old['candidate_ids'] if k not in reassigned]
                old['keep_ids']=[k for k in old['keep_ids'] if k not in reassigned]
            result['unassigned']=[u for u in result['unassigned'] if u['candidate_id'] not in reassigned]
            for row in accepted:
                ident='external_%03d/'%index+row['id']
                while any(o['id']==ident for o in observations['objects']):ident+='r'
                result['objects'].append({**row,'id':ident,
                    'review_status':'external_source_identity_completed'})
                obj=next(o for o in extra['objects'] if o['id']==row['id']);b=np.asarray(obj['bbox'])/np.tile(detail_scale,2)+np.tile(focus[:2],2)
                observations['objects'].append({**obj,'id':ident,'bbox':b.tolist(),
                    'source_observation':'original_only_external_completion'})
            audits.append({'status':'source_completion_observed','candidate_ids':sorted(seeds),'asset_id':asset_id,
                'source_gap_score':source_gap_score(((asset_id,owner,parent),seeds)),
                'reassigned_ids':sorted(reassigned),'source_objects':len(accepted)})
        except Exception as error:
            audits.append({'status':'uncertain','candidate_ids':sorted(seeds),'error':str(error),'error_type':type(error).__name__})
    save(folder/'external_source_completion.json',audits)
    return observations,result,calls
