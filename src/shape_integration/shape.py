"""Global shape candidates -> coarse layout -> visual local review -> ID application."""
import copy
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from PIL import Image
from native_qwen.element_protocol import protocol, validator as global_validator
from png2pptx.frames import scale_box
from png2pptx.images.search import native_crop
from .common import *
from .local_contract import SYSTEM, prompt_for, validator, candidate_overlay, source_to_local
from .scope import SHAPE_SCOPE, SHAPE_SCOPE_VERSION

def validate_plan(raw,size):
    if not isinstance(raw,dict) or set(raw)!={'regions','uncertain'} or not isinstance(raw['regions'],list) or not raw['regions']:raise ValueError('Return nonempty regions and uncertain')
    raw=copy.deepcopy(raw)
    seen=set()
    for r in raw['regions']:
        if not isinstance(r,dict) or set(r)!={'id','bbox_2d','description'} or not isinstance(r['id'],str) or not r['id'] or r['id'] in seen:raise ValueError('Invalid region identity/fields')
        if not all(c.isalnum() or c in '_-' for c in r['id']):raise ValueError('Unsafe region ID')
        # Coarse regions sometimes use the half-open image boundary despite
        # the pixel-center prompt. Convert ONLY that exact outer endpoint;
        # do not clip arbitrary out-of-range boxes or alter fitted geometry.
        box=r['bbox_2d']
        if isinstance(box,list) and len(box)==4:
            for axis in (0,1):
                if box[axis+2]==size[axis]:box[axis+2]=size[axis]-1
        if not valid_box(r['bbox_2d'],size):raise ValueError(f'Invalid region bounds; maximum x={size[0]-1},y={size[1]-1}')
        if not isinstance(r['description'],str):raise ValueError('Description must be text')
        seen.add(r['id'])
    if not isinstance(raw['uncertain'],list) or any(not isinstance(s,str) for s in raw['uncertain']):raise ValueError('uncertain must be strings')
    return raw

def assign(objects,regions,global_size,source_size):
    owners={};unassigned=[]
    for o in objects:
        scores=[intersect(o['bbox_2d'],mapped_box(r['bbox_2d'],global_size,source_size)) for r in regions]
        if max(scores,default=0)<=0:unassigned.append(o['id'])
        else:owners[o['id']]=regions[max(range(len(scores)),key=lambda j:scores[j])]['id']
    return owners,unassigned

def clipped_new_object(box,meta,source_size):
    """An internal crop cut is not an observed whole-object boundary."""
    w,h=meta['actual_crop_input_size_px'];crop=meta['source_crop_half_open']
    return any((box[0]<=1 and crop[0]>0,box[1]<=1 and crop[1]>0,
                box[2]>=w-2 and crop[2]<source_size[0],
                box[3]>=h-2 and crop[3]<source_size[1]))

def apply_review(originals,result,frame,source_size,partial_ids=()):
    """Keep each original ID once; a cropped-off hint cannot be silently approved."""
    known={o['id']:o for o in originals};active=[];history=[];rejected=[];issues=[]
    for review in result['reviews']:
        old=known[review['id']];action=review['action'];reason=review['reason']
        if action in {'keep','redraw'} and old['id'] in partial_ids:
            action='needs_context';reason='Controller: the crop does not show the complete target; retain its registered extent for whole-object fitting. '+reason
        after=copy.deepcopy(old)
        if action=='redraw':
            b=review['bbox_2d'];g=frame.to_global(b[:2])+frame.to_global(b[2:]);after['bbox_2d']=mapped_box(g,frame.global_size,source_size)
        if action=='reject':rejected.append(copy.deepcopy(old));after=None
        else:
            after['review_status']=action;active.append(after)
            if action=='needs_context':issues.append({'id':old['id'],'reason':reason})
        history.append({'id':old['id'],'action':action,'model_action':review['action'],'reason':reason,'before':old,'after':after,'local_bbox':review['bbox_2d']})
    return active,history,rejected,issues

def discover(source_path,input_path,root,runtime):
    """Cache first-pass discovery/layout so SAM can run before local review."""
    root.mkdir(parents=True,exist_ok=True);source=Image.open(source_path).convert('RGB');input_image=Image.open(input_path);gs=list(input_image.size);ss=list(source.size);config=runtime.config;issues=[]
    save(root/'status.json',{'state':'global_and_layout','updated_utc':now()})
    system,p=protocol(gs,'shape_bbox','absolute')
    p.update(scope=SHAPE_SCOPE,shape_scope_version=SHAPE_SCOPE_VERSION)
    p['kind_examples']=['rectangle','rounded_rectangle','ellipse','polygon','compound_symbol','projected_3d']
    p['kind_rule']='kind is a short visual type and a localization hint, not a fitted primitive.'
    p['task']='Locate schematic shapes. Return one JSON object with objects, more_pending and uncertain.'
    def global_call():return runtime.call(root/'calls','global',system,p,[('Clean full figure',input_path)],global_validator(gs,'shape_bbox','absolute'),config['global_max_tokens'])
    template=read(Path(__file__).with_name('layout_prompt.json'));plan_prompt=template['prompt'];plan_prompt['input_size']=gs
    plan_prompt['coordinates']=f'Only this image pixel centers. bbox_2d=[xmin,ymin,xmax,ymax], x=0..{gs[0]-1}, y=0..{gs[1]-1}. Width/height are not valid maximum endpoints.'
    def plan_call():return runtime.call(root/'calls','layout',template['system'],plan_prompt,[('Clean full figure',input_path)],lambda raw:validate_plan(raw,gs),config['plan_max_tokens'])
    with ThreadPoolExecutor(max_workers=2) as pool:
        gjob=pool.submit(global_call);pjob=pool.submit(plan_call)
        raw=gjob.result()
        try:plan=pjob.result()
        except Exception as exc:
            issues.append({'stage':'layout','reason':str(exc),'fallback':'full_image_review'})
            plan={'regions':[{'id':'R1','bbox_2d':[0,0,gs[0]-1,gs[1]-1],'description':'Full image fallback after failed layout'}],'uncertain':[str(exc)]}
    return raw,plan,issues

def run_shape(source_path,input_path,root,runtime):
    root.mkdir(parents=True,exist_ok=True);source=Image.open(source_path).convert('RGB');input_image=Image.open(input_path);gs=list(input_image.size);ss=list(source.size);config=runtime.config
    raw,plan,issues=discover(source_path,input_path,root,runtime)
    objects=[{**o,'id':f'G{i:03d}','global_model_id':o['id'],'bbox_2d':mapped_box(o['bbox_2d'],gs,ss)} for i,o in enumerate(raw['objects'],1)]
    save(root/'global_raw.json',raw);save(root/'global_geometry.json',geometry(objects,ss,raw['uncertain']));save(root/'plan.json',plan)
    overlay(source,objects,root/'global_overlay.png');overlay(source,[{'id':r['id'],'bbox_2d':mapped_box(r['bbox_2d'],gs,ss)} for r in plan['regions']],root/'plan_overlay.png')
    issues.extend({'stage':'global','reason':s} for s in raw['uncertain'])
    if raw['more_pending']:issues.append({'stage':'global','reason':'Model reports more_pending'})
    owners,unassigned=assign(objects,plan['regions'],gs,ss);jobs=[];routing=[]
    for r in plan['regions']:
        folder=root/'regions'/r['id'];folder.mkdir(parents=True,exist_ok=True);b=r['bbox_2d'];dx=(b[2]-b[0])*config['crop_padding'];dy=(b[3]-b[1])*config['crop_padding']
        expanded=[max(0,b[0]-dx),max(0,b[1]-dy),min(gs[0]-1,b[2]+dx),min(gs[1]-1,b[3]+dy)]
        frame,native,meta=native_crop(source,expanded,gs);native.save(folder/'native.png');clean=native.resize(frame.input_size,Image.Resampling.LANCZOS);clean.save(folder/'clean.png');save(folder/'frame.json',meta)
        candidates=[{**o,'role':'assigned' if owners.get(o['id'])==r['id'] else 'context'} for o in objects if owners.get(o['id'])==r['id'] or intersect(o['bbox_2d'],meta['source_crop_half_open'])>0]
        assignment={'region':r,'candidates':candidates,'assigned_ids':[o['id'] for o in candidates if o['role']=='assigned']};save(folder/'assignment.json',assignment)
        partial=[];roundtrip=[]
        for o in candidates:
            q=source_to_local(o['bbox_2d'],frame,ss)
            if o['role']=='assigned' and (q[0]<0 or q[1]<0 or q[2]>frame.input_size[0]-1 or q[3]>frame.input_size[1]-1):partial.append(o['id'])
            back=scale_box(frame.to_global(q[:2])+frame.to_global(q[2:]),gs,ss);roundtrip.extend(abs(a-b) for a,b in zip(o['bbox_2d'],back))
        assert max(roundtrip,default=0)<1e-8
        candidate_overlay(clean,candidates,frame,mapped_box(b,gs,ss),folder/'overlay.png',ss)
        prompt=prompt_for(assignment,meta);save(folder/'prompt.json',{'system':SYSTEM,'prompt':prompt})
        routing.append({'region_id':r['id'],'assigned_ids':assignment['assigned_ids'],'partial_ids':partial,'local_size':list(frame.input_size),'roundtrip_error':max(roundtrip,default=0)})
        jobs.append((r['id'],folder,frame,assignment,prompt,partial))
    save(root/'routing.json',{'owners':owners,'unassigned':unassigned,'regions':routing,'method':'max_intersection_for_routing_only'})
    results={};failures={};save(root/'status.json',{'state':'local_review','regions':len(jobs),'updated_utc':now()})
    def review(job):
        rid,folder,frame,assignment,prompt,partial=job
        images=[('Image 1: clean crop, authoritative pixels',folder/'clean.png'),('Image 2: aligned candidate ID/box visualization',folder/'overlay.png')]
        evidence=getattr(runtime,'spatial_evidence',None)
        if evidence is not None:
            prompt,images=evidence.region(prompt,images,read(folder/'frame.json'),assignment,folder)
        checker=validator(assignment['assigned_ids'],frame.input_size)
        if evidence is not None and hasattr(evidence,'review_validator'):
            checker=evidence.review_validator(checker,folder,assignment['candidates'])
        result=runtime.call(root/'calls','review_'+rid,SYSTEM,prompt,images,checker,config['local_max_tokens'])
        save(folder/'result.json',result);return result
    with ThreadPoolExecutor(max_workers=config['local_workers']) as pool:
        futures={pool.submit(review,j):j for j in jobs}
        for f in as_completed(futures):
            job=futures[f]
            try:results[job[0]]=f.result()
            except Exception as exc:failures[job[0]]=str(exc);save(job[1]/'failure.json',{'error':str(exc)})
    active=[];history=[];rejected=[]
    for rid,folder,frame,assignment,prompt,partial in jobs:
        originals=[o for o in objects if owners.get(o['id'])==rid]
        result=results.get(rid)
        if result is None:
            result={'reviews':[{'id':o['id'],'action':'needs_context','bbox_2d':None,'reason':'Local call failed: '+failures[rid]} for o in originals],'new_objects':[],'uncertain':['Local review failed'],'more_pending':True}
        kept,records,dropped,unresolved=apply_review(originals,result,frame,ss,partial)
        for o in kept:o['region_id']=rid
        for h in records:h['region_id']=rid
        for o in result['new_objects']:
            if clipped_new_object(o['bbox_2d'],read(folder/'frame.json'),ss):
                ident=rid+'_'+o['id']
                reason='New candidate touches an internal review-crop cut; whole extent/ownership is unverified. Deferred to the existing whole-image fitting coverage review.'
                records.append({'id':ident,'region_id':rid,'action':'needs_context',
                                'before':None,'after':None,'proposal':o,'reason':reason})
                issues.append({'stage':'local_review','region_id':rid,'id':ident,'reason':reason})
                continue
            b=o['bbox_2d'];g=frame.to_global(b[:2])+frame.to_global(b[2:]);o={**o,'id':rid+'_'+o['id'],'bbox_2d':mapped_box(g,gs,ss),'region_id':rid,'review_status':'new'}
            kept.append(o);records.append({'id':o['id'],'region_id':rid,'action':'new','before':None,'after':o})
        active.extend(kept);history.extend(records);rejected.extend(dropped);issues.extend({'stage':'local_review','region_id':rid,**x} for x in unresolved)
        issues.extend({'stage':'local_review','region_id':rid,'reason':s} for s in result['uncertain'])
        if result['more_pending']:issues.append({'stage':'local_review','region_id':rid,'reason':'Model reports more_pending'})
        overlay(Image.open(folder/'clean.png'),[{**o,'bbox_2d':source_to_local(o['bbox_2d'],frame,ss)} for o in kept],folder/'after.png')
        compare([folder/'clean.png',folder/'overlay.png',folder/'after.png'],['Clean','Visual old boxes','Applied decisions'],folder/'comparison.png')
    for o in objects:
        if o['id'] in unassigned:
            after={**o,'review_status':'unresolved_no_region'};active.append(after);history.append({'id':o['id'],'action':'needs_context','before':o,'after':after,'reason':'No layout region overlaps old candidate'})
            issues.append({'id':o['id'],'reason':'No responsible layout region'})
    assert len([h for h in history if h['before'] is not None])==len(objects)
    assert len({o['id'] for o in active})==len(active)
    flags=[]
    from .dedup import intersection_over_union
    for i,a in enumerate(active):
        for b in active[i+1:]:
            if a.get('region_id')!=b.get('region_id') and 'new' in (a.get('review_status'),b.get('review_status')):
                score=intersection_over_union(a['bbox_2d'],b['bbox_2d'])
                if score>.7:flags.append({'ids':[a['id'],b['id']],'iou':score,'action':'flag_only'})
    save(root/'history.json',history);save(root/'rejected.json',rejected);save(root/'issues.json',issues);save(root/'cross_region_flags.json',flags)
    final=geometry(active,ss,issues);save(root/'geometry.json',final);overlay(source,active,root/'enhanced_overlay.png')
    compare([root/'global_overlay.png',root/'enhanced_overlay.png'],['Global candidates','After crop review'],root/'comparison.png')
    summary={'global_candidates':len(objects),'regions':len(plan['regions']),'enhanced_candidates':len(active),'actions':dict(Counter(h['action'] for h in history)),'issues':len(issues),'local_failures':failures,'cross_region_flags':len(flags),'all_old_ids_accounted':True}
    save(root/'summary.json',summary);save(root/'status.json',{'state':'complete_with_issues' if issues or failures else 'complete','summary':summary,'updated_utc':now()})
    return final
