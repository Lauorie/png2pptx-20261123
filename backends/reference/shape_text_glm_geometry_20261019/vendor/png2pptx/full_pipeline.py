"""All-candidate shape refinement, independent local checks and model-selected discovery."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
from pathlib import Path
import shutil
import threading
import traceback
import httpx
from PIL import Image
from . import full_prompts as prompts
from .api.stream_client import StreamingRelayClient
from .frames import Crop, scale_box
from .pipeline import prepare_image, digest
from .render import candidate_overlay, local_overlay, global_overlay, export_scene
from .scene import save, now, apply_refinement


def inventory(scene, crop=None):
    result=[]
    for o in scene['objects'].values():
        parts=[]
        if o.get('export_enabled',True):
            for p in o.get('fit',{}).get('parts',[]):
                xy=[(c[i],c[i+1]) for c in p['commands'] for i in range(1,len(c),2)]
                if xy:
                    xs,ys=zip(*xy);box=scale_box([min(xs),min(ys),max(xs),max(ys)],scene['image']['source_size'],scene['image']['input_size'])
                    part={'id':p['id'],'kind':p['kind'],'bounding_hint_global_pixels':box}
                    if crop:part['bounding_hint_crop_pixels']=crop.to_local(box[:2])+crop.to_local(box[2:])
                    parts.append(part)
        item={k:o.get(k) for k in ['id','label','proposed_type','bbox_global_pixels','status','duplicate_of','represented_by','parent_id']}
        item.update(active_fitted_parts=parts,export_enabled=o.get('export_enabled',False))
        if crop:item['bbox_crop_pixels']=crop.to_local(o['bbox_global_pixels'][:2])+crop.to_local(o['bbox_global_pixels'][2:])
        result.append(item)
    return result


def check_ownership(objects, ident, duplicate_of, represented_by):
    graph={i:([o['duplicate_of']] if o.get('duplicate_of') else [])+o.get('represented_by',[]) for i,o in objects.items()}
    graph[ident]=([duplicate_of] if duplicate_of else [])+represented_by
    visiting=set();done=set()
    def visit(i):
        if i not in graph:raise ValueError('Unknown ownership reference '+i)
        if i in visiting:raise ValueError('Cyclic duplicate/child ownership at '+i)
        if i in done:return
        visiting.add(i)
        for child in graph[i]:visit(child)
        visiting.remove(i);done.add(i)
    for i in graph:visit(i)


def apply_full_refinement(scene, ident, raw, crop, provenance):
    check_ownership(scene['objects'],ident,raw['duplicate_of'],raw['represented_by'])
    if raw['decision'] in ('duplicate','represented_by_children'):
        o=scene['objects'][ident]
        o.update(status=raw['decision'],export_enabled=False)
        o['history'].append({'utc':now(),'stage':'identity_resolution','provenance':provenance,**raw})
    else:
        apply_refinement(scene,ident,{k:v for k,v in raw.items() if k not in ('duplicate_of','represented_by')},crop,provenance)
        o=scene['objects'][ident]
        if raw['decision']=='refined':o['export_enabled']=True
        if raw['decision']=='reject':o['export_enabled']=False
    o.update(duplicate_of=raw['duplicate_of'],represented_by=raw['represented_by'])


def ownership_state(scene):
    memo={}
    def covered(i, stack=()):
        if i in memo:return memo[i]
        if i in stack:return False
        o=scene['objects'][i]
        refs=([o['duplicate_of']] if o.get('duplicate_of') else [])+o.get('represented_by',[])
        own=bool(o.get('fit') and o.get('export_enabled',False))
        result=(own or bool(refs)) and all(covered(r,stack+(i,)) for r in refs)
        memo[i]=result;return result
    for i,o in scene['objects'].items():o['has_geometry_or_resolved_owner']=covered(i)


def full_summary(scene):
    ownership_state(scene);objects=list(scene['objects'].values());active=[o for o in objects if o.get('fit') and o.get('export_enabled',False)]
    initial=[o for o in objects if o['origin']=='seed'];checks=scene.get('coverage_checks',[])
    return {'case':scene['image']['id'],'initial_candidates':len(initial),'discovered_candidates':len(objects)-len(initial),
        'candidates':len(objects),'attempted_initial':sum(o.get('attempted',False) for o in initial),
        'attempted_total':sum(o.get('attempted',False) for o in objects),'unattempted_ids':[o['id'] for o in objects if not o.get('attempted')],
        'active_fitted_objects':len(active),'active_fitted_parts':sum(len(o['fit']['parts']) for o in active),
        'accepted_objects':sum(o['status']=='accepted_by_model' for o in active),
        'status_counts':dict(Counter(o['status'] for o in objects)),
        'unresolved_ownership_ids':[o['id'] for o in objects if (o.get('duplicate_of') or o.get('represented_by')) and not o['has_geometry_or_resolved_owner']],
        'coverage_checks':len(checks),'last_coverage_assessment':checks[-1]['coverage_assessment'] if checks else None,
        'pending_search_regions':sum(r['status'] in ('pending','failed','depth_budget_exhausted','coverage_budget_exhausted','nonshrinking_subregion') for r in scene.get('search_regions',[])),
        'completion_boundary':'Every candidate attempted is a controller coverage check, not measured visual recall.'}


class FullImageRunner:
    # One thread owns each scene. Network concurrency is across images, so scene commits are serialized.
    def __init__(self,scene,root,clients,args):
        self.scene=scene;self.root=root;self.clients=clients;self.args=args;self.case=scene['image']['id']
        self.folder=root/'results'/self.case;self.folder.mkdir(parents=True,exist_ok=True)
        self.image=Image.open(root/scene['image']['input']).convert('RGB')
        self.scene_path=self.folder/'scene.json'
        self.segment=getattr(args,'segment',None)
        if getattr(args,'resume',False):
            scene.setdefault('continuations',[]).append({'segment':self.segment,'utc':now(),
                'previous_run_state':scene.get('run_state'),'previous_summary':full_summary(scene),
                'crop_scale':args.crop_scale,'crop_max_edge':args.crop_max_edge})
            scene['run_state']='running'
        else:
            scene.update(scope='Full candidate shape experiment; separate geometry checks and model-selected discovery; other categories pending',
                         coverage_checks=[],search_regions=[],crop_plans=[],run_state='running')
            for o in scene['objects'].values():o.update(selected=True,status='required_pending',origin='seed',attempted=False,export_enabled=False)
        self.checkpoint()

    def make_crop(self,box):
        crop=Crop.from_model(box,self.image.size)
        factor=getattr(self.args,'crop_scale',None)
        return crop.scaled(factor,getattr(self.args,'crop_max_edge',2048)) if factor is not None else crop

    def call_tag(self,tag):
        return f'{self.case}/segments/{self.segment}/{tag}' if self.segment else f'{self.case}/{tag}'

    def request_path(self,tag):return 'calls/'+self.call_tag(tag)+'/request.json'

    def checkpoint(self,export=False):
        result=full_summary(self.scene);save(self.scene_path,self.scene);save(self.folder/'summary.json',result)
        if export:
            try:
                result['export_counts']=export_scene(self.scene,self.folder,self.image)
                save(self.folder/'summary.json',result)
            except Exception as exc:self.error('export',exc)
        return result

    def error(self,stage,exc,ident=None):
        event={'type':'execution_error','stage':stage,'object_id':ident,'error':str(exc),'utc':now()}
        self.scene['issues'].append(event)
        save(self.folder/'errors'/(stage.replace('/','_')+'.json'),{**event,'traceback':traceback.format_exc()})
        print(f'{self.case} {stage}: ERROR {str(exc)[:200]}',flush=True)

    def call(self,stage,tag,prompt,images,validator):
        def normalized_result(raw):
            return validator(prompts.round_coordinates(validator(raw)))
        return self.clients[stage].call(self.call_tag(tag),prompts.SYSTEM,prompts.coordinate_prompt(prompt),images,normalized_result)

    def plan(self,ids,round_name):
        crops={};all_items=inventory(self.scene)
        if getattr(self.args,'resume',False):
            for previous in self.scene['crop_plans']:
                for region in previous['regions']:
                    for ident in region['target_ids']:
                        if ident in ids:crops[ident]=self.make_crop(region['bbox_2d'])
            if crops:self.scene['issues'].append({'type':'reused_crop_proposals','segment':self.segment,'target_ids':list(crops),
                'note':'Existing model-selected global boxes reused; actual crop resize uses current policy.'})
        missing=[ident for ident in ids if ident not in crops]
        for start in range(0,len(missing),self.args.plan_batch_size):
            batch_ids=missing[start:start+self.args.plan_batch_size]
            targets=[i for i in all_items if i['id'] in batch_ids]
            tag=f'{round_name}/plan/batch_{start//self.args.plan_batch_size:03d}'
            try:
                raw=self.call('plan',tag,prompts.full_plan(self.image.size,targets,all_items,getattr(self.args,'crop_scale',None),getattr(self.args,'crop_max_edge',2048)),
                    [('Clean global input; crop coordinates use this frame',self.root/self.scene['image']['input'])],
                    lambda r:prompts.check_full_plan(r,targets,self.image.size))
                self.scene['crop_plans'].append({'tag':tag,**raw})
                for region in raw['regions']:
                    for ident in region['target_ids']:crops[ident]=self.make_crop(region['bbox_2d'])
            except Exception as exc:
                self.error(tag,exc)
                # Safe fallback still attempts every target. The model may request a better crop itself.
                for ident in batch_ids:crops[ident]=self.make_crop([0,0,self.image.width-1,self.image.height-1])
                self.scene['issues'].append({'type':'plan_fallback','target_ids':batch_ids,'fallback':'full clean image; target may request model-selected context'})
            self.checkpoint()
        if set(crops)!=set(ids):raise AssertionError('Planning dropped required targets')
        return crops

    def candidate_context(self,ident,crop):
        o=self.scene['objects'][ident];box=o['bbox_global_pixels']
        local_box=crop.to_local(box[:2])+crop.to_local(box[2:])
        candidate={'id':ident,'label':o['label'],'proposed_type':o['proposed_type'],'bbox_in_crop_input_pixels':local_box,
            'bbox_in_global_input_pixels':box,'bbox_warning':'Fallible identity hint, possibly outside crop; clean pixels establish geometry.'}
        peers=[i for i in inventory(self.scene,crop) if i['id']!=ident]
        return candidate,peers

    def check_object(self,ident,crop,folder,tag_base):
        """Transport/schema retries stay in the check stage and preserve fitted geometry."""
        o=self.scene['objects'][ident];candidate,peers=self.candidate_context(ident,crop)
        frame=crop.metadata(self.scene['image']['source_size'])
        for retry in range(self.args.error_retries+1):
            tag=f'{tag_base}/retry_{retry:02d}'
            try:
                review=self.call('check',tag,prompts.local_check(ident,frame,o['shape_type'],o.get('represented_by',[]),candidate,
                    [i for i in peers if i['id'] in o.get('represented_by',[])]),
                    [('Image 1: clean local crop',folder/'input.png'),('Image 2: fitted local geometry',folder/'geometry_overlay.png')],
                    lambda r:prompts.validate(r,prompts.local_check_schema(ident)))
                o['history'].append({'utc':now(),'stage':'local_geometry_check','request':self.request_path(tag),**review})
                o['status']={'accept':'accepted_by_model','revise':'needs_revision','reject':'rejected_by_model'}[review['decision']]
                if review['decision']=='reject':o['export_enabled']=False
                self.checkpoint();return review
            except Exception as exc:
                self.error(tag,exc,ident)
                o['history'].append({'utc':now(),'stage':'local_geometry_check_error','request':self.request_path(tag),'error':str(exc)})
                self.checkpoint()
        o['status']='failed_with_diagnostic_fit';self.checkpoint();return None

    def recheck_existing(self,ident):
        o=self.scene['objects'][ident];frame=self.scene['points'][o['active_point_ids'][0]]['provenance']['crop']
        box=frame['crop_bounds_global_half_open']
        crop=self.make_crop([box[0],box[1],box[2]-1,box[3]-1])
        folder=self.folder/'objects'/ident/f'recheck_{self.segment}';folder.mkdir(parents=True)
        clean=crop.image(self.image);clean.save(folder/'input.png')
        save(folder/'frame.json',crop.metadata(self.scene['image']['source_size']))
        local_overlay(self.scene,ident,crop,clean).save(folder/'geometry_overlay.png')
        print(f'{self.case} {ident}: recheck existing geometry only',flush=True)
        review=self.check_object(ident,crop,folder,f'local_check/{ident}/resume_existing')
        if review and review['decision']=='revise':self.refine(ident,crop,initial_feedback=review)

    def refine(self,ident,crop,initial_feedback=None):
        o=self.scene['objects'][ident];o.update(attempted=True,status='processing');self.checkpoint()
        feedback=initial_feedback;contexts=0;corrections=0;errors=0
        old_attempts=[int(p.name.split('_')[-1]) for p in (self.folder/'objects'/ident).glob('attempt_*') if p.is_dir() and p.name.split('_')[-1].isdigit()]
        attempt=max(old_attempts,default=-1)+1
        while True:
            current=attempt;attempt+=1;folder=self.folder/'objects'/ident/f'attempt_{current:02d}';folder.mkdir(parents=True)
            tag=f'refine/{ident}/attempt_{current:02d}'
            try:
                clean=crop.image(self.image);clean.save(folder/'input.png')
                candidate,peers=self.candidate_context(ident,crop);other_ids=[i['id'] for i in peers]
                candidate_overlay(clean,candidate['bbox_in_crop_input_pixels'],ident).save(folder/'candidate_overlay.png')
                frame=crop.metadata(self.scene['image']['source_size']);save(folder/'frame.json',frame)
                raw=self.call('refine',tag,prompts.full_refine(candidate,frame,peers,feedback),
                    [('Image 1: clean crop; output local points',folder/'input.png'),
                     ('Image 2: approximate target hint, same crop alignment',folder/'candidate_overlay.png')],
                    lambda r:prompts.check_full_refine(r,ident,crop,other_ids))
                provenance={'request':self.request_path(tag),'crop':frame,'source_sha256':digest(folder/'input.png'),
                    'response_sha256':digest(self.clients['refine'].root/self.call_tag(tag)/'response_0.json')}
                apply_full_refinement(self.scene,ident,raw,crop,provenance);o=self.scene['objects'][ident]
                self.checkpoint();print(f'{self.case} {ident} attempt {current}: {raw["decision"]}',flush=True)
                if raw['decision']=='needs_context':
                    if contexts>=self.args.context_rounds:o['status']='context_budget_exhausted';break
                    next_crop=self.make_crop(raw['suggested_global_crop'])
                    if next_crop.box==crop.box:o['status']='unresolved_repeated_context';break
                    contexts+=1;crop=next_crop;feedback={'reason':raw['reason'],'uncertain':raw['uncertain']};continue
                if raw['decision']!='refined':break
                save(folder/'fitted_object.json',o)
                local_overlay(self.scene,ident,crop,clean).save(folder/'geometry_overlay.png')
                review=self.check_object(ident,crop,folder,f'local_check/{ident}/attempt_{current:02d}')
                if review is None or review['decision']!='revise' or corrections>=self.args.correction_rounds:break
                corrections+=1;feedback=review;self.checkpoint();continue
            except Exception as exc:
                self.error(tag,exc,ident);o=self.scene['objects'][ident]
                if errors<self.args.error_retries:
                    errors+=1;feedback={'execution_error':str(exc),'instruction':'Return a complete valid response; resolve the stated contract/ownership issue from supplied pixels.'};self.checkpoint();continue
                o['status']='failed_with_diagnostic_fit' if o.get('fit') and o.get('export_enabled') else 'failed';break
        self.checkpoint()

    def process_queue(self,ids,round_name):
        crops=self.plan(ids,round_name)
        for index,ident in enumerate(ids):
            try:self.refine(ident,crops[ident])
            except Exception as exc:
                self.error('object_fatal_'+ident,exc,ident);self.scene['objects'][ident].update(attempted=True,status='failed')
            if (index+1)%self.args.export_every==0:self.checkpoint(export=True)
        self.checkpoint(export=True)

    def discover(self,record,depth):
        if depth>self.args.discovery_depth:record['status']='depth_budget_exhausted';return []
        crop=self.make_crop(record['bbox_global_pixels'])
        tag=f'discovery/{record["id"]}';folder=self.folder/'discovery'/(self.segment or 'initial')/record['id'];folder.mkdir(parents=True)
        clean=crop.image(self.image);clean.save(folder/'input.png');frame=crop.metadata(self.scene['image']['source_size']);save(folder/'frame.json',frame)
        try:
            existing=list(self.scene['objects']);raw=self.call('discovery',tag,prompts.discovery(frame,inventory(self.scene,crop),record['reason']),
                [('Clean crop chosen by coverage/discovery model',folder/'input.png')],lambda r:prompts.check_discovery(r,crop,existing))
            record.update(status='searched',request=self.request_path(tag),uncertain=raw['uncertain'],matches=[])
            ids=[]
            for item in raw['objects']:
                if item['matches_existing'] is not None:
                    record['matches'].append(item);continue
                ident=f'S{len(self.scene["objects"])+1:04d}';box=crop.to_global(item['bbox_2d'][:2])+crop.to_global(item['bbox_2d'][2:])
                # Pixel center resize can place an edge-center fractionally beyond the global center interval.
                box=[max(0,min(v,self.image.size[i%2]-1)) for i,v in enumerate(box)]
                Crop.from_model(box,self.image.size)
                self.scene['objects'][ident]={'id':ident,'label':item['label'],'proposed_type':item['shape_type'],'bbox_global_pixels':box,
                    'bbox_original_pixels':scale_box(box,self.image.size,self.scene['image']['source_size']),
                    'category':'shape','origin':'discovered','parent_id':item['parent_id'],'status':'required_pending',
                    'selected':True,'attempted':False,'export_enabled':False,'history':[],
                    'provenance':{'request':record['request'],'local_id':item['local_id'],'bbox_crop_pixels':item['bbox_2d'],'frame':frame,'reason':item['reason']}}
                ids.append(ident)
            record['new_candidate_ids']=ids.copy();self.checkpoint()
            for sub in raw['subregions']:
                box=crop.to_global(sub['bbox_2d'][:2])+crop.to_global(sub['bbox_2d'][2:])
                box=[max(0,min(v,self.image.size[i%2]-1)) for i,v in enumerate(box)]
                child=self.add_region(box,sub['reason'],record['id'])
                if Crop.from_model(box,self.image.size).box==crop.box:child['status']='nonshrinking_subregion'
                else:ids+=self.discover(child,depth+1)
            self.checkpoint();return ids
        except Exception as exc:
            record.update(status='failed',error=str(exc));self.error(tag,exc);self.checkpoint()
            # Objects committed before a later subregion failure must still be processed.
            return [i for i in record.get('new_candidate_ids',[]) if not self.scene['objects'][i]['attempted']]

    def add_region(self,box,reason,parent=None):
        record={'id':f'R{len(self.scene["search_regions"])+1:04d}','bbox_global_pixels':box,'reason':reason,'parent_region':parent,'status':'pending'}
        self.scene['search_regions'].append(record);return record

    def apply_duplicate(self,pair,tag):
        keep,other=pair['keep'],pair['other'];target=self.scene['objects'][keep];obj=self.scene['objects'][other]
        event={'type':'duplicate_candidate_proposal','request':tag,**pair}
        try:
            check_ownership(self.scene['objects'],other,keep,[])
            # Never suppress surviving geometry in favor of an owner with no geometry yet.
            ownership_state(self.scene)
            if not target['has_geometry_or_resolved_owner']:raise ValueError('Proposed keeper lacks resolved geometry')
            obj.update(duplicate_of=keep,represented_by=[],status='duplicate',export_enabled=False)
            obj['history'].append({'utc':now(),'stage':'coverage_duplicate',**event});event['status']='applied'
        except Exception as exc:event.update(status='pending_adjudication',error=str(exc))
        self.scene['issues'].append(event)

    def coverage(self,round_index):
        tag=f'coverage/round_{round_index:02d}';folder=self.folder/'coverage'/(self.segment or 'initial')/f'round_{round_index:02d}';folder.mkdir(parents=True)
        global_overlay(self.scene,self.image).save(folder/'inventory_2048.png');items=inventory(self.scene)
        raw=self.call('coverage',tag,prompts.coverage(self.image.size,items,round_index),
            [('Image 1: clean global source',self.root/self.scene['image']['input']),('Image 2: inventory and active geometry',folder/'inventory_2048.png')],
            lambda r:prompts.check_coverage(r,self.image.size,[i['id'] for i in items]))
        self.scene['coverage_checks'].append({'request':self.request_path(tag),**raw})
        for pair in raw['duplicate_candidates']:self.apply_duplicate(pair,tag)
        records=[self.add_region(r['bbox_2d'],r['reason']) for r in raw['missing_regions']];self.checkpoint();return records

    def run(self):
        if getattr(self.args,'resume',False):
            check_ids=[i for i,o in self.scene['objects'].items() if o.get('fit') and o.get('export_enabled') and
                o['status'] in ('refined_unverified','failed_with_diagnostic_fit','processing')]
            skip={'accepted_by_model','rejected','rejected_by_model','duplicate','represented_by_children'}
            retry_ids=[i for i,o in self.scene['objects'].items() if i not in check_ids and (not o['attempted'] or o['status'] not in skip)]
            save(self.folder/f'continuation_{self.segment}.json',{'check_existing_ids':check_ids,'refinement_queue':retry_ids,
                'preserved_ids':[i for i in self.scene['objects'] if i not in check_ids and i not in retry_ids]})
            for ident in check_ids:
                try:self.recheck_existing(ident)
                except Exception as exc:self.error('resume_check_'+ident,exc,ident)
            if retry_ids:self.process_queue(retry_ids,'resumed_pending')
            for record in list(self.scene['search_regions']):
                if record['status'] in ('pending','failed'):
                    record.setdefault('history',[]).append({'status':record['status'],'utc':now(),'segment':self.segment})
                    self.discover(record,0)
            pending=[i for i,o in self.scene['objects'].items() if not o['attempted']]
            if pending:self.process_queue(pending,'resumed_discovery')
        else:self.process_queue(list(self.scene['objects']),'initial')
        for r in range(self.args.coverage_rounds+1):
            try:records=self.coverage(r)
            except Exception as exc:self.error(f'coverage_round_{r}',exc);continue
            if not records:break
            if r==self.args.coverage_rounds:
                for record in records:record['status']='coverage_budget_exhausted'
                break
            for record in records:self.discover(record,0)
            # No per-region or per-image target limit; sweep persisted inventory, including partial discoveries.
            ids=[i for i,o in self.scene['objects'].items() if not o['attempted']]
            if ids:self.process_queue(ids,f'expansion_{r}')
        # Catch any pending objects even if a discovery failed after committing candidates.
        pending=[i for i,o in self.scene['objects'].items() if not o['attempted']]
        if pending:self.process_queue(pending,'final_pending')
        self.scene.update(run_state='ended',ended_utc=now());result=self.checkpoint(export=True)
        print(json.dumps(result,ensure_ascii=False),flush=True);return result


def main(argv=None):
    from datetime import datetime
    import fcntl
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed-run',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--segment')
    p.add_argument('--base-url',default='http://10.102.252.163:10088/v1');p.add_argument('--model',default='GLM-5.3-Flash')
    p.add_argument('--workers',type=int,default=12);p.add_argument('--plan-batch-size',type=int,default=8)
    p.add_argument('--context-rounds',type=int,default=2);p.add_argument('--correction-rounds',type=int,default=1)
    p.add_argument('--error-retries',type=int,default=1);p.add_argument('--coverage-rounds',type=int,default=2)
    p.add_argument('--discovery-depth',type=int,default=2);p.add_argument('--export-every',type=int,default=4)
    p.add_argument('--max-tokens',type=int,default=150000,help='One uniform output budget for every model stage, including thinking.')
    p.add_argument('--crop-scale',type=float,default=3);p.add_argument('--crop-max-edge',type=int,default=2048)
    args=p.parse_args(argv)
    if min(args.workers,args.plan_batch_size,args.export_every,args.max_tokens)<1:p.error('Positive execution sizes required')
    if min(args.context_rounds,args.correction_rounds,args.error_retries,args.coverage_rounds,args.discovery_depth)<0:p.error('Budgets must be nonnegative')
    Crop.from_model([0,0,10,10],[2048,2048]).scaled(args.crop_scale,args.crop_max_edge)
    root=args.output.resolve();previous=None
    if args.resume:
        previous=json.loads((root/'manifest.json').read_text())
        args.segment=args.segment or 'resume_'+datetime.now().strftime('%Y%m%d_%H%M%S')
        if not args.segment or any(not(c.isalnum() or c in '_-') for c in args.segment):p.error('Invalid segment ID')
        if not args.seed_run:args.seed_run=Path(previous['seed_run'])
    else:
        if not args.seed_run:p.error('--seed-run is required for a fresh run')
        if args.segment:p.error('--segment is reserved for continuation')
        root.mkdir(parents=True,exist_ok=False)
    run_lock=(root/'run.lock').open('a+')
    try:fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another controller already holds this run lock')
    seed=args.seed_run.resolve();project=Path(__file__).resolve().parents[1]
    with httpx.Client(trust_env=False,timeout=20) as session:
        response=session.get(args.base_url.rstrip('/')+'/models');response.raise_for_status();models=response.json()
    matching=[m for m in models['data'] if m['id']==args.model]
    if not matching:raise ValueError('Requested model absent from live service')
    if matching[0].get('max_model_len',args.max_tokens+1)<=args.max_tokens:raise ValueError('Output token budget leaves no room for inputs')
    segment_root=root/'segments'/args.segment if args.resume else root
    if args.resume:segment_root.mkdir(parents=True,exist_ok=False)
    save(segment_root/'live_models.json',models)
    snapshot=segment_root/'code_snapshot';snapshot.mkdir()
    for name in ['png2pptx','tests','design']:shutil.copytree(project/name,snapshot/name,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['run_full.py','review_full.py','review_m1.py','REQUIREMENTS.zh-CN.md','README.md']:shutil.copyfile(project/name,snapshot/name)
    save(segment_root/'code_sha256.json',{str(f.relative_to(snapshot)):digest(f) for f in sorted(snapshot.rglob('*')) if f.is_file()})
    if args.resume:
        save(segment_root/'previous_manifest.json',previous)
        for item in previous['images']:
            checkpoint=segment_root/'source_checkpoints'/item['id'];checkpoint.mkdir(parents=True)
            for name in ['scene.json','summary.json']:shutil.copyfile(root/'results'/item['id']/name,checkpoint/name)
        scenes=[json.loads((root/'results'/item['id']/'scene.json').read_text()) for item in previous['images']]
        for scene in scenes:
            with Image.open(root/scene['image']['input']) as image:
                if list(image.size)!=scene['image']['input_size'] or max(image.size)!=2048:raise ValueError('Global input changed during resume')
        manifest=copy.deepcopy(previous)
        manifest.setdefault('initial_config',copy.deepcopy(previous['config']))
    else:
        old=json.loads((seed/'manifest.json').read_text())
        scenes=[prepare_image(seed,root,item) for item in sorted(old['images'],key=lambda i:i['id'])]
        manifest={'schema_version':'png2pptx.shape_full.2','started_utc':now(),'seed_run':str(seed),
            'initial_candidate_count':sum(len(s['objects']) for s in scenes),'images':[s['image'] for s in scenes]}
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    segment_record={'id':args.segment or 'initial','started_utc':now(),'prompt_version':prompts.VERSION,'config':config,
        'code_snapshot':str(snapshot.relative_to(root)),'source':'existing scene checkpoints' if args.resume else 'seed bbox inventory'}
    manifest.setdefault('execution_segments',[]).append(segment_record)
    manifest.update(run_kind='Continuation from existing scene checkpoints; accepted geometry preserved with original provenance; current requests use new settings' if args.resume else 'Fresh full-inventory calls using reused seed bboxes and clean images',
        prompt_version=prompts.VERSION,config=config,model=args.model,base_url=args.base_url,thinking='on',input_long_edge=2048,
        max_tokens=args.max_tokens,max_targets_per_image=None,active_segment=args.segment,
        crop_resize_policy={'requested_scale':args.crop_scale,'max_long_edge':args.crop_max_edge,'formula':'min(requested_scale, max_long_edge / crop_long_edge)'},
        coordinate_precision={'model_coordinate_decimals':2,'internal_transforms':'full floating-point precision','raw_model_responses':'preserved unchanged'},
        budgets_note='Every candidate is processed. Iteration budgets can leave explicit unresolved objects/search regions; candidate processing is not measured visual recall.',
        coordinate_policy='Global input long edge 2048; crop scale is min(3,2048/crop_long_edge) by default. New model coordinates round to 2 decimals; internal Pillow half-pixel transforms retain full precision. Historical request frames remain authoritative.')
    save(root/'manifest.json',manifest);save(segment_root/'segment.json',segment_record)
    semaphore=threading.Semaphore(args.workers);lock=threading.Lock();clients={}
    for stage in ['refine','plan','check','coverage','discovery']:
        clients[stage]=StreamingRelayClient(root/'calls',args.model,semaphore,args.base_url,None,max_tokens=args.max_tokens,max_attempts=1,read_timeout=1800,thinking='on');clients[stage].lock=lock
    results=[]
    def global_status(state):
        summaries=[json.loads(f.read_text()) for f in sorted((root/'results').glob('*/summary.json'))]
        save(root/'summary.json',{'state':state,'active_segment':args.segment,'updated_utc':now(),'completed_images':len(results),'total_images':len(scenes),
            'results':summaries,'fatal_errors':[r for r in results if 'fatal_error' in r]})
    runners=[FullImageRunner(s,root,clients,args) for s in scenes];global_status('running')
    print(f'RUN segment={args.segment or "initial"} max_tokens={args.max_tokens} crop_scale={args.crop_scale} crop_max_edge={args.crop_max_edge} coordinates=2dp',flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(r.run):r.case for r in runners}
        for future in as_completed(jobs):
            try:results.append(future.result())
            except Exception as exc:
                results.append({'case':jobs[future],'fatal_error':str(exc),'traceback':traceback.format_exc()})
                save(root/'results'/jobs[future]/('fatal_error_'+(args.segment or 'initial')+'.json'),results[-1])
            global_status('running')
    global_status('ended');print('FULL RUN ENDED '+str(root),flush=True)


if __name__=='__main__':main()
