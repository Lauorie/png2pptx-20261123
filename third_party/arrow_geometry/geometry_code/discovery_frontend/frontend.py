#!/usr/bin/env python3
"""Frozen bbox-first identity front end. Original observations remain auditable."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import traceback

import numpy as np
from PIL import Image
from pipeline_frames import Crop
from transport.common import read_json as read,write_json as write,sha256
from transport.model import Model
from protocol import SYSTEM,discovery,refinement,relation_prompt,validate_discovery,validate_refine,validate_relations
from core import batches,crop_box,union,make_pairs,pair_batches,resolve,pending_object,ledger,area
from evidence import frame,save,boxes,get_sam,select_masks,overlay_masks,sam_only_observations,native_box,local_box,bounds

def now():return datetime.now(timezone.utc).isoformat()
REPORT_LOCK=threading.Lock()

def prepare(root,workspace,prior,working,benchmark):
 if root.exists():raise RuntimeError('Fresh run directory required')
 root.mkdir(parents=True);baseline=(workspace/'references/pptbench125_run').resolve()
 cfg=read(prior.parent/'identity_cleanup_sam_adaptive_20261023_v8/config.json')
 cfg.update(api_concurrency=32,case_workers=8,view_workers=6,refine_workers=4,request_workers=32,
            simple_batch_max=8,complex_batch_max=2,context_candidate_max=8,sam_evidence_max=32,
            max_expansion_rounds=2,pair_batch_max=6,pair_object_max=10,source_run=str(prior),baseline_run=str(baseline),
            review_output_dir=str(benchmark/'outputs/PPTBench/arrow_localization/reviews'/root.name),
            purpose='bbox-first independent discovery, pixel refinement, explicit ownership before geometry',
            protocol_version='bbox_frontend_v1',discovery_reuse=False)
 cfg.pop('arms',None);assert cfg['max_tokens']==113000 and 'reasoning_effort' not in cfg
 write(root/'config.json',cfg);shutil.copytree(working,root/'code',ignore=shutil.ignore_patterns('__pycache__'))
 snapshot=root/'constraints_snapshot';snapshot.mkdir()
 protocol=workspace/'working/code/protocol.py';tree=ast.parse(protocol.read_text())
 scope=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SCOPE' for t in n.targets))
 (snapshot/'scope.txt').write_text(scope+'\n');shutil.copy2(protocol,snapshot/'source_protocol.py')
 for name in ['箭头重建范围.md','分子结构与外部连线边界_20261023.md','模型思考强度约定_20261023.md','发现分组用户修订_20261023_v2.md']:
  shutil.copy2(workspace/'communication'/name,snapshot/name)
 shutil.copy2(workspace/'analysis/baseline_duplicates_0236_0201_20261024/重复原因诊断.md',snapshot/'baseline_duplicate_diagnosis.md')
 # Freeze baseline visualization reference without importing its reconstruction stack.
 shutil.copy2(baseline/'code/visual.py',snapshot/'baseline_visual.py')
 case_ids=[r['case_id'] for r in read(prior/'cases.json')];assert len(case_ids)==len(set(case_ids))==125
 write(root/'cases.json',case_ids);provenance=[]
 for cid in case_ids:
  inp=root/'inputs'/cid;shutil.copytree(prior/'inputs'/cid,inp)
  native=baseline/'cases'/cid/'clean.png';old=Image.open(inp/'source.png').convert('RGB');new=Image.open(native).convert('RGB')
  if old.size!=new.size or old.tobytes()!=new.tobytes():raise RuntimeError('Different baseline image: '+cid)
  shutil.copy2(native,inp/'source.png')
  views=read(inp/'views.json')
  for v in views['views']:
   if sha256(root/v['path'])!=v['sha256']:raise RuntimeError('Input view hash mismatch')
  source=read(baseline/'cases'/cid/'source_manifest.json')
  cache=Path(source['sam_prediction_path']).parent
  write(inp/'baseline.json',{'baseline_case':str(baseline/'cases'/cid),'sam_cache':str(cache),
       'image_sha256':sha256(native),'identity_target_count':sum(len(read(f).get('targets',[])) for f in (baseline/'cases'/cid/'groups').glob('*/identity/result.json')),
       'observation_group_count':len(read(baseline/'cases'/cid/'observation_plan.json')['groups'])})
  provenance.append({'case_id':cid,'native_sha256':sha256(native),'views':len(views['views']),'same_pixels_as_v7':True,'global_sam_cache':str(cache)})
 write(root/'input_provenance.json',provenance)
 write(root/'preparation.json',{'state':'prepared','cases':125,'global_discovery_calls':125,
       'local_discovery_calls':sum(p['views']-1 for p in provenance),'refinement_and_ownership_calls':'data dependent bounded batches',
       'global_sam':'reuse only exact image/model/category/threshold cache','new_glm_discovery':True,'created_at':now()})
 files=[p for folder in ['code','constraints_snapshot','inputs'] for p in (root/folder).rglob('*') if p.is_file()]
 files += [root/'config.json',root/'cases.json',root/'input_provenance.json',root/'preparation.json']
 write(root/'manifest.json',{'files':{str(p.relative_to(root)):sha256(p) for p in files},'created_at':now(),'credentials_copied':False})
 publish(root)
 print({'prepared_cases':125,'directory':str(root)},flush=True)

def public(row,view):
 return {'id':row['id'],'kind':row['kind'],'bbox':local_box(row['bbox_native'],view),'locator':row['locator'],
         'extent':row['extent'],'sam_proposal':bool(row.get('sam_only'))}

def discover_view(root,cid,v,global_v,scope,model):
 out=root/'cases'/cid/'discovery'/v['view_id'];target=root/v['path'];view=frame(v['transform'])
 q=discovery(scope,view.input_size,v['view_id']);gf=frame(global_v['transform'])
 q['target_region_in_global_px']=local_box(list(view.box),gf)
 images=[('TARGET',target)]
 if v['view_id']!='G':images.insert(0,('GLOBAL',root/global_v['path']))
 ans=model.ask(out/'request',q,images,lambda a:validate_discovery(a,view.input_size))
 rows=[]
 for o in ans['objects']:
  raw=o['bbox'];clipped=[max(0,min(float(x),view.input_size[i%2]-1)) for i,x in enumerate(raw)]
  b=native_box(clipped,view);b=[max(0,min(x,view.global_size[i%2]-1)) for i,x in enumerate(b)]
  rows.append({**o,'id':v['view_id']+'/'+o['id'],'bbox_native':b,'raw_bbox':raw,'source_view':v['view_id'],
               'coordinate_audit':{'input_frame':v['transform'],'edge_clipped':raw!=clipped},'sam_only':False})
 from visual_labels import render
 write(out/'observations.json',rows);render(target,rows,out/'overlay.png',view)
 return rows

def materialize(answer,focus,selected,view,prefix):
 index={r['id']:r for r in focus};masks={m['alias']:m for m in selected};rows=[]
 for o in answer['objects']:
  source=o['bbox_source'];audit={'source':source,'raw_bbox':o.get('bbox'),'mask_aliases':o['sam_ids']}
  if source=='original':b=index[o['original_id']]['bbox_native'];local=local_box(b,view)
  elif source=='model':
   local=[max(0,min(float(x),view.input_size[i%2]-1)) for i,x in enumerate(o['bbox'])];b=native_box(local,view)
  else:
   union_mask=np.zeros(view.input_size[::-1],dtype=bool)
   for mid in o['sam_ids']:union_mask |= masks[mid]['projected']
   bb=bounds(union_mask)
   if bb is None:raise ValueError('Selected SAM evidence is empty')
   local=[max(0,bb[0]-2),max(0,bb[1]-2),min(view.input_size[0]-1,bb[2]+2),min(view.input_size[1]-1,bb[3]+2)]
   b=native_box(local,view);audit.update(mask_area=int(union_mask.sum()),source_mask_ids=[masks[mid]['id'] for mid in o['sam_ids']])
  b=[max(0,min(float(x),view.global_size[i%2]-1)) for i,x in enumerate(b)]
  touch=(local[0]<=1 and view.box[0]>0) or (local[1]<=1 and view.box[1]>0) or (local[2]>=view.input_size[0]-2 and view.box[2]<view.global_size[0]) or (local[3]>=view.input_size[1]-2 and view.box[3]<view.global_size[1])
  old_box=union([index[s]['bbox_native'] for s in o['source_ids']]);growth=area(b)/max(1,area(old_box))
  rows.append({**o,'id':prefix+'/'+o['id'],'bbox_native':b,'bbox_target':local,
       'extent':'partial' if touch else o['extent'],'uncertain':o['uncertain'],'needs_split':o['kind']=='dense_region',
       'coordinate_audit':audit,'extent_audit':{'area_ratio_to_sources':growth,'touches_internal_boundary':bool(touch),
               'scope_changed_to_network':o['kind'] in {'shared_network','dense_network'} and any(index[s]['kind']=='single' for s in o['source_ids'])}})
 return rows

def refine_packet(root,cid,spec,index,masks,scope,model,stage='refine',expansion=False):
 cfg=model.config;folder=root/'cases'/cid/stage/spec['batch_id'];result_file=folder/'result.json'
 if result_file.exists():return read(result_file)
 inp=root/'inputs'/cid;native=Image.open(inp/'source.png').convert('RGB');cb=spec['crop'];view=Crop(native.size,tuple(cb),(2*(cb[2]-cb[0]),2*(cb[3]-cb[1])))
 focus=[index[k] for k in spec['focus_ids']];context=[index[k] for k in spec['context_ids']]
 save(view.image(native),folder/'target.png');boxes(folder/'target.png',focus,folder/'boxes.png',view,context)
 relevant=masks
 # Expansion has a new exact input; obtain local evidence for the expanded identity window.
 if expansion:
  relevant=masks+get_sam(folder/'target.png',folder/'sam',cfg,view,'E'+spec['batch_id'])
 selected,total=select_masks(relevant,focus,view,cfg['sam_evidence_max']);overlay_masks(folder/'target.png',selected,folder/'sam.png')
 gf=frame(read(inp/'views.json')['views'][0]['transform'])
 metadata=[{k:v for k,v in m.items() if k!='projected'} for m in selected];write(folder/'sam_evidence.json',{'selected':metadata,'relevant_total':total,'not_all_masks_shown':total>len(selected)})
 q=refinement(scope,view.input_size,[public(r,view) for r in focus],[public(r,view) for r in context],
      [{'id':m['alias'],'bbox':m['bbox_target'],'category':m['category'],'score':m['score']} for m in selected],local_box(cb,gf),expansion)
 write(folder/'packet.json',{**spec,'view':view.metadata(native.size),'source_ids':list(index)})
 answer=model.ask(folder/'request',q,[('GLOBAL',inp/'G/target.png'),('TARGET',folder/'target.png'),('BOXES',folder/'boxes.png'),('SAM',folder/'sam.png')],
                  lambda a:validate_refine(a,view.input_size,spec['focus_ids'],[m['alias'] for m in selected]))
 objects=materialize(answer,focus,selected,view,stage+'/'+spec['batch_id'])
 for r in answer['unresolved']:objects.append(pending_object(index[r['source_id']],r['reason'],prefix=stage+'/'+spec['batch_id']+'/PENDING'))
 result={'objects':objects,'removed':answer['removed'],'unresolved':answer['unresolved'],'focus_ids':spec['focus_ids']}
 write(result_file,result);boxes(folder/'target.png',objects,folder/'overlay.png',view)
 return result

def review_relations(root,cid,objects,scope,model,stage):
 folder=root/'cases'/cid/stage
 pairs=make_pairs(objects);index={o['id']:o for o in objects};packets=pair_batches(pairs,index,model.config['pair_batch_max'],model.config['pair_object_max'])
 write(folder/'pairs.json',pairs);decisions=[];issues=[]
 def review(number,subset):
  out=folder/f'R{number:04d}';ids=sorted({p[k] for p in subset for k in ['a','b']});focus=[index[k] for k in ids]
  inp=root/'inputs'/cid;native=Image.open(inp/'source.png').convert('RGB');cb=crop_box([o['bbox_native'] for o in focus],native.size,32)
  view=Crop(native.size,tuple(cb),(2*(cb[2]-cb[0]),2*(cb[3]-cb[1])));save(view.image(native),out/'target.png');boxes(out/'target.png',focus,out/'boxes.png',view)
  q=relation_prompt(scope,[public(o,view) for o in focus],subset)
  ans=model.ask(out/'request',q,[('GLOBAL',inp/'G/target.png'),('TARGET',out/'target.png'),('BOXES',out/'boxes.png')],lambda a:validate_relations(a,subset))
  write(out/'result.json',ans);return ans['decisions']
 with ThreadPoolExecutor(max_workers=model.config['refine_workers']) as pool:
  jobs={pool.submit(review,i,p):i for i,p in enumerate(packets,1)}
  for f in as_completed(jobs):
   try:decisions.extend(f.result())
   except Exception as exc:issues.append({'stage':stage,'packet':jobs[f],'error':str(exc)})
 result=resolve(objects,pairs,decisions);result['issues']=issues;write(folder/'result.json',result)
 return result

def run_case(root,cid,model,scope):
 cfg=model.config;folder=root/'cases'/cid;folder.mkdir(parents=True,exist_ok=True)
 if (folder/'finished.json').exists():return read(folder/'result.json')
 inp=root/'inputs'/cid;packet=read(inp/'views.json');views=packet['views'];global_v=views[0];native=Image.open(inp/'source.png').convert('RGB');size=native.size
 issues=[];observations=[];masks=[]
 def status(phase,**extra):
  write(folder/'status.json',{'state':'running','phase':phase,'updated_at':now(),**extra});publish(root)
 status('independent_discovery_and_sam')
 def sam_view(v):
  if v['view_id']=='G':
   view=Crop(size,(0,0,*size),size);image=inp/'source.png';cache=Path(read(inp/'baseline.json')['sam_cache']);prefix='SG'
  else:view=frame(v['transform']);image=root/v['path'];cache=None;prefix='S'+v['view_id']
  return get_sam(image,folder/'sam'/v['view_id'],cfg,view,prefix,cache)
 with ThreadPoolExecutor(max_workers=cfg['view_workers']) as pool:
  jobs={}
  for v in views:
   jobs[pool.submit(discover_view,root,cid,v,global_v,scope,model)]=('discovery',v['view_id'])
   jobs[pool.submit(sam_view,v)]=('sam',v['view_id'])
  for f in as_completed(jobs):
   stage,vid=jobs[f]
   try:
    result=f.result()
    if stage=='discovery':observations.extend(result)
    else:masks.extend(result)
   except Exception as exc:
    issue={'stage':stage,'view':vid,'error_type':type(exc).__name__,'error':str(exc)};issues.append(issue);write(folder/'failures'/f'{stage}_{vid}.json',issue)
   write(folder/'discovery_progress.json',{'finished':sum(x.done() for x in jobs),'total':len(jobs),'observations':len(observations),'masks':len(masks),'issues':issues,'updated_at':now()})
 observations.sort(key=lambda r:r['id']);write(folder/'glm_observations.json',observations);write(folder/'sam_evidence.json',masks)
 sam_rows,audit=sam_only_observations(masks,observations);write(folder/'sam_nomination_audit.json',audit)
 observations+=sam_rows;write(folder/'all_observations.json',observations)
 boxes(inp/'G/target.png',observations,folder/'observations.png',frame(global_v['transform']))
 specs=batches(observations,size,cfg);write(folder/'refinement_plan.json',specs);status('bbox_refinement',batches=len(specs),observations=len(observations))
 objects=[];removed=[];index={o['id']:o for o in observations}
 with ThreadPoolExecutor(max_workers=cfg['refine_workers']) as pool:
  jobs={pool.submit(refine_packet,root,cid,s,index,masks,scope,model):s for s in specs}
  for f in as_completed(jobs):
   spec=jobs[f]
   try:r=f.result();objects.extend(r['objects']);removed.extend(r['removed'])
   except Exception as exc:
    issue={'stage':'refine','batch':spec['batch_id'],'error_type':type(exc).__name__,'error':str(exc)};issues.append(issue);write(folder/'refine'/spec['batch_id']/'failed.json',issue)
    objects.extend(pending_object(index[sid],'Refinement failed; original observation retained',prefix='refine/'+spec['batch_id']+'/PENDING') for sid in spec['focus_ids'])
 write(folder/'refined_objects.json',objects);boxes(inp/'G/target.png',objects,folder/'refined.png',frame(global_v['transform']))
 status('identity_ownership');result=review_relations(root,cid,objects,scope,model,'ownership');issues+=result['issues']
 first_ownership=result
 for round_no in range(1,cfg['max_expansion_rounds']+1):
  targets=[o for o in result['objects'] if o['extent']=='partial']
  if not targets:break
  status('partial_extent_expansion',round=round_no,targets=len(targets));current={o['id']:o for o in result['objects']};replacements={}
  def expand(o,number):
   b=o['bbox_native'];width=max(96,b[2]-b[0]);height=max(96,b[3]-b[1]);cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2
   cb=crop_box([[cx-width,cy-height,cx+width,cy+height]],size,48) if round_no==1 else [0,0,*size]
   context=sorted([x for x in result['objects'] if x['id']!=o['id']],key=lambda x:__import__('core').gap(x['bbox_native'],b))[:cfg['context_candidate_max']]
   spec={'batch_id':f'E{number:04d}','focus_ids':[o['id']],'context_ids':[c['id'] for c in context],'crop':cb}
   r=refine_packet(root,cid,spec,current,masks,scope,model,stage='expand'+str(round_no),expansion=True)
   # Retain original observations and prior ownership when the refined task gets a new box/split.
   for new in r['objects']:
    new['expansion_parent_id']=o['id'];new['source_ids']=list(o['source_ids']);new['prior_extent_audit']=o.get('extent_audit')
   if r['removed']:
    # A partial identity cannot disappear solely while obtaining a bigger crop.
    return [pending_object(o,'Expansion proposes exclusion; retained for explicit scope review',prefix='EXPANSION_PENDING')]
   return r['objects']
  with ThreadPoolExecutor(max_workers=cfg['refine_workers']) as pool:
   jobs={pool.submit(expand,o,i):o for i,o in enumerate(targets,1)}
   for f in as_completed(jobs):
    old=jobs[f]
    try:replacements[old['id']]=f.result()
    except Exception as exc:issues.append({'stage':'expansion','object_id':old['id'],'error':str(exc)})
  objects=[x for o in result['objects'] for x in replacements.get(o['id'],[o])]
  result=review_relations(root,cid,objects,scope,model,'ownership_after_expand'+str(round_no));issues+=result['issues']
 result.update(case_id=cid,state='completed_with_issues' if issues else 'completed',completed_at=now(),issues=issues,
   input_observation_count=len(observations),glm_observation_count=sum(not o.get('sam_only') for o in observations),
   sam_only_observation_count=len(sam_rows),source_ledger=ledger(observations,result,removed),excluded_observations=removed,
   first_ownership_history={'merges':first_ownership['merges'],'absorbed_members':first_ownership['absorbed_members']},
   output_identity_count=len(result['objects']),pending_count=sum(o['uncertain'] or o['extent']=='partial' or o['kind'] in {'dense_region','sam_proposal'} for o in result['objects']),
   formal_accuracy_measured=False)
 for o in result['objects']:
  o['geometry_ready']=not o['uncertain'] and o['extent']=='complete' and o['kind'] in {'single','shared_network','dense_network'}
  o['reconstruction_contract']={'owner_id':o['id'],'scope_bbox_native':o['bbox_native'],'crop_is_not_ownership':True,
     'independent_neighbor_ids':[x['id'] for x in result['objects'] if x['id']!=o['id'] and __import__('core').gap(o['bbox_native'],x['bbox_native'])<32],
     'scope_change_requires_identity_review':True,'shared_paths_once':o['kind']=='shared_network'}
 boxes(inp/'G/target.png',result['objects'],folder/'overlay.png',frame(global_v['transform']));write(folder/'result.json',result)
 write(folder/'finished.json',{'state':result['state'],'completed_at':now()});write(folder/'status.json',{k:result[k] for k in ['state','case_id','output_identity_count','pending_count','completed_at']});publish(root)
 return result

def publish(root):
 with REPORT_LOCK:
  cfg=read(root/'config.json');pub=Path(cfg['review_output_dir']);(pub/'images').mkdir(parents=True,exist_ok=True);(pub/'data').mkdir(exist_ok=True)
  lines=['# bbox前置箭头发现：125图总览','','原图与当前身份框两栏。每张图使用同一配置；橙色single，蓝色网络/待拆密集区，紫色待确认或跨块不完整对象。候选数量和执行完成不代表召回率。','',
         '全图/局部独立GLM发现＋全局/局部SAM证据 → 小批bbox校准 → 显式去重与网络成员归属 → 必要的跨块扩图 → 交付身份及归属。','']
  summary=[]
  for cid in read(root/'cases.json'):
   inp=root/'inputs'/cid;folder=root/'cases'/cid
   source=pub/'images'/(cid+'_source.png')
   if not source.exists():shutil.copy2(inp/'G/target.png',source)
   status=read(folder/'status.json') if (folder/'status.json').exists() else {'state':'queued'}
   lines+=['## '+cid,'',f'状态：{status.get("state")}；阶段：{status.get("phase","finished" if status.get("completed_at") else "queued")}。','']
   if (folder/'result.json').exists():
    r=read(folder/'result.json');shutil.copy2(folder/'overlay.png',pub/'images'/(cid+'_result.png'));shutil.copy2(folder/'result.json',pub/'data'/(cid+'.json'))
    lines+=[f'观察候选 {r["input_observation_count"]} → 身份 {r["output_identity_count"]}；待确认/待补全/待拆分 {r["pending_count"]}。','',
       '| 原图 | 新身份框 |','|---|---|',f'| [![](images/{cid}_source.png)](images/{cid}_source.png) | [![](images/{cid}_result.png)](images/{cid}_result.png) |','',f'[每个候选的去向与理由、归属和请求证据](data/{cid}.json)','']
    audit=['# '+cid+' 候选归属记录','','| 候选 | 状态 | 最终身份 | 理由 |','|---|---|---|---|']
    for row in r['source_ledger']:audit.append('| '+row['source_id']+' | '+row['status']+' | '+', '.join(row['owner_ids'])+' | '+'；'.join(row['reasons']).replace('|','/').replace('\n',' ')+' |')
    (pub/'data'/(cid+'_归属.md')).write_text('\n'.join(audit)+'\n');lines += [f'[逐候选处理理由](data/{cid}_归属.md)','']
    summary.append({k:r[k] for k in ['case_id','state','input_observation_count','output_identity_count','pending_count']})
   else:
    available=folder/'refined.png' if (folder/'refined.png').exists() else folder/'observations.png'
    if available.exists():
     shutil.copy2(available,pub/'images'/(cid+'_progress.png'));right=f'[![](images/{cid}_progress.png)](images/{cid}_progress.png)<br>中间候选，尚未完成归属'
    else:right='尚未产生身份框'
    lines+=['| 原图 | 当前产物 |','|---|---|',f'| [![](images/{cid}_source.png)](images/{cid}_source.png) | {right} |','']
  temp=pub/'.report.tmp';temp.write_text('\n'.join(lines)+'\n');temp.replace(pub/'前置箭头身份发现_125例.md')
  write(root/'summary.json',{'updated_at':now(),'finished_cases':len(summary),'total_cases':125,'cases':summary,'report':str(pub/'前置箭头身份发现_125例.md')})

def preflight(root,model):
 from transport.credentials import load_credentials
 from sam_client import SamClient
 import httpx
 cfg=model.config;health=SamClient(cfg['sam_url']).health();write(root/'preflight/sam_health.json',health)
 if health['phase']!='ready' or health['model']['checkpoint_sha256']!=cfg['sam_checkpoint_sha256']:raise RuntimeError('SAM preflight mismatch')
 base,key,meta=load_credentials(cfg['base_url'])
 with httpx.Client(timeout=30,trust_env=False) as client:response=client.get(base+'/models',headers={'Authorization':'Bearer '+key})
 if response.status_code!=200 or cfg['model'] not in [x['id'] for x in response.json().get('data',[])]:raise RuntimeError('GLM model/auth preflight failed')
 write(root/'preflight/models.json',{'http_status':response.status_code,'model_available':True,'auth':meta})
 cid=read(root/'cases.json')[0]
 model.ask(root/'preflight/request',{'task':'Return {"ok":true}; transport preflight only.'},[('TARGET',root/'inputs'/cid/'G/target.png')],lambda a:a if a.get('ok') is True else (_ for _ in ()).throw(ValueError('ok required')))
 write(root/'preflight/passed.json',{'passed_at':now(),'thinking':cfg['thinking'],'max_tokens':cfg['max_tokens'],'reasoning_effort_present':False})

def execute(root):
 lock=(root/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for rel,digest in read(root/'manifest.json')['files'].items():
  if sha256(root/rel)!=digest:raise RuntimeError('Frozen artifact changed: '+rel)
 cfg=read(root/'config.json');assert cfg['max_tokens']==113000 and 'reasoning_effort' not in cfg
 model=Model(cfg,run_dir=root,system=SYSTEM);preflight(root,model);scope=(root/'constraints_snapshot/scope.txt').read_text()
 case_ids=read(root/'cases.json');write(root/'process.json',{'pid':os.getpid(),'started_at':now()})
 # Diagnostics may be seen early, but every case uses exactly the same algorithm and prompts.
 priority=['task_0236','task_0201','task_0320','task_0241','task_0237','task_0024','task_0056','task_0956']
 order=[c for c in priority if c in case_ids]+[c for c in case_ids if c not in priority]
 write(root/'execution_order.json',{'case_ids':order,'changes_only_scheduling':True});records=[]
 write(root/'progress.json',{'state':'running','total':125,'ended':0,'records':[],'updated_at':now()})
 with ThreadPoolExecutor(max_workers=cfg['case_workers']) as pool:
  jobs={pool.submit(run_case,root,cid,model,scope):cid for cid in order}
  for f in as_completed(jobs):
   cid=jobs[f]
   try:
    r=f.result();record={k:r[k] for k in ['case_id','state','output_identity_count','pending_count']}
   except Exception as exc:
    record={'case_id':cid,'state':'failed','error_type':type(exc).__name__,'error':str(exc)};write(root/'cases'/cid/'failed.json',{**record,'traceback':traceback.format_exc()})
   records.append(record);write(root/'progress.json',{'state':'running','total':125,'ended':len(records),'records':records,'updated_at':now()});publish(root);print(record,flush=True)
 write(root/'progress.json',{'state':'completed' if all(r['state']=='completed' for r in records) else 'completed_with_issues','total':125,'ended':len(records),'records':records,'updated_at':now()})
 write(root/'finished.json',{'completed_at':now(),'records':records});publish(root)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--workspace',type=Path);p.add_argument('--prior',type=Path);p.add_argument('--benchmark',type=Path);a=p.parse_args()
 if a.prepare:prepare(a.run_dir.resolve(),a.workspace.resolve(),a.prior.resolve(),Path(__file__).resolve().parent,a.benchmark.resolve())
 else:
  if Path(__file__).resolve().parent!=a.run_dir.resolve()/'code':raise RuntimeError('Execute frozen run code')
  execute(a.run_dir.resolve())
