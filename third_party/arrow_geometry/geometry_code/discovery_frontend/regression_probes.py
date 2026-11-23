#!/usr/bin/env python3
"""Live small probes and saved-trace compatibility checks for the integrated release."""
import argparse
import copy
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from transport.common import read_json as read,write_json as write,sha256
from transport.model import Model
import owner_refs
import post_tile_sam

def prepare(root,exp):
    if root.exists():raise ValueError('Fresh validation run required')
    root.mkdir(parents=True)
    shutil.copytree(Path(__file__).parent,root/'code',ignore=shutil.ignore_patterns('__pycache__'))
    cfg=read(Path(__file__).with_name('default_config.json'));cfg['review_output_dir']=str(root/'reports');write(root/'config.json',cfg)
    v25=exp/'runs/bbox_frontend_relations_complete16_20261025_v25';v24=exp/'runs/bbox_frontend_owner_refs16_20261025_v24';v16=exp/'runs/bbox_frontend_post_tile_sam4_20261025_v16'
    for cid in ['task_0346','task_0183']:shutil.copytree(v25/'inputs'/cid,root/'inputs'/cid)
    write(root/'cases.json',['task_0346','task_0183'])
    old=v25/'cases/task_0346/relations/P063';p=root/'long_pair';p.mkdir()
    for name in ['target.png','boxes_01.png']:shutil.copy2(old/name,p/name)
    shutil.copy2(old/'request/prompt.json',p/'old_prompt.json');shutil.copy2(old/'request/call_000/status.json',p/'old_status.json')
    objects=read(v25/'cases/task_0183/pre_relation_objects.json');write(root/'component_pair.json',[o for o in objects if o['id'] in ['C0006','C0015']])
    sam=root/'sam_replay';sam.mkdir();samcfg={**cfg,'post_sam_cache_run':str(v16),'source_run':str(v16),'review_output_dir':str(root/'reports/sam')}
    write(sam/'config.json',samcfg);samids=['task_0320','task_0027','task_0024','task_0241'];write(sam/'cases.json',samids)
    for cid in samids:
        shutil.copytree(v16/'inputs'/cid,sam/'inputs'/cid);c=sam/'cases'/cid;c.mkdir(parents=True)
        shutil.copy2(v16/'cases'/cid/'baseline_result.json',c/'baseline_result.json')
    format_checks=[]
    for cid,key in [('task_0028','G_C0016'),('task_0170','G_C0003'),('task_0178','A002'),('task_0236','G_C0003')]:
        found=[]
        for p in (v24/'cases'/cid/'checks'/key).rglob('parsed_*.json'):
            d=read(p)
            if any(x.get('same_as_readonly') is False or x.get('independent_complete') is None or x.get('source_identity_preserved') is None for x in d.get('decisions',[])):
                prompt=read(p.parent/'prompt.json')['prompt'];normalized=owner_refs.validate_identity_decisions(d,{str(i+1) for i in range(len(prompt['candidate']))},len(prompt['readonly']))
                found.append({'source':str(p),'sha256':sha256(p),'normalized':normalized})
        if not found:raise RuntimeError('Missing saved format failure '+cid)
        format_checks.append({'case_id':cid,'checks':found})
    write(root/'format_replay.json',{'cases':format_checks,'new_model_calls':0,'all_passed':True})
    write(root/'sources.json',{'v25':str(v25),'v24':str(v24),'v16':str(v16)})
    files={str(p.relative_to(root)):sha256(p) for p in root.rglob('*') if p.is_file()}
    write(root/'manifest.json',{'files':files});write(root/'progress.json',{'state':'prepared'})

def run(root):
    for path,h in read(root/'manifest.json')['files'].items():
        if sha256(root/path)!=h:raise ValueError('Frozen validation file changed '+path)
    cfg=read(root/'config.json');slots=threading.BoundedSemaphore(16)
    model=Model(cfg,run_dir=root,system=owner_refs.SYSTEM);model.slots=slots
    def long_pair():
        p=root/'long_pair';old=read(p/'old_prompt.json');m=Model(cfg,run_dir=root,system=old['system']);m.slots=slots
        def valid(a):
            if a.get('relation') not in ['same','different','uncertain'] or not a.get('reason'):raise ValueError('relation required')
            return a
        start=time.time();answer=m.ask(p/'request',old['prompt'],[('TARGET',p/'target.png'),('BOXES_1',p/'boxes_01.png')],valid)
        result={'answer':answer,'seconds':time.time()-start,'old_seconds':read(p/'old_status.json')['seconds'],'single_case_not_a_general_speed_claim':True}
        write(p/'comparison.json',result);return result
    def component():
        rows=read(root/'component_pair.json');answer=owner_refs.pair_job(root,'task_0183',1,rows,cfg,model)
        active,merged,conflicts=owner_refs.reconcile(rows,[answer]);owner_refs.export_handoff(root,'task_0183',active,cfg)
        h=read(root/'cases/task_0183/reconstruction_handoff.json');refs={t['target_id']:[x['owner_id'] for x in t['readonly_identities']] for t in h['tasks']}
        result={'answer':answer,'merged':merged,'conflicts':conflicts,'readonly':refs,'expected_component_link_present':'C0015' in refs.get('C0006',[])}
        write(root/'component_result.json',result);return result
    def sam():
        p=root/'sam_replay';scfg=read(p/'config.json');m=Model(scfg,run_dir=p,system=post_tile_sam.SYSTEM);m.slots=slots
        records=[]
        for cid in read(p/'cases.json'):
            r=post_tile_sam.execute_case(p,cid,scfg,m,'all')
            records.append({k:r[k] for k in ['case_id','added_count','pending_count','source_objects_unchanged','state']})
        write(root/'sam_results.json',records);return records
    write(root/'process.json',{'pid':os.getpid(),'started_at':owner_refs.now()});results={}
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs={pool.submit(fn):name for name,fn in [('coordinate_hint',long_pair),('component_relation',component),('post_tile_sam',sam)]}
        for f in as_completed(jobs):
            name=jobs[f]
            try:results[name]={'state':'completed','result':f.result()}
            except Exception as exc:results[name]={'state':'failed','error':str(exc)}
            write(root/'progress.json',{'state':'running','results':results,'updated_at':owner_refs.now()})
    write(root/'progress.json',{'state':'finished','results':results,'updated_at':owner_refs.now()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--experiment-dir',type=Path)
    a=p.parse_args();root=a.run_dir.resolve()
    prepare(root,a.experiment_dir.resolve()) if a.action=='prepare' else run(root)
