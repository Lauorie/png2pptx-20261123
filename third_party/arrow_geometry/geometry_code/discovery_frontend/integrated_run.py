#!/usr/bin/env python3
"""PNG -> discovery -> safe splitting -> identities -> append-only tiled SAM."""
import argparse
import copy
import fcntl
import math
import os
import shutil
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from PIL import Image
from pipeline_frames import Crop
from evidence import save,frame
from transport.common import read_json as read,write_json as write,sha256
from transport.model import Model
import protocol
import light_run
import region_partition
import owner_refs
import post_tile_sam

STAGES=('discovery','partition','owners','post_sam')
REPORT_LOCK=threading.Lock()

def now():return owner_refs.now()

def input_views(root,cid,image,cfg):
    inp=root/'inputs'/cid;inp.mkdir(parents=True)
    with Image.open(image) as src:native=src.convert('RGB')
    native.save(inp/'source.png');w,h=native.size;scale=2048/max(w,h)
    specs=[('G',[0,0,w,h],(max(1,round(w*scale)),max(1,round(h*scale))),None)]
    cols,rows=max(1,math.ceil(w/cfg['tile_core_native_px'])),max(1,math.ceil(h/cfg['tile_core_native_px']))
    for y in range(rows):
        for x in range(cols):
            core=[round(x*w/cols),round(y*h/rows),round((x+1)*w/cols),round((y+1)*h/rows)]
            px=math.ceil((core[2]-core[0])*cfg['tile_overlap_ratio']);py=math.ceil((core[3]-core[1])*cfg['tile_overlap_ratio'])
            bounds=[max(0,core[0]-px),max(0,core[1]-py),min(w,core[2]+px),min(h,core[3]+py)]
            v=region_partition.local_view(native.size,bounds,cfg)
            specs.append((f'R{y+1}C{x+1}',bounds,v.input_size,core))
    views=[]
    for vid,bounds,size,core in specs:
        v=Crop(native.size,tuple(bounds),tuple(size));path=inp/vid/'target.png';save(v.image(native),path)
        row={'view_id':vid,'path':str(path.relative_to(root)),'sha256':sha256(path),'transform':v.metadata(native.size),'core':core,'grid_drawn':False}
        write(path.parent/'view.json',row);views.append(row)
    write(inp/'views.json',{'case_id':cid,'native_size':[w,h],'source_sha256':sha256(inp/'source.png'),'views':views})

def prepare(root,images,config=None,from_run=None,cases=None,seed_stage=None,seed_run=None):
    if root.exists():raise ValueError('Run already exists; use run to resume it')
    cfg=read(config or Path(__file__).with_name('default_config.json'))
    assert cfg['max_tokens']==113000 and cfg['thinking']['type']=='enabled' and 'reasoning_effort' not in cfg
    assert cfg['request_timeout_seconds']==1200 and cfg.get('pair_review_budget') is None
    root.mkdir(parents=True)
    shutil.copytree(Path(__file__).parent,root/'code',ignore=shutil.ignore_patterns('__pycache__','constraints_snapshot'))
    shutil.copytree(Path(__file__).parent/'constraints_snapshot',root/'constraints_snapshot')
    ids=[];provenance=[]
    if from_run:
        ids=cases or read(from_run/'cases.json')
        for cid in ids:
            source=from_run/'inputs'/cid;shutil.copytree(source,root/'inputs'/cid)
            for v in read(root/'inputs'/cid/'views.json')['views']:
                if sha256(root/v['path'])!=v['sha256']:raise ValueError('Imported image view mismatch '+cid)
            provenance.append({'case_id':cid,'input_source_run':str(from_run),'image_sha256':sha256(source/'source.png')})
    else:
        for path in images:
            cid=path.stem
            if cid in ids:raise ValueError('Duplicate image stem '+cid)
            ids.append(cid);input_views(root,cid,path,cfg)
            provenance.append({'case_id':cid,'source_path':str(path.resolve()),'source_sha256':sha256(path),'image_sha256':sha256(root/'inputs'/cid/'source.png')})
    if not ids:raise ValueError('No input images')
    if seed_stage:
        if not seed_run:raise ValueError('A seed run is required')
        for cid in ids:
            src=seed_run/'cases'/cid/'result.json'
            if sha256(seed_run/'inputs'/cid/'source.png')!=sha256(root/'inputs'/cid/'source.png'):raise ValueError('Seed pixels differ '+cid)
            dst=root/'seeds'/cid;dst.mkdir(parents=True);shutil.copy2(src,dst/'result.json')
            write(dst/'provenance.json',{'stage':seed_stage,'source':str(src),'sha256':sha256(src),'policy':'explicit stage replay; not a new discovery or same-prompt claim'})
    cfg.update(source_run=str(from_run or root),review_output_dir=str(root/'reports'),seed_stage=seed_stage)
    write(root/'config.json',cfg);write(root/'cases.json',ids);write(root/'input_provenance.json',provenance)
    for stage in STAGES:
        folder=root/'stages'/stage;folder.mkdir(parents=True)
        (folder/'inputs').symlink_to(root/'inputs',target_is_directory=True)
        (folder/'constraints_snapshot').symlink_to(root/'constraints_snapshot',target_is_directory=True)
        scfg={**cfg,'review_output_dir':str(root/'reports'/stage)}
        write(folder/'config.json',scfg);write(folder/'cases.json',ids)
        write(folder/'execution_order.json',{'case_ids':ids})
    write(root/'progress.json',{'state':'prepared','finished':0,'total':len(ids)})
    manifest={str(p.relative_to(root)):sha256(p) for base in ['code','inputs','constraints_snapshot','seeds'] if (root/base).exists() for p in (root/base).rglob('*') if p.is_file()}
    manifest.update({str(p.relative_to(root)):sha256(p) for p in [root/'config.json',root/'cases.json',root/'input_provenance.json']})
    for stage in STAGES:manifest['stages/'+stage+'/config.json']=sha256(root/'stages'/stage/'config.json')
    write(root/'manifest.json',{'files':manifest,'created_at':now()})
    publish(root)
    return root

def stage_issues(stage,result):
    problems=copy.deepcopy(result.get('errors',result.get('issues',[])))
    if stage=='partition':
        for item in result.get('region_ledger',[]):
            problems.extend({'region':item['parent_id'],**c} for c in item.get('conflicts',[]) if 'error' in c)
    return problems

def run_case(root,cid,models,cfg):
    out=root/'cases'/cid;out.mkdir(parents=True,exist_ok=True)
    if (out/'finished.json').exists():return read(out/'result.json')
    seed=cfg.get('seed_stage');start=STAGES.index(seed)+1 if seed else 0
    previous=read(root/'seeds'/cid/'result.json') if seed else None
    stage_records=[]
    scope=(root/'constraints_snapshot/scope.txt').read_text()
    for number,stage in enumerate(STAGES):
        if number<start:
            stage_records.append({'stage':stage,'state':'explicitly_seeded','source':str(root/'seeds'/cid/'provenance.json')});continue
        write(out/'status.json',{'state':'running','stage':stage,'updated_at':now()})
        folder=root/'stages'/stage;c=folder/'cases'/cid;c.mkdir(parents=True,exist_ok=True)
        scfg=read(folder/'config.json')
        if previous is not None:
            prior=c/'baseline_result.json'
            if prior.exists() and read(prior)!=previous:raise RuntimeError('Stage baseline changed on resume '+stage+'/'+cid)
            if not prior.exists():write(prior,previous)
        if stage=='discovery':result=light_run.run_case(folder,cid,models[stage],scope)
        elif stage=='partition':result=region_partition.execute_case(folder,cid,scfg,models[stage])
        elif stage=='owners':result=owner_refs.execute_case(folder,cid,scfg,models[stage])
        else:result=post_tile_sam.execute_case(folder,cid,scfg,models[stage],'all')
        stage_records.append({'stage':stage,'state':'completed','result_path':str(c/'result.json'),'result_sha256':sha256(c/'result.json'),
                              'objects':len(result['objects']),'issues':stage_issues(stage,result)})
        previous=result
        publish(root)
    owner_file=root/'stages/owners/cases'/cid/'result.json'
    owners=read(owner_file) if owner_file.exists() else read(root/'seeds'/cid/'result.json') if seed=='owners' else {}
    objects=previous['objects'];issues=[{'stage':s['stage'],'detail':i} for s in stage_records for i in s.get('issues',[])]
    if seed:issues.extend({'stage':'seed','detail':i} for i in stage_issues(seed,read(root/'seeds'/cid/'result.json')))
    result={'case_id':cid,'state':'completed_with_issues' if issues else 'completed','objects':objects,'output_identity_count':len(objects),
            'stages':stage_records,'issues':issues,'pending_identity_count':sum(bool(o.get('uncertain')) or o.get('extent')!='complete' for o in objects),
            'pending_relation_count':owners.get('pending_relation_count',0),'post_sam_pending_count':previous.get('pending_count',0),
            'sam_added_count':previous.get('added_count',0),'sam_existing_unchanged':previous.get('source_objects_unchanged',False),
            'region_ledger':owners.get('region_ledger',[]),'formal_accuracy_measured':False,'geometry_executed':False,'completed_at':now()}
    write(out/'result.json',result)
    inp=root/'inputs'/cid;view=frame(read(inp/'views.json')['views'][0]['transform'])
    owner_refs.render(inp/'G/target.png',objects,out/'overlay.png',view)
    owner_refs.export_handoff(root,cid,objects,cfg)
    handoff=read(out/'reconstruction_handoff.json')
    partition_baseline=root/'stages/partition/cases'/cid/'baseline_result.json'
    originals={o['id']:o for o in read(partition_baseline)['objects']} if partition_baseline.exists() else {}
    handoff['context_regions']=[{'parent_id':row['parent_id'],'role':'context_only','owner_ids':row['active_owner_ids'],
                                'original_region':originals.get(row['parent_id']),'coverage_evidence':row.get('coverage_checks',[])}
                               for row in result['region_ledger'] if row.get('commit')]
    write(out/'reconstruction_handoff.json',handoff)
    write(out/'finished.json',{'completed_at':now()});write(out/'status.json',{'state':result['state'],'stage':'finished'})
    publish(root)
    return result

def publish(root):
    with REPORT_LOCK:
        ids=read(root/'cases.json');dest=root/'reports';dest.mkdir(exist_ok=True)
        lines=['# 当前箭头发现版本：完整结果对照','','发现 → 可分区域拆分 → 独立身份与只读归属 → 后置分块 SAM 补漏。','',
               '完成数量与候选数不代表召回率；本入口输出身份与交接材料，未执行几何/PPT。','']
        records=[]
        for cid in ids:
            c=root/'cases'/cid;lines += ['## '+cid,'']
            if not (c/'finished.json').exists():
                status=read(c/'status.json') if (c/'status.json').exists() else {'stage':'queued'}
                lines += ['阶段：'+status.get('stage','unknown'),''];continue
            r=read(c/'result.json');h=read(c/'reconstruction_handoff.json');records.append({k:r[k] for k in ['case_id','state','output_identity_count','sam_added_count','pending_identity_count','pending_relation_count','post_sam_pending_count']})
            lines += [f"状态 {r['state']}；身份/任务 {r['output_identity_count']}；SAM追加 {r['sam_added_count']}；待核验身份 {r['pending_identity_count']}、关系 {r['pending_relation_count']}、SAM候选 {r['post_sam_pending_count']}。",'',
                      '| 原图 | 最终身份框 |','|---|---|',f'| ![原图]({root}/inputs/{cid}/G/target.png) | ![结果]({c}/overlay.png) |','',f'[完整结果]({c}/result.json) · [交接清单]({c}/reconstruction_handoff.json)','']
            if h['gallery']:
                g=max(h['gallery'],key=lambda x:x['readonly_count']);lines += [f"只读子身份示例：{g['target']}，{g['readonly_count']} 个参考。",'',f"![只读参考]({g['image']})",'']
        from report_images import write_report
        write_report(dest/'完整结果对照.md','\n'.join(lines)+'\n',root)
        write(dest/'summary.json',{'finished':len(records),'total':len(ids),'cases':records})

def run(root):
    if Path(__file__).resolve().parent!=root/'code':
        os.execv(sys.executable,[sys.executable,'-u',str(root/'code/integrated_run.py'),'run','--run-dir',str(root)])
    lock=(root/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name,h in read(root/'manifest.json')['files'].items():
        if sha256(root/name)!=h:raise RuntimeError('Frozen input/code changed '+name)
    cfg=read(root/'config.json');ids=read(root/'cases.json');slots=threading.BoundedSemaphore(cfg['api_concurrency'])
    models={}
    for stage in STAGES:
        scfg=read(root/'stages'/stage/'config.json')
        system=protocol.SYSTEM if stage=='discovery' else owner_refs.SYSTEM if stage=='owners' else post_tile_sam.SYSTEM if stage=='post_sam' else region_partition.SYSTEM
        model=Model(scfg,run_dir=root/'stages'/stage,system=system);model.slots=slots;models[stage]=model
    write(root/'process.json',{'pid':os.getpid(),'started_at':now()});records=[]
    write(root/'progress.json',{'state':'running','finished':0,'total':len(ids),'records':records,'updated_at':now()})
    with ThreadPoolExecutor(max_workers=cfg['case_workers']) as pool:
        jobs={pool.submit(run_case,root,cid,models,cfg):cid for cid in ids}
        for future in as_completed(jobs):
            cid=jobs[future]
            try:
                result=future.result();records.append({'case_id':cid,'state':result['state']})
            except Exception as exc:
                failure={'case_id':cid,'state':'failed','error':str(exc),'traceback':traceback.format_exc()};write(root/'cases'/cid/'failed.json',failure);records.append(failure)
            write(root/'progress.json',{'state':'running','finished':len(records),'total':len(ids),'records':records,'updated_at':now()})
    write(root/'progress.json',{'state':'finished','finished':len(records),'total':len(ids),'records':records,'updated_at':now()});publish(root)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','run']);parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--images',type=Path,nargs='*',default=[]);parser.add_argument('--input-dir',type=Path);parser.add_argument('--config',type=Path)
    parser.add_argument('--from-run',type=Path);parser.add_argument('--cases',nargs='+');parser.add_argument('--seed-stage',choices=STAGES[:-1]);parser.add_argument('--seed-run',type=Path)
    args=parser.parse_args();root=args.run_dir.resolve()
    if args.action=='prepare':
        images=args.images+(sorted(args.input_dir.glob('*.png')) if args.input_dir else [])
        prepare(root,images,args.config,args.from_run,args.cases,args.seed_stage,args.seed_run)
    else:run(root)
