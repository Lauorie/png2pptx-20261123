"""Frozen full-suite runner and an optional append stage for the shape pipeline."""
import os
import fcntl
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from ..common import read,save,sha,now,usage
from ..styling.runner import heartbeat
from . import options,PROTOCOL
from .controller import run_case


def append(workspace,folder,item,summary,runtime,config,pool,slots):
    base=next((p for p in [folder/'styling/styled.pptx',folder/'fitting/geometry_final.pptx'] if p.exists()),None)
    try:result=run_case(workspace/item['original'],folder/'arrows',runtime,options(config),pool,slots,base)
    except Exception as exc:
        result={'state':'failed','error':str(exc)};save(folder/'arrows/runner_error.json',result)
    out={**summary,'arrows':result};save(folder/'fitting/summary.json',out);return out


def run(workspace,root,config,items,runtime,*,style_source=None,arrow_source=None,retry_missing_geometry=False):
    if style_source and arrow_source:raise ValueError('Choose one frozen source')
    if retry_missing_geometry and not arrow_source:raise ValueError('Missing-geometry recovery requires an arrow source')
    opt=options(config);root.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    with (root/'.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        files=[*workspace.joinpath('src').rglob('*.py'),*workspace.joinpath('src/shape_integration/arrows/assets').glob('*'),workspace/'run_arrows.py']
        hashes={str(p.relative_to(workspace)):sha(p) for p in files}
        manifest={'protocol':PROTOCOL,'images':items,'style_source':str(style_source) if style_source else None,
                  'arrow_source':str(arrow_source) if arrow_source else None,'retry_missing_geometry':retry_missing_geometry,'base_files':{},'geometry_sources':{}}
        bases={}
        if style_source:
            old=read(style_source/'manifest.json');olditems={i['id']:i for i in old['images']}
            for item in items:
                if olditems.get(item['id'])!=item:raise ValueError('Style source image manifest differs')
                folder=style_source/'cases'/item['id']/'styling'
                if read(folder/'summary.json')['state'] not in ('completed','completed_with_unresolved','rules_only'):raise ValueError('Style source must be terminal')
                if read(folder/'scene_source.json')['source_sha256']!=item['source_sha256']:raise ValueError('Base geometry source differs')
                base=folder/'styled.pptx';digest=sha(base);manifest['base_files'][str(base)]=digest
                target=root/'cases'/item['id']/'base/styled.pptx';target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists() and sha(target)!=digest:raise ValueError('Frozen base changed')
                if not target.exists():shutil.copy2(base,target)
                bases[item['id']]=target
        if arrow_source:
            if read(arrow_source/'summary.json')['state'] not in ('completed','completed_with_unresolved'):raise ValueError('Arrow source run must be terminal')
            olditems={i['id']:i for i in read(arrow_source/'manifest.json')['images']}
            for item in items:
                if olditems.get(item['id'])!=item:raise ValueError('Arrow source image manifest differs')
                source_case=arrow_source/'cases'/item['id'];scene=source_case/'arrows/scene.json'
                if read(scene)['source_sha256']!=item['source_sha256']:raise ValueError('Arrow source geometry image differs')
                manifest['geometry_sources'][str(scene)]=sha(scene)
                base=source_case/'base/styled.pptx'
                if base.exists():
                    digest=sha(base);manifest['base_files'][str(base)]=digest
                    target=root/'cases'/item['id']/'base/styled.pptx';target.parent.mkdir(parents=True,exist_ok=True)
                    if target.exists() and sha(target)!=digest:raise ValueError('Frozen base changed')
                    if not target.exists():shutil.copy2(base,target)
                    bases[item['id']]=target
        for item in items:
            if sha(workspace/item['original'])!=item['source_sha256']:raise ValueError('Original image changed')
        for name,value in [('manifest.json',manifest),('config.json',config),('code_sha256.json',hashes)]:
            path=root/name
            if path.exists() and read(path)!=value:raise ValueError('Frozen run changed; use a new run: '+name)
            save(path,value)
        for p in files:
            target=root/'code_snapshot'/p.relative_to(workspace);target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copy2(p,target)
        states={};results={};mutex=threading.Lock();slots=threading.BoundedSemaphore(opt['render_workers'])
        def report():
            with mutex:
                stage='running' if len(results)<len(items) else 'completed_with_unresolved' if any(s['state']!='completed' for s in results.values()) else 'completed'
                save(root/'progress.json',{'state':stage,'cases':states,'api':runtime.slots.snapshot(),'updated_utc':now()})
                save(root/'summary.json',{'state':stage,'cases':results,'updated_utc':now(),'visual_accuracy_verified':False})
                save(root/'process_status.json',{'pid':os.getpid(),'state':stage,'elapsed_seconds':time.monotonic()-started,'finished_cases':len(results),'total_cases':len(items),'api':runtime.slots.snapshot(),'updated_utc':now()})
                lines=['# 细线箭头整合实验','',f'状态：{stage}，{len(results)}/{len(items)} 图结束。',
                       '', '| 图 | 进度 | 箭头层 | 组合 |','| --- | --- | --- | --- |']
                for item in items:
                    ident=item['id'];s=states.get(ident,{})
                    lines.append(f'| [{ident}](cases/{ident}/arrows/REPORT.zh-CN.md) | {s.get("stage","pending")} {s.get("done",0)}/{s.get("total",0)} | [PPT](cases/{ident}/arrows/arrows.pptx) | [PPT](cases/{ident}/arrows/combined.pptx) |')
                for item in items:
                    im=f'cases/{item["id"]}/arrows/comparison.png'
                    if (root/im).exists():lines+=['',f'![{item["id"]}]({im})']
                temp=root/'REPORT.zh-CN.md.tmp';temp.write_text('\n'.join(lines)+'\n');temp.replace(root/'REPORT.zh-CN.md')
        def progress(ident,stage,done,total):
            with mutex:states[ident]={'stage':stage,'done':done,'total':total}
            print(f'ARROWS {ident} {stage} {done}/{total}',flush=True)
        report()
        with heartbeat(report),ThreadPoolExecutor(max_workers=opt['workers']) as pool,ThreadPoolExecutor(max_workers=config['case_workers']) as cases:
            jobs={cases.submit(run_case,workspace/i['original'],root/'cases'/i['id']/'arrows',runtime,opt,pool,slots,bases.get(i['id']),
                               lambda stage,done,total,ident=i['id']:progress(ident,stage,done,total),
                               frozen=arrow_source/'cases'/i['id']/'arrows' if arrow_source else None,retry_missing_geometry=retry_missing_geometry):i for i in items}
            for fut in as_completed(jobs):
                item=jobs[fut]
                try:result=fut.result()
                except Exception as exc:
                    result={'state':'failed','error':str(exc)};save(root/'cases'/item['id']/'arrows/runner_error.json',result)
                with mutex:results[item['id']]=result
                report()
        report();save(root/'usage.json',usage(root/'cases'));return read(root/'summary.json')
