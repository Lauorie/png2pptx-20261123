"""Text-only frozen runs and an append hook for the integrated pipeline."""
import fcntl
import os
import shutil
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..common import save, read, sha, now, usage
from ..styling.runner import heartbeat
from . import PROTOCOL, options
from .controller import run_case


def base_ppt(case):
    return next((case/p for p in ('image_composition/combined.pptx','arrows/combined.pptx','styling/styled.pptx',
                                 'arrows/arrows.pptx','fitting/geometry_final.pptx') if (case/p).is_file()),None)


def audit(root, manifest):
    from pptx import Presentation
    from lxml import etree
    records=[]
    for item in manifest['images']:
        folder=root/'cases'/item['id'];text=folder/'texts'
        summary=read(text/'summary.json') if (text/'summary.json').exists() else {'state':'failed'}
        record={'id':item['id'],'state':summary['state'],'artifacts':{}}
        for name in ('text.pptx','native.png','native.pdf','scene.json','overlay.png'):
            path=text/name;record['artifacts'][name]=sha(path) if path.exists() else None
        base=folder/'base/source.pptx'
        combined=text/'combined.pptx'
        if combined.exists() and base.exists():
            a=Presentation(str(base));b=Presentation(str(combined));n=len(a.slides[0].shapes)
            unchanged=[etree.tostring(s._element) for s in a.slides[0].shapes]==[etree.tostring(s._element) for s in list(b.slides[0].shapes)[:n]]
            if not unchanged:raise ValueError('Combined PPT changed a frozen base object: '+item['id'])
            record['base_objects_unchanged']=True
            record['editable_textboxes']=sum(s.name.startswith('TXT ') for s in list(b.slides[0].shapes)[n:])
            if record['editable_textboxes']!=summary.get('exported_lines'):raise ValueError('Exported text count mismatch')
            record['artifacts']['combined.pptx']=sha(combined)
        if summary['state']!='failed' and (not all(record['artifacts'].values()) or (base.exists() and not combined.exists())):
            raise ValueError('Completed case lacks final artifacts: '+item['id'])
        records.append(record)
    for path,expected in manifest['base_files'].items():
        if sha(path)!=expected:raise ValueError('Source base changed during text run: '+path)
    return {'artifact_checks_passed':True,'source_base_hashes_unchanged':True,'cases':records,
            'visual_accuracy_verified':False}


def append(workspace,folder,item,summary,runtime,config,pool,slots):
    try:result=run_case(workspace/item['original'],folder/'texts',runtime,options(config),pool,slots,base_ppt(folder))
    except Exception as exc:
        result={'state':'failed','error':str(exc)};save(folder/'texts/runner_error.json',result)
    out={**summary,'texts':result}
    if result['state']!='completed' and out['state']=='completed':out['state']='completed_with_unresolved'
    save(folder/'fitting/summary.json',out);return out


def run(workspace,root,config,items,runtime,*,source=None):
    workspace=Path(workspace);root=Path(root);source=Path(source) if source else None
    if source and (root.resolve()==source.resolve() or source.resolve() in root.resolve().parents):
        raise ValueError('Use an independent text run directory')
    opt=options(config);root.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    with (root/'.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        olditems={v['id']:v for v in read(source/'manifest.json')['images']} if source else {}
        if source and read(source/'summary.json').get('state') not in ('completed','completed_with_unresolved'):
            raise ValueError('Text source must be a completed run')
        bases={};manifest={'protocol':PROTOCOL,'images':items,'base_source':str(source) if source else None,'base_files':{}}
        for item in items:
            original=workspace/item['original']
            if sha(original)!=item['source_sha256']:raise ValueError('Original changed: '+item['id'])
            if source:
                previous=olditems.get(item['id'],{})
                if previous.get('source_sha256')!=item['source_sha256'] or previous.get('source_size')!=item['source_size']:
                    raise ValueError('Text base source image differs: '+item['id'])
                base=base_ppt(source/'cases'/item['id'])
                if base is None:raise ValueError('No native PPT base for '+item['id'])
                dest=root/'cases'/item['id']/'base/source.pptx';dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists() and sha(dest)!=sha(base):raise ValueError('Frozen base changed')
                if not dest.exists():shutil.copy2(base,dest)
                bases[item['id']]=dest;manifest['base_files'][str(base)]=sha(base)
        files=[*workspace.joinpath('src').rglob('*.py'),workspace/'run_texts.py',workspace/'run.py']
        hashes={str(p.relative_to(workspace)):sha(p) for p in files}
        for name,value in [('manifest.json',manifest),('config.json',config),('code_sha256.json',hashes)]:
            p=root/name
            if p.exists() and read(p)!=value:raise ValueError('Run identity changed; use a new directory: '+name)
            save(p,value)
        for p in files:
            dest=root/'code_snapshot'/p.relative_to(workspace);dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():shutil.copy2(p,dest)
        results={};states={};mutex=threading.Lock();slots=threading.BoundedSemaphore(opt['render_workers'])
        def progress(ident,stage,done,total):
            with mutex:states[ident]={'stage':stage,'done':done,'total':total,'updated_utc':now()}
            print(f'TEXT {ident} {stage} {done}/{total}',flush=True)
        def report():
            with mutex:
                stage='running' if len(results)<len(items) else 'completed_with_unresolved' if any(v['state']!='completed' for v in results.values()) else 'completed'
                save(root/'progress.json',{'state':stage,'cases':states,'api':runtime.slots.snapshot(),'updated_utc':now()})
                used=usage(root/'cases');save(root/'usage.json',used)
                save(root/'summary.json',{'state':stage,'cases':results,'usage':used['totals'],'visual_accuracy_verified':False})
                save(root/'process_status.json',{'pid':os.getpid(),'state':stage,'finished_cases':len(results),'total_cases':len(items),
                                                'elapsed_seconds':time.monotonic()-started,'api':runtime.slots.snapshot()})
                lines=['# 轻量文字恢复实验','',f'状态：{stage}；完成 {len(results)}/{len(items)} 图。',
                       '', '| 图 | 状态 | 文字行 | 局部复查 | 结果 |','| --- | --- | ---: | ---: | --- |']
                for item in items:
                    ident=item['id'];r=results.get(ident,{})
                    lines.append(f'| {ident} | {r.get("state",states.get(ident,{}).get("stage","pending"))} | {r.get("text_lines","—")} | {r.get("local_regions","—")} | [查看](cases/{ident}/texts/REPORT.zh-CN.md) |')
                for item in items:
                    ident=item['id'];p=f'cases/{ident}/texts/combined_comparison.png'
                    if not (root/p).exists():p=f'cases/{ident}/texts/comparison.png'
                    if (root/p).exists():lines+=['',f'![{ident}]({p})']
                temp=root/'REPORT.zh-CN.md.tmp';temp.write_text('\n'.join(lines)+'\n');temp.replace(root/'REPORT.zh-CN.md')
        report()
        with heartbeat(report),ThreadPoolExecutor(max_workers=opt['workers']) as pool,ThreadPoolExecutor(max_workers=config['case_workers']) as cases:
            jobs={cases.submit(run_case,workspace/i['original'],root/'cases'/i['id']/'texts',runtime,opt,pool,slots,
                               bases.get(i['id']),lambda stage,done,total,ident=i['id']:progress(ident,stage,done,total)):i for i in items}
            for future in as_completed(jobs):
                item=jobs[future]
                try:result=future.result()
                except Exception as exc:
                    result={'state':'failed','error':str(exc)};save(root/'cases'/item['id']/'texts/runner_error.json',result)
                with mutex:results[item['id']]=result
                report()
        save(root/'final_audit.json',audit(root,manifest))
        report();return read(root/'summary.json')
