"""Compose frozen shapes, arrows, images and text into a final PPT without models."""
import argparse
import fcntl
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'vendor')]
from shape_integration.common import read, save, sha, now
from shape_integration.composition import PROTOCOL, base_ppt, resolve_registry, compose_case


def run(source,output,items,*,image_source=None,workers=4,render_workers=2):
    if source.resolve()==output.resolve() or source.resolve() in output.resolve().parents:
        raise ValueError('Use a separate output directory')
    state=read(source/'summary.json')['state']
    if state not in ('completed','completed_with_unresolved'):raise ValueError('Source run is not complete')
    source_items={i['id']:i for i in read(source/'manifest.json')['images']}
    jobs=[];files={};lineage={}
    for item in items:
        if item!=source_items.get(item['id']):raise ValueError('Source input manifest mismatch')
        if sha(ROOT/item['original'])!=item['source_sha256']:raise ValueError('Original input changed')
        case=source/'cases'/item['id'];base=base_ppt(case)
        if base is None:raise ValueError('Missing source PPT: '+item['id'])
        registry,chain=resolve_registry(ROOT,image_source or source,item)
        text=case/'texts/scene.json';text=text if text.exists() else None
        jobs.append((item,base,registry,text));lineage[item['id']]=chain
        for p in [base,registry,*([text] if text else [])]:files[str(p.resolve())]=sha(p)
        for link in chain:files[link['path']]=link['sha256']
    manifest={'protocol':PROTOCOL,'images':items,'composition_source':str(source.resolve()),
              'explicit_image_source':str(image_source.resolve()) if image_source else None,
              'source_files':files,'image_source_lineage':lineage,'model_calls':0}
    output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();results={};slots=threading.BoundedSemaphore(render_workers)
    with (output/'.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name,value in [('manifest.json',manifest)]:
            if (output/name).exists() and read(output/name)!=value:raise ValueError('Run input identity changed')
            save(output/name,value)
        code=[*ROOT.joinpath('src').rglob('*.py'),ROOT/'run_complete.py',ROOT/'run.py']
        hashes={str(p.relative_to(ROOT)):sha(p) for p in code}
        if (output/'code_sha256.json').exists() and read(output/'code_sha256.json')!=hashes:
            raise ValueError('Code changed; use a new output run')
        save(output/'code_sha256.json',hashes)
        for p in code:
            dest=output/'code_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():shutil.copy2(p,dest)
        def progress():
            stage='running' if len(results)<len(items) else 'completed_with_unresolved' if any(v['state']!='completed' for v in results.values()) else 'completed'
            save(output/'summary.json',{'state':stage,'cases':results,'updated_utc':now(),'model_calls':0,'visual_accuracy_verified':False})
            save(output/'progress.json',{'state':stage,'done':len(results),'total':len(items),'elapsed_seconds':time.monotonic()-started})
        def process(item,base,registry,text):
            folder=output/'cases'/item['id']/'complete'
            result=compose_case(ROOT,folder,item,base,registry,slots,text_scene=text)
            if state=='completed_with_unresolved':
                result['upstream_run_state']=state
                if result['state']=='completed':result['state']='completed_with_unresolved'
                save(folder/'summary.json',result)
            return result
        progress()
        with ThreadPoolExecutor(workers) as pool:
            futures={pool.submit(process,*job):job[0]['id'] for job in jobs}
            for future in as_completed(futures):
                ident=futures[future]
                try:results[ident]=future.result()
                except Exception as exc:
                    results[ident]={'state':'failed','error':str(exc)}
                    save(output/'cases'/ident/'complete/runner_error.json',results[ident])
                print(ident,results[ident],flush=True);progress()
        for p,expected in files.items():
            if sha(p)!=expected:raise ValueError('Frozen composition source changed: '+p)
        for item in items:
            if sha(ROOT/item['original'])!=item['source_sha256']:raise ValueError('Original changed during composition')
        save(output/'audit.json',{'source_files_unchanged':True,'model_calls':0,
             'images':sum(v.get('images',0) for v in results.values()),
             'editable_textboxes':sum(v.get('editable_textboxes',0) for v in results.values()),
             'all_images_below_text':all(v.get('images_below_text',False) for v in results.values()),
             'all_base_objects_unchanged':all(v.get('base_objects_unchanged',False) for v in results.values()),
             'visual_accuracy_verified':False})
        lines=['# 图片 + 形状 + 箭头 + 文字／公式：完整合成','',
               '复用已完成的各阶段结果，零模型调用。图片位于原生文字／公式下方；原有形状、箭头和文字对象保持不变。',
               '', '| 图 | 图片 | 原生文本框 | 边界未决 | 结果 |','| --- | ---: | ---: | ---: | --- |']
        for item in items:
            ident=item['id'];v=results[ident]
            lines.append(f'| {ident} | {v.get("images",0)} | {v.get("editable_textboxes",0)} | {len(v.get("unresolved_image_boundaries",[]))} | [报告](cases/{ident}/complete/REPORT.zh-CN.md) · [PPT](cases/{ident}/complete/combined.pptx) |')
        lines+=['','图片使用当前同源清单：有已导出且身份匹配的 RGBA 时复用，否则原样裁切原图。标为 alpha 但没有有效遮罩的目标保留矩形原像素，并记录回退；未决边界不因插入成功而变成已确认。',
                '', '本次保留上一轮全部文字。图片内部已带字时可能有重复，候选列在逐图 text_image_overlaps.json；新完整运行先插入图片，再让 OCR 通过 IMG 标注排除图片内部已保留文字。']
        for item in items:
            ident=item['id'];lines+=['',f'![{ident}](cases/{ident}/complete/comparison.png)']
        (output/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
        return read(output/'summary.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name',required=True);p.add_argument('--from-run',type=Path,required=True)
    p.add_argument('--image-source',type=Path);p.add_argument('--ids',nargs='+')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--render-workers',type=int,default=2)
    a=p.parse_args()
    if Path(a.run_name).name!=a.run_name or a.run_name in ('.','..'):raise ValueError('Invalid run name')
    if min(a.workers,a.render_workers)<1:raise ValueError('Workers must be positive')
    items=read(ROOT/'inputs/manifest.json')['images']
    if a.ids:
        if not set(a.ids)<={i['id'] for i in items}:raise ValueError('Unknown case ID')
        items=[i for i in items if i['id'] in a.ids]
    result=run((ROOT/a.from_run).resolve(),ROOT/'runs'/a.run_name,items,
               image_source=(ROOT/a.image_source).resolve() if a.image_source else None,
               workers=a.workers,render_workers=a.render_workers)
    return int(any(r['state']=='failed' for r in result['cases'].values()))


if __name__=='__main__':raise SystemExit(main())
