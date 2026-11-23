"""Frozen style-only experiments, also usable after the integrated geometry stage."""
import fcntl
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from ..common import read,save,sha,now,usage
from . import options
from .controller import run_case


@contextmanager
def heartbeat(callback):
    stop=threading.Event()
    def tick():
        while not stop.wait(5):callback()
    thread=threading.Thread(target=tick,daemon=True);thread.start()
    try:yield
    finally:stop.set();thread.join()


def run(workspace, root, config, items, runtime, *, source, selected=None, rules_only=False):
    workspace=Path(workspace);root=Path(root);source=Path(source)
    if root.resolve()==source.resolve():raise ValueError('Style outputs must be separate from geometry inputs')
    opt=options(config);manifest={'images':items,'geometry_source':str(source.resolve()),'selected':selected,'rules_only':rules_only,'files':{}}
    source_items={i['id']:i for i in read(source/'manifest.json')['images']}
    for item in items:
        ident=item['id'];case=source/'cases'/ident/'fitting'
        if source_items.get(ident)!=item:raise ValueError('Style input manifest mismatch: '+ident)
        status=read(case/'summary.json').get('state')
        if status not in ('completed','completed_with_unresolved'):raise ValueError('Geometry is not terminal: '+ident)
        scene=read(case/'scene.json')
        if scene['source_sha256']!=sha(workspace/item['original']):raise ValueError('Source image hash mismatch')
        files=[case/'scene.json',case/'summary.json',workspace/item['original']]
        files += [case/o['fit_path'] for o in scene['objects'].values() if o.get('fit_path')]
        for p in files:manifest['files'][str(p.resolve())]=sha(p)
        for o in scene['objects'].values():
            if o.get('fit_path') and sha(case/o['fit_path'])!=o['fit_sha256']:raise ValueError('Selected fit changed')
        asset=case.parent/'dedup/image_boxes_original.json'
        if asset.exists():manifest['files'][str(asset.resolve())]=sha(asset)
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name,value in [('manifest.json',manifest),('config.json',config)]:
            path=root/name
            if path.exists() and read(path)!=value:raise ValueError('Frozen style run identity changed: '+name)
            save(path,value)
        sources=[*workspace.joinpath('src').rglob('*.py'),workspace/'run.py',workspace/'run_styling.py']
        sources += [workspace/'src/shape_integration/styling/SOURCE.json']
        hashes={str(p.relative_to(workspace)):sha(p) for p in sources}
        provenance=root/'code_sha256.json'
        if provenance.exists():
            if read(provenance)!=hashes:raise ValueError('Code changed; use a new style run directory')
        else:
            for p in sources:
                dest=root/'code_snapshot'/p.relative_to(workspace);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
            save(provenance,hashes)
        states={};summaries={};mutex=threading.Lock();renders=threading.BoundedSemaphore(opt['render_workers'])
        pixels=threading.BoundedSemaphore(opt['pixel_workers'])
        def progress(ident,stage,done,total):
            with mutex:
                states[ident]={'stage':stage,'done':done,'total':total,'updated_utc':now()}
                save(root/'progress.json',{'state':'running','cases':states,'api':runtime.slots.snapshot() if runtime else None,'updated_utc':now()})
                print(f'STYLE {ident} {stage} {done}/{total}',flush=True)
        def refresh():
            with mutex:
                save(root/'progress.json',{'state':'running','cases':states,'api':runtime.slots.snapshot() if runtime else None,'updated_utc':now()})
                report()
        def report():
            save(root/'summary.json',{'state':'running' if len(summaries)<len(items) else
                                     'completed_with_unresolved' if any(s['state'] not in ('completed','rules_only') for s in summaries.values()) else 'rules_only' if rules_only else 'completed',
                                     'cases':summaries,'updated_utc':now(),'visual_accuracy_verified':False})
            lines=['# 样式整合实验','',f'几何来源：`{source.resolve()}`。只运行样式，不重新拟合。',
                   '', '| 图 | 状态 | 目标部件 | critic 接受 | 未决 |','| --- | --- | ---: | ---: | ---: |']
            for item in items:
                ident=item['id'];r=summaries.get(ident,{})
                lines.append(f'| [{ident}](cases/{ident}/styling/REPORT.zh-CN.md) · [PPT](cases/{ident}/styling/styled.pptx) · [进度](cases/{ident}/styling/progress.json) | {r.get("state","running")} | {r.get("targeted_parts","—")} | {r.get("critic_accepted","—")} | {r.get("unresolved_parts","—")} |')
            for item in items:
                ident=item['id']
                if (root/f'cases/{ident}/styling/comparison.png').exists():
                    lines.extend(['',f'![{ident} 原图与实际 PPT](cases/{ident}/styling/comparison.png)',''])
            temporary=root/'REPORT.zh-CN.md.tmp'
            temporary.write_text('\n'.join(lines)+'\n');temporary.replace(root/'REPORT.zh-CN.md')
        report()
        with heartbeat(refresh), ThreadPoolExecutor(max_workers=opt['workers']) as pool, ThreadPoolExecutor(max_workers=config['case_workers']) as cases:
            jobs={}
            for item in items:
                ident=item['id'];case=source/'cases'/ident;dest=root/'cases'/ident/'styling';dest.mkdir(parents=True,exist_ok=True)
                original=dest/'original.png'
                if original.exists() and sha(original)!=item['source_sha256']:raise ValueError('Frozen source image changed')
                if not original.exists():shutil.copy2(workspace/item['original'],original)
                scene=read(case/'fitting/scene.json');assets=read(case/'dedup/image_boxes_original.json')['assets'] if (case/'dedup/image_boxes_original.json').exists() else []
                objects=selected.get(ident) if selected else None
                save(dest/'source.json',{'geometry_scene':str((case/'fitting/scene.json').resolve()),'scene_sha256':sha(case/'fitting/scene.json'),
                                         'original':str((workspace/item['original']).resolve()),'source_sha256':item['source_sha256']})
                future=cases.submit(run_case,scene,original,dest,runtime,opt,pool,renders,assets=assets,selected_objects=objects,
                                    rules_only=rules_only,pixel_slots=pixels,progress=lambda stage,done,total,ident=ident:progress(ident,stage,done,total))
                jobs[future]=ident
            for future in as_completed(jobs):
                ident=jobs[future]
                try:result=future.result()
                except Exception as exc:
                    import traceback
                    result={'state':'failed','error':str(exc),'traceback':traceback.format_exc()}
                    save(root/'cases'/ident/'styling/runner_error.json',result)
                with mutex:
                    summaries[ident]=result;report()
        changed=[p for p,d in manifest['files'].items() if sha(p)!=d]
        save(root/'input_validation.json',{'unchanged':not changed,'changed':changed})
        save(root/'usage.json',usage(root/'cases'))
        summary=read(root/'summary.json')
        save(root/'progress.json',{'state':summary['state'],'cases':states,'api':runtime.slots.snapshot() if runtime else None,'updated_utc':now()})
        if changed:raise ValueError('Source geometry/input changed during style run')
        return summary
