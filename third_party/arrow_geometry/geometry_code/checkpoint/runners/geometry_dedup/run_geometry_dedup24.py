"""Reviewed localization -> A geometry/critic -> existing local reconciliation."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path

TERMINAL = {'completed', 'completed_with_issues', 'failed', 'upstream_failed'}
REPORT_LOCK = threading.Lock()

def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')

def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default

def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+'.tmp')
    tmp.write_text(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def ready(source, cid):
    case=source/'cases'/cid
    return (read(case/'status.json',{}).get('state') in {'completed','completed_with_issues'}
            and all((case/name).exists() for name in ('result.json','reconstruction_handoff.json')))

def freeze_case(source, root, cid):
    if not ready(source,cid):
        raise ValueError('Localization review is not finished: '+cid)
    out=root/'cases'/cid;out.mkdir(parents=True,exist_ok=True)
    files={'clean.png':source/'inputs'/cid/'source.png',
           'reconstruction_handoff.json':source/'cases'/cid/'reconstruction_handoff.json',
           'localization_result.json':source/'cases'/cid/'result.json',
           'localization_status.json':source/'cases'/cid/'status.json',
           'localization.png':source/'cases'/cid/'current.png'}
    audit={}
    for name,src in files.items():
        expected=sha(src);dst=out/name
        if dst.exists() and sha(dst)!=expected:
            raise ValueError('Previously frozen input changed: '+str(src))
        if not dst.exists():shutil.copy2(src,dst)
        if sha(src)!=expected or sha(dst)!=expected:
            raise ValueError('Input changed while freezing: '+str(src))
        audit[name]={'source':str(src),'sha256':expected}
    write(out/'input_provenance.json',audit)
    return out

def publish(root):
    with REPORT_LOCK:
        cfg=read(root/'config.json');source=Path(cfg['source_localization']);ids=read(root/'cases.json')
        records=[];sections=[];assets=root/'reports/assets';assets.mkdir(parents=True,exist_ok=True)
        def picture(cid,name,path):
            if not path.exists():return '等待'
            stamp=path.stat().st_mtime_ns;dst=assets/f'{cid}_{name}_{stamp}.png'
            if not dst.exists():shutil.copy2(path,dst)
            url='assets/'+dst.name
            return f'[![{name}]({url})]({url})'
        for cid in ids:
            out=root/'cases'/cid
            status=read(out/'progress.json',dict(state='queued',stage='waiting_localization' if not ready(source,cid) else 'queued'))
            record=dict(case=cid,**status);records.append(record)
            src=out/'clean.png' if (out/'clean.png').exists() else source/'inputs'/cid/'source.png'
            before=out/'before_dedup/result/overlap.png';after=out/'result/overlap.png'
            sections += [f'## {cid}', '',
                f"{status['state']} · {status['stage']} · 几何目标 {status.get('ended',0)}/{status.get('total','?')}；失败目标 {status.get('failed',0)}。",'',
                f'[PPT](../cases/{cid}/arrows.pptx) · [纯重建](../cases/{cid}/result/styled.png) · [去重前PPT](../cases/{cid}/before_dedup/arrows.pptx) · [定位框](../cases/{cid}/localization.png) · [几何trace](../cases/{cid}/geometry_tasks/) · [去重trace](../cases/{cid}/reconciliation/) · [问题](../cases/{cid}/completed.json)','',
                '| 原图 | 当前重建叠图 |','|---|---|',
                '| '+picture(cid,'source',src)+' | '+picture(cid,'overlay',after)+' |','',
                '| 去重前几何叠图 | 去重后几何叠图 |','|---|---|',
                '| '+picture(cid,'before_dedup',before)+' | '+(picture(cid,'after_dedup',after) if status.get('dedup_finished') else '等待去重完成')+' |','']
        ended=sum(r['state'] in TERMINAL for r in records)
        progress=dict(updated_at=now(),total=len(ids),ended=ended,
            geometry_finished=sum(bool(r.get('geometry_finished')) for r in records),
            dedup_finished=sum(bool(r.get('dedup_finished')) for r in records),
            active=sum(r['state']=='running' for r in records),case_concurrency=cfg['case_workers'],
            api_concurrency=cfg['api_concurrency'],cases=records)
        write(root/'progress.json',progress)
        header=['# 24图：当前定位接入几何恢复与模型去重','',
            f"更新 {progress['updated_at']}；结束 {ended}/{len(ids)}，几何结束 {progress['geometry_finished']}，去重结束 {progress['dedup_finished']}。",'',
            '输入为v031 receiver_v3_clamp审核后的原框与新增框；定位/SAM补漏不重跑。几何、一次几何critic及一次局部联合去重重新调用。块状箭头保留独立shape/crop SAM分支；本轮去重模块处理普通线条/箭头，块状形状不混入点线路径。',
            f"默认思考；{cfg['case_workers']}图并发、API{cfg['api_concurrency']}；普通线默认2px、头med/med，无精细样式、额外补漏或范围审核。", 
            '运行中叠图只展示已返回目标。完成状态、路径减少和模型accept不代表视觉正确。去重失败或不确定时保留原路径。','',
            '[方案与配置](../README.md) · [执行状态](../execution.json) · [交接校验](../preflight/handoff_checks.json)','']
        write(root/'reports/24图_几何去重实时对照.md','\n'.join(header+sections)+'\n')
        write(root/'RESULTS.md','\n'.join(header)+'\n\n[24图实时对照](reports/24图_几何去重实时对照.md)\n')

def run(root):
    lock=(root/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name,expected in read(root/'manifest.json')['files'].items():
        if sha(root/name)!=expected:raise ValueError('Frozen file changed: '+name)
    cfg=read(root/'config.json');ids=read(root/'cases.json');source=Path(cfg['source_localization'])
    sys.path[:0]=[str(root/'geometry_code'),str(root/'geometry_code/lib')]
    from PIL import Image
    import bbox_inputs,bbox_geometry,pipeline,protocol,viz,reconcile_stage
    from scene_snapshot import canonical_rows
    from native_runtime import validated_renderer
    from native_foreground import render_layer
    from lib import native
    from transport.model import Model
    renderer,version=validated_renderer()
    write(root/'renderer_preflight.json',dict(executable=str(renderer),version=version,validated_at=now()))
    protocol.OCCLUSION=('Reconstruct visible stroke portions only. Do not infer or draw hidden connectors behind text, nodes or opaque objects. '
        'Separate visible portions may be separate paths. If only the local crop truncates a stroke, use the whole-image context to recover its actually visible continuation. '
        'Crossings alone are not junctions; a group region may contain multiple independent paths.')
    model=Model(cfg,run_dir=root,system=protocol.SYSTEM)
    locks={'cpu':threading.Semaphore(cfg['cpu_workers']),'sam':threading.Semaphore(cfg['sam_workers'])}
    write(root/'process.json',dict(pid=os.getpid(),started_at=now()))
    write(root/'execution.json',dict(state='running',started_at=now()))
    stop=threading.Event()
    def watch():
        while not stop.wait(15):publish(root)
    watcher=threading.Thread(target=watch,daemon=True);watcher.start()

    def case(cid):
        out=freeze_case(source,root,cid);clean=Image.open(out/'clean.png').convert('RGB')
        raw=read(out/'reconstruction_handoff.json');plan=bbox_inputs.normalize(raw,clean.size,cid)
        write(out/'geometry_plan.json',plan);upstream=read(out/'localization_result.json')
        provenance=dict(provider='reviewed_bbox_with_sam_missing',source=str(source),
            upstream_state=upstream['state'],upstream_issues=upstream.get('issues',[]),
            pending_localization=upstream.get('pending_count',0),handoff_sha256=sha(out/'reconstruction_handoff.json'),new_geometry_requests=True)
        write(out/'discovery_provenance.json',provenance)
        results={};start=time.monotonic();last_preview=0;geometry_done=False;dedup_done=False
        def ordered():return [results[t['id']] for t in plan['tasks'] if t['id'] in results]
        def progress(stage,state='running'):
            write(out/'progress.json',dict(stage=stage,state=state,ended=len(results),total=len(plan['tasks']),
                failed=sum(r['state'] in {'failed','unresolved'} for r in results.values()),
                geometry_finished=geometry_done,dedup_finished=dedup_done,seconds=time.monotonic()-start))
        def render(rows,dest):
            dest.mkdir(parents=True,exist_ok=True)
            if not (dest/'clean.png').exists():shutil.copy2(out/'clean.png',dest/'clean.png')
            write(dest/'discovery_provenance.json',provenance)
            with locks['cpu']:
                drawable=pipeline.publish(clean,rows,dest)
                if not any(r.get('geometry_heads_reported') for r in drawable):
                    native.render(dest/'arrows.pptx',dest/'native_geometry',clean.size)
                    shutil.copy2(dest/'native_geometry/styled.png',dest/'result/styled.png')
                    fg=render_layer(dest/'native_geometry/styled.svg',clean.size,dest/'native_geometry')
                    overlay=clean.convert('RGBA');overlay.alpha_composite(fg)
                    overlay.convert('RGB').save(dest/'result/overlap.png')
        def recover(task):
            folder=out/'geometry_tasks'/task['id'];saved=folder/'replay_result.json'
            if saved.exists():return read(saved)
            try:value=bbox_geometry.recover_target(clean,task,folder,model,cfg,locks)
            except Exception as exc:
                value=dict(group=task['id'],state='failed',rows=[],issues=[dict(stage='geometry',target=task['id'],error=str(exc),traceback=traceback.format_exc())])
            write(saved,value);return value
        progress('geometry_and_critic')
        with ThreadPoolExecutor(max_workers=cfg['task_workers']) as pool:
            jobs={pool.submit(recover,t):t['id'] for t in plan['tasks']}
            for future in as_completed(jobs):
                results[jobs[future]]=future.result();progress('geometry_and_critic')
                write(out/'geometry_results_partial.json',ordered())
                if time.monotonic()-last_preview>=cfg['preview_interval_seconds']:
                    try:
                        with locks['cpu']:viz.render(out/'clean.png',canonical_rows(ordered()),out/'result')
                    except Exception as exc:write(out/'preview_error.json',dict(error=str(exc)))
                    last_preview=time.monotonic()
        geometry_done=True;initial=ordered();write(out/'before_reconciliation.json',initial)
        issues=list(plan['issues'])+[e for r in initial for e in r.get('issues',[])]
        progress('before_dedup_render')
        try:render(initial,out/'before_dedup')
        except Exception as exc:
            issue=dict(stage='before_dedup_render',error=str(exc));issues.append(issue);write(out/'before_dedup_render_error.json',issue)
        progress('model_dedup')
        try:final,errors=reconcile_stage.run(clean,initial,out/'reconciliation',cfg,model,locks);issues.extend(errors)
        except Exception as exc:
            final=initial;issue=dict(stage='reconciliation',error=str(exc),retained_original=True)
            issues.append(issue);write(out/'reconciliation/fatal.json',dict(issue,traceback=traceback.format_exc()))
        dedup_done=True;write(out/'geometry_results.json',final);progress('final_render')
        render(final,out)
        unresolved=[k for k,v in results.items() if v['state'] in {'failed','unresolved'}]
        inherited=upstream.get('issues',[])
        state='completed_with_issues' if issues or unresolved or inherited else 'completed'
        write(out/'completed.json',dict(case=cid,state=state,issues=issues,unresolved=unresolved,
            targets=len(plan['tasks']),paths_before=len(canonical_rows(initial)),paths_after=len(canonical_rows(final)),
            upstream_issues=inherited,upstream_state=upstream['state'],visual_accuracy_measured=False))
        write(out/'RESULTS.md',f'# {cid}：几何与去重\n\n状态：{state}\n\n| 去重前 | 去重后 |\n|---|---|\n| ![](before_dedup/result/overlap.png) | ![](result/overlap.png) |\n\n[PPT](arrows.pptx) · [问题](completed.json) · [去重审计](reconciliation/audit.json)\n')
        progress('finished',state);print(cid,state,flush=True)

    pending=set(ids);active={}
    try:
        for cid in ids:
            if (root/'cases'/cid/'completed.json').exists():pending.discard(cid)
        with ThreadPoolExecutor(max_workers=cfg['case_workers']) as pool:
            while pending or active:
                for future,cid in list(active.items()):
                    if not future.done():continue
                    try:future.result()
                    except Exception as exc:
                        write(root/'cases'/cid/'fatal.json',dict(error=str(exc),traceback=traceback.format_exc()))
                        write(root/'cases'/cid/'progress.json',dict(state='failed',stage='failed',error=str(exc)))
                    del active[future]
                for cid in ids:
                    if cid not in pending:continue
                    if ready(source,cid) and len(active)<cfg['case_workers']:
                        pending.remove(cid);active[pool.submit(case,cid)]=cid
                    elif not ready(source,cid):
                        status=read(source/'cases'/cid/'status.json',{})
                        control=read(Path(cfg['source_controller_run'])/'execution.json',{})
                        if status.get('state')=='failed' or control.get('state') in {'finished','failed'}:
                            write(root/'cases'/cid/'progress.json',dict(state='upstream_failed',stage='no_completed_localization'))
                            pending.remove(cid)
                publish(root)
                if pending or active:time.sleep(5)
        write(root/'execution.json',dict(state='finished',finished_at=now()))
        write(root/'finished.json',dict(finished_at=now(),cases_ended=len(ids)))
    except BaseException as exc:
        write(root/'execution.json',dict(state='failed',error=str(exc),finished_at=now()));raise
    finally:stop.set();watcher.join();publish(root)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--report-only',action='store_true')
    a=ap.parse_args();(publish if a.report_only else run)(a.run_dir.resolve())
