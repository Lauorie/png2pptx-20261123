"""Reviewed localization -> A geometry/critic -> existing local reconciliation."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Standalone frozen runners load the same verified copy implementation.
for _copy_root in (Path(__file__).resolve().parents[3], Path(__file__).resolve().parents[1]/'geometry_code'):
    if (_copy_root/'checked_copy.py').is_file():
        sys.path.insert(0,str(_copy_root));break
from checked_copy import copy_file

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
           'localization_status.json':source/'cases'/cid/'status.json'}
    audit={}
    for name,src in files.items():
        expected=sha(src);dst=out/name
        if dst.exists() and sha(dst)!=expected:
            raise ValueError('Previously frozen input changed: '+str(src))
        if not dst.exists():copy_file(src,dst)
        if sha(src)!=expected or sha(dst)!=expected:
            raise ValueError('Input changed while freezing: '+str(src))
        audit[name]={'source':str(src),'sha256':expected}
    # A report image is not geometry input. Discovery publishes latest.overlay
    # relative to its run root; legacy producers used cases/<id>/current.png.
    preview=dict(required=False,state='unavailable')
    try:
        latest=read(source/'cases'/cid/'latest.json',{})
        overlay=latest.get('overlay') if isinstance(latest,dict) else None
        candidates=[source/overlay] if isinstance(overlay,str) and overlay else []
        candidates.append(source/'cases'/cid/'current.png')
        for src in candidates:
            src=src.resolve()
            if source.resolve() not in src.parents or not src.is_file():continue
            digest=sha(src);copy_file(src,out/'localization.png')
            preview.update(state='copied',source=str(src),sha256=digest)
            audit['localization.png']=dict(source=str(src),sha256=digest,required=False)
            break
    except (OSError,ValueError,TypeError) as exc:
        preview.update(state='unavailable',error=str(exc))
    write(out/'localization_preview.json',preview)
    write(out/'input_provenance.json',audit)
    return out

def publish(root):
    with REPORT_LOCK:
        cfg=read(root/'config.json');source=Path(cfg['source_localization']);ids=read(root/'cases.json')
        records=[];sections=[];assets=root/'reports/assets';assets.mkdir(parents=True,exist_ok=True)
        def picture(cid,name,path):
            if not path.exists():return '等待'
            stamp=path.stat().st_mtime_ns;dst=assets/f'{cid}_{name}_{stamp}.png'
            if not dst.exists():copy_file(path,dst)
            url='assets/'+dst.name
            return f'[![{name}]({url})]({url})'
        for cid in ids:
            out=root/'cases'/cid
            status=read(out/'progress.json',dict(state='queued',stage='waiting_localization' if not ready(source,cid) else 'queued'))
            record=dict(case=cid,**status);records.append(record)
            src=out/'clean.png' if (out/'clean.png').exists() else source/'inputs'/cid/'source.png'
            before=out/'before_dedup/result/overlap.png';after=out/'result/overlap.png';dedup=out/'after_dedup/result/overlap.png'
            sections += [f'## {cid}', '',
                f"{status['state']} · {status['stage']} · 几何目标 {status.get('ended',0)}/{status.get('total','?')}；失败目标 {status.get('failed',0)}；样式批次 {status.get('style_ended',0)}/{status.get('style_total',0)}。",'',
                f'[PPT](../cases/{cid}/arrows.pptx) · [纯重建](../cases/{cid}/result/styled.png) · [去重前PPT](../cases/{cid}/before_dedup/arrows.pptx) · [定位框](../cases/{cid}/localization.png) · [几何trace](../cases/{cid}/geometry_tasks/) · [去重trace](../cases/{cid}/reconciliation/) · [样式trace](../cases/{cid}/styles/) · [问题](../cases/{cid}/completed.json)','',
                '| 原图 | 当前重建叠图 |','|---|---|',
                '| '+picture(cid,'source',src)+' | '+picture(cid,'overlay',after)+' |','',
                '| 去重前几何叠图 | 去重后几何叠图 |','|---|---|',
                '| '+picture(cid,'before_dedup',before)+' | '+(picture(cid,'after_dedup',dedup) if status.get('dedup_finished') else '等待去重完成')+' |','']
        ended=sum(r['state'] in TERMINAL for r in records)
        progress=dict(updated_at=now(),total=len(ids),ended=ended,
            geometry_finished=sum(bool(r.get('geometry_finished')) for r in records),
            dedup_finished=sum(bool(r.get('dedup_finished')) for r in records),
            style_finished=sum(bool(r.get('style_finished')) for r in records),
            active=sum(r['state']=='running' for r in records),case_concurrency=cfg['case_workers'],
            api_concurrency=cfg['api_concurrency'],cases=records)
        write(root/'progress.json',progress)
        header=['# 几何恢复 → 全局与SAM逐框补漏 → 去重 → 样式','',
            f"更新 {progress['updated_at']}；结束 {ended}/{len(ids)}，几何结束 {progress['geometry_finished']}，去重结束 {progress['dedup_finished']}。",'',
            '输入为本轮bbox发现产物；初次几何后，全局GLM与过滤后的SAM候选逐框核验补漏。几何、一次几何critic及一次局部联合去重重新调用。块状箭头保留独立shape/crop SAM分支；本轮去重模块处理普通线条/箭头，块状形状不混入点线路径。',
            f"默认思考；{cfg['case_workers']}图并发、API{cfg['api_concurrency']}；普通线默认2px、头med/med，几何预览默认2px；样式使用三类头型、固定mid、普通线条箭头原图2px下限，最多refine一次。补漏只新增；无全图范围修正。", 
            '运行中叠图只展示已返回目标。完成状态、路径减少和模型accept不代表视觉正确。去重失败或不确定时保留原路径。','',
            '[方案与配置](../README.md) · [执行状态](../execution.json) · [交接校验](../preflight/handoff_checks.json)','']
        if cfg.get('stop_after_reconciliation',False):
            discovery='GLM全局发现＋SAM候选' if cfg.get('post_geometry_global_discovery_enabled',True) else 'SAM候选（GLM全局发现关闭）'
            header=['# 几何恢复 → 补漏 → 联合整理，样式前停止','',
                f"更新 {progress['updated_at']}；结束 {ended}/{len(ids)}，初次几何结束 {progress['geometry_finished']}，联合整理结束 {progress['dedup_finished']}。",'',
                f'候选整理复用；初次几何/critic → {discovery}及原有uncertain队列逐框审核 → 新增几何/critic → 联合整理。',
                f"API并发上限{cfg['api_concurrency']}；thinking开启。样式恢复、样式critic和独立头型模块不调用；图片为默认样式的几何预览。",'',
                '完成状态不代表视觉正确。','']
        write(root/'reports/重建实时对照.md','\n'.join(header+sections)+'\n')
        write(root/'RESULTS.md','\n'.join(header)+'\n\n[重建实时对照](reports/重建实时对照.md)\n')

def run(root):
    lock=(root/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name,expected in read(root/'manifest.json')['files'].items():
        if sha(root/name)!=expected:raise ValueError('Frozen file changed: '+name)
    cfg=read(root/'config.json');ids=read(root/'cases.json');source=Path(cfg['source_localization'])
    sys.path[:0]=[str(root/'geometry_code'),str(root/'geometry_code/lib')]
    from PIL import Image
    import bbox_inputs,bbox_geometry,pipeline,protocol,viz,reconcile_stage,style_pipeline,style_contract
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
    style_model=Model(cfg,run_dir=root,system='Use CLEAN as source evidence. Recover styles only; verify the physical endpoint categories. Return the requested JSON.')
    style_model.slots=model.slots
    style_model.blocked=model.blocked
    from request_budget import FileSlots
    locks={'cpu':FileSlots(cfg['shared_cpu_slot_dir'],cfg['cpu_workers']),'sam':FileSlots(cfg['shared_sam_slot_dir'],cfg['sam_workers'])}
    write(root/'process.json',dict(pid=os.getpid(),started_at=now()))
    write(root/'execution.json',dict(state='running',started_at=now()))
    stop=threading.Event()
    def watch():
        while not stop.wait(15):publish(root)
    watcher=threading.Thread(target=watch,daemon=True);watcher.start()

    def case(cid):
        out=freeze_case(source,root,cid);clean=Image.open(out/'clean.png').convert('RGB')
        raw=read(out/'reconstruction_handoff.json');plan=bbox_inputs.normalize(raw,clean.size,cid)
        deferred=plan.get('deferred_candidates',[])
        if deferred and not cfg.get('post_geometry_missing_enabled',False):
            raise ValueError('Deferred localization candidates require post_geometry_missing_enabled')
        write(out/'geometry_plan.json',plan);upstream=read(out/'localization_result.json')
        provenance=dict(provider='localization_then_geometry_then_single_bbox_missing',source=str(source),
            upstream_state=upstream['state'],upstream_issues=upstream.get('issues',[]),
            pending_localization=upstream.get('pending_count',0),handoff_sha256=sha(out/'reconstruction_handoff.json'),new_geometry_requests=not bool(cfg.get('resume_after_dedup_from') or cfg.get('resume_before_missing_from')))
        write(out/'discovery_provenance.json',provenance)
        results={};start=time.monotonic();last_preview=0;geometry_done=False;dedup_done=False;style_done=False;style_ended=0;style_total=0
        def ordered():return [results[t['id']] for t in plan['tasks'] if t['id'] in results]
        def progress(stage,state='running'):
            write(out/'progress.json',dict(stage=stage,state=state,ended=len(results),total=len(plan['tasks']),
                failed=sum(r['state'] in {'failed','unresolved'} for r in results.values()),
                geometry_finished=geometry_done,dedup_finished=dedup_done,style_finished=style_done,style_ended=style_ended,style_total=style_total,seconds=time.monotonic()-start))
        def render(rows,dest):
            dest.mkdir(parents=True,exist_ok=True)
            if not (dest/'clean.png').exists():copy_file(out/'clean.png',dest/'clean.png')
            write(dest/'discovery_provenance.json',provenance)
            with locks['cpu']:
                drawable=pipeline.publish(clean,rows,dest)
                if not any(r.get('geometry_heads_reported') for r in drawable):
                    native.render(dest/'arrows.pptx',dest/'native_geometry',clean.size)
                    copy_file(dest/'native_geometry/styled.png',dest/'result/styled.png')
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
        if cfg.get('resume_after_dedup_from'):
            import resume_checkpoint
            checkpoint=Path(cfg['resume_after_dedup_from'])/cid
            restored,initial,final,restored_issues=resume_checkpoint.load(checkpoint)
            results.update({group['group']:group for group in restored})
            geometry_done=True
            issues=list(plan['issues'])+restored_issues
            write(out/'initial_geometry_results.json',restored)
            write(out/'before_reconciliation.json',initial)
            write(out/'checkpoint_resume.json',read(checkpoint/'provenance.json'))
            write(out/'post_geometry_missing/audit.json',read(checkpoint/'post_geometry_missing/audit.json'))
            write(out/'reconciliation/audit.json',read(checkpoint/'reconciliation/audit.json'))
            progress('restored_geometry_before_render')
            render(initial,out/'before_dedup')
        else:
            if cfg.get('resume_before_missing_from'):
                from resume_missing_checkpoint import load
                initial, _, reused = load(Path(cfg['resume_before_missing_from'])/cid)
                results.update({g['group']: g for g in initial})
                geometry_done = True
                issues = list(plan['issues'])+[e for g in initial for e in g.get('issues', [])]
                write(out/'initial_geometry_results.json', initial)
                write(out/'checkpoint_resume.json', reused)
                progress('restored_initial_geometry')
            else:
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
                geometry_done=True;initial=ordered();write(out/'initial_geometry_results.json',initial)
                issues=list(plan['issues'])+[e for r in initial for e in r.get('issues',[])]
            if cfg.get('post_geometry_missing_enabled',False):
                if cfg.get('post_geometry_sam_scan_enabled',False):
                    progress('post_geometry_sam_scan')
                    controller=Path(cfg['source_controller_run'])
                    log=out/'post_geometry_sam_scan.log'
                    with log.open('a') as stream:
                        scan=subprocess.run([sys.executable,'-B','-u',str(controller/'code/stage.py'),
                            '--root',str(controller),'--case',cid,'--stage','sam_scan'],
                            stdout=stream,stderr=subprocess.STDOUT,cwd=controller)
                    if scan.returncode:
                        issues.append(dict(stage='sam_scan',error=f'exit {scan.returncode}',log=str(log)))
                import post_geometry_missing
                initial,missing_issues=post_geometry_missing.run(clean,initial,out/'post_geometry_missing',cfg,model,locks,cid,
                                                              notify=progress,deferred_candidates=deferred)
                issues.extend(missing_issues)
            write(out/'before_reconciliation.json',initial)
            progress('before_dedup_render')
            try:render(initial,out/'before_dedup')
            except Exception as exc:
                issue=dict(stage='before_dedup_render',error=str(exc));issues.append(issue);write(out/'before_dedup_render_error.json',issue)
            progress('model_dedup')
            try:final,errors=reconcile_stage.run(clean,initial,out/'reconciliation',cfg,model,locks);issues.extend(errors)
            except Exception as exc:
                final=initial;issue=dict(stage='reconciliation',error=str(exc),retained_original=True)
                issues.append(issue);write(out/'reconciliation/fatal.json',dict(issue,traceback=traceback.format_exc()))
        dedup_done=True;write(out/'geometry_results.json',final);progress('after_dedup_render')
        render(final,out/'after_dedup')
        geometry_rows=canonical_rows(final);geometry_signature=style_contract.geometry_signature(geometry_rows)
        if cfg.get('stop_after_reconciliation', False):
            # Publish the already rendered geometry checkpoint, with no style
            # estimate, style critic/refine or independent head requests.
            for name in ('scene.json', 'arrows.pptx', 'result/styled.png', 'result/overlap.png'):
                dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True)
                copy_file(out/'after_dedup'/name,dest)
            write(out/'rows.json',geometry_rows)
            write(out/'style_skipped.json',dict(reason='stop_after_reconciliation',model_calls=0))
            unresolved=[k for k,v in results.items() if v['state'] in {'failed','unresolved'}]
            inherited=upstream.get('issues',[])
            state='completed_with_issues' if issues or unresolved or inherited else 'completed'
            write(out/'completed.json',dict(case=cid,state=state,issues=issues,unresolved=unresolved,
                targets=len(plan['tasks']),paths_before=len(canonical_rows(initial)),paths_after=len(geometry_rows),
                upstream_issues=inherited,upstream_state=upstream['state'],visual_accuracy_measured=False,
                stop_after='reconciliation',style_recovery_run=False))
            write(out/'RESULTS.md',f'# {cid}：几何、SAM补漏与联合整理\n\n状态：{state}；样式恢复未运行。\n\n| 原图 | 几何结果 |\n|---|---|\n| ![](clean.png) | ![](result/overlap.png) |\n\n[PPT](arrows.pptx) · [问题](completed.json) · [联合整理](reconciliation/audit.json)\n')
            progress('finished_before_style',state);print(cid,state,flush=True)
            return
        def style_preview(rows):
            import cairosvg,io
            rgba=Image.open(io.BytesIO(cairosvg.svg2png(bytestring=viz.svg(clean.size,rows).encode()))).convert('RGBA')
            white=Image.new('RGBA',clean.size,'white');white.alpha_composite(rgba)
            overlay=clean.convert('RGBA');overlay.alpha_composite(rgba)
            dest=out/'result';dest.mkdir(exist_ok=True)
            white.convert('RGB').save(dest/'styled.png');overlay.convert('RGB').save(dest/'overlap.png')
        def style_callback(ended,total,updates,errors):
            nonlocal style_ended,style_total,last_preview
            style_ended=ended;style_total=total
            progress('style_recovery');write(out/'partial_styles.json',updates)
            if time.monotonic()-last_preview>=cfg['preview_interval_seconds']:
                from simple_style_contract import Contract,merge_style
                current=Contract().enforce_rows([merge_style(r,updates.get(r['id'],r)) for r in geometry_rows])
                with locks['cpu']:style_preview(current)
                last_preview=time.monotonic()
        progress('style_recovery')
        styled,errors=style_pipeline.finish_case(clean,final,out,cfg,style_model,locks,notify=style_callback)
        issues.extend(errors);styled_rows=canonical_rows(styled)
        assert style_contract.geometry_signature(styled_rows)==geometry_signature
        write(out/'rows.json',styled_rows);write(out/'scene.json',dict(size=list(clean.size),arrows=styled_rows,discovery_provider='bbox_handoff',heads_source='independent_head_recovery' if cfg.get('head_recovery_enabled') else 'style_three_category_review'));write(out/'geometry_preservation.json',dict(exact_geometry_unchanged=True,paths=len(styled_rows)))
        style_done=True;progress('final_native_render')
        with locks['cpu']:
            native.write(clean.size,styled_rows,[r['style'] for r in styled_rows],out/'arrows.pptx',head_floor_tolerance=.02)
            native.render(out/'arrows.pptx',out/'native',clean.size)
            fg=render_layer(out/'native/styled.svg',clean.size,out/'native')
            (out/'result').mkdir(exist_ok=True)
            copy_file(out/'native/styled.png',out/'result/styled.png')
            overlay=clean.convert('RGBA');overlay.alpha_composite(fg);overlay.convert('RGB').save(out/'result/overlap.png')
        unresolved=[k for k,v in results.items() if v['state'] in {'failed','unresolved'}]
        inherited=upstream.get('issues',[])
        state='completed_with_issues' if issues or unresolved or inherited else 'completed'
        write(out/'completed.json',dict(case=cid,state=state,issues=issues,unresolved=unresolved,
            targets=len(plan['tasks']),paths_before=len(canonical_rows(initial)),paths_after=len(canonical_rows(final)),
            upstream_issues=inherited,upstream_state=upstream['state'],visual_accuracy_measured=False))
        write(out/'RESULTS.md',f'# {cid}：几何、去重与样式\n\n状态：{state}\n\n| 原图 | 样式恢复结果 |\n|---|---|\n| ![](clean.png) | ![](result/overlap.png) |\n\n[PPT](arrows.pptx) · [问题](completed.json) · [去重审计](reconciliation/audit.json)\n')
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
