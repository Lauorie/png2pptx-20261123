"""Rolling discovery -> geometry -> SAM/global omissions -> dedup/style, with shared API slots."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

TERMINAL={'completed','completed_with_issues','failed'}
REPORT_LOCK=threading.Lock()

def now():return datetime.now().astimezone().isoformat(timespec='seconds')
def read(p, default=None):return json.loads(p.read_text()) if p.exists() else default
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    t.replace(p)

def process_alive(record):
    """Adopt only the recorded stage process, never an unrelated reused PID."""
    if not record.get('pid') or not record.get('command'):return False
    try:
        proc=Path('/proc')/str(record['pid'])
        if (proc/'stat').read_text().split()[2]=='Z':return False
        return [x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]==record['command']
    except (OSError,UnicodeError):return False

def stage_finished(root,cid,stage):
    if stage=='geometry':
        return read(root/'geometry/cases'/cid/'completed.json',{}).get('state') in {'completed','completed_with_issues'}
    case=root/'localization/cases'/cid
    return (read(case/'status.json',{}).get('state') in {'completed','completed_with_issues'}
            and all((case/name).is_file() for name in ('result.json','reconstruction_handoff.json')))

def detail(root,cid,status):
    stage=status.get('stage')
    if stage=='localization':p=root/'localization/cases'/cid/'status.json'
    elif stage=='sam_scan':
        p=root/'sam/missing/cases'/cid/'status.json'
    else:p=root/'geometry/cases'/cid/'progress.json'
    return read(p,{})

def publish(root):
    with REPORT_LOCK:
        ids=read(root/'cases.json');cfg=read(root/'config.json')
        rows=[]
        for cid in ids:
            s=read(root/'case_status'/f'{cid}.json',dict(state='queued',stage='queued'))
            rows.append(dict(case=cid,**s,detail=detail(root,cid,s)))
        states=Counter(r['state'] for r in rows)
        stats=dict(updated_at=now(),total=len(ids),ended=sum(states[s] for s in TERMINAL),
                   active=states['running'],queued=states['queued'],completed=states['completed'],
                   completed_with_issues=states['completed_with_issues'],failed=states['failed'],
                   case_concurrency=cfg['case_workers'],api_concurrency=cfg['api_concurrency'],cases=rows)
        write(root/'progress.json',stats)
        head=['# 几何后补漏 · 全局发现与SAM单候选核验','',
              f"更新 {stats['updated_at']}；结束 {stats['ended']}/{stats['total']}，运行 {stats['active']}，排队 {stats['queued']}；失败 {stats['failed']}，带问题结束 {stats['completed_with_issues']}。",'',
              f"{cfg['case_workers']}图流水并发；所有阶段共享API{cfg['api_concurrency']}上限；默认思考，未设置reasoning_effort。",
              ('复用候选整理、初次几何与补漏候选 → 新版逐框审核 → 新增几何/critic → 联合整理 → 样式/critic → ' if cfg.get('resume_before_missing_from') else
               ('复用已完成的候选整理、几何、补漏和联合整理 → 样式/critic → ' if cfg.get('resume_after_dedup_from') else
               ('复用已完成的候选整理 → ' if cfg.get('localization_reuse_from') else '定位整理 → ')+
               '初次几何/critic → 全局GLM漏区发现与SAM单候选核验 → 新增几何/critic → 联合去重 → 样式/critic → '))+
              ('逐条头型恢复 → ' if cfg.get('head_recovery_enabled') else '')+'PPT。SAM扫描不调用GLM补漏；已有几何只读。',
              f"初次几何及几何critic输入长边按原始crop尺寸限制到1080～1536，区间内不变，保持宽高比和crop覆盖；未改定位图尺寸。每个GLM逻辑请求最多{cfg['request_max_attempts']}次真实调用（含格式修复），单次超时{cfg['request_timeout_seconds']/60:g}分钟。",
              '头尺寸固定mid，普通线条箭头线宽原图2px下限。执行完成与视觉正确性分开；本轮尚未人工验收。','',
              '[配置](../config.json) · [来源](../provenance.json) · [进度JSON](../progress.json) · [验证](../preflight/validation.json)','']
        if cfg.get('stop_after_reconciliation',False):
            discovery='GLM全局发现＋SAM候选' if cfg.get('post_geometry_global_discovery_enabled',True) else 'SAM候选（GLM全局发现关闭）'
            head=['# 几何恢复、SAM补漏与联合整理：样式前停止','',
                f"更新 {stats['updated_at']}；结束 {stats['ended']}/{stats['total']}，运行 {stats['active']}，排队 {stats['queued']}，失败 {stats['failed']}。",'',
                f"{cfg['case_workers']}图并发、总API上限{cfg['api_concurrency']}；3 key配置通过仓库外环境加载；thinking开启，省略reasoning_effort。" if cfg.get('credential_key_count')==3 else f"总API上限{cfg['api_concurrency']}；thinking开启，省略reasoning_effort。",
                ('复用候选整理、初次几何与补漏候选 → 新版逐框审核 → 新增框合并/过滤 → 新增几何/critic → 联合整理 → 几何PNG/PPT。' if cfg.get('resume_before_missing_from') and cfg.get('missing_box_consolidation_enabled') else
                 ('复用候选整理、初次几何与补漏候选 → 新版逐框审核 → 新增几何/critic → 联合整理 → 几何PNG/PPT。' if cfg.get('resume_before_missing_from') else
                  f'复用候选整理 → 初次几何/critic → {discovery}及原有uncertain队列逐框审核 → 新增几何/critic → 联合整理 → 几何PNG/PPT。')),
                'GLM全局补漏：'+('开启' if cfg.get('post_geometry_global_discovery_enabled',True) else '关闭')+'。',
                '不调用样式恢复、样式critic或独立头型模块。预览使用几何默认样式；完成状态不代表视觉正确。','',
                '[配置](../config.json) · [来源](../provenance.json) · [进度JSON](../progress.json)','']
        sections=[]
        def img(p,label):
            if not p.exists():return '等待'
            url='../'+str(p.relative_to(root))
            return f'[![{label}]({url})]({url})'
        for row in rows:
            cid=row['case'];d=row['detail'];g=root/'geometry/cases'/cid
            latest=read(root/'localization/cases'/cid/'latest.json',{})
            loc=root/'localization'/latest.get('overlay','not_returned.png')
            sam=root/'sam/missing/cases'/cid/'retained_candidates.png'
            sections += [f'## {cid}','',
                f"{row['state']} · {row['stage']} / {d.get('stage','')}；几何 {d.get('ended',0)}/{d.get('total','?')}；样式 {d.get('style_ended',0)}/{d.get('style_total','?')}",'',
                f'[定位trace](../localization/cases/{cid}/) · [SAM候选](../sam/missing/cases/{cid}/) · [几何后逐框补漏](../geometry/cases/{cid}/post_geometry_missing/) · [几何/去重/样式trace](../geometry/cases/{cid}/) · [PPT](../geometry/cases/{cid}/arrows.pptx) · [纯重建](../geometry/cases/{cid}/result/styled.png)','',
                '| 原图 | 当前重建叠图 |','|---|---|',
                '| '+img(root/'localization/inputs'/cid/'source.png','原图')+' | '+img(g/'result/overlap.png','重建')+' |','',
                '| 初始定位整理 | SAM过滤后候选 |','|---|---|',
                '| '+img(loc,'定位')+' | '+img(sam,'补漏整理')+' |','']
        report=f"reports/{len(ids)}图_实时对照.md"
        write(root/report,'\n'.join(head+sections)+'\n')
        # This entry stays lightweight; the full comparison has 125 sections.
        write(root/'RESULTS.md','\n'.join(h.replace('../','') for h in head)+
              f'\n[{len(ids)}图实时对照]({report})\n')
        if cfg.get('comparison_run'):
            comparison=['# 旧版与新版：联合整理后的几何','',
                '两列均为样式恢复前的几何，附源图与请求。',
                '旧版：'+Path(cfg['comparison_run']).name+'；新版：'+root.name+'。',
                'GLM全局补漏：旧版'+('开启' if read(Path(cfg['comparison_run'])/'config.json',{}).get('post_geometry_global_discovery_enabled',True) else '关闭')+'，新版'+('开启' if cfg.get('post_geometry_global_discovery_enabled',True) else '关闭')+'。',
                '复用边界与本次改动见配置、来源和运行说明。','']
            def compare_image(path,label):
                if not path.is_file():return '等待结果'
                rel=os.path.relpath(path,root/'reports')
                return f'[![{label}]({rel})]({rel})'
            for cid in ids:
                old=Path(cfg['comparison_run'])/'geometry/cases'/cid
                new=root/'geometry/cases'/cid
                comparison += [f'## {cid}','',
                    compare_image(root/'localization/inputs'/cid/'source.png','原图'),'',
                    '| 旧版几何 | 新版几何 |','|---|---|',
                    '| '+compare_image(old/'after_dedup/result/styled.png','旧版')+' | '+compare_image(new/'after_dedup/result/styled.png','新版')+' |',
                    '| '+compare_image(old/'after_dedup/result/overlap.png','旧版叠图')+' | '+compare_image(new/'after_dedup/result/overlap.png','新版叠图')+' |','',
                    f'[新版请求](../geometry/cases/{cid}/) · [新版问题](../geometry/cases/{cid}/completed.json)','']
            comparison_name=f'{len(ids)}图_新旧几何对照.md'
            write(root/'reports'/comparison_name,'\n'.join(comparison)+'\n')
            with (root/'RESULTS.md').open('a') as stream:
                stream.write(f'\n[{len(ids)}图新旧几何对照](reports/{comparison_name})\n')

def run(root):
    lock=(root/'writer.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name,h in read(root/'manifest.json')['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen file changed: '+name)
    cfg=read(root/'config.json');ids=read(root/'cases.json')
    write(root/'execution.json',dict(state='running',pid=os.getpid(),started_at=now()))
    write(root/'sam/execution.json',dict(state='running',started_at=now()))
    stop=threading.Event()
    def watch():
        while not stop.wait(15):publish(root)
    watcher=threading.Thread(target=watch,daemon=True);watcher.start()
    def case(cid):
        p=root/'case_status'/f'{cid}.json'
        if read(p,{}).get('state') in TERMINAL:return
        start=time.monotonic()
        try:
            for stage in ('localization','geometry'):
                previous=read(root/'processes'/cid/(stage+'.json'),{})
                adopted=process_alive(previous)
                if not adopted and stage_finished(root,cid,stage):continue
                if stage=='localization' and cfg.get('localization_reuse_from'):
                    raise RuntimeError('Frozen reused localization is incomplete; refusing to re-run discovery/review')
                write(p,dict(state='running',stage=stage,started_at=now(),elapsed_seconds=time.monotonic()-start))
                if stage=='geometry':
                    cmd=[sys.executable,'-B','-u',str(root/'code/reconstruct.py'),'--run-dir',str(root/'geometry/jobs'/cid)]
                else:
                    cmd=[sys.executable,'-B','-u',str(root/'code/stage.py'),'--root',str(root),'--case',cid,'--stage',stage]
                log=root/'logs'/cid/(stage+'.log');log.parent.mkdir(parents=True,exist_ok=True)
                if adopted:
                    write(root/'processes'/cid/(stage+'_adopted.json'),dict(pid=previous['pid'],adopted_at=now(),controller_pid=os.getpid()))
                    while process_alive(previous):time.sleep(.5)
                    code=0 if stage_finished(root,cid,stage) else 1
                else:
                    with log.open('a') as output:
                        proc=subprocess.Popen(cmd,stdout=output,stderr=subprocess.STDOUT,cwd=root)
                        write(root/'processes'/cid/(stage+'.json'),dict(pid=proc.pid,stage=stage,started_at=now(),command=cmd))
                        code=proc.wait()
                if code:raise RuntimeError(f'{stage} exited {code}; see {log}')
            g=root/'geometry/cases'/cid
            result=read(g/'completed.json')
            if not result:raise RuntimeError('Geometry did not complete: '+str(read(g/'fatal.json',{})))
            write(p,dict(state=result['state'],stage='finished',finished_at=now(),elapsed_seconds=time.monotonic()-start))
        except Exception as exc:
            write(p,dict(state='failed',stage=stage,finished_at=now(),error=str(exc),traceback=traceback.format_exc()))
        print(cid,read(p)['state'],flush=True)
        publish(root)
    try:
        publish(root)
        with ThreadPoolExecutor(max_workers=cfg['case_workers']) as pool:
            # Existing stage processes count against the same case cap. Resume
            # them first, then explicitly requeued failures, then new cases.
            def priority(cid):
                if any(process_alive(read(root/'processes'/cid/(stage+'.json'),{})) for stage in ('localization','geometry')):return 0
                return 1 if read(root/'case_status'/f'{cid}.json',{}).get('resume_after_fix') else 2
            jobs=[pool.submit(case,cid) for cid in sorted(ids,key=priority)]
            for future in as_completed(jobs):future.result()
        write(root/'execution.json',dict(state='finished',pid=os.getpid(),finished_at=now()))
        write(root/'sam/execution.json',dict(state='finished',finished_at=now()))
    finally:
        stop.set();watcher.join();publish(root)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--report-only',action='store_true')
    a=ap.parse_args();(publish if a.report_only else run)(a.run_dir.resolve())
