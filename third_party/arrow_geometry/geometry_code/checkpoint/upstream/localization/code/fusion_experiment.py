"""Replay saved localization through model region fusion, then paired geometry."""
import argparse
import copy
import fcntl
import os
import shutil
import subprocess
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image
import contracts as C
from frames import global_frame, crop_from_global, visible_global_bounds
from handoff import export
from presentation import overlay, compact, now
from fusion_report import report, atomic
from transport.common import read_json as read, write_json as write, sha256
from transport.model import Model


def fusion_prompt(size, rows, focused=False, question=None):
    q=C.review_prompt(size, compact(rows), allow_inspect=not focused, focused=focused)
    q['task']='重新安排这些可见箭头/线条的重建区域。重点检查过细拆分的相关分支、共享主干、扇形和小网络；适合整体恢复就原子替换成group，保留全部不同线条。孤立零星箭头保持独立，信息量过大的组仍拆开。不是把每条线去证明为同一物理箭头，也不是尽量减少框数。'
    q['instructions'] += ['这是已有发现结果的融合对照，不重新扫描全图。仅修正给定区域及直接相关的遗漏；不能让已有group与其内部子任务同时重复保留。',
                          '每个replacement.reason简述是重复观察去重、不同线条的区域合组、拆分还是范围修正，以及可见依据。被文字/节点遮挡的部分不要补画；不为了合组推导隐藏连接。']
    if question:q['inspection_question']=question
    return q


def prepare(root, source, geometry_code, geometry_config, cases, constraints):
    if root.exists():raise ValueError('Run already exists; preserve frozen evidence')
    cfg=read(source/'config.json')
    assert cfg['thinking']=={'type':'enabled'} and 'reasoning_effort' not in cfg and cfg['max_tokens']==113000
    cfg.update(case_workers=10, protocol_version='model_led_regions_v3_useful_work_regions',
               comparison_run=str(source),localization_only=False)
    root.mkdir(parents=True)
    shutil.copytree(Path(__file__).parent,root/'code',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','*.pyc'))
    shutil.copytree(geometry_code,root/'geometry_code',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','*.pyc'))
    shutil.copytree(constraints,root/'constraints_snapshot',ignore=shutil.ignore_patterns('*.png','*.jpg'))
    write(root/'config.json',cfg);write(root/'cases.json',cases)
    gc=read(geometry_config)
    gc.update(case_workers=10,api_concurrency=128,task_workers=16,pipeline_mode='geometry',coverage_rounds=0,
              scope_review=False,reconcile_enabled=False,separate_style=False,geometry_critic_enabled=True,
              geometry_critic_max_refinements=1,geometry_critic_mask_enabled=False)
    assert gc['thinking']=={'type':'enabled'} and 'reasoning_effort' not in gc and gc['max_tokens']==113000
    write(root/'geometry_config.json',gc)
    for cid in cases:
        shutil.copytree(source/'inputs'/cid,root/'inputs'/cid)
        target=root/'cases'/cid;target.mkdir(parents=True)
        for f in ('result.json','reconstruction_handoff.json','coordinate_audit.json'):
            shutil.copy2(source/'cases'/cid/f,target/('before_'+f))
        old=read(target/'before_result.json')
        overlay(root/'inputs'/cid/'global.png',old['regions'],target/'before_boxes.png')
        write(target/'status.json',{'stage':'queued','state':'queued'})
    write(root/'provenance.json',{'source_localization':str(source),'source_geometry_code':str(geometry_code),
          'source_geometry_config':str(geometry_config),'same_geometry_both_arms':True,
          'unchanged_regions_reconstructed_once':True,'no_new_discovery_or_missing_stage':True,
          'geometry_protocol_override':'Visible segments only; no inferred occluded continuation, both arms',
          'sam_policy':'No new localization SAM; keep existing block evidence and fallback configuration'})
    atomic(root/'README.md','''# v003：模型融合恢复区域，再对照几何

复用 v002 的 8 个问题例与 0320、0153 两个孤立箭头对照。先让模型对照干净全图与 bbox 重新安排恢复区域；需要细看时由模型请求局部原图，每个窗口再判断一次。旧区域被原子替换，不把父组和子任务同时留下。

两边共用冻结的 bbox 几何、1 轮几何 refine、默认 med/med 头尺寸和 2px 线宽。几何原图最小 crop extent 25px、上下文 2 倍、长边 1080；均以实际输入像素坐标输出。两边统一采用“不补画原图遮挡、保留可见片段”的新约定。其他几何代码不变。

完全未变的定位区域共用一次几何结果，用于隔离融合的影响；不是两次独立随机重复实验。新旧区域均通过同一个几何入口，保留逐请求图、prompt、reasoning、回复、critic、坐标变换和 PPT。运行中叠图是已返回部分；完整导出用 LibreOffice 26.8，页尺寸安全缩放，最终以原尺寸 PNG 对照。

默认 thinking，不传 reasoning_effort，输出上限 113000，超时 1200 秒。10 图并行、API 上限 128；融合阶段结束后自动运行几何。无全图重新发现、补漏、范围审核、去重、精细样式，避免混入其他变量。定位不新增 SAM；几何保留原有块状箭头 SAM 辅助/轮廓后备。

当前只测试后置融合，尚未验证全局/局部发现使用新分组提示从头运行的效果。历史 v001/v002 与主线不变。完成数/框数不代表质量。

[实时两栏](reports/10图_实时对照.md) · [15 秒刷新网页](reports/live.html) · [最新进度](progress.json)
''')
    paths=[p for sub in ('code','geometry_code','inputs','constraints_snapshot') for p in (root/sub).rglob('*') if p.is_file()]
    paths += list(root.glob('*.json'))+[p for p in (root/'cases').rglob('before_*.json')]
    write(root/'manifest.json',{'created_at':now(),'files':{str(p.relative_to(root)):sha256(p) for p in paths}})
    report(root)


def process(root,cid,model):
    out=root/'cases'/cid;inp=root/'inputs'/cid
    if (out/'result.json').exists():return read(out/'result.json')
    before=read(out/'before_result.json');rows=before['regions'];frames=read(inp/'frames.json')
    size=frames['global_size'];view=global_frame(frames['native_size']);issues=[]
    write(out/'status.json',{'stage':'model_fusion','state':'running','started_at':now()})
    try:
        answer=model.ask(out/'fusion/request',fusion_prompt(size,rows),
                [('GLOBAL clean: sole output coordinate frame',inp/'global.png'),('BOXES: fallible locators, not strokes',out/'before_boxes.png')],
                lambda a:C.review_valid(a,size,rows,True))
        final=C.apply_review(rows,answer,'U');index={r['id']:r for r in final}
        def detail(i,spec):
            folder=out/'fusion/detail'/f'D{i:03d}';folder.mkdir(parents=True,exist_ok=True)
            crop=crop_from_global(spec['bbox'],view)
            with Image.open(inp/'source.png') as im:crop.image(im.convert('RGB')).save(folder/'local.png')
            write(folder/'frame.json',crop.metadata(frames['native_size']))
            focus=[index[k] for k in spec['refs']]
            prompt=fusion_prompt(size,focus,True,spec['question'])
            prompt['local_region_in_global_px']=visible_global_bounds(crop,view)
            try:
                a=model.ask(folder/'request',prompt,[('GLOBAL clean',inp/'global.png'),('LOCAL clean: visual evidence only',folder/'local.png')],lambda a:C.review_valid(a,size,focus,False))
                return i,focus,C.apply_review(focus,a,f'UD{i}'),None
            except Exception as exc:return i,focus,focus,{'stage':str(folder.relative_to(out)),'error':str(exc)}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(detail,i,spec) for i,spec in enumerate(answer['inspect'],1)]
            for f in as_completed(futures):
                i,focus,replaced,issue=f.result();refs={r['id'] for r in focus}
                final=[r for r in final if r['id'] not in refs]+replaced
                if issue:issues.append(issue)
        # No second global optimization: focused requests edit disjoint retained IDs.
        # Newly ambiguous overlaps remain visible for review, never auto-filtered.
    except Exception as exc:
        issues.append({'stage':'fusion','error':str(exc),'traceback':traceback.format_exc()})
        final=copy.deepcopy(rows)
    C.regions_valid(final,size)
    h,audit=export(cid,final,view,inp/'source.png')
    write(out/'reconstruction_handoff.json',h);write(out/'coordinate_audit.json',audit)
    overlay(inp/'global.png',final,out/'after_boxes.png')
    result={'case_id':cid,'state':'completed_with_issues' if issues else 'completed','before_count':len(rows),
            'region_count':len(final),'regions':final,'issues':issues,'upstream_issues':before.get('issues',[]),
            'pending_count':sum(r['extent']!='complete' for r in final),'geometry_executed':False,'finished_at':now()}
    write(out/'result.json',result);write(out/'status.json',{'stage':'fusion_finished_geometry_queued','state':result['state']})
    print(cid,'fusion',len(rows),'->',len(final),'issues',len(issues),flush=True)
    return result


def run(root):
    if Path(__file__).resolve().parent!=root/'code':
        os.execv(sys.executable,[sys.executable,'-B','-u',str(root/'code/fusion_experiment.py'),'run','--run-dir',str(root)])
    lock=(root/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for p,h in read(root/'manifest.json')['files'].items():
        if sha256(root/p)!=h:raise ValueError('Frozen file changed: '+p)
    write(root/'process.json',{'pid':os.getpid(),'started_at':now()})
    cfg=read(root/'config.json');model=Model(cfg,run_dir=root,system=C.SYSTEM)
    stop=threading.Event()
    def watch():
        while not stop.wait(10):
            try:report(root)
            except Exception:traceback.print_exc()
    watcher=threading.Thread(target=watch,daemon=True);watcher.start()
    try:
        with ThreadPoolExecutor(max_workers=cfg['case_workers']) as pool:
            futures={pool.submit(process,root,cid,model):cid for cid in read(root/'cases.json')}
            for f in as_completed(futures):f.result();report(root)
        write(root/'fusion_finished.json',{'finished_at':now()})
        subprocess.run([sys.executable,'-B','-u',str(root/'code/fusion_geometry.py'),'--run-dir',str(root)],check=True)
        write(root/'finished.json',{'finished_at':now(),'geometry_finished':True})
    finally:stop.set();watcher.join();report(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','report']);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--source',type=Path);p.add_argument('--geometry-code',type=Path);p.add_argument('--geometry-config',type=Path)
    p.add_argument('--constraints',type=Path);p.add_argument('--cases',nargs='+');a=p.parse_args();root=a.run_dir.resolve()
    if a.action=='prepare':prepare(root,a.source.resolve(),a.geometry_code.resolve(),a.geometry_config.resolve(),a.cases,a.constraints.resolve())
    elif a.action=='run':run(root)
    else:report(root)
