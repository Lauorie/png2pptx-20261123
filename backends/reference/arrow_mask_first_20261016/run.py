#!/usr/bin/env python3
"""Frozen mask-first experiment; scope selection does not change algorithms."""
import argparse,fcntl,json,os,shutil,sys,threading,time,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from transport.common import read_json as read,write_json as write,sha256 as sha
from transport.model import Model
import pipeline,protocol


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--cases',nargs='+')
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'));p.add_argument('--frozen',action='store_true');p.add_argument('--prepare-only',action='store_true');p.add_argument('--freeze-only',action='store_true')
    a=p.parse_args();run=a.run.resolve();run.mkdir(parents=True,exist_ok=True);cfg=read(a.config);scope=dict(cases=a.cases)
    if not a.frozen:
        root=Path(__file__).resolve().parent
        if not (run/'code_manifest.json').exists():
            files={}
            for src in list(root.glob('*.py'))+list((root/'lib').glob('*.py'))+list((root/'transport').glob('*.py'))+list((root/'shape_core').rglob('*.py')):
                rel=src.relative_to(root);dest=run/'code'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);files[str(rel)]=sha(dest)
            if (root/'shape_core/SOURCE.json').exists():
                shutil.copy2(root/'shape_core/SOURCE.json',run/'code/shape_core/SOURCE.json')
                files['shape_core/SOURCE.json']=sha(run/'code/shape_core/SOURCE.json')
            shutil.copy2(root/'BASELINE.json',run/'BASELINE.json');write(run/'config.json',cfg);write(run/'scope.json',scope)
            write(run/'code_manifest.json',dict(files=files,config_sha256=sha(run/'config.json')))
        if read(run/'config.json')!=cfg or read(run/'scope.json')!=scope:raise ValueError('Frozen run differs; use a new run directory')
        frozen=read(run/'code_manifest.json')
        if any(sha(run/'code'/name)!=h for name,h in frozen['files'].items()):raise ValueError('Frozen code changed')
        if a.freeze_only:return
        command=[sys.executable,'-u',str(run/'code/run.py'),'--run',str(run),'--config',str(run/'config.json'),'--frozen']
        if a.cases:command+=['--cases',*a.cases]
        if a.prepare_only:command+=['--prepare-only']
        os.execv(sys.executable,command)
    lock=(run/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=Path(cfg['global_predictions_run']);manifest=read(source/'manifest.json');all_ids=[r['id'] for r in manifest['images']]
    if not all_ids or len(all_ids)!=len(set(all_ids)):raise ValueError('Input manifest must contain nonempty, unique case IDs')
    ids=a.cases or all_ids
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate selected case IDs')
    if not set(ids)<=set(all_ids):raise ValueError('Unknown case IDs')
    write(run/'manifest.json',dict(ids=ids,source=str(source),source_sha256=sha(source/'manifest.json'),same_algorithm_all_cases=True))
    write(run/'process.json',dict(pid=os.getpid(),started=time.time(),prepare_only=a.prepare_only))
    model=Model(cfg,run_dir=run,reuse_from=cfg.get('model_cache_from') or run,system=protocol.SYSTEM);locks={k:threading.Semaphore(cfg[k+'_workers']) for k in ['cpu','sam']}
    progress={};done={};errors={};mutex=threading.RLock();started=time.monotonic()
    def report(cid=None,value=None):
        with mutex:
            if cid:progress[cid]=value
            write(run/'progress.json',dict(total=len(ids),completed=len(done),cases=progress,errors=errors,seconds=time.monotonic()-started))
            continuation=bool(cfg.get('coverage_resume_from'))
            lines=['# 主线双图补漏 → 局部重建 → 最终结果' if continuation else '# 身份路由 → mask / GLM 分段几何 → 样式','',
                f'已结束 {len(done)}/{len(ids)}。普通目标保留 mask 拟合；复杂目标单独拟合线/曲线分段，共享网络联合处理。图并发 {cfg["case_workers"]}，请求并发上限 {cfg["api_concurrency"]}。','',
                ('从归档的 continuation 快照继续，使用主管线 coverage_stage.run；本轮未重跑身份发现。运行中不把旧快照显示成最终图。' if continuation else '每个 case 使用相同主线入口。'),'',
                '| case | 状态 | 原图 | '+('补漏前' if continuation else '实例')+' | '+('最终 PPT 渲染' if continuation else '样式')+' | 叠图 |','|---|---|---|---|---|---|']
            for case in ids:
                folder=run/'cases'/case;rel=f'cases/{case}';entry=f'[{case}]({rel}/RESULTS.md)' if (folder/'RESULTS.md').exists() else case
                names=['clean.png','before/result/styled.png','native/styled.png','result/overlap.png'] if continuation else ['clean.png','result/instances.png','result/styled.png','result/overlap.png']
                if not continuation and (folder/'native/styled.png').exists():names[2]='native/styled.png'
                images=[f'![]({rel}/{name})' if (folder/name).exists() and (not continuation or i<2 or (folder/'completed.json').exists()) else '等待' for i,name in enumerate(names)]
                lines.append('| '+' | '.join([entry,progress.get(case,{}).get('stage','等待'),*images])+' |')
            lines+=['','每组保留 crop、SAM mask、身份决定、routing.json 和 geometry/ 下的实际几何请求；程序完成不等于视觉正确。','']
            temp=run/'RESULTS.md.tmp';temp.write_text('\n'.join(lines));temp.replace(run/'RESULTS.md')
    report()
    with ThreadPoolExecutor(max_workers=cfg['case_workers']) as pool:
        jobs={pool.submit(pipeline.run_case,cid,cfg,run/'cases'/cid,model,locks,report,a.prepare_only):cid for cid in ids}
        for f in as_completed(jobs):
            cid=jobs[f]
            try:done[cid]=f.result();print(cid,done[cid]['state'],flush=True)
            except Exception as exc:
                errors[cid]=str(exc);progress[cid]=dict(stage='failed',error=str(exc));write(run/'cases'/cid/'error.json',dict(error=str(exc),traceback=traceback.format_exc()))
                print(cid,'FAILED',str(exc),flush=True)
            report()
    write(run/('prepared.json' if a.prepare_only else 'finished.json'),dict(total=len(ids),completed=len(done),cases=done,errors=errors,seconds=time.monotonic()-started))


if __name__=='__main__':main()
