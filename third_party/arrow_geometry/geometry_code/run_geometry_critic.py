"""Replay saved local geometry through the same mainline geometry-review hook."""
import argparse
import fcntl
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent/'lib')]
from PIL import Image
import geometry_critic
import protocol
from transport.model import Model
from transport.common import read_json as read, write_json as write, sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    run=parser.parse_args().run.resolve()
    lock=(run/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    frozen=read(run/'code_manifest.json')
    if any(sha256(run/'code'/p)!=h for p,h in frozen['files'].items()):raise ValueError('Changed frozen code')
    for name in ('config.json','selection.json'):
        if sha256(run/name)!=frozen[name+'_sha256']:raise ValueError('Changed frozen '+name)
    config=read(run/'config.json');jobs=read(run/'selection.json')['jobs']
    model=Model(config,run_dir=run,system=protocol.SYSTEM)
    states={j['case']:dict(stage='prepared') for j in jobs};mutex=threading.RLock()
    cpu=threading.Semaphore(config.get('cpu_workers',4));start=time.monotonic()
    write(run/'process.json',dict(pid=os.getpid(),started=time.time()))

    def report():
        with mutex:
            write(run/'progress.json',dict(total=len(jobs),completed=sum(s['stage'] in ('finished','failed') for s in states.values()),
                  cases=states,seconds=time.monotonic()-start,updated=time.time()))
            lines=['# 几何 critic/refine：四图已保存点几何回放','',
                   '只判断几何；两张输入为局部 CLEAN 与点线叠图。最多两次修订请求；accept 提前结束，第二次仍修订则标为预算用尽、未复核。mask 分支本轮关闭。',
                   '', '| 图 / 目标 | 状态 | 原图 | 修订前 | 修订后 |', '|---|---|---|---|---|---|']
            for job in jobs:
                cid=job['case'];folder=run/job['folder'];state=states[cid]
                target_folders=list((folder/'geometry_critic').glob('*/target.json'))
                where=target_folders[0].parent if target_folders else folder/'preview'
                images=[]
                for name in ('before/clean.png','before/geometry.png','after/geometry.png'):
                    p=where/name;images.append('![]('+str(p.relative_to(run))+')' if p.exists() else '等待')
                status=state['stage']
                if state.get('targets'):status+=' / '+','.join(t['state'] for t in state['targets'])
                lines.append('| '+cid+' / '+job['owner']+' | '+status+' | '+' | '.join(images)+' |')
            lines+=['','[实验说明](实验说明.md) · [配置](config.json) · [实时状态](progress.json)',
                    '', '状态、模型 accept 和视觉正确分别判断；结果中不使用之前错误的虚线样式。']
            tmp=run/'RESULTS.md.tmp';tmp.write_text('\n'.join(lines)+'\n');tmp.replace(run/'RESULTS.md')

    def work(job):
        folder=run/job['folder'];begin=time.monotonic()
        with mutex:states[job['case']]=dict(stage='geometry_critic');report()
        try:
            seed=read(folder/'seed_rows.json');clean=Image.open(folder/'input/clean.png').convert('RGB')
            source_view_path=folder/'input/source_view.json'
            source_view=read(source_view_path) if source_view_path.exists() else seed[0].get('geometry_frame')
            rows,support,issues,calls=geometry_critic.recover(clean,seed,{r['id']:None for r in seed},
                 folder/'geometry_critic',model,config,cpu,source_view=source_view)
            write(folder/'rows_local.json',rows)
            target_states=[read(p) for p in (folder/'geometry_critic').glob('*/state.json')]
            state=dict(stage='finished',targets=target_states,calls=calls,issues=issues,
                 paths_before=len(seed),paths_after=len(rows),seconds=time.monotonic()-begin)
        except Exception as exc:state=dict(stage='failed',error=str(exc),seconds=time.monotonic()-begin)
        write(folder/'completed.json',state)
        with mutex:states[job['case']]=state;report()
        return state

    report()
    stop=threading.Event()
    def monitor():
        while not stop.wait(10):report()
    monitor_thread=threading.Thread(target=monitor,daemon=True);monitor_thread.start()
    with ThreadPoolExecutor(max_workers=min(4,config.get('case_workers',4))) as pool:
        for future in as_completed([pool.submit(work,j) for j in jobs]):
            value=future.result();print(value['stage'],value.get('paths_before'),value.get('paths_after'),flush=True)
    stop.set();monitor_thread.join(timeout=1);report()
    write(run/'finished.json',dict(state='finished',cases=states,seconds=time.monotonic()-start,accuracy_measured=False))


if __name__=='__main__':main()
