"""Fresh single-review requests on four saved cases, with a 2x2 factor design."""
import argparse
import fcntl
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import critic_context_ablation as experiment
import geometry_critic
from transport.common import read_json as read, write_json as write, sha256
from transport.model import Model


def usage(folder):
    result = dict(attempts=0, reported_attempts=0, input=0, output=0, reasoning=0, total=0, request_seconds=0., active=0)
    for p in folder.glob('request/**/call_*/status.json'):
        try:
            s=read(p)
        except (ValueError,OSError):
            continue
        result['attempts']+=1
        result['active']+=int(s.get('state')=='requesting')
        result['request_seconds']+=s.get('seconds',0)
        u=s.get('usage') or {}
        if u:
            result['reported_attempts']+=1
            for k, field in [('input','prompt_tokens'),('output','completion_tokens'),('total','total_tokens')]:
                result[k]+=u.get(field,0)
            result['reasoning']+=(u.get('completion_tokens_details') or {}).get('reasoning_tokens',0)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    run=parser.parse_args().run.resolve()
    lock=(run/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=read(run/'code_manifest.json')
    for relative,digest in manifest['files'].items():
        if sha256(run/'code'/relative)!=digest:raise ValueError('Changed frozen code: '+relative)
    for relative,digest in manifest['inputs'].items():
        if sha256(run/relative)!=digest:raise ValueError('Changed input: '+relative)
    for name in ('config.json','selection.json'):
        if sha256(run/name)!=manifest[name+'_sha256']:raise ValueError('Changed '+name)
    config=read(run/'config.json');selection=read(run/'selection.json')
    model=Model(config,run_dir=run,system=selection['system'])
    started=time.time();write(run/'process.json',dict(pid=os.getpid(),started=started))
    variants=selection.get('variants',list(experiment.VARIANTS))
    jobs=[(case,variant) for case in selection['cases'] for variant in variants]
    states={case['id']+'/'+v:dict(state='pending') for case,v in jobs}
    mutex=threading.RLock()

    def report():
        with mutex:
            records=[]
            for case,v in jobs:
                key=case['id']+'/'+v;folder=run/'variants'/v/'cases'/case['id']
                records.append(dict(case=case['id'],variant=v,**states[key],usage=usage(folder)))
            write(run/'progress.json',dict(total=len(jobs),completed=sum(r['state'] in ('finished','failed') for r in records),
                  jobs=records,seconds=time.time()-started,updated=time.time()))
            lines=['# 几何 critic：上下文 × 审核范围消融','',
                '四图同一份已保存点几何。A 当前局部；B 加整图；C 局部且关注结构；D 整图且关注结构。每格一次新审核，113k 上限保持。模型接受不等于视觉正确。','',
                '| 图 | 原图 | 输入几何 | '+' | '.join(variants)+' |',
                '|'+ '|'.join(['---']*(3+len(variants)))+'|']
            for case in selection['cases']:
                cid=case['id'];cells=[]
                for v in variants:
                    folder=run/'variants'/v/'cases'/cid;s=states[cid+'/'+v];u=usage(folder)
                    label=s.get('decision',s['state'])+' / '+f'{u["total"]:,} token / {u["request_seconds"]:.1f}s'
                    picture=folder/'after/geometry.png'
                    body=label+('<br>![]('+str(picture.relative_to(run))+')' if picture.exists() else '')
                    request=folder/'request/prompt.json'
                    if request.exists():body+=' <br>[请求]('+str(request.relative_to(run))+')'
                    cells.append(body)
                lines.append('| '+cid+' | ![](inputs/'+cid+'/clean.png) | ![](inputs/'+cid+'/geometry.png) | '+' | '.join(cells)+' |')
            lines+=['','[实验设计](实验说明.md) · [逐请求统计](progress.json) · [坐标与输入核验](preflight.json)','',
                '| 图 | 整图上下文 |','|---|---|']
            for case in selection['cases']:
                lines.append('| '+case['id']+' | ![](inputs/'+case['id']+'/context.png) |')
            tmp=run/'RESULTS.md.tmp';tmp.write_text('\n'.join(lines)+'\n');tmp.replace(run/'RESULTS.md')

    def work(case,variant):
        cid=case['id'];key=cid+'/'+variant;folder=run/'variants'/variant/'cases'/cid;begin=time.time()
        with mutex:states[key]=dict(state='running');report()
        try:
            inp=run/'inputs'/cid
            prompt,images=experiment.build(read(inp/'original_prompt.json'),inp,variant,read(inp/'context.json'))
            write(folder/'input_manifest.json',dict(images=[dict(label=l,path=str(p),sha256=sha256(p)) for l,p in images],
                 prompt=prompt,original_source=case['source']))
            answer=model.ask(folder/'request',prompt,images,lambda a:geometry_critic.validate(a,prompt['size']))
            write(folder/'answer.json',answer)
            delta=experiment.render_answer(inp,answer,folder)
            write(folder/'geometry_change.json',delta)
            result=dict(state='finished',decision=answer['decision'],reason=answer.get('reason'),geometry_change=delta,seconds=time.time()-begin)
        except Exception as exc:
            result=dict(state='failed',error=str(exc),traceback=traceback.format_exc(),seconds=time.time()-begin)
        write(folder/'completed.json',result)
        with mutex:states[key]=result;report()
        print(cid,variant,result['state'],result.get('decision'),flush=True)
        return result

    stop=threading.Event();report()
    def monitor():
        while not stop.wait(15):report()
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    # Four image cases; variants within each case execute independently.
    def work_case(case):
        with ThreadPoolExecutor(max_workers=4) as pool:
            return [f.result() for f in as_completed([pool.submit(work,case,v) for v in variants])]
    with ThreadPoolExecutor(max_workers=min(4,config['case_workers'])) as pool:
        for f in as_completed([pool.submit(work_case,case) for case in selection['cases']]):f.result()
    stop.set();thread.join(timeout=1);report()
    write(run/'finished.json',dict(state='finished',seconds=time.time()-started,jobs=states,accuracy_measured=False))


if __name__=='__main__':main()
