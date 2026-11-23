"""Exercise selected real groups through the SAME frozen mainline entry."""
import argparse,json,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import pipeline,evidence as ev,mask_grouping,protocol
from transport.model import Model
from transport.common import read_json as read,write_json as write


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--targets',nargs='+',required=True);a=p.parse_args();a.run.mkdir(parents=True,exist_ok=True)
    cfg=read(a.config);model=Model(cfg,run_dir=a.run,system=protocol.SYSTEM)
    locks={k:threading.Semaphore(cfg[k+'_workers']) for k in ['cpu','sam']}
    cache={};jobs=[];results={};failures=[]
    for pair in a.targets:
        cid,gid=pair.split(':');case=a.run/'cases'/cid
        if cid not in cache:
            clean,hints,_=ev.load_case(cid,cfg,case);groups,_=mask_grouping.group(hints,clean.size,cfg['observation_grouping'])
            group_of={r['id']:f'G{i:03d}' for i,g in enumerate(groups,1) for r in g}
            cache[cid]=(clean,hints,groups,group_of)
        clean,hints,groups,group_of=cache[cid]
        jobs.append((cid,gid,clean,groups[int(gid[1:])-1],hints,group_of,case))
    write(a.run/'smoke_manifest.json',dict(targets=a.targets,config=str(a.config.resolve()),code=str(Path(__file__).resolve().parent),same_group_entry_as_full_run=True))
    with ThreadPoolExecutor(max_workers=8) as pool:
        fs={pool.submit(pipeline.group_task,clean,g,hints,mapping,gid,case/'groups'/gid,cfg,model,locks):(cid,gid)
            for cid,gid,clean,g,hints,mapping,case in jobs}
        for f in as_completed(fs):
            cid,gid=fs[f]
            try:r=f.result();results.setdefault(cid,[]).append(r);print(cid,gid,r['state'],flush=True)
            except Exception as exc:failures.append(dict(case=cid,group=gid,error=str(exc)));print(cid,gid,'FAILED',str(exc),flush=True)
            write(a.run/'progress.json',dict(ended=sum(map(len,results.values()))+len(failures),total=len(jobs),failures=failures))
    lines=['# 同一主管线入口：针对性验证','','| case | 原图 | 本次所选组的实例 | 叠图 |','|---|---|---|---|']
    for cid,rs in sorted(results.items()):
        case=a.run/'cases'/cid;pipeline.publish(cache[cid][0],sorted(rs,key=lambda r:r['group']),case)
        rel='cases/'+cid;lines.append(f'| {cid} | ![]({rel}/clean.png) | ![]({rel}/result/instances.png) | ![]({rel}/result/overlap.png) |')
    (a.run/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    write(a.run/'finished.json',dict(groups=len(jobs),completed=sum(map(len,results.values())),failures=failures))


if __name__=='__main__':main()
