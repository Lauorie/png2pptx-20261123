"""Matched A/B/C observation trials; other groups supply pixels, never answers."""
import argparse,shutil,threading,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from PIL import Image
from transport.common import read_json as read,write_json as write,sha256 as sha
from transport.model import Model
import group_recovery,protocol


def main(run,prepared):
    run=Path(run).resolve();prepared=Path(prepared).resolve();cfg=read(run/'config.json');out=run/'ablation'
    model=Model(cfg,run_dir=out,system=protocol.SYSTEM);locks={k:threading.Semaphore(cfg[k+'_workers']) for k in ['cpu','sam']}
    # Scope-only regression list: no answer coordinates, head types or route counts.
    selected=[('001','G011'),('ds004','G012'),('ds012','G004'),('ds007','G016'),('032','G002')]
    tasks=[(v,*item) for v in ['A','B','C'] for item in selected];done=[];errors=[];mutex=threading.Lock()
    def report():
        write(out/'progress.json',dict(total=len(tasks),completed=done,errors=errors))
        lines=['# 跨组证据消融（局部观察组）','',
            'A：本组；B：局部共享；C：局部＋全局共享。目标、原图、SAM、几何/样式代码和模型参数相同。只恢复指定组，不等同于整图已完成。','',
            '| 条件 | case / 组 | 状态 | 对照 |','|---|---|---|---|']
        byid={(x['variant'],x['case'],x['group']):x for x in done}
        for v,cid,gid in tasks:
            record=byid.get((v,cid,gid));rel=f'{v}/cases/{cid}/groups/{gid}'
            lines.append(f'| {v} | {cid}/{gid} | '+(record['state'] if record else '等待／运行中')+f' | [实际输入与组结果]({rel}/README.md) |')
        (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    def one(v,cid,gid):
        source=prepared/'cases'/cid;target=out/v/'cases'/cid/'groups'/gid;target.mkdir(parents=True,exist_ok=True)
        shutil.copytree(source/'groups'/gid/'input',target/'input',dirs_exist_ok=True)
        meta=read(source/'groups'/gid/'prepared_observation.json')
        plan=next(x for x in read(source/'observation_plan.json')['groups'] if x['id']==gid)
        local_cfg={**cfg,'shared_mask_context':v!='A','shared_context_global':v=='C'}
        pool=read(source/'shared_observations.json')['masks'] if v!='A' else None
        clean=Image.open(source/'clean.png').convert('RGB')
        result=group_recovery.group_task(clean,[dict(id=x) for x in plan['sources']],[],{},gid,target,local_cfg,model,locks,prepared=meta,evidence_pool=pool)
        # prepare() restores the same M/D layer in all variants; all source pixels
        # and numeric proposals are still identical before the contextual overlay.
        result=dict(variant=v,case=cid,group=gid,state=result['state'],rows=len(result['rows']),
            followups=result.get('followups',[]),issues=result.get('issues',[]),
            clean_sha256=sha(target/'input/clean.png'),proposals_sha256=sha(target/'proposals.json'),
            geometry_calls=result.get('geometry_calls',0),head_calls=result.get('head_calls',0))
        write(target/'trial.json',result);return result
    report()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures={executor.submit(one,*t):t for t in tasks}
        for f in as_completed(futures):
            try:result=f.result();done.append(result);print(result['variant'],result['case'],result['state'],flush=True)
            except Exception as exc:errors.append(dict(task=futures[f],error=str(exc),traceback=traceback.format_exc()))
            report()
    write(out/'finished.json',dict(total=len(tasks),completed=done,errors=errors,full_case_evaluation=False))
    if errors:raise RuntimeError(f'{len(errors)} ablation tasks failed; see ablation/finished.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True)
    a=p.parse_args();main(a.run,a.prepared)
