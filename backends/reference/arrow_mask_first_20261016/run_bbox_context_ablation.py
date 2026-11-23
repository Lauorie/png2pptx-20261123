"""Matched locator, handoff and optional-mask experiments on saved observations."""
import argparse
import copy
import fcntl
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent/'lib'))
from PIL import Image
import evidence as ev
import geometry_routing
import group_bbox_identity
import group_recovery
import protocol
import semantic_geometry
import style_contract
from lib import viz
from run_group_bbox_scope_trial import prepare
from run_region_ablation import usage
from transport.common import read_json as read,write_json as write,sha256 as sha
from transport.model import Model


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    run=args.run.resolve();lock=(run/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=read(run/'code_manifest.json')
    if any(sha(run/'code'/n)!=h for n,h in manifest['files'].items()):raise ValueError('Frozen code changed')
    for n in ('config.json','selection.json'):
        if sha(run/n)!=manifest[n+'_sha256']:raise ValueError('Frozen selection/config changed')
    config=read(run/'config.json');specs=read(run/'selection.json')
    model=Model(config,run_dir=run,system=protocol.SYSTEM)
    locks={k:threading.Semaphore(config[k+'_workers']) for k in ('cpu','sam')}
    for spec in specs:
        folder=run/'cases'/spec['name']
        if not (folder/'input_audit.json').exists():prepare({**spec,'mode':'group'},folder)
        audit=read(folder/'input_audit.json')
        if audit['selection']!={**spec,'mode':'group'} or any(sha(n)!=h for n,h in audit['source_hashes'].items()):
            raise ValueError('Prepared inputs changed')
        cfg={**config,**spec.get('overrides',{})};write(folder/'effective_config.json',cfg)
        crop=Image.open(folder/'input/clean.png').convert('RGB')
        context_size=Image.open(folder/'input/context.png').size
        q,images=group_bbox_identity.build(crop,read(folder/'group.json'),
            read(folder/'prepared_observation.json')['view'],context_size,folder,cfg)
        write(folder/'prepared_identity_prompt.json',q)
        write(folder/'input/images.json',[dict(label=l,path=str(p.resolve()),sha256=sha(p)) for l,p in images])
    states={};mutex=threading.RLock();started=time.monotonic()
    def report(name=None,value=None):
        with mutex:
            if name:states[name]=value
            ended=sum((run/'cases'/s['name']/'finished.json').exists() for s in specs)
            write(run/'progress.json',dict(total=len(specs),completed=ended,cases=states,updated=time.time(),seconds=time.monotonic()-started))
            lines=['# Bbox 显示与几何证据小实验','',f'已结束 {ended}/{len(specs)}。',
                   '', '本轮只评估身份与几何；颜色、虚线和头型尚未进入最终样式恢复。实例预览的颜色用于区分目标。',
                   '', '| 对照 | 原图 | 实际定位输入 | 几何结果 |', '|---|---|---|---|']
            for s in specs:
                root='cases/'+s['name'];folder=run/root
                image=f'![]({root}/result/instances.png)' if (folder/'result/instances.png').exists() else '身份对照／等待几何'
                status=states.get(s['name'],{}).get('state','prepared')
                lines.append(f"| {s['name']}<br>{s['purpose']}<br>{status} | ![]({root}/input/clean.png) | ![]({root}/input/target_boxes.png) | {image} |")
            lines+=['','| 对照 | 原结果 | 本轮叠图 | 记录 |','|---|---|---|---|']
            for s in specs:
                root='cases/'+s['name'];folder=run/root
                old=f'![]({root}/before/instances.png)' if (folder/'before/instances.png').exists() else '无有效原结果'
                new=f'![]({root}/result/overlap.png)' if (folder/'result/overlap.png').exists() else '等待'
                lines.append(f"| {s['name']} | {old} | {new} | [身份输出]({root}/identity_decision.json) · [状态]({root}/finished.json) |")
            lines+=['','[实验设计](实验设计.md) · [实时状态](progress.json) · [选择与变量](selection.json)',
                    '', 'Baseline 来源及所有实际请求均保留。单次成功不代表稳定性；执行状态不代表视觉准确。']
            tmp=run/'结果对照.md.tmp';tmp.write_text('\n'.join(lines)+'\n');tmp.replace(run/'结果对照.md')
    report()
    if args.prepare_only:return
    write(run/'process.json',dict(pid=os.getpid(),started=time.time()))
    def work(spec):
        folder=run/'cases'/spec['name'];begin=time.monotonic()
        if (folder/'finished.json').exists():
            saved=read(folder/'finished.json');report(spec['name'],saved);return saved
        cfg={**config,**spec.get('overrides',{})};state=dict(state='running',mode=spec['mode']);report(spec['name'],state)
        try:
            group=read(folder/'group.json');meta=read(folder/'prepared_observation.json')
            crop=Image.open(folder/'input/clean.png').convert('RGB')
            case=Path(spec['source_case'])
            if spec['mode']=='identity':
                q=read(folder/'prepared_identity_prompt.json');images=[(r['label'],Path(r['path'])) for r in read(folder/'input/images.json')]
                ids=[g['id'] for g in group] if cfg.get('identity_neutral_handoff') else None
                answer=model.ask(folder/'identity',q,images,lambda a:group_bbox_identity.validate(a,crop.size,q['context_size'],ids))
                write(folder/'identity_decision.json',answer);state.update(targets=len(answer['targets']))
            elif spec['mode']=='geometry':
                saved=Path(spec['source_decision']);answer=read(saved)
                answer['routes']=[r for r in answer['routes'] if r['id'] in spec['route_ids']]
                if len(answer['routes'])!=len(spec['route_ids']):raise ValueError('Unknown fixed target')
                write(folder/'fixed_decision.json',answer)
                index={c['id']:c for c in meta['candidates']}
                candidates=[dict(index[m],mask=ev.mask(index[m]['path'])) for m in meta['selected_ids']]
                with locks['cpu']:proposals,_=semantic_geometry.prepare(crop,group,candidates,folder)
                rows,supports,issues,calls=geometry_routing.recover(answer,proposals,candidates,spec['group'],crop,folder,model,locks,cfg)
                with locks['cpu']:
                    for row in rows:row['style']=style_contract.provisional(crop,supports[row['id']],row)
                    write(folder/'rows_local.json',rows);viz.render(folder/'input/clean.png',rows,folder/'result')
                state.update(paths=len(rows),issues=issues,geometry_calls=calls)
            elif spec['mode']=='group':
                clean=Image.open(case/'clean.png').convert('RGB');hints=read(case/'global_hints.json')
                plan=read(case/'observation_plan.json')['groups'];group_of={s:p['id'] for p in plan for s in p['sources']}
                result=group_recovery.group_task(clean,group,hints,group_of,spec['group'],folder,cfg,model,locks,prepared=meta)
                state.update(paths=len(result['rows']),issues=result['issues'],targets=len(result['identities']))
            else:raise ValueError('Unknown replay mode')
            state['state']='completed_with_issues' if state.get('issues') else 'completed'
        except Exception as exc:
            state.update(state='failed',error=str(exc),traceback=traceback.format_exc())
        state.update(seconds=time.monotonic()-begin,api=usage(folder),style_fitted=False,accuracy_measured=False)
        write(folder/'finished.json',state);report(spec['name'],state)
        print(spec['name'],state['state'],flush=True);return state
    with ThreadPoolExecutor(max_workers=min(8,config['task_workers'])) as pool:
        jobs={pool.submit(work,s):s for s in specs}
        for future in as_completed(jobs):future.result()
    write(run/'finished.json',dict(cases=states,seconds=time.monotonic()-started))


if __name__=='__main__':main()
