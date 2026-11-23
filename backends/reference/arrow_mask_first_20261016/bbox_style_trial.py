"""Matched identity-input trials plus style replay on frozen physical geometry.

No SAM calls; no repair/coverage calls. A/B/C share the exact old prompt except
for the declared bbox and shaft-sample additions. Full-case reruns are separate.
"""
import argparse,copy,json,os,shutil,sys,threading,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'lib'))
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from PIL import Image,ImageDraw
import semantic_geometry,target_locator,shaft_sampling,style_windows,visual
from lib import viz,fit,style
from transport.common import read_json as read,write_json as write,sha256 as sha
from transport.model import Model


def main(run,baseline):
    run=Path(run).resolve();baseline=Path(baseline).resolve();cfg=read(run/'config.json')
    cases=[('001','G020'),('ds027','G009'),('558','G003'),('ds007','G003'),('ds024','G011'),('ds017','G001')]
    tasks=[(cid,gid,v) for cid,gid in cases for v in 'ABC'];results={};started=time.time()
    model=Model(cfg,run_dir=run,system=read(baseline/'cases/001/groups/G020/identity/identity.json')['system'])
    write(run/'process.json',dict(pid=os.getpid(),started=started));write(run/'scope.json',dict(groups=cases,variants=list('ABC'),full_cases=False))
    write(run/'config_effective.json',dict(model=cfg['model'],max_tokens=cfg['max_tokens'],thinking=cfg['thinking'],
        api_concurrency=cfg['api_concurrency'],case_count=len(cases),fresh_sam_calls=0,coverage_calls=0))

    def report():
        records=list(results.values());write(run/'progress.json',dict(total=len(tasks),finished=len(records),records=records,seconds=time.time()-started))
        lines=['# bbox 定位与箭身取样：局部对照','',
            f'结束 {len(records)}/{len(tasks)} 个身份请求任务。不是整图重跑；复用同一 crop、SAM 和模型参数。',
            '', 'A：原请求重新调用；B：用全局候选 bbox 替换全局 mask 提示图；C：B 加可选箭身取样。三组都保留局部 mask 图和干净原图。',
            '', '样式图固定旧几何、头型和类别，只应用 C 的取样建议；可单独检验样式，避免混入新身份/坐标变化。无法唯一对应的路线不套用取样。',
            '', '| case / 组 | 原图 | 旧样式 | C：固定几何重测样式 | 请求对照 |', '|---|---|---|---|---|']
        for cid,gid in cases:
            rel=Path('cases')/cid/gid;old=baseline/'cases'/cid/'groups'/gid
            src=os.path.relpath(old/'input/clean.png',run);before=os.path.relpath(old/'result/styled.png',run)
            after=rel/'C/fixed_style/styled.png'
            refs=[]
            for v in 'ABC':
                key=(cid,gid,v);state=results.get(key,{}).get('state','运行/等待')
                refs.append(f'[{v}: {state}]({rel}/{v}/README.md)')
            lines.append(f'| {cid}/{gid} | ![]({src}) | ![]({before}) | '+(f'![]({after})' if (run/after).exists() else '')+' | '+' · '.join(refs)+' |')
        lines+=['','[所有状态和用量](progress.json) · [代码与对照说明](CHANGES.md)','']
        (run/'RESULTS.md').write_text('\n'.join(lines))

    def one(cid,gid,variant):
        old=baseline/'cases'/cid/'groups'/gid;out=run/'cases'/cid/gid/variant;out.mkdir(parents=True,exist_ok=True)
        shutil.copytree(old/'input',out/'input',dirs_exist_ok=True)
        req=read(old/'identity/identity.json');prompt=copy.deepcopy(req['prompt'])
        old_images=read(old/'identity/call_000/request_metadata.json')['images']
        images=[(x['label'],out/'input'/Path(x['path']).name) for x in old_images]
        assert all(sha(path)==item['sha256'] for (_,path),item in zip(images,old_images))
        hints={x['id']:x for x in read(baseline/'cases'/cid/'global_hints.json')}
        group=[hints[k] for k in prompt['sources']];view=read(out/'input/transform.json')
        clean=Image.open(out/'input/clean.png').convert('RGB')
        if variant in 'BC':images=target_locator.apply(prompt,images,group,view,clean,out)
        if variant=='C':shaft_sampling.add_prompt(prompt)
        write(out/'prompt.json',prompt);write(out/'input/images.json',[dict(label=k,path=str(p),sha256=sha(p)) for k,p in images])
        write(out/'condition.json',dict(variant=variant,source=str(old),changed_images=0 if variant=='A' else 1,
            model_selects_sample=variant=='C',clean_unchanged=sha(out/'input/clean.png')==sha(old/'input/clean.png')))
        meta=read(old/'prepared_observation.json');index={c['id']:c for c in meta['candidates']}
        candidates=[dict(index[k],mask=np.array(Image.open(index[k]['path']).convert('L'))>0) for k in meta['selected_ids']]
        proposals=read(old/'proposals.json');mids=[c['id'] for c in candidates]
        write(out/'state.json',dict(state='requesting'))
        text=['# '+cid+'/'+gid+' — '+variant,'','当前条件：'+{'A':'原请求重放','B':'全局候选 bbox 定位','C':'bbox + GLM 可选箭身取样'}[variant],
              '', '[完整 prompt](prompt.json) · [实际调用](identity/prompt.json)', '', '| 输入 | 图片 |','|---|---|']
        text += [f'| {label} | ![](input/{path.name}) |' for label,path in images]
        (out/'README.md').write_text('\n'.join(text)+'\n')
        answer=model.ask(out/'identity',prompt,images,
            lambda a:semantic_geometry.validate(a,clean.size,prompt['sources'],proposals,mids,candidates,prompt['context_size']))
        write(out/'decision.json',answer)
        result=dict(case=cid,group=gid,variant=variant,state='completed',routes=len(answer['routes']),
            excluded=answer['excluded'],uncertain=answer['uncertain'],followups=answer.get('followups',[]),
            route_decisions=[{k:r.get(k) for k in ['id','sources','kind','geometry','masks','shaft_sample','description']} for r in answer['routes']])
        if variant=='C':
            rows=read(old/'rows_local.json') if (old/'rows_local.json').exists() else []
            checks=[];preview=clean.copy();draw=ImageDraw.Draw(preview)
            for row in rows:
                candidates_for_row=[r for r in answer['routes'] if set(r['sources'])==set(row['globals'])]
                baseline_matches=[r for r in rows if set(r['globals'])==set(row['globals'])]
                check=dict(id=row['id'],old_width_local=row.get('style',{}).get('width_px'),applied=False)
                if len(candidates_for_row)==1 and len(baseline_matches)==1 and 'shaft_sample' in candidates_for_row[0]:
                    sample=candidates_for_row[0]['shaft_sample'];row['shaft_sample']=sample
                    if sample!='auto':
                        row['style']=style_windows.recover(np.array(clean),None,row);check['applied']=True
                        if isinstance(sample,list):
                            draw.line(list(map(tuple,sample)),fill='#008080',width=2);visual.label(draw,sample[0],row['id'].split(':')[-1],'#008080',preview)
                    check.update(shaft_sample=sample,width_local=row['style'].get('width_px'),pending=style.pending(row))
                else:check['reason']='No unique identity correspondence; old style retained'
                checks.append(check)
            write(out/'fixed_rows_local.json',rows);write(out/'style_checks.json',checks)
            preview.save(out/'input/shaft_samples.png')
            if rows:viz.render(out/'input/clean.png',rows,out/'fixed_style')
            scale=float(np.sqrt(abs(np.linalg.det(np.asarray(view['view_to_global'])[:2,:2]))))
            for check in checks:
                for key in ['old_width_local','width_local']:
                    if check.get(key) is not None:check[key.replace('local','global')]=check[key]*scale
            result['style_checks']=checks;write(out/'style_checks.json',checks)
            text+=['','## 固定旧几何和头型，仅重测样式','','| 取样位置 | 样式 | 叠图 |','|---|---|---|',
                   '| ![](input/shaft_samples.png) | ![](fixed_style/styled.png) | ![](fixed_style/overlap.png) |',
                   '', '[测量对照](style_checks.json)']
        usage=[]
        for p in sorted((out/'identity').glob('call_*/status.json')):
            s=read(p);usage.append(dict(seconds=s.get('seconds'),usage=s.get('usage'),state=s.get('state')))
        result['calls']=usage
        text+=['','[模型判断](decision.json) · [任务记录](trial.json)']
        (out/'README.md').write_text('\n'.join(text)+'\n');write(out/'trial.json',result);write(out/'state.json',dict(state='completed'))
        return result

    report()
    with ThreadPoolExecutor(max_workers=min(24,cfg['api_concurrency'])) as pool:
        futures={pool.submit(one,*task):task for task in tasks}
        for future in as_completed(futures):
            key=futures[future]
            try:record=future.result()
            except Exception as exc:
                record=dict(case=key[0],group=key[1],variant=key[2],state='failed',error=str(exc),traceback=traceback.format_exc())
                write(run/'cases'/key[0]/key[1]/key[2]/'error.json',record)
            results[key]=record;report();print(*key,record['state'],flush=True)
    write(run/'finished.json',dict(total=len(tasks),completed=sum(r['state']=='completed' for r in results.values()),records=list(results.values()),seconds=time.time()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--baseline',required=True)
    args=p.parse_args();main(args.run,args.baseline)
