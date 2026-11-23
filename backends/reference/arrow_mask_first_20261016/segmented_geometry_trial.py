#!/usr/bin/env python3
"""Frozen local GLM geometry trial: fixed identities/styles, free line/curve parts."""
import argparse,copy,hashlib,json,math,os,shutil,sys,threading,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed


def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(p)


from segmented_geometry import validate,geometry,prompt,draw_points


def prepare(source,baseline_run,case,gid,dest):
    import numpy as np
    from PIL import Image,ImageDraw
    import visual,viz
    src=source/'cases'/case/'groups'/gid;rows=read(src/'rows_local.json')
    if not rows:raise ValueError('Empty existing group')
    inputs=dest/'input';inputs.mkdir(parents=True,exist_ok=True)
    for name in ['clean','context']:
        shutil.copy2(src/'input'/f'{name}.png',inputs/f'{name}.png')
    shutil.copy2(src/'input/transform.json',inputs/'transform.json')
    baseline={};targets=[];layers=[]
    for i,old in enumerate(rows,1):
        rid=f'R{i}';folder=old['id'].replace(':','_').replace('/','_')
        revised=baseline_run/'cases'/case/'curves'/folder/'after_local.json'
        row=read(revised) if revised.exists() else old
        if row['id']!=old['id'] or row['style']!=old['style']:raise ValueError('Baseline identity/style mismatch')
        row=copy.deepcopy(row);row['label']=rid;baseline[rid]=row
        masks=[]
        for j,p in enumerate(row['source_mask_paths']):
            dst=inputs/f'{rid}_mask_{j}.png';shutil.copy2(p,dst);masks.append(np.asarray(Image.open(dst).convert('L'))>0)
        union=np.logical_or.reduce(masks) if masks else np.zeros((Image.open(inputs/'clean.png').height,Image.open(inputs/'clean.png').width),bool)
        Image.fromarray(union.astype('uint8')*255).save(inputs/f'{rid}_mask.png');layers.append((rid,union))
        targets.append(dict(id=rid,heads=row['heads'],global_targets=row.get('globals',[])))
    clean=Image.open(inputs/'clean.png').convert('RGB');visual.masks(clean,layers,inputs/'masks.png')
    # No computed path or old point coordinates are included in the fit request.
    write(dest/'baseline.json',baseline)
    viz.render(inputs/'clean.png',list(baseline.values()),dest/'baseline')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir() if p.is_file()}
    write(inputs/'manifest.json',hashes)
    return targets,clean.size,baseline


def export_local_ppt(clean,rows,folder):
    """Pad a very narrow crop for PPT slide limits, keeping every pixel coordinate."""
    from PIL import Image
    import native
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    image=Image.open(clean).convert('RGB');w,h=image.size
    size=(max(w,math.ceil(h/4)),max(h,math.ceil(w/4)))
    write(folder/'ppt_canvas.json',dict(original_size=[w,h],canvas_size=list(size),offset=[0,0],
                                       geometry_and_style_coordinates_unchanged=True))
    native.write(size,rows,[r['style'] for r in rows],folder/'arrows.pptx')


def report(run,results):
    lines=['# GLM 自由分段几何：局部对照','','固定目标身份、原图、mask 和样式。GLM 可重新给端点与直线/曲线分段；不再由 mask 改写其几何。',
           '每组点击名称可查看实际输入、prompt、输出关键点及原生 PPT。运行中的项会陆续更新。','',
           '| 组 | 状态 | 原图 | 当前拟合 | GLM 首轮 | 一次反馈修正 |','|---|---|---|---|---|---|']
    for item in results:
        prefix='cases/'+item['case']+'/'+item['group'];p=run/prefix
        state=item['state'];after=f'![]({prefix}/glm/overlap.png)' if (p/'glm/overlap.png').exists() else '等待结果'
        refined=f'![]({prefix}/refined/overlap.png)' if (p/'refined/overlap.png').exists() else '未追加'
        lines.append(f'| [{item["case"]}/{item["group"]}]({prefix}/README.md) | {state} | ![]({prefix}/input/clean.png) | ![]({prefix}/baseline/overlap.png) | {after} | {refined} |')
    (run/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--scope',type=Path);parser.add_argument('--reuse',type=Path)
    parser.add_argument('--frozen',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parent;run=args.run.resolve();run.mkdir(parents=True,exist_ok=True)
    if not args.frozen:
        if (run/'code_manifest.json').exists():raise ValueError('New code requires a new run snapshot')
        read(args.scope)
        files=[*root.glob('*.py'),*(root/'lib').glob('*.py'),*(root/'transport').glob('*.py')];hashes={}
        for src in files:
            rel=src.relative_to(root);dst=run/'code'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
            hashes[str(rel)]=hashlib.sha256(dst.read_bytes()).hexdigest()
        write(run/'code_manifest.json',hashes);shutil.copy2(args.scope,run/'scope.json');shutil.copy2(root/'config.json',run/'config.json')
        if args.reuse:write(run/'reuse_from.json',dict(path=str(args.reuse.resolve())))
        os.execv(sys.executable,[sys.executable,str(run/'code/segmented_geometry_trial.py'),'--run',str(run),'--frozen'])
    sys.path.insert(0,str(root/'lib'))
    import viz,native
    from PIL import Image
    from transport.model import Model
    config=read(run/'config.json');scope=read(run/'scope.json')
    reuse=Path(read(run/'reuse_from.json')['path']) if (run/'reuse_from.json').exists() else None
    model=Model(config,run_dir=run,reuse_from=reuse,system='Fit the specified local geometry from the original image. Return the requested JSON only.')
    source=Path(scope['source']);baseline_run=Path(scope['baseline']);records=[];jobs=[]
    for task in scope['groups']:
        case,gid=task['case'],task['group'];dest=run/'cases'/case/gid
        targets,size,baseline=prepare(source,baseline_run,case,gid,dest)
        p=prompt(targets,size);write(dest/'prompt.json',p)
        record=dict(case=case,group=gid,state='prepared',reason=task['reason']);records.append(record)
        jobs.append((record,dest,targets,size,baseline,p))
        (dest/'README.md').write_text(f'# {case}/{gid}\n\n{task["reason"]}\n\n[文本 prompt](prompt.json) · [基线](baseline.json)\n\n| CLEAN | 目标 mask | 全图上下文 |\n|---|---|---|\n| ![](input/clean.png) | ![](input/masks.png) | ![](input/context.png) |\n')
    report(run,records);write(run/'progress.json',records)
    write(run/'runtime.json',dict(pid=os.getpid(),started=time.time(),workers=len(jobs),api_concurrency=config['api_concurrency'],sam_calls=0))
    lock=threading.Lock()
    def work(job):
        record,dest,targets,size,baseline,p=job;start=time.monotonic()
        with lock:record['state']='running';write(run/'progress.json',records)
        try:
            images=[('CLEAN: local coordinates',dest/'input/clean.png'),('MASKS: target evidence',dest/'input/masks.png'),('CONTEXT: whole figure',dest/'input/context.png')]
            answer=model.ask(dest/'request',p,images,lambda value:validate(value,baseline,size));write(dest/'answer.json',answer)
            rows=geometry(answer,baseline);write(dest/'rows_local.json',rows)
            viz.render(dest/'input/clean.png',rows,dest/'glm');draw_points(answer,dest/'input/clean.png',dest/'glm/points.png')
            export_local_ppt(dest/'input/clean.png',rows,dest)
            with (dest/'README.md').open('a') as f:
                f.write('\n[GLM 原始输出](answer.json) · [完整请求 trace](request/) · [原生 PPT](arrows.pptx)\n\n| 当前拟合 | GLM 点位 | GLM 几何 |\n|---|---|---|\n| ![](baseline/overlap.png) | ![](glm/points.png) | ![](glm/overlap.png) |\n')
            record.update(state='completed',routes=len(rows),nodes=len(answer['nodes']))
            if record['case']+'/'+record['group'] in scope.get('feedback_groups',[]):
                try:
                    revise=copy.deepcopy(p)
                    revise['task']='Compare CURRENT against CLEAN and correct this existing geometry if needed. Return the complete geometry in the same format.'
                    revise['previous_geometry']=answer
                    revise['instructions'] += ['CURRENT is a rendered prediction, never ground truth. Check endpoints, bends, line/curve transitions and true junctions against CLEAN. Keep already correct portions; do not add targets.']
                    correction=model.ask(dest/'feedback',revise,[('CLEAN',dest/'input/clean.png'),
                        ('MASKS: original target evidence',dest/'input/masks.png'),
                        ('CURRENT: prediction to check',dest/'glm/overlap.png')],lambda value:validate(value,baseline,size))
                    write(dest/'refined_answer.json',correction);newrows=geometry(correction,baseline)
                    write(dest/'refined_rows.json',newrows);viz.render(dest/'input/clean.png',newrows,dest/'refined')
                    draw_points(correction,dest/'input/clean.png',dest/'refined/points.png')
                    export_local_ppt(dest/'input/clean.png',newrows,dest/'refined');record['feedback']='completed'
                    with (dest/'README.md').open('a') as f:
                        f.write('\n一次渲染反馈修正（不提供人工坐标）：[输出](refined_answer.json) · [请求](feedback/)\n\n![](refined/overlap.png)\n')
                except Exception as exc:
                    write(dest/'feedback_failure.json',dict(error=str(exc)));record['feedback']='failed'
        except Exception as exc:
            write(dest/'failure.json',dict(error=str(exc)));record.update(state='failed',error=str(exc))
        record['seconds']=round(time.monotonic()-start,3);write(dest/'completed.json',record)
        with lock:write(run/'progress.json',records);report(run,records)
        print(record['case'],record['group'],record['state'],record['seconds'],flush=True)
    with ThreadPoolExecutor(max_workers=min(config['api_concurrency'],len(jobs))) as pool:list(pool.map(work,jobs))
    write(run/'completed.json',dict(groups=len(records),completed=sum(r['state']=='completed' for r in records),
                                   failed=sum(r['state']=='failed' for r in records),sam_calls=0))


if __name__=='__main__':main()
