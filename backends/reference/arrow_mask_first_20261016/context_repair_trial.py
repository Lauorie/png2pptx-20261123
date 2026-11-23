"""Local continuation experiment with frozen inputs and explicit existing paths.

No production algorithm changes. A short scope decision precedes geometry.
New paths are geometry previews; style recovery and PPT export are not tested.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import evidence as ev
import segmented_geometry as sg
import visual
from lib import fit
from transport.model import Model


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def points(row):
    if row.get('beziers'):
        parts = []
        for c in np.asarray(row['beziers']):
            t = np.linspace(0, 1, 16)[:, None]
            parts.append((1-t)**3*c[0]+3*(1-t)**2*t*c[1]+3*(1-t)*t**2*c[2]+t**3*c[3])
        return np.concatenate(parts)
    return np.asarray([row['start'], *row.get('via', []), row['end']], float)


def transform_points(p, matrix):
    return (np.c_[p, np.ones(len(p))] @ np.asarray(matrix).T)[:, :2]


def overlaps(row, box):
    p = points(row)
    return not (np.any(p.max(0) < box[:2]) or np.any(p.min(0) > box[2:]))


def draw_paths(clean, rows, view, path, new_ids=(), labels=False):
    image = clean.copy()
    d = ImageDraw.Draw(image)
    for r in rows:
        p = transform_points(points(r), view['global_to_view'])
        color = '#dc168c' if r['id'] in new_ids else '#1684cc'
        d.line([tuple(x) for x in p], fill=color, width=2)
        if labels:
            visible = p[(p[:,0]>=0)&(p[:,0]<image.width)&(p[:,1]>=0)&(p[:,1]<image.height)]
            if len(visible):visual.label(d, visible[len(visible)//2], r.get('short_id', r['id']), color, image)
    image.save(path)


def prepare_frame(case, group, dest, config, factor=1., requested=None):
    original = Image.open(case / 'clean.png').convert('RGB')
    old_view = read(group / 'input/transform.json')
    if requested is None and factor == 1:
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(group / 'input/clean.png', dest / 'clean.png')
        view = old_view
        write(dest / 'transform.json', view)
        crop = Image.open(dest / 'clean.png').convert('RGB')
    else:
        box = np.array(requested if requested is not None else old_view['bbox_global'], float)
        center = (box[:2]+box[2:])/2
        extent = (box[2:]-box[:2])*(factor if requested is None else 1.)
        cfg = {**config, 'display_side': 1080, 'context_padding': 0}
        crop, view = ev.frame(original, [], dest, cfg, box=[*(center-extent/2), *(center+extent/2)])
    context = original.copy()
    context.thumbnail((768,768), Image.Resampling.LANCZOS)
    cb = np.array(view['bbox_global'])*np.tile(np.array(context.size)/original.size, 2)
    ImageDraw.Draw(context).rectangle(cb.tolist(), outline='#c00080', width=2)
    context.save(dest / 'context.png')
    mask_ids = set(read(group / 'sam_summary.json')['selected_ids'])
    candidates = read(group / 'candidates.json')
    layers, metadata = [], []
    for c in candidates:
        if c['id'] not in mask_ids:continue
        m = ev.project(dict(path=c['path'], view=old_view), view)
        y,x = np.nonzero(m)
        if not len(x):continue
        layers.append((c['id'],m))
        metadata.append(dict(id=c['id'],bbox=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]))
    visual.masks(crop,layers,dest / 'masks.png')
    write(dest / 'masks.json',metadata)
    # Focus is a spatial locator, not a permission boundary for connected parts.
    hints = {h['id']:h for h in read(case / 'global_hints.json')}
    gs = next(g['sources'] for g in read(case / 'observation_plan.json')['groups'] if g['id']==group.name)
    focus=[]
    for gid in gs:
        b=hints[gid]['box'];p=transform_points([b[:2],b[2:]],view['global_to_view'])
        focus.append(dict(id=gid,bbox=np.round(p.ravel(),1).tolist()))
    return crop, view, context.size, metadata, focus


def inventory(rows, view):
    items=[]
    for row in rows:
        p=transform_points(points(row),view['global_to_view'])
        # Prompt simplification only. Rendered input retains the actual geometry.
        q=cv2.approxPolyDP(p.astype('float32'),.4,False).reshape(-1,2)
        items.append(dict(id=row['short_id'],kind=row['kind'],points=[[round(float(x),1),round(float(y),1)] for x,y in q],
            heads=row.get('heads',['none','none'])))
    return items


def plan_prompt(size,context_size,focus,masks,existing,allow_expand):
    return dict(task='Check whether the connection around FOCUS is completely recovered. Decide only the needed local repair; do not fit geometry or style yet.',
        size=list(size),context_size=list(context_size),focus=focus,masks=masks,existing=existing,
        instructions=[
            'CLEAN is the source. CURRENT shows recovered paths, not ground truth. MASKS are optional cached evidence; their coverage may be partial. FOCUS locates the connection, not an exhaustive list of its branches.',
            'Include the complete related connector or shared network, including a missing short turn or branch. Crossing alone does not mean connected. Exclude text, shape borders, chart axes/baselines, and internal icon wiring. Unrelated neighbors are context.',
            'Compare visible source lines with actual existing paths. Do not assume another group has recovered a line. Keep correct existing parts; replace lists only existing IDs that must be rebuilt. Use replace=[] to add only a missing portion. Do not recreate existing paths as new copies.',
            'Return complete if the connection is already complete, exclude if FOCUS is outside scope, repair with a short identifying description if incomplete, or expand if the needed continuation cannot be judged inside CLEAN. Existing coordinates can extend outside CLEAN; those are known continuations, not invalid points.',
            ('For expand, give context_bbox in CONTEXT image pixels covering the relevant connection and its endpoints.' if allow_expand else 'This is the last context expansion; use complete, exclude or repair for visible evidence. State uncertainty in reason if unresolved.')],
        output_schema=dict(status='complete|exclude|repair|expand',replace='[existing IDs]',description='short repair scope, or empty',context_bbox='[x0,y0,x1,y1] for expand; otherwise null',reason='brief'))


def validate_plan(answer, ids, context_size, allow_expand):
    if not isinstance(answer,dict) or answer.get('status') not in ['complete','exclude','repair','expand']:raise ValueError('Invalid status')
    replace=answer.get('replace',[])
    if not isinstance(replace,list) or not set(replace)<=set(ids):raise ValueError('replace must reference existing IDs')
    if answer['status']=='repair' and not isinstance(answer.get('description'),str):raise ValueError('Repair needs a description')
    if answer['status']=='expand':
        b=answer.get('context_bbox')
        if not allow_expand or not isinstance(b,list) or len(b)!=4:raise ValueError('Expansion unavailable or invalid bbox')
        if not all(type(x) in (int,float) and math.isfinite(x) for x in b):raise ValueError('Finite bbox required')
        if not (0<=b[0]<b[2]<=context_size[0] and 0<=b[1]<b[3]<=context_size[1]):raise ValueError('CONTEXT bbox out of bounds')
    return answer


def geometry_prompt(size, plan, masks, existing):
    return dict(task='Recover only the local repair described below. Return centerline geometry, not style measurements.',
        size=list(size),repair=plan,existing=existing,mask_measurements=masks,
        instructions=[
            'Use CLEAN as evidence. Preserve correct existing paths; output only new portions, plus complete replacements for the IDs in repair.replace. Existing paths outside that list remain in the result. Never output their copies.',
            'The repair scope is one related connection/network; choose its actual route count from the image. Use shared node IDs for true junctions. Mere crossings stay separate. Existing point coordinates are useful attachment references, not ground truth.',
            'Use line segments at sharp corners. Use curve segments only for visible curves, with at least three ordered on-curve points. Endpoints matter. Masks/bboxes are optional measurements, not centerlines. All output coordinates are CLEAN pixels.',
            'Each route has heads [first,last]: none no head; triangle filled triangle; stealth notched triangle; arrow open V; diamond rhombus; oval ellipse. Return compact JSON.'],
        output_schema=dict(nodes='[[id,x,y],...]',routes='[{id,heads:[type,type],segments:[{kind:line|curve,nodes:[ids]}]}]'))


def validate_geometry(answer,size):
    if not isinstance(answer,dict) or not isinstance(answer.get('routes'),list) or not answer['routes']:raise ValueError('Repair must contain routes')
    baseline={}
    for r in answer['routes']:
        if not isinstance(r.get('heads'),list) or len(r['heads'])!=2 or any(h not in ['none','triangle','stealth','arrow','diamond','oval'] for h in r['heads']):raise ValueError('Two valid heads required')
        baseline[r['id']]={}
    return sg.validate(answer,baseline,size)


def report(run,records):
    lines=['# 局部补全：简化 prompt、扩大上下文、共享已有几何','','只验证局部几何覆盖；复用冻结 SAM，未改变样式算法或主线结果。每项链接可看实际输入和输出。',
        '', '| 局部 / 条件 | 状态 | 原图 | 修复前实际结果 | 几何修复预览 |', '|---|---|---|---|---|']
    for r in records:
        rel=f'cases/{r["case"]}/{r["group"]}/{r["arm"]}';p=run/rel
        stage=r.get('view_folder','input')
        after=f'![]({rel}/after.png)' if (p/'after.png').exists() else '等待'
        lines.append(f'| [{r["case"]}/{r["group"]}/{r["arm"]}]({rel}/README.md) | {r["state"]} | ![]({rel}/{stage}/clean.png) | ![]({rel}/before.png) | {after} |')
    (run/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--scope',type=Path);parser.add_argument('--frozen',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parent;run=args.run.resolve();run.mkdir(parents=True,exist_ok=True)
    if not args.frozen:
        if (run/'code_manifest.json').exists():raise ValueError('Use a new run snapshot')
        scope=read(args.scope);source=Path(scope['source']);hashes={}
        for src in [*(source/'code').glob('*.py'),*(source/'code/lib').glob('*.py'),*(source/'code/transport').glob('*.py')]:
            rel=src.relative_to(source/'code');dest=run/'code'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);hashes[str(rel)]=sha(dest)
        dest=run/'code/context_repair_trial.py';shutil.copy2(Path(__file__),dest);hashes['context_repair_trial.py']=sha(dest)
        write(run/'code_manifest.json',hashes);write(run/'scope.json',scope)
        shutil.copy2(source/'config.json',run/'config.json')
        os.execv(sys.executable,[sys.executable,str(dest),'--run',str(run),'--frozen'])
    config=read(run/'config.json');scope=read(run/'scope.json');source=Path(scope['source'])
    model=Model(config,run_dir=run,system='Use the original image as evidence. Existing geometry and masks are fallible observations. Return the requested JSON only.')
    records=[{**t,'state':'prepared'} for t in scope['tasks']];lock=threading.Lock()
    def update():
        with lock:write(run/'progress.json',records);report(run,records)
    write(run/'runtime.json',dict(pid=os.getpid(),started=time.time(),workers=min(8,len(records)),api_concurrency=64,new_sam_calls=0))
    update()
    def work(record):
        started=time.monotonic();case=source/'cases'/record['case'];group=case/'groups'/record['group'];dest=run/'cases'/record['case']/record['group']/record['arm'];dest.mkdir(parents=True,exist_ok=True)
        record['state']='preparing';update()
        try:
            allrows=read(case/'scene.json')['arrows'];allrows=sorted(allrows,key=lambda r:r['id'])
            allrows=[{**copy.deepcopy(r),'short_id':f'E{i+1}'} for i,r in enumerate(allrows)]
            selected=[r for r in allrows if r['id'].startswith(record['group']+':')]
            requested=None;plan=None
            for round_id in range(2):
                folder=dest/('input' if round_id==0 else 'expanded_input');record['view_folder']=folder.name
                crop,view,context_size,masks,focus=prepare_frame(case,group,folder,config,record['factor'],requested)
                neighbors=[r for r in allrows if overlaps(r,view['bbox_global'])]
                existing=neighbors if record['share'] else [r for r in selected if overlaps(r,view['bbox_global'])]
                inv=inventory(existing,view)
                draw_paths(crop,existing,view,folder/'current.png',labels=True)
                draw_paths(crop,neighbors,view,dest/'before.png')
                p=plan_prompt(crop.size,context_size,focus,masks,inv,round_id==0)
                write(folder/'plan_payload.json',p);write(folder/'existing_map.json',{r['short_id']:r['id'] for r in existing})
                images=[('CLEAN: local source',folder/'clean.png'),('MASKS: cached local SAM evidence, not exhaustive',folder/'masks.png'),('CURRENT: existing predicted geometry, may be incomplete',folder/'current.png'),('CONTEXT: whole figure and CLEAN locator',folder/'context.png')]
                write(folder/'images.json',[dict(label=l,path=str(f),sha256=sha(f)) for l,f in images])
                (dest/'README.md').write_text(f'# {record["case"]}/{record["group"]}/{record["arm"]}\n\n{record["purpose"]}\n\n当前视野：{folder.name}。蓝色为已有路径，最终预览中的紫色为新增/替换几何。只测几何，不重新拟合样式。\n\n| 原图 | SAM 证据 | 提供给模型的已有结果 |\n|---|---|---|\n| ![]({folder.name}/clean.png) | ![]({folder.name}/masks.png) | ![]({folder.name}/current.png) |\n\n[实际 prompt]({folder.name}/plan_payload.json) · [已有 ID 对应]({folder.name}/existing_map.json)\n')
                record['state']='planning';update()
                plan=model.ask(dest/f'plan_{round_id}',p,images,lambda a:validate_plan(a,[x['id'] for x in inv],context_size,round_id==0))
                write(dest/'plan_answer.json',plan)
                if plan['status']!='expand':break
                requested=(np.array(plan['context_bbox'])*np.tile(np.array(Image.open(case/'clean.png').size)/context_size,2)).tolist()
            patch=[];replace=set()
            if plan['status']=='repair':
                record['state']='geometry';update()
                q=geometry_prompt(crop.size,plan,masks,inv);write(dest/'geometry_payload.json',q)
                answer=model.ask(dest/'geometry',q,images,lambda a:validate_geometry(a,crop.size));write(dest/'geometry_answer.json',answer)
                baseline={r['id']:dict(id='repair:'+record['group']+':'+r['id'],heads=r['heads'],line_style='unknown') for r in answer['routes']}
                local=sg.geometry(answer,baseline)
                for row in local:
                    p=transform_points(points(row),view['view_to_global'])
                    row.update(start=p[0].tolist(),via=p[1:-1].tolist(),end=p[-1].tolist(),beziers=[])
                    patch.append(row)
                mapping={r['short_id']:r['id'] for r in existing};replace={mapping[x] for x in plan['replace']}
            elif plan['status']=='exclude':
                mapping={r['short_id']:r['id'] for r in existing};replace={mapping[x] for x in plan.get('replace',[])}
            after=[r for r in allrows if r['id'] not in replace]+patch
            write(dest/'patch.json',dict(replace=sorted(replace),added=patch))
            write(dest/'after_rows_global.json',after)
            draw_paths(crop,[r for r in after if overlaps(r,view['bbox_global'])],view,dest/'after.png',new_ids=[r['id'] for r in patch])
            with (dest/'README.md').open('a') as f:f.write('\n[完整性判断](plan_answer.json) · [补丁](patch.json)\n\n| 修复前实际结果 | 修复后几何预览 |\n|---|---|\n| ![](before.png) | ![](after.png) |\n')
            record.update(state='completed',decision=plan['status'],added=len(patch),replaced=len(replace),expanded=requested is not None)
        except Exception as exc:
            write(dest/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()));record.update(state='failed',error=str(exc))
        record['seconds']=round(time.monotonic()-started,2);write(dest/'completed.json',record);update()
        print(json.dumps(record,ensure_ascii=False),flush=True)
    with ThreadPoolExecutor(max_workers=min(8,len(records))) as pool:list(pool.map(work,records))
    write(run/'completed.json',dict(total=len(records),completed=sum(x['state']=='completed' for x in records),failed=sum(x['state']=='failed' for x in records),new_sam_calls=0))


if __name__=='__main__':main()
