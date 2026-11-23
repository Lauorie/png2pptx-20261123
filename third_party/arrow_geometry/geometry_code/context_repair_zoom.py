"""Re-examine a proposed repair in a closer view; old paths stay unchanged."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
import context_repair_trial as trial
from transport.model import Model


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--case',required=True);parser.add_argument('--group',required=True);args=parser.parse_args()
    run=args.run.resolve();source=Path(trial.read(run/'scope.json')['source']);case=source/'cases'/args.case;group=case/'groups'/args.group
    previous=run/'cases'/args.case/args.group/'wide_shared';out=previous/'zoom_patch';out.mkdir(parents=True,exist_ok=True)
    patch=trial.read(previous/'patch.json')['added'];points=np.concatenate([trial.points(r) for r in patch]);lo,hi=points.min(0),points.max(0);center=(lo+hi)/2;extent=3*(hi-lo)
    requested=[*(center-extent/2),*(center+extent/2)]
    config=trial.read(run/'config.json');crop,view,context_size,masks,focus=trial.prepare_frame(case,group,out/'input',config,requested=requested)
    rows=sorted(trial.read(case/'scene.json')['arrows'],key=lambda r:r['id']);rows=[{**copy.deepcopy(r),'short_id':f'E{i+1}'} for i,r in enumerate(rows)]
    nearby=[r for r in rows if trial.overlaps(r,view['bbox_global'])]
    inventory=trial.inventory(nearby,view)
    for item in inventory:item['points']=[[round(float(x),1),round(float(y),1)] for x,y in item['points']]
    trial.draw_paths(crop,nearby,view,out/'input/current.png',labels=True)
    trial.draw_paths(crop,nearby+patch,view,out/'input/proposed.png',new_ids=[r['id'] for r in patch])
    plan=dict(status='repair',replace=[],description='Recover only the missing portions of the connection identified by PROPOSED. Its magenta paths are a fallible proposal: independently determine actual branch count, endpoints, and connections from CLEAN. Do not reproduce extra proposed lines. Keep original CURRENT paths.')
    prompt=trial.geometry_prompt(crop.size,plan,masks,inventory)
    images=[('CLEAN: closer original view',out/'input/clean.png'),('MASKS: cached evidence',out/'input/masks.png'),('CURRENT: original existing paths, before any repair',out/'input/current.png'),('PROPOSED: previous repair to check; magenta may be wrong or incomplete',out/'input/proposed.png'),('CONTEXT: whole figure',out/'input/context.png')]
    trial.write(out/'prompt.json',prompt);trial.write(out/'inputs.json',dict(region_source='bbox of previous model-generated patch, expanded 3x',bbox_global=view['bbox_global'],new_sam_calls=0,images=[dict(label=l,path=str(p),sha256=trial.sha(p)) for l,p in images]))
    model=Model(config,run_dir=run,system='Recover local connection geometry from the original image. Proposals are fallible, not ground truth. Return the requested JSON only.')
    started=time.monotonic();answer=model.ask(out/'request',prompt,images,lambda a:trial.validate_geometry(a,crop.size));trial.write(out/'answer.json',answer)
    metadata={r['id']:dict(id='zoom:'+r['id'],heads=r['heads'],line_style='unknown') for r in answer['routes']}
    new=trial.sg.geometry(answer,metadata)
    for r in new:
        p=trial.transform_points(trial.points(r),view['view_to_global']);r.update(start=p[0].tolist(),via=p[1:-1].tolist(),end=p[-1].tolist(),beziers=[])
    trial.write(out/'patch.json',dict(replace=[],added=new))
    trial.draw_paths(crop,nearby+new,view,out/'after.png',new_ids=[r['id'] for r in new])
    trial.write(out/'completed.json',dict(seconds=time.monotonic()-started,added=len(new),source_modified=False))
    (out/'README.md').write_text('# 对修复区域再放大一次\n\n区域由上一轮模型生成的补丁范围自动扩大三倍，没有手写坐标或支路数量。已有路径不变，仅替换试验补丁；不重测样式。\n\n| 原图 | 第一轮提案 | 放大后重新判断 |\n|---|---|---|\n| ![](input/clean.png) | ![](input/proposed.png) | ![](after.png) |\n\n[实际 prompt](prompt.json) · [输入与范围](inputs.json) · [模型输出](answer.json) · [请求 trace](request/)\n')
    print(json.dumps(dict(folder=str(out),added=len(new)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
