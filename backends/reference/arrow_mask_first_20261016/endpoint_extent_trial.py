"""Offline 40-case ablation: preserve model-owned longitudinal endpoint extent.

Only explicit-anchor straight mask fits change; D proposals/GLM networks/curves
and all styles remain fixed. No skeletons, new mask filtering or new identities.
"""
import argparse,copy,os,shutil,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from PIL import Image


def read(p):
    import json
    return json.loads(Path(p).read_text())


def write(p,v):
    import json
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def case_task(args):
    baseline,output,cid=args;source=Path(baseline)/'cases'/cid;out=Path(output)/'cases'/cid;out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(Path(__file__).resolve().parent/'lib'))
    import fit,viz
    scene=read(source/'scene.json');rows=copy.deepcopy(scene['arrows']);byid={r['id']:r for r in rows};changes=[]
    for section in ['groups','missing']:
        for file in sorted((source/section).glob('*/rows_local.json')):
            for row in read(file):
                if row.get('geometry_method')!='mask_robust_axis' or row['id'] not in byid:continue
                anchors=np.asarray(row['anchors'],float)
                masks=[np.asarray(Image.open(p).convert('L'))>0 for p in row['source_mask_paths']]
                mask=np.logical_or.reduce(masks);y,x=np.nonzero(mask)
                axis=fit.robust_axis(np.c_[x,y].astype(float),anchors[0],anchors[-1])
                local=np.array([axis[2],axis[3]])
                view=read(Path(row['local_input']).with_name('transform.json'))
                global_points=(np.c_[local,np.ones(2)]@np.asarray(view['view_to_global']).T)[:,:2]
                dest=byid[row['id']];before=np.array([dest['start'],dest['end']]);shift=np.linalg.norm(global_points-before,axis=1)
                dest.update(start=global_points[0].tolist(),via=[],end=global_points[1].tolist(),beziers=[],
                    geometry_method='probe_mask_axis_model_extent',endpoint_trial=True)
                changes.append(dict(id=row['id'],label=dest.get('label'),before=before.tolist(),after=global_points.tolist(),
                    shift_global=shift.tolist(),source_row_file=str(file),local_anchors=anchors.tolist()))
    # Validate that this ablation changed no physical identities or styles.
    assert [r['id'] for r in rows]==[r['id'] for r in scene['arrows']]
    assert all(a['style']==b['style'] and a['heads']==b['heads'] for a,b in zip(rows,scene['arrows']))
    shutil.copy2(source/'clean.png',out/'clean.png')
    viz.render(out/'clean.png',rows,out/'after')
    before=out/'before';before.mkdir(exist_ok=True)
    for name in ['overlap.png','styled.png','instances.png']:shutil.copy2(source/'result'/name,before/name)
    write(out/'scene.json',{**scene,'arrows':rows});write(out/'changes.json',changes)
    (out/'README.md').write_text('# '+cid+'：直线范围交接对照\n\n固定身份、样式与其他路径；仅对原 mask_robust_axis 直线，保留模型指定纵向范围、用同一 mask 估计轴向。\n\n[变化记录](changes.json)\n\n| 原图 | V4 | 实验结果 |\n|---|---|---|\n| ![](clean.png) | ![](before/overlap.png) | ![](after/overlap.png) |\n')
    return dict(case=cid,changed=len(changes),max_shift=max([max(r['shift_global']) for r in changes] or [0]),changes=changes)


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--frozen',action='store_true');a=p.parse_args()
    out=a.run.resolve();base=a.baseline.resolve();out.mkdir(parents=True,exist_ok=True)
    if not a.frozen:
        if (out/'code').exists():raise ValueError('Snapshot exists')
        shutil.copytree(base/'code',out/'code');shutil.copy2(__file__,out/'code/endpoint_extent_trial.py')
        import hashlib
        write(out/'code_manifest.json',{str(f.relative_to(out/'code')):hashlib.sha256(f.read_bytes()).hexdigest() for f in (out/'code').rglob('*.py')})
        write(out/'config.json',dict(baseline=str(base),workers=8,rule='axis projections of explicit physical endpoints; keep D proposals and all other branches unchanged'))
        os.execv(sys.executable,[sys.executable,'-u',str(out/'code/endpoint_extent_trial.py'),'--run',str(out),'--baseline',str(base),'--frozen'])
    cases=sorted(read(base/'finished.json')['cases']);results=[];started=time.monotonic()
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(case_task,(str(base),str(out),c)) for c in cases]
        for f in as_completed(futures):results.append(f.result())
    shifts=[v for r in results for c in r['changes'] for v in c['shift_global']]
    write(out/'summary.json',dict(cases=len(results),arrows_changed=sum(r['changed'] for r in results),endpoint_shift_quantiles=np.quantile(shifts,[.5,.9,.95,1]).tolist(),seconds=time.monotonic()-started,new_api_calls=0,production_modified=False,results=results))
    text='# 直线端点范围交接：40 张离线对照\n\n未调用模型；仅测试已显式给出起终点的直线是否应保留该纵向范围。轴向仍由 mask 测量，身份和样式不变。该方案不能保证模型端点本身正确，待目视复查。\n\n| Case | 改动条数 | 原图 | V4 | 对照 |\n|---|---:|---|---|---|\n'
    for r in sorted(results,key=lambda x:x['case']):
        c=r['case'];text+=f'| [{c}](cases/{c}/README.md) | {r["changed"]} | ![](cases/{c}/clean.png) | ![](cases/{c}/before/overlap.png) | ![](cases/{c}/after/overlap.png) |\n'
    (out/'RESULTS.md').write_text(text);print('completed',len(results),sum(r['changed'] for r in results),flush=True)


if __name__=='__main__':main()
