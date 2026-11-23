#!/usr/bin/env python3
"""Parallel curve-only replay through the same fitter as the main pipeline."""
import argparse,copy,json,os,shutil,sys,hashlib,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed


def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2))


def refit_target(folder):
    """Read immutable local inputs, not a still-running source scene."""
    import numpy as np
    from PIL import Image
    root=Path(__file__).resolve().parent
    sys.path.insert(0,str(root/'lib'))
    import curve_sampling,mask_vectors,fit,viz
    folder=Path(folder);local=read(folder/'before_local.json')
    mask=np.asarray(Image.open(folder/'mask.png'))>0
    start=time.monotonic()
    if local.get('anchors'):
        measured=mask_vectors.fit_route(mask,'curve',local['anchors'],local['heads'],
                                       'mask',local.get('line_style','solid'))
    else:measured=curve_sampling.refine(mask,local)
    new=copy.deepcopy(local);new.update(measured)
    write(folder/'after_local.json',new)
    viz.render(folder/'clean.png',[local],folder/'before')
    viz.render(folder/'clean.png',[new],folder/'after')
    global_new=fit.transform(new,read(folder/'transform.json'))
    angle=lambda r:float(np.degrees(np.arctan2(*(np.subtract(r['beziers'][-1][3],r['beziers'][-1][2]))[::-1])))
    return global_new,dict(status='refit',seconds=round(time.monotonic()-start,3),
        old_end_tangent_deg=angle(local),new_end_tangent_deg=angle(new),
        before_segments=len(local['beziers']),after_segments=len(new['beziers']),
        evidence=new['fit_evidence'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--cases',nargs='+')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--frozen',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parent;run=args.run.resolve();run.mkdir(parents=True,exist_ok=True)
    if not args.frozen:
        if (run/'code_manifest.json').exists():raise ValueError('Use a new run for changed numerical code')
        hashes={}
        for src in [*root.glob('*.py'),*(root/'lib').glob('*.py'),*(root/'transport').glob('*.py')]:
            rel=src.relative_to(root);dst=run/'code'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
            hashes[str(rel)]=hashlib.sha256(dst.read_bytes()).hexdigest()
        write(run/'code_manifest.json',hashes)
        command=[sys.executable,str(run/'code/curve_replay.py'),'--source',str(args.source.resolve()),
                 '--run',str(run),'--workers',str(args.workers),'--frozen']
        if args.cases:command+=['--cases',*args.cases]
        os.execv(sys.executable,command)
    import numpy as np
    from PIL import Image
    sys.path.insert(0,str(root/'lib'))
    import viz,native
    started=time.monotonic();cases={};tasks=[];stats=[]
    write(run/'runtime.json',dict(pid=os.getpid(),started_unix=time.time(),command=[sys.executable,*sys.argv],
                                 code=str(root),worker_count=args.workers))
    write(run/'config.json',dict(mode='offline geometry replay',workers=args.workers,
        requested_cases=args.cases,identity_fixed=True,style_fixed=True,sam_calls=0,glm_calls=0))
    if (args.source/'config.json').exists():shutil.copy2(args.source/'config.json',run/'source_config.json')
    for source in sorted((args.source/'cases').iterdir()):
        if args.cases and source.name not in args.cases:continue
        if not (source/'scene.json').exists():continue
        # When comparing another numerical revision, reuse the original frozen
        # baseline, not the previous revision's partially improved geometry.
        baseline=source/'before.json' if (source/'before.json').exists() else source/'scene.json'
        scene=read(baseline);dest=run/'cases'/source.name;dest.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/'clean.png',dest/'clean.png');write(dest/'before.json',scene)
        state_path=source/'source_state.json' if (source/'source_state.json').exists() else source/'completed.json'
        source_state=read(state_path) if state_path.exists() else dict(state='partial_snapshot')
        write(dest/'source_state.json',source_state)
        cases[source.name]=dict(scene=scene,dest=dest,curves=[],source_state=source_state,
                               source_sha256=hashlib.sha256((dest/'before.json').read_bytes()).hexdigest())
        for index,row in enumerate(scene['arrows']):
            if row['kind']!='curve' or row.get('coordinate_source')!='mask_no_skeleton':continue
            record=dict(case=source.name,id=row['id'],label=row.get('label'),status='pending',index=index)
            try:
                gid=row['id'].split(':')[0];group=source/'groups'/gid
                evidence_id=gid+'_'+row['id'].split(':',1)[-1].replace('/','_')
                cached=source/'curves'/evidence_id
                if cached.exists():
                    group=cached;local=read(group/'before_local.json')
                    clean_path=group/'clean.png';view_path=group/'transform.json'
                    mask_paths=sorted(group.glob('mask_[0-9]*.png'))
                else:
                    if not group.exists():group=next(p.parent for p in source.rglob('rows_local.json') if any(r['id']==row['id'] for r in read(p)))
                    local=next(r for r in read(group/'rows_local.json') if r['id']==row['id'])
                    clean_path=group/'input/clean.png';view_path=group/'input/transform.json'
                    mask_paths=local['source_mask_paths']
                evidence=dest/'curves'/evidence_id;evidence.mkdir(parents=True,exist_ok=True)
                write(evidence/'before_local.json',local);shutil.copy2(clean_path,evidence/'clean.png')
                write(evidence/'transform.json',read(view_path))
                masks=[]
                for i,path in enumerate(mask_paths):
                    dst=evidence/f'mask_{i}.png';shutil.copy2(path,dst)
                    masks.append(np.asarray(Image.open(dst).convert('L'))>0)
                mask=np.logical_or.reduce(masks);Image.fromarray(mask.astype('uint8')*255).save(evidence/'mask.png')
                hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence.iterdir() if p.is_file()}
                write(evidence/'input_manifest.json',hashes)
                record['evidence_folder']=str(evidence.relative_to(run));tasks.append((evidence,record))
                cases[source.name]['curves'].append(evidence_id)
            except Exception as exc:
                record.update(status='preserved',error=str(exc));stats.append(record)
    if args.cases and set(args.cases)-set(cases):raise ValueError('Requested cases lack source scenes: '+str(set(args.cases)-set(cases)))
    write(run/'manifest.json',dict(source=str(args.source.resolve()),cases=list(cases),
        source_snapshots={c:dict(sha256=d['source_sha256'],state=d['source_state']) for c,d in cases.items()},
        mode='offline curve geometry only; no SAM/GLM calls',same_algorithm_all_cases=True,styles_fixed=True))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(refit_target,str(folder)):record for folder,record in tasks}
        for f in as_completed(jobs):
            record=jobs[f];row=cases[record['case']]['scene']['arrows'][record['index']]
            try:
                global_new,audit=f.result();record.update(audit)
                record.update(start_shift_px=float(np.linalg.norm(np.subtract(global_new['start'],row['start']))),
                              end_shift_px=float(np.linalg.norm(np.subtract(global_new['end'],row['end']))))
                for key in ['start','via','end','beziers','geometry_method','fit_evidence']:
                    row[key]=global_new[key]
            except Exception as exc:record.update(status='preserved',error=str(exc))
            stats.append(record);write(run/'metrics.json',stats)
            write(run/'progress.json',dict(curves_ended=len(stats),curve_tasks=len(tasks),cases=len(cases),
                  seconds=round(time.monotonic()-started,2)))
            print(record['case'],record['id'],record['status'],flush=True)
    for cid,data in cases.items():
        dest=data['dest'];scene=data['scene'];before=read(dest/'before.json')
        write(dest/'scene.json',scene);viz.render(dest/'clean.png',before['arrows'],dest/'before')
        shutil.copy2(dest/'before/styled.png',dest/'before.png')
        viz.render(dest/'clean.png',scene['arrows'],dest/'result')
        native.write(Image.open(dest/'clean.png').size,scene['arrows'],[r['style'] for r in scene['arrows']],dest/'arrows.pptx')
        case_stats=[s for s in stats if s['case']==cid]
        write(dest/'completed.json',dict(state='completed_with_issues' if any(s['status']!='refit' for s in case_stats) else 'completed',
            refit=sum(s['status']=='refit' for s in case_stats),unchanged=len(scene['arrows'])-sum(s['status']=='refit' for s in case_stats),
            source_state=data['source_state'],source_is_snapshot=True))
        lines=['# '+cid+'：曲线修正对照','','身份、头型、样式固定；使用主线同一拟合入口。源图尚未完成的部分不在本次快照中。','',
            '| 原图 | 原结果 | 曲线修正后 | 叠图 |','|---|---|---|---|',
            '| ![](clean.png) | ![](before.png) | ![](result/styled.png) | ![](result/overlap.png) |','',
            '| 局部组 | 原图 | mask | 原几何 | 新几何 |','|---|---|---|---|---|']
        for gid in data['curves']:
            p='curves/'+gid;lines.append(f'| {gid} | ![]({p}/clean.png) | ![]({p}/mask.png) | ![]({p}/before/overlap.png) | ![]({p}/after/overlap.png) |')
        (dest/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    lines=['# 曲线末端修正：选定 case 回归','','复用已确认身份和 SAM mask；图内独立曲线使用 '+str(args.workers)+' 个 CPU worker 并行。GLM/SAM 新请求均为 0。',
        '原图、身份、样式固定，仅替换 mask 曲线的数值拟合；每张另有可编辑原生 PPT。','',
        '| case | 原图 | 原结果 | 修正后样式 | 实例 | 叠图 |','|---|---|---|---|---|---|']
    for cid in cases:
        p='cases/'+cid;lines.append(f'| [{cid}]({p}/RESULTS.md) | ![]({p}/clean.png) | ![]({p}/before.png) | ![]({p}/result/styled.png) | ![]({p}/result/instances.png) | ![]({p}/result/overlap.png) |')
    (run/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    write(run/'completed.json',dict(state='completed_with_issues' if any(s['status']!='refit' for s in stats) else 'completed',
        cases=len(cases),refit=sum(s['status']=='refit' for s in stats),issues=sum(s['status']!='refit' for s in stats),
        seconds=round(time.monotonic()-started,2),sam_calls=0,glm_calls=0))
    write(run/'progress.json',dict(stage='completed',cases_ended=len(cases),curves_ended=len(stats),
          curve_tasks=len(tasks),seconds=round(time.monotonic()-started,2)))


if __name__=='__main__':main()
