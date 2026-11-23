"""Offline ablations and cached straight-arrow checks; never call SAM or GLM."""
import argparse,collections,copy,json,os,shutil,time,traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from PIL import Image
import pipeline
from transport.common import read_json as read,write_json as write,sha256


def mask_at(folder,row):
    mids=row.get('source_masks',[])
    if mids and all((folder/'masks'/f'{k}.png').exists() for k in mids):
        files=[folder/'masks'/f'{k}.png' for k in mids]
    else:
        local=row['id'].split(':',1)[-1]
        support=folder/f'repair_{local.split(":")[0][1:]}/proposal/support.png' if local.startswith('R') and ':' in local else folder/'proposal/support.png'
        files=[support]
    return np.logical_or.reduce([np.array(Image.open(p).convert('L'))>0 for p in files]),[dict(path=str(p.resolve()),sha256=sha256(p)) for p in files]


def ablate(pilot,out):
    results=[]
    for f in sorted(pilot.glob('cases/*/groups/*/rows_local.json')):
        cid,gid=f.parts[-4],f.parts[-2];folder=out/'ablation'/cid/gid;folder.mkdir(parents=True,exist_ok=True)
        clean=f.parent/'input/input/clean.png';rgb=np.array(Image.open(clean).convert('RGB'));rows=read(f)
        panels={};notes=[]
        for name,adaptive,semantic in [('A_previous',False,False),('B_head_region_only',True,False),
                                       ('C_model_category_only',False,True),('D_combined',True,True)]:
            changed=copy.deepcopy(rows)
            for r in changed:
                mask,_=mask_at(f.parent/'input',r)
                if not semantic:r.pop('line_style',None)
                r['style']=pipeline.style.recover(rgb,mask,r,adaptive_heads=adaptive)
                notes.append(dict(variant=name,id=r['id'],dash=r['style']['dash'],head_region=r['style']['evidence'].get('head_exclusion')))
            panels[name]=changed;write(folder/name/'rows.json',changed)
            pipeline.viz.render(clean,changed,folder/name)
        write(folder/'audit.json',notes)
        shutil.copy2(clean,folder/'clean.png')
        (folder/'README.md').write_text(f'# {cid} / {gid}: 样式消融\n\n固定同一份几何、头型、原图和 mask；只切换头部统计区间与 GLM 类别。\n\n'
            '| 原图 | 旧统计 | 仅头部区间 | 仅模型类别 | 两者结合 |\n|---|---|---|---|---|\n'
            '| ![](clean.png) | ![](A_previous/styled.png) | ![](B_head_region_only/styled.png) | ![](C_model_category_only/styled.png) | ![](D_combined/styled.png) |\n\n[audit](audit.json)\n')
        results.append(dict(case=cid,group=gid,details=notes))
    write(out/'ablation.json',results)
    return results


def broad_group(f,out):
    cid,category,gid=f.parts[-4:-1];rows=read(f);base=f.parent/'baseline';clean=base/'input/clean.png'
    eligible=[r for r in rows if r['kind']=='straight' and (r.get('start_head') or r.get('end_head'))]
    result=dict(case=cid,category=category,group=gid,rows_in_group=len(rows),tested=[],errors=[])
    if not eligible:return result
    rgb=np.array(Image.open(clean).convert('RGB'));before=copy.deepcopy(rows);after=copy.deepcopy(rows)
    indices={r['id']:i for i,r in enumerate(rows)};changed=[]
    for row in eligible:
        try:
            mask,provenance=mask_at(base,row);r=copy.deepcopy(row);r.pop('line_style',None)
            a=pipeline.style.recover(rgb,mask,r,adaptive_heads=False)
            b=pipeline.style.recover(rgb,mask,r,adaptive_heads=True)
            i=indices[r['id']];before[i]['style']=a;after[i]['style']=b
            period_a=a['dash'].get('period_px');period_b=b['dash'].get('period_px')
            major=a['dash']['type']!=b['dash']['type'] or (period_a and period_b and abs(period_a-period_b)>.1*period_a)
            result['tested'].append(dict(id=r['id'],before=a['dash'],after=b['dash'],
                head_region=b['evidence'].get('head_exclusion'),major_change=bool(major),support=provenance))
            if major:changed.append(r['id'])
        except Exception as exc:result['errors'].append(dict(id=row['id'],error=str(exc)))
    if changed:
        folder=out/'broad40'/cid/category/gid;folder.mkdir(parents=True,exist_ok=True)
        pipeline.viz.render(clean,before,folder/'before');pipeline.viz.render(clean,after,folder/'after')
        shutil.copy2(clean,folder/'clean.png');write(folder/'rows_before.json',before);write(folder/'rows_after.json',after)
        write(folder/'audit.json',result)
        (folder/'README.md').write_text(f'# {cid} / {gid}\n\n仅更改头部统计范围，无新的 GLM 虚实线判断；旧几何/头型/取证关系原样保留。\n\n'
            '| 原图 | 旧统计 | 新统计 |\n|---|---|---|\n| ![](clean.png) | ![](before/styled.png) | ![](after/styled.png) |\n\n[audit](audit.json)\n')
        result['preview']=str(folder.relative_to(out))+'/README.md'
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,required=True);p.add_argument('--baseline40',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True);root=Path(__file__).resolve().parent
    files={}
    for src in [Path(__file__),root/'lib/style.py']:
        dst=out/'code'/src.relative_to(root);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);files[str(src)]=sha256(dst)
    write(out/'code_manifest.json',files)
    ablations=ablate(a.pilot.resolve(),out);jobs=[]
    for category in ('groups','missing'):jobs.extend(a.baseline40.resolve().glob(f'cases/*/{category}/*/rows_local.json'))
    results=[];start=time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        tasks={pool.submit(broad_group,f,out):f for f in sorted(jobs)}
        for f in as_completed(tasks):
            try:results.append(f.result())
            except Exception as exc:results.append(dict(case=tasks[f].parts[-4],errors=[dict(error=str(exc),traceback=traceback.format_exc())],tested=[]))
            if len(results)%80==0:
                print(len(results),'/',len(tasks),flush=True)
                write(out/'progress.json',dict(done=len(results),total=len(tasks),seconds=time.time()-start))
    counts=collections.Counter();transitions=collections.Counter();cases=collections.defaultdict(lambda:collections.Counter())
    for g in results:
        cases[g['case']]['groups']+=1;cases[g['case']]['tested']+=len(g['tested'])
        for r in g['tested']:
            transitions[r['before']['type']+' -> '+r['after']['type']]+=1
            counts['tested']+=1;counts['major_changes']+=r['major_change'];cases[g['case']]['changed']+=r['major_change']
        counts['errors']+=len(g['errors'])
    summary=dict(cases=dict(cases),groups=len(results),counts=dict(counts),transitions=dict(transitions),seconds=time.time()-start,
        new_sam_calls=0,new_glm_calls=0,no_accuracy_ground_truth=True,
        note='Numerical head-window ablation only; inherited geometry/head/mask errors remain. Changes are not accuracy scores.')
    write(out/'summary.json',summary);write(out/'broad40.json',results)
    lines=['# 虚实线类别与头部统计范围：消融','','GLM 新接口只在 6 组试验。40 张缓存复验仅检查数值头部统计，不是新模型全量，也没有人工准确率。','',
           '[数值复验汇总](summary.json) · [全部测量](broad40.json)','','## 固定几何的四组对照','','| case | 对照 |','|---|---|']
    for r in ablations:
        b=f'ablation/{r["case"]}/{r["group"]}'
        lines.append(f'| {r["case"]} / {r["group"]} | [原图和四组结果]({b}/README.md) |')
    lines+=['','## 40 张缓存中变化较大的组','','| case / group | 原图 | 旧统计 | 新统计 |','|---|---|---|---|']
    for r in sorted(results,key=lambda x:(x['case'],x.get('group',''))):
        if not r.get('preview'):continue
        b=str(Path(r['preview']).parent)
        lines.append(f'| [{r["case"]} / {r["group"]}]({r["preview"]}) | ![]({b}/clean.png) | ![]({b}/before/styled.png) | ![]({b}/after/styled.png) |')
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
