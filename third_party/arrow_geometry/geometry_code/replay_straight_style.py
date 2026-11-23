"""Offline replay: freeze existing identities, heads, local masks; refit geometry/style.

No SAM/GLM, no new identities, no coverage. Uses the same numerical entry points
as pipeline.py. Incomplete upstream groups remain explicitly listed as missing.
"""
import argparse,copy,collections,json,shutil,time,traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from PIL import Image
import pipeline,straight_mask
from transport.common import read_json as read,write_json as write,sha256 as sha


def snapshot(source,out,cases):
    records=[]
    for cid in cases:
        src=source/'cases'/cid;dst=out/'cases'/cid;dst.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src/'clean.png',dst/'clean.png');groups=[];missing=[];hashes={}
        for category in ('groups','missing'):
            for g in sorted((src/category).glob('*')):
                if not g.is_dir():continue
                if not (g/'rows_local.json').exists():missing.append(str(g.relative_to(src)));continue
                target=dst/category/g.name;target.mkdir(parents=True,exist_ok=True)
                files=['rows_local.json','candidates.json','decision.json','input/clean.png','input/transform.json','proposal/support.png','proposal/draft.json']
                selected=[]
                for c in read(g/'candidates.json'):
                    if c.get('selected') and not c.get('represented_by'):
                        files.append('masks/'+c['id']+'.png');selected.append(c['id'])
                for p in g.glob('repair_*/proposal/support.png'):files.append(str(p.relative_to(g)))
                for f in files:
                    if not (g/f).exists():continue
                    dest=target/'baseline'/f;dest.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(g/f,dest);hashes[str(g/f)]=sha(dest)
                groups.append(dict(category=category,group=g.name,selected=selected))
        records.append(dict(case=cid,groups=groups,upstream_missing=missing,source_hashes=hashes))
    write(out/'snapshot.json',dict(source=str(source),captured_at=time.time(),cases=records))
    return records


def replay_group(root,cid,info):
    folder=root/'cases'/cid/info['category']/info['group'];base=folder/'baseline'
    before=read(base/'rows_local.json');clean=Image.open(base/'input/clean.png').convert('RGB')
    support=np.asarray(Image.open(base/'proposal/support.png').convert('L'))>0
    candidates=[dict(id=k,mask=np.asarray(Image.open(base/'masks'/f'{k}.png').convert('L'))>0) for k in info['selected']]
    rows=straight_mask.refit_confirmed(copy.deepcopy(before),candidates,read(base/'proposal/draft.json'))
    changes=[];preserved=[]
    for row in rows:
        m=support
        local_id=row['id'].split(':',1)[-1]
        if local_id.startswith('R') and ':' in local_id:
            p=base/f'repair_{local_id.split(":")[0][1:]}/proposal/support.png'
            if not p.exists():preserved.append(row['id']);continue
            m=np.asarray(Image.open(p).convert('L'))>0
        pipeline.style_rows(clean,[row],m)
    write(folder/'rows_local.json',rows)
    view=read(base/'input/transform.json')
    global_rows=[pipeline.fit.transform(r,view) for r in rows]
    for local,glob in zip(rows,global_rows):
        glob['native_route_path']=local.get('route_path');glob.pop('route_path',None)
    write(folder/'rows_global.json',global_rows)
    index={r['id']:r for r in before}
    for row in rows:
        old=index.get(row['id'],{})
        a=old.get('style',{}).get('dash',{}).get('type');b=row['style'].get('dash',{}).get('type')
        if a!=b or row.get('geometry_method')=='direct_straight_mask':
            changes.append(dict(id=row['id'],before_dash=a,after_dash=b,from_paths=row.get('refit_from_paths',[])))
    note=dict(group=info['group'],before_rows=len(before),after_rows=len(rows),
        refitted=any(r.get('geometry_method')=='direct_straight_mask' for r in rows),changes=changes,
        preserved_style_missing_support=preserved)
    write(folder/'audit.json',note)
    if changes:
        pipeline.viz.render(base/'input/clean.png',before,folder/'before')
        pipeline.viz.render(base/'input/clean.png',rows,folder/'result')
        (folder/'README.md').write_text('# '+info['group']+'\n\n| 原图 | 修正前 | 修正后 |\n|---|---|---|\n'
            '| ![](baseline/input/clean.png) | ![](before/styled.png) | ![](result/styled.png) |\n\n[测量记录](audit.json)\n')
    before_global=[pipeline.fit.transform(r,view) for r in before]
    return dict(case=cid,rows=global_rows,before=before_global,audit=note)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cases',nargs='*');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'snapshot.json').exists():raise RuntimeError('Use a fresh output directory to preserve the snapshot')
    root=Path(__file__).resolve().parent
    code=out/'code';code.mkdir();files={}
    for src in [*root.glob('*.py'),*root.glob('lib/*.py'),*root.glob('transport/*.py')]:
        dst=code/src.relative_to(root);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);files[str(src.relative_to(root))]=sha(dst)
    write(out/'code_manifest.json',files)
    cases=a.cases or sorted(x.name for x in (a.source/'cases').iterdir() if (x/'clean.png').exists())
    records=snapshot(a.source.resolve(),out,cases);results=collections.defaultdict(list);errors=[];started=time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        jobs={pool.submit(replay_group,out,record['case'],g):(record['case'],g) for record in records for g in record['groups']}
        for f in as_completed(jobs):
            cid,g=jobs[f]
            try:results[cid].append(f.result())
            except Exception as exc:errors.append(dict(case=cid,group=g['group'],error=str(exc),traceback=traceback.format_exc()))
            n=sum(map(len,results.values()))+len(errors)
            write(out/'progress.json',dict(done=n,total=len(jobs),errors=errors,seconds=time.time()-started))
            if n%40==0:print(n,'/',len(jobs),flush=True)
    summary=[]
    for record in records:
        cid=record['case'];folder=out/'cases'/cid;items=sorted(results[cid],key=lambda r:r['audit']['group'])
        clean=Image.open(folder/'clean.png').convert('RGB')
        pipeline.publish(clean,[dict(rows=r['rows']) for r in items],folder)
        pipeline.viz.render(folder/'clean.png',[row for r in items for row in r['before']],folder/'before')
        row=dict(case=cid,groups=len(items),refitted_groups=sum(x['audit']['refitted'] for x in items),
            before_rows=sum(len(x['before']) for x in items),after_rows=sum(len(x['rows']) for x in items),
            before_dash=dict(collections.Counter(r['style'].get('dash',{}).get('type','filled') for x in items for r in x['before'])),
            after_dash=dict(collections.Counter(r['style'].get('dash',{}).get('type','filled') for x in items for r in x['rows'])),
            upstream_missing=record['upstream_missing'])
        summary.append(row);write(folder/'audit.json',row)
        lines=['# '+cid,'','复用已有身份、头型和局部 mask；无模型请求、无补漏。未完成的上游分组仍列在 audit.json。','',
            '| 原图 | 修正前 | 修正后 | 叠图 |','|---|---|---|---|',
            '| ![](clean.png) | ![](before/styled.png) | ![](result/styled.png) | ![](result/overlap.png) |','',
            '[PPT](arrows.pptx) · [实例图](result/instances.png) · [统计](audit.json)','']
        for g in record['groups']:
            path=Path(g['category'])/g['group']/'README.md'
            if (folder/path).exists():lines.append(f'- [{g["group"]}]({path})')
        (folder/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    write(out/'summary.json',dict(cases=summary,errors=errors,seconds=time.time()-started,new_model_calls=0))
    lines=['# 孤立直线和虚线修正：离线复验','',
        '使用同一套代码处理全部缓存 case，保留模型身份与头型；没有新增 SAM/GLM 请求或补漏。上游未完成的分组不计为本次成功恢复。',
        'PPT 保留可编辑连接线及实测虚线长度/间隔；原生 PPT 不支持精确虚线相位，起始位置仍可能与 SVG 不同。','',
        '[汇总](summary.json) · [输入快照与哈希](snapshot.json)','',
        '| case | 原图 | 修正前 | 修正后 |','|---|---|---|---|']
    for cid in cases:lines.append(f'| [{cid}](cases/{cid}/RESULTS.md) | ![](cases/{cid}/clean.png) | ![](cases/{cid}/before/styled.png) | ![](cases/{cid}/result/styled.png) |')
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('COMPLETE',len(summary),'cases',len(errors),'errors',flush=True)


if __name__=='__main__':main()
