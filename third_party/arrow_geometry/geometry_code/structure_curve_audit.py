"""Fixed-identity offline regression over every saved dashed mask curve."""
import argparse,copy,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'lib'))
import numpy as np
from PIL import Image
import mask_vectors
from lib import fit,viz
from transport.common import read_json as read,write_json as write


def main(base,out):
    base=Path(base).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True);records=[];groups=[]
    for path in sorted(base.glob('cases/*/groups/*/rows_local.json')):
        rows=read(path);indices=[i for i,r in enumerate(rows) if r.get('geometry_method')=='mask_mark_axis_samples']
        if not indices:continue
        cid,gid=path.parents[2].name,path.parent.name;folder=out/cid/gid;folder.mkdir(parents=True,exist_ok=True)
        revised=copy.deepcopy(rows)
        for i in indices:
            old=rows[i];p=fit.points(old)
            try:
                mask=np.logical_or.reduce([np.array(Image.open(x).convert('L'))>0 for x in old['source_mask_paths']])
                new=mask_vectors.fit_route(mask,'curve',old['anchors'],old['heads'],'mask',old['line_style'])
                q=fit.points(new)
                for k in ['start','via','end','beziers','geometry_method','fit_evidence']:revised[i][k]=new[k]
                records.append(dict(case=cid,group=gid,id=old['id'],state='refit',
                    before_length=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()),after_length=float(np.linalg.norm(np.diff(q,axis=0),axis=1).sum()),
                    endpoint_shift=float(np.linalg.norm(q[[0,-1]]-p[[0,-1]],axis=1).max()),
                    mask_ids=old['source_masks'],evidence=new['fit_evidence']))
            except Exception as exc:records.append(dict(case=cid,group=gid,id=old['id'],state='failed',error=str(exc)))
        clean=path.parent/'input/clean.png';viz.render(clean,rows,folder/'before');viz.render(clean,revised,folder/'after')
        write(folder/'before_rows.json',rows);write(folder/'after_rows.json',revised)
        groups.append((cid,gid,clean,folder))
    write(out/'audit.json',dict(model_calls=0,scope='All saved mask_mark_axis_samples in 40-case baseline',records=records))
    lines=['# 虚线曲线：固定身份、样式和端点的离线对照','',
        f'检查 40 张缓存；找到 {len(records)} 条旧碎块插值曲线、{len(groups)} 个局部组。无模型或 SAM 调用。',
        '', '路径长度用于定位折返异常，不是准确率。身份/端点错误不会因为换拟合器自动修复。',
        '', '| case / 路径 | 原长度（局部像素） | 新长度 | 最大端点变化 |','|---|---|---|---|']
    for r in records:
        lines.append(f'| {r["case"]}/{r["id"]} | {r.get("before_length",0):.1f} | {r.get("after_length",0):.1f} | {r.get("endpoint_shift",0):.3f} |')
    lines+=['','| case / 组 | 原图 | 原实例 | 加权拟合后 | 叠图 |','|---|---|---|---|---|']
    for cid,gid,clean,folder in groups:
        rel=folder.relative_to(out)
        lines.append(f'| {cid}/{gid} | ![]({os.path.relpath(clean,out)}) | ![]({rel}/before/instances.png) | ![]({rel}/after/instances.png) | ![]({rel}/after/overlap.png) |')
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print('curves',len(records),'groups',len(groups))
    for r in records:print(r['case'],r['id'],r.get('before_length'),r.get('after_length'),r.get('endpoint_shift'),r['state'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();main(a.baseline,a.out)
