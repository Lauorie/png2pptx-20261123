"""Offline mask-overlap ablation. Masks remain evidence, never arrow identities.

Compare intact-instance suppression with destructive pixel exclusivity using
one configuration on the baseline's 40 cases. No SAM/GLM or mainline mutation.
"""
import argparse, hashlib, json, shutil, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

cv2.setNumThreads(1)
VARIANTS = ('single90', 'union85', 'union90', 'union95', 'single90_t1', 'union90_t1', 'peel')
COLORS = ['#d52522','#007da8','#ae23bd','#549315','#de7f00','#174fc5','#00947d','#c74680']


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(mask, box, ident, **meta):
    mask = np.asarray(mask, bool)
    yy, xx = np.nonzero(mask)
    if not len(xx): return None
    a,b,c,d = int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)
    return dict(id=ident, box=[int(box[0]+a),int(box[1]+b),int(box[0]+c),int(box[1]+d)],
                ink=mask[b:d,a:c].copy(), area=int(mask.sum()), **meta)


def load_mask(path, ident, **meta):
    with Image.open(path) as im:
        im=im.convert('L'); box=im.getbbox()
        if box is None: return None
        return record(np.asarray(im.crop(box))>0, box, ident, native_path=str(Path(path).resolve()), **meta)


def intersections(a,b):
    x0,y0=max(a['box'][0],b['box'][0]),max(a['box'][1],b['box'][1])
    x1,y1=min(a['box'][2],b['box'][2]),min(a['box'][3],b['box'][3])
    if x1<=x0 or y1<=y0:return 0
    aa=a['ink'][y0-a['box'][1]:y1-a['box'][1],x0-a['box'][0]:x1-a['box'][0]]
    bb=b['ink'][y0-b['box'][1]:y1-b['box'][1],x0-b['box'][0]:x1-b['box'][0]]
    return int((aa&bb).sum())


def components(mask):
    return int(cv2.connectedComponents(mask.astype('uint8'),connectivity=8)[0]-1)


def coverage(rows,size):
    a=np.zeros((size[1],size[0]),np.uint16)
    for r in rows:
        x0,y0,x1,y1=r['box']; a[y0:y1,x0:x1]+=r['ink']
    return a


def suppress(rows,size,method):
    occupied=np.zeros((size[1],size[0]),bool); matching=occupied.copy(); kept=[]; expanded=[]; audit=[]
    tolerance=1 if method.endswith('_t1') else 0
    base_method=method.removesuffix('_t1')
    order=sorted(rows,key=lambda r:(-r['area'],-r.get('score',0),r['id']))
    threshold={'single90':.9,'union85':.85,'union90':.9,'union95':.95}.get(base_method)
    for r in order:
        x0,y0,x1,y1=r['box']; existing=occupied[y0:y1,x0:x1]
        overlap=int((r['ink']&existing).sum()); exact_fraction=overlap/r['area']
        fraction=float((r['ink']&matching[y0:y1,x0:x1]).sum()/r['area']) if tolerance else exact_fraction
        parents=[]
        drop=False; output=r
        if base_method=='single90':
            parents=[p['id'] for p in (expanded if tolerance else kept) if intersections(r,p)/r['area']>=threshold]
            drop=bool(parents)
        elif method.startswith('union'):
            drop=fraction>=threshold
        elif method=='peel':
            residual=r['ink']&~existing
            output=record(residual,r['box'],r['id'],**{k:v for k,v in r.items() if k not in ('id','box','ink','area')})
            drop=output is None
        else:raise ValueError(method)
        if drop or method=='peel' and overlap:
            if not parents:parents=[p['id'] for p in kept if intersections(r,p)>0]
            event=dict(id=r['id'],action='drop' if drop else 'subtract',area_before=r['area'],
                       area_after=0 if drop else output['area'],covered_fraction=fraction,
                       exact_covered_fraction=exact_fraction,matching_tolerance_pixels=tolerance,parents=parents)
            if method=='peel':
                event.update(components_before=components(r['ink']),components_after=0 if drop else components(output['ink']))
            audit.append(event)
        if not drop:
            kept.append(output)
            x0,y0,x1,y1=output['box'];occupied[y0:y1,x0:x1]|=output['ink']
            if tolerance:
                x,y,z,w=max(0,x0-1),max(0,y0-1),min(size[0],x1+1),min(size[1],y1+1)
                patch=np.zeros((w-y,z-x),np.uint8);patch[y0-y:y1-y,x0-x:x1-x]=output['ink']
                patch=cv2.dilate(patch,np.ones((3,3),np.uint8))>0
                matching[y:w,x:z]|=patch;expanded.append(record(patch,[x,y,z,w],output['id']))
    return kept,audit


def evaluate(rows,size,reference):
    count=coverage(rows,size); union=count>0
    return dict(instances=len(rows),union_pixels=int(union.sum()),
        lost_union_pixels=int((reference&~union).sum()),duplicate_pixel_mass=int(count.sum()-union.sum()),
        overlap_pixels=int((count>1).sum()))


def serial(r):
    return {k:v for k,v in r.items() if k!='ink'}


def save_masks(folder,rows,modified=False):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);data=[]
    for r in rows:
        item=serial(r)
        if modified:
            dest=folder/(r['id']+'.png');Image.fromarray(r['ink'].astype('uint8')*255).save(dest)
            item['tight_mask_path']=str(dest.resolve())
        data.append(item)
    write(folder/'instances.json',data)


def preview(clean,rows,path,labels=True):
    """Small display-only renderer: no mask resizing/geometry changes to evidence."""
    canvas=np.asarray(clean.convert('RGB')).astype(float);sums=np.zeros_like(canvas);counts=np.zeros(canvas.shape[:2])
    for i,r in enumerate(rows):
        color=np.array([int(COLORS[i%len(COLORS)][j:j+2],16) for j in (1,3,5)])
        x0,y0,x1,y1=r['box'];m=r['ink'];sums[y0:y1,x0:x1][m]+=color;counts[y0:y1,x0:x1]+=m
    hit=counts>0;canvas[hit]=.4*canvas[hit]+.6*sums[hit]/counts[hit,None]
    image=Image.fromarray(canvas.astype('uint8'))
    if labels:
        draw=ImageDraw.Draw(image);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
        for i,r in enumerate(rows):
            x0,y0,x1,y1=r['box'];draw.text((min(image.width-48,x1+2),min(image.height-23,y1+2)),r['id'],fill=COLORS[i%len(COLORS)],font=font,stroke_width=1,stroke_fill='white')
    image.save(path)


def overlap_preview(rows,size,path):
    c=coverage(rows,size);rgb=np.full((size[1],size[0],3),255,np.uint8)
    rgb[c==1]=[150,175,195];rgb[c>=2]=[235,130,50];rgb[c>=4]=[195,40,80]
    Image.fromarray(rgb).save(path)


def project(r,view,global_size,ident):
    """Global proxy only; native mask remains the geometry/style evidence."""
    a=np.array(view['view_to_global']);x,y,x1,y1=r['box']
    corners=np.array([[x,y,1],[x1,y1,1]])@a.T
    lo=np.maximum(0,np.floor(corners[0,:2]-.5).astype(int));hi=np.minimum(global_size,np.ceil(corners[1,:2]+.5).astype(int))
    if np.any(hi<=lo):return None
    mat=a[:2,:].copy();mat[:,2]+=a[:2,:2]@np.array([x,y])-lo
    mask=cv2.warpAffine(r['ink'].astype('uint8'),mat,tuple((hi-lo).tolist()),flags=cv2.INTER_NEAREST)>0
    meta={k:v for k,v in r.items() if k not in ('ink','box','id','area')}
    return record(mask,[*lo,*hi],ident,native_id=r['id'],native_box=r['box'],view=view,**meta)


def process_case(args):
    baseline,out,cid=args;case=Path(baseline)/'cases'/cid;dest=Path(out)/'cases'/cid;dest.mkdir(parents=True,exist_ok=True)
    clean=Image.open(case/'clean.png').convert('RGB');size=clean.size
    frames=[];pool=[];global_rows=[];seq=0
    for g in read(case/'global_hints.json'):
        seq+=1;r=load_mask(g['path'],f'S{seq:04d}',score=g.get('score',0),source_kind='global',origin=g['id'])
        if r:pool.append(r);global_rows.append(r)
    for section in ('groups','missing'):
        for folder in sorted((case/section).glob('*')):
            if not (folder/'candidates.json').exists():continue
            active=folder;attempts=read(folder/'sampling_attempts.json') if (folder/'sampling_attempts.json').exists() else []
            if attempts and attempts[-1]['folder']!='.' and attempts[-1]['selected']:active=folder/attempts[-1]['folder']
            view=read(active/'input/transform.json');candidates=read(active/'candidates.json');native=[];all_native=[]
            for c in candidates:
                r=load_mask(c['path'],c['id'],score=c.get('score',0),source_kind='local',
                    origin=str(folder.relative_to(case)),category=c['category'],baseline_represented_by=c.get('represented_by'))
                if not r:continue
                all_native.append(r)
                if c['selected']:native.append(r)
                seq+=1;projected=project(r,view,size,f'S{seq:04d}')
                if projected:pool.append(projected)
            key=section+'_'+folder.name;fdest=dest/'groups'/key;fdest.mkdir(parents=True,exist_ok=True)
            ref=coverage(native,view['size'])>0
            actual=[r for r in native if not r['baseline_represented_by']]
            metrics={'input':evaluate(native,view['size'],ref),'baseline':evaluate(actual,view['size'],ref)};outputs={};audits={}
            for method in VARIANTS:
                kept,events=suppress(native,view['size'],method);outputs[method]=kept;audits[method]=events
                metrics[method]=evaluate(kept,view['size'],ref)
                save_masks(fdest/method,kept,modified=method=='peel')
                write(fdest/method/'actions.json',events)
            metrics['peel']['instances_fragmented']=sum(e.get('components_after',0)>e.get('components_before',0) for e in audits['peel'])
            save_masks(fdest/'input',native);save_masks(fdest/'baseline',actual)
            write(fdest/'metrics.json',metrics)
            info=dict(group=key,source=str(folder),active_source=str(active),raw_local=len(all_native),metrics=metrics)
            frames.append(info)
            # Render only observations that actually contain overlapping pixels.
            if metrics['input']['overlap_pixels']:
                local_clean=Image.open(active/'input/clean.png').convert('RGB')
                shutil.copy2(active/'input/clean.png',fdest/'clean.png')
                for name,rows in [('input',native),('baseline',actual),('union90',outputs['union90']),('union90_t1',outputs['union90_t1']),('peel',outputs['peel'])]:preview(local_clean,rows,fdest/(name+'.png'))
                text=f'# {cid} / {key}\n\n输入为当前组有归属的局部 mask，过滤与阈值沿用基线。M 编号跨对照保持不变；颜色只供定位。\n\n'
                text+='| 原图 | 基线送入 GLM | 完整实例抑制 union90 | 扣重叠像素 peel |\n|---|---|---|---|\n| ![](clean.png) | ![](baseline.png) | ![](union90.png) | ![](peel.png) |\n\n'
                text+=f'归属筛选后 {len(native)} → 基线 {len(actual)} → union90 {len(outputs["union90"])} → peel {len(outputs["peel"])} 个。\n\n'
                text+='[抑制前全部输入](input.png) · [指标](metrics.json) · [union90 删除记录](union90/actions.json) · [peel 改像素记录](peel/actions.json)。\n'
                text+='\n允许 1 像素边缘误差的 union90_t1（仅匹配容差，不改保留 mask）：\n\n![](union90_t1.png)\n\n[删除记录](union90_t1/actions.json)\n'
                (fdest/'README.md').write_text(text)
    # A separate stress test pools all cached local proposals, including those
    # previously rejected by owner retrieval. This is not the current mainline.
    pdest=dest/'pooled';pdest.mkdir(exist_ok=True);ref=coverage(pool,size)>0
    metrics={'input':evaluate(pool,size,ref)};outputs={}
    for method in VARIANTS:
        kept,events=suppress(pool,size,method);outputs[method]=kept
        metrics[method]=evaluate(kept,size,ref)
        if method=='peel':metrics[method]['instances_fragmented']=sum(e.get('components_after',0)>e.get('components_before',0) for e in events)
        save_masks(pdest/method,kept,modified=True);write(pdest/method/'actions.json',events)
    save_masks(pdest/'input',pool,modified=True)
    write(pdest/'metrics.json',metrics)
    shutil.copy2(case/'clean.png',dest/'clean.png')
    preview(clean,global_rows,dest/'global_baseline.png',False)
    preview(clean,outputs['union90'],dest/'pooled_union90.png',False)
    preview(clean,outputs['union90_t1'],dest/'pooled_union90_t1.png',False)
    overlap_preview(pool,size,dest/'pooled_overlap.png')
    review='# '+cid+'：mask 抑制对照\n\n全图只是候选证据，不是箭头恢复结果。全图不堆叠标签，局部对照有短 M 标签。\n\n'
    review+='| 原图 | 基线全局 mask | 全局＋全部局部候选重叠 | union90 保留的完整实例 |\n|---|---|---|---|\n| ![](clean.png) | ![](global_baseline.png) | ![](pooled_overlap.png) | ![](pooled_union90.png) |\n\n'
    review+='重叠图：蓝灰=1 个实例，橙色=2–3 个，红色=至少 4 个。全图投影只做对照；原始局部 mask 路径保留，用于后续精确几何/样式。\n\n'
    review+='union90_t1：90% 并集覆盖，匹配时允许 1 像素边缘误差（不改保留 mask）。\n\n![](pooled_union90_t1.png)\n\n'
    review+='| 方法 | 候选数 | 未保留的并集像素 | 重复像素总量 |\n|---|---:|---:|---:|\n'
    for m in ('input',*VARIANTS):review+=f'| {m} | {metrics[m]["instances"]} | {metrics[m]["lost_union_pixels"]} | {metrics[m]["duplicate_pixel_mass"]} |\n'
    review+='\n局部对照（仅列有重叠的组）：\n\n'
    for f in frames:
        if f['metrics']['input']['overlap_pixels']:
            mm=f['metrics'];review+=f'- [{f["group"]}](groups/{f["group"]}/README.md)：基线 {mm["baseline"]["instances"]} → union90 {mm["union90"]["instances"]}；peel 使 {mm["peel"]["instances_fragmented"]} 个 mask 连通片增多。\n'
    (dest/'README.md').write_text(review)
    result=dict(case=cid,frames=frames,pooled=metrics,global_masks=len(global_rows),pooled_input=len(pool),
                provenance=dict(baseline=str(case),native_masks_preserved=True),accuracy_measured=False)
    write(dest/'summary.json',result)
    return result


def self_test():
    def r(a,n):return record(a,[0,0,a.shape[1],a.shape[0]],n,score=1.)
    a=np.zeros((20,20),bool);a[8:11,2:18]=1
    same=[r(a,'a'),r(a,'b')]
    assert len(suppress(same,(20,20),'single90')[0])==1
    b=np.zeros_like(a);b[2:18,8:11]=1
    cross=[r(a,'a'),r(b,'b')]
    assert len(suppress(cross,(20,20),'union90')[0])==2
    peeled,_=suppress(cross,(20,20),'peel');cov=coverage(peeled,(20,20))
    assert np.array_equal(cov>0,a|b) and cov.max()==1
    assert any(components(x['ink'])>1 for x in peeled)
    c=np.zeros_like(a);c[8:11,2:7]=1;c[3:7,8:11]=1
    union=suppress([*cross,r(c,'c')],(20,20),'union90')[0]
    single=suppress([*cross,r(c,'c')],(20,20),'single90')[0]
    assert len(union)==2 and len(single)==3
    assert all(np.array_equal(x['ink'],next(z['ink'] for z in [*cross,r(c,'c')] if z['id']==x['id'])) for x in union)
    print('self_test: exact duplicate, crossing, union coverage, intact masks, pixel-union invariants passed',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path);p.add_argument('--out',type=Path);p.add_argument('--workers',type=int,default=8);p.add_argument('--self-test',action='store_true');args=p.parse_args()
    self_test()
    if args.self_test:return
    base=args.baseline.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if not (out/'code').exists():shutil.copytree(base/'code',out/'code')
    shutil.copy2(__file__,out/'code/mask_overlap_trial.py');shutil.copy2(base/'config.json',out/'baseline_config.json')
    cases=sorted(read(base/'finished.json')['cases']);assert len(cases)==40
    write(out/'config.json',dict(baseline=str(base),cases=cases,case_workers=args.workers,variants=VARIANTS,
        priority='descending pixel area, then SAM score, then stable ID',thresholds=[.85,.9,.95],matching_tolerance_pixels=[0,1],
        local_scope='owner-selected masks before baseline containment',pooled_scope='global + all raw-filtered cached local masks',
        sam_thresholds_unchanged=True,new_api_calls=0,production_modified=False,global_projection='nearest-neighbor proxy; preserve native mask evidence'))
    write(out/'code_manifest.json',{str(f.relative_to(out/'code')):digest(f) for f in sorted((out/'code').rglob('*.py'))})
    started=time.monotonic();results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(process_case,(str(base),str(out),c)):c for c in cases}
        for future in as_completed(jobs):
            result=future.result();results.append(result)
            write(out/'progress.json',dict(completed=len(results),total=len(cases),last_case=result['case']))
            print(f'{len(results)}/{len(cases)} {result["case"]}: pool {result["pooled_input"]} -> {result["pooled"]["union90"]["instances"]}',flush=True)
    totals={}
    for scope in ('local','pooled'):
        rows=[f['metrics'] for r in results for f in r['frames']] if scope=='local' else [r['pooled'] for r in results]
        totals[scope]={}
        for method in rows[0]:
            totals[scope][method]={key:sum(r[method].get(key,0) for r in rows) for key in set().union(*(r[method].keys() for r in rows))}
    write(out/'summary.json',dict(cases=40,seconds=time.monotonic()-started,totals=totals,accuracy_measured=False))
    text='# 40 张 mask 重叠处理实验\n\n基线：`all40_corner_scope_20261017_v4`。只改变缓存 mask 的保留/像素归属，不调用模型，不修改基线和主管线。规则用于本轮可逆对照，不作为最终箭头身份判定。\n\n'
    text+='方法：single90=被单个较大实例覆盖至少90%则整张抑制；union90=被已保留实例并集覆盖至少90%则整张抑制；union85/95只改变阈值；peel=按同一优先序扣掉已占像素，保留所有非空残余。完整实例抑制不改保留 mask 的像素。\n\n'
    text+='后缀 `_t1`：仅在匹配时允许 1 像素边缘误差，不膨胀或修改输出 mask。全图合池以原图网格评估，组内以原局部网格评估；两种尺度的统计应分别阅读。\n\n'
    for scope,title in [('local','当前组内：保持原有归属筛选'),('pooled','扩展候选池：全局 + 全部缓存局部 SAM')]:
        text+='## '+title+'\n\n| 方法 | 实例数 | 未保留并集像素 | 重复像素总量 | 连通片增多的实例 |\n|---|---:|---:|---:|---:|\n'
        for method,values in totals[scope].items():text+=f'| {method} | {values["instances"]} | {values["lost_union_pixels"]} | {values["duplicate_pixel_mass"]} | {values.get("instances_fragmented","—")} |\n'
        text+='\n'
    text+='数字仅描述证据冗余/损失，不是箭头准确率。并集覆盖不变也可能损坏实例归属，共享主干尤其如此；全图投影不替代 native mask。没有新增模型规划区域，也没有证明下游恢复已经改善。\n\n'
    text+='## 全部 40 张\n\n| Case | 原图 | 扩展池重叠 | union90 |\n|---|---|---|---|\n'
    for r in sorted(results,key=lambda x:x['case']):
        c=r['case'];text+=f'| [{c}](cases/{c}/README.md) | ![](cases/{c}/clean.png) | ![](cases/{c}/pooled_overlap.png) | ![](cases/{c}/pooled_union90.png) |\n'
    (out/'RESULTS.md').write_text(text)
    write(out/'finished.json',dict(completed=40,seconds=time.monotonic()-started,new_api_calls=0,production_modified=False))
    print('finished '+str(out/'RESULTS.md'),flush=True)


if __name__=='__main__':main()
