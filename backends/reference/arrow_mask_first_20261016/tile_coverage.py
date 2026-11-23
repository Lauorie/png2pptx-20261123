"""Parallel regional omission locators. Regions never define arrow identities."""
import argparse,os,threading,time,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from PIL import Image,ImageDraw
import evidence as ev,protocol,shared_repair,visual
from transport.common import read_json as read,write_json as write,sha256 as sha


def windows(size,columns=3,rows=2,overlap=.12):
    w,h=size;dx,dy=w/columns,h/rows
    return [[max(0,int(np.floor((x-overlap/2)*dx))),max(0,int(np.floor((y-overlap/2)*dy))),
             min(w,int(np.ceil((x+1+overlap/2)*dx))),min(h,int(np.ceil((y+1+overlap/2)*dy)))]
            for y in range(rows) for x in range(columns)]


def discover(clean,current,out,model,config,locks):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);bounds=windows(clean.size);records={}
    def one(i,box):
        folder=out/f'T{i+1}';inputs=folder/'input'
        with locks['cpu']:
            crop,view=ev.frame(clean,[],inputs,config,box=box)
            shared_repair.draw_current(crop,current,view,inputs/'current.png')
        q=dict(task='Locate potentially missing in-scope lines in this observation window. Return small review regions only.',
            size=list(crop.size),instructions=[protocol.SCOPE,protocol.OCCLUSION,
                'ORIGINAL is the reference. CURRENT shows recovered paths in deep blue over a faded gray source, including known heads, dashes and filled contours. Thin shafts are thickened only for visibility; gray source ink is not reconstruction. Blue results are not ground truth. Look for original strokes or branches absent from CURRENT; boxes or a nearby path do not imply completeness.',
                'Return tight bbox regions around suspect missing strokes and their visible attachments, including short dimension/auxiliary strokes. A region may be rejected after local review. Do not estimate arrow counts, topology or geometry, or explain the figure architecture. Descriptions only locate visible marks.',
                'Inspect visible edge fragments too: the later local recovery can request more context. Ignore minor style differences. Return an empty regions list when nothing is visibly missing.'],
            output_schema=dict(regions='[{bbox:[left,top,right,bottom],description:short visual locator}]'))
        images=[('ORIGINAL: intact local source',inputs/'clean.png'),('CURRENT: actual paths on the same local source',inputs/'current.png')]
        write(folder/'prompt.json',q);write(inputs/'images.json',[dict(label=n,path=str(p),sha256=sha(p)) for n,p in images])
        answer=model.ask(folder/'request',q,images,lambda a:protocol.regions(a,crop.size));write(folder/'answer.json',answer)
        regions=[];marked=crop.copy();draw=ImageDraw.Draw(marked)
        for j,r in enumerate(answer['regions'],1):
            b=r['bbox'];m=np.asarray(view['view_to_global']);p=np.array([[b[0],b[1],1],[b[2],b[3],1]])@m.T
            g=np.clip(p[:,:2],0,np.asarray(clean.size)).reshape(-1).tolist()
            if g[0]>=g[2] or g[1]>=g[3]:raise ValueError('Degenerate reprojected region')
            regions.append(dict(bbox=g,description=r['description'],observation=f'T{i+1}'))
            draw.rectangle(b,outline=visual.color(j),width=2);visual.label(draw,b[:2],str(j),visual.color(j),marked)
        marked.save(folder/'regions.png');return dict(tile=f'T{i+1}',bbox=box,regions=regions,state='completed')
    with ThreadPoolExecutor(max_workers=min(config['task_workers'],len(bounds))) as pool:
        jobs={pool.submit(one,i,b):i for i,b in enumerate(bounds)}
        for future in as_completed(jobs):
            i=jobs[future]
            try:records[i]=future.result()
            except Exception as exc:records[i]=dict(tile=f'T{i+1}',bbox=bounds[i],regions=[],state='failed',error=str(exc));write(out/f'T{i+1}'/'error.json',dict(**records[i],traceback=traceback.format_exc()))
            write(out/'progress.json',dict(total=len(bounds),finished=len(records),records=list(records.values())))
    regions=[r for i in sorted(records) for r in records[i]['regions']];errors=[r for r in records.values() if r['state']=='failed']
    result=dict(regions=regions,errors=errors,state='failed' if len(errors)==len(bounds) else 'partial' if errors else 'completed')
    write(out/'result.json',result)
    (out/'README.md').write_text('# 六区补漏候选\n\n只定位区域，不预设实例数量或连接；重叠区域由主线统一局部核查。\n\n'+
        '| 区域 | 原图 | 已恢复路径 | 提议区域 |\n|---|---|---|---|\n'+
        '\n'.join(f'| [T{i+1}](T{i+1}/prompt.json) | ![](T{i+1}/input/clean.png) | ![](T{i+1}/input/current.png) | ![](T{i+1}/regions.png) |' for i in range(len(bounds)))+'\n')
    return result


if __name__=='__main__':
    from transport.model import Model
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--baseline',required=True);p.add_argument('--cases',nargs='+',required=True)
    a=p.parse_args();r=Path(a.run).resolve();base=Path(a.baseline).resolve();cfg=read(r/'config.json');model=Model(cfg,run_dir=r,system=protocol.SYSTEM)
    locks=dict(cpu=threading.Semaphore(cfg['cpu_workers']));out=r/'tile_coverage';out.mkdir(exist_ok=True)
    write(out/'process.json',dict(pid=os.getpid(),started=time.time()))
    def one(cid):
        scene=read(base/'cases'/cid/'scene.json');return cid,discover(Image.open(base/'cases'/cid/'clean.png').convert('RGB'),scene['arrows'],out/cid,model,cfg,locks)
    records={}
    with ThreadPoolExecutor(max_workers=min(8,len(a.cases))) as pool:
        jobs=[pool.submit(one,cid) for cid in a.cases]
        for f in as_completed(jobs):
            cid,result=f.result();records[cid]=result;write(out/'progress.json',records);print(cid,result['state'],len(result['regions']),flush=True)
    write(out/'finished.json',records)
