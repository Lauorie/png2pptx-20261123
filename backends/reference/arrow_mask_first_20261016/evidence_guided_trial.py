"""Fixed-identity geometry ablation: carry bbox/D, then add mask line points.

Freeze baseline inputs/styles. No SAM calls, no case-specific fitting rules.
"""
import argparse,copy,os,shutil,sys,threading,time,traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
import geometry_measurements as gm
import segmented_geometry as sg
import protocol
from transport.common import read_json as read,write_json as write,sha256 as sha
from transport.model import Model


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--scope',type=Path);p.add_argument('--frozen',action='store_true');a=p.parse_args()
    run=a.run.resolve();run.mkdir(parents=True,exist_ok=True);root=Path(__file__).resolve().parent
    if not a.frozen:
        if (run/'code_manifest.json').exists():raise ValueError('Snapshot exists; resume with --frozen')
        scope=read(a.scope);base=Path(scope['baseline']);shutil.copytree(base/'code',run/'code')
        for name in ['geometry_measurements.py','evidence_guided_trial.py']:shutil.copy2(root/name,run/'code'/name)
        write(run/'scope.json',scope);config=read(base/'config.json');write(run/'config.json',config)
        write(run/'code_manifest.json',{str(f.relative_to(run/'code')):sha(f) for f in sorted((run/'code').rglob('*.py'))})
        os.execv(sys.executable,[sys.executable,'-u',str(run/'code/evidence_guided_trial.py'),'--run',str(run),'--frozen'])
    sys.path.insert(0,str(root/'lib'))
    import viz
    config=read(run/'config.json');scope=read(run/'scope.json');base=Path(scope['baseline'])
    model=Model(config,run_dir=run,system=protocol.SYSTEM);records=[];lock=threading.RLock();render_lock=threading.Semaphore(4)
    for t in scope['groups']:
        for mode in ['bbox','points']:records.append({**t,'mode':mode,'state':'waiting'})
    def report():
        with lock:
            write(run/'progress.json',records)
            text='# 几何数字证据：bbox / 关键点对照\n\n冻结 V4 身份、原图、mask、样式，仅改变几何请求的测量输入。未运行整图补漏。\n\n'
            text+='| 局部 | 条件 | 状态 | 原图 | V4 | 实验结果 |\n|---|---|---|---|---|---|\n'
            for r in records:
                rel=f'cases/{r["case"]}/{r["group"]}/{r["mode"]}';dest=run/rel
                pics=[f'![]({rel}/{x})' if (dest/x).exists() else '' for x in ['input/clean.png','before/overlap.png','after/overlap.png']]
                text+='| '+' | '.join([f'[{r["case"]}/{r["group"]}]({rel}/README.md)',r['mode'],r['state'],*pics])+' |\n'
            (run/'RESULTS.md').write_text(text)
    report();write(run/'process.json',dict(pid=os.getpid(),started=time.time(),case_workers=8,api_concurrency=64,new_sam_calls=0))
    def work(record):
        started=time.monotonic();cid,gid,mode=record['case'],record['group'],record['mode']
        src=base/'cases'/cid/'groups'/gid;previous=src/'geometry/F001';out=run/'cases'/cid/gid/mode;inputs=out/'input';inputs.mkdir(parents=True,exist_ok=True)
        try:
            with lock:record['state']='preparing';report()
            for name,origin in [('clean.png',src/'input/clean.png'),('context.png',src/'input/context.png'),('masks.png',previous/'input/masks.png')]:shutil.copy2(origin,inputs/name)
            shutil.copy2(src/'input/transform.json',inputs/'transform.json')
            metadata=read(previous/'baseline_metadata.json');old={r['id']:r for r in read(src/'rows_local.json')}
            baseline={rid:copy.deepcopy(old[value['id']]) for rid,value in metadata.items()}
            for rid,row in baseline.items():row['label']=rid
            write(out/'baseline.json',baseline)
            with render_lock:viz.render(inputs/'clean.png',list(baseline.values()),out/'before')
            identity=read(src/'prompt.json');size=Image.open(inputs/'clean.png').size
            arrays={}
            for row in baseline.values():
                for mid,path in zip(row['source_masks'],row['source_mask_paths']):arrays[mid]=np.asarray(Image.open(path).convert('L'))>0
            prompt=copy.deepcopy(read(previous/'prompt.json'))
            measurement=gm.evidence(prompt['targets'],metadata,identity['masks'],identity['proposals'],arrays,size,mode=='points')
            prompt['mask_measurements']=measurement
            prompt['instructions'] += [
                'mask_measurements carries numeric evidence from the SAME local CLEAN pixel frame. route_masks maps each R to the original M masks. Bboxes locate ink, not centerlines; D endpoints are optional measurements, not mandatory geometry. Use CLEAN to correct incomplete or mixed evidence.',
            ]
            images=[('CLEAN: local pixel coordinates',inputs/'clean.png'),('MASKS: selected target evidence, R labels',inputs/'masks.png'),('CONTEXT: whole figure',inputs/'context.png')]
            if mode=='points':
                prompt['instructions'] += ['L line_fragments are measured straight ink fragments, NOT arrows, topology, complete endpoints or ground truth. You may reuse their coordinate/axis information, join collinear fragments through dashed gaps, or reject them. Do not turn a real curve into straight segments merely because it has L observations. Choose shared point IDs yourself at real junctions; crossings alone are not junctions.']
                gm.draw(inputs/'clean.png',measurement['line_fragments'],inputs/'measurements.png')
                images.append(('L MEASUREMENTS: optional mask line fragments, never ground truth',inputs/'measurements.png'))
            write(out/'prompt.json',prompt);write(inputs/'manifest.json',[dict(label=l,path=str(p),sha256=sha(p)) for l,p in images])
            before_answer=read(previous/'answer.json');write(out/'before_answer.json',before_answer)
            readme=f'# {cid}/{gid}：{mode}\n\n固定身份与样式；不改变图像坐标。bbox 模式补回所选 mask 的框与原有 D；points 额外给自动提取的直段坐标。算法不决定最终点位，不设置几何门控。\n\n[实际 prompt](prompt.json) · [请求 trace](request/) · [旧节点](before_answer.json)\n\n'
            readme+='| 原图 | mask | 全局上下文 |\n|---|---|---|\n| ![](input/clean.png) | ![](input/masks.png) | ![](input/context.png) |\n'
            if mode=='points':readme+='\n自动测量的直段候选：\n\n![](input/measurements.png)\n'
            (out/'README.md').write_text(readme)
            with lock:record['state']='running';report()
            answer=model.ask(out/'request',prompt,images,lambda value:sg.validate(value,baseline,size));write(out/'answer.json',answer)
            rows=sg.geometry(answer,baseline);write(out/'rows_local.json',rows)
            with render_lock:viz.render(inputs/'clean.png',rows,out/'after');sg.draw_points(answer,inputs/'clean.png',out/'after/points.png')
            assert all(row['style']==baseline[rid]['style'] for rid,row in zip(baseline,rows))
            with (out/'README.md').open('a') as f:f.write('\n[新几何](answer.json) · [逐条结果](rows_local.json)\n\n| V4 | 实验结果 | 模型点位 |\n|---|---|---|\n| ![](before/overlap.png) | ![](after/overlap.png) | ![](after/points.png) |\n')
            record.update(state='completed',routes=len(rows),points=len(answer['nodes']),line_hints=len(measurement.get('line_fragments',[])))
        except Exception as exc:
            record.update(state='failed',error=str(exc));write(out/'error.json',dict(error=str(exc),traceback=traceback.format_exc()))
        record['seconds']=round(time.monotonic()-started,2);write(out/'completed.json',record)
        with lock:report()
        print(cid,gid,mode,record['state'],record['seconds'],flush=True)
    def case_work(cid):
        tasks=[r for r in records if r['case']==cid]
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(work,tasks))
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(case_work,sorted({r['case'] for r in records})))
    write(run/'finished.json',dict(records=records,new_sam_calls=0,production_modified=False))


if __name__=='__main__':main()
