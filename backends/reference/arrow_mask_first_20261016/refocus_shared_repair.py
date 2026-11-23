"""Closer recovery windows from previous model-nominated missing regions.

The earlier geometry supplies only a locator bbox, not points or branch counts.
Uses the same production shared_repair entry; no case-specific recovery rules.
"""
import argparse,copy,json,os,shutil,sys,threading,time,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import evidence as ev,shared_repair,pipeline,protocol
from lib import fit
from transport.common import read_json as read,write_json as write,sha256 as sha
from transport.model import Model


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--scope',type=Path);p.add_argument('--frozen',action='store_true');a=p.parse_args()
    run=a.run.resolve();run.mkdir(parents=True,exist_ok=True);root=Path(__file__).resolve().parent
    if not a.frozen:
        if (run/'code_manifest.json').exists():raise ValueError('Refocus snapshot already exists')
        write(run/'scope.json',read(a.scope));config=read(root/'config.json');config['case_workers']=3;write(run/'config.json',config)
        files={}
        for src in [*root.glob('*.py'),*(root/'lib').glob('*.py'),*(root/'transport').glob('*.py')]:
            rel=src.relative_to(root);dest=run/'code'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);files[str(rel)]=sha(dest)
        write(run/'code_manifest.json',dict(files=files,config_sha256=sha(run/'config.json')))
        os.execv(sys.executable,[sys.executable,'-u',str(run/'code'/Path(__file__).name),'--run',str(run),'--frozen'])
    scope=read(run/'scope.json');config=read(run/'config.json');source=Path(scope['source'])
    model=Model(config,run_dir=run,system=protocol.SYSTEM);locks={k:threading.Semaphore(config[k+'_workers']) for k in ['cpu','sam']};mutex=threading.RLock()
    records=[{**t,'state':'waiting'} for t in scope['tasks']]
    write(run/'process.json',dict(pid=os.getpid(),started=time.time(),case_workers=3,api_concurrency=64))
    def report():
        with mutex:
            write(run/'progress.json',records)
            lines=['# 按此前模型提出的缺段范围重新取景','','只用之前模型提案的 bbox 定位，不把它的坐标、支路数量或拟合结果传给 GLM。重新 crop 3×、SAM；共享原有实际路径。','',
                   '| case | 状态 | 原图 | 原有结果 | 当前实例 | 当前叠图 |','|---|---|---|---|---|---|']
            for r in records:
                rel='cases/'+r['case'];folder=run/rel
                images=[f'![]({rel}/{n})' if (folder/n).exists() else '' for n in ['clean.png','before/overlap.png','result/instances.png','result/overlap.png']]
                lines.append('| '+' | '.join([f'[{r["case"]}]({rel}/continuation/README.md)',r['state'],*images])+' |')
            (run/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    report()
    def work(r):
        out=run/'cases'/r['case'];out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
        try:
            case=source/'cases'/r['case'];clean=Image.open(case/'clean.png').convert('RGB');shutil.copy2(case/'clean.png',out/'clean.png');shutil.copytree(case/'result',out/'before',dirs_exist_ok=True)
            suggested=read(Path(r['suggestion']))['added'];p=np.concatenate([fit.points(x) for x in suggested]);box=np.r_[p.min(0),p.max(0)].tolist()
            task=dict(bbox=box,description='Complete the related connector/network around this candidate missing region. The locator came from an earlier fallible suggestion; independently determine the visible missing branches and connections, preserving current paths.')
            write(out/'locator_provenance.json',dict(source=r['suggestion'],sha256=sha(Path(r['suggestion'])),bbox_global=box,prior_geometry_sent=False))
            current=read(case/'scene.json')['arrows'];r['state']='repair';report()
            after,patch=shared_repair.recover(clean,task,current,'P1',out/'continuation',config,model,locks)
            result=shared_repair.merge_results([dict(group='preserved',state='completed',rows=current)],patch)
            with locks['cpu']:pipeline.publish(clean,result,out)
            write(out/'results.json',result);r.update(state=patch['state'],decision=patch['status'],added=len(patch['rows']),replaced=len(patch['replace']))
        except Exception as exc:r.update(state='failed',error=str(exc));write(out/'error.json',dict(error=str(exc),traceback=traceback.format_exc()))
        r['seconds']=round(time.monotonic()-started,1);write(out/'completed.json',r);report();print(json.dumps(r,ensure_ascii=False),flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(work,records))
    write(run/'finished.json',dict(cases=records))


if __name__=='__main__':main()
