"""Same-input scope wording ablation; identity only, no geometry/style changes."""
import argparse,os,shutil,sys,threading,time,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from transport.common import read_json as read,write_json as write,sha256 as sha
from transport.model import Model
import semantic_geometry,protocol

SCOPE=('Include object-to-object connectors, standalone separators and auxiliary lines, and real arrows indicating direction, displacement, sampling, or a legend example. '
       'An arrow is not excluded merely because it is inside an illustrated feature map, legend, panel or module. '
       'Exclude shape/frame edges, text, mathematical operator strokes, chart axes/baselines/grids (including headed chart axes), and headless dense wiring internal to neural-network icons. '
       'Use CLEAN and whole-figure context to distinguish these roles; SAM candidates are fallible evidence, not proof of eligibility.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--scope',type=Path);p.add_argument('--frozen',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parent;out=a.run.resolve();out.mkdir(parents=True,exist_ok=True)
    if not a.frozen:
        if (out/'code_manifest.json').exists():raise ValueError('Snapshot exists')
        scope=read(a.scope);base=Path(scope['baseline']);shutil.copytree(base/'code',out/'code');shutil.copy2(__file__,out/'code/scope_clarity_trial.py')
        write(out/'scope.json',scope);shutil.copy2(base/'config.json',out/'config.json')
        write(out/'code_manifest.json',{str(f.relative_to(out/'code')):sha(f) for f in sorted((out/'code').rglob('*.py'))})
        os.execv(sys.executable,[sys.executable,'-u',str(out/'code/scope_clarity_trial.py'),'--run',str(out),'--frozen'])
    scope=read(out/'scope.json');cfg=read(out/'config.json');base=Path(scope['baseline']);model=Model(cfg,run_dir=out,system=protocol.SYSTEM)
    rows=[{**x,'state':'waiting'} for x in scope['groups']];lock=threading.RLock()
    def report():
        with lock:
            write(out/'progress.json',rows)
            text='# 范围排除措辞：同输入对照\n\n仅替换原身份请求的范围条款；其余任务、候选、图片和推理配置保持一致。未拟合几何。\n\n| 局部 | 状态 | 旧 used / excluded | 新 used / excluded | 原图 |\n|---|---|---|---|---|\n'
            for r in rows:
                rel=f'cases/{r["case"]}/{r["group"]}';text+=f'| [{r["case"]}/{r["group"]}]({rel}/README.md) | {r["state"]} | {r.get("before", "")} | {r.get("after", "")} | ![]({rel}/input/clean.png) |\n'
            (out/'RESULTS.md').write_text(text)
    def counts(d):return dict(used=sorted({g for r in d['routes'] for g in r['sources']}),excluded=d['excluded'],uncertain=d['uncertain'])
    def work(r):
        t=time.monotonic();source=base/'cases'/r['case']/'groups'/r['group'];dest=out/'cases'/r['case']/r['group'];inp=dest/'input';inp.mkdir(parents=True,exist_ok=True)
        try:
            prompt=read(source/'prompt.json');old_prompt=read(source/'prompt.json');prompt['instructions'][0]=SCOPE
            images=[]
            for item in read(source/'input/images.json'):
                src=Path(item['path']);target=inp/src.name;shutil.copy2(src,target);assert sha(src)==sha(target);images.append((item['label'],target))
            r['before']=counts(read(source/'decision.json'));r['state']='running';report()
            write(dest/'before_prompt.json',old_prompt);write(dest/'prompt.json',prompt)
            (dest/'README.md').write_text('# '+r['case']+'/'+r['group']+'\n\n[旧 prompt](before_prompt.json) · [新 prompt](prompt.json) · [完整请求](request/)\n\n| 原图 | mask |\n|---|---|\n| ![](input/clean.png) | ![](input/masks.png) |\n')
            answer=model.ask(dest/'request',prompt,images,lambda v:semantic_geometry.validate(v,prompt['size'],prompt['sources'],read(source/'proposals.json'),[m['id'] for m in prompt['masks']]))
            write(dest/'answer.json',answer);r.update(state='completed',after=counts(answer))
        except Exception as exc:r.update(state='failed',error=str(exc));write(dest/'error.json',dict(error=str(exc),traceback=traceback.format_exc()))
        r['seconds']=round(time.monotonic()-t,2);report();print(r['case'],r['group'],r['state'],r['seconds'],flush=True)
    report()
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(work,rows))
    write(out/'finished.json',dict(records=rows,new_sam_calls=0,geometry_changed=False,production_modified=False))


if __name__=='__main__':main()
