"""One prompt-text ablation over frozen style batches from four figures."""
import argparse,fcntl,hashlib,json,os,shutil,sys,threading,time,traceback
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor,as_completed

LOCK=threading.Lock()
def now():return datetime.now().astimezone().isoformat(timespec='seconds')
def read(p,default=None):return json.loads(Path(p).read_text()) if Path(p).exists() else default
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.'+p.name+'.tmp')
    tmp.write_text(x if isinstance(x,str) else json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def report(root):
    with LOCK:
        body=[];records=[]
        for j in read(root/'jobs.json'):
            d=root/'batches'/j['id'];s=read(d/'status.json',dict(state='queued'))
            records.append(dict(job=j['id'],case=j['case'],**s))
            def img(path):
                f=d/path
                if not f.exists():return '等待'
                asset=root/'reports/assets'/f'{j["id"]}_{f.parent.name}_{f.name}'
                asset.parent.mkdir(parents=True,exist_ok=True)
                if not asset.exists():shutil.copy2(f,asset)
                return f'[![](assets/{asset.name})](assets/{asset.name})'
            body += [f'## {j["case"]} / {j["role"]}', '',f'{s["state"]}；目标：'+', '.join(j['targets']),'',
                f'[原始trace]({j["source"]}) · [新trace](../batches/{j["id"]}/style/) · [端点变化](../batches/{j["id"]}/head_changes.json)','',
                '| 模型实际CLEAN输入 | 新样式实际PPT |','|---|---|',f'| {img("input/clean.png")} | {img("style/final/render.png")} |','',
                '| 旧样式实际PPT | 模型实际定位输入 |','|---|---|',f'| {img("input/before.png")} | {img("style/input/locators.png")} |','']
        ended=sum(r['state'] in ('completed','completed_with_issues','failed') for r in records)
        header=['# 圆头与图节点：样式说明单变量实验','',f'更新 {now()}；结束 {ended}/{len(records)} 批，来自4张图。','',
            '只替换样式prompt中端点类别说明。明确：独立小圆端点标记即使贴着大节点边界也应保留；大节点本身、圆线帽、交叉不自动算头。几何、crop、输入点、样式schema、mid、1px下限和审核预算不变。','',
            '复用v036样式前几何与同一CLEAN像素，新跑估计和critic，最多refine一次；7批并发，API上限128，默认思考。旧结果作历史对照，单次差异不能单独证明因果或泛化。执行结束/accept不等于视觉正确。','',
            '[配置](../config.json) · [说明来源与原始问题](../PROVENANCE.md) · [变更](../prompt_change.json)','']
        write(root/'progress.json',dict(updated_at=now(),ended=ended,total=len(records),batches=records))
        write(root/'reports/四图_端点说明对照.md','\n'.join(header+body)+'\n')
        write(root/'RESULTS.md','\n'.join(header)+'\n[四图逐批结果](reports/四图_端点说明对照.md)\n')

def prepare(root,parent):
    work=Path(__file__).resolve().parent.parent;source=parent/'runs/v036_style_heads24/receiver_v2'
    if root.exists():raise ValueError('Run already exists')
    root.mkdir(parents=True)
    for name in ('code','geometry_code'):shutil.copytree(work/name,root/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache'))
    shutil.copytree(source/'constraints_snapshot',root/'constraints_snapshot')
    cfg=read(source/'config.json');cfg.update(api_concurrency=128,case_workers=4,task_workers=7,cpu_workers=4,model_cache_from=None,style_reuse_from=None,run_label=root.name)
    assert 'reasoning_effort' not in cfg and cfg['thinking']=={'type':'enabled'}
    write(root/'config.json',cfg)
    specs=[('task_0077','faec39c21199','圆头被误删'),('task_0077','b2c2cf060b6e','已有圆头对照'),('task_0077','134baa8b4c88','邻近节点的普通头'),
           ('task_0028','07aff40611c0','普通直箭头'),('task_0056','0ea710465b61','串联线段有头与无头'),
           ('task_0122','16ae635eeaa8','无头网络对照1'),('task_0122','ae3db219edb6','无头网络对照2')]
    jobs=[]
    for cid,batch,role in specs:
        old=source/'cases'/cid/'styles'/batch;key=cid+'_'+batch;d=root/'batches'/key/'input';d.mkdir(parents=True)
        for src,name in [('input/clean.png','clean.png'),('input/frame.json','frame.json'),('input_rows.json','rows.json'),('rows.json','before_rows.json'),('final/render.png','before.png')]:
            shutil.copy2(old/src,d/name)
        rows=read(d/'rows.json');jobs.append(dict(id=key,case=cid,role=role,targets=[r['id'] for r in rows],source=str(old),source_clean_sha256=sha(old/'input/clean.png')))
    write(root/'jobs.json',jobs)
    sys.path[:0]=[str(root/'geometry_code'),str(root/'geometry_code/lib')]
    from simple_style_contract import Contract
    j=jobs[0];d=root/'batches'/j['id']/'input';frame=read(d/'frame.json');rows=read(d/'rows.json')
    old=read(Path(j['source'])/'rev_00/estimate/prompt.json')['prompt']
    new=Contract(frame['view']).prompt(Contract(frame['view']).enforce_rows(rows),frame['view']['size'])
    changed=[k for k in old if old[k]!=new[k]]
    assert changed==['instructions'],changed
    differences=[i for i,(a,b) in enumerate(zip(old['instructions'],new['instructions'])) if a!=b]
    assert differences==[1],differences
    write(root/'prompt_change.json',dict(changed_fields=changed,changed_instruction_indices=differences,old=old['instructions'][1],new=new['instructions'][1],schema_unchanged=True))
    write(root/'PROVENANCE.md','''# 来源与问题边界

“Do not mistake a nearby node, a crossing or a round line cap for a head”这句在2026-10-29接入v036三类端点样式协议时加入；simple_style_contract.py文件时间15:32:08，首轮v036于15:48:01启动。v035尚无该文件。更早的v003几何协议已有“A round node or a round line cap is not automatically an endpoint mark”。这些是防止把图节点误识别为圆头的实现说明，不是用户要求排除贴着节点的端点圆头。

0077/R0020：几何初次oval/none，几何critic保留，去重后保留；样式输入circle/none，初次估计改none/none。首轮样式critic明确认为起点圆点是graph node，要求只修虚线；第二轮接受none/none。CLEAN中小圆点清晰可见，最终PPT XML的headEnd/tailEnd都是none，未发生导出丢头。可以确认样式误判，不能仅凭本例证明prompt是唯一原因。

本轮冻结同一crop、局部几何、坐标变换与当前样式。只改端点说明一条；普通线条几何表达、几何critic、块状箭头、样式接收和渲染代码均未改。0077含圆头正例、三角头对照；0028普通箭头、0056串联线段、0122无头网络用于观察副作用。不会自动将本轮结果合入或覆盖旧24图。
''')
    write(root/'manifest.json',dict(files={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()}));report(root)

def run(root):
    lock=(root/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for rel,digest in read(root/'manifest.json')['files'].items():
        assert sha(root/rel)==digest,rel
    sys.path[:0]=[str(root/'geometry_code'),str(root/'geometry_code/lib')]
    from PIL import Image
    import style_pipeline,style_contract
    from native_runtime import validated_renderer
    from transport.model import Model
    cfg=read(root/'config.json');renderer,version=validated_renderer();write(root/'renderer.json',dict(path=str(renderer),version=version))
    model=Model(cfg,run_dir=root,system='Use CLEAN as source evidence. Recover styles only; verify the physical endpoint categories. Return the requested JSON.')
    locks={'cpu':threading.Semaphore(cfg['cpu_workers'])}
    write(root/'execution.json',dict(state='running',pid=os.getpid(),started_at=now()))
    def work(j):
        d=root/'batches'/j['id'];start=time.monotonic();write(d/'status.json',dict(state='running',started_at=now()));report(root)
        try:
            crop=Image.open(d/'input/clean.png').convert('RGB');rows=read(d/'input/rows.json');frame=read(d/'input/frame.json')
            final,issues=style_pipeline.recover(crop,rows,{},d/'style',cfg,model,locks,frame['view'],frame['source_size'])
            assert style_contract.geometry_signature(rows)==style_contract.geometry_signature(final)
            assert sha(d/'style/input/clean.png')==j['source_clean_sha256']
            before={r['id']:r for r in read(d/'input/before_rows.json')};initial={r['id']:r for r in rows}
            changes=[dict(id=r['id'],geometry_heads=initial[r['id']]['heads'],old_style_heads=before[r['id']]['heads'],new_style_heads=r['heads'],state=r['style'].get('state')) for r in final]
            write(d/'head_changes.json',changes);write(d/'status.json',dict(state='completed_with_issues' if issues else 'completed',issues=issues,seconds=round(time.monotonic()-start,1),geometry_unchanged=True,clean_unchanged=True))
        except Exception as exc:write(d/'status.json',dict(state='failed',error=str(exc),traceback=traceback.format_exc()))
        report(root);print(j['id'],read(d/'status.json')['state'],flush=True)
    with ThreadPoolExecutor(max_workers=cfg['task_workers']) as pool:
        for f in as_completed([pool.submit(work,j) for j in read(root/'jobs.json')]):f.result()
    write(root/'execution.json',dict(state='finished',finished_at=now(),pid=os.getpid()));report(root)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--prepare',type=Path);a=p.parse_args()
    prepare(a.run.resolve(),a.prepare.resolve()) if a.prepare else run(a.run.resolve())
