"""Targeted block-entry regression; untouched rows are copied byte-for-byte."""
import argparse,copy,fcntl,hashlib,json,math,os,shutil,sys,threading,time,traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime
from pathlib import Path

LOCK=threading.Lock()
def now(): return datetime.now().astimezone().isoformat(timespec='seconds')
def read(p,default=None): return json.loads(Path(p).read_text()) if Path(p).exists() else default
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.'+p.name+'.tmp')
    tmp.write_text(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def report(root):
    with LOCK:
        body=[];records=[]
        for j in read(root/'targets.json'):
            cid=j['case'];d=root/'cases'/cid;s=read(d/'status.json',{'state':'queued','stage':'queued'})
            records.append(dict(case=cid,target=j['target'],**s))
            def image(rel):
                f=d/rel
                if not f.exists():return '等待'
                asset=root/'reports/assets'/f'{cid}_{f.parent.name}_{f.stem}_{f.stat().st_mtime_ns}.png';asset.parent.mkdir(parents=True,exist_ok=True)
                if not asset.exists():shutil.copy2(f,asset)
                return f'[![](assets/{asset.name})](assets/{asset.name})'
            body += [f'## {cid} / {j["target"]}', '',f'{s["state"]} · {s["stage"]}','',
                f'[trace](../cases/{cid}/geometry/) · [样式trace](../cases/{cid}/styles/) · [类型路由](../cases/{cid}/geometry/dispatch.json) · [局部PPT](../cases/{cid}/final/target.pptx) · [整图PPT](../cases/{cid}/full/arrows.pptx)','',
                '| 原图局部 | 修复后局部（实际PPT） |','|---|---|',f'| {image("input/focus.png")} | {image("final/focus.png")} |','',
                '| 修复前局部 | 新几何（样式前） |','|---|---|',f'| {image("before/focus.png")} | {image("geometry_preview/focus.png")} |','',
                '| 原图 | 整图叠加（只替换本目标） |','|---|---|',f'| {image("input/source.png")} | {image("full/overlap.png")} |','']
        ended=sum(s['state'] in ('completed','completed_with_issues','failed') for s in records)
        head=['# 定位块状类型直达shape：四目标验证','',f'更新 {now()}；结束 {ended}/{len(records)}。','',
            '只修改块状箭头的入口。普通线条提示词、几何表达、critic、样式协议全部不改。定位和其余几何/样式复用；四目标重新生成shape、语义审核、恢复填充/描边及样式审核。不是四张图的全部目标重跑。','',
            '4目标并发，API上限4，默认思考；保留现有crop SAM证据/轮廓兜底。每个几何审核和样式审核最多refine一次。保留原图、实际模型输入/输出、坐标变换与原生PPT渲染。','',
            '原图不覆盖白底，叠加使用实际PPT透明前景；执行成功/accept不等于视觉正确。','',
            '[配置](../config.json) · [来源](../provenance.json) · [冻结哈希](../manifest.json) · [本轮状态](../progress.json)','']
        write(root/'progress.json',dict(updated_at=now(),total=len(records),ended=ended,cases=records))
        write(root/'reports/块状箭头_四图对照.md','\n'.join(head+body)+'\n')
        write(root/'RESULTS.md','\n'.join(head)+'\n[四图对照](reports/块状箭头_四图对照.md)\n')

def prepare(root,parent):
    work=Path(__file__).resolve().parent.parent
    if root.exists():raise ValueError('Run exists')
    root.mkdir(parents=True)
    for name in ('code','geometry_code'):
        shutil.copytree(work/name,root/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache'))
    shutil.copytree(parent/'runs/v035_block_shape_semantic_rerun4/constraints_snapshot',root/'constraints_snapshot')
    style=parent/'runs/v036_style_heads24/receiver_v2';source=parent/'runs/v032_geometry_dedup24'
    cfg=read(style/'config.json');cfg.update(case_workers=4,task_workers=1,api_concurrency=4,cpu_workers=2,sam_workers=1,
        block_shape_review_enabled=True,block_shape_max_refinements=1,model_cache_from=None,style_reuse_from=None,run_label=root.name)
    cfg.pop('reasoning_effort',None);assert cfg['thinking']=={'type':'enabled'}
    write(root/'config.json',cfg)
    sys.path[:0]=[str(root/'geometry_code'),str(root/'geometry_code/lib')]
    from PIL import Image
    from simple_style_contract import merge_style
    jobs=[];origins=[]
    for cid,tid in [('task_0024','R0001'),('task_0028','R0023'),('task_0104','R0002'),('task_0639','R0002')]:
        old=source/'cases'/cid;sd=style/'cases'/cid;d=root/'cases'/cid/'input';d.mkdir(parents=True)
        obj=next(x for x in read(old/'geometry_plan.json')['tasks'] if x['id']==tid)
        assert obj['render_type']=='block_arrow'
        write(d/'target.json',obj);shutil.copy2(old/'clean.png',d/'source.png')
        if (sd/'rows.json').exists():rows=read(sd/'rows.json');origin=sd/'rows.json'
        else:
            origin=sd/'partial_styles.json';updates=read(origin,{})
            rows=[merge_style(r,updates.get(r['id'],r)) for g in read(sd/'input/geometry_results.json') for r in g['rows']]
        write(d/'baseline_rows.json',rows)
        selected=[r for r in rows if r.get('source_discovery_id',r.get('parent_id',r['id']))==tid]
        assert selected;write(d/'baseline_target.json',selected)
        im=Image.open(d/'source.png');b=obj['bbox_native'];pad=max(8,.1*max(b[2]-b[0],b[3]-b[1]))
        focus=[max(0,math.floor(b[0]-pad)),max(0,math.floor(b[1]-pad)),min(im.width,math.ceil(b[2]+pad)),min(im.height,math.ceil(b[3]+pad))]
        im.crop(focus).save(d/'focus.png');jobs.append(dict(case=cid,target=tid,focus_box=focus))
        origins.append(dict(case=cid,target=tid,plan=str(old/'geometry_plan.json'),plan_sha256=sha(old/'geometry_plan.json'),baseline=str(origin),baseline_snapshot_sha256=sha(d/'baseline_rows.json')))
    write(root/'targets.json',jobs);write(root/'provenance.json',dict(source_code=str(parent/'working/style_heads24/geometry_code'),inputs=origins,
        ordinary_line_code_and_prompts_unchanged=True,new_discovery_calls=False,new_block_geometry_calls=True,new_style_calls=True,new_crop_sam_calls=True,whole_cases_rerun=False))
    write(root/'constraints_snapshot/block_only_user_20261029.md','只修定位已标记block_arrow的入口；普通线条几何表达、提示词和审核不修改。task0122加头问题不在本轮范围。\n')
    write(root/'manifest.json',dict(files={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()}))
    report(root)

def run(root):
    lock=(root/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for rel,digest in read(root/'manifest.json')['files'].items():
        if sha(root/rel)!=digest:raise ValueError('Frozen file changed: '+rel)
    sys.path[:0]=[str(root/'geometry_code'),str(root/'geometry_code/lib')]
    from PIL import Image
    import bbox_geometry,style_pipeline,protocol
    from lib import native
    from native_foreground import render_layer
    from native_runtime import validated_renderer
    from transport.model import Model
    renderer,version=validated_renderer();write(root/'renderer.json',dict(executable=str(renderer),version=version))
    cfg=read(root/'config.json');model=Model(cfg,run_dir=root,system=protocol.SYSTEM)
    locks={'cpu':threading.Semaphore(cfg['cpu_workers']),'sam':threading.Semaphore(cfg['sam_workers'])}
    write(root/'execution.json',dict(state='running',started_at=now(),pid=os.getpid()))
    def work(job):
        cid=job['case'];d=root/'cases'/cid;source=Image.open(d/'input/source.png').convert('RGB');start=time.monotonic()
        def status(stage,state='running',**extra):write(d/'status.json',dict(stage=stage,state=state,seconds=round(time.monotonic()-start,1),**extra));report(root)
        def render(rows,dest,name='target'):
            dest.mkdir(parents=True,exist_ok=True)
            with locks['cpu']:
                native.write(source.size,rows,[r['style'] for r in rows],dest/(name+'.pptx'),head_floor_tolerance=.02)
                native.render(dest/(name+'.pptx'),dest/'native',source.size)
                layer=render_layer(dest/'native/styled.svg',source.size,dest/'native')
                overlay=source.convert('RGBA');overlay.alpha_composite(layer);overlay.convert('RGB').save(dest/'overlap.png')
                Image.open(dest/'native/styled.png').convert('RGB').crop(job['focus_box']).save(dest/'focus.png')
            write(dest/'rows.json',rows)
        try:
            status('block_geometry_and_review')
            result=bbox_geometry.recover_target(source,read(d/'input/target.json'),d/'geometry',model,cfg,locks)
            assert result['rows'] and all(r['kind']=='block_arrow' and r.get('shape_parts') for r in result['rows'])
            status('geometry_preview');render(result['rows'],d/'geometry_preview');report(root)
            status('shape_style')
            groups,style_issues=style_pipeline.finish_case(source,[result],d,cfg,model,locks)
            final=[r for g in groups for r in g['rows']];write(d/'final_rows.json',final)
            status('native_render');render(final,d/'final');render(read(d/'input/baseline_target.json'),d/'before')
            original=read(d/'input/baseline_rows.json');old_ids={r['id'] for r in read(d/'input/baseline_target.json')}
            unchanged=[r for r in original if r['id'] not in old_ids];updated=unchanged+final
            assert [r for r in updated if r['id'] not in {x['id'] for x in final}]==unchanged
            render(updated,d/'full','arrows')
            issues=result['issues']+style_issues
            write(d/'completed.json',dict(issues=issues,unrelated_rows_unchanged=True,ordinary_line_requests=0,
                result_kinds=[r['kind'] for r in final],styles=[r['style'] for r in final],visual_accuracy_measured=False))
            status('finished','completed_with_issues' if issues else 'completed',issues=len(issues))
        except Exception as exc:
            write(d/'fatal.json',dict(error=str(exc),traceback=traceback.format_exc()));status('failed','failed',error=str(exc))
        print(cid,read(d/'status.json'),flush=True)
    with ThreadPoolExecutor(max_workers=cfg['case_workers']) as pool:
        for future in as_completed([pool.submit(work,j) for j in read(root/'targets.json')]):future.result()
    write(root/'execution.json',dict(state='finished',finished_at=now(),pid=os.getpid()));report(root)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--prepare',type=Path);p.add_argument('--report-only',action='store_true');a=p.parse_args()
    if a.prepare:prepare(a.run.resolve(),a.prepare.resolve())
    elif a.report_only:report(a.run.resolve())
    else:run(a.run.resolve())
