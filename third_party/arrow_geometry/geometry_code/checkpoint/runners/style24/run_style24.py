"""Frozen-geometry style replay with three head categories and live Markdown."""
import argparse,copy,fcntl,hashlib,json,os,shutil,sys,threading,time,traceback
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor,as_completed

REPORT_LOCK=threading.Lock()
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def read(p,default=None):return json.loads(Path(p).read_text()) if Path(p).exists() else default
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name('.'+p.name+'.tmp')
    q.write_text(x if isinstance(x,str) else json.dumps(x,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def report(root):
    with REPORT_LOCK:
        records=[];body=[]
        def image(cid,label,path):
            if not path.exists():return '等待'
            asset=root/'reports/assets'/f'{cid}_{label}_{path.stat().st_mtime_ns}.png';asset.parent.mkdir(parents=True,exist_ok=True)
            if not asset.exists():shutil.copy2(path,asset)
            return f'[![{label}](assets/{asset.name})](assets/{asset.name})'
        for cid in read(root/'cases.json'):
            d=root/'cases'/cid;s=read(d/'progress.json',dict(state='queued',stage='queued',ended=0,total=0));records.append(dict(case=cid,**s))
            body += [f'## {cid}', '',f'{s["state"]} · {s["stage"]}；样式批次 {s.get("ended",0)}/{s.get("total",0)}。', '',
                f'[样式请求与critic](../cases/{cid}/styles/) · [状态](../cases/{cid}/progress.json) · [问题](../cases/{cid}/completed.json) · [PPT](../cases/{cid}/arrows.pptx)','',
                '| 原图 | 样式恢复叠图（运行中为SVG预览，结束后为实际PPT） |','|---|---|',
                '| '+image(cid,'source',d/'clean.png')+' | '+image(cid,'styled',d/'result/overlap.png')+' |','',
                '| 输入几何 | 样式恢复纯重建 |','|---|---|',
                '| '+image(cid,'before',d/'input/before.png')+' | '+image(cid,'pure',d/'result/styled.png')+' |','']
        ended=sum(x['state'] in ['completed','completed_with_issues','failed'] for x in records)
        write(root/'progress.json',dict(updated_at=now(),total=len(records),ended=ended,active=sum(x['state']=='running' for x in records),cases=records))
        header=['# 24图：三类头型、固定mid、原图1px线宽下限','',f'更新 {now()}；结束 {ended}/{len(records)}。','',
            '复用v032全部24图已去重的几何，四个块状箭头采用v035最新结果。重新调用样式估计及实际PPT样式critic，不重跑定位、几何、去重或SAM。0639的新几何来自SAM兜底，源问题单独记录。','',
            '端点只分none/circle/arrow，可纠正几何阶段的头型；后台映射none/oval/triangle，头宽/长固定med/med。线宽和有效shape描边至少1原图像素，局部采样单位严格换算，原始模型输出保留。颜色、虚实和shape填充/描边正常恢复；不新增shape描边或改几何。','',
            '16图并行，共享API128，每图最多8个样式批次并行，每批最多4条路径。默认thinking，113000输出上限，样式最多refine一次；只对实际候选做PPT渲染审核，最终复用同一渲染。无范围审核/几何后补漏。','',
            'completed_with_issues可能继承几何阶段问题或本轮样式未解决；二者分开记录。运行完成和模型accept不等于视觉正确。','',
            '[协议](../STYLE_PROTOCOL.md) · [配置](../config.json) · [输入来源](../input_provenance.json) · [检查](../preflight/)','']
        write(root/'reports/24图_样式实时对照.md','\n'.join(header+body)+'\n')
        write(root/'RESULTS.md','\n'.join(header)+'\n\n[24图实时对照](reports/24图_样式实时对照.md)\n')

def prepare(root,source,shapes,workspace):
    if root.exists():raise ValueError('Run already exists')
    root.mkdir(parents=True);(root/'code').mkdir();shutil.copy2(__file__,root/'code'/Path(__file__).name)
    shutil.copytree(workspace/'geometry_code',root/'geometry_code',ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache'))
    shutil.copytree(shapes/'constraints_snapshot',root/'constraints_snapshot')
    sys.path[:0]=[str(root/'geometry_code'),str(root/'geometry_code/lib')]
    from lib import fit
    from simple_style_contract import Contract
    from PIL import Image
    ids=read(source/'cases.json');write(root/'cases.json',ids)
    cfg=read(source/'config.json');cfg.update(style_profile='three_heads_mid_min1',pipeline_mode='full',separate_style=True,style_entry='estimate',style_refine_rounds=1,
        style_pixel_evidence=False,style_detail_views=True,style_batch_max_paths=4,case_workers=16,task_workers=8,api_concurrency=128,cpu_workers=4,
        model_cache_from=None,coverage_rounds=0,scope_review=False,run_label=root.name,source_geometry=str(source),shape_overrides=str(shapes),preview_interval_seconds=15)
    cfg.pop('reasoning_effort',None);write(root/'config.json',cfg)
    override={j['case']:j for j in read(shapes/'targets.json')};provenance=[]
    for cid in ids:
        src=source/'cases'/cid;d=root/'cases'/cid;inputs=d/'input';inputs.mkdir(parents=True)
        shutil.copy2(src/'clean.png',d/'clean.png');shutil.copy2(src/'result/styled.png',inputs/'before.png')
        groups=read(src/'geometry_results.json');write(inputs/'source_geometry_results.json',groups)
        source_issues=read(src/'completed.json');extras=[]
        if cid in override:
            target=override[cid]['target'];sc=shapes/'cases'/cid
            assert (sc/'completed.json').exists(),'Shape override unfinished: '+cid
            local=read(sc/'final/row.json');fresh=fit.transform(local,local['geometry_frame']);matches=0
            for group in groups:
                for i,row in enumerate(group.get('rows',[])):
                    if row['id']==target:group['rows'][i]=fresh;matches+=1
            assert matches==1,(cid,target,matches)
            extras=read(sc/'completed.json').get('issues',[])
            write(inputs/'shape_override.json',dict(source=str(sc),target=target,coordinate_source=fresh['coordinate_source'],issues=extras))
        crop_map={}
        for group in groups:
            for row in group.get('rows',[]):
                old=row.get('local_input')
                if old:
                    f=Path(old);assert f.is_file(),str(f)
                    if old not in crop_map:
                        dst=inputs/'crops'/(hashlib.sha256(old.encode()).hexdigest()[:16]+'.png');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst)
                        crop_map[old]=str(dst);provenance.append(dict(case=cid,source=old,target=str(dst.relative_to(root)),sha256=sha(f)))
                    row['local_input']=crop_map[old]
        write(inputs/'geometry_results.json',groups);write(inputs/'source_issues.json',dict(geometry=source_issues,shape_override=extras))
        provenance.append(dict(case=cid,source=str(src/'geometry_results.json'),sha256=sha(src/'geometry_results.json'),shape_override=override.get(cid)))
        rows=[row for g in groups for row in g.get('rows',[])];example=rows[:1]
        if cid==ids[0]:write(root/'example_style_prompt.json',Contract().prompt(example,Image.open(d/'clean.png').size))
    write(root/'input_provenance.json',provenance)
    write(root/'STYLE_PROTOCOL.md','''# 样式协议

模型重新核对每条路径两个物理端点，只输出 none/circle/arrow；旧几何头型不是锁定真值。后台映射无头/oval/triangle，宽长固定med/med，不询问或审核头部尺寸。

模型估计CLEAN局部像素中的线宽、颜色和虚实；后台按view_to_global矩阵换算原图像素，低于1px抬到1px，同时保存原始和应用后的数值。shape无描边仍无描边；有描边时同样执行1px下限。形状填充可保持无填充/纯色/线性渐变。样式阶段不改路径/轮廓、不选择SAM。

每批最多4条路径；估计→实际PPT审核，明确样式错误最多修改一次。模型看到原图、定位、当前样式和实际PPT；审核附端点/箭身的同坐标细节对照。固定mid和1px下限的差异不要求修复。

2048px图在1px线宽下可能触及PPT 56英寸上限：原生渲染允许最多2%的头部底限误差，保留1px线宽和med/med参数不变，逐PPT写入scale.json。更大的超限仍报错，不默默改变源几何或线宽。
''')
    write(root/'constraints_snapshot/style_user_20261029.md','用户确认当前全部24张图；端点只分圆形头、无头、普通头；头尺寸mid不估计；线宽低于原图1px由后台截断。\n')
    files={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()}
    write(root/'manifest.json',dict(prepared_at=now(),files=files));report(root)

def run(root):
    lock=(root/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for rel,digest in read(root/'manifest.json')['files'].items():
        if sha(root/rel)!=digest:raise ValueError('Frozen input changed: '+rel)
    sys.path[:0]=[str(root/'geometry_code'),str(root/'geometry_code/lib')]
    from PIL import Image
    from lib import native,viz
    from native_foreground import render_layer
    from native_runtime import validated_renderer
    from simple_style_contract import Contract,merge_style
    import style_pipeline,style_contract
    from transport.model import Model
    cfg=read(root/'config.json');locks={'cpu':threading.Semaphore(cfg['cpu_workers'])}
    model=Model(cfg,run_dir=root,reuse_from=cfg.get('style_reuse_from'),system='Use CLEAN as source evidence. Recover styles only; verify the physical endpoint categories. Return the requested JSON.')
    renderer,version=validated_renderer();write(root/'renderer.json',dict(executable=str(renderer),version=version))
    started=now();write(root/'execution.json',dict(state='running',started_at=started,pid=os.getpid()))
    stop=threading.Event()
    def watch():
        while not stop.wait(15):report(root)
    watcher=threading.Thread(target=watch,daemon=True);watcher.start()
    def case(cid):
        d=root/'cases'/cid;clean=Image.open(d/'clean.png').convert('RGB');groups=read(d/'input/geometry_results.json');begin=time.monotonic();last_preview=0
        originals=[r for g in groups for r in g.get('rows',[])];signature=style_contract.geometry_signature(originals)
        def progress(stage,ended=0,total=0,state='running',**extra):write(d/'progress.json',dict(state=state,stage=stage,ended=ended,total=total,seconds=time.monotonic()-begin,**extra))
        def preview(rows):
            import cairosvg,io
            dest=d/'result';dest.mkdir(exist_ok=True)
            rgba=Image.open(io.BytesIO(cairosvg.svg2png(bytestring=viz.svg(clean.size,rows).encode()))).convert('RGBA')
            white=Image.new('RGBA',clean.size,'white');white.alpha_composite(rgba);white.convert('RGB').save(dest/'styled.png')
            overlay=clean.convert('RGBA');overlay.alpha_composite(rgba);overlay.convert('RGB').save(dest/'overlap.png')
        def callback(ended,total,updates,issues):
            nonlocal last_preview
            progress('style_recovery',ended,total,issues=len(issues));write(d/'partial_styles.json',updates)
            if time.monotonic()-last_preview>=cfg['preview_interval_seconds']:
                current=Contract().enforce_rows([merge_style(r,updates.get(r['id'],r)) for r in originals])
                with locks['cpu']:preview(current)
                last_preview=time.monotonic()
        try:
            progress('style_recovery');final,issues=style_pipeline.finish_case(clean,groups,d,cfg,model,locks,notify=callback)
            rows=[r for g in final for r in g.get('rows',[])];assert style_contract.geometry_signature(rows)==signature
            write(d/'geometry_preservation.json',dict(exact_geometry_unchanged=True,paths=len(rows)))
            write(d/'rows.json',rows);counts=read(d/'progress.json');progress('final_render',counts.get('ended',0),counts.get('total',0))
            with locks['cpu']:
                native.write(clean.size,rows,[r['style'] for r in rows],d/'arrows.pptx',head_floor_tolerance=.02)
                native.render(d/'arrows.pptx',d/'native',clean.size)
                layer=render_layer(d/'native/styled.svg',clean.size,d/'native')
                (d/'result').mkdir(exist_ok=True);shutil.copy2(d/'native/styled.png',d/'result/styled.png')
                overlay=clean.convert('RGBA');overlay.alpha_composite(layer);overlay.convert('RGB').save(d/'result/overlap.png')
            inherited=read(d/'input/source_issues.json');upstream=bool(inherited['geometry'].get('issues') or inherited['geometry'].get('unresolved') or inherited['geometry'].get('upstream_issues') or inherited['shape_override'])
            state='completed_with_issues' if issues or upstream else 'completed'
            write(d/'completed.json',dict(state=state,style_issues=issues,inherited_geometry_issues=inherited,paths=len(rows),visual_accuracy_measured=False))
            progress('finished',counts.get('ended',0),counts.get('total',0),state,style_issues=len(issues),inherited_issues=upstream)
            print(cid,state,'style_issues',len(issues),flush=True)
        except Exception as exc:
            write(d/'fatal.json',dict(error=str(exc),traceback=traceback.format_exc()));progress('failed',state='failed',error=str(exc));print(cid,'FAILED',str(exc),flush=True)
        report(root)
    try:
        with ThreadPoolExecutor(max_workers=cfg['case_workers']) as pool:
            for future in as_completed([pool.submit(case,cid) for cid in read(root/'cases.json') if not (root/'cases'/cid/'completed.json').exists()]):future.result()
        write(root/'execution.json',dict(state='finished',started_at=started,finished_at=now(),pid=os.getpid()))
    finally:stop.set();watcher.join();report(root)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--source',type=Path);p.add_argument('--shapes',type=Path);p.add_argument('--workspace',type=Path);p.add_argument('--report-only',action='store_true');a=p.parse_args()
    if a.prepare:prepare(a.run.resolve(),a.source.resolve(),a.shapes.resolve(),a.workspace.resolve())
    elif a.report_only:report(a.run.resolve())
    else:run(a.run.resolve())
