"""Frozen full-cache style replay: GLM categories + the mainline style entry.

All existing groups and omission groups are included. Geometry, heads, identity,
and original SAM evidence remain frozen; no discovery or segmentation calls.
"""
import argparse,collections,copy,fcntl,json,os,shutil,sys,threading,time,traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from PIL import Image
import pipeline,protocol,style_semantics
from transport.model import Model
from transport.common import read_json as read,write_json as write,sha256 as sha


def support_files(base,row):
    mids=row.get('source_masks',[])
    if mids and all((base/'masks'/f'{k}.png').exists() for k in mids):return [base/'masks'/f'{k}.png' for k in mids]
    local=row['id'].split(':',1)[-1]
    if local.startswith('R') and ':' in local:return [base/f'repair_{local.split(":")[0][1:]}/proposal/support.png']
    return [base/'proposal/support.png']


def group_task(src,out,model,cpu):
    saved=out/'completed.json';provenance=out/'provenance.json'
    if saved.exists() and provenance.exists():
        previous=read(saved);evidence=read(provenance)
        if not previous.get('errors') and evidence.get('source')==str(src.resolve()) and all(
                Path(p).exists() and sha(Path(p))==digest for p,digest in evidence['hashes'].items()):
            if all((out/p).exists() for p in ('rows_local.json','rows_global.json','result/styled.png','README.md')):
                write(out/'cache_resume.json',dict(source_inputs_unchanged=True,new_api_calls=0,numerical_result_reused=True))
                return previous
    out.mkdir(parents=True,exist_ok=True);base=src/'baseline';snapshot=out/'input'
    original=read(src/'rows_local.json');rows=copy.deepcopy(original);stroke=[r for r in rows if r['kind']!='block_arrow']
    hashes={};files=[src/'rows_local.json',base/'input/clean.png',base/'input/transform.json']
    for row in stroke:files+=support_files(base,row)
    for f in set(files):
        if not f.exists():continue
        relative=Path('before_rows.json') if f==src/'rows_local.json' else f.relative_to(base)
        target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target);hashes[str(f)]=sha(target)
    write(out/'provenance.json',dict(source=str(src.resolve()),hashes=hashes))
    clean_path=snapshot/'input/clean.png';clean=Image.open(clean_path).convert('RGB')
    styles={};errors=[];batches=[]
    for i in range(0,len(stroke),3):
        name=f'B{i//3+1:03d}';batch=stroke[i:i+3];folder=out/'classify'/name
        try:
            styles.update(style_semantics.classify(clean,batch,folder,model))
            batches.append(dict(batch=name,state='completed',rows=[r['id'] for r in batch]))
        except Exception as exc:
            record=dict(batch=name,rows=[r['id'] for r in batch],error=str(exc));errors.append(record)
            write(folder/'error.json',dict(**record,traceback=traceback.format_exc()))
            batches.append(dict(batch=name,state='failed',rows=record['rows']))
    measured=[];preserved=[]
    with cpu:
        for row in stroke:
            if row['id'] not in styles:preserved.append(dict(id=row['id'],reason='category request failed'));continue
            row['line_style']=styles[row['id']]
            paths=[snapshot/f.relative_to(base) for f in support_files(base,row)]
            if not all(p.exists() for p in paths):
                errors.append(dict(id=row['id'],error='missing frozen support'));preserved.append(dict(id=row['id'],reason='missing support'));continue
            support=np.logical_or.reduce([np.array(Image.open(p).convert('L'))>0 for p in paths])
            try:
                pipeline.style_rows(clean,[row],support);measured.append(row['id'])
            except Exception as exc:
                old=next(r for r in original if r['id']==row['id']);row['style']=copy.deepcopy(old['style'])
                errors.append(dict(id=row['id'],error=str(exc)));preserved.append(dict(id=row['id'],reason='style measurement failed; old style retained'))
        # Assert preservation, including order and editable geometry control points.
        for a,b in zip(original,rows):
            for k in ('id','parent_id','identity_index','start','via','end','beziers','contours','heads','node_ids','globals','source_masks'):
                if a.get(k)!=b.get(k):raise AssertionError('Style replay changed '+k)
        write(out/'rows_local.json',rows)
        pipeline.viz.render(clean_path,original,out/'before');pipeline.viz.render(clean_path,rows,out/'result')
        view=read(snapshot/'input/transform.json');global_rows=[pipeline.fit.transform(r,view) for r in rows]
        for local,glob in zip(rows,global_rows):glob['native_route_path']=local.get('route_path');glob.pop('route_path',None)
        write(out/'rows_global.json',global_rows)
    pending=[r['id'] for r in rows if pipeline.style.pending(r)];write(out/'pending_style.json',pending)
    summary=dict(state='completed_with_issues' if errors or pending else 'completed',rows=global_rows,
        stroke_rows=len(stroke),classified=len(styles),measured=len(measured),unchanged_filled_rows=len(rows)-len(stroke),
        batches=batches,errors=errors,preserved=preserved,pending=pending,geometry_and_heads_unchanged=True,
        kinds=dict(collections.Counter(r['kind'] for r in stroke)),origins=dict(collections.Counter(r['coordinate_source'] for r in stroke)),
        categories=dict(collections.Counter(styles.values())),
        transitions=dict(collections.Counter(a['style'].get('dash',{}).get('type','filled')+' -> '+b['style'].get('dash',{}).get('type','filled') for a,b in zip(original,rows))))
    write(out/'completed.json',summary)
    (out/'README.md').write_text('# '+src.parent.parent.name+' / '+src.name+'\n\n冻结身份、几何、两端头型；只重新判断虚实线类别并测样式。? 表示类别和像素均不确定：保留几何待复查，未提交到样式图/PPT。\n\n'
        '| 原图 | 旧样式 | 新样式 | 实例定位 |\n|---|---|---|---|\n'
        '| ![](input/input/clean.png) | ![](before/styled.png) | ![](result/styled.png) | ![](result/instances.png) |\n\n'
        '[统计与问题](completed.json) · [待复查](pending_style.json)\n\n'+
        '\n'.join(f'- [{b["batch"]} 输入](classify/{b["batch"]}/locator.png) · [原图](classify/{b["batch"]}/clean.png) · [请求](classify/{b["batch"]}/request/prompt.json)' for b in batches)+'\n')
    return summary


def failed_group(src,out,error):
    """An infrastructure failure cannot silently remove existing geometry."""
    rows=read(src/'rows_local.json');view=read(src/'baseline/input/transform.json')
    global_rows=[pipeline.fit.transform(r,view) for r in rows]
    for r in global_rows:r.pop('route_path',None)
    stroke=[r for r in rows if r['kind']!='block_arrow']
    summary=dict(state='completed_with_issues',rows=global_rows,stroke_rows=len(stroke),classified=0,measured=0,
        unchanged_filled_rows=len(rows)-len(stroke),batches=[],errors=[dict(error=error)],preserved=[r['id'] for r in rows],pending=[],
        kinds=dict(collections.Counter(r['kind'] for r in stroke)),origins=dict(collections.Counter(r['coordinate_source'] for r in stroke)),
        categories={},transitions={},geometry_and_heads_unchanged=True)
    write(out/'rows_local.json',rows);write(out/'rows_global.json',global_rows);write(out/'completed.json',summary)
    (out/'README.md').write_text('# 本组未完成新样式测试\n\n原有几何与样式保留，错误见 [error.json](error.json)。\n')
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'));p.add_argument('--cases',nargs='*');p.add_argument('--frozen',action='store_true')
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True);source=a.source.resolve();cfg=read(a.config)
    if not a.frozen:
        root=Path(__file__).resolve().parent;manifest={}
        if not (out/'code_manifest.json').exists():
            for src in [*root.glob('*.py'),*root.glob('lib/*.py'),*root.glob('transport/*.py')]:
                dst=out/'code'/src.relative_to(root);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);manifest[str(src.relative_to(root))]=sha(dst)
            write(out/'code_manifest.json',manifest);write(out/'config.json',cfg)
            write(out/'scope.json',dict(source=str(source),cases=a.cases))
        assert read(out/'config.json')==cfg and read(out/'scope.json')==dict(source=str(source),cases=a.cases)
        assert all(sha(out/'code'/k)==v for k,v in read(out/'code_manifest.json').items())
        command=[sys.executable,'-u',str(out/'code/run_style40.py'),'--source',str(source),'--out',str(out),'--config',str(out/'config.json'),'--frozen']
        if a.cases:command+=['--cases',*a.cases]
        os.execv(sys.executable,command)
    lock=(out/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(out/'process.json',dict(pid=os.getpid(),started=time.time()))
    cases=a.cases or sorted(p.name for p in (source/'cases').iterdir() if (p/'scene.json').exists())
    write(out/'manifest.json',dict(cases=cases,source=str(source),same_algorithm_all_cases=True,
        max_lines_per_request=3,api_concurrency=cfg['api_concurrency'],new_sam_calls=0,scope='all cached stroked geometry, including recovery and missing groups'))
    cpu=threading.Semaphore(cfg['cpu_workers']);model=Model(cfg,run_dir=out,system=protocol.SYSTEM)
    tasks=[];remaining={};done=collections.defaultdict(list);case_status={};failures=[];started=time.monotonic()
    for cid in cases:
        folder=out/'cases'/cid;folder.mkdir(parents=True,exist_ok=True);shutil.copy2(source/'cases'/cid/'clean.png',folder/'clean.png')
        shutil.copytree(source/'cases'/cid/'result',folder/'before',dirs_exist_ok=True)
        current=[]
        for category in ('groups','missing'):
            current += [(cid,category,p.parent) for p in sorted((source/'cases'/cid/category).glob('*/rows_local.json'))]
        tasks+=current;remaining[cid]=len(current);case_status[cid]='waiting'
    def report():
        write(out/'progress.json',dict(cases=len(cases),finished_cases=sum(v in ('completed','completed_with_issues') for v in case_status.values()),
            groups_done=sum(map(len,done.values())),groups_total=len(tasks),remaining=remaining,status=case_status,seconds=time.monotonic()-started,failures=failures))
        lines=['# 40 张：主管线全部已有线条的样式验证','','复用身份、几何、头型和 SAM；GLM 每次最多判断 3 条线的虚实类别，64 并发上限。普通组、补漏组和修复路径都进入同一主线样式函数。','',
            '实例图中的 ? 表示样式仍待确认；几何保存在 scene.json/pending_style.json，未提交到样式图/PPT。原先缺失/未完成的几何没有在本次重建。','',
            '| case | 状态 | 原图 | 旧样式 | 新样式 | 实例 |','|---|---|---|---|---|---|']
        for cid in cases:
            base=f'cases/{cid}';folder=out/base
            images=[f'![]({base}/{n})' if (folder/n).exists() else '' for n in ('clean.png','before/styled.png','result/styled.png','result/instances.png')]
            lines.append('| '+' | '.join([f'[{cid}]({base}/RESULTS.md)' if (folder/'RESULTS.md').exists() else cid,case_status[cid],*images])+' |')
        lines+=['','[执行与类别统计](finished.json) · [所有入口验证](ENTRYPOINTS.md)']
        (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    def publish_case(cid):
        folder=out/'cases'/cid;items=sorted(done[cid],key=lambda x:(x['category'],x['group']))
        with cpu:
            rows=pipeline.publish(Image.open(folder/'clean.png').convert('RGB'),items,folder)
        # Geometry labels are stable because every original group/row survives.
        issues=[i for i in items if i['state']!='completed'];pending=[r['id'] for r in rows if pipeline.style.pending(r)]
        case_status[cid]='completed_with_issues' if issues or any(f['case']==cid for f in failures) else 'completed'
        old_audit=read(source/'cases'/cid/'audit.json') if (source/'cases'/cid/'audit.json').exists() else {}
        write(folder/'completed.json',dict(state=case_status[cid],groups=len(items),draw_paths=len(rows),
            stroke_rows=sum(i['stroke_rows'] for i in items),classified=sum(i['classified'] for i in items),
            measured=sum(i['measured'] for i in items),pending=pending,issues=[dict(group=i['group'],errors=i['errors']) for i in issues],
            upstream_missing=old_audit.get('upstream_missing',[]),geometry_and_heads_unchanged=True))
        (folder/'RESULTS.md').write_text('# '+cid+'\n\n状态：'+case_status[cid]+'。? 为样式待确认，几何保留，未提交到样式图/PPT。\n\n'
            '| 原图 | 旧样式 | 新样式 | 实例 |\n|---|---|---|---|\n'
            '| ![](clean.png) | ![](before/styled.png) | ![](result/styled.png) | ![](result/instances.png) |\n\n'
            '[PPT](arrows.pptx) · [完整几何](scene.json) · [待确认](pending_style.json) · [统计](completed.json)\n\n'+
            '\n'.join(f'- [{i["category"]}/{i["group"]}]({i["category"]}/{i["group"]}/README.md)' for i in items)+'\n')
    report()
    with ThreadPoolExecutor(max_workers=cfg['api_concurrency']) as pool:
        futures={pool.submit(group_task,src,out/'cases'/cid/category/src.name,model,cpu):(cid,category,src) for cid,category,src in tasks}
        for f in as_completed(futures):
            cid,category,src=futures[f]
            try:done[cid].append(dict(f.result(),category=category,group=src.name))
            except Exception as exc:
                failures.append(dict(case=cid,category=category,group=src.name,error=str(exc)))
                write(out/'cases'/cid/category/src.name/'error.json',dict(error=str(exc),traceback=traceback.format_exc()))
                done[cid].append(dict(failed_group(src,out/'cases'/cid/category/src.name,str(exc)),category=category,group=src.name))
            remaining[cid]-=1;case_status[cid]='running'
            if remaining[cid]==0:publish_case(cid);print(cid,case_status[cid],flush=True)
            report()
    for cid in cases:
        if not done[cid]:publish_case(cid)
    counters={k:collections.Counter() for k in ('kinds','origins','categories','transitions')}
    for groups in done.values():
        for g in groups:
            for k in counters:counters[k].update(g[k])
    write(out/'finished.json',dict(cases=len(cases),groups=sum(map(len,done.values())),failures=failures,
        stroke_rows=sum(g['stroke_rows'] for groups in done.values() for g in groups),
        classified=sum(g['classified'] for groups in done.values() for g in groups),
        measured=sum(g['measured'] for groups in done.values() for g in groups),
        request_failures=[dict(case=cid,group=g['group'],errors=g['errors']) for cid,groups in done.items() for g in groups if g['errors']],
        **{k:dict(v) for k,v in counters.items()},seconds=time.monotonic()-started))
    report();print('FINISHED',flush=True)


if __name__=='__main__':main()
