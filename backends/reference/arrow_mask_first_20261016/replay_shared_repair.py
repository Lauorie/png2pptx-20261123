"""Focused mainline regression from frozen existing identities and pixel evidence.

Case/group selection is test scope only. No case IDs or answer coordinates enter
production recovery. Unselected routes remain byte-for-byte equal before publish.
"""
import argparse,copy,json,os,shutil,sys,threading,time,traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
import evidence as ev,semantic_geometry as semantic,geometry_routing,style_windows
import shared_repair,pipeline,protocol
from lib import fit
from transport.model import Model
from transport.common import read_json as read,write_json as write,sha256 as sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--scope',type=Path);p.add_argument('--frozen',action='store_true');a=p.parse_args()
    run=a.run.resolve();run.mkdir(parents=True,exist_ok=True);root=Path(__file__).resolve().parent
    if not a.frozen:
        if (run/'code_manifest.json').exists():raise ValueError('Choose a new frozen run')
        scope=read(a.scope);write(run/'scope.json',scope);config=read(root/'config.json');config['model_cache_from']=None
        config['case_workers']=scope.get('case_workers',config['case_workers'])
        write(run/'config.json',config);files={}
        for src in [*root.glob('*.py'),*(root/'lib').glob('*.py'),*(root/'transport').glob('*.py')]:
            rel=src.relative_to(root);dest=run/'code'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);files[str(rel)]=sha(dest)
        write(run/'code_manifest.json',dict(files=files,config_sha256=sha(run/'config.json')))
        os.execv(sys.executable,[sys.executable,'-u',str(run/'code'/Path(__file__).name),'--run',str(run),'--frozen'])
    config=read(run/'config.json');scope=read(run/'scope.json');source=Path(scope['source'])
    manifest=read(run/'code_manifest.json')
    if any(sha(root/n)!=v for n,v in manifest['files'].items()):raise ValueError('Frozen code changed')
    model=Model(config,run_dir=run,system=protocol.SYSTEM)
    locks={k:threading.Semaphore(config[k+'_workers']) for k in ['cpu','sam']}
    records=[{**t,'state':'waiting'} for t in scope['tasks']];mutex=threading.RLock();started=time.time()
    write(run/'process.json',dict(pid=os.getpid(),started=started,case_workers=min(8,len(records)),api_concurrency=64))
    def update():
        with mutex:
            write(run/'progress.json',records)
            lines=['# 旧坏例回归：跨组实际路径补全与几何证据传递','','复用上一轮整图结果，只重跑表中局部；普通数值拟合、样式算法不变。不是一次新的全量发现评测。','',
                   '| case / 局部 | 状态 | 原图 | 之前样式 | 当前样式 | 当前叠图 |','|---|---|---|---|---|---|']
            for r in records:
                rel='cases/'+r['case'];folder=run/rel
                cols=[f'[{r["case"]}/{r["group"]}]({rel}/RESULTS.md)',r['state']]
                cols += [f'![]({rel}/{name})' if (folder/name).exists() else '等待' for name in ['clean.png','before/styled.png','result/styled.png','result/overlap.png']]
                lines.append('| '+' | '.join(cols)+' |')
            (run/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    update()
    def work(record):
        cid,gid=record['case'],record['group'];case=source/'cases'/cid;old=case/'groups'/gid;out=run/'cases'/cid;dest=out/'groups'/gid
        out.mkdir(parents=True,exist_ok=True);dest.mkdir(parents=True,exist_ok=True);t=time.monotonic()
        try:
            record['state']='identity' if record['mode']=='identity_repair' else 'geometry';update()
            shutil.copy2(case/'clean.png',out/'clean.png');shutil.copytree(case/'result',out/'before',dirs_exist_ok=True)
            shutil.copytree(old/'input',dest/'input',dirs_exist_ok=True)
            clean=Image.open(out/'clean.png').convert('RGB');crop=Image.open(dest/'input/clean.png').convert('RGB');view=read(dest/'input/transform.json')
            evidence=old
            attempts=read(old/'sampling_attempts.json')
            if len(attempts)>1 and attempts[-1]['selected']:evidence=old/attempts[-1]['folder']
            selected_ids=set(read(evidence/'sam_summary.json')['selected_ids'])
            candidates=[{**r,'mask':ev.mask(r['path'])} for r in read(evidence/'candidates.json') if r['id'] in selected_ids]
            with locks['cpu']:proposals,info=semantic.prepare(crop,[],candidates,dest)
            if record['mode'] in ('geometry','repair_only'):
                # Frozen identity references D IDs: never regenerate/reorder this table.
                proposals=read(old/'proposals.json');write(dest/'proposals.json',proposals)
            previous=read(old/'decision.json');labels=list(dict.fromkeys([s for r in previous['routes'] for s in r['sources']]+previous['excluded']+previous['uncertain']))
            context_size=Image.open(dest/'input/context.png').size
            if record['mode']=='identity_repair':
                q=semantic.prompt(labels,crop.size,proposals,info,'',context_size,ev.crop_in_context(view,context_size))
                images=[('CONTEXT: whole original figure',dest/'input/context.png'),('CLEAN: local pixel frame',dest/'input/clean.png'),('GLOBAL HINTS: target locators',dest/'input/global_hints.png'),('MASKS: local SAM evidence',dest/'input/masks.png')]
                write(dest/'prompt.json',q)
                answer=model.ask(dest/'identity',q,images,lambda x:semantic.validate(x,crop.size,labels,proposals,selected_ids,candidates,context_size))
            else:answer=previous
            write(dest/'decision.json',answer)
            record['state']='geometry';update()
            if record['mode']=='repair_only':
                rows=read(old/'rows_local.json');global_rows=read(old/'rows_global.json');issues=[];calls=0
            else:
                rows,supports,issues,calls=geometry_routing.recover(answer,proposals,candidates,gid,crop,dest,model,locks)
                with locks['cpu']:
                    for row in rows:row['style']=style_windows.recover(np.asarray(crop),supports[row['id']],row);row['local_input']=str(dest/'input/clean.png')
                    global_rows=[fit.transform(r,view) for r in rows]
            write(dest/'rows_local.json',rows);write(dest/'rows_global.json',global_rows)
            all_before=read(case/'scene.json')['arrows'];untouched=[r for r in all_before if not r['id'].startswith(gid+':')]
            results=[dict(group='preserved',rows=untouched,state='completed'),dict(group=gid,rows=global_rows,state='completed_with_issues' if issues else 'completed')]
            write(out/'initial_results.json',results)
            with locks['cpu']:current=pipeline.publish(clean,results,out)
            tasks=[]
            if record['mode'] in ('identity_repair','repair_only'):
                tasks=[dict(bbox=(np.asarray(f['context_bbox'])*np.tile(np.asarray(clean.size)/context_size,2)).tolist(),description=f['description']) for f in answer.get('followups',[])]
                record['model_followups']=len(tasks)
                if not tasks:
                    # Diagnostic test scope only: the regression explicitly asks
                    # to revisit this known case/control. Production does not
                    # review every group or manufacture this fallback queue.
                    hint_index={h['id']:h for h in read(case/'global_hints.json')}
                    focus_bbox=[float(v) for v in ev.bounds([hint_index[g]['box'] for g in labels])]
                    tasks=[dict(bbox=view['bbox_global'],focus_bbox=focus_bbox,description='Check completeness of the connection around the supplied original target scope. Keep correct existing paths; repair only related missing portions or wrong topology, and exclude out-of-scope graphic components. Only the connection at FOCUS is under review, not every visible neighbor.')]
                    record['diagnostic_forced_review']=True
            write(out/'repair_queue.json',tasks)
            for i,task in enumerate(tasks,1):
                record['state']=f'repair {i}/{len(tasks)}';update()
                _,patch=shared_repair.recover(clean,task,current,f'P{i}',out/'continuation'/f'P{i}',config,model,locks)
                results=shared_repair.merge_results(results,patch)
                with locks['cpu']:current=pipeline.publish(clean,results,out)
            write(out/'results.json',results)
            removed={rid for r in results for rid in r.get('replace',[])}
            final_byid={r['id']:r for result in results for r in result.get('rows',[])}
            preserved=all(r['id'] in final_byid and r==final_byid[r['id']] for r in untouched if r['id'] not in removed)
            if not preserved:raise AssertionError('Unselected geometry/style changed')
            write(out/'provenance.json',dict(source=str(case),source_scene_sha256=sha(case/'scene.json'),replayed_group=gid,mode=record['mode'],preserved_unselected=True,
                geometry_issues=issues,whole_pipeline_rerun=False))
            (out/'RESULTS.md').write_text(f'# {cid}/{gid}\n\n{record["purpose"]}\n\n'+
                '| 原图 | 之前实例 | 当前实例 | 当前样式 | 叠图 |\n|---|---|---|---|---|\n| ![](clean.png) | ![](before/instances.png) | ![](result/instances.png) | ![](result/styled.png) | ![](result/overlap.png) |\n\n'+
                f'[本次身份决定](groups/{gid}/decision.json) · [几何请求](groups/{gid}/geometry/) · [补全区域](repair_queue.json) · [可编辑 PPT](arrows.pptx)\n\n'+
                '\n'.join(f'- [局部补全 P{i}](continuation/P{i}/README.md)' for i in range(1,len(tasks)+1))+'\n')
            record.update(state='completed_with_issues' if issues or any(x.get('state')=='completed_with_issues' for x in results) else 'completed',paths_before=len(all_before),paths_after=len(current),geometry_calls=calls,issues=issues)
        except Exception as exc:
            record.update(state='failed',error=str(exc));write(out/'error.json',dict(error=str(exc),traceback=traceback.format_exc()))
        record['seconds']=round(time.monotonic()-t,2);write(out/'completed.json',record);update();print(json.dumps(record,ensure_ascii=False),flush=True)
    with ThreadPoolExecutor(max_workers=min(config['case_workers'],len(records))) as pool:list(pool.map(work,records))
    write(run/'finished.json',dict(total=len(records),completed=sum(r['state'].startswith('completed') for r in records),seconds=time.time()-started,cases=records))


if __name__=='__main__':main()
