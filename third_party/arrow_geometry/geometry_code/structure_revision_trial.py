"""Replay exact old views with revised semantics/contracts, before full cases."""
import argparse,copy,json,os,shutil,sys,threading,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'lib'))
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from PIL import Image
import shared_repair as repair,semantic_geometry as semantic,geometry_routing as routing
import protocol,mask_vectors,style_windows,head_semantics,target_locator,shaft_sampling
from lib import fit,viz
from transport.common import read_json as read,write_json as write,sha256 as sha
from transport.model import Model


def main(run,base,task_file=None):
    run=Path(run).resolve();base=Path(base).resolve();cfg=read(run/'config.json')
    tasks=[dict(case=c,source=p,mode='repair',name=p.split('/')[-1]) for c,p in [
        ('073','missing/missing1_580343facc'),('073','missing/missing4_e53ee9c0f6'),
        ('ds013','missing/missing4_d418809f29'),('ds013','missing/missing1_a9e932c61a'),
        ('ds025','continuation/continuation3_dbfe8ba09e'),('ds025','continuation/continuation6_fde66049e3'),
        ('001','continuation/continuation2_0ce78e7614'),('063','missing/missing3_930f6b3acd')]]
    tasks += [dict(case=c,source='groups/'+g,mode='identity',name=g+'_'+v,variant=v) for c,g,v in [
        ('ds017','G001','normal'),('ds017','G019','normal'),('ds024','G011','normal'),
        ('ds024','G011','boxes_only'),('ds022','G005','normal'),('ds022','G005','boxes_only')]]
    if task_file:tasks=read(task_file)
    model=Model(cfg,run_dir=run,system=protocol.SYSTEM);locks=dict(cpu=threading.Semaphore(cfg['cpu_workers']))
    results={};started=time.time();write(run/'scope.json',dict(tasks=tasks,baseline=str(base),ds027_deferred=True))
    write(run/'process.json',dict(pid=os.getpid(),started=started))
    def report():
        write(run/'progress.json',dict(total=len(tasks),finished=len(results),records=list(results.values()),seconds=time.time()-started))
        lines=['# 身份、块状轮廓与曲线坏例修正','',f'局部任务结束 {len(results)}/{len(tasks)}；ds027 暂不处理。不是全图重跑。',
            '', '[曲线固定身份离线对照](offline_curves/README.md) · [本次修改](CHANGES.md)',
            '', '| case / 任务 | 状态 | 原图 | 旧局部结果 | 新局部结果 | 叠图 |','|---|---|---|---|---|---|']
        for i,t in enumerate(tasks):
            rel=Path('cases')/t['case']/t['name'];folder=run/rel;r=results.get(i,{})
            paths=['input/clean.png','before/styled.png','result/styled.png','result/overlap.png']
            images=[f'![]({rel/p})' if (folder/p).exists() else '' for p in paths]
            lines.append('| '+' | '.join([f'[{t["case"]}/{t["name"]}]({rel}/README.md)',r.get('state','运行／等待'),*images])+' |')
        (run/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    def work(t):
        source=base/'cases'/t['case']/t['source'];old=source/'local' if t['mode']=='repair' else source
        out=run/'cases'/t['case']/t['name'];out.mkdir(parents=True,exist_ok=True);shutil.copytree(old/'input',out/'input',dirs_exist_ok=True)
        clean=Image.open(out/'input/clean.png').convert('RGB');view=read(out/'input/transform.json')
        if (source/'patch_result').exists():shutil.copytree(source/'patch_result',out/'before',dirs_exist_ok=True)
        elif (source/'result').exists():shutil.copytree(source/'result',out/'before',dirs_exist_ok=True)
        oldq=read(old/'prompt.json');rows=[];issues=[]
        write(out/'source.json',dict(path=str(source),prompt_sha256=sha(old/'prompt.json'),crop_sha256=sha(out/'input/clean.png')))
        if t['mode']=='repair':
            images=[(x['label'],out/'input'/Path(x['path']).name) for x in read(old/'input/images.json')]
            q=repair.prompt(oldq['size'],oldq['context_size'],oldq['focus'],oldq['mask_measurements'],oldq['existing'],True)
            q['crop_in_context']=oldq.get('crop_in_context');q['instructions'].append('crop_in_context locates CLEAN in CONTEXT; do not infer architecture from the region size.')
            mids=[m['id'] for m in oldq['mask_measurements']];existing={r['id']:r['points'] for r in oldq['existing']}
            validate=lambda a:repair.validate(a,clean.size,oldq['context_size'],existing,mids,True,existing)
        else:
            oldq=read(old/'identity/identity.json')['prompt'];proposals=read(old/'proposals.json')
            meta=read(old/'prepared_observation.json');index={c['id']:c for c in meta['candidates']}
            candidates=[dict(index[mid],mask=np.array(Image.open(index[mid]['path']).convert('L'))>0) for mid in meta['selected_ids']]
            q=semantic.prompt(oldq['sources'],clean.size,proposals,oldq['masks'],'',oldq['context_size'],oldq['crop_in_context'])
            hints={r['id']:r for r in read(base/'cases'/t['case']/'global_hints.json')}
            images=[(x['label'],out/'input'/Path(x['path']).name) for x in read(old/'input/images.json')]
            images=target_locator.apply(q,images,[hints[k] for k in oldq['sources']],view,clean,out)
            if t['variant']=='boxes_only':
                candidates=[];proposals={};q['masks']=[];q['proposals']=[]
                images=[(name,path) for name,path in images if not name.startswith('MASKS')]
                q['instructions'].append('This observation deliberately supplies no mask measurements. Use original pixels for physical identities; choose GLM geometry for complex routes. Missing SAM is not a reason to omit a visible line.')
            validate=lambda a:semantic.validate(a,clean.size,q['sources'],proposals,[c['id'] for c in candidates],candidates,q['context_size'])
        write(out/'prompt.json',q);write(out/'input/images.json',[dict(label=n,path=str(p),sha256=sha(p)) for n,p in images])
        (out/'README.md').write_text('# '+t['case']+'/'+t['name']+'\n\n[输入 prompt](prompt.json)\n\n'+
            '\n\n'.join('**'+n+'**\n\n![](input/'+p.name+')' for n,p in images)+'\n')
        answer=model.ask(out/'request',q,images,validate);write(out/'answer.json',answer)
        if t['mode']=='identity':
            write(out/'decision.json',answer)
            rows,supports,issues,calls=routing.recover(answer,proposals,candidates,t['name'],clean,out,model,locks)
        elif answer['status']=='repair':
            index={c['id']:c for c in read(old/'candidates.json')} if (old/'candidates.json').exists() else {}
            metadata={r['id']:dict(id=t['name']+':'+r['id'],heads=r.get('heads',['unknown','unknown']),line_style=r['line_style'],
                source_masks=r['masks'],source_mask_paths=[index[m]['path'] for m in r['masks']],globals=[],identity_index=i+1,
                local_input=str(out/'input/clean.png')) for i,r in enumerate(answer['routes'])}
            center={**answer,'routes':[r for r in answer['routes'] if r.get('kind')!='block_arrow']}
            rows=repair.sg.geometry(center,metadata)
            for r in answer['routes']:
                if r.get('kind')=='block_arrow':
                    ix={k:dict(v,mask=np.array(Image.open(v['path']).convert('L'))>0) for k,v in index.items()}
                    rows.append({**metadata[r['id']],**repair.block_geometry(r,answer['nodes'],ix),'heads':['none','none'],'shared_nodes':[]})
            supports={r['id']:np.logical_or.reduce([np.array(Image.open(p).convert('L'))>0 for p in r['source_mask_paths']]) if r['source_mask_paths'] else None for r in rows}
            rows,head_issues,_=head_semantics.recover(clean,rows,supports,out/'heads',model,cpu_slot=locks['cpu']);issues+=head_issues
            for row in rows:
                for end,h in zip(['start','end'],row['heads']):row[end+'_head']=h not in ('none','unknown');row[end+'_head_type']=h if h not in ('none','unknown') else None
                row.setdefault('node_ids',[]);row.setdefault('shared_nodes',[])
            mask_vectors.assign_parents(rows,t['name'])
        else:
            supports={};issues=[dict(status=answer['status'])] if answer['status'] in ('unresolved','expand') else []
        with locks['cpu']:
            for row in rows:row['style']=style_windows.recover(np.array(clean),supports[row['id']],row);row['local_input']=str(out/'input/clean.png')
            write(out/'rows_local.json',rows);viz.render(out/'input/clean.png',rows,out/'result')
            write(out/'rows_global.json',[fit.transform(r,view) for r in rows])
        text=(out/'README.md').read_text()+'\n\n[模型决定](answer.json) · [完整 trace](request/prompt.json)\n\n| 旧局部结果 | 新局部结果 | 新结果叠图 |\n|---|---|---|\n| ![](before/styled.png) | ![](result/styled.png) | ![](result/overlap.png) |\n'
        if t['mode']=='repair':text+='\n局部结果只显示本次补丁；complete 表示无需添加，空白图不代表原有箭头被删除。\n'
        (out/'README.md').write_text(text)
        r=dict(**t,state='completed_with_issues' if issues else 'completed',rows=len(rows),issues=issues,status=answer.get('status'),excluded=answer.get('excluded'),replace=answer.get('replace'),routes=[{k:x.get(k) for k in ['id','kind','sources','masks','description']} for x in answer['routes']])
        write(out/'completed.json',r);return r
    report()
    with ThreadPoolExecutor(max_workers=24) as pool:
        futures={pool.submit(work,t):i for i,t in enumerate(tasks)}
        for f in as_completed(futures):
            i=futures[f]
            try:results[i]=f.result()
            except Exception as exc:
                results[i]=dict(**tasks[i],state='failed',error=str(exc));write(run/'cases'/tasks[i]['case']/tasks[i]['name']/'error.json',dict(**results[i],traceback=traceback.format_exc()))
            report();print(tasks[i]['case'],tasks[i]['name'],results[i]['state'],flush=True)
    write(run/'finished.json',dict(total=len(tasks),records=list(results.values()),seconds=time.time()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--baseline',required=True);p.add_argument('--tasks')
    a=p.parse_args();main(a.run,a.baseline,a.tasks)
