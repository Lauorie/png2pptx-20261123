"""Frozen, selection-driven local experiments; no case-specific pipeline logic."""
import argparse,copy,json,os,shutil,sys,threading,time,traceback,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
sys.path.insert(0,str(Path(__file__).resolve().parent/'lib'))
import numpy as np
from PIL import Image
import evidence as ev,style_pipeline,semantic_geometry,protocol,target_locator
from lib import fit
from transport.model import Model
from transport.common import read_json as read,write_json as write,sha256 as sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--frozen',action='store_true')
    args=parser.parse_args();out=args.run.resolve();source=args.source.resolve()
    jobs=read(args.selection);config=read(args.config);out.mkdir(parents=True,exist_ok=True)
    if not args.frozen:
        write(out/'selection.json',jobs)
        command=[sys.executable,str(Path(__file__).with_name('run.py')),'--run',str(out),'--config',str(args.config.resolve()),'--freeze-only']
        subprocess.run(command,check=True)
        command=[sys.executable,'-u',str(out/'code/replay_upgrade.py'),'--run',str(out),'--source',str(source),
                 '--selection',str(out/'selection.json'),'--config',str(out/'config.json'),'--frozen']
        os.execv(sys.executable,command)
    write(out/'process.json',dict(pid=os.getpid(),started=time.time(),source=str(source),kind='local replay'))
    model=Model(config,run_dir=out,system=protocol.SYSTEM)
    locks={'cpu':threading.Semaphore(config['cpu_workers']),'sam':threading.Semaphore(config['sam_workers'])}
    states={};mutex=threading.Lock()
    def report(name=None,value=None):
        with mutex:
            if name:states[name]=value
            write(out/'progress.json',dict(updated=time.time(),jobs=states))
            lines=['# 范围和独立样式流程：局部验证','',
                   '复用已保存的原图、mask 和几何；不是整图从头重跑。参数预算不变。样式对照使用实际 PPT，身份回放只证明发现／路由输出，不能算最终召回。','',
                   '| 样例 | 状态 | 原图 | 原样式（PPT） | 新样式（PPT） |','|---|---|---|---|---|']
            for j in jobs:
                key=j['name'];folder=out/key
                imgs=[f'![]({key}/{p})' if (folder/p).exists() else '等待' for p in ['input/clean.png','before/render.png','final/render.png']]
                lines.append('| '+' | '.join([f'[{key}]({key}/status.json)',states.get(key,{}).get('state','pending'),*imgs])+' |')
            lines+=['','[选择与来源](selection.json) · [冻结代码](code_manifest.json) · [进度](progress.json)','']
            (out/'RESULTS.md').write_text('\n'.join(lines))
    def work(job):
        folder=out/job['name'];folder.mkdir(parents=True,exist_ok=True);case=source/'cases'/job['case']
        clean=Image.open(case/'clean.png').convert('RGB');started=time.time();report(job['name'],dict(state='running'))
        if job['mode']=='style':
            scene=read(source/job['rows']) if job.get('rows') else read(case/'scene.json')['arrows']
            rows=[r for r in scene if r['id'] in job['ids']]
            if len(rows)!=len(job['ids']):raise ValueError('Selected row IDs missing')
            crop,view=style_pipeline.frame_for(rows[0],clean)
            local=[fit.transform(r,view,False) for r in rows];supports={}
            for r in local:
                masks=[ev.mask(p) for p in r.get('source_mask_paths',[])]
                if any(m.shape!=(crop.height,crop.width) for m in masks):raise ValueError('Mask frame mismatch')
                supports[r['id']]=np.logical_or.reduce(masks) if masks else None
            write(folder/'provenance.json',dict(source_rows=job.get('rows',str(case/'scene.json')),
                ids=job['ids'],clean_path=str(rows[0].get('local_input')),source_clean_sha256=sha(case/'clean.png'),
                no_new_sam=True,geometry_replay=True))
            result,issues=style_pipeline.recover(crop,local,supports,folder,config,model,locks,view,clean.size,prepare_shapes=True)
            return dict(state='completed_with_issues' if issues else 'completed',paths=len(result),issues=issues,seconds=time.time()-started)
        group=case/'groups'/job['group'];meta=read(group/'prepared_observation.json')
        shutil.copytree(group/'input',folder/'input',dirs_exist_ok=True)
        crop=Image.open(folder/'input/clean.png').convert('RGB');view=meta['view']
        hints=read(case/'global_hints.json');plan=read(case/'observation_plan.json')
        named=next(g['sources'] for g in plan['groups'] if g['id']==job['group'])
        anchors=[h for h in hints if h['id'] in named];selected=[]
        # Same retrieval as local_sampling: keep related and previously unowned
        # evidence, with the existing representative containment suppression.
        for c in meta['candidates']:
            related=any(c.get('scores',{}).get(g,0)>0 for g in named)
            if related or c.get('best_global') is None:
                selected.append(dict(c,mask=ev.mask(c['path'])))
        kept=[]
        for c in sorted(selected,key=lambda r:-r['native_area']):
            if any((c['mask']&p['mask']).sum()/max(1,c['native_area'])>=config['duplicate_containment'] for p in kept):continue
            kept.append(c)
        proposals,info=semantic_geometry.prepare(crop,anchors,kept,folder)
        context_size=Image.open(folder/'input/context.png').size
        prompt=semantic_geometry.prompt(named,crop.size,proposals,info,'',context_size,ev.crop_in_context(view,context_size))
        images=[('CONTEXT',folder/'input/context.png'),('CLEAN',folder/'input/clean.png'),
                ('GLOBAL HINTS',folder/'input/global_hints.png'),('MASKS + D ENDPOINTS',folder/'input/proposals.png')]
        images=target_locator.apply(prompt,images,anchors,view,crop,folder)
        def validate_identity(a):
            if config.get('identity_source_mode')=='observation':
                import observation_identity
                a=observation_identity.answer(a,named)
            return semantic_geometry.validate(a,crop.size,named,proposals,[c['id'] for c in kept],kept,context_size)
        if config.get('identity_source_mode')=='observation':
            import observation_identity
            prompt=observation_identity.prompt(prompt)
        answer=model.ask(folder/'identity',prompt,images,validate_identity)
        write(folder/'decision.json',answer)
        result=dict(state='completed',routes=len(answer['routes']),geometry_calls_not_run=True,
                    before_selected=len(meta['selected_ids']),after_selected=len(kept),excluded=answer['excluded'],uncertain=answer['uncertain'],seconds=time.time()-started)
        write(folder/'status.json',result);return result
    report()
    with ThreadPoolExecutor(max_workers=config.get('case_workers',8)) as pool:
        futures={pool.submit(work,j):j for j in jobs}
        for f in as_completed(futures):
            job=futures[f]
            try:result=f.result()
            except Exception as exc:
                result=dict(state='failed',error=str(exc));write(out/job['name']/'error.json',dict(error=str(exc),traceback=traceback.format_exc()))
            report(job['name'],result);print(job['name'],result['state'],flush=True)
    write(out/'finished.json',dict(jobs=states,visual_accuracy_verified=False))


if __name__=='__main__':main()
