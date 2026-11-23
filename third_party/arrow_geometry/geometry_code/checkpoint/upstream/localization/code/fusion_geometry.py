"""Paired geometry via frozen mainline components; unchanged tasks run once."""
import argparse
import copy
import hashlib
import json
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def canonical_task(task):
    """Only provenance/ordinal fields excluded; all geometry inputs remain bound."""
    ignored={'target_id','display_id','region_id','source_image','target_image',
             'observation_ids','bbox_global_input_px','bbox_native_unclipped'}
    return {k:v for k,v in task.items() if k not in ignored}


def plan_pair(before, after):
    tasks=[];index={};arms={};maps={}
    for arm,handoff in [('before',before),('after',after)]:
        arms[arm]=[];maps[arm]=[]
        for item in handoff['tasks']:
            key=json.dumps(canonical_task(item),sort_keys=True,ensure_ascii=False,separators=(',',':'))
            if key not in index:
                tid=f'K{len(tasks)+1:04d}';index[key]=tid
                tasks.append({**copy.deepcopy(item),'target_id':tid,'display_id':tid})
            tid=index[key];arms[arm].append(tid)
            maps[arm].append({'original_target_id':item['target_id'],'region_id':item['region_id'],
                              'shared_geometry_task':tid,'input_sha256':hashlib.sha256(key.encode()).hexdigest()})
    return tasks,arms,maps


def main(root):
    sys.path[:0]=[str(root/'geometry_code'),str(root/'geometry_code/lib')]
    from PIL import Image
    import bbox_geometry,bbox_inputs,pipeline,protocol,viz
    from scene_snapshot import canonical_rows
    from lib import native
    from transport.common import read_json as read,write_json as write
    from transport.model import Model
    cfg=read(root/'geometry_config.json')
    # Same visible-only contract in BOTH arms; no path/pixel post-processing.
    protocol.OCCLUSION=('Reconstruct visible stroke portions only. Do not infer or draw hidden connectors behind text, nodes or opaque objects. '
        'Separate visible portions may be separate paths. If only the local crop truncates a stroke, use the whole-image context to recover its actually visible continuation. '
        'Crossings alone are not junctions; a group region may contain multiple independent paths.')
    model=Model(cfg,run_dir=root,system=protocol.SYSTEM)
    locks={'cpu':threading.Semaphore(cfg.get('cpu_workers',4)),'sam':threading.Semaphore(cfg.get('sam_workers',2))}
    def case(cid):
        out=root/'cases'/cid
        if (out/'geometry_finished.json').exists():return read(out/'geometry_finished.json')
        clean=Image.open(root/'inputs'/cid/'source.png').convert('RGB')
        before=read(out/'before_reconstruction_handoff.json');after=read(out/'reconstruction_handoff.json')
        tasks,arms,maps=plan_pair(before,after)
        normalized=bbox_inputs.normalize({'case_id':cid,'tasks':tasks},clean.size,cid)
        write(out/'geometry_plan.json',{'arms':maps,'unique_tasks':normalized['tasks'],
              'shared_task_count':len(set(arms['before'])&set(arms['after'])),
              'note':'Exact unchanged task inputs share one result, not independent replicates.'})
        for arm in arms:
            folder=out/arm;folder.mkdir(parents=True,exist_ok=True);clean.save(folder/'clean.png')
            write(folder/'discovery_provenance.json',{'provider':'model_led_regions_v3','arm':arm,'paired':True})
        results={};started=time.monotonic();last_preview=0
        def snapshot(final=False):
            errors=[]
            for arm,ordered in arms.items():
                selected=[copy.deepcopy(results[k]) for k in ordered if k in results]
                folder=out/arm
                write(folder/'geometry_results.json',selected)
                rows=canonical_rows(selected)
                try:
                    with locks['cpu']:
                        if final:
                            pipeline.publish(clean,selected,folder)
                            if not (folder/'native_geometry/styled.png').exists():
                                native.render(folder/'arrows.pptx',folder/'native_geometry',clean.size)
                                # Composite native RGBA over CLEAN, not a white background.
                                import shutil
                                shutil.copy2(folder/'native_geometry/styled.png',folder/'result/styled.png')
                                from native_foreground import render_layer
                                foreground=render_layer(folder/'native_geometry/styled.svg',clean.size,folder/'native_geometry')
                                overlay=clean.convert('RGBA');overlay.alpha_composite(foreground)
                                overlay.convert('RGB').save(folder/'result/overlap.png')
                        else:
                            # Cheap interim line/path view; final uses validated native renderer.
                            for i,r in enumerate(rows,1):r['label']=r.get('parent_id',str(i))
                            viz.render(folder/'clean.png',rows,folder/'result')
                    if final:write(folder/'completed.json',{'paths':len(rows),'tasks':len(ordered),'unresolved':[k for k in ordered if not results.get(k,{}).get('rows') and results.get(k,{}).get('state')!='completed'],
                                  'issues':[e for r in selected for e in r.get('issues',[])],'native_rendered':True})
                except Exception as exc:
                    issue={'stage':'final_render' if final else 'partial_preview','arm':arm,'error':str(exc),'traceback':traceback.format_exc()}
                    write(folder/('render_error.json' if final else 'preview_error.json'),issue);errors.append(issue)
            return errors
        def recover(task):
            folder=out/'geometry_tasks'/task['id']
            if (folder/'replay_result.json').exists():return read(folder/'replay_result.json')
            try:result=bbox_geometry.recover_target(clean,task,folder,model,cfg,locks)
            except Exception as exc:
                result={'group':task['id'],'state':'failed','rows':[],'issues':[{'stage':'geometry','error':str(exc),'traceback':traceback.format_exc()}]}
            write(folder/'replay_result.json',result)
            return result
        write(out/'geometry_progress.json',{'stage':'running','ended':0,'total':len(tasks),'failed':0})
        with ThreadPoolExecutor(max_workers=cfg['task_workers']) as pool:
            futures={pool.submit(recover,t):t['id'] for t in normalized['tasks']}
            for f in as_completed(futures):
                tid=futures[f];results[tid]=f.result()
                write(out/'geometry_progress.json',{'stage':'running','ended':len(results),'total':len(tasks),
                      'failed':sum(r['state'] in ('failed','unresolved') for r in results.values()),'seconds':time.monotonic()-started})
                if time.monotonic()-last_preview>30:snapshot();last_preview=time.monotonic()
        render_errors=snapshot(True)
        summary={'case':cid,'unique_tasks':len(tasks),'shared_tasks':len(set(arms['before'])&set(arms['after'])),
                 'before_tasks':len(arms['before']),'after_tasks':len(arms['after']),
                 'failed_tasks':[k for k,r in results.items() if r['state'] in ('failed','unresolved')],
                 'issues':[e for r in results.values() for e in r.get('issues',[])]+render_errors,'seconds':time.monotonic()-started}
        write(out/'geometry_finished.json',summary)
        write(out/'geometry_progress.json',{'stage':'finished','ended':len(results),'total':len(tasks),'failed':len(summary['failed_tasks']),'seconds':summary['seconds']})
        print(cid,'geometry_finished',len(tasks),'failed',len(summary['failed_tasks']),flush=True)
        return summary
    summaries=[]
    with ThreadPoolExecutor(max_workers=cfg['case_workers']) as pool:
        futures={pool.submit(case,c):c for c in read(root/'cases.json')}
        for f in as_completed(futures):
            try:summaries.append(f.result())
            except Exception as exc:
                cid=futures[f];write(root/'cases'/cid/'geometry_fatal.json',{'error':str(exc),'traceback':traceback.format_exc()})
                print(cid,'GEOMETRY_FATAL',str(exc),flush=True)
    write(root/'geometry_finished.json',{'cases':summaries,'total':len(read(root/'cases.json'))})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    main(p.parse_args().run_dir.resolve())
