"""Shared omission stage for full runs and continuation-checkpoint runs."""
import copy
import hashlib
import json
import shutil
import traceback
from pathlib import Path

import evidence as ev
import parallel_repairs
import protocol
import shared_repair
from transport.common import write_json as write, sha256 as sha


def discover(clean,current,out,config,model,locks):
    from pipeline import inventory
    out=Path(out);folder=out/'coverage';inputs=folder/'input';inputs.mkdir(parents=True,exist_ok=True)
    mode=config.get('coverage_input_mode','two_images')
    if mode not in ('two_images','boxes_and_inventory'):
        raise ValueError('Unknown coverage_input_mode: '+str(mode))
    rows=inventory(current)
    # Keep locators for audit and human inspection even when not sent to GLM.
    write(inputs/'inventory.json',rows)
    if (out/'result/boxes.png').exists():shutil.copy2(out/'result/boxes.png',inputs/'already_reconstructed.png')
    with locks['cpu']:
        shared_repair.draw_current(clean,current,ev.identity_view(clean.size),inputs/'actual_geometry.png')
    if config.get('coverage_mode')=='tiles':
        import tile_coverage
        answer=tile_coverage.discover(clean,current,folder/'tiles',model,config,locks)
    else:
        images=[('ORIGINAL intact reference; desired arrows are visible here',out/'clean.png')]
        if mode=='boxes_and_inventory':
            images += [('BOXES: locators only; a box does not imply complete recovery',inputs/'already_reconstructed.png'),
                       ('ACTUAL GEOMETRY: recovered paths on ORIGINAL, not ground truth',inputs/'actual_geometry.png')]
        else:
            images += [('RECOVERED: blue recovered paths; gray is source only',inputs/'actual_geometry.png')]
        prompt=protocol.coverage(list(clean.size),rows,input_mode=mode)
        write(inputs/'images.json',[dict(label=label,path=str(path.resolve()),sha256=sha(path)) for label,path in images])
        write(folder/'input_policy.json',dict(mode=mode,images_sent=len(images),numeric_inventory_sent='already_reconstructed' in prompt,
            inventory_archived=True,scope='global region nomination; not geometry or identity ownership'))
        answer=model.ask(folder,prompt,images,lambda a:protocol.regions(a,list(clean.size)))
    write(out/'coverage_status.json',dict(state=answer.get('state','completed'),regions=len(answer['regions']),input_mode=mode))
    write(out/'missing_regions.json',answer)
    return answer


def run(clean,results,out,config,model,locks,evidence_pool=None,notify=None):
    """Discover, verify/rebuild locally, and commit explicit model patches."""
    from pipeline import publish
    from scene_snapshot import canonical_rows
    out=Path(out);results=copy.deepcopy(results);current=canonical_rows(results)
    issues=[];patches=[]
    def progress(stage,**kw):
        if notify:notify(stage,**kw)
    progress('missing_only')
    try:
        answer=discover(clean,current,out,config,model,locks)
    except Exception as exc:
        issue=dict(stage='coverage',error=str(exc));issues.append(issue)
        write(out/'coverage_status.json',dict(state='failed',error=str(exc)))
        write(out/'coverage/error.json',dict(**issue,traceback=traceback.format_exc()))
        return results,issues,dict(added=[],replaced=[],regions=None)
    issues.extend(dict(stage='coverage_tile',**e) for e in answer.get('errors',[]))
    tasks=answer['regions'];write(out/'missing_queue.json',tasks)
    progress('missing',repair_ended=0,candidate_regions=len(tasks))
    if config.get('parallel_repairs',False):
        patches,errors=parallel_repairs.execute(clean,tasks,current,'missing',out/'missing',config,model,locks,evidence_pool,
            notify=lambda n,total:progress('missing',repair_ended=n,repair_total=total))
        issues.extend(errors)
        for patch in patches:results=shared_repair.merge_results(results,patch)
    else:
        for i,task in enumerate(tasks,1):
            key=hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest()[:10];pid=f'missing{i}'
            folder=out/'missing'/(pid+'_'+key)
            try:
                _,patch=shared_repair.recover(clean,task,current,pid,folder,config,model,locks,evidence_pool=evidence_pool)
                results=shared_repair.merge_results(results,patch);patches.append(patch)
                with locks['cpu']:current=publish(clean,results,out)
            except Exception as exc:
                issue=dict(stage='missing',task=task,error=str(exc));issues.append(issue)
                write(folder/'error.json',dict(**issue,traceback=traceback.format_exc()))
            progress('missing',repair_ended=i,repair_total=len(tasks))
    for patch in patches:
        issues.extend(dict(e,phase='missing',group=patch['group']) for e in patch.get('issues',[]))
        if patch.get('status')=='unresolved':issues.append(dict(stage='missing',group=patch['group'],error='Model left the target unresolved'))
    changes=dict(added=[r['id'] for p in patches for r in p.get('rows',[])],
                 replaced=sorted({rid for p in patches for rid in p.get('replace',[])}),regions=len(tasks))
    write(out/'coverage_changes.json',changes)
    with locks['cpu']:publish(clean,results,out)
    return results,issues,changes
