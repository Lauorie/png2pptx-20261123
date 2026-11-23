"""Parallel snapshot repairs; shared write conflicts are returned to the model."""
import copy,hashlib,json,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import shared_repair as repair
from transport.common import write_json as write


def box(task):return task.get('focus_bbox',task['bbox'])


def overlaps(a,b):return max(a[0],b[0])<min(a[2],b[2]) and max(a[1],b[1])<min(a[3],b[3])


def bundle(tasks,rows):
    """Group conflicting observation scopes, never infer physical connections."""
    resources=[{r.get('parent_id',r['id']) for r in rows if repair.visible(r,box(t))} for t in tasks]
    pending=list(range(len(tasks)));groups=[]
    while pending:
        members=[pending.pop(0)];changed=True
        while changed:
            changed=False
            for i in list(pending):
                if any(resources[i]&resources[j] or overlaps(box(tasks[i]),box(tasks[j])) for j in members):
                    pending.remove(i);members.append(i);changed=True
        if len(members)==1:groups.append(copy.deepcopy(tasks[members[0]]));continue
        foci=[dict(bbox=list(box(tasks[i])),description=tasks[i]['description']) for i in members]
        bounds=[min(x['bbox'][0] for x in foci),min(x['bbox'][1] for x in foci),max(x['bbox'][2] for x in foci),max(x['bbox'][3] for x in foci)]
        groups.append(dict(bbox=bounds,focus_regions=foci,
            description='Check the separately listed suspected missing portions; they are not assumed to form one network.',
            bundled_tasks=members))
    return groups


def execute(clean,tasks,current,phase,out,config,model,locks,evidence_pool=None,notify=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);tasks=bundle(tasks,current);snapshot=copy.deepcopy(current)
    write(out/'tasks.json',tasks);write(out/'snapshot_ids.json',[r['id'] for r in snapshot]);results={};issues=[]
    def one(i,task):
        key=hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest()[:10];pid=f'{phase}{i+1}'
        folder=out/(pid+'_'+key)
        try:
            _,patch=repair.recover(clean,task,snapshot,pid,folder,config,model,locks,evidence_pool=evidence_pool)
            return dict(task=task,pid=pid,folder=folder,patch=patch)
        except Exception as exc:
            error=dict(stage=phase,task=task,error=str(exc));write(folder/'error.json',dict(**error,traceback=traceback.format_exc()))
            return dict(error=error)
    with ThreadPoolExecutor(max_workers=config['task_workers']) as pool:
        futures={pool.submit(one,i,t):i for i,t in enumerate(tasks)}
        for f in as_completed(futures):
            results[futures[f]]=f.result()
            if notify:notify(len(results),len(tasks))
    after=copy.deepcopy(current);patches=[];removed=set();conflicts=[]
    for i in sorted(results):
        item=results[i]
        if 'error' in item:issues.append(item['error']);continue
        patch=item['patch'];conflict=removed.intersection(patch['replace'])
        if conflict:
            # This is a stale-write conflict on exact IDs, not a geometry gate.
            # Never arbitrarily drop or merge a model's edits.
            write(item['folder']/'write_conflict.json',dict(ids=sorted(conflict),action='Ask again with current committed paths'))
            try:
                _,patch=repair.recover(clean,item['task'],after,item['pid'],item['folder']/'rebase',config,model,locks,evidence_pool=evidence_pool)
                for field in ['repair_calls','geometry_calls','head_calls']:patch[field]=patch.get(field,0)+item['patch'].get(field,0)
                conflicts.append(dict(task=i,ids=sorted(conflict),resolved_by='fresh_model_decision'))
            except Exception as exc:
                issues.append(dict(stage=phase,task=item['task'],error=str(exc),conflict_ids=sorted(conflict)));continue
        removed.update(patch['replace']);after=[r for r in after if r['id'] not in set(patch['replace'])]+patch['rows'];patches.append(patch)
    write(out/'parallel_summary.json',dict(tasks=len(tasks),committed=len(patches),issues=issues,write_conflicts=conflicts,
        snapshot_shared=True,geometry_based_final_gate=False))
    return patches,issues
