"""Source-backed local reconciliation after geometry/coverage, before style."""
import copy,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import evidence as ev,segmented_geometry as sg
import reconcile_geometry as rg
from reconcile_scopes import propose
from bbox_geometry import preview_style
from lib import fit
from transport.common import write_json as write

def run(clean,results,out,config,model,locks):
    results=copy.deepcopy(results);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    allrows=[r for g in results for r in g.get('rows',[])]
    rows=[r for r in allrows if r['kind']!='block_arrow']
    cfg={'scope_distance_at_2048':12.,'scope_sample_step_at_2048':2.,'scope_tangent_cosine':.92,**config}
    plan=propose(rows,clean.size,cfg);write(out/'scope_plan.json',plan)
    index={r['id']:r for r in rows};patches={};issues=[];records=[]
    def work(item):
        i,ids=item;gid='reconcile_%03d'%i;folder=out/gid;original=[index[x] for x in ids]
        if len(ids)>config.get('reconcile_max_paths',8):
            return ids,original,dict(group=gid,decision='deferred',reason='Oversized overlap component; no members truncated'),[dict(stage='reconciliation_scope',group=gid,paths=len(ids))]
        doc,mapping=rg.global_document(original);xy=np.asarray([n[1:] for n in doc['nodes']]);lo=xy.min(0);hi=xy.max(0)
        center=(lo+hi)/2;span=np.maximum(hi-lo,config.get('reconcile_min_extent',25))*config.get('reconcile_context_scale',1.8)
        with locks['cpu']:
            crop,frame=ev.frame(clean,[np.r_[center-span/2,center+span/2]],folder/'input',
                {**config,'display_side':config.get('reconcile_display_side',1080),'min_crop_side':config.get('reconcile_min_extent',25)})
            local_doc=rg.transform_document(doc,frame['global_to_view'])
            local=[fit.transform(r,frame,False) for r in original]
            bb=frame['bbox_global'];context=[]
            for r in allrows:
                if r['id'] in ids:continue
                b=rg.bounds([r])
                if b[0]<=bb[2] and b[2]>=bb[0] and b[1]<=bb[3] and b[3]>=bb[1]:context.append(fit.transform(r,frame,False))
            rg.labeled_current(crop,local,context,folder/'input',config.get('geometry_preview_width_original_px',1.2)*frame['scale'][0])
        write(folder/'input/current.json',local_doc);write(folder/'input/path_map.json',mapping)
        write(folder/'before_rows_global.json',original);write(folder/'scope.json',dict(edit_ids=ids,frame=frame,readonly_ids=[r['id'] for r in context]))
        request=folder/'geometry_critic/revision_00/request'
        answer=model.ask(request,rg.prompt(local_doc,crop.size),
            [('CLEAN source',folder/'input/clean.png'),('CURRENT editable paths; gray context readonly',folder/'input/current.png')],lambda a:rg.validate(a,crop.size))
        write(folder/'answer.json',answer)
        errors=[]
        if answer['decision']=='refine':
            base=dict(id=gid,parent_id=gid,label=gid,source_masks=[],source_mask_paths=[],geometry_frame=frame,
                geometry_request=str(request),local_input=str(folder/'input/clean.png'),reconciles=ids,
                source_discovery_ids=sorted({r.get('source_discovery_id',r.get('parent_id',r['id'])) for r in original}))
            local=sg.geometry(answer['geometry'],{'R1':base})
            for r in local:preview_style(r,config.get('geometry_preview_width_original_px',1.2)*frame['scale'][0])
            revised=[fit.transform(r,frame) for r in local]
        else:
            revised=original
            if answer['decision']=='uncertain':errors.append(dict(stage='reconciliation',group=gid,reason=answer.get('reason')))
        write(folder/'after_rows_global.json',revised)
        with locks['cpu']:rg.native_views(crop,local,folder/'after',config.get('geometry_preview_width_original_px',1.2)*frame['scale'][0])
        return ids,revised,dict(group=gid,decision=answer['decision'],before=len(ids),after=len(revised)),errors
    with ThreadPoolExecutor(max_workers=config.get('task_workers',16)) as pool:
        futures={pool.submit(work,item):item for item in enumerate(plan['groups'],1)}
        for f in as_completed(futures):
            i,ids=futures[f]
            try:
                ids,revised,record,errors=f.result();records.append(record);issues.extend(errors)
                if record['decision']=='refine':patches[tuple(ids)]=revised
            except Exception as exc:
                errors=dict(stage='reconciliation',ids=ids,error=str(exc),retained_original=True);issues.append(errors)
                write(out/('reconcile_%03d'%i)/'error.json',dict(errors,traceback=traceback.format_exc()))
    changed={i for ids in patches for i in ids};replacement=[r for patch in patches.values() for r in patch]
    for group in results:group['rows']=[r for r in group.get('rows',[]) if r['id'] not in changed]
    if replacement:results.append(dict(group='reconciliation',state='completed',rows=replacement,issues=[]))
    old={r['id']:r for r in allrows if r['id'] not in changed}
    new={r['id']:r for g in results for r in g.get('rows',[]) if r['id'] in old}
    if old!=new:raise ValueError('Reconciliation changed unselected paths')
    write(out/'audit.json',dict(records=records,issues=issues,edited_ids=sorted(changed),nonedited_rows_preserved=True))
    write(out/'results.json',results);return results,issues

