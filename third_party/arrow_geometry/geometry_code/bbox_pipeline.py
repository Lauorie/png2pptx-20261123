"""Bbox handoff -> per-target geometry -> omissions -> reconciliation -> style."""
import copy,hashlib,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import bbox_inputs,bbox_geometry
from transport.common import read_json as read,write_json as write

def run_case(cid,config,out,model,locks,notify,prepare_only=False):
    import pipeline,coverage_stage,scope_review,style_pipeline,recall_review,reconcile_stage
    from lib import native
    out=Path(out);clean,plan=bbox_inputs.load_case(cid,out);tasks=plan['tasks'];results=[];issues=list(plan['issues'])
    deferred=plan.get('deferred_candidates',[])
    if deferred and not prepare_only and not config.get('post_geometry_missing_enabled',False):
        raise ValueError('Deferred localization candidates require post_geometry_missing_enabled')
    write(out/'observation_plan.json',dict(provider='bbox_handoff',groups=[dict(id='G%03d'%i,sources=[t['id']]) for i,t in enumerate(tasks,1)]))
    folders={t['id']:out/'groups'/('G%03d'%i) for i,t in enumerate(tasks,1)}
    def progress(stage,**extra):notify(cid,dict(stage=stage,groups=len(tasks),ended=len(results),failures=len(issues),**extra))
    def recover(t,folder):
        try:return bbox_geometry.recover_target(clean,t,folder,model,config,locks,prepare_only)
        except Exception as exc:
            issue=dict(stage='bbox_geometry',target=t['id'],error=str(exc))
            write(folder/'error.json',dict(issue,traceback=traceback.format_exc()))
            return dict(group=t['id'],state='failed',rows=[],issues=[issue],discovery_task=t)
    progress('bbox_geometry_preparing' if prepare_only else 'bbox_geometry')
    with ThreadPoolExecutor(max_workers=config.get('task_workers',16)) as pool:
        jobs={pool.submit(recover,t,folders[t['id']]):t for t in tasks}
        for f in as_completed(jobs):
            results.append(f.result());progress('bbox_geometry_preparing' if prepare_only else 'bbox_geometry')
    results.sort(key=lambda r:r['group'])
    if prepare_only:
        summary=dict(case=cid,state='prepared',discovery_tasks=len(tasks),issues=[e for r in results for e in r.get('issues',[])])
        write(out/'prepared.json',summary);return summary
    # Only after parallel initial requests: broken owners cannot suppress parents.
    failed={r['group'] for r in results if not r.get('rows')}
    retry=[]
    for t in tasks:
        result=next(r for r in results if r['group']==t['id'])
        dependent=any(r.get('exclusion_verified') and r.get('owner_id') in failed for r in t.get('readonly',[]))
        if result['state'] in ('failed','unresolved') or dependent:
            adjusted=copy.deepcopy(t)
            for r in adjusted.get('readonly',[]):
                if r.get('owner_id') in failed:r['exclusion_verified']=False
            retry.append(adjusted)
    if config.get('bbox_retry_unresolved',True) and retry:
        progress('bbox_dependency_recovery',targets=len(retry))
        with ThreadPoolExecutor(max_workers=config.get('task_workers',16)) as pool:
            updates=list(pool.map(lambda t:recover(t,folders[t['id']]/'recovery_01'),retry))
        table={r['group']:r for r in updates}
        # Preserve good original geometry when a dependency recheck fails.
        for i,r in enumerate(results):
            new=table.get(r['group'])
            if new and (new.get('rows') or not r.get('rows')):results[i]=new
            elif new:issues.append(dict(stage='bbox_dependency_recovery',target=r['group'],retained_previous=True,reason='Dependency recheck returned no replacement geometry'))
        write(out/'dependency_recovery.json',dict(failed_owner_ids=sorted(failed),rechecked=[t['id'] for t in retry]))
    issues.extend(e for r in results for e in r.get('issues',[]))
    write(out/'initial_geometry_results.json',results)
    with locks['cpu']:pipeline.publish(clean,results,out)
    recall_review.snapshot(out,'continuation')
    if config.get('post_geometry_missing_enabled',False):
        import post_geometry_missing
        results,errors=post_geometry_missing.run(clean,results,out/'post_geometry_missing',config,model,locks,cid,
                                               notify=progress,deferred_candidates=deferred)
        issues.extend(errors)
    elif config.get('coverage_rounds',0):
        progress('coverage')
        results,errors,_=coverage_stage.run(clean,results,out,config,model,locks,evidence_pool=[],notify=progress)
        issues.extend(errors)
    if config.get('scope_review',False):
        progress('scope_review');results,errors=scope_review.run(clean,results,out/'scope_review',model,locks);issues.extend(errors)
    results,errors=style_pipeline.prepare_case_geometry(clean,results,out,config,model,locks);issues.extend(errors)
    write(out/'before_reconciliation.json',results)
    if config.get('reconcile_enabled',True):
        progress('reconciliation')
        results,errors=reconcile_stage.run(clean,results,out/'reconciliation',config,model,locks);issues.extend(errors)
    write(out/'geometry_results.json',results)
    with locks['cpu']:pipeline.publish(clean,results,out)
    if config.get('pipeline_mode','full')!='geometry':
        progress('style_recovery');results,errors=style_pipeline.finish_case(clean,results,out,config,model,locks);issues.extend(errors)
    with locks['cpu']:
        rows=pipeline.publish(clean,results,out)
        if config.get('native_render',True):native.render(out/'arrows.pptx',out/'native',clean.size)
    unresolved=[r['group'] for r in results if r.get('state') in ('failed','unresolved')]
    summary=dict(case=cid,state='completed_with_issues' if issues or unresolved else 'completed',
        discovery_provider='bbox_handoff',discovery_tasks=len(tasks),draw_paths=len(rows),targets=len({r['parent_id'] for r in rows}),
        issues=issues,unresolved_targets=unresolved,mask_identity_executed=False,
        geometry_only=config.get('pipeline_mode')=='geometry',accuracy_measured=False)
    write(out/'completed.json',summary)
    (out/'RESULTS.md').write_text('# '+cid+'：bbox发现到最终重建\n\n状态：'+summary['state']+'\n\n| 原图 | 几何/样式结果 | 叠图 |\n|---|---|---|\n| ![](clean.png) | ![](native/styled.png) | ![](result/overlap.png) |\n\n[发现交接](discovery_plan.json) · [去重审计](reconciliation/audit.json) · [问题](completed.json) · [PPT](arrows.pptx)\n')
    progress('completed',paths=len(rows));return summary

