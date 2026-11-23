"""Global hints -> crop SAM evidence -> semantic structure -> mask geometry.

One semantic request selects isolated mask proposals or supplies structural
keypoints. No skeleton, gap graph or independent endpoint override is executed.
"""
import copy,sys,traceback,shutil,hashlib,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parent/'lib'))
import fit,style,viz,native
import evidence as ev,graph as gr,visual,mask_grouping,protocol,local_sampling
import contracts as ct
import straight_mask,shared_repair,observation_pool,parallel_repairs,recall_review
from transport.common import write_json as write,read_json as read,sha256 as sha


# One main implementation. Historical graph-based recovery remains only in frozen runs.
from group_recovery import group_task,prepare_group


def style_rows(crop,rows,support,adaptive_heads=True):
    import style_windows
    for row in rows:row['style']=style_windows.recover(np.asarray(crop),support,row)
    return rows


def publish(clean,results,out):
    from scene_snapshot import canonical_rows
    allrows=canonical_rows(results);parents=sorted({r['parent_id'] for r in allrows})
    labels={key:f'A{i}' for i,key in enumerate(parents,1)}
    for r in allrows:r['label']=labels[r['parent_id']]
    write(out/'scene.json',dict(size=list(clean.size),arrows=allrows,heads_source='glm',normal_geometry_source='local_sam_mask',
        global_sam_role='identity hints and observation scope',geometry_repaired_rows=sum(r['coordinate_source'] in ('glm_keypoints','glm_segmented_points') for r in allrows)))
    write(out/'pending_style.json',[r for r in allrows if style.pending(r)])
    viz.render(out/'clean.png',allrows,out/'result');native.write(clean.size,allrows,[r['style'] for r in allrows],out/'arrows.pptx')
    locator=clean.copy();d=ImageDraw.Draw(locator)
    for key in parents:
        pts=np.concatenate([fit.points(r) for r in allrows if r['parent_id']==key]);b=[*pts.min(axis=0),*pts.max(axis=0)]
        d.rectangle(b,outline='#c00080',width=2);visual.label(d,b[:2],labels[key],'#c00080',locator)
    locator.save(out/'result/boxes.png');return allrows


def inventory(rows):
    result=[]
    for key in sorted({r['parent_id'] for r in rows}):
        members=[r for r in rows if r['parent_id']==key];pts=np.concatenate([fit.points(r) for r in members])
        result.append(dict(id=members[0]['label'],bbox=np.round(np.r_[pts.min(axis=0),pts.max(axis=0)],1).tolist(),
            kinds=sorted({r['kind'] for r in members}),heads=[r['heads'] for r in members]))
    return result


def run_case(cid,config,out,model,locks,notify,prepare_only=False):
    if config.get('style_resume_from'):
        return resume_style_case(cid,config,out,model,locks,notify)
    if config.get('coverage_resume_from'):
        if prepare_only:raise ValueError('Coverage continuation cannot be prepare-only')
        return resume_coverage_case(cid,config,out,model,locks,notify)
    out=Path(out);issues=[];results=[]
    with locks['cpu']:
        clean,hints,_=ev.load_case(cid,config,out);groups,relations=mask_grouping.group(hints,clean.size,config['observation_grouping'])
    plan=[dict(id=f'G{i:03d}',sources=[r['id'] for r in g]) for i,g in enumerate(groups,1)]
    group_of={s:p['id'] for p in plan for s in p['sources']};write(out/'observation_plan.json',dict(groups=plan,relations=relations))
    def progress(stage,**kw):notify(cid,dict(stage=stage,groups=len(groups),ended=len(results)+len(issues),failures=len(issues),**kw))
    progress('local_sam')
    prepared={};pool_evidence=None
    if config.get('prepare_all_observations',False):
        with ThreadPoolExecutor(max_workers=config['task_workers']) as pool:
            jobs={pool.submit(prepare_group,clean,g,hints,group_of,p['id'],out/'groups'/p['id'],config,locks):p for g,p in zip(groups,plan)}
            for f in as_completed(jobs):
                p=jobs[f]
                try:prepared[p['id']]=f.result()
                except Exception as exc:
                    issues.append(dict(group=p['id'],stage='local_sam',error=str(exc)))
                    write(out/'groups'/p['id']/'error.json',dict(error=str(exc),traceback=traceback.format_exc()))
                progress('local_sam',prepared_groups=len(prepared))
        if config.get('shared_mask_context',False):
            with locks['cpu']:pool_evidence=observation_pool.build(list(prepared.values()),out)
    with ThreadPoolExecutor(max_workers=config['task_workers']) as pool:
        # Small independent scopes first; this changes scheduling only, not IDs,
        # grouping, prompts or the algorithm applied to any case.
        jobs={pool.submit(group_task,clean,g,hints,group_of,p['id'],out/'groups'/p['id'],config,model,locks,prepare_only,
                          prepared=prepared.get(p['id']),evidence_pool=pool_evidence):(g,p)
              for g,p in sorted(zip(groups,plan),key=lambda pair:len(pair[0]))
              if not config.get('prepare_all_observations',False) or p['id'] in prepared}
        for f in as_completed(jobs):
            g,p=jobs[f]
            try:results.append(f.result())
            except Exception as exc:
                issues.append(dict(group=p['id'],sources=p['sources'],error=str(exc)))
                write(out/'groups'/p['id']/'error.json',dict(error=str(exc),traceback=traceback.format_exc()))
            if not prepare_only and results and len(results)%4==0:
                with locks['cpu']:publish(clean,sorted(results,key=lambda x:x['group']),out)
                (out/'RESULTS.md').write_text('# '+cid+'：进行中\n\n以下是部分完成组，实例编号可能随完整结果更新。\n\n'
                    '| 原图 | 当前实例 | 当前样式 | 叠图 |\n|---|---|---|---|\n'
                    '| ![](clean.png) | ![](result/instances.png) | ![](result/styled.png) | ![](result/overlap.png) |\n')
            progress('preparing' if prepare_only else 'identity_geometry_style')
    results.sort(key=lambda x:x['group'])
    if prepare_only:
        summary=dict(case=cid,state='prepared' if not issues else 'prepared_with_issues',groups=len(groups),results=results,issues=issues)
        write(out/'prepared.json',summary);return summary
    with locks['cpu']:allrows=publish(clean,results,out)
    write(out/'initial_results.json',results)
    recall_review.snapshot(out,'initial')
    def repair_tasks(tasks,phase):
        nonlocal results,allrows
        write(out/(phase+'_queue.json'),tasks)
        if config.get('parallel_repairs',False):
            patches,errors=parallel_repairs.execute(clean,tasks,allrows,phase,out/phase,config,model,locks,pool_evidence,
                notify=lambda n,total:progress(phase,repair_ended=n,repair_total=total))
            issues.extend(errors)
            for patch in patches:results=shared_repair.merge_results(results,patch)
            with locks['cpu']:allrows=publish(clean,results,out)
            return
        # Within a figure, later requests must see earlier committed patches.
        # Different figures still run concurrently under the shared API limit.
        for i,task in enumerate(tasks,1):
            key=hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest()[:10]
            pid=f'{phase}{i}'
            folder=out/phase/(pid+'_'+key)
            try:
                _,patch=shared_repair.recover(clean,task,allrows,pid,folder,config,model,locks,evidence_pool=pool_evidence)
                results=shared_repair.merge_results(results,patch)
                with locks['cpu']:allrows=publish(clean,results,out)
            except Exception as exc:
                issues.append(dict(stage=phase,task=task,error=str(exc)))
                write(folder/'error.json',dict(error=str(exc),traceback=traceback.format_exc()))
            progress(phase,repair_ended=i,repair_total=len(tasks))
    followups=[f for r in results for f in r.get('followups',[])]
    if config.get('recover_failed_groups',False):
        # Explicitly unresolved observations remain work, not silent exclusions.
        bygroup={r['group']:r for r in results}
        for g,p in zip(groups,plan):
            result=bygroup.get(p['id'])
            failed=result is None or result.get('uncertain') or result.get('observation_incomplete') or any(x.get('stage')=='glm_geometry' for x in result.get('issues',[]))
            if failed:
                followups.append(dict(bbox=list(map(float,ev.bounds([r['box'] for r in g]))),
                    description='This observation failed or left visible linework unresolved. Compare CLEAN with existing paths and recover only missing visible portions.',
                    source_group=p['id'],reason='unresolved_observation'))
    if followups:repair_tasks(followups,'continuation')
    recall_review.snapshot(out,'continuation')
    if config['coverage_rounds']:
        import coverage_stage
        results,coverage_issues,_=coverage_stage.run(clean,results,out,config,model,locks,pool_evidence,notify=progress)
        issues.extend(coverage_issues)
    if config.get('scope_review',False):
        import scope_review
        progress('scope_review')
        results,review_issues=scope_review.run(clean,results,out/'scope_review',model,locks)
        issues.extend(review_issues)
        with locks['cpu']:publish(clean,results,out)
        recall_review.snapshot(out,'scope_review')
    if config.get('separate_style',False):
        import style_pipeline
        progress('shape_geometry')
        results,shape_issues=style_pipeline.prepare_case_geometry(clean,results,out,config,model,locks)
        issues.extend(shape_issues)
        write(out/'geometry_results.json',results)
        if config.get('pipeline_mode','full')!='geometry':
            progress('style_recovery')
            results,style_issues=style_pipeline.finish_case(clean,results,out,config,model,locks)
            issues.extend(style_issues)
    with locks['cpu']:allrows=publish(clean,results,out)
    rendered=False
    if config.get('native_render',False):
        with locks['cpu']:native.render(out/'arrows.pptx',out/'native',clean.size)
        rendered=True
    summary=dict(case=cid,state='completed_with_issues' if issues or any(r['state']!='completed' for r in results) else 'completed',
        groups=len(groups),targets=len({r['parent_id'] for r in allrows}),draw_paths=len(allrows),issues=issues,
        semantic_calls=sum(r.get('semantic_calls',0) for r in results),repair_calls=sum(r.get('repair_calls',0) for r in results),geometry_calls=sum(r.get('geometry_calls',0) for r in results),
        head_calls=sum(r.get('head_calls',0) for r in results),
        mask_geometry_calls=sum(r.get('mask_geometry_calls',0) for r in results),
        scope_review=read(out/'scope_review/audit.json') if config.get('scope_review',False) else None,
        coordinate_repair_rows=sum(r['coordinate_source'] in ('glm_keypoints','glm_segmented_points') for r in allrows),mask_geometry_rows=sum(r['coordinate_source']=='mask_no_skeleton' for r in allrows),
        native_ppt=str(out/'arrows.pptx'),native_rendered=rendered,accuracy_measured=False)
    write(out/'completed.json',summary)
    lines=[f'# {cid}: 几何与样式恢复','',f'状态 {summary["state"]}；目标 {summary["targets"]}，mask 几何 {summary["mask_geometry_rows"]} 段，GLM 分段几何 {summary["coordinate_repair_rows"]} 段。','',
        '| 原图 | 最终实例 | '+('实际 PPT' if rendered else '样式预览')+' | 叠图 |','|---|---|---|---|','| ![](clean.png) | ![](result/instances.png) | ![]('+('native/styled.png' if rendered else 'result/styled.png')+') | ![](result/overlap.png) |','',
        '[几何、样式与来源](scene.json) · [问题记录](completed.json) · [可编辑 PPT](arrows.pptx)','']
    for p in plan:lines.append(f'- [{p["id"]}: '+', '.join(p['sources'])+f'](groups/{p["id"]}/README.md)')
    lines+=['','[分阶段召回](RECALL.md)']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n');recall_review.write_case(out)
    progress(summary['state'],targets=summary['targets']);return summary


def resume_style_case(cid,config,out,model,locks,notify):
    """Restyle saved geometry without SAM, identity, geometry fitting or coverage."""
    import style_pipeline
    source=Path(config['style_resume_from'])/'cases'/cid;out=Path(out);out.mkdir(parents=True,exist_ok=True)
    checkpoint=source/'geometry_results.json'
    if checkpoint.exists():results=read(checkpoint)
    else:
        checkpoint=source/'scene.json'
        results=[dict(group='saved_geometry',state='completed',rows=read(checkpoint)['arrows'])]
    shutil.copy2(source/'clean.png',out/'clean.png');clean=Image.open(out/'clean.png').convert('RGB')
    write(out/'resume_provenance.json',dict(mode='style_only',source=str(checkpoint),sha256=sha(checkpoint),new_sam=False,new_identity=False))
    write(out/'geometry_results.json',results);notify(cid,dict(stage='style_recovery'))
    results,issues=style_pipeline.finish_case(clean,results,out,config,model,locks)
    with locks['cpu']:
        rows=publish(clean,results,out);native.render(out/'arrows.pptx',out/'native',clean.size)
    summary=dict(case=cid,state='completed_with_issues' if issues else 'completed',draw_paths=len(rows),issues=issues,
                 native_rendered=True,geometry_source=str(checkpoint),style_only=True,accuracy_measured=False)
    write(out/'completed.json',summary)
    (out/'RESULTS.md').write_text('# '+cid+'：独立样式重跑\n\n| 原图 | 实际 PPT |\n|---|---|\n| ![](clean.png) | ![](native/styled.png) |\n\n[样式问题](style_issues.json) · [固定几何来源](resume_provenance.json)\n')
    notify(cid,dict(stage=summary['state']));return summary


def resume_coverage_case(cid,config,out,model,locks,notify):
    """Resume at the same omission entry used by run_case, with provenance."""
    import coverage_stage
    source=Path(config['coverage_resume_from'])/'cases'/cid
    checkpoint=source/'stages/continuation/scene.json'
    saved=read(checkpoint);before=copy.deepcopy(saved['arrows']);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source/'clean.png',out/'clean.png');clean=Image.open(out/'clean.png').convert('RGB')
    if list(clean.size)!=saved['size']:raise ValueError('Checkpoint size does not match CLEAN')
    results=[dict(group='preserved_baseline',state='completed',rows=before)]
    (out/'before').mkdir(exist_ok=True);shutil.copy2(checkpoint,out/'before/scene.json')
    inherited=read(source/'completed.json') if (source/'completed.json').exists() else {}
    write(out/'before/source_completed.json',inherited)
    write(out/'resume_provenance.json',dict(source_stage=str(checkpoint.resolve()),scene_sha256=sha(checkpoint),
        clean_sha256=sha(source/'clean.png'),source_stage_name='continuation',identity_rerun=False,coverage_entry='coverage_stage.run'))
    with locks['cpu']:
        publish(clean,results,out)
        native.write(clean.size,before,[r['style'] for r in before],out/'before/arrows.pptx')
    shutil.copytree(out/'result',out/'before/result',dirs_exist_ok=True)
    (out/'RESULTS.md').write_text('# '+cid+'：双图补漏运行中\n\n当前 result 仍是补漏前快照，补漏提交后更新；尚无最终结果。\n\n![补漏前](before/result/styled.png)\n')
    pool_path=source/'shared_observations.json';pool=read(pool_path)['masks'] if pool_path.exists() else None
    if pool_path.exists():shutil.copy2(pool_path,out/'shared_observations.json')
    def progress(stage,**kw):notify(cid,dict(stage=stage,resumed_from='continuation',**kw))
    results,issues,changes=coverage_stage.run(clean,results,out,config,model,locks,pool,notify=progress)
    scope_removed=set()
    if config.get('scope_review',False):
        import scope_review
        progress('scope_review')
        results,review_issues=scope_review.run(clean,results,out/'scope_review',model,locks)
        issues.extend(review_issues)
        scope_removed=set(read(out/'scope_review/audit.json').get('removed',[]))
    with locks['cpu']:after=publish(clean,results,out)
    removed=set(changes['replaced']);byid={r['id']:r for r in after}
    stable=lambda r:{k:v for k,v in r.items() if k!='label'}
    if any(r['id'] not in byid or stable(r)!=stable(byid[r['id']]) for r in before if r['id'] not in removed|scope_removed):
        raise AssertionError('A baseline path changed without an explicit replacement')
    write(out/'change_audit.json',dict(changes,scope_review_removed=sorted(scope_removed),unmodified_baseline_preserved=True))
    if config.get('separate_style',False) and changes['added']:
        import style_pipeline
        added=set(changes['added'])
        selected=[dict(group=r['group'],state=r['state'],rows=[row for row in r.get('rows',[]) if row['id'] in added])
                  for r in results if any(row['id'] in added for row in r.get('rows',[]))]
        progress('shape_geometry')
        selected,shape_issues=style_pipeline.prepare_case_geometry(clean,selected,out,config,model,locks)
        issues.extend(shape_issues)
        updates={r['id']:r for g in selected for r in g['rows']}
        for group in results:group['rows']=[updates.get(r['id'],r) for r in group.get('rows',[])]
        write(out/'geometry_results.json',results)
        progress('style_recovery')
        selected,style_issues=style_pipeline.finish_case(clean,selected,out,config,model,locks)
        issues.extend(style_issues)
        updates={r['id']:r for g in selected for r in g['rows']}
        for group in results:group['rows']=[updates.get(r['id'],r) for r in group.get('rows',[])]
        with locks['cpu']:after=publish(clean,results,out)
    write(out/'committed_results.json',results)
    rendered=False
    if config.get('native_render',False):
        progress('native_render')
        try:
            with locks['cpu']:
                native.render(out/'before/arrows.pptx',out/'before/native',clean.size)
                native.render(out/'arrows.pptx',out/'native',clean.size)
            rendered=True
        except Exception as exc:issues.append(dict(stage='native_render',error=str(exc)))
    summary=dict(case=cid,state='completed_with_issues' if issues else 'completed',issues=issues,groups=0,
        targets=len({r['parent_id'] for r in after}),draw_paths=len(after),paths_before=len(before),
        added_paths=len(changes['added']),replaced_paths=len(removed),native_rendered=rendered,accuracy_measured=False,
        scope_review_removed_paths=len(scope_removed),
        inherited_baseline_state=inherited.get('state'),source_stage=str(checkpoint.resolve()),identity_rerun=False,
        native_ppt=str(out/'arrows.pptx'))
    write(out/'completed.json',summary)
    lines=[f'# {cid}：主线双图补漏','',f'状态：{summary["state"]}。沿用补漏前快照，新增 {summary["added_paths"]} 条、替换 {summary["replaced_paths"]} 条旧路径。','',
        '| 原图 | 补漏前 | 最终重建 | 叠图 |','|---|---|---|---|',
        '| ![](clean.png) | ![](before/result/styled.png) | ![]('+('native/styled.png' if rendered else 'result/styled.png')+') | ![](result/overlap.png) |','',
        '[PPT](arrows.pptx) · [实际候选请求](coverage/prompt.json) · [本轮问题](completed.json) · [基线继承问题](before/source_completed.json) · [增删审计](change_audit.json)','']
    for p in sorted((out/'missing').glob('*/README.md')):lines.append(f'- [{p.parent.name}]({p.relative_to(out)})')
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n');progress(summary['state'],targets=summary['targets'])
    return summary
