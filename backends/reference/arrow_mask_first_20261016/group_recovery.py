"""The main group recovery path: semantic instances, then mask measurements."""
import shutil,traceback,sys
from pathlib import Path
import numpy as np
from PIL import Image
import evidence as ev,local_sampling,semantic_geometry,mask_vectors,style_windows,geometry_routing,observation_pool
sys.path.insert(0,str(Path(__file__).resolve().parent/'lib'))
import fit,viz,style
from transport.common import read_json as read,write_json as write,sha256 as sha


def prepare_group(clean,group,all_hints,group_of,gid,out,config,locks,box=None):
    """Persist native evidence before model decisions, without creating targets."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with locks['cpu']:crop,view=local_sampling.prepare(clean,group,out,config,box)
    selected,all_candidates=local_sampling.sample(crop,view,group,all_hints,group_of,gid,out,config,locks['sam'])
    attempts=[dict(folder='.',selected=len(selected),bbox=view['bbox_global'])]
    if not selected and config.get('sam_retry_context_scale'):
        retry=out/'context_retry';cfg={**config,'crop_context_scale':config['sam_retry_context_scale']}
        # Retry multiplier is relative to the actual failed crop, including its
        # minimum width/padding, so a thin target also receives more side context.
        with locks['cpu']:wider,wview=local_sampling.prepare(clean,group,retry,cfg,view['bbox_global'])
        more,all_more=local_sampling.sample(wider,wview,group,all_hints,group_of,gid,retry,cfg,locks['sam'])
        attempts.append(dict(folder='context_retry',selected=len(more),bbox=wview['bbox_global']))
        if more:
            shutil.copytree(out/'input',out/'first_input',dirs_exist_ok=True)
            shutil.copytree(retry/'input',out/'input',dirs_exist_ok=True)
            crop,view,selected,all_candidates=wider,wview,more,all_more
    write(out/'sampling_attempts.json',attempts)
    candidates=[]
    for c in all_candidates:
        item={k:v for k,v in c.items() if k!='mask'}
        y,x=np.nonzero(c['mask'])
        if len(x):
            corners=np.array([[x.min(),y.min(),1],[x.max()+1,y.max()+1,1]])@np.asarray(view['view_to_global']).T
            item['bbox_global']=[*corners[0,:2].tolist(),*corners[1,:2].tolist()]
        candidates.append(item)
    meta=dict(group=gid,folder=str(out.resolve()),view=view,candidates=candidates,selected_ids=[c['id'] for c in selected],attempts=attempts)
    write(out/'prepared_observation.json',meta)
    return meta


def group_task(clean,group,all_hints,group_of,gid,out,config,model,locks,prepare_only=False,box=None,description='',synthetic_source=None,prepared=None,evidence_pool=None):
    out=Path(out);labels=[r['id'] for r in group] or [synthetic_source]
    meta=prepared or prepare_group(clean,group,all_hints,group_of,gid,out,config,locks,box)
    view=meta['view'];attempts=meta['attempts'];all_candidates=meta['candidates']
    write(out/'sampling_attempts.json',attempts)
    if prepared is not None:write(out/'prepared_source.json',dict(folder=meta['folder'],view=view,selected_ids=meta['selected_ids']))
    crop=Image.open(out/'input/clean.png').convert('RGB');index={c['id']:c for c in all_candidates}
    selected=[dict(index[mid],mask=ev.mask(index[mid]['path'])) for mid in meta['selected_ids']]
    point_mode=config.get('identity_geometry_mode')=='glm_points'
    if point_mode:proposals,info={},[]
    else:
        with locks['cpu']:proposals,info=semantic_geometry.prepare(crop,group,selected,out)
    reference_mode=config.get('identity_input_mode')=='group_refs'
    bbox_mode=config.get('identity_input_mode') in ('group_boxes','group_refs')
    shared=None
    if point_mode:
        import point_identity
        with locks['cpu']:point_identity.evidence_image(crop,group,view,all_candidates,gid,evidence_pool,out)
    elif evidence_pool is not None and not bbox_mode:
        with locks['cpu']:shared=observation_pool.overlay(crop,view,selected,gid,evidence_pool,out)
        if config.get('shared_context_global',False):
            with locks['cpu']:observation_pool.overview(view,evidence_pool,out)
    elif evidence_pool is not None and config.get('identity_shared_coverage',False):
        # Restore cross-observation pixels without reinstating an M/D census.
        # Keep CLEAN and the original candidate-box image independently visible.
        with locks['cpu']:
            crop.save(out/'input/shared_coverage.png')
            shared=observation_pool.overlay(crop,view,selected,gid,evidence_pool,out,layer='shared_coverage.png')
    images=[('CONTEXT: unverified global candidates and original whole figure',out/'input/context.png'),
            ('CLEAN: local original, coordinate frame',out/'input/clean.png'),
            ('GLOBAL HINTS: same local coordinate frame',out/'input/global_hints.png'),
            ('MASKS + D ENDPOINTS: Da is first, Db is last; measured axes are optional evidence',out/'input/proposals.png')]
    with Image.open(out/'input/context.png') as context:context_size=context.size
    if bbox_mode:
        if reference_mode:import group_reference_identity as group_bbox_identity
        else:import group_bbox_identity
        prompt,images=group_bbox_identity.build(crop,group,view,context_size,out,config)
    else:
        prompt=semantic_geometry.prompt(labels,crop.size,proposals,info,description,context_size,ev.crop_in_context(view,context_size))
        if config.get('identity_target_view')=='bbox':
            import target_locator
            images=target_locator.apply(prompt,images,group,view,crop,out)
        if config.get('identity_source_mode')=='observation':
            import observation_identity
            prompt=observation_identity.prompt(prompt)
    if config.get('model_shaft_sample',False) and not config.get('separate_style',False):
        import shaft_sampling
        shaft_sampling.add_prompt(prompt)
    if shared is not None and bbox_mode:
        images.append(('SHARED COVERAGE: pale cyan is additional SAM evidence in the same local frame',out/'input/shared_coverage.png'))
    if point_mode:
        images.append(('EVIDENCE: orange primary pixels; faint cyan surrounding pixels; locators only',out/'input/identity_evidence.png'))
    write(out/'prompt.json',prompt)
    write(out/'input/images.json',[dict(label=l,path=str(p.resolve()),sha256=sha(p)) for l,p in images])
    if shared is not None:
        prompt['shared_evidence']=dict(additional_mask_pixels=shared['additional_pixels'],native_observations=shared['observations'])
        if bbox_mode:
            prompt['instructions'].append('SHARED COVERAGE uses pale cyan for extra SAM pixels from overlapping observations or unselected candidates. '
                'It is context, not already reconstructed geometry, ownership, or proof of an arrow. No M/D matching is requested. '
                'The supplied original G boxes still locate this group task. Use the coverage to recognize nearby continuations; '
                'do not promote every highlighted neighbor or crop-truncated stub to a complete new target. '
                'For a related connector that needs more context, request followup rather than publishing its visible stub as a complete object. '
                'A dense band may still have one or several targets; missing cyan is not proof of absence.')
        else:
            prompt['instructions'].append('Pale cyan behind the named masks shows additional local SAM evidence shared by other observations. It is NOT already recovered geometry, a new identity, or proof of a valid connector. Use it to notice a related continuation and request followup if needed. A missing cyan highlight does not prove no line exists; inspect CLEAN.')
        if config.get('shared_context_global',False):prompt['instructions'].append('CONTEXT also shows the pooled local SAM coverage in pale cyan, including portions outside this crop. This is segmentation coverage, not already reconstructed arrows. The magenta rectangle still locates CLEAN.')
        write(out/'prompt.json',prompt)
    if prepare_only:return dict(group=gid,state='prepared',sam_candidates=len(all_candidates),selected=len(selected),proposals=len(proposals))
    def validate_identity(a):
        if point_mode:return group_bbox_identity.validate(a,crop.size,context_size,labels,config)
        if bbox_mode:return group_bbox_identity.validate(a,crop.size,context_size,
            labels if reference_mode or config.get('identity_neutral_handoff') else None)
        if config.get('identity_source_mode')=='observation':
            import observation_identity
            a=observation_identity.answer(a,labels)
        return semantic_geometry.validate(a,crop.size,labels,proposals,[c['id'] for c in selected],selected,context_size)
    answer=model.ask(out/'identity',prompt,images,validate_identity)
    mask_geometry_calls=0;mask_geometry_issues=[]
    if bbox_mode:
        write(out/'identity_decision.json',answer)
        answer,mask_geometry_calls,mask_geometry_issues=group_bbox_identity.resolve(
            answer,labels,crop,proposals,selected,info,out,model,config,prompt['group_boxes'])
    write(out/'decision.json',answer)
    rows,supports,issues,geometry_calls=geometry_routing.recover(answer,proposals,selected,gid,crop,out,model,locks,config)
    issues=mask_geometry_issues+issues
    if config.get('geometry_critic_enabled',False):
        import geometry_critic
        rows,supports,review_issues,review_calls=geometry_critic.recover(
            crop,rows,supports,out/'geometry_critic',model,config,locks['cpu'],source_view=view)
        issues.extend(review_issues)
    with locks['cpu']:
        for row in rows:
            if config.get('separate_style',False):
                import style_contract
                row['style']=style_contract.provisional(crop,supports[row['id']],row)
                row['geometry_frame']=view
            else:row['style']=style_windows.recover(np.asarray(crop),supports[row['id']],row)
            row['local_input']=str(out/'input/clean.png')
            if row['style'].get('evidence',{}).get('fill_samples')==0:issues.append(dict(stage='color_evidence_missing',route=row['id']))
        write(out/'rows_local.json',rows);viz.render(out/'input/clean.png',rows,out/'result')
        global_rows=[fit.transform(row,view) for row in rows];write(out/'rows_global.json',global_rows)
        pending=[r for r in rows if style.pending(r)];write(out/'pending_style.json',pending)
        if pending and not config.get('separate_style',False):issues.append(dict(stage='style_pending',rows=[r['id'] for r in pending]))
    result=dict(group=gid,state='completed_with_issues' if issues or answer['uncertain'] or answer.get('observation_incomplete') else 'completed',rows=global_rows,
        observation_incomplete=answer.get('observation_incomplete',False),
        identities=answer['routes'],excluded=answer['excluded'],uncertain=answer['uncertain'],issues=issues,
        followups=[dict(bbox=(np.asarray(f['context_bbox'])*np.tile(np.asarray(clean.size)/context_size,2)).tolist(),description=f['description'],source_group=gid) for f in answer.get('followups',[])],
        sam_calls=len(attempts),semantic_calls=1,repair_calls=geometry_calls,geometry_calls=geometry_calls,
        mask_geometry_calls=mask_geometry_calls,identity_input_mode=config.get('identity_input_mode','legacy'),
        head_calls=read(out/'head_calls.json')['calls'],
        coordinate_repairs=sum(r['coordinate_source'] in ('glm_keypoints','glm_segmented_points') for r in rows),
        mask_geometry_rows=sum(r['coordinate_source']=='mask_no_skeleton' for r in rows),selected_masks=[r['id'] for r in selected])
    write(out/'completed.json',result)
    if config.get('geometry_critic_enabled',False):
        result['geometry_critic_calls']=review_calls
        write(out/'completed.json',result)
    if (out/'error.json').exists():
        history=out/'error_history';history.mkdir(exist_ok=True)
        shutil.move(str(out/'error.json'),str(history/f'{len(list(history.glob("*.json")))+1:03d}.json'))
    identity_preview='input/target_boxes.png' if bbox_mode else 'input/proposals.png'
    (out/'README.md').write_text('# '+gid+'：身份路由与几何恢复\n\n'
        '| 原图 | 身份判断的定位输入 | 最终实例 | 样式叠图 |\n|---|---|---|---|\n'
        '| ![](input/clean.png) | ![]('+identity_preview+') | ![](result/instances.png) | ![](result/overlap.png) |\n\n'
        '[模型决定](decision.json) · [几何路由与端点头型请求](routing.json) · [实际请求](identity/prompt.json) · [采样次数与范围](sampling_attempts.json) · [结果](completed.json)\n')
    if shared is not None and bbox_mode:
        with (out/'README.md').open('a') as file:file.write('\n[身份阶段实际共享覆盖输入](input/shared_coverage.png) · [共享像素来源](shared_evidence.json)\n')
    return result
