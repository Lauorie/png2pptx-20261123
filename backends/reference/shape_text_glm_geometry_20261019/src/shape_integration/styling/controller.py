"""One case: immutable geometry -> styles -> actual render -> critic -> one refine."""
import copy
import time
from concurrent.futures import as_completed, ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import nullcontext
from pathlib import Path
from PIL import Image
from shapely.geometry import box as shape_box
from shapely.ops import unary_union
from ..common import read, save, sha, digest, now
from . import rules, calibration, occlusion, views, model, grouping
from .geometry import parts, closed, crop_box
from .export import export, pattern
from .live import publish

SCHEDULING_PROTOCOL = "component_pipeline.1"


def run_case(scene, source, folder, runtime, options, pool, render_slots, *, assets=(), selected_objects=None,
             rules_only=False, pixel_slots=None, progress=lambda *a:None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);table=parts(scene)
    source=Path(source)
    if sha(source)!=scene['source_sha256']:raise ValueError('Style source pixels do not match geometry')
    identity={'scene_sha256':digest(scene),'source_sha256':sha(source),'options':options,'style_protocol':model.PROTOCOL,
              'component_group_protocol':grouping.PROTOCOL,'scheduling_protocol':SCHEDULING_PROTOCOL,
              'pdf_crop_protocol':views.PDF_CROP_PROTOCOL,
              'assets_sha256':digest(list(assets)),'selected_objects':selected_objects,'rules_only':rules_only,
              'model':runtime.config['model'] if runtime else None,
              'timeout':runtime.config.get('request_timeout_seconds') if runtime else None}
    if (folder/'identity.json').exists() and read(folder/'identity.json')!=identity:
        raise ValueError('Style case identity changed; create a new run')
    save(folder/'identity.json',identity);save(folder/'scene_source.json',scene)
    if (folder/'summary.json').exists():
        existing=read(folder/'summary.json')
        if existing['state'] in ('completed','completed_with_unresolved','rules_only'):
            if existing.get('styles_sha256')!=sha(folder/'styles.json'):raise ValueError('Saved styles changed')
            return existing
    started=time.monotonic();timings=[]
    def elapsed(stage, beginning, revision=None):
        timings.append({'stage':stage,'revision':revision,'wall_seconds':time.monotonic()-beginning})
        save(folder/'timings.json',{'stages':timings,'elapsed_wall_seconds':time.monotonic()-started,
                                   'note':'Stage wall time includes local queueing; individual API queue/network times are in calls/*/stream_status.json.'})
    selected={k for k,p in table.items() if selected_objects is None or p['object_id'] in selected_objects}
    if selected_objects and not set(selected_objects)<={p['object_id'] for p in table.values()}:
        raise ValueError('Selected object has no exportable geometry')
    progress('relations',0,len(table))
    geoms,relations=occlusion.discover(table,options)
    selected_components={table[k]['object_id'] for k in selected}
    overlap=[p for p in relations['pending'] if p['kind']=='partial_overlap' and ({p['a'],p['b']}&selected_components)]
    if runtime is not None and not rules_only:
        futures={}
        for index,batch in enumerate(occlusion.relation_batches(overlap,options['relation_batch_pairs'])):
            keys=list(dict.fromkeys(k for p in batch for k in (p['a'],p['b'])))
            mapping={f'C{i}':k for i,k in enumerate(keys)};aliases={k:a for a,k in mapping.items()}
            root=folder/'relations'/f'batch_{index:03d}'
            try:images,units,_=views.prepare(source,table,mapping,root/'input',options,assets=assets,
                                           component_members=relations['component_parts'])
            except Exception as exc:
                save(root/'input_error.json',{'mapping':mapping,'error':str(exc)})
                for pair in batch:pair['reason']='input_error: '+str(exc)
                continue
            pair_table={f'R{i}':[aliases[p['a']],aliases[p['b']]] for i,p in enumerate(batch)}
            p=model.prompt('relations',[{'id':a} for a in mapping],units,pairs=[{'id':a,'parts':v} for a,v in pair_table.items()])
            future=pool.submit(model.ask,runtime,root/'calls','relations',p,images,list(pair_table),{},options,pairs=pair_table)
            futures[future]=(batch,mapping)
        done=0
        for f in as_completed(futures):
            batch,mapping=futures[f]
            try: result=f.result()
            except Exception as exc:result={'valid':{},'errors':{'batch':str(exc)}}
            for i,pair in enumerate(batch):
                row=result['valid'].get(f'R{i}')
                if row and row['front'] is not None:
                    front=mapping[row['front']];back=pair['b'] if front==pair['a'] else pair['a']
                    relations['edges'].append({'back':back,'front':front,'origin':'model','evidence':row['evidence']})
                    relations['pending'].remove(pair)
                else:pair['reason']=row['evidence'] if row else result['errors'].get(f'R{i}',result['errors'].get('batch','no_relation'))
            done+=1;progress('relations',done,len(futures))
    relations=occlusion.order(scene,table,relations);save(folder/'occlusion.json',relations)
    elapsed('relations',started);rules_started=time.monotonic()
    styles={};evidence={};records={}
    image=Image.open(source).convert('RGB')
    asset_masks=[shape_box(*a['bbox_2d']) for a in assets if len(a.get('bbox_2d',[]))==4]
    def excluded(key, current):
        masks=[occlusion.exclusions(key,geoms,relations,current)]
        # Asset bboxes are conservative unknown regions, never claimed exact opaque masks.
        masks.extend(g for g in asset_masks if geoms[key].intersects(g))
        return unary_union(masks)
    for index,key in enumerate(reversed(relations['flat_order'])):
        try:
            with pixel_slots or nullcontext():
                e=rules.estimate(image,table[key],excluded(key,styles),options)
            styles[key]=e['style'];evidence[key]=e
            records[key]={'origin':'rules','history':[{'stage':'rules','style':copy.deepcopy(styles[key])}],
                          'selected_for_model':key in selected,'geometry_state':scene['objects'][table[key]['object_id']]['state']}
        except Exception as exc:
            styles[key]={'fill':None,'stroke':'#D000C0','width_px':1.2,'dash':'solid'}
            evidence[key]={'style':styles[key],'closed':closed(table[key]),'needs_model':True,'reasons':['rule_error: '+str(exc)],
                           'rule_error':str(exc),'candidates':{'fills':[None],'strokes':[None]}}
            records[key]={'origin':'diagnostic_fallback','history':[{'stage':'rules','error':str(exc)}],
                          'selected_for_model':key in selected,'geometry_state':scene['objects'][table[key]['object_id']]['state']}
        if index%10==0:progress('rules',index+1,len(table))
    save(folder/'rules.json',evidence)
    elapsed('rules',rules_started)
    progress('component_groups',0,len(selected));group_started=time.monotonic()
    group_audit=grouping.discover(scene,source,table,selected,folder/'component_groups',
                                 runtime if not rules_only else None,options,pool,assets)
    elapsed('component_groups',group_started)
    grouped={k for group in group_audit['groups'] for k in group['members']}
    work_units=[]
    for obj in scene['objects'].values():
        keys=[k for k in obj['part_keys'] if k in table and k in selected]
        if keys and not grouped.intersection(keys):
            work_units.append({'id':obj['id'],'keys':keys,'root':folder/'objects'/obj['id'],'kind':'object'})
    for group in group_audit['groups']:
        keys=[group['representative']]+[k for k in group['members'] if k!=group['representative']]
        work_units.append({'id':group['id'],'keys':keys,'root':folder/'component_groups'/group['id'],
                           'kind':'shared_component_style'})
        for key in keys:records[key]['style_group']=group['id']
    save(folder/'style_units.json',[{**unit,'root':str(unit['root'].relative_to(folder))} for unit in work_units])
    problems={k:[{'rule_error':e['rule_error'],'action':'No source-derived candidates; estimate freely during the one correction'}]
              for k,e in evidence.items() if e.get('rule_error')}

    # Only this case coordinator mutates styles, records and unit states. Workers
    # receive frozen input copies and never wait on tasks in their own executor.
    checks={};sequence=0;latest=None;render_future=None;last_render=-float('inf')
    jobs={};calibration_audit={};units_by_key={k:u for u in work_units for k in u['keys']}
    context_audited=False;failed_sequence=None
    model_enabled=runtime is not None and not rules_only
    for unit in work_units:
        unit.update(phase='new',revision='revision_00',review_snapshot=None)
        try:
            unit['region']=shape_box(*crop_box([table[k] for k in unit['keys']],image.size,
                                               options['crop_padding'],options['crop_min_padding_px']))
        except ValueError:unit['region']=None

    def floor(keys,revision):
        applied={};minimum=options['minimum_stroke_width_px']
        for key in keys:
            style=styles[key]
            if style['stroke'] is not None and style['width_px']<minimum:
                styles[key]={**style,'width_px':minimum}
                applied[key]={'before_width_px':style['width_px'],'after_width_px':minimum}
                records[key]['history'].append({'stage':'minimum_stroke_width','revision':revision,**applied[key]})
        save(folder/revision/'stroke_width_floor.json',{'unit':'original_source_pixels',
             'minimum_stroke_width_px':minimum,'applied':applied})

    floor(table,'rules_baseline')

    def normalize(unit):
        nonlocal sequence
        keys=unit['keys'];base=copy.deepcopy(styles);audit={}
        revision=str((unit['root']/unit['revision']).relative_to(folder));beginning=time.monotonic()
        shared=unit['kind']=='shared_component_style'
        for key in (keys[:1] if shared else keys):
            if not options['pixel_calibration_enabled']:
                audit[key]={'enabled':False,'apply':False,'changed':False,'reason':'disabled_by_config'}
                continue
            try:
                with pixel_slots or nullcontext():
                    styles[key],audit[key]=calibration.calibrate(image,table[key],base[key],excluded(key,base))
            except Exception as exc:audit[key]={'apply':False,'error':str(exc)}
            audit[key]['enabled']=True
        if shared:
            for key in keys[1:]:
                styles[key]=copy.deepcopy(styles[keys[0]])
                audit[key]={**audit[keys[0]],'shared_from':keys[0],'style_group':unit['id']}
        for key in keys:
            if styles[key]!=base[key]:
                records[key]['history'].append({'stage':'pixel_calibration','revision':revision,
                                               'before':base[key],'after':copy.deepcopy(styles[key])})
            records[key]['selected_style_revision']=revision
        calibration_audit.update(audit);save(folder/revision/'calibration.json',audit)
        floor(keys,revision);save(folder/revision/'styles.json',{k:styles[k] for k in keys})
        elapsed('calibration',beginning,revision);sequence+=1

    def request(unit,mode,revision,base,feedback_by_key,snapshot):
        keys=unit['keys'];shared=unit['kind']=='shared_component_style'
        mapping={f'P{i}':k for i,k in enumerate(keys)}
        output_mapping={'P0':keys} if shared else {a:[k] for a,k in mapping.items()}
        root=unit['root']/revision/f'{mode}_000'
        native_pdf=snapshot['root']/'native.pdf' if snapshot else None
        try:
            images,units,frame=views.prepare(source,table,mapping,root/'input',options,native_pdf=native_pdf,
                                            assets=assets,with_stroke_reference=True)
            with pixel_slots or nullcontext():
                ranked=rules.ranked_source_colors(image,{k:table[k] for k in keys},
                           lambda k:excluded(k,base),options['color_candidate_limit']) if mode=='select' else None
        except Exception as exc:
            save(root/'input_error.json',{'mapping':mapping,'error':str(exc)})
            return {k:{'value':None,'error':'input_error: '+str(exc),'path':str(root.relative_to(folder))} for k in keys}
        colors=[entry['color'] for entry in ranked['colors']] if ranked else []
        if ranked is not None:save(root/'color_candidates.json',ranked)
        targets=[]
        for alias,members in output_mapping.items():
            key=members[0];target={'id':alias,'closed':closed(table[key])}
            if shared:target['shared_with']=list(mapping)[1:]
            if mode in ('critic','refine'):
                # Both reviewers see the exact styles behind their immutable PDF.
                displayed=copy.deepcopy(base[key]);displayed.pop('issue',None)
                effective_pattern=pattern(displayed)
                if effective_pattern:displayed['dash_pattern_px']=effective_pattern
                scale=frame['source_to_crop_scale'];displayed['width_px']*=scale
                if 'dash_pattern_px' in displayed:
                    displayed['dash_pattern_px']=[v*scale for v in displayed['dash_pattern_px']]
                target['current_style' if mode=='critic' else 'previous_style']=displayed
            gradients=[f for f in evidence[key]['candidates']['fills'] if isinstance(f,dict)]
            if mode=='select' and gradients:target['gradient_candidates']=gradients
            targets.append(target)
        feedback=None
        if mode=='refine':
            feedback=[{'id':alias,'problems':list({digest(p):p for k in members for p in feedback_by_key.get(k,[])}.values())}
                      for alias,members in output_mapping.items() if any(feedback_by_key.get(k) for k in members)]
        p=model.prompt(mode,targets,units,feedback=feedback,color_candidates=colors)
        save(root/'job.json',{'mapping':mapping,'output_mapping':output_mapping,'unit_kind':unit['kind'],
             'prompt':p,'geometry_digest':digest({k:table[k] for k in keys}),
             'render_snapshot':str(snapshot['root'].relative_to(folder)) if snapshot else None,
             'render_sha256':frame['rendered_source']['sha256'] if native_pdf else None,
             'rendered_source':frame['rendered_source'],'frame_sha256':sha(root/'input/frame.json')})
        try:
            res=model.ask(runtime,root/'calls',mode,p,images,list(output_mapping),
                          {a:table[m[0]] for a,m in output_mapping.items()},options,
                          source_to_crop_scale=frame['source_to_crop_scale'])
        except Exception as exc:res={'valid':{},'errors':{a:str(exc) for a in output_mapping}}
        return {k:{'value':copy.deepcopy(res['valid'].get(a)),'error':res['errors'].get(a),
                   'path':str(root.relative_to(folder))} for a,members in output_mapping.items() for k in members}

    def submit(unit,mode,snapshot=None,*,context_recheck=False,resume=False):
        revision='context_recheck' if context_recheck else unit['revision']
        base=copy.deepcopy(snapshot['styles'] if snapshot else styles)
        context=unit['root']/revision/(mode+'_000')/'request_context.json'
        if resume:
            saved=read(context);base=saved['styles'];feedback=saved['problems']
        else:
            feedback=copy.deepcopy(problems)
            save(context,{'styles':base,'problems':feedback})
        future=pool.submit(request,unit,mode,revision,base,feedback,snapshot)
        jobs[future]=(unit,mode,revision,snapshot,time.monotonic(),context_recheck)
        unit['phase']='context_recheck' if context_recheck else mode

    def stale(unit,snapshot):
        if snapshot is None:return True
        for key,style in styles.items():
            old=snapshot['styles'][key]
            if old==style:continue
            if key in unit['keys']:return True
            region=unit['region']
            # A changed stroke can enter the crop even when its centerline lies outside.
            width=max(old['width_px'] if old['stroke'] else 0,style['width_px'] if style['stroke'] else 0)
            if region is not None and geoms[key].buffer(max(width/2,.01)).intersects(region):return True
        return False

    def checkpoint(terminal=False):
        current_reviews={id(u):not stale(u,u['review_snapshot']) for u in work_units if u['phase']=='done'}
        for key,record in records.items():
            unit=units_by_key.get(key);value=checks.get(key,{}).get('value')
            if key not in selected:state='context_only'
            elif rules_only:state='rules_only'
            elif unit['phase']!='done':state='running_'+unit['phase']
            elif not current_reviews[id(unit)]:state='pending_context_recheck'
            else:state='critic_'+value['verdict'] if value else 'unresolved'
            # No render means there cannot be a current visual check.
            if terminal and state=='pending_context_recheck':state='unresolved'
            record.update(state=state,style=copy.deepcopy(styles[key]),critic=checks.get(key,{}))
            record.setdefault('selected_style_revision','rules_baseline')
            record['invisible_geometry']=styles[key]['fill'] is None and styles[key]['stroke'] is None
        final=str(latest['root'].relative_to(folder)) if latest else None
        save(folder/'styles.json',{'schema_version':'shape_integration.styles.1','input':identity,
             'length_unit':'original_source_pixels','minimum_stroke_width_px':options['minimum_stroke_width_px'],
             'state':'finished' if terminal else 'running','sequence':sequence,'selected_revision':final,'parts':records,
             'style_groups':[{**g,'style':styles[g['representative']]} for g in group_audit['groups']],
             'visual_accuracy_verified':False})
        done=sum(u['phase']=='done' for u in work_units)
        save(folder/'progress.json',{'state':'finished' if terminal else 'running','updated_utc':now(),
             'sequence':sequence,'rendered_sequence':latest['sequence'] if latest else None,
             'rendered_snapshot':final,'completed_units':done,'total_units':len(work_units),
             'units':{str(u['root'].relative_to(folder)):{'stage':u['phase'],'revision':u['revision'],
                      'parts':len(u['keys'])} for u in work_units}})
        def ref(snapshot):return str(snapshot['root'].relative_to(folder)) if snapshot else None
        save(folder/'scheduler.json',{'sequence':sequence,'styles':styles,'records':records,'checks':checks,
             'problems':problems,'calibration_audit':calibration_audit,'latest':ref(latest),
             'context_audited':context_audited,
             'units':{str(u['root'].relative_to(folder)):{'phase':u['phase'],'revision':u['revision'],
                      'review_snapshot':ref(u['review_snapshot'])} for u in work_units},
             'pending':[{'unit':str(u['root'].relative_to(folder)),'mode':mode,'snapshot':ref(snap),
                         'context_recheck':ctx} for u,mode,rev,snap,t,ctx in jobs.values()]})
        progress('components',done,len(work_units))

    def render_snapshot(snapshot,audit):
        root=snapshot['root'];save(root/'styles.json',snapshot['styles']);save(root/'calibration.json',audit)
        export(scene,snapshot['styles'],relations,root,source,options,render_slots)
        return snapshot

    if (folder/'scheduler.json').exists():
        saved=read(folder/'scheduler.json')
        styles=saved['styles'];records=saved['records'];checks=saved['checks'];problems=saved['problems']
        sequence=saved['sequence'];calibration_audit=saved['calibration_audit'];context_audited=saved['context_audited']
        snapshots={}
        def restore(ref):
            if ref is None:return None
            if ref not in snapshots:
                root=folder/ref
                snapshots[ref]={'root':root,'styles':read(root/'styles.json'),
                                'sequence':read(root/'live.json')['sequence']}
            return snapshots[ref]
        latest=restore(saved['latest'])
        by_path={str(u['root'].relative_to(folder)):u for u in work_units}
        for path,u in by_path.items():
            u.update(saved['units'][path]);u['review_snapshot']=restore(u['review_snapshot'])
        for row in saved['pending']:
            submit(by_path[row['unit']],row['mode'],restore(row['snapshot']),
                   context_recheck=row['context_recheck'],resume=True)
    else:
        for unit in work_units:
            keys=unit['keys']
            if model_enabled and (len(keys)>1 or any(evidence[k]['needs_model'] for k in keys)):
                submit(unit,'select')
            else:
                normalize(unit);unit['phase']='waiting_render' if model_enabled else 'done'
    checkpoint()
    (folder/'REPORT.zh-CN.md').write_text('# 样式恢复进行中\n\n组件独立推进；以下链接自动指向最近完整渲染。\n\n'
        '[实时进度](progress.json) · [样式状态](styles.json) · [可编辑 PPTX](styled.pptx) · [渲染版本](live.json)\n\n'
        '![原图与实际 PPT 渲染](comparison.png)\n')
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='style-render') as renderer:
        while True:
            # Fast components use a finished immutable PDF immediately. Subsequent
            # neighbour changes are checked once against the final frozen scene.
            dispatched=False
            for unit in work_units:
                if unit['phase']=='waiting_render' and latest is not None and all(
                        latest['styles'][k]==styles[k] for k in unit['keys']):
                    submit(unit,'critic',latest);dispatched=True
            if dispatched:checkpoint()
            dirty=latest is None or latest['sequence']!=sequence
            if render_future is None and dirty and failed_sequence!=sequence and (
                    not jobs or time.monotonic()-last_render>=options['live_render_interval_seconds']):
                snapshot={'root':folder/'snapshots'/f'snapshot_{sequence:06d}_{digest(styles)[:12]}',
                          'sequence':sequence,'styles':copy.deepcopy(styles)}
                last_render=time.monotonic()
                render_future=renderer.submit(render_snapshot,snapshot,copy.deepcopy(calibration_audit))
            all_done=all(u['phase']=='done' for u in work_units)
            if all_done and not jobs and render_future is None:
                if dirty:
                    checkpoint(terminal=True)
                    raise RuntimeError('Latest styles could not be rendered; previous live PPT retained; see snapshots/*/error.json')
                if not context_audited:
                    context_audited=True
                    if model_enabled:
                        for unit in work_units:
                            if stale(unit,unit['review_snapshot']):submit(unit,'critic',latest,context_recheck=True)
                    checkpoint()
                    if jobs:continue
                break
            futures=list(jobs)+([render_future] if render_future else [])
            if not futures:raise RuntimeError('Style scheduler has unfinished units without work')
            ready,_=wait(futures,timeout=.5,return_when=FIRST_COMPLETED)
            changed=False
            for future in ready:
                if future is render_future:
                    render_future=None
                    try:
                        candidate=future.result();publish(folder,candidate['root'],candidate['sequence']);latest=candidate
                        progress('preview',latest['sequence'],sequence)
                    except Exception as exc:
                        failed_sequence=snapshot['sequence']
                        save(snapshot['root']/'error.json',{'error':str(exc),'sequence':failed_sequence})
                        # Preserve all selected styles even if a native export fails.
                        for unit in work_units:
                            if unit['phase']=='waiting_render' and all(snapshot['styles'][k]==styles[k] for k in unit['keys']):
                                unit['phase']='done'
                                for key in unit['keys']:checks[key]={'value':None,'error':'render_error: '+str(exc)}
                    elapsed('render',last_render,str(snapshot['root'].relative_to(folder)))
                    changed=True;continue
                unit,mode,revision,used_snapshot,beginning,is_context=jobs.pop(future)
                try:results=future.result()
                except Exception as exc:results={k:{'value':None,'error':str(exc)} for k in unit['keys']}
                elapsed(mode,beginning,str((unit['root']/revision).relative_to(folder)))
                for key,result in results.items():
                    records[key]['history'].append({'stage':mode,'revision':revision,**result})
                if mode in ('select','refine'):
                    before={k:copy.deepcopy(styles[k]) for k in unit['keys']}
                    valid=False
                    for key,result in results.items():
                        if result['value'] is not None:
                            valid=True;styles[key]=result['value']
                            records[key]['origin']='model_selection' if mode=='select' else 'model_refine'
                            if mode=='select' and result['value'].get('issue'):
                                problems.setdefault(key,[]).append(result['value']['issue'])
                        elif mode=='select':problems.setdefault(key,[]).append({'selection_error':result['error']})
                    if mode=='select' or valid:normalize(unit)
                    modified=any(before[k]!=styles[k] for k in unit['keys'])
                    unit['phase']='waiting_render' if mode=='select' or modified else 'done'
                else:
                    checks.update(results);unit['review_snapshot']=used_snapshot
                    if revision=='revision_00' and not is_context:
                        for key,result in results.items():
                            if result['value'] is None or result['value']['verdict']!='accept':
                                problems.setdefault(key,[]).append(result['value'] or {'critic_error':result['error']})
                        if options['refine_rounds'] and any(problems.get(k) for k in unit['keys']):
                            unit['revision']='revision_01';submit(unit,'refine',used_snapshot)
                        else:unit['phase']='done'
                    else:unit['phase']='done'
                changed=True
            if changed:checkpoint()
    checkpoint(terminal=True)
    final=str(latest['root'].relative_to(folder))
    unresolved=[k for k in selected if records[k]['state']!='critic_accept']
    summary={'state':'rules_only' if rules_only else 'completed_with_unresolved' if unresolved or relations['pending'] or relations['order_conflicts'] else 'completed',
             'parts':len(table),'targeted_parts':len(selected),'critic_accepted':sum(records[k]['state']=='critic_accept' for k in selected),
             'unresolved_parts':len(unresolved),'relation_unresolved':len(relations['pending']),
             'order_conflicts':len(relations['order_conflicts']),'gradients':sum(isinstance(styles[k]['fill'],dict) for k in selected),
             'pixel_calibration_enabled':options['pixel_calibration_enabled'],
             'component_style_groups':len(group_audit['groups']),
             'grouped_components':len(grouped),'grouping_errors':len(group_audit['errors']),
             'scheduling_protocol':SCHEDULING_PROTOCOL,'completed_units':len(work_units),
             'selected_revision':final,'styles_sha256':sha(folder/'styles.json'),'updated_utc':now(),'visual_accuracy_verified':False}
    if digest(scene)!=identity['scene_sha256']:raise AssertionError('Style recovery mutated geometry')
    save(folder/'summary.json',summary)
    lines=['# 样式恢复结果','',f'状态：`{summary["state"]}`。模型接受不等于视觉准确率。',
           '',f'目标部件 {len(selected)}，critic 接受 {summary["critic_accepted"]}，未决 {len(unresolved)}；几何状态不变。',
           '', '后台像素校准：'+('开启' if options['pixel_calibration_enabled'] else '关闭')+'。',
           '', '[样式与状态](styles.json) · [组件进度](progress.json) · [渲染版本](live.json) · [组件分组](component_groups/groups.json) · [整体恢复单元](style_units.json) · [遮挡关系](occlusion.json) · [可编辑 PPTX](styled.pptx) · [SVG](styled.svg) · [矢量 PDF](native.pdf)',
           '', '![原图与实际 PPT 渲染](comparison.png)','',
           '| 对象/部件 | 样式状态 | 来源 |','| --- | --- | --- |']
    for k in relations['flat_order']:
        if k in selected:lines.append(f'| {table[k]["object_id"]}/{table[k]["id"]} | {records[k]["state"]} | {records[k]["origin"]} |')
    (folder/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
    progress('completed',len(selected),len(selected));return summary
