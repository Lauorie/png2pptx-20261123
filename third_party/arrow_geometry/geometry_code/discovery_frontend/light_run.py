#!/usr/bin/env python3
"""GLM-first discovery, compressed candidates, selective SAM and one repair pass."""
import argparse
import copy
import fcntl
import json
import os
import shutil
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
from transport.common import read_json as read, write_json as write, sha256
from transport.model import Model
from protocol import SYSTEM, validate_refine
from pipeline_frames import Crop
from frontend import discover_view, materialize, public, preflight
from evidence import frame, save, boxes, get_sam, select_masks, overlay_masks, sam_only_observations, local_box
from core import crop_box, area, inside, pending_object
from light_core import compress, classify, repair_batches, select_clean_sam, compact_prompt, iou
from light_report import publish
from repair_policy import local_view, network_units, link_fragments, packet_plan, normalize_answer, repair_prompt
from fair_pool import FairPool
from residual_review import preserve_chain_scope, reconcile
from visual_labels import render as render_labels, short_request_rows, restore_answer_ids


def now(): return datetime.now(timezone.utc).isoformat()


def refresh_local_inputs(root,cfg):
    audit=[]
    for cid in read(root/'cases.json'):
        inp=root/'inputs'/cid;packet=read(inp/'views.json')
        native=Image.open(inp/'source.png').convert('RGB')
        for v in packet['views']:
            old=copy.deepcopy(v)
            if v['view_id']!='G':
                view=local_view(native.size,v['transform']['crop_bounds_global_half_open'],cfg)
                save(view.image(native),root/v['path'])
                v['transform']=view.metadata(native.size);v['sha256']=sha256(root/v['path'])
            write((root/v['path']).parent/'view.json',v)
            audit.append({'case_id':cid,'view_id':v['view_id'],'before_size':old['transform']['actual_crop_input_size_px'],
                'after_size':v['transform']['actual_crop_input_size_px'],'before_sha256':old['sha256'],
                'after_sha256':v['sha256'],'crop_bounds_unchanged':old['transform']['crop_bounds_global_half_open']==v['transform']['crop_bounds_global_half_open']})
        write(inp/'views.json',packet)
    write(root/'input_resize_audit.json',audit)
    return audit


def prepare(root, previous, benchmark):
    if root.exists(): raise RuntimeError('Fresh run directory required')
    root.mkdir(parents=True)
    cfg=read(previous/'config.json')
    discovery_source=previous.parent/'bbox_frontend_trace_smoke_20261024_v13'
    if not discovery_source.exists():discovery_source=previous
    cfg.pop('upstream_replay_run',None)
    cfg.update(api_concurrency=128,request_workers=128,case_workers=8,case_batch_size=8,
        view_workers=128,refine_workers=128,sam_evidence_max=2,
        repair_batch_max=3,repair_neighbor_gap=48,repair_span_px=320,repair_single_large_px=320,
        sam_extra_limit=8,sam_extra_min_score=.65,sam_extra_uncovered_fraction=.80,
        local_scale=3,local_min_edge=1080,local_max_edge=2048,
        extent_refine_passes=1,network_extent_expansion=False,
        residual_review_enabled=True,residual_pair_limit=12,
        max_expansion_rounds=0,protocol_version='bbox_full125_v15',scheduler='rolling_8_shared_fair_api_pool',
        purpose='125-case full discovery using the accepted pilot network/extent/duplicate strategy and text-only format repair',
        source_run=str(previous),discovery_reuse=True,discovery_reuse_run=str(discovery_source),
        review_output_dir=str(benchmark/'outputs/PPTBench/arrow_localization/reviews'/root.name))
    assert cfg['max_tokens']==113000 and 'reasoning_effort' not in cfg
    write(root/'config.json',cfg)
    shutil.copytree(Path(__file__).parent,root/'code',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(previous/'inputs',root/'inputs')
    shutil.copytree(previous/'constraints_snapshot',root/'constraints_snapshot')
    (root/'constraints_snapshot/user_speed_priority_20261024.md').write_text(
        '用户要求：GLM为主；SAM主要补少数遗漏及局部准确bbox。高重叠/包含候选可以程序取并集，减少重复判断。'
        '当前以尽量发现和整图返回速度为主，不要求百分百发现；密集神经网络整体。每批8张，总API并发128。'
        '默认thinking和113k输出上限沿用。\n')
    (root/'constraints_snapshot/user_debug_and_visual_20261024.md').write_text(
        '用户要求先修正长ID和标签重叠，然后继续实验；同时运行图片上限8，总API并发上限128。'
        '不允许把大量纠缠框和不相关区域混成复杂请求；重点检查0236点线残缺/空白误框及0320、0201的漏检。'
        '框稍偏/松可接受，发现和身份优先。\n')
    shutil.copy2(previous/'cases.json',root/'cases.json')
    shutil.copy2(previous/'execution_order.json',root/'execution_order.json')
    cases=read(root/'cases.json')
    if len(cases)!=125 or len(set(cases))!=125:raise ValueError('Full experiment requires the same 125 unique cases')
    # Review order only; no case-dependent inference or geometry behavior.
    priority=['task_0201','task_0241','task_0056','task_0008','task_0028','task_0236','task_0320','task_0027']
    previous_order=read(root/'execution_order.json')['case_ids']
    write(root/'execution_order.json',{'case_ids':priority+[c for c in previous_order if c not in priority],
          'reason':'User-mentioned cases and a simple regression first; same rules for all 125 images'})
    refresh_local_inputs(root,cfg)
    constraint_name='发现阶段完整范围优先_20261024.md'
    current_constraint=next((p/'communication'/constraint_name for p in Path(__file__).resolve().parents
                             if (p/'communication'/constraint_name).exists()),None)
    if current_constraint is None:raise RuntimeError('Current user scope snapshot is missing')
    shutil.copy2(current_constraint,root/'constraints_snapshot'/constraint_name)
    (root/'constraints_snapshot/user_full125_authorization_20261024.md').write_text(
        '用户认可8图小实验效果，明确要求继续全量125图，边运行边观察，发现问题继续分析。'
        '默认thinking、113000上限、同时8图、总API上限128；本轮包含残留两框审核。\n')
    shutil.copy2(Path(__file__).parent/'LIGHT_PROTOCOL.md',root/'PROTOCOL.md')
    files=[p for d in ['code','inputs','constraints_snapshot'] for p in (root/d).rglob('*') if p.is_file()]
    files += [root/'config.json',root/'cases.json',root/'execution_order.json',root/'input_resize_audit.json']
    write(root/'manifest.json',{'files':{str(p.relative_to(root)):sha256(p) for p in files},'created_at':now()})
    write(root/'preparation.json',{'source_run':str(previous),'cases':len(read(root/'cases.json')),
        'fresh_discovery_if_no_exact_complete_cache':True,'reuse_validated_transport_only':True,
        'speed_comparison_must_separate_reused_discovery':True,'prepared_at':now()})
    write(root/'progress.json',{'state':'prepared','total':len(read(root/'cases.json')),'ended':0,'records':[]})
    publish(root)


def global_sam(root,cid,cfg):
    inp=root/'inputs'/cid
    with Image.open(inp/'source.png') as image:size=image.size
    source=Path(cfg.get('source_run') or root)
    saved=source/'cases'/cid/'sam/G/evidence.json'
    previous=Path(read(saved)['prediction_path']).parent if saved.exists() else source/'cases'/cid/'sam/G/raw'
    cache=previous if (previous/'predictions.json').exists() else None
    if cache is None and (inp/'baseline.json').exists() and read(inp/'baseline.json').get('sam_cache'):
        cache=Path(read(inp/'baseline.json')['sam_cache'])
    return get_sam(inp/'source.png',root/'cases'/cid/'sam/G',cfg,Crop(size,(0,0,*size),size),'SG',cache)


def local_sam(root,cid,row,cfg):
    folder=root/'cases'/cid/'local_sam'/row['id'];native=Image.open(root/'inputs'/cid/'source.png').convert('RGB')
    b=crop_box([row['bbox_native']],native.size,16);view=local_view(native.size,b,cfg)
    save(view.image(native),folder/'target.png')
    prior=Path(cfg['source_run'])/'cases'/cid/'local_sam'/row['id']/'evidence'/'evidence.json'
    cache=Path(read(prior)['prediction_path']).parent if prior.exists() else None
    masks=get_sam(folder/'target.png',folder/'evidence',cfg,view,'L'+row['id'],cache)
    best,why=select_clean_sam(row,masks);r=copy.deepcopy(row)
    if best is not None:
        mb=best['bbox_native'];r['bbox_native']=[max(0,mb[0]-1),max(0,mb[1]-1),min(native.width-1,mb[2]+1),min(native.height-1,mb[3]+1)]
        r.update(bbox_source='sam',sam_ids=[best['id']],reason='GLM identity retained; unique local SAM mask tightens bbox')
    elif why=='competing_masks':
        r['review_reasons'].append(why)
    else:
        # No mask does not mean no arrow. It must not silently certify an
        # offset box as finished, either: send a small relocalization task.
        r['reason']='No unique local SAM support; GLM relocalization required'
        r['review_reasons'].append('local_sam_no_support')
    write(folder/'selection.json',{'before':row['bbox_native'],'after':r['bbox_native'],'reason':why,
        'selected_mask_id':best['id'] if best else None,'all_masks':len(masks)})
    return r,masks


def extra_candidates(masks, rows, cfg):
    proposals,audit=sam_only_observations(masks,rows);index={m['id']:m for m in masks}
    coverage={a['mask_id']:a['fraction_inside_glm_boxes'] for a in audit}
    eligible=[]
    for row in proposals:
        m=index[row['proposal_mask_id']]
        if m['score']<cfg['sam_extra_min_score'] or coverage[m['id']]>1-cfg['sam_extra_uncovered_fraction']:continue
        row['score']=m['score'];eligible.append(row)
    merged,merge_audit=compress(eligible,semantic_only=cfg.get('strict_identity_semantics',False))
    merged.sort(key=lambda r:(-r['score'],-area(r['bbox_native']),r['id']))
    limit=cfg.get('sam_extra_limit')
    selected=merged if limit is None else merged[:limit]
    deferred=[] if limit is None else merged[limit:]
    for i,r in enumerate(selected,1):
        # These IDs remain disjoint from the GLM compression IDs.
        r['id']=f'S{i:04d}';r['review_reasons']=['sam_proposal'];r['sam_only']=True
    return selected,{'raw_nomination':audit,'eligible_raw':len(eligible),'compression':merge_audit,
        'selected':selected,'deferred':deferred,'deferred_reason':'speed-first discovery budget; not declared non-arrows'}


def repair(root,cid,spec,index,masks,scope,model):
    folder=root/'cases'/cid/'light_refine'/spec['batch_id'];cfg=model.config
    if (folder/'result.json').exists():return read(folder/'result.json')
    inp=root/'inputs'/cid;native=Image.open(inp/'source.png').convert('RGB');b=spec['crop']
    view=local_view(native.size,b,cfg)
    focus=[index[k] for k in spec['focus_ids']];context=[index[k] for k in spec['context_ids']]
    save(view.image(native),folder/'target.png')
    render_labels(folder/'target.png',focus,folder/'boxes.png',view,context=context,mode='request')
    selected,total=select_masks(masks,focus,view,cfg['sam_evidence_max'])
    overlay_masks(folder/'target.png',selected,folder/'sam.png')
    def short(r):
        o=public(r,view)
        # Old endpoint descriptions anchored the model to a partial line or the
        # wrong repeated arrow. Let local pixels establish identity/extent.
        o.pop('locator',None);o.pop('extent',None)
        o['bbox']=[round(x) for x in o['bbox']]
        if r.get('network_region'):
            o['group_scope']=r['locator']
        return o
    short_focus,short_context,alias_to_id=short_request_rows([short(r) for r in focus],[short(r) for r in context])
    write(folder/'visual_aliases.json',{'alias_to_original_id':alias_to_id,'labels_and_prompt_use_same_aliases':True})
    use_global=spec.get('network_region',False) or any(r['extent']=='partial' for r in focus)
    q=repair_prompt(view.input_size,short_focus,network=spec.get('network_region',False),global_context=use_global,sam=bool(selected))
    write(folder/'packet.json',{**spec,'view':view.metadata(native.size)})
    write(folder/'sam_evidence.json',{'selected':[{k:v for k,v in m.items() if k!='projected'} for m in selected],'relevant_total':total})
    images=[('TARGET',folder/'target.png'),('BOXES',folder/'boxes.png')]
    if use_global:images.insert(0,('GLOBAL',inp/'G/target.png'))
    if selected:images.append(('SAM',folder/'sam.png'))
    def validate_aliases(answer):
        return normalize_answer(answer,view.input_size,alias_to_id,restore_answer_ids)
    answer=restore_answer_ids(model.ask(folder/'request',q,images,validate_aliases),alias_to_id)
    write(folder/'extra_discoveries.json',{'extras':answer.get('extras',[]),'rejected':answer.get('rejected_extras',[]),
          'automatic_promotion':False,'reason':'Separate from numbered-focus corrections; preserve for later discovery'})
    objects=materialize(answer,focus,selected,view,'repair/'+spec['batch_id'])
    # One bounded, wider inspection when a crop cuts a real line. The old
    # request and response remain intact; never loop until a preferred answer.
    needs_extent=any(o['extent']=='partial' and o.get('extent_audit',{}).get('touches_internal_boundary') for o in objects)
    if needs_extent and not spec.get('extent_pass') and cfg.get('extent_refine_passes',1)>0 and (cfg.get('network_extent_expansion',False) or not spec.get('network_region')):
        wider={**spec,'batch_id':spec['batch_id']+'_extent','crop':[0,0,*native.size],'extent_pass':True}
        try:
            result=repair(root,cid,wider,index,masks,scope,model)
            result={**result,'extent_parent_batch':spec['batch_id']}
            write(folder/'result.json',result)
            return result
        except Exception as exc:
            write(folder/'extent_failure.json',{'error_type':type(exc).__name__,'error':str(exc),
                'fallback':'retain original partial response'})
    guards=[]
    from collections import Counter
    counts=Counter(s for o in objects for s in o['source_ids'])
    for o in objects:o['split_families']=[spec['batch_id']+'/'+sid for sid in o['source_ids'] if counts[sid]>1]
    for old in focus:
        proposed=[o for o in objects if old['id'] in o['source_ids']]
        if old['kind']!='single' or not proposed:continue
        support,why=select_clean_sam(old,masks)
        if support is None:continue
        b=support['bbox_native'];anchor=[(b[0]+b[2])/2,(b[1]+b[3])/2]
        if any(o['bbox_native'][0]-2<=anchor[0]<=o['bbox_native'][2]+2 and o['bbox_native'][1]-2<=anchor[1]<=o['bbox_native'][3]+2 for o in proposed):continue
        retained=copy.deepcopy(old);retained.update(id='coverage_guard/'+old['id'],source_ids=[old['id']],uncertain=True,
            reason='Correction moved away from original local line evidence; retain candidate pending review')
        objects.append(retained);guards.append({'source_id':old['id'],'mask_id':support['id'],'anchor_native':anchor})
    for unresolved in answer['unresolved']:
        o=copy.deepcopy(index[unresolved['source_id']]);o.update(id='pending/'+o['id'],source_ids=[unresolved['source_id']],uncertain=True,reason=unresolved['reason'])
        objects.append(o)
    for o in objects:
        clusters=list(o['source_ids']);o['compressed_source_ids']=clusters
        o['source_ids']=sorted({s for c in clusters for s in index[c]['source_ids']})
        o['reviewed_by_glm']=True
    removed=[{'source_id':sid,'reason':x['reason']} for x in answer['removed'] for sid in index[x['source_id']]['source_ids']]
    result={'objects':objects,'removed':removed,'focus_ids':spec['focus_ids'],'coverage_guards':guards}
    write(folder/'result.json',result);render_labels(folder/'target.png',objects,folder/'overlay.png',view)
    return result


def run_case(root,cid,model,scope):
    folder=root/'cases'/cid;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'finished.json').exists():return read(folder/'result.json')
    started=time.time();cfg=model.config;inp=root/'inputs'/cid;views=read(inp/'views.json')['views']
    with Image.open(inp/'source.png') as image:size=image.size
    issues=[];timings={};observations=[];masks=[]
    def status(phase,**extra):
        write(folder/'status.json',{'state':'running','phase':phase,'updated_at':now(),**extra})
    status('discovery');t=time.time()
    if cfg.get('upstream_replay_run'):
        origin=Path(cfg['upstream_replay_run'])
        replay=[]
        for v in views:
            previous=origin/'cases'/cid/'discovery'/v['view_id']/'observations.json'
            if sha256(inp/'source.png')!=sha256(origin/'inputs'/cid/'source.png'):
                raise RuntimeError('Upstream replay pixels differ')
            old_view=next(x for x in read(origin/'inputs'/cid/'views.json')['views'] if x['view_id']==v['view_id'])
            if v['transform']!=old_view['transform']:raise RuntimeError('Upstream replay frame differs')
            rows=read(previous);observations.extend(rows)
            write(folder/'discovery'/v['view_id']/'observations.json',rows)
            replay.append({'source':str(previous),'sha256':sha256(previous)})
        masks=global_sam(root,cid,cfg)
        write(folder/'upstream_replay.json',{'policy':'Controlled repair ablation: pin historical discovery observations',
            'same_current_prompt':False,'new_discovery_calls':0,'sources':replay})
    else:
        with ThreadPoolExecutor(max_workers=cfg.get('view_workers',16)) as fallback:
            submit=lambda fn,*args:model.task_pool.submit(cid,fn,*args) if hasattr(model,'task_pool') else fallback.submit(fn,*args)
            jobs={submit(discover_view,root,cid,v,views[0],scope,model):('discovery',v['view_id']) for v in views}
            jobs[submit(global_sam,root,cid,cfg)]=('sam','G')
            for f in as_completed(jobs):
                stage,vid=jobs[f]
                try:
                    result=f.result()
                    if stage=='discovery':observations.extend(result)
                    else:masks=result
                except Exception as exc:
                    issue={'stage':stage,'view':vid,'error_type':type(exc).__name__,'error':str(exc)};issues.append(issue)
                    write(folder/'failures'/f'{stage}_{vid}.json',issue)
                status('discovery',returned_views=sum((folder/'discovery'/v['view_id']/'observations.json').exists() for v in views),expected_views=len(views))
    timings['discovery_and_global_sam_seconds']=time.time()-t
    write(folder/'glm_observations.json',observations)
    rows,merge_audit=compress(observations,semantic_only=cfg.get('strict_identity_semantics',False))
    if cfg.get('strict_identity_semantics',False):network_audit=[]
    else:rows,network_audit=network_units(rows)
    rows=classify(rows);rows,fragment_audit=link_fragments(rows)
    write(folder/'network_units.json',network_audit);write(folder/'fragment_links.json',fragment_audit)
    merge_audit['post_network_unit_count']=len(rows);write(folder/'compression.json',merge_audit)
    write(folder/'compressed_objects.json',rows);render_labels(inp/'G/target.png',rows,folder/'compressed.png',frame(views[0]['transform']))
    status('local_sam_tightening',compressed_count=len(rows));t=time.time()
    targets=[r for r in rows if r['kind']=='single' and not r['review_reasons']]
    replacements={}
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs={pool.submit(local_sam,root,cid,row,cfg):row for row in targets}
        for f in as_completed(jobs):
            old=jobs[f]
            try:
                r,local_masks=f.result();replacements[old['id']]=r;masks.extend(local_masks)
            except Exception as exc:
                issues.append({'stage':'local_sam','object_id':old['id'],'error_type':type(exc).__name__,'error':str(exc)})
    rows=[replacements.get(r['id'],r) for r in rows]
    # Only the global SAM contributes discovery proposals; local crops serve
    # precise bbox refinement and cannot flood discovery with nearby borders.
    extra,extra_audit=extra_candidates([m for m in masks if m['id'].startswith('SG/')],rows,cfg)
    rows.extend(extra);write(folder/'sam_supplement.json',extra_audit);write(folder/'sam_evidence.json',masks)
    timings['local_sam_seconds']=time.time()-t
    index={r['id']:r for r in rows};plan=packet_plan(rows,size,cfg,repair_batches)
    write(folder/'repair_plan.json',plan);write(folder/'all_compressed_objects.json',rows)
    render_labels(inp/'G/target.png',rows,folder/'provisional.png',frame(views[0]['transform']))
    status('selective_glm_repair',batches=len(plan),objects_needing_review=sum(bool(r['review_reasons']) for r in rows))
    t=time.time();objects=[copy.deepcopy(r) for r in rows if not r['review_reasons']];removed=[];handled=set()
    for r in objects:r['reviewed_by_glm']=False
    with ThreadPoolExecutor(max_workers=cfg.get('refine_workers',16)) as fallback:
        submit=lambda fn,*args:model.task_pool.submit(cid,fn,*args) if hasattr(model,'task_pool') else fallback.submit(fn,*args)
        jobs={submit(repair,root,cid,s,index,masks,scope,model):s for s in plan}
        for f in as_completed(jobs):
            spec=jobs[f]
            try:
                result=f.result();objects.extend(result['objects']);removed.extend(result['removed'])
            except Exception as exc:
                issue={'stage':'light_refine','batch':spec['batch_id'],'error_type':type(exc).__name__,'error':str(exc)};issues.append(issue)
                write(folder/'light_refine'/spec['batch_id']/'failed.json',issue)
                objects.extend(pending_object(index[sid],'Repair failed; compressed observation retained') for sid in spec['focus_ids'])
            handled.update(spec['focus_ids'])
            preview=objects+[pending_object(index[sid],'Awaiting repair; original candidate remains visible',prefix='WAIT') for s in plan for sid in s['focus_ids'] if sid not in handled]
            write(folder/'preview_objects.json',preview)
            render_labels(inp/'G/target.png',preview,folder/'repair_progress.png',frame(views[0]['transform']))
    timings['selective_repair_seconds']=time.time()-t
    # A final numerical pass removes duplicates that converge after repair.
    # Explicit model splits with common sources cannot be merged back.
    objects,envelope_audit=preserve_chain_scope(objects,observations)
    write(folder/'chain_scope_envelopes.json',envelope_audit)
    final,post_audit=compress(objects,protect_splits=True,semantic_only=cfg.get('strict_identity_semantics',False))
    residual_audit={'decisions':[],'applied':[],'errors':[],'conflicts':[],'deferred':[]}
    if cfg.get('residual_review_enabled',False):
        status('residual_review',pre_review_identity_count=len(final))
        render_labels(inp/'G/target.png',final,folder/'pre_residual.png',frame(views[0]['transform']))
        t=time.time();final,residual_audit=reconcile(root,cid,final,model);timings['residual_review_seconds']=time.time()-t
        issues.extend({'stage':'residual_review',**e} for e in residual_audit['errors'])
    for i,o in enumerate(sorted(final,key=lambda r:(r['bbox_native'][1],r['bbox_native'][0])),1):
        o['display_id']=f'I{i:04d}';o['needs_split']=o['kind']=='dense_region'
        o['geometry_ready']=False;o['handoff_stage']='discovery_only_not_geometry_validated'
    all_sources={r['id']:r for r in observations}
    for r in extra:
        for sid in r['source_ids']:all_sources[sid]=r
    removed_by={r['source_id']:r['reason'] for r in removed};ledger=[]
    for sid,old in all_sources.items():
        owners=[o for o in final if sid in o['source_ids']]
        if not owners and sid not in removed_by:raise AssertionError('Silently lost source: '+sid)
        ledger.append({'source_id':sid,'owner_ids':[o['display_id'] for o in owners],
            'status':'split' if len(owners)>1 else 'assigned' if owners else 'excluded',
            'before_bbox':old['bbox_native'],'reasons':[o['reason'] for o in owners] if owners else [removed_by[sid]]})
    result={'case_id':cid,'state':'completed_with_issues' if issues else 'completed','completed_at':now(),
        'objects':final,'issues':issues,'input_observation_count':len(observations),'compressed_count':len(index)-len(extra),
        'sam_proposals_reviewed':len(extra),'sam_proposals_deferred':len(extra_audit['deferred']),
        'repair_batches':len(plan),'direct_objects_before_final_merge':sum(not r['review_reasons'] for r in rows),
        'local_sam_crops':len(targets),'local_sam_tightened':sum(r['bbox_source']=='sam' for r in replacements.values()),
        'output_identity_count':len(final),'pending_count':sum(o['uncertain'] or o['extent']=='partial' or o['kind'] in {'dense_region','sam_proposal'} for o in final),
        'timings':{**timings,'total_seconds':time.time()-started},'source_ledger':ledger,'excluded_observations':removed,
        'post_compression':post_audit,'residual_review':residual_audit,'chain_scope_envelopes':envelope_audit,
        'formal_accuracy_measured':False,'ownership_semantically_validated':False}
    render_labels(inp/'G/target.png',final,folder/'overlay.png',frame(views[0]['transform']))
    write(folder/'result.json',result);write(folder/'finished.json',{'completed_at':now()});write(folder/'status.json',{'state':result['state'],'phase':'finished','updated_at':now()})
    publish(root);return result


def execute(root):
    lock=(root/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for rel,digest in read(root/'manifest.json')['files'].items():
        if sha256(root/rel)!=digest:raise RuntimeError('Frozen artifact changed: '+rel)
    cfg=read(root/'config.json');assert cfg['max_tokens']==113000 and 'reasoning_effort' not in cfg
    write(root/'process.json',{'pid':os.getpid(),'started_at':now()})
    model=Model(cfg,run_dir=root,reuse_from=Path(cfg.get('discovery_reuse_run',cfg['source_run'])),system=SYSTEM)
    model.task_pool=FairPool(cfg['api_concurrency'])
    preflight(root,model);scope=(root/'constraints_snapshot/scope.txt').read_text()
    stop=threading.Event()
    def watch():
        while not stop.wait(30):
            try:publish(root)
            except Exception as exc:print('report_error',type(exc).__name__,str(exc),flush=True)
    threading.Thread(target=watch,daemon=True).start()
    order=read(root/'execution_order.json')['case_ids'];records=[];queue=iter(order);launches=[]
    with ThreadPoolExecutor(max_workers=cfg['case_workers']) as pool:
        jobs={}
        def fill():
            while len(jobs)<cfg['case_workers'] and not model.blocked.is_set() and not (root/'STOP_AFTER_BATCH').exists():
                cid=next(queue,None)
                if cid is None:break
                jobs[pool.submit(run_case,root,cid,model,scope)]=cid;launches.append({'case_id':cid,'started_at':now()})
            write(root/'launch_order.json',launches)
            write(root/'progress.json',{'state':'running','total':len(order),'ended':len(records),'records':records,
                'scheduler':'rolling_8','current_cases':list(jobs.values()),'updated_at':now()})
        fill()
        while jobs:
            done,_=wait(jobs,return_when=FIRST_COMPLETED)
            for f in done:
                cid=jobs.pop(f)
                try:
                    r=f.result();record={k:r[k] for k in ['case_id','state','output_identity_count','pending_count','repair_batches','timings']}
                except Exception as exc:
                    record={'case_id':cid,'state':'failed','error_type':type(exc).__name__,'error':str(exc)}
                    write(root/'cases'/cid/'failed.json',{**record,'traceback':traceback.format_exc()})
                    write(root/'cases'/cid/'status.json',record)
                records.append(record);print(record,flush=True)
            fill();publish(root)
    state='completed' if len(records)==len(order) and all(r['state']=='completed' for r in records) else 'completed_with_issues' if len(records)==len(order) else 'stopped_between_batches'
    write(root/'progress.json',{'state':state,'total':len(order),'ended':len(records),'records':records,'updated_at':now()})
    write(root/'finished.json',{'state':state,'completed_at':now()});stop.set();model.task_pool.shutdown();publish(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--prepare',action='store_true');parser.add_argument('--previous',type=Path);parser.add_argument('--benchmark',type=Path);args=parser.parse_args()
    if args.prepare:prepare(args.run_dir.resolve(),args.previous.resolve(),args.benchmark.resolve())
    else:
        if Path(__file__).resolve().parent!=args.run_dir.resolve()/'code':raise RuntimeError('Run frozen code only')
        execute(args.run_dir.resolve())
