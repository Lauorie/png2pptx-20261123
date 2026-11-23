"""Batch localization and image conflict removal; no geometry fitting or editing."""
import copy
import csv
import fcntl
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from png2pptx.frames import scale_box
from .common import *
from .shape import run_shape
from .image_conflicts import run_filter
from . import image_bridge

def finish_case(root,source,item,shapes,inventory,errors,config,runtime=None,*,original_assets=None,prepare_only=False):
    out=root/'dedup';out.mkdir(exist_ok=True);image=Image.open(source).convert('RGB')
    assets=original_assets if original_assets is not None else ([] if inventory is None else [{**a,'bbox_2d':scale_box(a['bbox'],inventory['input_size'],image.size)} for a in inventory['assets']])
    matches=run_filter(shapes['objects'],assets,
        {'path':str(source),'sha256':item['source_sha256'],'size':list(image.size)},out/'model_review',runtime,config,prepare_only=prepare_only)
    filtered={**shapes,'objects':matches['kept'],'dedup_provenance':{'threshold':matches['threshold'],'comparison':'>',
        'mode':matches['mode'],'image_inventory':str(root/'images/inventory.json'),
        'model_review_calls':matches['model_request_calls'],'review_candidates':matches['review_candidates'],
        'deferred_ids':matches['deferred_ids'],'review_pending':matches['review_pending'],
        'manual_overrides':[],'image_detection_available':inventory is not None or original_assets is not None}}
    save(out/'shapes_filtered.json',filtered);save(root/'geometry.json',filtered)
    save(out/'image_boxes_original.json',{'source_sha256':item['source_sha256'],'assets':assets})
    for key in ['decisions','pairs','removed']:save(out/(key+'.json'),matches[key])
    save(out/'review_requests_backend.json',matches['review_requests'])
    if matches['mode']=='rule_only':
        save(out/'model_review_not_called.json',{'enabled':False,'calls':0,'requests':matches['review_requests']})
    with (out/'pairwise_iou.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['shape_id','image_id','iou','above_threshold']);w.writeheader();w.writerows(matches['pairs'])
    overlay(image,matches['kept'],root/'final_overlay.png');overlay(image,[{'id':a['id'],'bbox_2d':a['bbox_2d']} for a in assets],root/'image_overlay.png','#008765')
    compare([root/'shapes/global_overlay.png',root/'shapes/enhanced_overlay.png',root/'final_overlay.png'],['Global','Enhanced','After image conflict filter'],root/'comparison.png')
    s=read(root/'shapes/summary.json');image_status=read(root/'images/status.json') if (root/'images/status.json').exists() else {}
    summary={'id':item['id'],'source_sha256':item['source_sha256'],'state':'failed' if errors else 'prepared' if prepare_only else 'completed_with_unresolved' if matches['deferred_ids'] else 'completed',
             'global_shapes':s['global_candidates'],'regions':s['regions'],'enhanced_shapes':len(shapes['objects']),
             'image_assets':len(assets),'rule_removed':len(matches['removed']) if matches['mode']=='rule_only' else 0,
             'model_removed':len(matches['removed']) if matches['mode']=='model_review' else 0,
             'image_removed':len(matches['removed']),'removed_ids':[o['id'] for o in matches['removed']],
             'final_shapes':len(matches['kept']),'shape_issues':s['issues'],'local_failures':s['local_failures'],
             'image_coverage_status':inventory.get('coverage_status') if inventory else 'failed',
             'image_boundary_status':inventory.get('boundary_review',{}).get('status') if inventory else 'failed',
             'image_unresolved_ids':inventory.get('boundary_review',{}).get('unresolved_ids',[]) if inventory else [],
             'image_stage_status':image_status.get('stage_status'),'errors':errors,'model_conflict_review_calls':matches['model_request_calls'],
             'model_conflict_review_batches':matches['reviewer_calls'],'model_conflict_review_candidates':matches['review_candidates'] if matches['mode']=='model_review' else 0,
             'model_conflict_review_failed_batches':matches['review_failed_batches'],
             'image_conflict_deferred_ids':matches['deferred_ids'],'image_conflict_review_pending':matches['review_pending'],
             'dedup_mode':matches['mode'],'iou_threshold':matches['threshold'],'fitting_performed':False,'visual_accuracy_verified':False}
    u=usage(root);save(root/'usage.json',u);summary['usage']=u['totals'];save(root/'summary.json',summary)
    return summary

def case(root,item,workspace,runtime):
    folder=root/'cases'/item['id'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'summary.json').exists():return read(folder/'summary.json')
    source=workspace/item['original'];input_path=workspace/item['input'];assert sha(source)==item['source_sha256'];assert sha(input_path)==item['input_sha256']
    save(folder/'status.json',{'state':'running','started_utc':now(),'case':item['id']});errors={};shapes=None;inventory=None
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            sf=pool.submit(run_shape,source,input_path,folder/'shapes',runtime)
            imf=pool.submit(image_bridge.detect,source,folder/'images',runtime.config)
            try:shapes=sf.result()
            except Exception as exc:errors['shape']=str(exc)
            try:inventory=imf.result()
            except Exception as exc:errors['image']=str(exc)
        if shapes is None:
            summary={'id':item['id'],'state':'failed','errors':errors,'usage':usage(folder)['totals']};save(folder/'summary.json',summary)
        else:summary=finish_case(folder,source,item,shapes,inventory,errors,runtime.config,runtime)
        save(folder/'status.json',{'state':summary['state'],'summary':summary,'updated_utc':now()});return summary
    except Exception as exc:
        save(folder/'status.json',{'state':'failed','error':str(exc),'updated_utc':now()});raise

def audit_run(root):
    rows=[]
    for f in (root/'cases').glob('*/shapes/calls/review_*/request.json'):
        req=read(f);p=req['prompt'];ims=req['images']
        assert len(ims)==2
        assert all(set(o)=={'id','label','kind','role'} for o in p['candidates'])
        assert not ({'region','frame','global_size','source_size','crop_offset','bbox_hint'} & set(p))
        assert all(sha(x['path'])==x['sha256'] for x in ims)
        sizes=[list(Image.open(x['path']).size) for x in ims];assert sizes[0]==sizes[1]==p['local_image_size'];assert max(sizes[0])==2048
        rows.append({'case':f.parents[3].name,'call':f.parent.name,'two_aligned_images':True,'no_numeric_old_bbox_or_global_transform':True,'image_hashes_verified':True})
    edits=list((root/'cases').glob('*/images/calls/editor/*/request.json'));assert not edits
    conflict_calls=list((root/'cases').glob('*/dedup/model_review/calls/*/request.json'))
    for f in conflict_calls:
        req=read(f);p=req['prompt'];assert len(req['images'])==2
        assert all(set(c)=={'shape_id','image_ids'} for c in p['candidates'])
        assert not {'bbox_2d','iou','frame','source','target'} & set(p)
        assert all(sha(x['path'])==x['sha256'] for x in req['images'])
    return {'passed':True,'local_review_calls_checked':len(rows),'local_requests':rows,'image_edit_calls':0,'fitting_calls':0,'model_conflict_review_calls':len(conflict_calls)}

def report(root,items,summaries):
    ordered=[summaries[i['id']] for i in items if i['id'] in summaries]
    totals={k:sum(s.get(k,0) for s in ordered) for k in ['global_shapes','enhanced_shapes','image_assets','rule_removed','model_removed','image_removed','final_shapes','model_conflict_review_candidates','model_conflict_review_calls']}
    state={'state':'completed' if len(ordered)==len(items) else 'running','processed':len(ordered),'total':len(items),'cases':ordered,'totals':totals,'updated_utc':now()}
    if len(ordered)==len(items) and any(s['state']=='failed' for s in ordered):state['state']='completed_with_failures'
    elif len(ordered)==len(items) and any(s['state']=='prepared' for s in ordered):state['state']='prepared'
    elif len(ordered)==len(items) and any(s['state']=='completed_with_unresolved' for s in ordered):state['state']='completed_with_unresolved'
    save(root/'summary.json',state)
    lines=['# 定位与 image 去重','',f'已处理 {len(ordered)}/{len(items)} 张。模型审核模式下，先按严格 IoU 门槛筛选，再批量审核冲突候选；其余直接保留。存疑或审核失败也保留并记录。本轮不运行拟合。','',
           '| 图 | 初检 | 增强后 | image | 去重删除 | 最终 shape | shape 问题记录 | image 覆盖 / 边界 | 查看 |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |']
    for s in ordered:
        if s['state']=='failed':lines.append(f'| {s["id"]} | — | — | — | — | — | — | failed | [状态](cases/{s["id"]}/status.json) |');continue
        lines.append(f'| {s["id"]} | {s["global_shapes"]} | {s["enhanced_shapes"]} | {s["image_assets"]} | {s.get("image_removed",s["rule_removed"])} | {s["final_shapes"]} | {s["shape_issues"]} | {s["image_coverage_status"]} / {s["image_boundary_status"]} | [前后图](cases/{s["id"]}/comparison.png) · [image](cases/{s["id"]}/image_overlay.png) |')
    lines+=['','数量和模型状态不代表视觉准确率。未解决项、无效响应和局部失败均保留在逐图目录中。','',
             '每张图的 `geometry.json` 是本轮去重后的原图坐标候选；`shapes/history.json` 为逐 ID 决定；`dedup/decisions.json` 与 `pairwise_iou.csv` 为去重证据。','',
             '模型模式的 IoU 仅决定审核候选，不直接决定删除；原图、候选框、prompt 和原始响应保存在 `dedup/model_review/`。无图片或对象 ID 特例。']
    for s in ordered:
        if s['state']=='failed':continue
        lines+=['',f'## {s["id"]}','',
                '定位对比（从左到右：全图初检、子图增强、image 冲突过滤后）：','',
                f'![{s["id"]} 定位与去重对比](cases/{s["id"]}/comparison.png)','',
                'image 检测框：','',
                f'![{s["id"]} image 检测结果](cases/{s["id"]}/image_overlay.png)']
    (root/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
    return state

def freeze(workspace,root,config,items,localization_source=None):
    files=list((workspace/'src').rglob('*.py'))+list((workspace/'src').rglob('*.json'))+list((workspace/'vendor/native_qwen').glob('*.py'))+[workspace/'run.py']
    linked=list((workspace/'vendor/png2pptx').rglob('*.py'))
    hashes={str(p.relative_to(workspace)):sha(p) for p in files+linked}
    frozen={'config':config,'images':items,'code_sha256':hashes,'image_code_symlink_target':str((workspace/'vendor/png2pptx').resolve()),'predictions_reused':localization_source is not None}
    if localization_source is not None:frozen['localization_source']=localization_source
    if (root/'manifest.json').exists():
        if read(root/'manifest.json')!=frozen:raise ValueError('Cannot resume after code/config/input changes; use a new run')
    else:
        save(root/'manifest.json',frozen)
        for p in files+linked:
            dest=root/'code_snapshot'/p.relative_to(workspace);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    return frozen

def run(workspace,root,config,items,runtime):
    root.mkdir(parents=True,exist_ok=True);lock=(root/'.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    freeze(workspace,root,config,items);image_bridge.configure(runtime);summaries={}
    with ThreadPoolExecutor(max_workers=config['case_workers']) as pool:
        jobs={pool.submit(case,root,item,workspace,runtime):item for item in items}
        for f in as_completed(jobs):
            item=jobs[f]
            try:summaries[item['id']]=f.result()
            except Exception as exc:summaries[item['id']]={'id':item['id'],'state':'failed','errors':{'runner':str(exc)}}
            state=report(root,items,summaries);print(json.dumps({'processed':state['processed'],'total':state['total'],'case':summaries[item['id']]},ensure_ascii=False),flush=True)
    save(root/'audit.json',audit_run(root));save(root/'usage.json',usage(root))
    report(root,items,summaries)


def replay_case(workspace,root,item,runtime,config,source,provenance,prepare_only):
    """Reapply the gate to immutable localization inventories, without redetection."""
    from .pipeline import import_localization, validate_candidates
    folder=root/'cases'/item['id']; upstream=folder/'localization_input'
    old=import_localization(upstream,source,item,provenance)
    original=workspace/item['original']
    if sha(original)!=item['source_sha256']:raise ValueError('Original input changed')
    if (upstream/'shapes/geometry.json').exists():
        shapes=read(upstream/'shapes/geometry.json')
    else:
        # Older fitting snapshots saved the post-dedup candidates plus removals.
        shapes=read(upstream/'geometry.json')
        removed=read(upstream/'dedup/removed.json') if (upstream/'dedup/removed.json').exists() else []
        shapes={**shapes,'objects':shapes['objects']+removed}
    validate_candidates(shapes,item)
    asset_data=read(upstream/'dedup/image_boxes_original.json')
    if asset_data.get('source_sha256',item['source_sha256'])!=item['source_sha256']:
        raise ValueError('Image assets belong to a different source')
    for name in ['shapes/history.json','shapes/summary.json','shapes/global_overlay.png',
                 'shapes/enhanced_overlay.png','images/status.json','images/inventory.json']:
        src=upstream/name; dst=folder/name
        if src.exists():
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    save(folder/'shapes/geometry.json',shapes)
    if not (folder/'shapes/summary.json').exists():
        save(folder/'shapes/summary.json',{'global_candidates':old.get('global_shapes',len(shapes['objects'])),
            'regions':old.get('regions',0),'issues':old.get('shape_issues',0),'local_failures':old.get('local_failures',0)})
    with Image.open(original) as image:
        for name in ['global_overlay.png','enhanced_overlay.png']:
            if not (folder/'shapes'/name).exists():overlay(image,shapes['objects'],folder/'shapes'/name)
    inventory=read(upstream/'images/inventory.json') if (upstream/'images/inventory.json').exists() else None
    result=finish_case(folder,original,item,shapes,inventory,old.get('errors',{}),config,runtime,
        original_assets=asset_data['assets'],prepare_only=prepare_only)
    save(folder/'status.json',{'state':result['state'],'updated_utc':now()})
    return result


def run_dedup(workspace,root,config,items,runtime,*,source,prepare_only=False):
    from .pipeline import source_manifest
    provenance=source_manifest(source,items)
    provenance['reused_stages']=['shape_localization','image_detection']
    provenance['rerun_stages']=['shape_image_conflict_filter']
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        freeze(workspace,root,config,items,provenance);summaries={}
        report(root,items,summaries)
        with ThreadPoolExecutor(max_workers=config['case_workers']) as pool:
            jobs={pool.submit(replay_case,workspace,root,item,runtime,config,source,provenance,prepare_only):item for item in items}
            for future in as_completed(jobs):
                item=jobs[future]
                try:summaries[item['id']]=future.result()
                except Exception as exc:
                    summaries[item['id']]={'id':item['id'],'state':'failed','errors':{'dedup_replay':str(exc)}}
                    save(root/'cases'/item['id']/'status.json',summaries[item['id']])
                report(root,items,summaries)
                print(json.dumps(summaries[item['id']],ensure_ascii=False),flush=True)
        save(root/'audit.json',audit_run(root));save(root/'usage.json',usage(root/'cases'))
        if runtime is not None:save(root/'api_concurrency.json',runtime.slots.snapshot())
        return report(root,items,summaries)
