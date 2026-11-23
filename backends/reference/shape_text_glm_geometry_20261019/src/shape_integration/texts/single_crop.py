"""Dispatch to two-stage crop review; retain the old single-pass mode explicitly."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import shutil
import threading
from pathlib import Path

from PIL import Image

from ..common import read,save,sha,compare,usage,intersect
from . import layout,colors,export,local_content,ocr_multiline
from . import render_feedback as feedback
from .bbox_color_review import DEFAULT_MAX_SCORE_DROP,retain_nonregressing

POLICY='single_crop_low_score_v1'
FIELDS=(*local_content.FIELDS,*local_content.STYLE_FIELDS,'visual_line_count','line_layout_status')


def changes(original, revised, target_ids):
    """Build atomic parent updates. High-score context can never enter updates."""
    known={r['id']:r for r in original};parts={ident:[] for ident in target_ids}
    for row in revised:
        parent=row['id'] if row['id'] in known else row.get('parent_id')
        if parent not in known:raise ValueError('Crop invented an unrelated text ID')
        if parent in parts:parts[parent].append(deepcopy(row))
    result={}
    for ident,rows in parts.items():
        if len(rows)==1 and rows[0]['id']==ident and all(
                rows[0].get(k)==known[ident].get(k) for k in FIELDS):continue
        for row in rows:
            for key in ('layout','frame_px','font_size_px','resolved_font','text_insets_px','glyph_registration'):
                row.pop(key,None)
        result[ident]=rows
    return result


def finish(source,output,baseline,opt,runtime,*,threshold=.80,max_score_drop=DEFAULT_MAX_SCORE_DROP,
           base=None,parameters=None,progress=lambda *a,**k:None,recognition=None):
    if opt.get('crop_review_mode','two_stage') != 'legacy_single':
        from .two_stage_crop import finish as finish_two_stage
        return finish_two_stage(source,output,baseline,opt,runtime,threshold=threshold,
            max_score_drop=max_score_drop,base=base,parameters=parameters,
            progress=progress,recognition=recognition)
    with Image.open(source) as image:reference=image.convert('RGB')
    with Image.open(baseline/'native.png') as image:before=image.convert('RGB')
    scene=read(baseline/'scene.json');rows=scene['texts'];by_id={r['id']:r for r in rows}
    slots=threading.BoundedSemaphore(1)
    degraded={r['id'] for r in rows if r.get('layout',{}).get('measurement',{}).get('preview_degraded')}
    scores={r['id']:feedback.evidence(reference,before,before,[r]) for r in rows}
    low=[r['id'] for r in rows if r['id'] not in degraded and scores[r['id']]['before']['score']<threshold]
    frozen=set(by_id)-set(low)
    save(output/'selection.json',{'policy':POLICY,'threshold':threshold,'comparison':'score < threshold',
        'total_texts':len(rows),'low_score_ids':low,'automatic_ids':sorted(frozen),'scores':scores,
        'degraded_preview_ids':sorted(degraded)})
    warnings=deepcopy((recognition or {}).get('regions_needing_review',[]))
    warnings.extend({'stage':'baseline_font_metrics','reason':e} for e in scene.get('errors',[]))
    review=output/'low_score_crop';updates={};group_count=0;review_unresolved=[]
    if low:
        progress('low_score_joint_crop',low_score_texts=len(low))
        try:
            if runtime is None:raise ValueError('API credential required for low-score crop review')
            with ThreadPoolExecutor(max_workers=4) as pool:
                revised,errors=local_content.refine(deepcopy(rows),source,review,runtime,
                    {**opt,'geometry_max_targets':6},pool,
                    lambda stage,done,total:progress('low_score_joint_crop',done=done,total=total),
                    target_ids=low,rendered=baseline/'native.png',scores=scores,collision_passes=0)
            report=read(review/'summary.json');group_count=report['groups']
            review_unresolved=report.get('unresolved_ids',[])
            warnings.extend({'stage':'single_crop','reason':e} for e in errors)
            updates=changes(rows,revised,low)
            save(output/'crop_recognition.json',{'texts':revised,'policy':POLICY,
                 'reviewed_ids':low,'read_only_ids':sorted(frozen)})
        except Exception as exc:
            warnings.append({'stage':'single_crop','reason':str(exc),'fallback':'baseline'})
            review_unresolved=list(low)
    else:
        save(review/'plan.json',{'policy':POLICY,'groups':[],'target_ids':[],
             'max_collision_passes':0,'reason':'no_low_score_targets'})
    if updates:
        try:
            progress('fit_joint_crop_candidates',candidates=len(updates))
            candidates=[row for parts in updates.values() for row in parts]
            if candidates:
                fitted,errors=layout.fit_all(candidates,output/'typography',slots,
                    {**opt,'pixel_reflow_enabled':False,'glyph_registration_enabled':False},source=source)
                warnings.extend({'stage':'candidate_layout','reason':e} for e in errors)
                fitted,audit=colors.recover(fitted,source,output/'color_assets')
                save(output/'color_recovery.json',audit)
                updates=ocr_multiline.regroup_fitted(updates,fitted)
            bad={i for i,parts in updates.items() if any(
                row.get('layout',{}).get('measurement',{}).get('preview_degraded') for row in parts)}
            for ident in bad:
                updates.pop(ident)
                warnings.append({'id':ident,'stage':'candidate_preview','fallback':'baseline'})
        except Exception as exc:
            warnings.append({'stage':'candidate_layout','reason':str(exc),'fallback':'baseline'})
            updates={}
    conflicts=feedback.new_collisions(rows,feedback.replace(rows,updates))
    blocked=ocr_multiline.conflict_parents(updates,conflicts)
    updates={i:v for i,v in updates.items() if i not in blocked}
    rounds=[];state={}
    def evaluate(candidates):
        folder=output/'cpu_guard'/f'round_{len(rounds)+1:02d}';folder.mkdir(parents=True,exist_ok=True)
        chosen=feedback.replace(rows,candidates)
        progress('cpu_candidate_render',round=len(rounds)+1,candidates=len(candidates))
        if candidates:export.final(chosen,source,folder,slots,opt)
        else:
            for name in ('text.pptx','native.png','native.pdf'):shutil.copy2(baseline/name,folder/name)
        with Image.open(folder/'native.png') as image:actual=image.convert('RGB')
        metrics={i:feedback.evidence(reference,before,actual,[by_id[i]],parts) for i,parts in candidates.items()}
        # Protect unchanged high-score text against low-score edits overlapping it.
        neighbors={i:feedback.evidence(reference,before,actual,[by_id[i]]) for i in frozen}
        for metric in metrics.values():
            metric['neighbor_regressions']=[i for i,m in neighbors.items()
                if m['after']['score']<m['before']['score']-1e-9 and intersect(metric['roi'],m['roi'])>0]
        save(folder/'scene.json',{'texts':chosen,'source_size':list(reference.size)})
        save(folder/'scores.json',metrics);save(folder/'context_scores.json',neighbors)
        rounds.append(folder);state.update(folder=folder,image=actual,rows=chosen)
        return metrics
    try:accepted,history=retain_nonregressing(updates,evaluate,max_score_drop)
    except Exception as exc:
        warnings.append({'stage':'candidate_render','reason':str(exc),'fallback':'baseline'})
        accepted={};history=[{'error':str(exc),'fallback':'baseline'}]
        rounds.append(output/'cpu_guard/failed_attempt');evaluate({})
    save(output/'cpu_guard/history.json',history)
    final=output/'final'
    if final.exists():raise ValueError('Final already exists; use a fresh output')
    shutil.copytree(state['folder'],final)
    if base:
        export.final(state['rows'],source,final,slots,opt,base)
        save(final/'scene.json',{'texts':state['rows'],'source_size':list(reference.size)})
    save(final/'provenance.json',{'image_sha256':sha(source),'parameters':parameters or {},'review_policy':POLICY})
    metrics={r['id']:feedback.evidence(reference,before,state['image'],[r],accepted.get(r['id'],[])) for r in rows}
    for ident in degraded:metrics[ident].update(reliable=False,excluded_from_fidelity=True)
    low_final=[i for i,m in metrics.items() if m['after']['score']<threshold or not m['reliable']]
    quality='needs_review' if low_final or warnings else 'proxy_checks_passed'
    save(final/'quality.json',{'quality_status':quality,'low_score_ids':low_final,'scores':metrics,
         'regions_needing_review':warnings,'note':'CPU scores are proxies, not ground-truth visual approval.'})
    compare([source,baseline/'native.png',final/'native.png'],
        ['Original','Global baseline','Single low-score crop + CPU gate'],output/'comparison.png',width_cap=1800)
    accounting=usage(output)
    if (output/'recovery_seed.json').exists():
        cached={str(Path(name).parent) for name in read(output/'recovery_seed.json')['files']
                if name.endswith('/response_0.json')}
        new=[c for c in accounting['calls'] if c['call'] not in cached]
        accounting.update(reused_cached_calls=len(accounting['calls'])-len(new),new_calls=new,
            new_totals={key:sum(c[key] for c in new) for key in accounting['totals']})
    save(output/'usage.json',accounting)
    known_multiline={r['id'] for r in rows if r.get('line_layout_status')=='requires_split'}
    resolved={i for i,v in accepted.items() if len(v)>1}
    summary={'total_texts':len(rows),'final_texts':len(state['rows']),'geometry_backend':'glm',
        'review_policy':POLICY,'review_rounds':int(bool(low)),'low_score_texts':len(low),'groups':group_count,
        'accepted_ids':list(accepted),'unresolved_ids':sorted((set(low)-set(accepted))|degraded),
        'read_only_ids':sorted(frozen),'final_low_score_ids':low_final,'quality_status':quality,
        'regions_needing_review':warnings,'known_multiline_ids':sorted(known_multiline),
        'resolved_multiline_ids':sorted(resolved),'unresolved_multiline_ids':sorted(known_multiline-resolved),
        'crop_unresolved_ids':review_unresolved,'degraded_formula_preview_ids':sorted(degraded),
        'collision_rejected_ids':sorted(blocked),'max_score_drop':max_score_drop,
        'score_policy':'same ROI; reject decreases beyond explicit tolerance; protect read-only neighbors',
        'metric_is_proxy_not_ground_truth':True}
    save(output/'summary.json',summary);progress('completed',**summary)
    return summary
