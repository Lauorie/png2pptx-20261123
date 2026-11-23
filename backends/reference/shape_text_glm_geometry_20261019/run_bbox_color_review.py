"""CPU >80 keeps text; low-score VLM review changes bbox/color only."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import os
import shutil
import sys
import threading

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'.deps'),str(ROOT/'src'),str(ROOT/'vendor')]
from PIL import Image
from shape_integration.common import read,save,sha,Runtime,compare,usage,intersect
from shape_integration.texts import options,layout,export
from shape_integration.texts import render_feedback as feedback,bbox_color_review as protocol
from shape_integration.texts.controller import frame,mapped_box,local_hints
from shape_integration.texts.local_geometry import union

SYSTEM='Locate tight SOURCE glyph bounds and foreground color for every supplied target before comparing it with PPT. There is no keep decision: return source localization even when the texts appear similar. Report uncertainty only when source localization is unreliable. Return the requested JSON only. Do not change content or font properties.'


def main(args):
    if args.output.exists():raise ValueError('Use a fresh output directory')
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    source=Image.open(args.image).convert('RGB');before=Image.open(args.baseline/'native.png').convert('RGB')
    if source.size!=before.size:raise ValueError('Canvas mismatch')
    if read(args.baseline/'provenance.json')['image_sha256']!=sha(args.image):raise ValueError('Source mismatch')
    baseline=read(args.baseline/'scene.json');rows=baseline['texts'];by_id={r['id']:r for r in rows}
    wanted=getattr(args,'target_ids',None)
    if wanted and (len(set(wanted))!=len(wanted) or set(wanted)-set(by_id)):raise ValueError('Invalid target IDs')
    scope=[by_id[i] for i in wanted] if wanted else rows
    groups,selection=protocol.select(scope,source,before,args.max_groups)
    cfg=read(ROOT/'config.json');cfg['api_concurrency']=args.concurrency;opt=options(cfg)
    key=os.environ.get('OpenAI_API') or os.environ.get('OPENAI_API_KEY') or os.environ.get('BOYUE_API_KEY')
    if groups and not key:raise ValueError('API key required in environment')
    runtime=Runtime(cfg,cfg['expected_boyue_base_url'],key).for_stage(reasoning_effort='low',shape_schema_retries=1)
    args.output.mkdir(parents=True)
    hashes={str(p.resolve()):sha(p) for p in [args.image,args.baseline/'scene.json',args.baseline/'text.pptx',args.baseline/'native.png']}
    save(args.output/'provenance.json',{'policy':protocol.POLICY,'sources':hashes,'scope_ids':[r['id'] for r in scope],
         'legacy_cache_disabled':True,'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'review_rendered_text.py',*ROOT.joinpath('src').rglob('*.py')]}})
    save(args.output/'selection.json',selection)
    def job(index,group):
        folder=args.output/'review'/f'G{index:03d}'
        short=max(min(r['bbox_2d'][2]-r['bbox_2d'][0]+1,r['bbox_2d'][3]-r['bbox_2d'][1]+1) for r in group)
        f=frame(args.image,folder/'input',{**opt,'crop_min_padding_px':max(64,short*2),'crop_padding':0,'crop_scale':100},union(group))
        feedback.render_evidence(args.baseline/'native.pdf',before,f,folder/'input')
        local_hints(folder/'input/clean.png',group,f,folder/'input/reference.png')
        targets=[{'id':r['id'],'text_read_only':r['text'],'runs_read_only':r['runs'],
            'predicted_bbox_2d':mapped_box(r['bbox_2d'],f),'current_color':r['color'],
            'cpu_similarity_score_0_to_100':round(selection['scores'][r['id']]['before']['score']*100,4),
            'score_roi_on_crop':mapped_box(selection['scores'][r['id']]['roi'],f)} for r in group]
        for target in targets:
            target['similarity_statement']=f"The two images have similarity {target['cpu_similarity_score_0_to_100']:.2f}/100 within this target ROI. Diagnose cause 1 (background), cause 2 (target text), or both, with specific visible evidence."
        context=[{'id':r['id'],'text':r['text'],'bbox_2d':mapped_box(r['bbox_2d'],f)} for r in rows
                 if r['id'] not in {t['id'] for t in group} and intersect(r['bbox_2d'],f['box'])>0]
        prompt=protocol.prompt(f['size'],targets,context);save(folder/'prompt.json',prompt)
        result=runtime.call(folder/'calls','bbox_color',SYSTEM,prompt,
            [('SOURCE: original, including background graphics.',folder/'input/clean.png'),
             ('PPT: text-only render, same crop and scale.',folder/'input/render.png'),
             ('REFERENCE: original with fallible current bbox highlights.',folder/'input/reference.png')],
            lambda raw:protocol.validate(raw,group,f),opt['max_tokens'])
        updates=protocol.proposals(result,group,f)
        save(folder/'validated.json',result);save(folder/'proposals.json',updates)
        return {'group':f'G{index:03d}',**result},updates
    updates={};reviews=[];errors=[]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        jobs={pool.submit(job,i,g):i for i,g in enumerate(groups)}
        for future in as_completed(jobs):
            try:
                record,new=future.result();reviews.append(record);updates.update(new)
                print(record['group'],'bbox/color returned',len(new),flush=True)
            except Exception as exc:errors.append(f'G{jobs[future]:03d}: {exc}')
            save(args.output/'review_progress.json',{'completed':len(reviews)+len(errors),'total':len(groups),'errors':errors})
    save(args.output/'reviews.json',reviews)
    diagnosed={item['id']:item for record in reviews for item in record['items']}
    model_keep=set()  # Kept empty for compatibility; VLM no longer decides keep.
    accepted={};acceptance=[];slots=threading.BoundedSemaphore(1)
    if updates:
        fitted,fit_errors=layout.fit_all([r for group in updates.values() for r in group],args.output/'typography',slots,
            {**opt,'pixel_reflow_enabled':False,'source_font_selection':False,'glyph_registration_enabled':False},source=args.image)
        errors.extend(fit_errors)
        fitted_by_id={r['id']:r for r in fitted}
        for ident in sorted(updates):
            new=[fitted_by_id[ident]] if ident in fitted_by_id else []
            conflicts=feedback.new_collisions(rows,feedback.replace(rows,{**accepted,ident:new})) if new else []
            ok=bool(new) and not fit_errors and not conflicts
            if ok:accepted[ident]=new
            acceptance.append({'id':ident,'accepted':ok,'reason':'vlm_bbox_color_applied' if ok else 'measurement_or_collision_failure','collisions':conflicts,'diagnosis':diagnosed[ident]})
    rendered_state={};render_round=0
    def evaluate(candidates):
        nonlocal render_round
        render_round+=1
        folder=args.output/'cpu_guard'/f'round_{render_round:02d}';folder.mkdir(parents=True)
        candidate_rows=feedback.replace(rows,candidates)
        if candidates:counts=export.final(candidate_rows,args.image,folder,slots,opt)
        else:
            for name in ('text.pptx','native.png','native.pdf','comparison.png','export.json'):
                if (args.baseline/name).is_file():shutil.copy2(args.baseline/name,folder/name)
            counts=read(args.baseline/'summary.json')['counts']
        actual=Image.open(folder/'native.png').convert('RGB')
        metrics={ident:feedback.evidence(source,before,actual,[by_id[ident]],new) for ident,new in candidates.items()}
        save(folder/'cpu_comparisons.json',metrics)
        rendered_state.update(folder=folder,rows=candidate_rows,counts=counts,actual=actual)
        return metrics
    accepted,cpu_history=protocol.retain_nonregressing(accepted,evaluate)
    save(args.output/'cpu_guard/history.json',cpu_history)
    failed={r['id']:r for step in cpu_history for r in step['checks'] if not r['accepted']}
    for item in acceptance:
        if item['id'] in failed:
            item.update(accepted=False,kept_original=True,reason=failed[item['id']]['reason'],cpu_evidence=failed[item['id']]['evidence'])
    final_rows=rendered_state['rows'];final=rendered_state['folder'];counts=rendered_state['counts'];actual=rendered_state['actual']
    diagnostics=[{'id':r['id'],'evidence':r['evidence'],'role':'hard gate: final score must not decrease on the same ROI'} for r in cpu_history[-1]['checks']]
    quality=feedback.audit_cpu_quality(source,actual,final_rows)
    save(final/'cpu_quality.json',quality)
    save(final/'scene.json',{'texts':final_rows,'source_size':list(source.size),'errors':[*baseline.get('errors',[]),*errors]})
    final.rename(args.output/'final');final=args.output/'final'
    unresolved=[r['id'] for r in scope if r['id'] not in selection['automatic_ids'] and r['id'] not in accepted and r['id'] not in model_keep]
    save(args.output/'acceptance.json',acceptance)
    save(args.output/'final_validation.json',{'policy':protocol.POLICY,'automatic_ids':selection['automatic_ids'],
        'vlm_applied_ids':sorted(accepted),'vlm_background_keep_ids':sorted(model_keep),'diagnoses':list(diagnosed.values()),'unresolved_ids':unresolved,'post_render_diagnostics':diagnostics,
        'second_vlm_verification':False,'post_vlm_cpu_veto':True,'post_vlm_cpu_rule':'after >= before on identical ROI',
        'cpu_rollback_ids':sorted(failed),'cpu_guard_rounds':len(cpu_history)})
    save(args.output/'usage.json',usage(args.output/'review'))
    compare([args.image,args.baseline/'native.png',final/'native.png'],['Original','Before: PPT text','After: bbox/color review'],args.output/'comparison.png',width_cap=1200)
    summary={'state':'completed_with_errors' if errors else 'completed','policy':protocol.POLICY,
        'reviewed_groups':len(reviews),'proposed_targets':len(updates),'accepted_targets':len(accepted),
        'rejected_targets':len(updates)-len(accepted),'automatic_targets':len(selection['automatic_ids']),
        'vlm_background_kept_targets':len(model_keep),
        'cpu_rollback_ids':sorted(failed),'cpu_guard_rounds':len(cpu_history),
        'deferred_ids':selection['deferred_ids'],'unresolved_ids':unresolved,'final_cpu_failed_ids':quality['cpu_failed_ids'],
        'quality_status':'needs_review' if unresolved else 'cpu_or_vlm_review_complete_not_ground_truth',
        'counts':counts,'errors':errors,'cpu_score_threshold':protocol.THRESHOLD,'cpu_score_comparison':'>',
        'visual_accuracy_verified':False,'metric_is_proxy_not_ground_truth':True}
    save(args.output/'summary.json',summary)
    assert all(sha(Path(p))==value for p,value in hashes.items()),'Source/baseline modified'
    print({k:v for k,v in summary.items() if k!='counts'},flush=True)
