"""One bounded post-render review round; old exports and source images are read-only."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys
import threading

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'.deps'),str(ROOT/'src'),str(ROOT/'vendor')]
from PIL import Image
from shape_integration.common import Runtime,read,save,sha,usage,compare,intersect
from shape_integration.texts import options,layout,colors,export
from shape_integration.texts.controller import frame,mapped_box,local_hints
from shape_integration.texts.local_geometry import union
from shape_integration.texts.prompt_protocol import SYSTEM
from shape_integration.texts import render_feedback as feedback


def verify_rendered(runtime, output, image, baseline, candidate, rows, accepted, groups, opt, concurrency):
    """Verify precisely the rendered combination that will be published."""
    source=Image.open(image).convert('RGB')
    before=Image.open(baseline/'native.png').convert('RGB')
    after=Image.open(candidate/'native.png').convert('RGB')
    records=[];errors=[]
    save(output/'artifacts.json',{'candidate_sha256':{
        name:sha(candidate/name) for name in ('text.pptx','native.png','native.pdf') if (candidate/name).is_file()},
        'policy':'all CPU-approved edits must pass; otherwise rollback the whole combined render'})
    def check(index, group):
        group=[r for r in group if r['id'] in accepted]
        if not group:return []
        folder=output/f'G{index:03d}'
        all_rows=[*group,*[r for old in group for r in accepted[old['id']]]]
        short=max(min(r['bbox_2d'][2]-r['bbox_2d'][0]+1,r['bbox_2d'][3]-r['bbox_2d'][1]+1) for r in all_rows)
        f=frame(image,folder/'source',{**opt,'crop_min_padding_px':max(24,short*2),
            'crop_padding':0,'crop_scale':100},union(all_rows))
        for name,root,render in [('before',baseline,before),('after',candidate,after)]:
            target=folder/name;target.mkdir(parents=True)
            feedback.render_evidence(root/'native.pdf',render,f,target)
        local_hints(folder/'source/clean.png',group,f,folder/'source/ids.png')
        targets=[{'id':r['id'],'before':{'text':r['text'],'bbox_2d':mapped_box(r['bbox_2d'],f)},
            'after':[{'id':n['id'],'text':n['text'],'bbox_2d':mapped_box(n['bbox_2d'],f)} for n in accepted[r['id']]]} for r in group]
        value=feedback.verification_prompt(f['size'],targets)
        save(folder/'prompt.json',value)
        result=runtime.call(folder/'calls','verify',SYSTEM,value,
            [('SOURCE: original crop, sole content authority.',folder/'source/clean.png'),
             ('BEFORE: actual baseline PPT text render.',folder/'before/render.png'),
             ('AFTER: actual modified PPT text render to accept or reject.',folder/'after/render.png'),
             ('IDS: original identity hints only.',folder/'source/ids.png')],
            lambda raw:feedback.validate_verification(raw,[r['id'] for r in group]),opt['max_tokens'])
        save(folder/'validated.json',result)
        return result['items']
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        jobs={pool.submit(check,i,g):i for i,g in enumerate(groups) if any(r['id'] in accepted for r in g)}
        for future in as_completed(jobs):
            try:records.extend(future.result())
            except Exception as exc:errors.append(f"verification G{jobs[future]:03d}: {exc}")
    save(output/'verdicts.json',{'items':records,'errors':errors})
    return records,errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-groups',type=int,default=0,help='0 reviews all low-score groups')
    parser.add_argument('--policy',choices=['cpu80_bbox_color','dual_gate'],default='cpu80_bbox_color')
    parser.add_argument('--target-ids',nargs='+',help='Optional limited smoke-test scope for cpu80_bbox_color')
    parser.add_argument('--concurrency',type=int,default=2)
    parser.add_argument('--reuse-review',type=Path,help='Revalidate complete cached responses from this exact baseline; request only invalid/missing groups')
    args=parser.parse_args()
    if args.max_groups<0 or not 1<=args.concurrency<=4:raise ValueError('Invalid bounded review budget')
    if args.policy=='cpu80_bbox_color':
        from run_bbox_color_review import main as routed_main
        return routed_main(args)
    if args.target_ids:raise ValueError('target-ids is only supported by cpu80_bbox_color')
    if args.output.exists():raise ValueError('Use a fresh output directory')
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    before_hashes={str(p.resolve()):sha(p) for p in [args.image,args.baseline/'scene.json',args.baseline/'text.pptx',args.baseline/'native.png']}
    provenance=read(args.baseline/'provenance.json')
    if provenance['image_sha256']!=sha(args.image):raise ValueError('Baseline/source identity mismatch')
    if args.reuse_review and read(args.reuse_review/'provenance.json')['sources']!=before_hashes:
        raise ValueError('Cached review used a different baseline/source')
    baseline=read(args.baseline/'scene.json');rows=baseline['texts']
    args.max_groups=args.max_groups or max(1,len(rows))
    source=Image.open(args.image).convert('RGB');rendered=Image.open(args.baseline/'native.png').convert('RGB')
    if source.size!=rendered.size:raise ValueError('Native render must have exactly the source canvas size')
    args.output.mkdir(parents=True)
    cfg=read(ROOT/'config.json');cfg['api_concurrency']=args.concurrency
    opt=options(cfg);groups,selection=feedback.select_groups(rows,source,rendered,args.max_groups)
    save(args.output/'selection.json',selection)
    save(args.output/'provenance.json',{'sources':before_hashes,'rounds':1,'max_groups':args.max_groups,
         'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),*ROOT.joinpath('src').rglob('*.py')]},
         'formula_preview':'LibreOffice uses same-OMML compatibility assets; native PowerPoint appearance is not verified'})
    key=os.environ.get('OpenAI_API') or os.environ.get('OPENAI_API_KEY') or os.environ.get('BOYUE_API_KEY')
    if groups and not key:raise ValueError('API key must be supplied through environment')
    runtime=Runtime(cfg,cfg['expected_boyue_base_url'],key).for_stage(reasoning_effort='low',shape_schema_retries=1)
    updates={};reviews=[];errors=[]
    def review_group(index,group):
        folder=args.output/'review'/f'G{index:03d}'
        short=max(min(r['bbox_2d'][2]-r['bbox_2d'][0]+1,r['bbox_2d'][3]-r['bbox_2d'][1]+1) for r in group)
        f=frame(args.image,folder/'input',{**opt,'crop_min_padding_px':max(24,short*2),'crop_padding':0,'crop_scale':100},union(group))
        feedback.render_evidence(args.baseline/'native.pdf',rendered,f,folder/'input')
        local_hints(folder/'input/clean.png',group,f,folder/'input/ids.png')
        targets=[{k:deepcopy(r[k]) for k in ('id','text','font','bold','italic','color','rotation_deg','font_size_px','kind','latex') if k in r} for r in group]
        for t,r in zip(targets,group):t['bbox_2d']=mapped_box(r['bbox_2d'],f)
        context=[{'id':r['id'],'text':r['text'],'bbox_2d':mapped_box(r['bbox_2d'],f)} for r in rows
                 if r['id'] not in {v['id'] for v in group} and intersect(r['bbox_2d'],f['box'])>0]
        value=feedback.prompt(f['size'],targets,context)
        save(folder/'prompt.json',value)
        result=None;cached_path=None
        if args.reuse_review:
            old=args.reuse_review/'review'/f'G{index:03d}'
            old_prompt=read(old/'prompt.json') if (old/'prompt.json').is_file() else {}
            if old_prompt.get('targets')==targets and old_prompt.get('coordinates')==value['coordinates']:
                paths=sorted((old/'calls').glob('*/parsed_response.json'),key=lambda p:p.stat().st_mtime,reverse=True)
                old_reviews=read(args.reuse_review/'reviews.json') if (args.reuse_review/'reviews.json').is_file() else []
                for record in old_reviews:
                    if record.get('group')==f'G{index:03d}' and record.get('reused_response'):
                        path=Path(record['reused_response'])
                        if path.is_file() and sha(path)==record.get('reused_sha256'):paths.append(path)
                for path in paths:
                    status=read(path.parent/'stream_status.json')
                    if status.get('finish_reason')!='stop' or not status.get('sse_done'):continue
                    try:
                        result=feedback.validate_review(read(path),group,f,source)
                        feedback.proposals(result,group,f)
                        cached_path=path;break
                    except (ValueError,KeyError,TypeError):continue
        if result is None:result=runtime.call(folder/'calls','review',SYSTEM,value,
             [('SOURCE: original crop; coordinates refer here.',folder/'input/clean.png'),
              ('RENDER: actual editable text layer, same crop and scale; other graphics intentionally absent.',folder/'input/render.png'),
              ('IDS: identity hints only, not source content.',folder/'input/ids.png')],
             lambda raw:feedback.validate_review(raw,group,f,source),opt['max_tokens'])
        proposed=feedback.proposals(result,group,f)
        save(folder/'validated.json',result);save(folder/'proposals.json',proposed)
        return {'group':f'G{index:03d}',**result,
                'reused_response':str(cached_path.resolve()) if cached_path else None,
                'reused_sha256':sha(cached_path) if cached_path else None},proposed
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        jobs={pool.submit(review_group,i,g):i for i,g in enumerate(groups)}
        for future in as_completed(jobs):
            try:
                record,proposed=future.result();reviews.append(record);updates.update(proposed)
                print(record['group'],'reviewed',len(proposed),'revisions',flush=True)
            except Exception as exc:errors.append(f'G{jobs[future]:03d}: {exc}')
            save(args.output/'review_progress.json',{'completed':len(reviews)+len(errors),'total':len(groups),'errors':errors})
    save(args.output/'reviews.json',reviews);save(args.output/'usage.json',usage(args.output/'review'))
    slots=threading.BoundedSemaphore(1);fitted_updates={};gates=[]
    if updates:
        proposed=[r for group in updates.values() for r in group]
        fitted,fit_errors=layout.fit_all(proposed,args.output/'candidate_typography',slots,{**opt,'pixel_reflow_enabled':False},source=args.image)
        fitted,color_report=colors.recover(fitted,args.image,args.output/'candidate_colors')
        save(args.output/'candidate_color_recovery.json',color_report)
        errors.extend(fit_errors)
        by_id={r['id']:r for r in fitted}
        for ident,group in updates.items():
            if all(r['id'] in by_id for r in group):fitted_updates[ident]=[by_id[r['id']] for r in group]
        candidate_rows=feedback.replace(rows,fitted_updates)
        candidate=args.output/'candidate';candidate.mkdir()
        export.final(candidate_rows,args.image,candidate,slots,opt)
        save(candidate/'scene.json',{'texts':candidate_rows,'source_size':list(source.size),'errors':fit_errors})
        proposed_render=Image.open(candidate/'native.png').convert('RGB')
        old_by_id={r['id']:r for r in rows};accepted={}
        for ident,new in sorted(fitted_updates.items()):
            metric=feedback.evidence(source,rendered,proposed_render,[old_by_id[ident]],new)
            approve,reason=feedback.gate(metric)
            conflicts=feedback.new_collisions(rows,feedback.replace(rows,{**accepted,ident:new}))
            if conflicts:approve,reason=False,'new_occurrence_collision'
            if fit_errors:approve,reason=False,'candidate_measurement_errors'
            if approve:accepted[ident]=new
            gates.append({'id':ident,'accepted':approve,'reason':reason,'evidence':metric,'collisions':conflicts})
    else:accepted={}
    # Never expose an unverified render under the public final/ path.
    final_rows=feedback.replace(rows,accepted);final=args.output/'pending_final';final.mkdir()
    final_check=[];interaction_rollback=False;model_check=[];dual_rollback=False
    cpu_accepted_ids=sorted(accepted)
    if accepted:
        counts=export.final(final_rows,args.image,final,slots,opt)
        actual=Image.open(final/'native.png').convert('RGB')
        old_by_id={r['id']:r for r in rows}
        for ident,new in sorted(accepted.items()):
            metric=feedback.evidence(source,rendered,actual,[old_by_id[ident]],new)
            ok,reason=feedback.gate(metric)
            final_check.append({'id':ident,'passed':ok,'reason':reason,'evidence':metric})
        # Candidate scoring included proposals later rejected. Recheck the
        # actual accepted combination; do not publish a jointly regressing set.
        if any(not r['passed'] for r in final_check):
            final.rename(args.output/'rejected_combination');final.mkdir()
            interaction_rollback=True;accepted={};final_rows=deepcopy(rows)
        else:
            try:
                model_check,verification_errors=verify_rendered(runtime,args.output/'verification',args.image,
                    args.baseline,final,rows,accepted,groups,opt,args.concurrency)
            except Exception as exc:
                model_check=[];verification_errors=[f'verification preparation failed: {exc}']
            errors.extend(verification_errors)
            # Do not selectively remove edits after checking a combination: that
            # would publish a different, unverified render. Fail closed instead.
            if not feedback.dual_gate(final_check,model_check,accepted):
                final.rename(args.output/'rejected_model_combination');final.mkdir()
                dual_rollback=True;accepted={};final_rows=deepcopy(rows)
    if not accepted:
        for name in ('text.pptx','native.png','native.pdf','comparison.png','export.json'):
            if (args.baseline/name).is_file():shutil.copy2(args.baseline/name,final/name)
        counts=read(args.baseline/'summary.json')['counts']
    save(final/'scene.json',{'texts':final_rows,'source_size':list(source.size),'errors':[*baseline.get('errors',[]),*errors]})
    quality=feedback.audit_cpu_quality(source,Image.open(final/'native.png').convert('RGB'),final_rows)
    save(final/'cpu_quality.json',quality)
    final.rename(args.output/'final');final=args.output/'final'
    for record in gates:
        record['cpu_candidate_accepted']=record.pop('accepted')
        record['accepted']=record['id'] in accepted
    save(args.output/'acceptance.json',gates)
    save(args.output/'final_validation.json',{'checks':final_check,'model_checks':model_check,
         'cpu_accepted_ids':cpu_accepted_ids,'interaction_rollback':interaction_rollback,
         'dual_gate_rollback':dual_rollback,'accepted_ids':sorted(accepted),
         'cpu_score_threshold':feedback.CPU_SCORE_THRESHOLD,'cpu_score_comparison':'>',
         'policy':f'CPU score > {feedback.CPU_SCORE_THRESHOLD} AND improvement checks AND actual-render VLM; any failure/uncertainty reverts the complete candidate combination'})
    save(args.output/'usage.json',{'proposal':usage(args.output/'review'),'verification':usage(args.output/'verification')})
    compare([args.image,args.baseline/'native.png',final/'native.png'],['Original','Before: actual PPT text','After: accepted feedback'],args.output/'comparison.png',width_cap=1200)
    summary={'state':'completed_with_errors' if errors else 'completed','reviewed_groups':len(reviews),
             'proposed_targets':len(updates),'accepted_targets':len(accepted),'rejected_targets':len(updates)-len(accepted),
             'deferred_ids':selection['deferred_ids'],'counts':counts,'errors':errors,
             'uncertain_ids':[i['id'] for g in reviews for i in g['decisions'] if i['decision']=='uncertain'],
             'visual_accuracy_verified':False,'metric_is_proxy_not_ground_truth':True}
    summary['interaction_rollback']=interaction_rollback
    summary['dual_gate_rollback']=dual_rollback
    summary['cpu_accepted_targets']=len(cpu_accepted_ids)
    summary['dual_gate_policy']='cpu_and_actual_render_vlm_all_pass'
    summary['cpu_score_threshold']=feedback.CPU_SCORE_THRESHOLD
    summary['cpu_score_comparison']='>'
    summary['final_cpu_failed_ids']=quality['cpu_failed_ids']
    summary['quality_status']='needs_review' if quality['cpu_failed_ids'] else 'cpu_floor_passed_not_full_model_certification'
    save(args.output/'summary.json',summary)
    assert before_hashes=={p:sha(p) for p in before_hashes},'Baseline/source was modified'
    print({k:v for k,v in summary.items() if k!='counts'},flush=True)


if __name__=='__main__':main()
