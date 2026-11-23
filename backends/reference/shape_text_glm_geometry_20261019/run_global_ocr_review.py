"""GLM text -> native baseline -> crop geometry/paint -> crop attributes -> render.

GLM reviews every recognized text in two crop calls, with no CPU acceptance
gate or pixel/template tightening. Ambiguous geometry retains the previous box.
The optional OCR backend keeps its separate score-based validation path.
"""
import argparse
import json
from pathlib import Path
import os
import shutil
import subprocess
import sys
import threading

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'.deps'),str(ROOT/'src'),str(ROOT/'vendor')]
import numpy as np
from PIL import Image
from shape_integration.common import read,save,sha,Runtime,compare,usage
from shape_integration.texts import options,model,prompts,contracts,layout,colors,export,ocr_refinement
from shape_integration.texts.controller import frame,merge
from shape_integration.texts.local_geometry import compact_groups,union
from shape_integration.texts import render_feedback as feedback
from shape_integration.texts.bbox_color_review import retain_nonregressing
from shape_integration.texts import ocr_multiline
from shape_integration.texts.ocr_geometry import OCR_TEMPLATE_MIN_SCORE


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('image','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('ocr-python','models'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--geometry-backend',choices=('glm','ocr'),default='glm',
                        help='GLM uses geometry/background then attributes with no CPU gate; OCR retains its separate path')
    parser.add_argument('--baseline-run',type=Path,
                        help='Read-only prior run: reuse the identical pre-OCR baseline for a paired comparison')
    parser.add_argument('--base-pptx',type=Path,
                        help='Optional aligned non-text PPT: CPU selects white/transparent fill on final combined render')
    parser.add_argument('--recovery-run',type=Path,
                        help='Read-only failed staged run: reuse recognition evidence, render afresh')
    parser.add_argument('--threshold',type=float,default=.80)
    parser.add_argument('--max-score-drop',type=float,default=.05,
                        help='Absolute tolerance on the 0–1 scale; default .05 permits a five-point drop, not a relative five percent')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--recognition-mode',choices=('staged','legacy'),default='staged')
    parser.add_argument('--ocr-mode',choices=('geometry','legacy'),default='geometry')
    parser.add_argument('--ocr-max-side',type=int,default=1536)
    parser.add_argument('--ocr-template-validation',choices=('on','off'),default='on',
                        help='OCR backend only; GLM boxes bypass CPU ink/template tightening')
    parser.add_argument('--glm-crop-refine',action='store_true',
                        help='Compatibility flag: GLM now uses two crop calls for every recognized text after baseline')
    parser.add_argument('--stop-after-baseline',action='store_true',
                        help='Verify OMML/color native output before enabling OCR; resume without this flag')
    parser.add_argument('--saved-response',type=Path,help='Revalidate a completed response from this run without a new API call')
    parser.add_argument('--reuse-run',type=Path,help='Read-only completed run: reuse its final render and cached GLM/OCR evidence in a fresh output')
    parser.add_argument('--evidence-run',type=Path,help='Reuse only a complete global response, including schema-failed responses; render and OCR afresh')
    args=parser.parse_args()
    if args.recovery_run and (args.baseline_run or args.reuse_run or args.evidence_run or args.saved_response
                             or args.recognition_mode!='staged'):
        raise ValueError('Recovery requires staged mode without other evidence sources')
    if args.geometry_backend=='ocr' and (not args.ocr_python or not args.models):
        raise ValueError('OCR backend requires --ocr-python and --models')
    if args.geometry_backend=='glm' and (args.reuse_run or args.ocr_mode!='geometry'):
        raise ValueError('GLM backend requires geometry mode and cannot reuse an OCR final result')
    if args.baseline_run and (args.reuse_run or args.evidence_run or args.saved_response):
        raise ValueError('Frozen baseline cannot be combined with other evidence modes')
    if not 0<=args.threshold<=1 or not 0<=args.max_score_drop<=1:
        raise ValueError('Invalid threshold/tolerance')
    if args.output.exists() and not args.resume:raise ValueError('Use fresh output or explicit resume')
    args.output=args.output.resolve();args.image=args.image.resolve()
    if args.recognition_mode=='staged' and (args.reuse_run or args.evidence_run or args.saved_response):
        raise ValueError('Old single-pass evidence needs explicit --recognition-mode legacy')
    if args.glm_crop_refine and args.reuse_run:
        raise ValueError('Crop refinement requires rendering a fresh baseline, not completed-run reuse')
    if args.evidence_run:
        if args.reuse_run:raise ValueError('Choose evidence-only or completed-run reuse')
        args.evidence_run=args.evidence_run.resolve()
        prior=read(args.evidence_run/'provenance.json')['parameters']
        if prior['image_sha256']!=sha(args.image):raise ValueError('Evidence source mismatch')
        if not args.saved_response:raise ValueError('Evidence reuse needs a complete saved response')
    if args.reuse_run:
        args.reuse_run=args.reuse_run.resolve()
        if args.reuse_run==args.output:raise ValueError('Do not overwrite the source run')
        prior=read(args.reuse_run/'provenance.json')['parameters']
        if prior['image_sha256']!=sha(args.image):raise ValueError('Reuse source mismatch')
        if not args.saved_response:raise ValueError('Reusing requires the saved global response to restore metadata')
    args.output.mkdir(parents=True,exist_ok=True)
    fingerprint={'image_sha256':sha(args.image),'threshold':args.threshold,'max_score_drop':args.max_score_drop,
                 'text_geometry_revision':'monospace_occurrence_model_bbox_v2',
                 'text_background_revision':'transparent_only_v2' if args.geometry_backend=='glm' else 'cpu_white_or_transparent_v1',
                 'glm_bbox_policy':'model_bbox_without_cpu_tightening_v1',
                 'review_policy':'two_stage_crop_visual_recall_wide_recovery_no_cpu_gate_v5' if args.geometry_backend=='glm' else 'ocr_geometry',
                 'crop_rules_sha256':sha(ROOT/'src/shape_integration/texts/crop_review_rules.txt'),
                 'base_pptx_sha256':sha(args.base_pptx) if args.base_pptx else None,
                 'recognition_mode':args.recognition_mode,'ocr_template_validation':args.ocr_template_validation,
                 'ocr_template_min_score':OCR_TEMPLATE_MIN_SCORE,'geometry_backend':args.geometry_backend,
                 'baseline_run':str(args.baseline_run.resolve()) if args.baseline_run else None}
    if args.glm_crop_refine:fingerprint['glm_crop_refine']=True
    if args.recovery_run:fingerprint['recovery_run']=str(args.recovery_run.resolve())
    provenance=args.output/'provenance.json'
    if provenance.exists() and read(provenance)['parameters']!=fingerprint:
        raise ValueError('Resume parameters/source mismatch')
    if not provenance.exists():save(provenance,{'parameters':fingerprint,'code_sha256':{
        str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'compare_group_ocr.py',ROOT/'ocr_group_worker.py',*ROOT.joinpath('src').rglob('*.py'),ROOT/'src/shape_integration/texts/crop_review_rules.txt']}})
    if hasattr(os,'sched_getaffinity'):
        allowed=os.sched_getaffinity(0)
        requested=os.environ.get('SHAPE_CPU_AFFINITY')
        cpus={int(v) for v in requested.split(',')} if requested else set(sorted(allowed)[:2])
        if not cpus or not cpus<=allowed:raise ValueError('Invalid per-case CPU affinity')
        os.sched_setaffinity(0,cpus)
    def status(stage,**kw):
        save(args.output/'progress.json',{'stage':stage,**kw});print(stage,kw,flush=True)
    cfg=read(ROOT/'config.json');opt=options(cfg)
    if os.environ.get('SHAPE_BATCH_LIMIT_DIR'):
        import cv2
        cv2.setNumThreads(1)
    opt.update(pixel_reflow_enabled=False,source_font_selection=True,glyph_registration_enabled=False)
    if args.geometry_backend=='glm':
        opt.update(inventory_local_enabled=False,multiline_before_baseline=False)
    source=Image.open(args.image).convert('RGB');slots=threading.BoundedSemaphore(1)
    runtime_warnings=[]
    if args.recovery_run:
        from shape_integration.texts.run_recovery import seed
        seed(args.recovery_run,args.output,args.image)
    baseline=args.output/'baseline';recognition=args.output/'recognition.json'
    if args.baseline_run:
        prior=args.baseline_run.resolve()
        if prior==args.output:raise ValueError('Frozen baseline must be read-only in another directory')
        if read(prior/'baseline/provenance.json')['image_sha256']!=sha(args.image):
            raise ValueError('Frozen baseline source mismatch')
        cached_provenance=args.output/'baseline_reuse.json'
        files=[p for p in (prior/'baseline').rglob('*') if p.is_file()]
        names=['recognition.json','crop_recognition.json'] if args.glm_crop_refine and args.geometry_backend=='ocr' else ['recognition.json']
        files.extend(prior/name for name in names)
        digests={str(p.relative_to(prior)):sha(p) for p in files}
        if cached_provenance.exists():
            if read(cached_provenance)['files']!=digests:raise ValueError('Frozen baseline changed')
        else:
            if baseline.exists() or recognition.exists():raise ValueError('Cannot seed a partially populated baseline')
            shutil.copytree(prior/'baseline',baseline)
            for name in names:shutil.copy2(prior/name,args.output/name)
            save(cached_provenance,{'source':str(prior),'files':digests,
                'policy':'identical pre-OCR source and rendering; no repeated global or crop API calls'})
        status('frozen_baseline_reused',source=str(prior))
    if not recognition.exists() and args.recognition_mode=='staged':
        from shape_integration.texts.staged_formula import recognize
        key=os.environ.get('OpenAI_API') or os.environ.get('OPENAI_API_KEY') or os.environ.get('BOYUE_API_KEY')
        if not key:raise ValueError('API key required in environment')
        runtime=Runtime(cfg,cfg['expected_boyue_base_url'],key)
        result=recognize(args.image,args.output/'staged',runtime,opt,status)
        save(recognition,result)
    if not recognition.exists():
        status('glm_global_recognition')
        folder=(args.reuse_run or args.evidence_run or args.output)/'global'
        if args.saved_response:
            from shape_integration.styling.views import resample_crop
            cached=args.saved_response.resolve()
            if not cached.is_relative_to((folder/'calls').resolve()):raise ValueError('Response must belong to this run')
            s=read(cached.parent/'stream_status.json')
            if s.get('http_status')!=200 or s.get('finish_reason')!='stop' or not s.get('sse_done'):
                raise ValueError('Only complete successful HTTP responses may be revalidated')
            f=read(folder/'input/frame.json')
            expected=resample_crop(source,f['box'],f['size'],f['scale'],f['offset'])
            if not np.array_equal(np.asarray(expected),np.asarray(Image.open(folder/'input/clean.png').convert('RGB'))):
                raise ValueError('Saved response source mismatch')
            identity=read(cached.parent/'identity.json')
            if sha(folder/'input/clean.png') not in [i['sha256'] for i in identity['images']]:
                raise ValueError('Saved request image hash mismatch')
            text=cached.read_text().strip()
            if text.startswith('```') and text.endswith('```'):text='\n'.join(text.splitlines()[1:-1])
            result=contracts.recognition(json.loads(text),f['size'],default_font=opt['default_font'])
            save(args.output/'saved_response_revalidation.json',{'response':str(cached),'sha256':sha(cached),
                'api_calls':0,'format_repairs':result.get('format_repairs',[]),'accepted_texts':len(result['texts']),
                'validator_code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in ROOT.joinpath('src/shape_integration/texts').glob('*.py')}})
        else:
            key=os.environ.get('OpenAI_API') or os.environ.get('OPENAI_API_KEY') or os.environ.get('BOYUE_API_KEY')
            if not key:raise ValueError('API key required in environment')
            runtime=Runtime(cfg,cfg['expected_boyue_base_url'],key)
            f=frame(args.image,folder/'input',opt)
            result=model.ask(runtime,folder,prompts.recognize(f['size'],local=False,known=[],assets=False,
                max_regions=opt['max_regions'],default_font=opt['default_font']),
                [('Original full figure; recover editable text, not text inside photographs.',folder/'input/clean.png')],
                lambda raw:contracts.recognition(raw,f['size'],default_font=opt['default_font']),opt)
        rows=[];history=[];merge(rows,result['texts'],f,'glm_global',history,opt)
        if not rows:raise ValueError('GLM returned no text')
        save(recognition,{'texts':rows,'uncertain':result.get('uncertain',[]),'merge_history':history,
                          'regions_needing_review':result.get('regions',[]),
                          'format_repairs':result.get('format_repairs',[])})
    baseline_recognition=recognition
    from shape_integration.texts.occurrence_dedup import deduplicate
    inventory=read(recognition)
    inventory['texts'],inventory['occurrence_dedup']=deduplicate(inventory['texts'],opt['duplicate_iou'])
    baseline_recognition=args.output/'deduplicated_recognition.json'
    save(baseline_recognition,inventory)
    if args.glm_crop_refine and args.geometry_backend=='ocr':
        baseline_recognition=args.output/'crop_recognition.json'
        if not baseline_recognition.exists():
            from shape_integration.texts.local_geometry import refine_inventory
            key=os.environ.get('OpenAI_API') or os.environ.get('OPENAI_API_KEY') or os.environ.get('BOYUE_API_KEY')
            if not key:raise ValueError('API key required for GLM crop review')
            runtime=Runtime(cfg,cfg['expected_boyue_base_url'],key)
            status('glm_crop_geometry')
            result=refine_inventory(inventory,args.image,args.output/'glm_crop_geometry',runtime,opt,
                lambda stage,done,total:status(stage,done=done,total=total))
            save(baseline_recognition,result)
    # Also cover cached recognitions/crop replies without mutating source evidence.
    from shape_integration.texts.numeric_routing import inventory as route_inventory
    routed=route_inventory(read(baseline_recognition),'before_native_layout')
    baseline_recognition=args.output/'routed_recognition.json'
    save(baseline_recognition,routed)
    if not (baseline/'summary.json').exists():
        status('reuse_completed_baseline' if args.reuse_run else 'baseline_native_render')
        baseline.mkdir(exist_ok=True)
        errors=[]
        if args.reuse_run:
            rows=read(args.reuse_run/'final/scene.json')['texts']
            metadata={r['id']:r for r in read(recognition)['texts']}
            if {r['id'] for r in rows}!=set(metadata):raise ValueError('Reuse IDs do not match saved recognition')
            for row in rows:
                original=metadata[row['id']]
                if row['text']!=original['text']:raise ValueError('Reuse text changed')
                row.update({k:original[k] for k in ('visual_line_count','line_layout_status')})
            for name in ('text.pptx','native.png','native.pdf'):
                shutil.copy2(args.reuse_run/'final'/name,baseline/name)
            counts={'textboxes':len(rows),'reused_previous_final':True}
            save(args.output/'reuse_provenance.json',{'run':str(args.reuse_run),'source_files':{
                name:sha(args.reuse_run/'final'/name) for name in ('scene.json','text.pptx','native.png','native.pdf')},
                'policy':'preserve previously accepted corrections; restore line metadata from complete GLM response'})
        else:
            rows,errors=layout.fit_all(read(baseline_recognition)['texts'],baseline/'typography',slots,opt,source=args.image)
            rows,audit=colors.recover(rows,args.image,baseline/'color_assets')
            save(baseline/'color_recovery.json',audit)
            counts=export.final(rows,args.image,baseline,slots,opt)
        save(baseline/'scene.json',{'texts':rows,'source_size':list(source.size),'errors':errors})
        save(baseline/'provenance.json',{'image_sha256':sha(args.image)})
        save(baseline/'summary.json',{'counts':counts})
    if args.stop_after_baseline:
        save(args.output/'usage.json',usage(args.output))
        status('baseline_ready_for_review',recognition_mode=args.recognition_mode)
        return
    if args.geometry_backend=='glm':
        from shape_integration.texts.single_crop import finish
        key=os.environ.get('OpenAI_API') or os.environ.get('OPENAI_API_KEY') or os.environ.get('BOYUE_API_KEY')
        runtime=Runtime(cfg,cfg['expected_boyue_base_url'],key) if key else None
        finish(args.image,args.output,baseline,opt,runtime,threshold=args.threshold,
               max_score_drop=args.max_score_drop,base=args.base_pptx,
               parameters=fingerprint,progress=status,recognition=read(baseline_recognition))
        return
    ocr_params={'backend':args.geometry_backend,'mode':args.ocr_mode,'max_side':args.ocr_max_side,'template_validation':args.ocr_template_validation if args.geometry_backend=='ocr' else 'not_applicable',
                'template_min_score':OCR_TEMPLATE_MIN_SCORE if args.geometry_backend=='ocr' else None}
    if (args.output/'ocr_parameters.json').exists() and read(args.output/'ocr_parameters.json')!=ocr_params:
        raise ValueError('OCR parameters changed; use a fresh run')
    save(args.output/'ocr_parameters.json',ocr_params)
    save(args.output/'ocr_code.json',{str(p.relative_to(ROOT)):sha(p)
         for p in [Path(__file__),ROOT/'ocr_group_worker.py',*ROOT.joinpath('src').rglob('*.py')]})
    rows=read(baseline/'scene.json')['texts'];by_id={r['id']:r for r in rows}
    degraded_ids={r['id'] for r in rows if r.get('layout',{}).get('measurement',{}).get('preview_degraded')}
    before=Image.open(baseline/'native.png').convert('RGB')
    baseline_scores={r['id']:feedback.evidence(source,before,before,[r]) for r in rows}
    low=[r for r in rows if r['id'] not in degraded_ids and
         (baseline_scores[r['id']]['before']['score']<args.threshold
         or r.get('line_layout_status')=='requires_split')]
    if args.reuse_run:
        low=[r for r in low if not r.get('ocr_refinement') or r.get('line_layout_status')=='requires_split']
    groups=compact_groups(low,source.size,6)
    save(args.output/'selection.json',{'total_texts':len(rows),'low_score_ids':[r['id'] for r in low],
        'automatic_ids':[r['id'] for r in rows if r not in low],'scores':baseline_scores,
        'comparison':'score < threshold OR known multiline; reuse preserves previously accepted edits'})
    jobs=[]
    for index,group in enumerate(groups):
        if args.reuse_run:break
        folder=args.output/'groups'/f'G{index:03d}'
        short=max(min(r['bbox_2d'][2]-r['bbox_2d'][0]+1,r['bbox_2d'][3]-r['bbox_2d'][1]+1) for r in group)
        frame(args.image,folder/'input',{**opt,'crop_min_padding_px':max(64,1.5*short),'crop_padding':0,'crop_scale':100},union(group))
        rotations=[0]
        if any(abs(r['rotation_deg'])%180>45 for r in group):rotations.extend([90,270])
        jobs.append({'group':folder.name,'crop_dir':str(folder/'input'),
                     'output':str(folder/('glm_geometry' if args.geometry_backend=='glm' else 'ocr')),
                     'ids':[r['id'] for r in group],'rotations':rotations,'max_side':args.ocr_max_side})
    if args.reuse_run:
        low_ids={r['id'] for r in low}
        previous_jobs=read(args.reuse_run/'ocr_manifest.json')['jobs']
        jobs=[{**job,'ids':[i for i in job['ids'] if i in low_ids]}
              for job in previous_jobs if any(i in low_ids for i in job['ids'])]
    manifest=args.output/'ocr_manifest.json'
    save(manifest,{'jobs':jobs,'scene':str(baseline/'scene.json'),
                   'backend':args.geometry_backend,'models':str(args.models.resolve()) if args.models else None})
    status('grouped_'+args.geometry_backend,total_texts=len(rows),low_score_texts=len(low),groups=len(jobs))
    if jobs and args.geometry_backend=='glm':
        from shape_integration.texts.glm_geometry import run_jobs
        key=os.environ.get('OpenAI_API') or os.environ.get('OPENAI_API_KEY') or os.environ.get('BOYUE_API_KEY')
        if not key:raise ValueError('API key required for GLM geometry')
        results=run_jobs(jobs,rows,args.image,baseline/'native.png',baseline_scores,
                         Runtime(cfg,cfg['expected_boyue_base_url'],key),opt,status)
        save(args.output/'glm_geometry_progress.json',{'completed':len(results),'total':len(jobs),'results':results})
        runtime_warnings.extend({'stage':'glm_geometry','group':r['group'],'reason':'request failed; retained prior hypothesis'}
                                for r in results if r['state']=='failed')
    if jobs and not args.reuse_run and args.geometry_backend=='ocr':
        env=dict(os.environ,PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True',MKL_NUM_THREADS='2')
        with (args.output/'ocr.log').open('a') as log:
            try:
                subprocess.run([str(args.ocr_python),'-B','-u',str(ROOT/'ocr_group_worker.py'),'--manifest',str(manifest)],
                    env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            except (subprocess.CalledProcessError,OSError) as exc:
                runtime_warnings.append({'stage':'ocr_worker','reason':str(exc),
                    'policy':'reuse_completed_groups_or_prior_geometry'})
    updates={};decisions=[]
    for job in jobs:
        path=Path(job['output'])/'report.json'
        if not path.is_file():
            if args.ocr_mode!='geometry':
                decisions.extend({'id':i,'status':'ocr_group_failed'} for i in job['ids']);continue
            report={'detections':[],'input_sha256':{},'ocr_failed':True}
        else:report=read(path)
        for name,digest in report['input_sha256'].items():
            if sha(Path(job['crop_dir'])/name)!=digest:raise ValueError('OCR input changed')
        used=set()
        for ident in job['ids']:
            try:
                original=by_id[ident]
                detections=[d for d in report['detections'] if d.get('target_id',ident)==ident]
                # A target is an occurrence, not necessarily one OCR box.
                if args.geometry_backend=='glm':
                    from shape_integration.texts.glm_geometry import propose_lines
                    children,multiline_audit=propose_lines(original,detections,source,rows)
                else:children,multiline_audit=ocr_multiline.propose_lines(original,detections,source,used)
                if children:
                    decisions.append({'id':ident,**multiline_audit})
                    updates[ident]=children;used.update(multiline_audit['detection_indices'])
                    continue
                if original.get('line_layout_status')=='requires_split':
                    if args.geometry_backend=='glm':
                        decisions.append({'id':ident,'status':'unresolved_multiline',
                            'glm':multiline_audit,'bbox_policy':'retain_prior_no_cpu_tightening'})
                        continue
                    # Restore the earlier pixel/template reflow as a conservative
                    # fallback when OCR cannot transcribe complete baselines.
                    from shape_integration.texts.reflow import propose as pixel_reflow
                    children,pixel_audit=pixel_reflow(original,source)
                    if children and len(children)==original['visual_line_count']:
                        for child in children:
                            child.update(visual_line_count=1,line_layout_status='split_baseline',
                                semantic_unit_id=original.get('semantic_unit_id',ident),font=original['font'])
                        updates[ident]=children
                        decisions.append({'id':ident,'status':'pixel_reflow_multiline',
                                          'ocr':multiline_audit,'pixel_reflow':pixel_audit})
                    else:
                        decisions.append({'id':ident,'status':'unresolved_multiline',
                                          'ocr':multiline_audit,'pixel_reflow':pixel_audit})
                    continue
                if args.ocr_mode=='geometry':
                    if args.geometry_backend=='glm':
                        from shape_integration.texts.glm_geometry import select_geometry
                        record=select_geometry(original,detections,source.size,rows)
                    else:
                        from shape_integration.texts.ocr_geometry import select
                        record=select(original,detections,source,used,rows,
                                      template_validation=args.ocr_template_validation=='on')
                    record['id']=ident
                    record['geometry_backend']=args.geometry_backend
                    record['geometry_group_failed']=report.get('geometry_failed',False)
                    record['ocr_group_failed']=report.get('ocr_failed',False)
                    decisions.append(record)
                    used.update(record['detection_indices'])
                    updates[ident]=[ocr_refinement.proposal(original,record['bbox_2d'],record)]
                    continue
                detection=ocr_refinement.select_match(original,report['detections'],source.size,used)
                ocr_refinement.validate_match(by_id[ident],detection,source.size)
                indices=ocr_refinement.detection_indices(detection)
                if set(indices).intersection(used):raise ValueError('Reused OCR detection')
                used.update(indices)
                record,mask=ocr_refinement.localize(original,detection,source)
                record.update(id=ident,detection=detection)
                local=args.output/'localization'/ident;local.mkdir(parents=True,exist_ok=True)
                Image.fromarray(mask.astype(np.uint8)*255).save(local/'mask.png');save(local/'record.json',record)
                decisions.append(record)
                if record['status']=='located':updates[ident]=[ocr_refinement.proposal(by_id[ident],record['bbox_2d'],record)]
            except (ValueError,TypeError,KeyError) as exc:decisions.append({'id':ident,'status':'unmatched','reason':str(exc)})
    save(args.output/'ocr_decisions.json',decisions)
    save(args.output/'geometry_decisions.json',{'backend':args.geometry_backend,'decisions':decisions})
    status('fit_ocr_candidates',candidates=len(updates))
    if updates:
        try:
            fitted,errors=layout.fit_all([r for rs in updates.values() for r in rs],args.output/'typography',slots,opt,source=args.image)
            runtime_warnings.extend({'stage':'ocr_candidate_font_metrics','reason':e} for e in errors)
            for row in fitted:
                score=row.get('ocr_refinement',{}).get('template_score',0)
                if score>=.80:
                    row['glyph_registration']={'status':'accepted','score':score,
                                                'anchor':'ocr_full_native_template'}
            fitted,color_audit=colors.recover(fitted,args.image,args.output/'ocr_color_assets')
            save(args.output/'ocr_color_recovery.json',color_audit)
            updates=ocr_multiline.regroup_fitted(updates,fitted)
            # A degraded preview is not usable evidence for accepting a geometry edit.
            degraded_updates={i for i,rs in updates.items() if any(
                r.get('layout',{}).get('measurement',{}).get('preview_degraded') for r in rs)}
            for ident in degraded_updates:
                updates.pop(ident)
                runtime_warnings.append({'id':ident,'stage':'candidate_preview','fallback':'baseline'})
        except Exception as exc:
            runtime_warnings.append({'stage':'candidate_layout','reason':str(exc),'fallback':'baseline'})
            updates={}
    conflicts=feedback.new_collisions(rows,feedback.replace(rows,updates))
    blocked=ocr_multiline.conflict_parents(updates,conflicts)
    updates={i:rs for i,rs in updates.items() if i not in blocked}
    state={};rounds=[]
    def evaluate(candidates):
        folder=args.output/'cpu_guard'/f'round_{len(rounds)+1:02d}';folder.mkdir(parents=True,exist_ok=True)
        final_rows=feedback.replace(rows,candidates)
        status('cpu_candidate_render',round=len(rounds)+1,candidates=len(candidates))
        if candidates:export.final(final_rows,args.image,folder,slots,opt)
        else:
            for name in ('native.png','native.pdf','text.pptx'):shutil.copy2(baseline/name,folder/name)
        actual=Image.open(folder/'native.png').convert('RGB')
        metrics={i:feedback.evidence(source,before,actual,[by_id[i]],rs) for i,rs in candidates.items()}
        save(folder/'scene.json',{'texts':final_rows,'source_size':list(source.size)})
        save(folder/'scores.json',metrics);rounds.append(folder);state.update(folder=folder,image=actual)
        return metrics
    try:
        accepted,history=retain_nonregressing(updates,evaluate,args.max_score_drop)
    except Exception as exc:
        # A failed candidate render must never invalidate the usable baseline.
        runtime_warnings.append({'stage':'candidate_render','reason':str(exc),'fallback':'baseline'})
        accepted={};history=[{'error':str(exc),'fallback':'baseline'}]
        # The failed attempt may have created an incomplete round directory.
        rounds.append(args.output/'cpu_guard/failed_attempt')
        evaluate({})
    save(args.output/'cpu_guard/history.json',history)
    final=args.output/'final'
    if final.exists():raise ValueError('Final already exists; completed runs are immutable')
    shutil.copytree(state['folder'],final)
    # Background is evaluated only against real non-text context, after text
    # geometry is finalized. It never causes an additional model request.
    final_scene=read(final/'scene.json')
    if args.base_pptx:
        status('cpu_background_selection')
        export.final(final_scene['texts'],args.image,final,slots,opt,args.base_pptx)
        save(final/'scene.json',final_scene)
    else:
        from shape_integration.texts.background_selection import select as select_background
        final_scene['texts']=select_background(final_scene['texts'],args.image,
            final/'background_selection',slots,opt)
        # Fresh baseline/candidate exports are already transparent. Reused old
        # runs must be rendered again if they contained model-assigned fills.
        old_rows=read(final/'scene.json')['texts']
        if any(r.get('background_fill') for r in old_rows):
            export.final(final_scene['texts'],args.image,final,slots,opt)
        save(final/'scene.json',final_scene)
    save(final/'provenance.json',{'image_sha256':sha(args.image),'parameters':fingerprint})
    final_metrics={r['id']:feedback.evidence(source,before,state['image'],[r],accepted.get(r['id'],[])) for r in rows}
    for ident in degraded_ids:
        final_metrics[ident].update(reliable=False,excluded_from_fidelity=True,
            reason='Degraded compatibility preview; native OMML retained, manual review required')
    low_final=[i for i,m in final_metrics.items() if m['after']['score']<args.threshold or not m['reliable']]
    region_warnings=read(baseline_recognition).get('regions_needing_review',[])+runtime_warnings
    region_warnings.extend({'stage':'baseline_font_metrics','reason':e}
                           for e in read(baseline/'scene.json').get('errors',[]))
    save(final/'quality.json',{'quality_status':'needs_review' if low_final or region_warnings else 'proxy_checks_passed',
        'low_score_ids':low_final,'scores':final_metrics,'regions_needing_review':region_warnings,
        'note':'Execution completed is not visual approval. White text on a white text-only preview has no measurable contrast.'})
    compare([args.image,baseline/'native.png',final/'native.png'],
            ['Original','GLM crop-refined text' if args.glm_crop_refine else ('Previous OCR result' if args.reuse_run else 'GLM global text'),
             'GLM bbox + CPU render gate' if args.geometry_backend=='glm' else 'OCR proposals + glyph matching'],args.output/'comparison.png',width_cap=1800)
    accounting=usage(args.output)
    if args.recovery_run:
        seeded=read(args.output/'recovery_seed.json')['files']
        cached_calls={str(Path(name).parent) for name in seeded if name.endswith('/response_0.json')}
        new_calls=[call for call in accounting['calls'] if call['call'] not in cached_calls]
        accounting['reused_cached_calls']=len(accounting['calls'])-len(new_calls)
        accounting['new_calls']=new_calls
        accounting['new_totals']={key:sum(call[key] for call in new_calls) for key in accounting['totals']}
        accounting['note']='totals include reused evidence; new_totals counts only new responses with usage'
    save(args.output/'usage.json',accounting)
    known_multiline=[r['id'] for r in rows if r.get('line_layout_status')=='requires_split']
    summary={'total_texts':len(rows),'final_texts':len(read(final/'scene.json')['texts']),
             'geometry_backend':args.geometry_backend,
             'baseline_run':str(args.baseline_run.resolve()) if args.baseline_run else None,
             'low_score_texts':len(low),'groups':len(jobs),
             'known_multiline_ids':known_multiline,
             'resolved_multiline_ids':sorted(i for i in accepted if len(accepted[i])>1),
             'unresolved_multiline_ids':sorted(set(known_multiline)-set(accepted)),
             'reused_run':str(args.reuse_run) if args.reuse_run else None,
             'evidence_run':str(args.evidence_run) if args.evidence_run else None,
             'quality_status':'needs_review' if low_final or region_warnings else 'proxy_checks_passed',
             'final_low_score_ids':low_final,'regions_needing_review':region_warnings,
             'accepted_ids':sorted(accepted),'unresolved_ids':sorted(({r['id'] for r in low}-set(accepted))|degraded_ids),
             'degraded_formula_preview_ids':sorted(degraded_ids),
             'collision_rejected_ids':sorted(blocked&set(by_id)),'max_score_drop':args.max_score_drop,
             'score_policy':'same ROI; after >= before - max_score_drop; absolute points, not relative percent',
             'ocr_is_not_formula_parser':True,'metric_is_proxy_not_ground_truth':True}
    save(args.output/'summary.json',summary);status('completed',**summary)
    if args.reuse_run:
        assert all(sha(args.reuse_run/'final'/name)==digest
                   for name,digest in read(args.output/'reuse_provenance.json')['source_files'].items())


if __name__=='__main__':main()
