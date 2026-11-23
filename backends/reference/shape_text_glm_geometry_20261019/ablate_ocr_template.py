"""Offline paired OCR geometry ablation on identical cached baselines/replies."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import threading

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'.deps'),str(ROOT/'src'),str(ROOT/'vendor')]
import numpy as np
from PIL import Image
from shape_integration.common import read,save,sha,compare
from shape_integration.texts import options,layout,colors,export,ocr_multiline,ocr_refinement
from shape_integration.texts.ocr_geometry import select
from shape_integration.texts import render_feedback as feedback
from shape_integration.texts.bbox_color_review import retain_nonregressing


def branch(source_path,cache,folder,enabled):
    folder.mkdir(parents=True,exist_ok=True)
    with Image.open(source_path) as im:source=im.convert('RGB')
    before=Image.open(cache/'baseline/native.png').convert('RGB')
    rows=read(cache/'baseline/scene.json')['texts'];by_id={r['id']:r for r in rows}
    opt=options(read(ROOT/'config.json'))
    opt.update(pixel_reflow_enabled=False,source_font_selection=False,glyph_registration_enabled=False)
    slots=threading.BoundedSemaphore(1)
    updates={};decisions=[];warnings=[]
    for job in read(cache/'ocr_manifest.json')['jobs']:
        report_path=Path(job['output'])/'report.json'
        report=read(report_path) if report_path.exists() else {'detections':[],'input_sha256':{}}
        for name,digest in report['input_sha256'].items():
            if sha(Path(job['crop_dir'])/name)!=digest:raise ValueError('OCR crop hash mismatch')
        used=set()
        for ident in job['ids']:
            if ident not in by_id:raise ValueError('Cached OCR ID absent from frozen baseline')
            original=by_id[ident]
            try:
                children,audit=ocr_multiline.propose_lines(original,report['detections'],source,used)
                if children:
                    updates[ident]=children;used.update(audit['detection_indices'])
                    decisions.append({'id':ident,**audit});continue
                if original.get('line_layout_status')=='requires_split':
                    from shape_integration.texts.reflow import propose
                    children,detail=propose(original,source)
                    if children and len(children)==original['visual_line_count']:
                        for child in children:
                            child.update(visual_line_count=1,line_layout_status='split_baseline',
                                semantic_unit_id=original.get('semantic_unit_id',ident),font=original['font'])
                        updates[ident]=children
                    decisions.append({'id':ident,'status':'common_multiline_fallback','detail':detail});continue
                record=select(original,report['detections'],source,used,rows,template_validation=enabled)
                record['id']=ident;decisions.append(record)
                used.update(record['detection_indices'])
                updates[ident]=[ocr_refinement.proposal(original,record['bbox_2d'],record)]
            except (ValueError,TypeError,KeyError) as exc:
                decisions.append({'id':ident,'status':'unmatched','reason':str(exc)})
    save(folder/'decisions.json',decisions)
    initial=len(updates)
    print(folder.parent.name,folder.name,'fit',initial,flush=True)
    if updates:
        fitted,errors=layout.fit_all([r for rs in updates.values() for r in rs],folder/'typography',slots,opt,source=source_path)
        warnings.extend(errors)
        for row in fitted:
            score=row.get('ocr_refinement',{}).get('template_score',0)
            if score>=.80:row['glyph_registration']={'status':'accepted','score':score,'anchor':'ocr_full_native_template'}
        fitted,audit=colors.recover(fitted,source_path,folder/'color_assets')
        save(folder/'color_recovery.json',audit)
        updates=ocr_multiline.regroup_fitted(updates,fitted)
    conflicts=feedback.new_collisions(rows,feedback.replace(rows,updates))
    blocked=ocr_multiline.conflict_parents(updates,conflicts)
    updates={i:rs for i,rs in updates.items() if i not in blocked}
    state={};rounds=[]
    def evaluate(candidates):
        out=folder/'cpu_guard'/f'round_{len(rounds)+1:02d}';out.mkdir(parents=True)
        new_rows=feedback.replace(rows,candidates)
        if candidates:export.final(new_rows,source_path,out,slots,opt)
        else:
            for name in ('native.png','native.pdf','text.pptx'):shutil.copy2(cache/'baseline'/name,out/name)
        actual=Image.open(out/'native.png').convert('RGB')
        metrics={i:feedback.evidence(source,before,actual,[by_id[i]],rs) for i,rs in candidates.items()}
        save(out/'scene.json',{'texts':new_rows,'source_size':list(source.size)})
        save(out/'scores.json',metrics);rounds.append(out);state.update(folder=out,rows=new_rows)
        return metrics
    tolerance=read(cache/'provenance.json')['parameters']['max_score_drop']
    accepted,history=retain_nonregressing(updates,evaluate,tolerance)
    save(folder/'cpu_guard/history.json',history)
    shutil.copytree(state['folder'],folder/'final')
    summary={'initial_candidates':initial,'accepted_ids':sorted(accepted),
             'collision_rejected':sorted(blocked),'warnings':warnings,'new_api_calls':0,
             'new_ocr_calls':0,'template_validation':enabled,'max_score_drop':tolerance}
    save(folder/'summary.json',summary)
    print(folder.parent.name,folder.name,'accepted',len(accepted),flush=True)
    return summary


def case(ident,cache,source,folder):
    if folder.exists():raise ValueError('Ablation outputs must be fresh')
    folder.mkdir(parents=True)
    expected=read(cache/'baseline/provenance.json')['image_sha256']
    if sha(source)!=expected:raise ValueError('Source differs from frozen baseline')
    save(folder/'provenance.json',{'source_run':str(cache),'source':str(source),'source_sha256':expected,
        'baseline_scene_sha256':sha(cache/'baseline/scene.json'),
        'baseline_render_sha256':sha(cache/'baseline/native.png'),
        'policy':'Both arms replay the same baseline and OCR evidence; only single-unit template policy differs',
        'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),*ROOT.joinpath('src').rglob('*.py')]}})
    summaries={}
    for enabled in (True,False):
        name='template_on' if enabled else 'template_off'
        summaries[name]=branch(source,cache,folder/name,enabled)
    with Image.open(source) as im:src=im.convert('RGB')
    on=Image.open(folder/'template_on/final/native.png').convert('RGB')
    off=Image.open(folder/'template_off/final/native.png').convert('RGB')
    initial=read(cache/'baseline/scene.json')['texts']
    def map_rows(name):
        result={}
        for r in read(folder/name/'final/scene.json')['texts']:
            result.setdefault(r.get('parent_id',r['id']),[]).append(r)
        return result
    a,b=map_rows('template_on'),map_rows('template_off')
    comparison=[]
    for row in initial:
        ident=row['id'];ra=a.get(ident,[row]);rb=b.get(ident,[row])
        metric=feedback.evidence(src,on,off,[row,*ra,*rb])
        delta=metric['after']['score']-metric['before']['score']
        comparison.append({'id':ident,'text':row['text'],'kind':row['kind'],'on':metric['before']['score'],
            'off':metric['after']['score'],'delta':delta,'metric':metric,
            'changed':[(r['bbox_2d'],r.get('font_size_px')) for r in ra]!=[(r['bbox_2d'],r.get('font_size_px')) for r in rb]})
    summary={'case':folder.name,'objects':len(initial),'on_mean':float(np.mean([r['on'] for r in comparison])),
        'off_mean':float(np.mean([r['off'] for r in comparison])),
        'improved_over_2_points':[r['id'] for r in comparison if r['delta']>.02],
        'regressed_over_2_points':[r['id'] for r in comparison if r['delta']<-.02],
        'changed_objects':sum(r['changed'] for r in comparison),'branches':summaries,
        'note':'Same-ROI CPU proxy includes non-text source graphics; not ground truth or visual approval.'}
    save(folder/'comparison_metrics.json',comparison);save(folder/'summary.json',summary)
    compare([source,folder/'template_on/final/native.png',folder/'template_off/final/native.png'],
            ['Original','Template ON','Template OFF'],folder/'comparison.png',width_cap=1200)
    print('CASE_DONE',json.dumps({k:v for k,v in summary.items() if k!='branches'}),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Use a fresh batch directory')
    args.output.mkdir(parents=True)
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    records=read(args.manifest)['cases'];results=[];errors=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(case,r['id'],Path(r['cache']).resolve(),Path(r['image']).resolve(),args.output/r['id']):r for r in records}
        for future in as_completed(jobs):
            r=jobs[future]
            try:results.append(future.result())
            except Exception as exc:
                import traceback
                traceback.print_exc();errors.append({'case':r['id'],'error':str(exc)})
            save(args.output/'progress.json',{'completed':len(results),'failed':errors,'total':len(records)})
    save(args.output/'summary.json',{'cases':sorted(results,key=lambda r:r['case']),'errors':errors,
        'new_api_calls':0,'new_ocr_calls':0,
        'improved_objects':sum(len(r['improved_over_2_points']) for r in results),
        'regressed_objects':sum(len(r['regressed_over_2_points']) for r in results)})


if __name__=='__main__':main()
