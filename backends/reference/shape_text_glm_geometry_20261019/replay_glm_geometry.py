"""Replay cached GLM boxes verbatim, with native CPU rendering and no API calls."""
import argparse
from copy import deepcopy
from pathlib import Path
import os
import shutil
import sys
import threading

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'.deps'),str(ROOT/'src'),str(ROOT/'vendor')]
from PIL import Image
from shape_integration.common import read,save,sha,compare
from shape_integration.texts import options,export,layout,glm_geometry
from shape_integration.texts import render_feedback as feedback
from shape_integration.texts.ocr_refinement import proposal
from shape_integration.texts.ocr_multiline import conflict_parents
from shape_integration.texts.bbox_color_review import DEFAULT_MAX_SCORE_DROP,retain_nonregressing


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('old-run','image','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--max-score-drop',type=float,default=DEFAULT_MAX_SCORE_DROP,
                        help='Absolute tolerance on the 0–1 scale; .05 means five points')
    args=parser.parse_args()
    if not 0<=args.max_score_drop<=1:raise ValueError('Invalid score tolerance')
    args.old_run=args.old_run.resolve();args.output=args.output.resolve();args.image=args.image.resolve()
    if args.output.exists():raise ValueError('Use a fresh output directory')
    expected=read(args.old_run/'baseline/provenance.json')['image_sha256']
    if sha(args.image)!=expected:raise ValueError('Cached source image mismatch')
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    args.output.mkdir(parents=True)
    opt=options(read(ROOT/'config.json'));slots=threading.BoundedSemaphore(1)
    with Image.open(args.image) as im:source=im.convert('RGB')
    baseline=args.old_run/'baseline'
    rows=read(baseline/'scene.json')['texts'];by_id={r['id']:r for r in rows}
    with Image.open(baseline/'native.png') as im:before=im.convert('RGB')
    jobs=read(args.old_run/'ocr_manifest.json')['jobs']
    updates={};decisions=[];inputs={}
    for job in jobs:
        report_path=Path(job['output'])/'report.json'
        if not report_path.is_relative_to(args.old_run):raise ValueError('Cached report outside source run')
        report=read(report_path) if report_path.is_file() else {'detections':[]}
        if report_path.is_file():inputs[str(report_path.relative_to(args.old_run))]=sha(report_path)
        for name,digest in report.get('input_sha256',{}).items():
            if sha(Path(job['crop_dir'])/name)!=digest:raise ValueError('Cached crop hash mismatch')
        for ident in job['ids']:
            original=by_id[ident]
            detections=[d for d in report['detections'] if d.get('target_id')==ident]
            try:
                children,audit=glm_geometry.propose_lines(original,detections,source,rows)
                if children:
                    candidates=children
                    decisions.append({'id':ident,**audit})
                elif original.get('line_layout_status')=='requires_split':
                    raise ValueError('No complete model line boxes; preserve original bbox')
                else:
                    audit=glm_geometry.select_geometry(original,detections,source.size,rows)
                    candidates=[proposal(original,audit['bbox_2d'],audit)]
                    decisions.append({'id':ident,**audit})
                # Existing native metrics are independent of source crop bounds.
                # Child lines need fresh complete-line metrics, never a parent template.
                if len(candidates)>1:
                    fitted,errors=layout.fit_all(candidates,args.output/'typography'/ident,slots,opt,source=None)
                    if errors:save(args.output/'typography'/ident/'warnings.json',errors)
                else:
                    metric=deepcopy(original['layout']['measurement'])
                    fitted=[layout.fit(candidates[0],metric,opt['font_fit_fraction'])]
                updates[ident]=fitted
            except (ValueError,KeyError,RuntimeError) as exc:
                decisions.append({'id':ident,'status':'retained_prior','reason':str(exc)})
    conflicts=feedback.new_collisions(rows,feedback.replace(rows,updates))
    blocked=conflict_parents(updates,conflicts)
    updates={i:v for i,v in updates.items() if i not in blocked}
    save(args.output/'geometry_decisions.json',decisions)
    save(args.output/'provenance.json',{'api_calls':0,'source_run':str(args.old_run),
        'image_sha256':expected,'source_scene_sha256':sha(baseline/'scene.json'),
        'cached_report_sha256':inputs,'bbox_policy':'model_bbox_without_cpu_tightening_v1',
        'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in
                       [Path(__file__),ROOT/'run_global_ocr_review.py',*ROOT.joinpath('src').rglob('*.py')]}})
    print('cached_geometry_candidates',len(updates),'baseline_texts',len(rows),flush=True)
    rounds=[];state={}
    def evaluate(candidates):
        folder=args.output/'cpu_guard'/f'round_{len(rounds)+1:02d}';folder.mkdir(parents=True)
        chosen=feedback.replace(rows,candidates)
        print('cpu_render',len(rounds)+1,'candidates',len(candidates),flush=True)
        if candidates:export.final(chosen,args.image,folder,slots,opt)
        else:
            for name in ('text.pptx','native.png','native.pdf'):shutil.copy2(baseline/name,folder/name)
        with Image.open(folder/'native.png') as im:actual=im.convert('RGB')
        metrics={i:feedback.evidence(source,before,actual,[by_id[i]],parts) for i,parts in candidates.items()}
        save(folder/'scene.json',{'texts':chosen,'source_size':list(source.size)})
        save(folder/'scores.json',metrics);rounds.append(folder);state.update(folder=folder,image=actual,rows=chosen)
        return metrics
    accepted,history=retain_nonregressing(updates,evaluate,args.max_score_drop)
    save(args.output/'cpu_guard/history.json',history)
    shutil.copytree(state['folder'],args.output/'final')
    compare([args.image,args.old_run/'final/native.png',args.output/'final/native.png'],
            ['Original','Previous CPU-tightened boxes','GLM boxes, no CPU tightening'],
            args.output/'comparison.png',width_cap=1800)
    old={r['id']:r for r in read(args.old_run/'final/scene.json')['texts']}
    changes=[{'id':r['id'],'text':r['text'],'old_bbox':old[r['id']]['bbox_2d'],'bbox_2d':r['bbox_2d']}
             for r in state['rows'] if r['id'] in old and r['bbox_2d']!=old[r['id']]['bbox_2d']]
    save(args.output/'summary.json',{'state':'completed','api_calls':0,'texts':len(state['rows']),
        'model_bbox_candidates':len(updates),'accepted_ids':list(accepted),
        'retained_prior_ids':[i for i in by_id if i not in accepted],
        'collision_rejected_ids':sorted(blocked),'changed_boxes':changes,
        'max_score_drop':args.max_score_drop,'metric':'same-ROI source-contrast proxy, not ground-truth fidelity'})
    print('completed','accepted',len(accepted),'changed_boxes',len(changes),args.output,flush=True)


if __name__=='__main__':main()
