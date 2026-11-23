"""Re-fit native formulas from a completed run, with no GLM/OCR requests."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys
import threading

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'.deps'),str(ROOT/'src'),str(ROOT/'vendor')]
from PIL import Image
from shape_integration.common import read,save,sha,compare
from shape_integration.texts import options,layout,export,render_feedback as feedback
from shape_integration.texts.bbox_color_review import DEFAULT_MAX_SCORE_DROP,retain_nonregressing


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','image','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--measurement-cache',type=Path)
    parser.add_argument('--ids',nargs='+',help='Only re-fit these formula identities; others stay unchanged')
    parser.add_argument('--no-layout-selection',action='store_true',help='Only remeasure converter output')
    parser.add_argument('--max-score-drop',type=float,default=DEFAULT_MAX_SCORE_DROP,
                        help='Absolute tolerance on the 0–1 scale; .05 means five points')
    args=parser.parse_args()
    if not 0<=args.max_score_drop<=1:raise ValueError('Invalid score tolerance')
    if args.output.exists() and not args.resume:raise ValueError('Use a fresh output directory')
    if (args.output/'final').exists():raise ValueError('Completed output must not be overwritten')
    if read(args.run/'provenance.json')['parameters']['image_sha256']!=sha(args.image):
        raise ValueError('Source image differs from original run')
    args.output.mkdir(parents=True,exist_ok=True)
    parameters={'source_run':str(args.run.resolve()),'image_sha256':sha(args.image),
                'max_score_drop':args.max_score_drop}
    if args.ids:parameters['target_ids']=sorted(args.ids)
    if args.no_layout_selection:parameters['layout_selection']=False
    if (args.output/'replay_parameters.json').exists() and read(args.output/'replay_parameters.json')!=parameters:
        raise ValueError('Resume input changed')
    save(args.output/'replay_parameters.json',parameters)
    if args.measurement_cache and not (args.output/'typography').exists():
        cache_parameters=read(args.measurement_cache/'replay_parameters.json')
        measurement_parameters=lambda p:{k:v for k,v in p.items() if k!='max_score_drop'}
        if measurement_parameters(cache_parameters)!=measurement_parameters(parameters):
            raise ValueError('Measurement cache belongs to different inputs')
        shutil.copytree(args.measurement_cache/'typography',args.output/'typography',
            ignore=shutil.ignore_patterns('profile','recovery_*'))
        save(args.output/'measurement_cache.json',{'source':str(args.measurement_cache.resolve()),
            'policy':'copied cache; exact DOCX XML and complete PDF are revalidated'})
    def progress(stage,**kwargs):
        save(args.output/'progress.json',{'stage':stage,**kwargs});print(stage,kwargs,flush=True)
    opt=options(read(ROOT/'config.json'))
    opt.update(pixel_reflow_enabled=False,glyph_registration_enabled=False,source_font_selection=False)
    if args.no_layout_selection:opt['formula_layout_enabled']=False
    slots=threading.BoundedSemaphore(1)
    if hasattr(__import__('os'),'sched_getaffinity'):
        import os
        os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
    rows=read(args.run/'final/scene.json')['texts'];by_id={r['id']:r for r in rows}
    source=Image.open(args.image).convert('RGB');before=Image.open(args.run/'final/native.png').convert('RGB')
    formulas=[deepcopy(r) for r in rows if r.get('kind')=='formula']
    if args.ids:
        if not set(args.ids)<={r['id'] for r in formulas}:raise ValueError('Unknown/nonformula target IDs')
        formulas=[r for r in formulas if r['id'] in set(args.ids)]
    progress('native_formula_measurement',formulas=len(formulas),api_calls=0)
    fitted,errors=layout.fit_all(formulas,args.output/'typography',slots,opt,source=args.image)
    if {r['id'] for r in fitted}!={r['id'] for r in formulas}:
        raise ValueError('Formula identities changed; refusing replacement')
    if any(r['bbox_2d']!=by_id[r['id']]['bbox_2d'] for r in fitted):
        raise ValueError('Formula source boxes changed; refusing replacement')
    updates={r['id']:[r] for r in fitted if not r['layout']['measurement'].get('preview_degraded')}
    rounds=[];state={}
    def evaluate(chosen):
        folder=args.output/'cpu_guard'/f'round_{len(rounds)+1:02d}'
        folder.mkdir(parents=True)
        final_rows=feedback.replace(rows,chosen)
        progress('native_render',round=len(rounds)+1,candidates=len(chosen),api_calls=0)
        if chosen:export.final(final_rows,args.image,folder,slots,opt)
        else:
            for name in ('text.pptx','native.png','native.pdf'):
                shutil.copy2(args.run/'final'/name,folder/name)
        actual=Image.open(folder/'native.png').convert('RGB')
        metrics={ident:feedback.evidence(source,before,actual,[by_id[ident]],parts)
                 for ident,parts in chosen.items()}
        save(folder/'scene.json',{'texts':final_rows,'source_size':list(source.size)})
        save(folder/'scores.json',metrics)
        rounds.append(folder);state.update(folder=folder,image=actual,rows=final_rows)
        return metrics
    accepted,history=retain_nonregressing(updates,evaluate,args.max_score_drop)
    shutil.copytree(state['folder'],args.output/'final')
    save(args.output/'cpu_guard/history.json',history)
    save(args.output/'provenance.json',{'parameters':{'image_sha256':sha(args.image),
        'source_run':str(args.run.resolve()),'policy':'native_formula_layout_only',
        'max_score_drop':args.max_score_drop},
        'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in ROOT.joinpath('src').rglob('*.py')}})
    compare([args.image,args.run/'final/native.png',args.output/'final/native.png'],
            ['Original','Previous native formulas','Native vector layout; same GLM boxes'],
            args.output/'comparison.png',width_cap=1800)
    report={'formulas':len(formulas),'total_texts':len(rows),'accepted_ids':sorted(accepted),
        'reverted_ids':sorted(set(updates)-set(accepted)),'api_calls':0,
        'source_boxes_unchanged':True,'nonformula_objects_unchanged':True,
        'errors':errors,'max_score_drop':args.max_score_drop,
        'quality_status':'needs_review','metric_is_proxy_not_ground_truth':True}
    save(args.output/'summary.json',report);save(args.output/'usage.json',{'new_api_calls':0})
    progress('completed',**report)


if __name__=='__main__':main()
