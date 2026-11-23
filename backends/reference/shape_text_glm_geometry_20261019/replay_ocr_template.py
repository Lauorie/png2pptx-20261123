"""Render the current template policy from cached evidence; compare prior renders."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
from pathlib import Path

from ablate_ocr_template import ROOT, branch, read, save, sha, compare, feedback, Image
from shape_integration.texts.ocr_geometry import OCR_TEMPLATE_MIN_SCORE


def run_case(record, previous, output):
    ident=record['id'];cache=Path(record['cache']).resolve();source=Path(record['image']).resolve()
    folder=output/ident;old=previous/ident
    if folder.exists():raise ValueError('Use a fresh case output')
    provenance=read(old/'provenance.json')
    if (sha(source)!=provenance['source_sha256']
            or sha(cache/'baseline/scene.json')!=provenance['baseline_scene_sha256']
            or sha(cache/'baseline/native.png')!=provenance['baseline_render_sha256']):
        raise ValueError('Comparison baseline/source mismatch')
    for job in read(cache/'ocr_manifest.json')['jobs']:
        if not (Path(job['output'])/'report.json').is_file():
            raise ValueError('Missing cached OCR report')
    folder.mkdir(parents=True)
    save(folder/'provenance.json',{'source':str(source),'cache':str(cache),'previous':str(old),
        'source_sha256':sha(source),'baseline_scene_sha256':sha(cache/'baseline/scene.json'),
        'template_min_score':OCR_TEMPLATE_MIN_SCORE,'new_api_calls':0,'new_ocr_calls':0,
        'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'ablate_ocr_template.py',*ROOT.joinpath('src').rglob('*.py')]}})
    result=branch(source,cache,folder,True)
    initial=read(cache/'baseline/scene.json')['texts'];ids={r['id'] for r in initial}
    paths={'old_template':old/'template_on/final','no_template':old/'template_off/final','current_template':folder/'final'}
    groups={};images={}
    for name,path in paths.items():
        groups[name]={}
        for r in read(path/'scene.json')['texts']:
            owner=r['id'] if r['id'] in ids else r.get('parent_id')
            if owner not in ids:raise ValueError('Unmapped rendered object')
            groups[name].setdefault(owner,[]).append(r)
        if set(groups[name])!=ids:raise ValueError('Rendered object coverage mismatch')
        with Image.open(path/'native.png') as im:images[name]=im.convert('RGB')
    with Image.open(source) as im:src=im.convert('RGB')
    if any(im.size!=src.size for im in images.values()):raise ValueError('Pixel grid mismatch')
    metrics=[]
    for r in initial:
        roi_rows=[r,*[v for group in groups.values() for v in group[r['id']]]]
        scores={name:feedback.evidence(src,im,im,roi_rows)['before']['score'] for name,im in images.items()}
        metrics.append({'id':r['id'],'text':r['text'],'scores':scores,
            'delta_vs_old':scores['current_template']-scores['old_template'],
            'delta_vs_off':scores['current_template']-scores['no_template']})
    summary={'case':ident,'objects':len(initial),'template_min_score':OCR_TEMPLATE_MIN_SCORE,
        'means':{name:sum(r['scores'][name] for r in metrics)/len(metrics) for name in paths},
        'improved_over_2_points_vs_old':[r['id'] for r in metrics if r['delta_vs_old']>.02],
        'regressed_over_2_points_vs_old':[r['id'] for r in metrics if r['delta_vs_old']<-.02],
        'render':result,'metric_note':'Same-ROI source-contrast CPU proxy, includes non-text graphics; not ground truth.'}
    save(folder/'comparison_metrics.json',metrics);save(folder/'summary.json',summary)
    compare([source,paths['old_template']/'native.png',paths['current_template']/'native.png',paths['no_template']/'native.png'],
        ['Original','Previous template 70%','Current template 50%','Template OFF'],folder/'comparison.png',width_cap=1200)
    print('DONE',ident,summary['means'],flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.output.exists() and not args.resume:raise ValueError('Use a fresh output or --resume')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    records=read(args.manifest)['cases'];results=[];errors=[];pending=[]
    for record in records:
        folder=args.output/record['id']
        if args.resume and folder.exists():
            prov=read(folder/'provenance.json')
            if prov['source_sha256']!=sha(Path(record['image'])) or prov['template_min_score']!=OCR_TEMPLATE_MIN_SCORE:
                raise ValueError('Resume source/policy mismatch')
            if prov['baseline_scene_sha256']!=sha(Path(record['cache'])/'baseline/scene.json'):
                raise ValueError('Resume baseline mismatch')
            summary=read(folder/'summary.json') if (folder/'summary.json').exists() else {}
            if 'means' in summary and all((folder/name).is_file() for name in ['comparison.png','final/text.pptx','final/native.png','comparison_metrics.json']):
                results.append(summary);continue
            archive=args.output/'interrupted_attempts';archive.mkdir(exist_ok=True)
            folder.rename(archive/(record['id']+'_'+str(time.time_ns())))
        pending.append(record)
    save(args.output/'progress.json',{'completed':len(results),'total':len(records),'failed':[]})
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(run_case,r,args.previous,args.output):r['id'] for r in pending}
        for future in as_completed(jobs):
            try:results.append(future.result())
            except Exception as exc:
                import traceback
                traceback.print_exc();errors.append({'case':jobs[future],'error':str(exc)})
            save(args.output/'progress.json',{'completed':len(results),'total':len(records),'failed':errors})
    save(args.output/'summary.json',{'cases':sorted(results,key=lambda r:r['case']),'errors':errors,
        'template_min_score':OCR_TEMPLATE_MIN_SCORE,'new_api_calls':0,'new_ocr_calls':0})
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
