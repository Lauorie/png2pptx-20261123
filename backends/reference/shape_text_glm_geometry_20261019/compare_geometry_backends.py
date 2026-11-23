"""Compare two geometry backends on an identical pre-refinement baseline."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'.deps'),str(ROOT/'src'),str(ROOT/'vendor')]
from PIL import Image
from shape_integration.common import read, save, sha, compare
from shape_integration.texts.render_feedback import evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('image','old','new'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    source_sha=sha(args.image)
    for run in (args.old,args.new):
        if read(run/'baseline/provenance.json')['image_sha256']!=source_sha:
            raise ValueError('Source image mismatch')
    for name in ('scene.json','native.png','text.pptx'):
        if sha(args.old/'baseline'/name)!=sha(args.new/'baseline'/name):
            raise ValueError('Baselines are not identical: '+name)
    baseline=read(args.old/'baseline/scene.json')['texts']
    scenes=[read(run/'final/scene.json')['texts'] for run in (args.old,args.new)]
    original=Image.open(args.image).convert('RGB')
    before=Image.open(args.old/'final/native.png').convert('RGB')
    after=Image.open(args.new/'final/native.png').convert('RGB')
    records=[]
    for row in baseline:
        candidates=[[r for r in rows if r['id']==row['id'] or r.get('parent_id')==row['id']] for rows in scenes]
        if any(not group for group in candidates):raise ValueError('Missing target: '+row['id'])
        m=evidence(original,before,after,[row],[*candidates[0],*candidates[1]])
        records.append({'id':row['id'],'text':row['text'],'old':100*m['before']['score'],
                        'new':100*m['after']['score'],'delta':100*(m['after']['score']-m['before']['score']),
                        'evidence':m})
    summary={'targets':len(records),'old_mean':sum(r['old'] for r in records)/len(records),
             'new_mean':sum(r['new'] for r in records)/len(records),
             'improved_over_2_points':[r['id'] for r in records if r['delta']>2],
             'regressed_over_2_points':[r['id'] for r in records if r['delta']< -2],
             'metric':'same ROI foreground soft-Dice/color proxy; not ground-truth fidelity',
             'records':records}
    save(args.new/'backend_comparison.json',summary)
    compare([args.image,args.old/'final/native.png',args.new/'final/native.png'],
            ['Original','PaddleOCR proposals + CPU','GLM proposals + CPU'],
            args.new/'backend_comparison.png',width_cap=1600)
    print({k:v for k,v in summary.items() if k!='records'})


if __name__=='__main__':main()
