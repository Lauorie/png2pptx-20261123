"""Controlled bbox-reference experiment; text and font properties remain fixed."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os
from pathlib import Path
import sys
import threading

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'.deps'),str(ROOT/'src'),str(ROOT/'vendor')]
from PIL import Image
from shape_integration.common import Runtime,read,save,sha,compare,usage,valid_box
from shape_integration.texts import options,layout,export
from shape_integration.texts.controller import frame,mapped_box,local_hints
from shape_integration.texts.local_geometry import validate,SYSTEM,union
from shape_integration.texts import render_feedback as feedback
from review_rendered_text import verify_rendered


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--ids',nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Use a fresh output directory')
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    rows=read(args.baseline/'scene.json')['texts'];by_id={r['id']:r for r in rows}
    targets=[by_id[i] for i in args.ids]
    if len(set(args.ids))!=len(args.ids):raise ValueError('Duplicate target')
    if read(args.baseline/'provenance.json')['image_sha256']!=sha(args.image):raise ValueError('Source mismatch')
    source=Image.open(args.image).convert('RGB');before=Image.open(args.baseline/'native.png').convert('RGB')
    if source.size!=before.size:raise ValueError('Canvas mismatch')
    key=os.environ.get('OpenAI_API') or os.environ.get('OPENAI_API_KEY') or os.environ.get('BOYUE_API_KEY')
    if not key:raise ValueError('API key required in environment')
    cfg=read(ROOT/'config.json');cfg['api_concurrency']=2;opt=options(cfg)
    runtime=Runtime(cfg,cfg['expected_boyue_base_url'],key).for_stage(reasoning_effort='low',shape_schema_retries=1)
    args.output.mkdir(parents=True)
    hashes={str(p):sha(p) for p in [args.image,args.baseline/'scene.json',args.baseline/'native.png',args.baseline/'text.pptx']}
    save(args.output/'provenance.json',{'sources':hashes,'experiment':'same prompt/source; additional bbox reference only',
         'frozen':'content, font family, weight, color and rotation; disable CPU bbox/font reselection to isolate VLM bbox effect',
         'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'review_rendered_text.py',*ROOT.joinpath('src').rglob('*.py')]}})
    f=frame(args.image,args.output/'input',{**opt,'crop_min_padding_px':64,'crop_padding':.3,'crop_scale':100},union(targets))
    local_hints(args.output/'input/clean.png',targets,f,args.output/'input/reference.png')
    prompt={'task':'Correct the tight visible glyph bounding box of each supplied text occurrence.',
        'image_size':f['size'],'coordinate_system':'local image pixel centers, [xmin,ymin,xmax,ymax]',
        'targets':[{'id':r['id'],'text':r['text'],'predicted_bbox_2d':mapped_box(r['bbox_2d'],f),'rotation_deg':r['rotation_deg']} for r in targets],
        'rules':['The clean SOURCE is the sole content and geometry authority. Numerical predicted boxes are fallible starting points, never ground truth.',
            'If a REFERENCE image is present, its colored thin rectangles denote those same predicted boxes and ID labels are annotations. SOURCE and REFERENCE have exactly the same crop and scale. Do not count a rectangle or ID label as part of a glyph.',
            'Inspect all visible characters, dots, ascenders and descenders. Determine whether the current box cuts off glyphs or contains empty margins. Expand, shrink or move it to enclose the complete actual text ink. Do not preserve a small box merely because it was supplied.',
            'Use neighboring baselines as spatial context but exclude their ink. Preserve target identity; do not jump to a repeated word. Do not change content, style, line grouping or rotation, and do not estimate font size.',
            'Return every ID once. If visible and unambiguous, status=located, direction_status=confirmed and preserve rotation_deg. If clipped or ambiguous, bbox_2d=null and status=clipped/ambiguous. Give a brief visible-evidence reason, not a reasoning trace.'],
        'response_format':{'items':[{'id':'supplied ID','status':'located/clipped/ambiguous','bbox_2d':[10,20,150,50],
            'rotation_deg':0,'direction_status':'confirmed/ambiguous','reason':'brief observed boundary issue'}]}}
    save(args.output/'prompt.json',prompt)
    def request(arm):
        images=[('SOURCE: clean original crop.',args.output/'input/clean.png')]
        if arm=='with_reference':images.append(('REFERENCE: fallible predicted bbox, same coordinates; not glyph contours.',args.output/'input/reference.png'))
        return runtime.call(args.output/arm/'calls','bbox',SYSTEM,prompt,images,
                            lambda raw:validate(raw,args.ids,f['size']),opt['max_tokens'])
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs={arm:pool.submit(request,arm) for arm in ('clean_only','with_reference')}
        results={arm:future.result() for arm,future in jobs.items()}
    slots=threading.BoundedSemaphore(1);summary=[]
    for arm,result in results.items():
        folder=args.output/arm;save(folder/'geometry.json',result)
        incoming=[]
        for item in result['items']:
            if item['status']!='located' or item['direction_status']!='confirmed':continue
            old=by_id[item['id']]
            if item['rotation_deg']!=old['rotation_deg']:raise ValueError('Rotation changed in bbox-only experiment')
            row={k:deepcopy(old[k]) for k in ('id','text','runs','kind','latex','font','bold','italic','underline','color','rotation_deg','legibility') if k in old}
            row['bbox_2d']=mapped_box(item['bbox_2d'],f,True)
            if not valid_box(row['bbox_2d'],source.size):raise ValueError('Mapped box invalid')
            incoming.append(row)
        fitted,errors=layout.fit_all(incoming,folder/'typography',slots,
            {**opt,'pixel_reflow_enabled':False,'source_font_selection':False,'glyph_registration_enabled':False},source=args.image)
        updates={r['id']:[r] for r in fitted}
        candidate_rows=feedback.replace(rows,updates);candidate=folder/'candidate';candidate.mkdir()
        export.final(candidate_rows,args.image,candidate,slots,opt)
        save(candidate/'scene.json',{'texts':candidate_rows})
        actual=Image.open(candidate/'native.png').convert('RGB');checks=[]
        for row in fitted:
            old=by_id[row['id']];metric=feedback.evidence(source,before,actual,[old],[row]);ok,reason=feedback.gate(metric)
            conflicts=feedback.new_collisions(rows,feedback.replace(rows,{row['id']:[row]}))
            if conflicts or errors:ok,reason=False,'new_collision_or_measurement_error'
            checks.append({'id':row['id'],'passed':ok,'reason':reason,'evidence':metric,
                'old_bbox':old['bbox_2d'],'new_bbox':row['bbox_2d'],'old_font_size_px':old['font_size_px'],'new_font_size_px':row['font_size_px']})
        model=[];verify_errors=[]
        if len(checks)==len(targets) and all(c['passed'] for c in checks):
            model,verify_errors=verify_rendered(runtime,folder/'verification',args.image,args.baseline,candidate,rows,updates,[targets],opt,2)
        accepted=feedback.dual_gate(checks,model,args.ids)
        save(folder/'evaluation.json',{'cpu_checks':checks,'model_checks':model,'accepted':accepted,'errors':errors+verify_errors})
        view=folder/'view';view.mkdir()
        feedback.render_evidence(candidate/'native.pdf',actual,f,view)
        summary.append({'arm':arm,'checks':checks,'accepted':accepted,'model_checks':model,'errors':errors+verify_errors})
        print(arm,[(c['id'],c['old_font_size_px'],c['new_font_size_px'],c['evidence']['after']['score'],c['passed']) for c in checks],flush=True)
    original_view=args.output/'before';original_view.mkdir()
    feedback.render_evidence(args.baseline/'native.pdf',before,f,original_view)
    compare([args.output/'input/clean.png',original_view/'render.png',args.output/'clean_only/view/render.png',args.output/'with_reference/view/render.png'],
        ['Source','Before','Clean-only candidate','BBox-reference candidate'],args.output/'comparison.png',width_cap=650)
    save(args.output/'summary.json',summary);save(args.output/'usage.json',usage(args.output))
    assert all(sha(Path(p))==value for p,value in hashes.items()),'Source or baseline changed'


if __name__=='__main__':main()
