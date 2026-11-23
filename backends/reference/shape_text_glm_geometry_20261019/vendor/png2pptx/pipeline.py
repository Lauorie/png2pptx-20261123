"""M1 only: existing shape candidates -> model crops -> local points -> fits -> shape audit."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import shutil
import threading
import traceback
from PIL import Image
import httpx
from .api.stream_client import StreamingRelayClient
from .contracts import check_plan,check_refinement,check_audit
from .frames import Crop,scale_point
from . import prompts
from .render import candidate_overlay,local_overlay,global_overlay,export_scene
from .scene import save,now,new_scene,prompt_candidates,apply_refinement,apply_audit,summary


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_image(seed_root,root,item):
    case=item['id'];folder=root/'images'/case;folder.mkdir(parents=True)
    image=dict(item)
    for key,sha in [('original','source_sha256'),('input','input_sha256')]:
        source=seed_root/item[key]
        if digest(source)!=item[sha]:raise ValueError(f'{case}: input SHA mismatch {key}')
        target=folder/('original.png' if key=='original' else 'input_2048.png');shutil.copyfile(source,target)
        image[key]=str(target.relative_to(root))
    with Image.open(root/image['input']) as im:
        if list(im.size)!=image['input_size'] or max(im.size)!=2048:raise ValueError('Input size mismatch')
    with Image.open(root/image['original']) as im:
        if list(im.size)!=image['source_size']:raise ValueError('Original image size mismatch')
    geometry_path=seed_root/'shape_bbox'/'results'/case/'absolute'/'geometry.json'
    geometry=json.loads(geometry_path.read_text())
    if geometry['task']!='shape_bbox' or geometry['model_units']!='absolute' or geometry['image_size']!=image['source_size']:raise ValueError('Wrong seed geometry contract')
    shutil.copyfile(geometry_path,folder/'seed_geometry.json')
    image['seed_geometry']={'path':str(geometry_path),'sha256':digest(geometry_path),
        'coordinate_import':'Invert legacy endpoint-aligned decoding to recover the originally predicted global INPUT bbox. New local observations use Pillow half-pixel mapping; imported candidates are only fallible proposals.'}
    candidates=[]
    for i,old in enumerate(geometry['objects'],1):
        box=[v*(image['input_size'][j%2]-1)/(image['source_size'][j%2]-1) for j,v in enumerate(old['bbox_2d'])]
        candidates.append({'id':f'S{i:04d}','seed_id':old['id'],'label':old.get('label',''),
            'proposed_type':old.get('kind','unknown'),'bbox_global_pixels':box,'seed_bbox_original_legacy':old['bbox_2d'],
            'bbox_original_pixels':scale_point(box[:2],image['input_size'],image['source_size'])+scale_point(box[2:],image['input_size'],image['source_size']),
            'provenance':image['seed_geometry']})
    return new_scene(image,candidates)


def process_image(scene,root,client,max_targets=4,context_rounds=1):
    case=scene['image']['id'];case_root=root/'results'/case;case_root.mkdir(parents=True,exist_ok=True)
    scene_path=case_root/'scene.json';save(scene_path,scene)
    image=Image.open(root/scene['image']['input']).convert('RGB');candidates=prompt_candidates(scene)
    def record_error(stage,exc,ident=None):
        scene['issues'].append({'type':'execution_error','stage':stage,'object_id':ident,'error':str(exc),'utc':now()})
        if ident:scene['objects'][ident]['status']='failed'
        save(case_root/(stage.replace('/','_')+'_error.json'),{'error':str(exc),'traceback':traceback.format_exc()})
        save(scene_path,scene);print(f'{case} {stage} FAILED: {str(exc)[:240]}',flush=True)
    try:
        raw=client.call(f'{case}/plan',prompts.SYSTEM,prompts.planning_prompt(image.size,candidates,max_targets),
            [('Clean global input; all crop coordinates refer to this image',root/scene['image']['input'])],
            lambda r:check_plan(r,candidates,max_targets,image.size))
        scene['crop_plan']=raw
        for region in raw['regions']:
            for ident in region['target_ids']:scene['objects'][ident].update(selected=True,status='selected_pending')
        save(scene_path,scene);print(f'{case} plan selected {sum(o["selected"] for o in scene["objects"].values())}',flush=True)
    except Exception as exc:record_error('plan',exc);raw={'regions':[]}
    for region in raw['regions']:
        for ident in region['target_ids']:
            try:
                crop=Crop.from_model(region['bbox_2d'],image.size)
                for attempt in range(context_rounds+1):
                    attempt_dir=case_root/'objects'/ident/f'context_{attempt}';attempt_dir.mkdir(parents=True)
                    clean=crop.image(image);clean.save(attempt_dir/'input_2048.png')
                    obj=scene['objects'][ident];box=obj['bbox_global_pixels']
                    local_box=crop.to_local(box[:2])+crop.to_local(box[2:])
                    candidate_overlay(clean,local_box,ident).save(attempt_dir/'candidate_overlay_2048.png')
                    frame=crop.metadata(scene['image']['source_size']);save(attempt_dir/'frame.json',frame)
                    candidate={'id':ident,'label':obj['label'],'proposed_type':obj['proposed_type'],
                        'bbox_in_crop_input_pixels':local_box,'bbox_in_global_input_pixels':box,
                        'bbox_warning':'Fallible identity hint; may extend outside crop or deviate from the actual contour. Infer target geometry from clean pixels.'}
                    response=client.call(f'{case}/refine/{ident}/context_{attempt}',prompts.SYSTEM,prompts.refine_prompt(candidate,frame),
                        [('Image 1: clean crop; return points in this actual pixel frame',attempt_dir/'input_2048.png'),
                         ('Image 2: same crop with approximate candidate box; annotation is not source geometry',attempt_dir/'candidate_overlay_2048.png')],
                        lambda r:check_refinement(r,ident,crop.input_size,crop.global_size))
                    provenance={'request':f'calls/{case}/refine/{ident}/context_{attempt}/request.json',
                        'crop':frame,'source_sha256':digest(attempt_dir/'input_2048.png'),'response_sha256':digest(client.root/f'{case}/refine/{ident}/context_{attempt}/response_0.json')}
                    apply_refinement(scene,ident,response,crop,provenance);save(scene_path,scene)
                    if response['decision']=='refined':
                        save(attempt_dir/'fitted_object.json',scene['objects'][ident])
                        local_overlay(scene,ident,crop,clean).save(attempt_dir/'geometry_overlay.png')
                    print(f'{case} {ident} context {attempt}: {response["decision"]}',flush=True)
                    if response['decision']!='needs_context':break
                    if attempt==context_rounds:
                        scene['objects'][ident]['status']='context_budget_exhausted'
                        scene['issues'].append({'type':'context_budget_exhausted','object_id':ident,'suggested_global_crop':response['suggested_global_crop']})
                        break
                    next_crop=Crop.from_model(response['suggested_global_crop'],image.size)
                    if next_crop.box==crop.box:raise ValueError('needs_context repeated the same crop; retained unresolved rather than looping')
                    crop=next_crop
            except Exception as exc:record_error('refine_'+ident,exc,ident)
            save(scene_path,scene)
    refined=[o['id'] for o in scene['objects'].values() if o['status']=='refined_unverified']
    # Keep audits even if a subset failed: omissions and all unselected candidates remain visible.
    if scene['crop_plan'] is not None:
        try:
            overlay=case_root/'audit_input_2048.png';global_overlay(scene,image).save(overlay)
            audit=client.call(f'{case}/audit',prompts.SYSTEM,prompts.shape_audit_prompt(image.size,prompt_candidates(scene),refined),
                [('Image 1: clean global source',root/scene['image']['input']),('Image 2: global candidate inventory and refined geometry',overlay)],
                lambda r:check_audit(r,list(scene['objects']),refined,image.size))
            apply_audit(scene,audit)
        except Exception as exc:record_error('audit',exc)
    scene['ended_utc']=now();save(scene_path,scene)
    result=summary(scene)
    try:result['export_counts']=export_scene(scene,case_root,image)
    except Exception as exc:record_error('export',exc);result['export_error']=str(exc)
    save(case_root/'summary.json',result);print(json.dumps(result,ensure_ascii=False),flush=True);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base-url',default='http://10.102.252.163:10088/v1');p.add_argument('--model',default='GLM-5.3-Flash')
    p.add_argument('--workers',type=int,default=6);p.add_argument('--max-targets',type=int,default=4)
    p.add_argument('--context-rounds',type=int,default=1);p.add_argument('--max-tokens',type=int,default=24000)
    p.add_argument('--cases',nargs='+',help='Optional case subset for explicit regression experiments; default is every input image.')
    args=p.parse_args(argv)
    if min(args.workers,args.max_targets,args.max_tokens)<1 or args.context_rounds<0:p.error('Invalid budget')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    seed=args.seed_run.resolve();old=json.loads((seed/'manifest.json').read_text())
    project=Path(__file__).resolve().parents[1]
    with httpx.Client(trust_env=False,timeout=20) as session:
        response=session.get(args.base_url.rstrip('/')+'/models');response.raise_for_status();models=response.json()
    if args.model not in [m['id'] for m in models['data']]:raise ValueError('Requested model absent from live endpoint')
    save(root/'live_models.json',models)
    snapshot=root/'code_snapshot';snapshot.mkdir()
    for name in ['png2pptx','tests','design']:
        shutil.copytree(project/name,snapshot/name,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['run_m1.py','REQUIREMENTS.zh-CN.md','README.md']:shutil.copyfile(project/name,snapshot/name)
    hashes={str(path.relative_to(snapshot)):digest(path) for path in sorted(snapshot.rglob('*')) if path.is_file()};save(root/'code_sha256.json',hashes)
    if args.cases and set(args.cases)-{i['id'] for i in old['images']}:raise ValueError('Requested case is absent from source manifest')
    scenes=[prepare_image(seed,root,item) for item in sorted(old['images'],key=lambda x:x['id']) if not args.cases or item['id'] in args.cases]
    manifest={'schema_version':'png2pptx.m1.1','started_utc':now(),'seed_run':str(seed),
        'run_kind':'fresh local refinement/audit calls using cached bbox candidates; representative development sample',
        'prompt_version':prompts.VERSION,'base_url':args.base_url,'model':args.model,'thinking':'on',
        'max_tokens':args.max_tokens,'max_attempts':1,'temperature':0.1,'workers':args.workers,
        'max_targets_per_image':args.max_targets,'context_rounds':args.context_rounds,'images':[s['image'] for s in scenes],
        'candidate_count':sum(len(s['objects']) for s in scenes),'input_long_edge':2048,
        'coordinate_policy':'Pillow half-pixel mapping for every new observation; legacy decoded candidate boxes inverted to their original input frame',
        'token_budget_note':'Single-object requests use a configurable 24000 default to bound pathological repetitive generation. This differs from the old 120000 budget. Exhaustion is a failed request, never a valid empty result.'}
    save(root/'manifest.json',manifest)
    client=StreamingRelayClient(root/'calls',args.model,threading.Semaphore(args.workers),args.base_url,None,
        max_tokens=args.max_tokens,max_attempts=1,read_timeout=1800,thinking='on')
    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(process_image,s,root,client,args.max_targets,args.context_rounds):s['image']['id'] for s in scenes}
        for future in as_completed(jobs):
            try:results.append(future.result())
            except Exception as exc:results.append({'case':jobs[future],'fatal_error':str(exc),'traceback':traceback.format_exc()})
            save(root/'summary.json',{'state':'running','completed_images':len(results),'results':sorted(results,key=lambda r:r['case'])})
    save(root/'summary.json',{'state':'ended','ended_utc':now(),'completed_images':len(results),'results':sorted(results,key=lambda r:r['case'])})


if __name__=='__main__':main()
