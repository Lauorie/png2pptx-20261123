"""Compose verified existing branches with the current ownership/image/text logic.

No shape/arrow/text recognition or SAM requests are scheduled. Inputs remain
read-only, and every case writes to its own resumable composition directory.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace
from zipfile import ZipFile,ZIP_DEFLATED

from .compose import merge,sha,COMPOSITION_PROTOCOL


def read(path):return json.loads(Path(path).read_text())
def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def inspect_sources(shape_run,text_run,arrow_run,ids=None):
    """Bind all three products to the same original; never substitute another run."""
    from PIL import Image,ImageChops
    images=read(shape_run/'source_manifest.json')['images']
    if ids:
        unknown=set(ids)-{i['id'] for i in images}
        if unknown:raise ValueError('Unknown requested cases: '+','.join(sorted(unknown)))
        images=[i for i in images if i['id'] in set(ids)]
    text_inputs={i['id']:i for i in read(text_run/'manifest.json')['cases']}
    arrow_cases=(arrow_run/'geometry/cases' if (arrow_run/'geometry/cases').is_dir() else
                 arrow_run/'cases' if (arrow_run/'cases').is_dir() else arrow_run)
    ready=[];missing=[]
    for item in images:
        cid=item['id'];shape=shape_run/'shape/geometry/cases'/cid;text=text_run/'cases'/cid;arrow=arrow_cases/cid
        if (text/'final').is_dir():text=text/'final'
        files={'shape_scene':shape/'fitting/scene.json','shape_styles':shape/'styling/styles.json',
               'shape_complete':shape/'complete.json',
               'shape_pptx':shape/'styling/styled.pptx','shape_preview':shape/'styling/native.png',
               'image_registry':shape_run/'shape/localization/cases'/cid/'dedup/image_boxes_original.json',
               'arrow_scene':arrow/'scene.json','arrow_pptx':arrow/'arrows.pptx',
               'text_scene':text/'scene.json','text_pptx':text/'text.pptx'}
        absent=[k for k,p in files.items() if not p.is_file()]
        if absent:
            missing.append({'id':cid,'missing':absent,'shape_failure':read(shape/'failure.json').get('error') if (shape/'failure.json').is_file() else None});continue
        completed=read(files['shape_complete'])
        if completed.get('pptx_sha256')!=sha(files['shape_pptx']) or completed.get('geometry_sha256')!=sha(files['shape_scene']):
            raise ValueError('Shape completion checkpoint differs from product: '+cid)
        if sha(item['original'])!=item['source_sha256']:raise ValueError('Original changed: '+cid)
        if text_inputs.get(cid,{}).get('sha256')!=item['source_sha256']:raise ValueError('Text provenance mismatch: '+cid)
        scene=read(files['shape_scene']);arrows=read(files['arrow_scene']);texts=read(files['text_scene']);registry=read(files['image_registry'])
        if scene['source_sha256']!=item['source_sha256'] or registry['source_sha256']!=item['source_sha256']:raise ValueError('Shape provenance mismatch: '+cid)
        if scene['source_size']!=item['source_size'] or arrows['size']!=item['source_size'] or texts['source_size']!=item['source_size']:raise ValueError('Coordinate grids differ: '+cid)
        with Image.open(item['original']) as im,Image.open(arrow/'clean.png') as clean:
            if im.size!=clean.size or ImageChops.difference(im.convert('RGB'),clean.convert('RGB')).getbbox() is not None:raise ValueError('Arrow source pixels differ: '+cid)
        # Existing text.pptx is self-contained: native equations and fallback
        # images travel in its OPC package. Historical measurement PNG paths
        # are exporter scratch files, not inputs to native text composition.
        ready.append({'item':item,'paths':{k:str(p.resolve()) for k,p in files.items()},
                      'sha256':{k:sha(p) for k,p in files.items()}})
    return ready,missing


def prepare(project,args):
    shape_run=args.shape_run.resolve();text_run=args.text_run.resolve();arrow_run=args.arrow_run.resolve();run=args.run_dir.resolve()
    if run.exists():raise ValueError('Use a new composition directory')
    ready,missing=inspect_sources(shape_run,text_run,arrow_run,getattr(args,'ids',None))
    if not ready:raise ValueError('No complete matching source products')
    if args.require_all and missing:raise ValueError('Wait for every requested branch product before composition: '+','.join(r['id'] for r in missing))
    import run_shapes
    dataset=Path(ready[0]['item']['original']).parents[2]
    run_shapes.prepare(SimpleNamespace(run_dir=run/'shape',input_root=dataset,ids=[r['item']['id'] for r in ready],max_stroke_pt=None))
    code=run/'code';code.mkdir()
    shutil.copytree(project/'full_recovery',code/'full_recovery',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (code/'merge_existing.py').write_text('from full_recovery.merge_existing import main\nif __name__ == "__main__": main()\n')
    shutil.copy2(project/'scripts/lo_guard.py',code/'lo_guard.py')
    for name in ['test_full_composition.py','test_full_line_dedup.py','test_full_text_occlusion.py','test_full_text_product.py']:
        shutil.copy2(project/'tests'/name,code/name)
    for name in ['test_source_occlusion_bands.py','test_quality_texture_composition.py']:
        shutil.copy2(project/'tests'/name,code/name.replace('test_','test_full_',1))
    shutil.copy2(project/'tests/test_unified_image_assets.py',code/'test_unified_image_assets.py')
    explicit_runtime=getattr(args,'runtime_source',None)
    anchor=(Path(explicit_runtime).resolve() if explicit_runtime else
            shape_run if (shape_run/'renderer_runtime').is_dir() and (shape_run/'backends/reference_fonts').is_dir()
            else Path(read(project/'runs/LATEST_SHAPE_API.json')['run_dir']))
    old=read(anchor/'settings.json');settings={k:old[k] for k in ['api_model','api_base_url']}
    if getattr(args,'api_base_url',None):settings['api_base_url']=args.api_base_url.rstrip('/')
    from .prepare_api import credential_sources,stage_credentials
    key_sources=credential_sources(project,args)
    staged_keys=stage_credentials(run,key_sources)
    settings.update(api_key_file=staged_keys[0],api_key_files=staged_keys,
                    api_credential_source_files=[str(p.resolve()) for p in key_sources])
    settings.update(workflow_scope='composition',source_project=str(project),project_root=str(project),
        source_shape_run=str(shape_run),source_text_run=str(text_run),source_arrow_run=str(arrow_run),
        runtime_source=str(anchor),
        case_workers=args.case_workers,api_global_concurrency=args.api_concurrency,api_gateway_port=18188,
        api_request_timeout_seconds=3600,glm_url='http://127.0.0.1:18188/v1',
        namespace=getattr(args,'namespace','ailab-intern10'),
        charged_group=getattr(args,'charged_group','intern10_proxy'),gpu=0,
        worker_cpu=getattr(args,'worker_cpu',16),worker_memory_mib=getattr(args,'worker_memory_mib',65536),
        worker_max_seconds=86400,requested=len(ready)+len(missing),expected=len(ready))
    if old.get('native_renderer'):settings['native_renderer']=old['native_renderer']
    if getattr(args,'image_edit_url',None):
        settings['image_edit']={'enabled':True,'base_url':args.image_edit_url,
            'model':getattr(args,'image_edit_model','SenseNova-U1.5-8B-MoT'),
            'concurrency':getattr(args,'image_edit_concurrency',2),'timeout_seconds':180}
    if getattr(args,'reuse_assets_from',None):settings['reuse_assets_from']=str(args.reuse_assets_from.resolve())
    if getattr(args,'reuse_assets_from',None) or args.resume_from:
        settings['reuse_request_roots']=[str(p.resolve()) for p in
            (getattr(args,'reuse_assets_from',None),args.resume_from) if p is not None]
    save(run/'settings.json',settings)
    backend=run/'backends';backend.mkdir()
    from .prepare import freeze_text
    from run_ppt import EXPERIMENTS
    text_source=Path(getattr(args,'text_source',None) or EXPERIMENTS/'shape_text_glm_geometry_20261019').resolve()
    freeze_text(text_source,backend/'text',settings['glm_url'])
    settings['source_text_project']=str(text_source);save(run/'settings.json',settings)
    (backend/'reference_fonts').symlink_to(anchor/'backends/reference_fonts',target_is_directory=True)
    (run/'renderer_runtime').symlink_to(anchor/'renderer_runtime',target_is_directory=True)
    for directory in ['logs','jobs','render_locks','delivery','cases']:(run/directory).mkdir(exist_ok=True)
    save(run/'sources.json',{'ready':ready,'missing':missing,'shape_geometry_recovery':False,'arrow_recovery':False,'text_recognition':False,'sam_requests':0})
    save(run/'source_manifest.json',{'images':[r['item'] for r in ready]})
    save(run/'delivery/missing_sources.json',{'requested':settings['requested'],'available':len(ready),'missing':missing})
    reused=[]
    if args.resume_from:
        previous=args.resume_from.resolve()
        for record in ready:
            cid=record['item']['id'];case=previous/'cases'/cid;marker=case/'completed.json'
            if not marker.is_file():continue
            done=read(marker)
            if done['composition_protocol']!=COMPOSITION_PROTOCOL or done['sources']['sha256']!=record['sha256']:
                raise ValueError('Completed composition has different inputs/policy: '+cid)
            if sha(case/'combined.pptx')!=done['pptx_sha256']:raise ValueError('Completed composition changed: '+cid)
            (run/'cases'/cid).symlink_to(case,target_is_directory=True)
            (run/'delivery'/cid).symlink_to(previous/'delivery'/cid,target_is_directory=True)
            reused.append(cid)
        save(run/'continuation.json',{'source':str(previous),'completed_cases_reused_read_only':reused,
                                     'incomplete_cases_use_fresh_directories':True})
    manifest={str(p.relative_to(run)):sha(p) for p in code.rglob('*') if p.is_file()}
    save(run/'code_manifest.json',manifest)
    save(run/'status.json',{'state':'prepared','expected':len(ready),'missing_sources':len(missing),'at':time.time()})
    print('PREPARED_COMPOSITION',run,'available',len(ready),'missing',len(missing),flush=True)


def execute(run, *, record_stream=None):
    from PIL import Image,ImageDraw
    import fitz
    sys.path[:0]=[str(run/'shape/code/src'),str(run/'shape/code/vendor')]
    from shape_integration.common import Runtime
    from shape_integration.composition import extract
    from .quality_choice import review as choose_native
    from .image_assets import review as choose_texture,commit as commit_assets
    from .handoff import reconcile,surviving_scene
    settings=read(run/'settings.json');sources=read(run/'sources.json')
    records=[] if record_stream is not None else list(sources['ready']);code=run/'code'
    expected=settings.get('expected',settings['requested']) if record_stream is not None else len(records)
    lock=(run/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=read(run/'shape/pipeline_config.json')
    runtime=Runtime({**config,'api_concurrency':settings['api_global_concurrency'],'shape_schema_retries':1,
                     'reasoning_effort':'high','request_timeout_seconds':360,
                     'source_scope_workers':settings.get('source_scope_workers',2),
                     'mapping_workers':settings.get('mapping_workers',2),
                     'reuse_output_root':str(run),'reuse_request_roots':settings.get('reuse_request_roots',[])},settings['glm_url'],'')
    editor=None
    if settings.get('image_edit',{}).get('enabled'):
        from .image_edit import Editor
        editor=Editor(settings['image_edit'])
    progress={};mutex=threading.Lock()
    def notify(cid,**changes):
        with mutex:
            progress.setdefault(cid,{}).update(changes)
            save(run/'progress.json',{'expected':expected,'requested':settings['requested'],
                'completed':sum(r.get('state')=='completed' for r in progress.values()),
                'failed':sum(r.get('state')=='failed' for r in progress.values()),
                'missing_sources':len(sources['missing']),'cases':progress,'at':time.time()})
    def command(cid,tag,args):
        with (run/'logs'/(cid+'_'+tag+'.log')).open('a') as log:
            p=subprocess.run([sys.executable,'-B',str(code/'full_recovery/text_stage.py'),*map(str,args)],cwd=code,stdout=log,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError(tag+' failed; see logs/'+cid+'_'+tag+'.log')
    def one(record):
        item=record['item'];cid=item['id'];paths={k:Path(p) for k,p in record['paths'].items()};folder=run/'cases'/cid;folder.mkdir(exist_ok=True)
        marker=folder/'completed.json'
        for key,p in paths.items():
            if sha(p)!=record['sha256'][key]:raise ValueError('Input product changed: '+cid+'/'+key)
        if marker.exists():
            done=read(marker)
            if done['composition_protocol']!=COMPOSITION_PROTOCOL or sha(folder/'combined.pptx')!=done['pptx_sha256']:raise ValueError('Completed composition changed')
            notify(cid,state='completed');return done
        notify(cid,state='running',stage='ownership')
        # Stroke colour, width and dash are visible facts: read them from the source.
        from .arrow_restyle import restyled
        from .replay_composition import reuse_arguments
        branch_paths=dict(paths)  # the branches' own files, checked again after composition
        paths.update(restyled(paths,Path(item['original']),folder/'arrow_style',settings,code))
        scene=read(paths['shape_scene']);style_data=read(paths['shape_styles']);styles={k:v['style'] for k,v in style_data['parts'].items()}
        arrows=read(paths['arrow_scene']);texts=read(paths['text_scene'])
        from .native_paint import bind as bind_native_paint
        arrows,paint_receipt=bind_native_paint(arrows,paths['arrow_pptx'])
        save(folder/'native_arrow_paint.json',paint_receipt)
        registered=extract(Path(settings['source_project']),item,paths['image_registry'],folder/'images')
        cached=[]
        if settings.get('reuse_assets_from'):
            cached_marker=Path(settings['reuse_assets_from'])/'cases'/cid/'completed.json'
            previous=read(cached_marker) if cached_marker.is_file() else None
            # A changed branch is a cache miss, not a composition failure.
            # The normal asset review below establishes the current scope.
            if previous:
                cached=previous['ownership']['texture_fallback']['assets']
                if previous['sources']['sha256']!=record['sha256']:
                    # Independent registered images do not depend on native
                    # shape geometry. Their source/crop/text/edit contracts and
                    # bytes are still checked by the current asset review.
                    cached=[a for a in cached if a.get('registry_ids') and not a.get('replace_parts')
                        and not a.get('edit',{}).get('separate_foreground_parts')
                        and not a.get('edit',{}).get('preserved_native_frames')]
        texture=choose_texture(scene,styles,arrows['arrows'],texts['texts'],item['original'],paths['shape_preview'],folder/'texture_fallback',runtime,
                               images=registered,editor=editor,cached_assets=cached,defer_cleanup=True,preserve_icon_pixels=True)
        save(folder/'texture_fallback/proposed_ownership.json',texture)
        active=surviving_scene(scene,texture)
        texture['proposed_drop_arrow_names']=texture.get('drop_arrow_names',[])
        texture['drop_arrow_names']=[]
        kept_arrows=arrows['arrows']
        quality=choose_native(active,styles,kept_arrows,item['original'],paths['shape_pptx'],paths['arrow_pptx'],folder/'ownership_quality',runtime,
                              assets=texture.get('assets', ()),texts=texts['texts'])
        # Only image pixels consume this early plan. The final plan below
        # always reconciles the actual assets, shapes, arrows and text.
        plan=(reconcile(scene,style_data,arrows,texts,item['original'],quality=quality,texture=texture)
              if texture['assets'] else {'drop_arrow_names':[]})
        from .asset_occlusion import reconcile as asset_occlusion
        image_order,image_audit=asset_occlusion(texture['assets'],arrows['arrows'],item['original'],plan['drop_arrow_names'],texts['texts'])
        plan['image_source_occlusion_order']=image_order
        from .image_realization import realize
        realize(texture,scene,styles,item['original'],plan,editor)
        if texture.get('realization_failures'):
            save(folder/'ownership_quality/before_image_fallback_audit.json',quality)
            quality=choose_native(surviving_scene(scene,texture),styles,arrows['arrows'],item['original'],
                paths['shape_pptx'],paths['arrow_pptx'],folder/'ownership_quality_after_image_fallback',runtime,
                assets=texture.get('assets',()),texts=texts['texts'])
            quality['source_ledger_root']='ownership_quality_after_image_fallback'
            quality['image_realization_failures']=texture['realization_failures']
            save(folder/'ownership_quality/audit.json',quality)
        # Final ownership consumes only realized images. This is deterministic
        # bookkeeping, not another model choice; failed assets retain native
        # content, successful ones retire exactly their reviewed replacements.
        plan=reconcile(scene,style_data,arrows,texts,item['original'],quality=quality,texture=texture)
        image_order,image_audit=asset_occlusion(texture['assets'],arrows['arrows'],item['original'],plan['drop_arrow_names'],texts['texts'])
        for ident,spec in image_order.items():
            target=plan['arrows_between_shape_parts'].setdefault(ident,{'before':[],'after':[]})
            for side,refs in spec.items():target[side]=sorted(set(target.get(side,[]))|set(refs))
        plan['image_source_occlusion_audit']=image_audit
        save(folder/'texture_fallback/audit.json',texture)
        save(folder/'ownership.json',plan)
        shape_pptx=paths['shape_pptx']
        if texture.get('native_patterns'):
            from .texture_fallback import apply_patterns
            shape_pptx=apply_patterns(shape_pptx,folder/'native_patterns.pptx',texture['native_patterns'])
        placed=commit_assets(shape_pptx,folder/'asset_shapes.pptx',texture['assets'],item['source_size'])
        shape_pptx=folder/'asset_shapes.pptx'
        first=merge(shape_pptx,[('arrows',paths['arrow_pptx'])],folder/'vectors.pptx',
                    exclude_base_names=plan['drop_shape_names'],suppress_base_segments=plan['shape_segment_suppression'],
                    suppress_base_subpaths=plan['shape_subpath_suppression'],
                    exclude_names={'arrows':plan['drop_arrow_names']},between_objects={'arrows':plan['arrows_between_shape_parts']},
                    suppress_round_terminals={'arrows':plan.get('arrow_round_terminal_suppression',{})})
        notify(cid,stage='images')
        shutil.copy2(folder/'vectors.pptx',folder/'images.pptx')
        save(folder/'images/validation.json',{'assets':texture['assets'],'placement':placed})
        notify(cid,stage='text_paint')
        selected=folder/'selected_text'
        command(cid,'text_paint',['--backend',run/'backends/text','--paint-on-base',folder/'images.pptx','--text-scene',paths['text_scene'],
            '--source-image',item['original'],'--shape-scene',paths['shape_scene'],'--shape-styles',paths['shape_styles'],
            '--arrow-scene',paths['arrow_scene'],'--ownership-plan',folder/'ownership.json','--output',selected,
            *reuse_arguments(settings,run)])
        second=merge(folder/'images.pptx',[('text',selected/'text.pptx')],folder/'combined.pptx')
        notify(cid,stage='rendering')
        command(cid,'render',['--backend',run/'backends/text','--render-pptx',folder/'combined.pptx','--output',folder/'native','--size',*item['source_size']])
        # Whatever no branch recovered is kept as a picture of the source.
        residual=[]
        if settings.get('source_residual_pictures',True):
            from .residual import recover_missing
            try:residual=recover_missing(folder/'combined.pptx',Path(item['original']),folder/'native/native.png',folder/'residual')
            except (OSError,ValueError,KeyError) as exc:save(folder/'residual_failure.json',{'error':str(exc)})
        if residual:
            notify(cid,stage='rendering_residual')
            try:
                command(cid,'render_residual',['--backend',run/'backends/text','--render-pptx',folder/'combined.pptx','--output',folder/'native_residual','--size',*item['source_size']])
                shutil.copy2(folder/'native_residual/native.png',folder/'native/native.png')
            except (RuntimeError,OSError) as exc:
                # The deck already holds the pictures; only its preview is one render behind.
                save(folder/'residual_failure.json',{'error':str(exc),'preview_shows_residual_pictures':False})
        with Image.open(item['original']) as im:original=im.convert('RGB')
        with Image.open(folder/'native/native.png') as im:restored=im.convert('RGB')
        w=min(1400,original.width);h=round(original.height*w/original.width);panel=Image.new('RGB',(w*2,h+28),'white');d=ImageDraw.Draw(panel)
        panel.paste(original.resize((w,h)),(0,28));panel.paste(restored.resize((w,h)),(w,28));d.text((8,8),cid+' SOURCE',fill='black');d.text((w+8,8),'RECOVERED PPT',fill='black');panel.save(folder/'comparison.png')
        for key,p in branch_paths.items():
            if sha(p)!=record['sha256'][key]:raise ValueError('Input product modified during composition')
        done={'id':cid,'composition_protocol':COMPOSITION_PROTOCOL,'pptx_sha256':sha(folder/'combined.pptx'),
              'sources':record,'stage_sources_unchanged':True,'native_rendered':True,'visual_accuracy_verified':False,
              'shape_geometry_refitted':False,'shape_type_reclassified':False,'sam_requests':0,
              'vectors':first,'text_composition':second,'ownership':plan,'text_paint':read(selected/'composition.json'),'at':time.time()}
        done['composition_ownership_complete']=bool(quality.get('line_ownership_complete'))
        done['source_residual_pictures']=[list(box) for box in residual]
        dest=run/'delivery'/cid;dest.mkdir(exist_ok=True)
        for source,name in [(folder/'combined.pptx','combined.pptx'),(folder/'native/native.png','preview.png'),(folder/'comparison.png','comparison.png')]:shutil.copy2(source,dest/name)
        save(marker,done);save(dest/'validation.json',done);notify(cid,state='completed',stage='completed')
        print('COMPOSED',cid,flush=True);return done
    completed={};errors={}
    def tracked_one(record):
        # Streaming can wait a long time for the next source product. Publish
        # a failed case as soon as its own work finishes, independently of it.
        try:return one(record)
        except Exception as exc:
            cid=record['item']['id']
            save(run/'cases'/cid/'failure.json',{'error':str(exc),'traceback':traceback.format_exc()})
            notify(cid,state='failed',error=str(exc))
            print('COMPOSITION_FAILED',cid,str(exc),flush=True)
            raise
    for record in records:notify(record['item']['id'],state='pending')
    with ThreadPoolExecutor(max_workers=settings['case_workers']) as pool:
        futures={}
        for record in record_stream if record_stream is not None else list(records):
            if record_stream is not None:
                records.append(record);notify(record['item']['id'],state='pending')
            futures[pool.submit(tracked_one,record)]=record['item']['id']
        for future in as_completed(futures):
            cid=futures[future]
            try:completed[cid]=future.result()
            except Exception as exc:
                errors[cid]=str(exc)
    if record_stream is not None:
        sources=read(run/'sources.json')
        order={r['id']:i for i,r in enumerate(read(run/'source_manifest.json')['images'])}
        records.sort(key=lambda r:order[r['item']['id']])
    document=fitz.open()
    with ZipFile(run/'delivery/editable_pptx.zip','w',ZIP_DEFLATED) as archive:
        for record in records:
            cid=record['item']['id'];folder=run/'delivery'/cid
            if cid not in completed:continue
            archive.write(folder/'combined.pptx',cid+'.pptx')
            with Image.open(folder/'comparison.png') as im:w,h=im.size
            page=document.new_page(width=1440,height=1440*h/w);page.insert_image(page.rect,filename=str(folder/'comparison.png'))
    if len(document):document.save(run/'delivery/comparison.pdf',garbage=4,deflate=True)
    document.close()
    unresolved=[cid for cid,value in completed.items() if not value.get('composition_ownership_complete',
                  value.get('ownership',{}).get('quality_choice',{}).get('line_ownership_complete',True))]
    summary={'state':'completed_with_failures' if errors else 'completed_with_unresolved' if unresolved else 'completed',
             'unresolved_ownership_cases':unresolved,'expected':len(records),'requested':settings['requested'],
             'exported':len(completed),'errors':errors,'missing_sources':sources['missing'],'sam_requests':0,
             'stage_recovery_requests':0,'composition_only':True,'at':time.time()}
    save(run/'summary.json',summary);save(run/'delivery/validation.json',summary)
    if errors:raise RuntimeError('Composition failures retained for resume')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    p.add_argument('--shape-run',type=Path);p.add_argument('--text-run',type=Path);p.add_argument('--arrow-run',type=Path)
    p.add_argument('--text-source',type=Path,help='Current text implementation; defaults to the same source as run_ppt.py')
    p.add_argument('--runtime-source',type=Path,help='API endpoint/font/renderer environment; credentials use the current default or --api-key-file')
    p.add_argument('--api-key-file',type=Path,action='append',help='Protected credential source; repeat for multiple files')
    p.add_argument('--api-base-url',help='Explicit current provider endpoint; otherwise use the selected runtime')
    p.add_argument('--namespace',default='ailab-intern10')
    p.add_argument('--charged-group',default='intern10_proxy')
    p.add_argument('--worker-cpu',type=int,default=16)
    p.add_argument('--worker-memory-mib',type=int,default=65536)
    p.add_argument('--ids',nargs='+',help='Compose only the named source cases, preserving all other deliveries')
    p.add_argument('--case-workers',type=int,default=8);p.add_argument('--api-concurrency',type=int,default=24)
    p.add_argument('--resume-from',type=Path,help='Reuse only verified completed compositions read-only')
    p.add_argument('--require-all',action='store_true',help='Refuse preparation while any requested branch product is incomplete')
    p.add_argument('--image-edit-url',help='Optional SenseNova JSON /v1 image edit endpoint; disabled by default')
    p.add_argument('--image-edit-model',default='SenseNova-U1.5-8B-MoT')
    p.add_argument('--image-edit-concurrency',type=int,default=2)
    p.add_argument('--reuse-assets-from',type=Path,help='Reuse validated image assets only with identical branch inputs and unchanged target/role/replacement scope')
    args=p.parse_args()
    if args.prepare:
        if not all([args.shape_run,args.text_run,args.arrow_run]):p.error('Prepare needs all three existing product paths')
        prepare(Path(__file__).resolve().parents[1],args)
    else:execute(args.run_dir.resolve())


if __name__=='__main__':main()
