"""Run preserved per-stage schedulers and compose completed cases as they arrive."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from zipfile import ZipFile,ZIP_DEFLATED

from .compose import merge,sha,COMPOSITION_PROTOCOL
from .handoff import reconcile


def read(path):return json.loads(Path(path).read_text())
def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def execute(run):
    if os.environ.get('STYLE_CLUSTER_WORKER')!='1':raise RuntimeError('Use scripts/submit_ppt.sh')
    from PIL import Image
    import fitz
    code=run/'code';settings=read(run/'settings.json');items=read(run/'source_manifest.json')['images']
    editor=None
    if settings.get('image_edit',{}).get('enabled'):
        from .image_edit import Editor
        editor=Editor(settings['image_edit'])
    owner=(run/'pipeline.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    children=[];children_lock=threading.Lock();cancel=threading.Event();statuses={};state_lock=threading.RLock()
    main_env={**os.environ,'PPT_LOCAL_GLM':'1','PYTHONDONTWRITEBYTECODE':'1'}
    stage_env={**main_env,'PYTHONPATH':str(Path(settings['source_project'])/'.runtime/shape_layers_py311')}
    def command(name,argv,env=stage_env,affinity=None):
        logpath=run/'logs'/(name+'.log');logpath.parent.mkdir(exist_ok=True)
        with logpath.open('a',buffering=1) as log:
            cpu_set=set(affinity) if affinity else None
            child=subprocess.Popen([sys.executable,'-B','-u',*map(str,argv)],env=env,cwd=run,
                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                 preexec_fn=(lambda:os.sched_setaffinity(0,cpu_set)) if cpu_set else None)
            with children_lock:children.append(child)
            while child.poll() is None:
                if cancel.wait(2):
                    try:os.killpg(child.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    child.wait(timeout=30);raise RuntimeError('Workflow cancelled')
            if child.returncode:raise RuntimeError(name+' exited '+str(child.returncode)+'; see '+str(logpath))
    def stop(*signals):
        cancel.set()
        with children_lock:
            for child in children:
                if child.poll() is None:
                    try:os.killpg(child.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
        if signals:raise SystemExit(143)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def notify(ident,**row):
        with state_lock:
            statuses.setdefault(ident,{}).update(row)
            save(run/'progress.json',{'expected':len(items),'completed':sum(v.get('state')=='completed' for v in statuses.values()),
                 'failed':sum(v.get('state')=='failed' for v in statuses.values()),'cases':statuses,'at':time.time()})
    shape_env={**stage_env,'SHAPE_REFERENCE_ROOT':str(run/'shape/code')}
    def shape_batch():
        command('shape_batch',[run/'shape/code/run_shapes.py','--run-dir',run/'shape'],shape_env)
    def arrow_batch():
        remaining=[i['id'] for i in items if not arrow_complete(i)]
        if not remaining:return
        command('arrow_batch',[code/'full_recovery/arrow_stage.py','--backend',run/'backends/arrow',
             '--run',run/'arrows','--config',run/'arrows/config.json','--frozen','--cases',*remaining])
    def text_case(item,worker_index):
        ident=item['id'];out=run/'texts'/ident
        if (out/'summary.json').exists() and (out/'final/text.pptx').exists():
            if read(out/'final/provenance.json')['image_sha256']!=item['source_sha256']:raise ValueError('Text resume input changed')
            return
        notify(ident,text='running')
        argv=[code/'full_recovery/text_stage.py','--backend',run/'backends/text','--image',item['original'],
              '--output',out,'--ocr-python',settings['text_ocr_python'],'--models',settings['text_models'],
              '--recognition-mode',settings['text_recognition_mode'],'--ocr-mode',settings['text_ocr_mode'],
              '--geometry-backend',settings.get('text_geometry_backend','ocr'),
              '--ocr-template-validation',settings['text_ocr_template_validation'],
              '--threshold',str(settings['text_threshold']),'--max-score-drop',str(settings['text_max_score_drop']),
              '--ocr-max-side',str(settings['text_ocr_max_side'])]
        if settings['text_crop_refine']:argv.append('--glm-crop-refine')
        if out.exists():argv.append('--resume')
        available=sorted(os.sched_getaffinity(0));offset=worker_index*2
        cpu_set=available[offset:offset+2] or available[:2]
        command('text_'+ident,argv,affinity=cpu_set)
        notify(ident,text='completed')
    def text_worker(index):
        for item in items[index::settings['text_workers']]:
            if cancel.is_set():break
            try:text_case(item,index)
            except Exception as exc:
                save(run/'texts'/item['id']/'failure.json',{'error':str(exc),'traceback':traceback.format_exc()})
                notify(item['id'],text='failed',text_error=str(exc))
    def arrow_complete(item):
        folder=run/'arrows/cases'/item['id']
        if not (folder/'completed.json').exists():return None
        if read(folder/'source_manifest.json')['image_sha256']!=item['source_sha256']:
            raise ValueError('Arrow source hash mismatch: '+item['id'])
        if not (folder/'arrows.pptx').exists():raise ValueError('Arrow completion lacks PPTX')
        return folder
    def compose_case(item,arrow):
        ident=item['id'];folder=run/'cases'/ident;folder.mkdir(parents=True,exist_ok=True)
        marker=folder/'completed.json'
        if marker.exists():
            completed=read(marker)
            if completed.get('composition_protocol')!=COMPOSITION_PROTOCOL:
                raise ValueError('Composition policy changed; resume stages into a new run: '+ident)
            if sha(folder/'combined.pptx')!=completed['pptx_sha256']:raise ValueError('Composed PPT changed')
            notify(ident,state='completed',composition='completed',arrow='completed',shape='completed',text='completed')
            return completed
        if sha(item['original'])!=item['source_sha256']:raise ValueError('Original PNG changed')
        shape=run/'shape/geometry/cases'/ident/'styling/styled.pptx'
        text=run/'texts'/ident/'final/text.pptx'
        notify(ident,composition='running')
        shape_case=run/'shape/geometry/cases'/ident
        text_scene=run/'texts'/ident/'final/scene.json'
        sys.path[:0]=[str(run/'shape/code/src'),str(run/'shape/code/vendor')]
        from shape_integration.common import Runtime
        from .quality_choice import review as choose_native
        from .image_assets import review as choose_texture,commit as commit_assets
        from shape_integration.composition import extract
        scene=read(shape_case/'fitting/scene.json');style_data=read(shape_case/'styling/styles.json')
        styles={k:v['style'] for k,v in style_data['parts'].items()}
        arrow_scene=read(arrow/'scene.json');texts=read(text_scene)
        from .native_paint import bind as bind_native_paint
        arrow_scene,paint_receipt=bind_native_paint(arrow_scene,arrow/'arrows.pptx')
        save(folder/'native_arrow_paint.json',paint_receipt)
        config=read(run/'shape/pipeline_config.json')
        reviewer=Runtime({**config,'api_concurrency':4,'shape_schema_retries':1,
                         'reasoning_effort':'high','request_timeout_seconds':360},settings['glm_url'],'')
        from .handoff import surviving_scene
        registered=extract(Path(settings['project_root']),item,run/'shape/localization/cases'/ident/'dedup/image_boxes_original.json',folder/'images')
        texture=choose_texture(scene,styles,arrow_scene['arrows'],texts['texts'],item['original'],
                               shape_case/'styling/native.png',folder/'texture_fallback',reviewer,
                               images=registered,editor=editor)
        active_scene=surviving_scene(scene,texture)
        active_arrows=[r for r in arrow_scene['arrows'] if r['id'] not in texture.get('drop_arrow_names',[])]
        quality=choose_native(active_scene,styles,active_arrows,item['original'],shape,
                              arrow/'arrows.pptx',folder/'ownership_quality',reviewer,assets=texture.get('assets', ()))
        plan=reconcile(scene,style_data,arrow_scene,texts,item['original'],quality=quality,texture=texture)
        save(folder/'ownership.json',plan)
        if texture.get('native_patterns'):
            from .texture_fallback import apply_patterns
            shape=apply_patterns(shape,folder/'native_patterns.pptx',texture['native_patterns'])
        placed=commit_assets(shape,folder/'asset_shapes.pptx',texture['assets'],item['source_size'])
        shape=folder/'asset_shapes.pptx'
        first=merge(shape,[('arrows',arrow/'arrows.pptx')],folder/'vectors.pptx',
                    exclude_base_names=plan['drop_shape_names'],
                    suppress_base_segments=plan['shape_segment_suppression'],
                    suppress_base_subpaths=plan['shape_subpath_suppression'],
                    exclude_names={'arrows':plan['drop_arrow_names']},
                    between_objects={'arrows':plan['arrows_between_shape_parts']},
                    suppress_round_terminals={'arrows':plan.get('arrow_round_terminal_suppression',{})})
        shutil.copy2(folder/'vectors.pptx',folder/'images.pptx')
        save(folder/'images/validation.json',{'assets':texture['assets'],'placement':placed})
        selected_text=folder/'selected_text'
        command('text_paint_'+ident,[code/'full_recovery/text_stage.py','--backend',run/'backends/text',
                '--paint-on-base',folder/'images.pptx','--text-scene',text_scene,'--source-image',item['original'],
                '--shape-scene',shape_case/'fitting/scene.json','--shape-styles',shape_case/'styling/styles.json',
                '--arrow-scene',arrow/'scene.json',
                '--ownership-plan',folder/'ownership.json','--output',selected_text])
        second=merge(folder/'images.pptx',[('text',selected_text/'text.pptx')],folder/'combined.pptx')
        command('render_'+ident,[code/'full_recovery/text_stage.py','--backend',run/'backends/text',
                '--render-pptx',folder/'combined.pptx','--output',folder/'native','--size',*item['source_size']])
        # The text backend's preview uses its own OMML compatibility copy.
        # The delivered combined.pptx always retains editable Office Math.
        from PIL import ImageDraw
        with Image.open(item['original']) as opened:original=opened.convert('RGB')
        with Image.open(folder/'native/native.png') as opened:restored=opened.convert('RGB')
        width=min(1400,original.width);height=round(original.height*width/original.width)
        panel=Image.new('RGB',(2*width,height+30),'white');draw=ImageDraw.Draw(panel)
        panel.paste(original.resize((width,height)),(0,30));panel.paste(restored.resize((width,height)),(width,30))
        draw.text((8,8),ident+' Original',fill='black');draw.text((width+8,8),'Recovered PPT',fill='black')
        panel.save(folder/'comparison.png')
        dest=run/'delivery'/ident;dest.mkdir(exist_ok=True)
        for src,name in ((folder/'combined.pptx','combined.pptx'),(folder/'native/native.png','preview.png'),
                         (folder/'comparison.png','comparison.png')):shutil.copy2(src,dest/name)
        validation={'id':ident,'composition_protocol':COMPOSITION_PROTOCOL,
             'pptx_sha256':sha(folder/'combined.pptx'),'source_sha256':item['source_sha256'],
             'shape_sha256':sha(shape),'arrow_sha256':sha(arrow/'arrows.pptx'),'text_sha256':sha(text),
             'vectors':first,'text_composition':second,'image_composition':read(folder/'images/validation.json'),
             'arrow_source':str(arrow),'arrow_status':read(arrow/'completed.json')['state'],
             'text_quality':read(run/'texts'/ident/'summary.json').get('quality_status'),
             'shape_quality':read(run/'shape/geometry/cases'/ident/'styling/summary.json')['state'],
             'shape_geometry_quality':read(shape_case/'fitting/summary.json')['state'],
             'native_rendered':True,'visual_accuracy_verified':False,'at':time.time()}
        validation['composition_ownership']=plan
        save(marker,validation);save(dest/'validation.json',validation)
        notify(ident,state='completed',composition='completed',arrow='completed',shape='completed')
        return validation
    for item in items:notify(item['id'],state='pending')
    composed={};pending={};errors={};pool=ThreadPoolExecutor(max_workers=2)
    with ThreadPoolExecutor(max_workers=2+settings['text_workers']) as stages:
        shape_future=stages.submit(shape_batch);arrow_future=stages.submit(arrow_batch)
        text_futures=[stages.submit(text_worker,i) for i in range(settings['text_workers'])]
        try:
            while len(composed)+len(errors)<len(items):
                if cancel.is_set():raise RuntimeError('Cancelled')
                for item in items:
                    ident=item['id']
                    if ident in composed or ident in errors or ident in pending:continue
                    if (run/'shape/geometry/cases'/ident/'failure.json').exists():
                        errors[ident]='Shape stage failed';notify(ident,state='failed',error=errors[ident]);continue
                    if (run/'texts'/ident/'failure.json').exists():
                        errors[ident]='Text stage failed';notify(ident,state='failed',error=errors[ident]);continue
                    arrow=arrow_complete(item)
                    if arrow and (run/'shape/geometry/cases'/ident/'complete.json').exists() and (run/'texts'/ident/'summary.json').exists():
                        pending[ident]=pool.submit(compose_case,item,arrow)
                    elif arrow_future.done() and not arrow:
                        errors[ident]='Arrow stage ended without this case';notify(ident,state='failed',error=errors[ident])
                    elif shape_future.done() and not (run/'shape/geometry/cases'/ident/'complete.json').exists():
                        errors[ident]='Shape stage ended without this case';notify(ident,state='failed',error=errors[ident])
                for ident,future in list(pending.items()):
                    if not future.done():continue
                    try:composed[ident]=future.result()
                    except Exception as exc:
                        errors[ident]=str(exc);notify(ident,state='failed',error=str(exc))
                        save(run/'cases'/ident/'failure.json',{'error':str(exc),'traceback':traceback.format_exc()})
                    del pending[ident]
                if shape_future.done() and shape_future.exception():
                    # Per-case shape failures are already explicit; a global
                    # failure with no such markers must stop the integration.
                    if not any((run/'shape/geometry/cases'/i['id']/'failure.json').exists() for i in items):raise shape_future.exception()
                if arrow_future.done() and arrow_future.exception():raise arrow_future.exception()
                for future in text_futures:
                    if future.done() and future.exception():raise future.exception()
                save(run/'stage_status.json',{'shape_batch_done':shape_future.done(),'text_workers_done':sum(f.done() for f in text_futures),
                    'composed':len(composed),'failed':len(errors),'expected':len(items),'at':time.time()})
                if len(composed)+len(errors)<len(items):cancel.wait(10)
        except BaseException:
            stop()
            raise
        finally:
            pool.shutdown(wait=True)
    document=fitz.open()
    with ZipFile(run/'delivery/editable_pptx.zip','w',ZIP_DEFLATED) as archive:
        for item in items:
            folder=run/'delivery'/item['id']
            if not (folder/'combined.pptx').exists():continue
            archive.write(folder/'combined.pptx',item['id']+'.pptx')
            with Image.open(folder/'comparison.png') as image:w,h=image.size
            page=document.new_page(width=1440,height=1440*h/w)
            page.insert_image(page.rect,filename=str(folder/'comparison.png'))
    if len(document):document.save(run/'delivery/comparison.pdf')
    document.close()
    summary={'state':'completed_with_failures' if errors else 'completed','expected':len(items),'exported':len(composed),
         'errors':errors,'cases':composed,'originals_unchanged':all(sha(i['original'])==i['source_sha256'] for i in items),
         'visual_accuracy_verified':False,'at':time.time()}
    save(run/'summary.json',summary);save(run/'delivery/validation.json',summary)
    if errors:raise RuntimeError('Some cases failed; inspect explicit per-case records')
