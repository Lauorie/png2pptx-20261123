"""New PNG recognition in all branches, followed by the shared compositor."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

from .prepare import read,save,sha


def archive_unpublished_requests(output):
    """Interrupted attempts without an identity/result are not reusable evidence."""
    output=Path(output)
    moved=[]
    for path in list(output.rglob('request.json')):
        folder=path.parent
        if 'interrupted_requests' in folder.parts or folder.parent.name!='calls':continue
        if (folder/'identity.json').is_file() or (folder/'result.json').is_file():continue
        # Never quarantine complete source responses: those have an explicit
        # revalidation path even when termination interrupted identity writing.
        status=read(folder/'stream_status.json') if (folder/'stream_status.json').is_file() else {}
        if (status.get('finish_reason')=='stop' and status.get('sse_done')
                and (folder/'content.txt').is_file()):continue
        destination=output/'interrupted_requests'/str(time.time_ns())/folder.relative_to(output)
        destination.parent.mkdir(parents=True,exist_ok=True)
        folder.rename(destination)
        moved.append({'source':str(folder),'archive':str(destination),
                      'reason':'Interrupted attempt has no published identity or validated result'})
    if moved:save(output/'interrupted_request_recovery.json',{'requests':moved,'complete_results_unchanged':True})
    return moved


def validate_sources(run,records,external_arrows=None,*,verify_external=True):
    """Every inference product must belong to this new run, never a prior run."""
    run=Path(run).resolve()
    external = {}
    if external_arrows is not None:
        from .external_arrow_products import verify
        if verify_external:verify(external_arrows)
        external = {record['id']: record for record in external_arrows['records']}
    for record in records:
        for name,value in record['paths'].items():
            path=Path(value).resolve()
            if name in ('arrow_scene','arrow_pptx') and external_arrows is not None:
                expected = external.get(record['item']['id'])
                if (expected is None or expected['source_sha256'] != record['item']['source_sha256']
                        or path != Path(expected['paths'][name]).resolve()
                        or record['sha256'][name] != expected['sha256'][name]):
                    raise ValueError('Arrow product is not the explicitly bound input: '+str(path))
                continue
            if not path.is_relative_to(run):
                raise ValueError('Historical branch product forbidden: '+name+' '+str(path))
            if sha(path)!=record['sha256'][name]:raise ValueError('Fresh product changed: '+str(path))


def validate_arrow_completion(root, ids):
    """A zero controller exit code does not establish case completion."""
    root = Path(root)
    progress = read(root/'progress.json') if (root/'progress.json').is_file() else {}
    failed = [r['case'] for r in progress.get('cases', []) if r.get('state') == 'failed']
    if progress.get('failed', 0) or failed:
        raise RuntimeError('Arrow controller exited with failed cases: '+str(failed or progress['failed']))
    missing = []
    for ident in ids:
        folder = root/'geometry/cases'/ident
        absent = [name for name in ('completed.json', 'scene.json', 'arrows.pptx')
                  if not (folder/name).is_file() or (folder/name).stat().st_size == 0]
        if absent:
            missing.append({'id': ident, 'missing': absent})
    if missing:
        raise RuntimeError('Arrow controller did not produce complete fresh case products: '+str(missing))


def arrow_case_products_ready(root, ident):
    """One case may compose while unrelated arrow cases are still running."""
    folder=Path(root)/'geometry/cases'/ident
    return all((folder/name).is_file() and (folder/name).stat().st_size>0
               for name in ('completed.json','scene.json','arrows.pptx'))


def execute(run, *, code_root=None):
    if os.environ.get('STYLE_CLUSTER_WORKER')!='1':raise RuntimeError('Use the CPU rjob launcher')
    run=Path(run).resolve();settings=read(run/'settings.json')
    if settings.get('reuse_request_roots') or settings.get('reuse_assets_from'):
        raise ValueError('Fresh workflow forbids historical caches')
    lock=(run/'fresh_branches.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    items=read(run/'source_manifest.json')['images'];code=Path(code_root) if code_root else run/'code';children=[]
    external_arrows = read(Path(settings['external_arrow_inputs'])) if settings.get('arrow_driver')=='existing_products' else None
    pending_arrows=external_arrows.get('pending',[]) if external_arrows is not None else []
    pending_ids={row['id'] for row in pending_arrows}
    composition_ids=[item['id'] for item in items if item['id'] not in pending_ids]
    if external_arrows is not None:
        from .external_arrow_products import verify
        verify(external_arrows)
    cancel=threading.Event();mutex=threading.Lock();progress={}
    env={**os.environ,'PPT_LOCAL_GLM':'1','PYTHONDONTWRITEBYTECODE':'1',
         'ARROW_API_KEY':'local-glm-without-auth','BOYUE_API_KEY':'local-glm-without-auth',
         'PYTHONPATH':str(Path(settings['source_project'])/'.runtime/shape_layers_py311')}
    def stamp(name,state,**details):
        with mutex:
            progress[name]={'state':state,**details,'at':time.time()}
            save(run/'fresh_stage_status.json',{'branches':progress,'at':time.time()})
    def stop(*args):
        cancel.set()
        with mutex:
            for child in children:
                if child.poll() is None:
                    try:os.killpg(child.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
        if args:raise SystemExit(143)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def command(name,argv,environment=env,completion_check=None):
        stamp(name,'running')
        path=run/'logs'/(name+'.log')
        with path.open('a',buffering=1) as log:
            child=subprocess.Popen([sys.executable,'-B','-u',*map(str,argv)],cwd=run,
                                   env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            with mutex:children.append(child)
            while child.poll() is None:
                if cancel.wait(1):
                    child.wait(timeout=30);raise RuntimeError('Fresh workflow cancelled')
            if child.returncode:
                stamp(name,'failed',exit_code=child.returncode)
                raise RuntimeError(name+' failed; see '+str(path))
        if completion_check is not None:
            try:completion_check()
            except Exception as exc:
                stamp(name,'failed',error=str(exc),exit_code=child.returncode)
                raise
        stamp(name,'completed')
    def text_worker(index):
        for item in items[index::settings['text_workers']]:
            output=run/'texts/cases'/item['id']
            if (output/'summary.json').is_file():
                from .text_product_validation import verify
                verify(item, output)
                stamp('fresh_text_'+item['id'],'completed',checkpoint_verified=True)
                continue
            argv=[code/'full_recovery/text_stage.py','--backend',run/'backends/text','--image',item['original'],
                  '--output',output,'--ocr-python',settings['text_ocr_python'],'--models',settings['text_models'],
                  '--recognition-mode',settings['text_recognition_mode'],'--ocr-mode',settings['text_ocr_mode'],
                  '--geometry-backend',settings.get('text_geometry_backend','glm'),
                  '--ocr-template-validation',settings['text_ocr_template_validation'],
                  '--threshold',settings['text_threshold'],'--max-score-drop',settings['text_max_score_drop'],
                  '--ocr-max-side',settings['text_ocr_max_side']]
            if settings['text_crop_refine']:argv.append('--glm-crop-refine')
            if output.exists():
                archive_unpublished_requests(output)
                argv.append('--resume')
            command('fresh_text_'+item['id'],argv)
    def arrow_batch():
        if settings.get('arrow_driver')=='frozen_geometry_first':
            argv=[Path(settings['fresh_arrow_root'])/'code/run_full125.py',
                  '--run-dir',settings['fresh_arrow_root']]
        else:
            argv=[code/'full_recovery/arrow_stage.py','--backend',run/'backends/arrow',
                  '--run',run/'arrows','--config',run/'arrows/config.json','--frozen',
                  '--cases',*[item['id'] for item in items]]
        check = (lambda: validate_arrow_completion(settings['fresh_arrow_root'], [i['id'] for i in items])) if settings.get('arrow_driver')=='frozen_geometry_first' else None
        command('fresh_arrows',argv,completion_check=check)
    try:
        if external_arrows is not None:
            stamp('existing_arrow_products','validated',root=external_arrows['root'],inference_enabled=False)
        from .merge_existing import inspect_sources,execute as compose
        save(run/'texts/manifest.json',{'cases':[{'id':i['id'],'sha256':i['source_sha256']} for i in items]})
        source_records=[]
        source_document={'ready':source_records,'missing':pending_arrows,
            'historical_products_reused':external_arrows is not None,
            'shape_geometry_recovery':True,'arrow_recovery':external_arrows is None,'text_recognition':True,
            'arrow_reuse_policy':'explicit_user_selected_run' if external_arrows else None}
        save(run/'sources.json',source_document)
        def products():
            pending=set(composition_ids)
            while pending:
                if cancel.is_set():raise RuntimeError('Fresh product stream cancelled')
                emitted=False
                for item in items:
                    cid=item['id']
                    if cid not in pending:continue
                    with mutex:
                        text_ready=progress.get('fresh_text_'+cid,{}).get('state')=='completed'
                        arrow_ready=external_arrows is not None or progress.get('fresh_arrows',{}).get('state')=='completed'
                    if not arrow_ready and settings.get('arrow_driver')=='frozen_geometry_first':
                        arrow_ready=arrow_case_products_ready(settings['fresh_arrow_root'],cid)
                    if not text_ready or not arrow_ready or not (run/'shape/geometry/cases'/cid/'complete.json').is_file():continue
                    ready,missing=inspect_sources(run,run/'texts',Path(settings['fresh_arrow_root']),[cid])
                    if missing:continue
                    # The complete external arrow binding was verified before
                    # starting. Check this fresh case and its exact bound rows;
                    # do not replay a 125-case verification for each emission.
                    validate_sources(run,ready,external_arrows,verify_external=False)
                    source_records.extend(ready);save(run/'sources.json',source_document)
                    pending.remove(cid);emitted=True
                    yield ready[0]
                if not emitted:cancel.wait(1)
        with ThreadPoolExecutor(max_workers=(2 if external_arrows is not None else 3)+settings['text_workers']) as pool:
            futures=[pool.submit(command,'fresh_shape',[run/'shape/code/run_shapes.py','--run-dir',run/'shape'],
                        {**env,'SHAPE_REFERENCE_ROOT':str(run/'shape/code'),'GLM_API_RELAY_URL':settings['glm_url']})]
            if external_arrows is None:futures.append(pool.submit(arrow_batch))
            futures.extend(pool.submit(text_worker,i) for i in range(settings['text_workers']))
            futures.append(pool.submit(compose,run,record_stream=products()))
            for future in as_completed(futures):
                try:future.result()
                except BaseException:stop();raise
        ready,missing=inspect_sources(run,run/'texts',Path(settings['fresh_arrow_root']),composition_ids)
        if missing:raise RuntimeError('Fresh branches incomplete: '+str(missing))
        validate_sources(run,ready,external_arrows)
        save(run/'sources.json',{'ready':ready,'missing':pending_arrows, 'historical_products_reused':external_arrows is not None,
             'shape_geometry_recovery':True,'arrow_recovery':external_arrows is None,'text_recognition':True,
             'arrow_reuse_policy':'explicit_user_selected_run' if external_arrows else None})
        stamp('fresh_product_audit','completed',cases=len(ready),historical_products_reused=external_arrows is not None)
        summary=read(run/'summary.json')
        summary['composition_sam_requests']=summary.pop('sam_requests',0)
        metrics=[read(p) for p in (run/'api').glob('*/api_metrics.json')]
        if (run/'api_metrics.json').is_file():metrics.append(read(run/'api_metrics.json'))
        summary.update(composition_only=False,stage_recovery_requests=sum(r.get('requests',0) for r in metrics),
                       historical_products_reused=external_arrows is not None,branches_fresh=external_arrows is None,
                       shape_text_branches_fresh=True,arrow_inference_enabled=external_arrows is None)
        if pending_arrows:
            summary.update(state='completed_with_pending_arrow_inputs',missing_sources=pending_arrows,
                           shape_text_requested=len(items),pending_arrow_ids=sorted(pending_ids))
        save(run/'summary.json',summary);save(run/'delivery/validation.json',summary)
    except BaseException as exc:
        save(run/'fresh_failure.json',{'error':str(exc),'traceback':traceback.format_exc(),'at':time.time()})
        raise
