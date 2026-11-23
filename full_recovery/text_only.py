"""Fresh original-image text inference with the shared CPU worker and renderer."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from .prepare import read,save,sha,freeze_text


def prepare(project,run,runtime_source,text_source):
    if run.exists():raise ValueError('Use a new text run')
    initial=read(runtime_source/'settings.json')
    items=read(runtime_source/'source_manifest.json')['images']
    import run_shapes
    run_shapes.prepare(SimpleNamespace(run_dir=run/'shape',
        input_root=Path(items[0]['original']).parents[2],ids=[r['id'] for r in items],max_stroke_pt=2.5))
    items=read(run/'shape/source_manifest.json')['images']
    code=run/'code';code.mkdir()
    shutil.copytree(project/'full_recovery',code/'full_recovery',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(project/'scripts/lo_guard.py',code/'lo_guard.py')
    (code/'text_only.py').write_text('from full_recovery.text_only import main\nif __name__ == "__main__": main()\n')
    for name in ['renderer_runtime','backends/reference_fonts']:
        shutil.copytree(runtime_source/name,run/name,symlinks=True)
    freeze_text(text_source,run/'backends/text','http://127.0.0.1:18188/v1')
    settings={**{k:v for k,v in initial.items() if k.startswith(('api_','text_')) or k=='source_project'},
        'workflow_scope':'text','api_global_concurrency':8,
        'api_gateway_port':18188,'glm_url':'http://127.0.0.1:18188/v1','text_workers':2,
        'worker_max_seconds':86400,'fresh_text_only':True}
    for key in ['api_relays','reuse_request_roots','reuse_assets_from','external_arrow_inputs',
                'shape_work_root','shape_base_run','arrow_driver']:
        settings.pop(key,None)
    for name in ['logs','jobs','render_locks','texts/cases','tests_tmp']: (run/name).mkdir(parents=True,exist_ok=True)
    save(run/'settings.json',settings)
    save(run/'source_manifest.json',{'images':items})
    save(run/'experiment.json',{'scope':'Fresh text from original PNGs only',
        'source_prediction_products':[],'shape_inference_enabled':False,'arrow_inference_enabled':False,
        'sam_requests':0,'runtime_source':str(runtime_source),'visual_accuracy_verified':False,
        'reason':'Reference text stages with bounded local recovery; no extra model stages'})
    files=[p for base in [code,run/'backends/text',run/'renderer_runtime',run/'backends/reference_fonts']
           for p in base.rglob('*') if p.is_file() and p.suffix not in ('.pyc',)]
    files.extend([run/'settings.json',run/'source_manifest.json',run/'experiment.json'])
    save(run/'code_manifest.json',{str(p.relative_to(run)):sha(p) for p in files})


def execute(run):
    if os.environ.get('STYLE_CLUSTER_WORKER')!='1':raise RuntimeError('Use the CPU rjob worker')
    owner=(run/'text.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    settings=read(run/'settings.json');items=read(run/'source_manifest.json')['images']
    env={**os.environ,'PPT_LOCAL_GLM':'1','BOYUE_API_KEY':'local-glm-without-auth'}
    children=[];mutex=threading.Lock();progress={}
    def stamp(cid,**value):
        with mutex:
            progress[cid]={**value,'at':time.time()}
            save(run/'progress.json',{'completed':sum(v.get('state')=='completed' for v in progress.values()),
                'failed':sum(v.get('state')=='failed' for v in progress.values()),'expected':len(items),'cases':progress})
    def stop(*args):
        for process in children:
            if process.poll() is None:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        if args:raise SystemExit(143)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def one(item):
        cid=item['id'];output=run/'texts/cases'/cid
        if sha(item['original'])!=item['source_sha256']:raise ValueError('Original changed: '+cid)
        stamp(cid,state='running')
        argv=[sys.executable,'-B','-u',run/'code/full_recovery/text_stage.py',
            '--backend',run/'backends/text','--image',item['original'],'--output',output,
            '--ocr-python',settings['text_ocr_python'],'--models',settings['text_models'],
            '--recognition-mode',settings['text_recognition_mode'],'--ocr-mode',settings['text_ocr_mode'],
            '--geometry-backend',settings.get('text_geometry_backend','glm'),
            '--ocr-template-validation',settings['text_ocr_template_validation'],
            '--threshold',str(settings['text_threshold']),'--max-score-drop',str(settings['text_max_score_drop']),
            '--ocr-max-side',str(settings['text_ocr_max_side'])]
        if settings['text_crop_refine']:argv.append('--glm-crop-refine')
        with (run/'logs'/('fresh_text_'+cid+'.log')).open('a') as log:
            process=subprocess.Popen(list(map(str,argv)),cwd=run,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            with mutex:children.append(process)
            process.wait()
        if process.returncode:raise RuntimeError('Fresh text failed: '+cid)
        from .text_product_validation import verify
        save(output/'completed.json',verify(item,output))
        stamp(cid,state='completed');print('TEXT_COMPLETED',cid,flush=True)
    errors={}
    try:
        with ThreadPoolExecutor(max_workers=settings['text_workers']) as pool:
            pending={pool.submit(one,item):item['id'] for item in items}
            for task in as_completed(pending):
                try:task.result()
                except Exception as error:
                    cid=pending[task];errors[cid]=str(error);stamp(cid,state='failed',error=str(error))
        save(run/'texts/manifest.json',{'cases':[{'id':i['id'],'sha256':i['source_sha256']} for i in items]})
        save(run/'summary.json',{'exported':len(items)-len(errors),'expected':len(items),'errors':errors,
            'fresh_original_inference':True,'sam_requests':0,'arrow_inference_enabled':False})
        if errors:raise RuntimeError('Some fresh text cases failed')
    finally:stop()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--runtime-source',type=Path);parser.add_argument('--text-source',type=Path)
    args=parser.parse_args();run=args.run_dir.resolve()
    if args.prepare:
        if not args.runtime_source or not args.text_source:parser.error('Preparation needs runtime and original text source')
        prepare(Path(__file__).resolve().parents[1],run,args.runtime_source.resolve(),args.text_source.resolve())
    else:execute(run)


if __name__=='__main__':main()
