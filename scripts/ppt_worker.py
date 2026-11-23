"""Own local GLM/SAM and the isolated full-recovery schedulers in one rjob."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time


def read(path):return json.loads(Path(path).read_text())
def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);run=p.parse_args().run_dir.resolve()
    owner=(run/'worker.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    settings=read(run/'settings.json');project=Path(settings['source_project']);services=run/'services'
    code=run/'code';shape=run/'shape/code';started=time.time();children=[];handles=[];phase='preflight'
    env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy')}
    env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    sampler_stop=threading.Event()
    def stamp(name,**extra):
        nonlocal phase
        phase=name;save(run/'status.json',{'state':name,'host':socket.gethostname(),'pid':os.getpid(),
             'at':time.time(),'elapsed_seconds':time.time()-started,**extra});print(name,extra,flush=True)
    def launch(name,argv,environment):
        handle=(run/(name+'.log')).open('a',buffering=1);handles.append(handle)
        proc=subprocess.Popen(list(map(str,argv)),env=environment,cwd=code,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(proc);return proc
    def stop(*signals):
        sampler_stop.set()
        save(services/'STOP.json',{'at':time.time(),'reason':'full_recovery_'+phase})
        for process in reversed(children):
            if process.poll() is None:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for process in children:
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait()
        if signals:raise SystemExit(143)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def sample_usage():
        while not sampler_stop.is_set():
            try:
                measured=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
                snapshot={'at':time.time(),'stage':phase,'columns':['index','utilization_percent','memory_mib','total_memory_mib']}
                if measured.returncode==0:snapshot['values']=measured.stdout.strip()
                else:snapshot['query_error']='nvidia-smi exit '+str(measured.returncode)
                pressure=Path('/proc/pressure/io')
                if pressure.exists():snapshot['io_pressure']=pressure.read_text().strip()
                save(run/'gpu_usage.json',snapshot)
                with (run/'gpu_usage.jsonl').open('a') as record:record.write(json.dumps(snapshot)+'\n')
            except (OSError,subprocess.TimeoutExpired):
                pass
            sampler_stop.wait(10)
    sampler=threading.Thread(target=sample_usage,name='gpu-metrics',daemon=True);sampler.start()
    try:
        stamp('verifying_frozen_inputs_and_old_sam')
        for relative,digest in read(run/'code_manifest.json').items():
            if hashlib.sha256((run/relative).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen code changed: '+relative)
        for relative,digest in read(run/'shape/code_sha256.json').items():
            if hashlib.sha256((shape/relative).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen shape code changed: '+relative)
        checkpoint=Path(settings['sam_checkpoint']);h=hashlib.sha256()
        with checkpoint.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
        if h.hexdigest()!=settings['sam_checkpoint_sha256'] or read(checkpoint.with_suffix('.json'))['epoch']!=18:
            raise ValueError('Old shape SAM identity changed')
        save(run/'sam_weight_validation.json',{'path':str(checkpoint),'sha256':h.hexdigest(),'epoch':18})
        for model in (() if settings.get('text_geometry_backend')=='glm' else ('PP-OCRv5_server_det','PP-OCRv5_server_rec')):
            path=Path(settings['text_models'])/model
            if not path.is_dir():raise ValueError('Existing OCR model is not mounted: '+str(path))
        sys.path.insert(0,str(shape/'scripts'))
        from shape_worker import renderer
        native=renderer(project,shape);portable=Path('/tmp/shape-recovery-lo')
        from full_recovery.fonts import configure
        save(run/'font_validation.json',configure(run/'backends/reference_fonts',portable))
        raw=portable/'bin/libreoffice-raw';shutil.copy2(portable/'bin/libreoffice',raw);raw.chmod(0o755)
        guarded=portable/'bin/libreoffice';shutil.copy2(code/'lo_guard.py',guarded);guarded.chmod(0o755)
        # Equations also call the absolute system path; use the same renderer
        # and cross-process limit for those disposable measurement documents.
        system=Path('/usr/bin/libreoffice')
        if system.exists() or system.is_symlink():system.unlink()
        system.symlink_to(guarded)
        cpu={**env,**native,'STYLE_CLUSTER_WORKER':'1','PPT_LOCAL_GLM':'1',
             'SHAPE_REFERENCE_ROOT':str(shape),
             'PPT_RENDER_LOCK_DIR':str(run/'render_locks'),'PPT_LIBREOFFICE_RAW':str(raw),
             'FONTCONFIG_FILE':str(portable/'fonts.conf'),
             'PYTHONPATH':str(project/'.runtime/shape_layers_py311'),
             'LD_LIBRARY_PATH':'/opt/shape-cpu/lib:'+env.get('LD_LIBRARY_PATH','')}
        # Fontconfig is present in the preserved CPU environment.
        cpu['PATH']=str(portable/'bin')+':/opt/shape-cpu/bin:'+env['PATH']
        stamp('shape_regression_tests')
        tests=launch('shape_tests',['/opt/shape-cpu/bin/python','-B','-m','unittest','discover','-s',shape/'tests','-p','test_*.py'],cpu)
        tests.wait()
        if tests.returncode:raise RuntimeError('Shape regression failed; see shape_tests.log')
        stamp('text_arrow_and_composition_regression_tests')
        tests=launch('text_tests',['/opt/shape-cpu/bin/python','-B','-u',code/'full_recovery/text_stage.py','--backend',run/'backends/text','--test'],cpu)
        tests.wait()
        if tests.returncode:raise RuntimeError('Text regression failed; see text_tests.log')
        tests=launch('arrow_tests',['/opt/shape-cpu/bin/python','-B',code/'full_recovery/arrow_stage.py','--backend',run/'backends/arrow','--test'],cpu)
        tests.wait()
        if tests.returncode:raise RuntimeError('Arrow regression failed; see arrow_tests.log')
        tests=launch('composition_tests',['/opt/shape-cpu/bin/python','-B','-m','unittest','discover','-s',code,'-p','test_full_composition.py'],cpu)
        tests.wait()
        if tests.returncode:raise RuntimeError('Composition regression failed; see composition_tests.log')
        save(run/'tests_passed.json',{'at':time.time(),'shape':True,'text':True,'arrows':True,'composition':True})
        service_code=services/'service_code';service_code.mkdir(exist_ok=True)
        for path in (shape/'experiment/style_line_sam/service').glob('*.py'):shutil.copy2(path,service_code/path.name)
        serving={**settings,'glm_only':False,'single_glm':True,'sam_first':False}
        import torch
        devices=[torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
        if len(devices)!=8 or any(p.major<9 for p in devices):raise ValueError('This serving configuration requires eight Hopper-or-newer GPUs')
        if min(p.total_memory for p in devices)<120*1024**3:
            raise ValueError('Independent TP4 profiles require H200-class memory; use a 16-GPU distributed deployment on smaller GPUs')
        save(run/'gpu_topology.json',{'devices':[{'name':p.name,'memory_bytes':p.total_memory} for p in devices],
            'glm_profiles':serving['glm_profiles'],'sam_shared_devices':[4,5,6,7]})
        save(services/'settings.json',serving)
        (services/'STOP.json').unlink(missing_ok=True)
        stamp('loading_local_glm_and_shape_sam')
        server=launch('services',[sys.executable,'-u',service_code/'worker.py','--run-dir',services],env)
        arrow_run=services/'arrow_sam';arrow_run.mkdir(exist_ok=True)
        arrow_server=launch('arrow_sam',[settings['sam_python'],'-B','-u',run/'backends/arrow_sam/server.py',
            '--config',run/'backends/arrow_sam/config.json','--run-dir',arrow_run],
            {**env,'CUDA_VISIBLE_DEVICES':str(settings['arrow_sam_device'])})
        deadline=time.time()+4200
        while True:
            save(services/'activity.json',{'at':time.time(),'stage':'loading'})
            if server.poll() is not None:raise RuntimeError('Local model supervisor exited')
            if arrow_server.poll() is not None:raise RuntimeError('Independent arrow SAM exited')
            state=read(services/'status.json') if (services/'status.json').exists() else {};rows=state.get('services',{})
            arrow_state=read(arrow_run/'service-state.json') if (arrow_run/'service-state.json').exists() else {}
            if len(rows.get('glm',[]))==len(settings['glm_profiles']) and len(rows.get('sam',[]))==len(settings['sam_ports']) and all(v['alive'] and v['phase']=='ready' for v in rows['glm']+rows['sam']) and arrow_state.get('phase')=='ready':
                model=arrow_state['model']
                if model['checkpoint']['epoch']!=7 or model['checkpoint_sha256']!=read(run/'arrows/config.json')['sam_checkpoint_sha256']:
                    raise ValueError('Independent arrow SAM identity mismatch')
                save(run/'arrow_sam_validation.json',{'epoch':7,'model':model,'fresh_service':True})
                break
            if time.time()>deadline:raise TimeoutError('Model startup deadline')
            stamp('loading_local_glm_and_shape_sam',services=rows,arrow_sam=arrow_state.get('phase'));time.sleep(10)
        stamp('recovering_full_ppt')
        process=launch('pipeline',['/opt/shape-cpu/bin/python','-B','-u',code/'run_ppt.py','--run-dir',run],cpu)
        while process.poll() is None:
            if server.poll() is not None:raise RuntimeError('A local model service exited')
            if arrow_server.poll() is not None:raise RuntimeError('Independent arrow SAM exited during recovery')
            if time.time()-started>settings['worker_max_seconds']:raise TimeoutError('Full recovery lifetime exceeded; checkpoints retained')
            save(services/'activity.json',{'at':time.time(),'stage':'full_recovery'})
            progress=read(run/'progress.json') if (run/'progress.json').exists() else {}
            stamp('recovering_full_ppt',completed=progress.get('completed',0),failed=progress.get('failed',0),expected=progress.get('expected'))
            time.sleep(10)
        if process.returncode:raise RuntimeError('Full recovery exited; inspect pipeline.log and per-stage logs')
        stamp('completed',exported=read(run/'summary.json')['exported'])
    except BaseException as exc:
        stamp('failed',error=str(exc));raise
    finally:
        stop()
        sampler.join(timeout=12)
        for handle in handles:handle.close()


if __name__=='__main__':main()
