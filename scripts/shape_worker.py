"""Run validation and the frozen workflow alongside persistent local GLM/SAM."""
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


def read(path): return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def renderer(project,code):
    source=project/'.runtime/portable_lo';local=Path('/tmp/shape-recovery-lo')
    shutil.copytree(source,local,dirs_exist_ok=True)
    for path in local.rglob('*'):
        if path.is_file() and (path.suffix in ('.rc','.rdb','.xml','.xcd','.xcu','.ini','.conf') or path.name.endswith('rc') or path.name=='libreoffice'):
            data=path.read_bytes()
            adapted=data.replace(b'{PNG2PPTX_PORTABLE_LO}',str(local).encode()).replace(str(source).encode(),str(local).encode())
            if adapted!=data:path.write_bytes(adapted)
    wrapper=local/'bin/libreoffice'
    shutil.copy2(code/'experiment/style_line_sam/lo_wrapper.py',wrapper);wrapper.chmod(0o755)
    font=local/'fonts/dejavu/DejaVuSans.ttf'
    system_font=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    if not system_font.exists():
        system_font.parent.mkdir(parents=True,exist_ok=True);system_font.symlink_to(font)
    return {'PATH':str(local/'bin')+':'+os.environ['PATH'],
            'XDG_DATA_DIRS':str(local)+':/usr/local/share:/usr/share','STYLE_FONT':str(font)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);run=p.parse_args().run_dir.resolve()
    lock=(run/'.worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    settings=read(run/'settings.json');code=run/'code';project=Path(settings['project_root']);services=run/'services'
    env={k:v for k,v in os.environ.items() if k.lower() not in {'http_proxy','https_proxy','all_proxy'}}
    env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    children=[];logs=[];started=time.time();phase='preflight'
    def stamp(name,**extra):
        nonlocal phase
        phase=name;save(run/'status.json',{'phase':name,'host':socket.gethostname(),'pid':os.getpid(),
             'at':time.time(),'elapsed_seconds':time.time()-started,**extra});print(name,extra,flush=True)
    def launch(name,argv,environment):
        log=(run/(name+'.log')).open('a',buffering=1);logs.append(log)
        child=subprocess.Popen(argv,env=environment,cwd=code,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(child);return child
    def stop(*signals):
        save(services/'STOP.json',{'at':time.time(),'reason':'shape_worker_'+phase})
        for child in reversed(children):
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for child in children:
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        if signals:raise SystemExit(143)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        stamp('verifying_old_sam')
        checkpoint=Path(settings['sam_checkpoint']);h=hashlib.sha256()
        with checkpoint.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
        if h.hexdigest()!=settings['sam_checkpoint_sha256'] or read(checkpoint.with_suffix('.json'))['epoch']!=18:
            raise ValueError('The selected old epoch18 SAM checkpoint changed')
        save(run/'sam_weight_validation.json',{'path':str(checkpoint),'sha256':h.hexdigest(),'epoch':18})
        for relative,expected in read(run/'code_sha256.json').items():
            if hashlib.sha256((code/relative).read_bytes()).hexdigest()!=expected:raise ValueError('Frozen code changed: '+relative)
        cpu={**env,**renderer(project,code),'STYLE_CLUSTER_WORKER':'1',
             'PYTHONPATH':str(project/'.runtime/shape_layers_py311'),
             'LD_LIBRARY_PATH':'/opt/shape-cpu/lib:'+env.get('LD_LIBRARY_PATH',''),
             'SHAPE_REFERENCE_ROOT':str(code)}
        stamp('tests')
        tests=launch('tests',['/opt/shape-cpu/bin/python','-B','-m','unittest','discover','-s','tests','-p','test_*.py'],cpu)
        while tests.poll() is None:time.sleep(5)
        if tests.returncode:raise RuntimeError('Regression tests failed; see tests.log')
        save(run/'tests_passed.json',{'at':time.time(),'exit_code':0})
        service_config={**settings,'glm_only':False,'single_glm':True,'sam_first':False}
        save(services/'settings.json',service_config)
        service_code=services/'service_code';service_code.mkdir(exist_ok=True)
        for path in (code/'experiment/style_line_sam/service').glob('*.py'):shutil.copy2(path,service_code/path.name)
        (services/'STOP.json').unlink(missing_ok=True)
        stamp('model_loading')
        server=launch('services',[sys.executable,'-u',str(service_code/'worker.py'),'--run-dir',str(services)],env)
        deadline=time.time()+4200
        while True:
            save(services/'activity.json',{'at':time.time(),'stage':'model_loading'})
            if server.poll() is not None:raise RuntimeError('Model host exited during loading')
            state=read(services/'status.json') if (services/'status.json').exists() else {}
            rows=state.get('services',{})
            if len(rows.get('glm',[]))==len(settings.get('glm_profiles') or {'glm_a':{}}) and len(rows.get('sam',[]))==len(settings['sam_ports']) and all(r['phase']=='ready' and r['alive'] for r in rows['glm']+rows['sam']):break
            if time.time()>deadline:raise TimeoutError('Model startup deadline exceeded')
            stamp('model_loading',services=rows);time.sleep(10)
        stamp('recovering')
        workflows=[Path(p) for p in settings.get('workflows',[str(run)])]
        workloads=[launch('recovery_'+str(i),['/opt/shape-cpu/bin/python','-B','-u',str(code/'run_shapes.py'),'--run-dir',str(path)],cpu)
                   for i,path in enumerate(workflows)]
        while any(workload.poll() is None for workload in workloads):
            if server.poll() is not None:raise RuntimeError('Model host died during recovery')
            if time.time()-started>settings['worker_max_seconds']:raise TimeoutError('Workflow deadline exceeded; checkpoints retained')
            save(services/'activity.json',{'at':time.time(),'stage':'recovering'})
            progress=read(run/'progress.json') if (run/'progress.json').exists() else {}
            stamp('recovering',done=progress.get('done',0),expected=progress.get('expected'),failed=progress.get('failed',0));time.sleep(10)
        if any(workload.returncode for workload in workloads):raise RuntimeError('Recovery failed; see recovery logs and per-case failure.json')
        if len(workflows)>1:
            report=launch('comparison',['/opt/shape-cpu/bin/python','-B','-m',
                'full_recovery.shape_experiment',*[str(path) for path in workflows]],cpu)
            while report.poll() is None:time.sleep(2)
            if report.returncode:raise RuntimeError('Experiment comparison export failed')
        stamp('completed',summaries={str(path):read(path/'summary.json') for path in workflows})
    except BaseException as exc:
        stamp('failed',error=str(exc));raise
    finally:
        stop()
        for log in logs:log.close()


if __name__=='__main__':main()
