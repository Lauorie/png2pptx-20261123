"""One fixed 8-GPU worker: persistent GLM/SAM, paired forward reconstruction and evaluation."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import re
from core import read,save


def completion_counts(comparison):
    manifest=comparison/'manifest.json'
    if not manifest.exists():return {}
    data=read(manifest)
    arms=data.get('config', {}).get('arms', ['glm_control','sam_forward'])
    completed=[[bool((comparison/arm/'cases'/item['id']/'forward_complete.json').exists())
                for arm in arms] for item in data['images']]
    return {'forward_pairs_complete':sum(all(pair) for pair in completed),
            'forward_arms_complete':sum(sum(pair) for pair in completed),
            'forward_pairs_expected':len(completed)}


def wait_workload(process, services, run, state, interval=10):
    """A live supervisor heartbeat is distinct from a completed output receipt."""
    save(run/'status.json', state)
    while process.poll() is None:
        if any(service.poll() is not None for service in services):
            raise RuntimeError('Model service died during '+state['phase'])
        progress=completion_counts(Path(state['comparison_path'])) if state.get('comparison_path') else {}
        save(run/'status.json', {**state, 'heartbeat_at': time.time(), **progress})
        time.sleep(interval)
    return process.returncode


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True);args=parser.parse_args()
    run=args.run_dir;code=run/'code';config=read(run/'settings.json');children=[];logs=[]
    owner=(run/'.worker.lock').open('a')
    try:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        owner.close();raise
    environment=dict(os.environ)
    for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:environment.pop(k,None)
    environment.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    project=Path(config['project_root']);cpu_env={**environment,'PYTHONPATH':str(project/'.runtime/shape_layers_py311'),
        'LD_LIBRARY_PATH':'/opt/shape-cpu/lib:'+environment.get('LD_LIBRARY_PATH',''),
        'SHAPE_REFERENCE_ROOT':str(code/'reference')}
    def launch(name,argv,env):
        log=(run/(name+'.log')).open('a',buffering=1);logs.append(log)
        proc=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);children.append(proc)
        return proc
    def stop(*unused):
        for child in children:
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for child in children:
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
        if unused:raise SystemExit(143)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    status={'hostname':socket.gethostname(),'pid':os.getpid(),'phase':'preflight','started_at':time.time()}
    save(run/'status.json',status)
    try:
        subprocess.run(['/opt/shape-cpu/bin/python','-B',str(code/'preflight.py'),'--config',str(run/'settings.json'),'--output',str(run/'preflight.json')],env=cpu_env,check=True)
        def wait_ready(processes, paths, phase):
            start=time.monotonic()
            while True:
                if any(p.poll() is not None for p in processes):raise RuntimeError('Model service exited during '+phase)
                phases=[read(p).get('phase') if p.exists() else 'loading' for p in paths]
                save(run/'status.json',{**status,'phase':phase,'services':phases,'updated_at':time.time()})
                if all(x=='ready' for x in phases):break
                if time.monotonic()-start>4200:raise TimeoutError('Model startup timeout')
                time.sleep(10)
        sam_services=[]
        devices=[] if config.get('sam_cache_only') else [4,5,6,7]
        for index,device in enumerate(devices):
            sam_services.append(launch(f'sam{index}',['/data/dongdaozhu/conda/envs/sam3/bin/python','-u',str(code/'sam_service.py'),'sam',
                '--port',str(config['sam_ports'][index]),'--config',str(run/'settings.json'),'--state',str(run/f'sam{index}-state.json')],
                {**environment,'CUDA_VISIBLE_DEVICES':str(device),'PYTHONPATH':config['training_run']+'/runtime/third_party/sam3'}))
        wait_ready(sam_services,[run/f'sam{i}-state.json' for i in range(len(sam_services))],'sam_loading')
        save(run/'status.json',{**status,'phase':'sam_precompute','updated_at':time.time()})
        prep=launch('prepare_sam',['/opt/shape-cpu/bin/python','-B','-u',str(code/'prepare_sam.py'),
            '--run-dir',str(run/'shared'),'--config',str(run/'settings.json')],cpu_env)
        if prep.wait()!=0:raise RuntimeError('SAM precomputation failed')
        receipt=read(run/'shared/sam_precompute.json')
        if receipt['completed']!=receipt['expected']:raise RuntimeError('Incomplete SAM cache')
        for child in sam_services:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        for child in sam_services:
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
        save(run/'sam_gpu_release.json',{'process_returncodes':[p.returncode for p in sam_services],
             'cached_cases':receipt['completed'],'released_for_glm_tp':config['glm_tp'],'at':time.time()})
        glm=launch('glm',[sys.executable,'-u',str(code/'serve_glm.py'),'--run-dir',str(run/'glm'),'--model',config['glm_model'],
            '--tp',str(config['glm_tp']),'--dp','1','--max-model-len','65536','--max-num-seqs','24','--max-num-batched-tokens','8192'],
            {**environment,'CUDA_VISIBLE_DEVICES':','.join(str(i) for i in range(config['glm_tp']))})
        services=[glm]
        wait_ready(services,[run/'glm/service-state.json'],'glm_loading')
        comparison=run/'comparison';active_code=code;active_config=run/'settings.json';resume_seen=None
        while True:
            active_env={**cpu_env,'SHAPE_REFERENCE_ROOT':str(active_code/'reference')}
            (comparison/'sam_cache').symlink_to(run/'shared/sam_cache') if not (comparison/'sam_cache').exists() else None
            phase={**status,'phase':'paired_forward','active_code':str(active_code),
                'comparison_path':str(comparison),'updated_at':time.time()}
            proc=launch('forward',['/opt/shape-cpu/bin/python','-B','-u',str(active_code/'run.py'),
                '--run-dir',str(comparison),'--config',str(active_config)],active_env)
            failed='forward' if wait_workload(proc,services,run,phase) else None
            if not failed:
                phase={**status,'phase':'paired_evaluation','active_code':str(active_code),
                    'comparison_path':str(comparison),'updated_at':time.time()}
                report=launch('report',['/opt/shape-cpu/bin/python','-B','-u',str(active_code/'report.py'),
                    '--run-dir',str(comparison),'--config',str(active_config)],active_env)
                if wait_workload(report,services,run,phase)!=0:failed='evaluation'
            if not failed:break
            # Keep already-loaded GPUs for bounded debugging, with one CPU writer.
            save(run/'status.json',{**status,'phase':'awaiting_repair','failed_stage':failed,
                'active_code':str(active_code),'comparison_path':str(comparison),'updated_at':time.time()})
            deadline=time.monotonic()+1800
            while True:
                if any(p.poll() is not None for p in services):raise RuntimeError('Model service died during repair wait')
                marker=run/'RESUME.json';value=marker.read_text() if marker.exists() else None
                if value and value!=resume_seen:
                    instruction=read(marker);revision=instruction['revision'];name=instruction['comparison']
                    if not re.fullmatch(r'r[0-9]+',revision) or not re.fullmatch(r'comparison(?:_r[0-9]+)?',name):
                        raise ValueError('Invalid repair paths')
                    active_code=run/'revisions'/revision;comparison=run/name
                    active_config=active_code/'settings.json'
                    revised=read(active_config)
                    for key in ['glm_model','glm_tp','glm_url','sam_threshold','training_run','baseline_root','baseline_run']:
                        if revised[key]!=config[key]:raise ValueError('Live model/baseline identity cannot change: '+key)
                    if not all((comparison/p).is_dir() for p in ['native_review','accepted']):
                        raise ValueError('Create host-owned output/native_review/accepted directories before RESUME')
                    subprocess.run(['/opt/shape-cpu/bin/python','-B',str(active_code/'preflight.py'),
                        '--config',str(active_config),'--output',str(active_code/'preflight.json')],
                        env={**cpu_env,'SHAPE_REFERENCE_ROOT':str(active_code/'reference')},check=True)
                    resume_seen=value;break
                if time.monotonic()>deadline:raise TimeoutError('Repair wait exceeded 30 minutes; releasing GPUs')
                time.sleep(10)
        save(run/'status.json',{**status,'phase':'completed','active_code':str(active_code),
            'comparison_path':str(comparison),'updated_at':time.time(),'aggregate':read(comparison/'aggregate.json')})
    except Exception as exc:
        save(run/'status.json',{**status,'phase':'failed','error':str(exc),'updated_at':time.time()});raise
    finally:
        stop()
        for log in logs:log.close()
        owner.close()


if __name__=='__main__':main()
