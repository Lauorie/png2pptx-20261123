"""8-GPU service host: GLM (vLLM, TP4) + trained SAM, reachable on the pod network.

The style pipeline itself runs on the host that has LibreOffice; this worker only
serves models. GPUs 0-3 always serve GLM replica A. GPUs 4-7 first serve four SAM
instances; after control.json sets ``release_sam`` they are released and reused
for GLM replica B. STOP.json, a dead service or the lifetime budget ends the job.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def pod_ip():
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(('100.64.0.1', 9))
        return probe.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())
    finally:
        probe.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    run = parser.parse_args().run_dir.resolve()
    config = read(run / 'settings.json')
    code = run / 'service_code'
    env = {k: v for k, v in os.environ.items() if k.lower() not in {'http_proxy', 'https_proxy', 'all_proxy'}}
    env.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', VLLM_USE_BREAKABLE_CUDAGRAPH='1')
    children, logs = {}, []
    started = time.time()
    ip = pod_ip()
    state = {'hostname': socket.gethostname(), 'ip': ip, 'pid': os.getpid(), 'started_at': started}

    def launch(name, argv, environment):
        handle = (run / 'logs' / (name + '.log')).open('a', buffering=1)
        logs.append(handle)
        children[name] = subprocess.Popen(argv, env=environment, stdout=handle, stderr=subprocess.STDOUT,
                                          start_new_session=True)

    def kill(name):
        child = children.pop(name, None)
        if child is None or child.poll() is not None:
            return
        try:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)
        except ProcessLookupError:
            pass

    def stop(*unused):
        for name in list(children):
            kill(name)
        if unused:
            raise SystemExit(143)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def glm(name, devices, port):
        profile={**config,**config.get('glm_profiles',{}).get(name,{})}
        launch(name, [sys.executable, '-u', str(code / 'serve_glm.py'), '--run-dir', str(run / name),
                      '--model', config['glm_model'], '--tp', str(profile.get('glm_tp',4)), '--dp', str(profile.get('glm_dp',1)), '--port', str(port),
                      '--max-model-len', str(profile['glm_max_model_len']),
                      '--max-num-seqs', str(profile['glm_max_num_seqs']), '--max-num-batched-tokens', '8192',
                      '--gpu-memory-utilization',str(profile.get('glm_gpu_memory_utilization',.9))],
               {**env, 'CUDA_VISIBLE_DEVICES': devices})

    def sam(index, device):
        launch(f'sam{index}', [config['sam_python'], '-u', str(code / 'sam_service.py'), 'sam',
                               '--port', str(config['sam_ports'][index]), '--config', str(run / 'settings.json'),
                               '--state', str(run / f'sam{index}-state.json')],
               {**env, 'CUDA_VISIBLE_DEVICES': str(device),
                'PYTHONPATH': config['training_run'] + '/runtime/third_party/sam3'})

    def service_rows():
        rows = {'glm': [], 'sam': []}
        for name, port in (('glm_a', config['glm_ports'][0]), ('glm_b', config['glm_ports'][1])):
            if name in children:
                path = run / name / 'service-state.json'
                phase = read(path).get('phase') if path.exists() else 'loading'
                rows['glm'].append({'name': name, 'url': f'http://{ip}:{port}/v1', 'phase': phase,
                                    'alive': children[name].poll() is None})
        for index, port in enumerate(config['sam_ports']):
            name = f'sam{index}'
            if name in children:
                path = run / f'{name}-state.json'
                phase = read(path).get('phase') if path.exists() else 'loading'
                rows['sam'].append({'name': name, 'url': f'http://{ip}:{port}', 'phase': phase,
                                    'alive': children[name].poll() is None})
        return rows

    try:
        (run / 'logs').mkdir(exist_ok=True)
        sam_first = bool(config.get('sam_first', False))
        colocated = bool(config.get('colocated_glm',False))
        if not sam_first and not colocated:
            glm('glm_a', '0,1,2,3', config['glm_ports'][0])
        sam_released = bool(config.get('glm_only', False))
        if sam_released and not config.get('single_glm', False):
            glm('glm_b', '4,5,6,7', config['glm_ports'][1])
        elif not sam_released:
            for index, device in enumerate(config.get('sam_devices',[0, 1, 2, 3] if sam_first else [4, 5, 6, 7])):
                sam(index, device)
        ready_at = None
        while not (run / 'STOP.json').exists():
            rows = service_rows()
            dead = [r['name'] for r in rows['glm'] + rows['sam'] if not r['alive']]
            if dead:
                raise RuntimeError('Service exited: ' + ','.join(dead))
            if config.get('glm_profiles') and 'glm_b' not in children:
                arrow_path=run/'arrow_sam/service-state.json'
                arrow=read(arrow_path) if arrow_path.exists() else {}
                if rows['sam'] and all(r['alive'] and r['phase']=='ready' for r in rows['sam']) and (not config.get('require_arrow_sam',True) or arrow.get('phase')=='ready'):
                    glm('glm_b','4,5,6,7',config['glm_ports'][1])
            if colocated and 'glm_a' not in children:
                arrow_path=run/'arrow_sam/service-state.json'
                arrow=read(arrow_path) if arrow_path.exists() else {}
                if rows['sam'] and all(r['alive'] and r['phase']=='ready' for r in rows['sam']) and arrow.get('phase')=='ready':
                    # Load SAM first so vLLM profiles the actual remaining
                    # memory. Its smaller cache budget leaves room for SAM
                    # inference beyond the resident model allocations.
                    glm('glm_a','0,1,2,3,4,5,6,7',config['glm_ports'][0])
            control = read(run / 'control.json') if (run / 'control.json').exists() else {}
            if control.get('release_sam') and not sam_released:
                for index in range(len(config['sam_ports'])):
                    kill(f'sam{index}')
                save(run / 'sam_release.json', {'released_at': time.time(), 'control': control})
                if sam_first:
                    glm('glm_a', '0,1,2,3', config['glm_ports'][0])
                else:
                    glm('glm_b', '4,5,6,7', config['glm_ports'][1])
                sam_released = True
                continue
            if time.time() - started > config['worker_max_seconds']:
                raise TimeoutError('Worker lifetime budget exhausted')
            # Idle release: the host-side driver refreshes activity.json while it uses the
            # services. Without it (driver dead or finished without STOP) the GPUs are freed.
            if ready_at is None and any(r['phase'] == 'ready' for r in rows['glm']):
                ready_at = time.time()
            if ready_at is not None:
                activity = read(run / 'activity.json')['at'] if (run / 'activity.json').exists() else 0
                idle = time.time() - max(ready_at, activity)
                if idle > config['idle_stop_seconds']:
                    save(run / 'STOP.json', {'reason': 'idle_release', 'idle_seconds': idle, 'at': time.time()})
                    continue
            save(run / 'status.json', {**state, 'phase': 'serving', 'heartbeat_at': time.time(),
                                       'sam_released': sam_released, 'services': rows})
            time.sleep(10)
        save(run / 'status.json', {**state, 'phase': 'completed', 'heartbeat_at': time.time(),
                                   'stop': read(run / 'STOP.json')})
    except Exception as exc:
        save(run / 'status.json', {**state, 'phase': 'failed', 'error': str(exc), 'heartbeat_at': time.time()})
        raise
    finally:
        stop()
        for handle in logs:
            handle.close()


if __name__ == '__main__':
    main()
