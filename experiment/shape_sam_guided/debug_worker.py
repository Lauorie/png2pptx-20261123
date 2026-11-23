"""Persistent model service, sequential frozen workloads, bounded debugging window."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    run = parser.parse_args().run_dir.resolve()
    lock = (run / '.worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = read(run / 'settings.json')
    env = {k: v for k, v in os.environ.items() if k.lower() not in {'http_proxy', 'https_proxy', 'all_proxy'}}
    env.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', VLLM_USE_BREAKABLE_CUDAGRAPH='1')
    children = []
    logs = []
    started = time.time()
    state = {'hostname': socket.gethostname(), 'pid': os.getpid(), 'started_at': started}
    def publish(phase, **fields):
        save(run / 'status.json', {**state, 'phase': phase, 'heartbeat_at': time.time(), **fields})
    def launch(argv, logfile, environment):
        handle = logfile.open('a', buffering=1)
        logs.append(handle)
        child = subprocess.Popen(argv, env=environment, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
        return child
    def stop(*unused):
        for child in reversed(children):
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
                except ProcessLookupError:
                    pass
        if unused:
            raise SystemExit(143)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        publish('glm_loading')
        glm = launch([sys.executable, '-u', str(run / 'service_code/serve_glm.py'),
                      '--run-dir', str(run / 'glm'), '--model', config['glm_model'],
                      '--tp', '8', '--dp', '1', '--max-model-len', '65536',
                      '--max-num-seqs', '24', '--max-num-batched-tokens', '8192'],
                     run / 'model_launcher.log', {**env, 'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'})
        while True:
            if glm.poll() is not None:
                raise RuntimeError('GLM service exited during startup')
            path = run / 'glm/service-state.json'
            service = read(path) if path.exists() else {}
            publish('glm_loading', service=service)
            if service.get('phase') == 'ready':
                break
            if time.time() - started > 4200:
                raise TimeoutError('GLM startup exceeded budget')
            time.sleep(10)
        last_work = time.time()
        while not (run / 'STOP.json').exists():
            if glm.poll() is not None:
                raise RuntimeError('GLM service exited')
            if time.time() - started > config['worker_max_seconds']:
                raise TimeoutError('Worker time budget exhausted')
            pending = [p for p in sorted((run / 'commands').glob('*.json'))
                       if not (run / 'receipts' / p.name).exists()]
            if not pending:
                publish('ready_for_command', idle_seconds=time.time() - last_work)
                if time.time() - last_work > config['debug_idle_seconds']:
                    raise TimeoutError('Debug idle budget exhausted')
                time.sleep(5)
                continue
            path = pending[0]
            spec = read(path)
            code = (run / 'revisions' / spec['revision']).resolve()
            if code.parent != run / 'revisions' or Path(spec['script']).name != spec['script']:
                raise ValueError('Invalid frozen workload path')
            cpu = {**env, 'PYTHONPATH': str(Path(config['project_root']) / '.runtime/shape_layers_py311'),
                   'SHAPE_REFERENCE_ROOT': str(code / 'reference'),
                   'LD_LIBRARY_PATH': '/opt/shape-cpu/lib:' + env.get('LD_LIBRARY_PATH', '')}
            receipt = {'spec': spec, 'started_at': time.time()}
            publish('workload_running', command=path.name, **receipt)
            task = launch(['/opt/shape-cpu/bin/python', '-B', '-u', str(code / spec['script']), *spec.get('args', [])],
                          run / 'jobs' / (path.stem + '.log'), cpu)
            while task.poll() is None:
                if glm.poll() is not None:
                    raise RuntimeError('GLM service exited during workload')
                publish('workload_running', command=path.name, **receipt)
                time.sleep(10)
            save(run / 'receipts' / path.name, {**receipt, 'exit_code': task.returncode, 'finished_at': time.time()})
            last_work = time.time()
        publish('completed', stop=read(run / 'STOP.json'))
    except Exception as exc:
        publish('failed', error=str(exc))
        raise
    finally:
        stop()
        for handle in logs:
            handle.close()


if __name__ == '__main__':
    main()
