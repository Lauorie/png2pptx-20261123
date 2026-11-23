#!/usr/bin/env python3
"""Own the SAM and power-controller processes until STOP or a termination signal."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

from engine import save


def worker_ip():
    if os.environ.get('POD_IP'):
        return os.environ['POD_IP']
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        # Select the internal route without transmitting a packet.
        sock.connect(('10.0.0.1', 80))
        return sock.getsockname()[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--power-pct', type=float, default=60)
    parser.add_argument('--startup-timeout', type=float, default=1800)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if not 0 < args.power_pct <= 100:
        parser.error('--power-pct must be in (0, 100]')
    with (run / 'worker.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run / 'STOP').exists():
            raise RuntimeError('STOP already exists; use a fresh deployment directory')
        config = json.loads((run / 'config.json').read_text())
        code = Path(__file__).resolve().parent
        environment = dict(os.environ)
        for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
            environment.pop(key, None)
        environment.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
            XDG_CACHE_HOME=str(run / 'cache'), MPLCONFIGDIR=str(run / 'cache/matplotlib'))
        # Preserve scheduler GPU visibility; CUDA ordinal 0 may be any host GPU.
        environment.pop('PYTHONPATH', None)
        stopped = threading.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: stopped.set())
        children, logs = {}, []
        state = {'phase': 'loading', 'hostname': socket.gethostname(), 'pid': os.getpid(),
                 'started_at': time.time(), 'power_target_pct': args.power_pct,
                 'idle_resume_seconds': config.get('idle_resume_seconds', 600)}
        status_path = run / 'worker-state.json'
        save(status_path, state)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        startup_lock = (run / 'gpu-work.lock').open('a')
        fcntl.flock(startup_lock, fcntl.LOCK_EX)
        try:
            state['pod_ip'] = worker_ip()
            commands = {
                'power': [sys.executable, '-u', str(code / 'power_keepalive.py'),
                          '--run-dir', str(run), '--power-pct', str(args.power_pct),
                          '--idle-resume-seconds', str(config.get('idle_resume_seconds', 600)),
                          '--low-power-threshold-pct', str(config.get('low_power_threshold_pct', 40))],
                'sam': [sys.executable, '-u', str(code / 'server.py'),
                        '--run-dir', str(run), '--config', str(run / 'config.json')]}
            for name, argv in commands.items():
                log = (run / (name + '.log')).open('a', buffering=1)
                logs.append(log)
                children[name] = subprocess.Popen(argv, env=environment, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + args.startup_timeout
            was_ready = False
            while not stopped.is_set() and not (run / 'STOP').exists():
                for name, process in children.items():
                    if process.poll() is not None:
                        raise RuntimeError(f'{name} exited with {process.returncode}; see {name}.log')
                health = None
                try:
                    with opener.open(f"http://127.0.0.1:{config['port']}/health", timeout=5) as response:
                        health = json.load(response)
                except (OSError, ValueError):
                    pass
                power_file = run / 'power-state.json'
                power = json.loads(power_file.read_text()) if power_file.exists() else {}
                power_ok = power.get('phase') in ('running', 'yielding_to_inference', 'cooldown', 'monitoring_low_power', 'waiting_for_service') and (
                    time.time() - power.get('updated_at', 0) < 30)
                if health and health.get('warmup_passed') and power_ok:
                    if power.get('gpu_uuid') != health.get('model', {}).get('gpu_uuid'):
                        raise RuntimeError('SAM and power controller resolved different GPU UUIDs')
                    if startup_lock is not None:
                        startup_lock.close()
                        startup_lock = None
                    was_ready = True
                    state.update(phase='ready', service=health, power=power)
                    save(run / 'endpoint.json', {'url': f"http://{state['pod_ip']}:{config['port']}",
                        'hostname': state['hostname'], 'updated_at': time.time()})
                else:
                    state.update(phase='degraded' if was_ready else 'loading', power=power)
                    if not was_ready and time.monotonic() > deadline:
                        raise TimeoutError('Model warmup / power telemetry startup timed out')
                state['updated_at'] = time.time()
                save(status_path, state)
                stopped.wait(5)
            state['phase'] = 'stopped'
        except Exception:
            state.update(phase='failed', error=traceback.format_exc())
            raise
        finally:
            if startup_lock is not None:
                startup_lock.close()
            for process in children.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
            for process in children.values():
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            (run / 'endpoint.json').unlink(missing_ok=True)
            state['stopped_at'] = time.time()
            save(status_path, state)
            for filename in ('service-state.json', 'power-state.json'):
                path = run / filename
                if path.exists():
                    child_state = json.loads(path.read_text())
                    child_state.update(phase=state['phase'], stopped_at=state['stopped_at'])
                    save(path, child_state)
            for log in logs:
                log.close()


if __name__ == '__main__':
    main()
