"""Supervise configured SAM power controllers and the SenseNova service."""
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

run = Path(os.environ['SENSENOVA_RUN_DIR'])
code = run / 'code_snapshot'
CONFIG = json.loads((run / 'config.json').read_text())
GPU_COUNT = int(CONFIG['gpu_count'])
POWER_PCT = float(CONFIG.get('power_target_pct', 60))
IDLE_SECONDS = float(CONFIG.get('idle_resume_seconds', 600))
stop = threading.Event()
for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, lambda *_: stop.set())
children, logs = {}, []
state = dict(phase='starting', pid=os.getpid(), started_at=time.time(), power_target_pct=POWER_PCT, idle_resume_seconds=IDLE_SECONDS)

def launch(name, argv, env=None):
    log = (run / 'evidence' / (name + '.log')).open('a', buffering=1)
    logs.append(log)
    p = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    children[name] = p
    return p

def terminate(p):
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        p.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def health():
    try:
        with opener.open('http://127.0.0.1:8091/health', timeout=4) as r:
            return r.status == 200
    except (OSError, ValueError):
        return False

startup_locks = []
def take_startup_locks():
    for i in range(GPU_COUNT):
        f = (run / 'power' / str(i) / 'gpu-work.lock').open('a')
        startup_locks.append(f)
        fcntl.flock(f, fcntl.LOCK_EX)

def release_startup_locks():
    for f in startup_locks:
        f.close()
    startup_locks.clear()

with (run / 'worker.lock').open('a') as owner:
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (run / 'STOP').exists():
        raise RuntimeError('STOP exists; use a fresh run')
    try:
        import torch
        if torch.cuda.device_count() != GPU_COUNT:
            raise RuntimeError(f'Expected exactly {GPU_COUNT} scheduler-allocated GPUs')
        uuids = [str(torch.cuda.get_device_properties(i).uuid) for i in range(GPU_COUNT)]
        uuids = [u if u.startswith('GPU-') else 'GPU-' + u for u in uuids]
        if len(set(uuids)) != GPU_COUNT:
            raise RuntimeError('GPU UUID mapping is not unique')
        state.update(gpu_uuids=uuids, hostname=socket.gethostname())
        save(run / 'evidence' / 'gpu-mapping.json', state)
        power_envs = []
        for i, uuid in enumerate(uuids):
            directory = run / 'power' / str(i)
            directory.mkdir(exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid)
            power_envs.append(env)
        take_startup_locks()
        for i in range(GPU_COUNT):
            launch(f'power-{i}', [sys.executable, '-u', str(code / 'power_keepalive.py'),
                   '--run-dir', str(run / 'power' / str(i)), '--power-pct', str(POWER_PCT), '--activity-dir', str(run),
                           '--activity-gpu-index', str(i), '--idle-resume-seconds', str(IDLE_SECONDS),
                           '--low-power-threshold-pct', str(CONFIG.get('low_power_threshold_pct', 40))], power_envs[i])
        launch('service', ['bash', str(code / 'service.sh')])
        deadline, unhealthy_since = time.monotonic() + 1800, None
        ready = False
        while not stop.is_set() and not (run / 'STOP').exists():
            for i in range(GPU_COUNT):
                name = f'power-{i}'
                if children[name].poll() is not None:
                    state.setdefault('restarts', {}).setdefault(name, 0)
                    state['restarts'][name] += 1
                    launch(name, [sys.executable, '-u', str(code / 'power_keepalive.py'),
                           '--run-dir', str(run / 'power' / str(i)), '--power-pct', str(POWER_PCT), '--activity-dir', str(run),
                           '--activity-gpu-index', str(i), '--idle-resume-seconds', str(IDLE_SECONDS),
                           '--low-power-threshold-pct', str(CONFIG.get('low_power_threshold_pct', 40))], power_envs[i])
            exited = children['service'].poll() is not None
            healthy = False if exited else health()
            if healthy:
                if not ready:
                    release_startup_locks()
                    ready = True
                unhealthy_since = None
                state['phase'] = 'ready'
            elif ready:
                unhealthy_since = unhealthy_since or time.monotonic()
                state['phase'] = 'degraded'
            if exited or (ready and unhealthy_since and time.monotonic() - unhealthy_since > 180) or (not ready and time.monotonic() > deadline):
                terminate(children['service'])
                release_startup_locks()
                take_startup_locks()
                state.setdefault('restarts', {}).setdefault('service', 0)
                state['restarts']['service'] += 1
                state['phase'] = 'restarting'
                save(run / 'evidence' / 'supervisor-state.json', state)
                if stop.wait(10):
                    break
                launch('service', ['bash', str(code / 'service.sh')])
                ready, unhealthy_since = False, None
                deadline = time.monotonic() + 1800
            powers = []
            for i in range(GPU_COUNT):
                p = run / 'power' / str(i) / 'power-state.json'
                powers.append(json.loads(p.read_text()) if p.exists() else {})
            state.update(power=powers, updated_at=time.time(), health=healthy,
                         children={n:p.pid for n,p in children.items()})
            save(run / 'evidence' / 'supervisor-state.json', state)
            stop.wait(5)
    except BaseException:
        state.update(phase='failed', error=traceback.format_exc())
        raise
    finally:
        release_startup_locks()
        for p in children.values():
            terminate(p)
        for log in logs:
            log.close()
        state.update(phase='failed' if state.get('phase') == 'failed' else 'stopped', stopped_at=time.time())
        save(run / 'evidence' / 'supervisor-state.json', state)
