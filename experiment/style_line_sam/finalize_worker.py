"""Resume cached comparisons and retry invalid final verdicts on the same GLM."""
import argparse
import fcntl
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import traceback
from clients import read, save, services, require_cluster

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    run = p.parse_args().run_dir.resolve()
    require_cluster()
    rec = run / 'recovery_20261016'
    service = rec / 'finalize_gpu'
    lock = (rec / '.cpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ.update(STYLE_SERVICE_DIR=str(service),
                     PYTHONPATH=str(PROJECT / '.runtime/shape_layers_py311'),
                     STYLE_FONT=str(PROJECT / '.runtime/portable_lo/fonts/dejavu/DejaVuSans.ttf'))
    state = {'host': socket.gethostname(), 'pid': os.getpid(), 'started_at': time.time()}
    stop = threading.Event()

    def stage(name, **extra):
        state.update(stage=name, updated_at=time.time(), **extra)
        save(rec / 'finalize_status.json', state)
        print(name, extra, flush=True)

    def heartbeat():
        while not stop.wait(20):
            save(service / 'activity.json', {'at': time.time(), 'stage': state.get('stage')})

    def call(name, script, *argv):
        stage(name)
        with (rec / (name + '.log')).open('a', buffering=1) as log:
            subprocess.run([sys.executable, '-B', '-u', str(HERE / script), *map(str, argv)],
                           cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        threading.Thread(target=heartbeat, daemon=True).start()
        call('judge_parser_tests', 'tests/test_judge_parser.py')
        stage('wait_glm')
        deadline = time.time() + 3600
        while True:
            try:
                services(run, 'glm')
                break
            except (OSError, RuntimeError, KeyError):
                if time.time() > deadline:
                    raise TimeoutError('Final review GLM readiness deadline exceeded')
                time.sleep(10)
        from deliver import PAIRS
        for a, b in PAIRS:
            # Valid cached requests replay; only invalid final responses are corrected.
            call('finalize_' + a + '_vs_' + b, 'evaluate.py', '--run-dir', run, '--arms', a, b)
        call('finalize_delivery', 'deliver.py', '--run-dir', run)
        stage('completed', wall_seconds=time.time() - state['started_at'])
    except BaseException as exc:
        stage('failed', error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        stop.set()
        save(service / 'STOP.json', {'at': time.time(), 'reason': 'finalize_' + state['stage']})


if __name__ == '__main__':
    main()
