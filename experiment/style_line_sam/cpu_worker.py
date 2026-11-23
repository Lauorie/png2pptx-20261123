"""Cluster-only recovery driver. Bounded CPU work; completed inference is reused."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path[:0] = [str(HERE)]
from clients import read, save, services


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    args = p.parse_args()
    from clients import require_cluster
    require_cluster()
    run = args.run_dir.resolve()
    rec = run / 'recovery_20261016'
    rec.mkdir(exist_ok=True)
    lock = (rec / '.cpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ.update(PYTHONPATH=str(PROJECT / '.runtime/shape_layers_py311'),
                      STYLE_SERVICE_DIR=str(rec / 'gpu'))
    sys.path.insert(0, str(PROJECT / '.runtime/shape_layers_py311'))
    sys.path.insert(0, str(run / 'code/v2/src'))
    sys.path.insert(0, str(PROJECT / 'vendor'))
    started = time.time()
    state = {'host': socket.gethostname(), 'pid': os.getpid(), 'started_at': started,
             'case_workers': 2, 'arms_sequential': True}
    stop = threading.Event()

    def stage(name, **extra):
        state.update(stage=name, **extra, updated_at=time.time())
        save(rec / 'cpu_status.json', state)
        print(name, extra, flush=True)

    def beat():
        while not stop.wait(20):
            save(rec / 'gpu/activity.json', {'at': time.time(), 'stage': state.get('stage')})
            save(rec / 'cpu_heartbeat.json', {'at': time.time(), 'stage': state.get('stage')})

    def call(name, script, *argv):
        stage(name)
        with (rec / (name + '.log')).open('a', buffering=1) as log:
            proc = subprocess.run([sys.executable, '-B', '-u', str(HERE / script), *map(str, argv)],
                                  stdout=log, stderr=subprocess.STDOUT, cwd=str(PROJECT))
        if proc.returncode:
            raise RuntimeError(f'{name} exited {proc.returncode}; see {name}.log')

    try:
        stage('preflight')
        import numpy, scipy, PIL, shapely, cairosvg, fitz, pptx
        from shape_integration.styling.export import render_native
        preflight = {'python': sys.executable, 'libreoffice': shutil.which('libreoffice'),
                     'versions': {m.__name__: getattr(m, '__version__', '?')
                                  for m in (numpy, scipy, PIL, shapely, cairosvg, fitz, pptx)},
                     'affinity': sorted(os.sched_getaffinity(0)), 'time': time.time()}
        save(rec / 'preflight.json', preflight)
        portable = PROJECT / '.runtime/portable_lo/bin'
        if not (portable / 'libreoffice').exists():
            stage('waiting_libreoffice')
            deadline = time.time() + 3600
            while not (portable / 'libreoffice').exists():
                if time.time() > deadline:
                    raise TimeoutError('LibreOffice is unavailable')
                time.sleep(10)
        # LibreOffice's binary registry crashes on the GPFS mmap implementation.
        # Reuse the exact staged renderer from local container storage instead.
        renderer_source = portable.parent
        renderer_local = Path('/tmp/style-recovery-lo')
        shutil.copytree(renderer_source, renderer_local, dirs_exist_ok=True)
        for path in renderer_local.rglob('*'):
            if path.is_file() and (path.suffix in ('.rc', '.rdb', '.xml', '.xcd', '.xcu', '.ini', '.conf')
                                   or path.name.endswith('rc') or path.name == 'libreoffice'):
                data = path.read_bytes()
                old = str(renderer_source).encode()
                if old in data:
                    path.write_bytes(data.replace(old, str(renderer_local).encode()))
        portable = renderer_local / 'bin'
        shutil.copy2(HERE / 'lo_wrapper.py', portable / 'libreoffice')
        (portable / 'libreoffice').chmod(0o755)
        os.environ['PATH'] = str(portable) + ':' + os.environ['PATH']
        os.environ['XDG_DATA_DIRS'] = str(renderer_local) + ':/usr/local/share:/usr/share'
        os.environ['STYLE_FONT'] = str(portable.parent / 'fonts/dejavu/DejaVuSans.ttf')
        version = subprocess.run(['libreoffice', '--version'], capture_output=True, text=True, timeout=30)
        save(rec / 'renderer.json', {'path': str(portable), 'version': version.stdout, 'error': version.stderr})
        if version.returncode:
            raise RuntimeError('Portable LibreOffice failed: ' + version.stderr)
        sample = run / 'arms/original/cases/ds007/styling/styled.pptx'
        scene = read(run / 'arms/original/cases/ds007/styling/scene_source.json')
        render_native(sample, rec / 'render_smoke/native_render', scene['source_size'])
        from clients import sha
        parity = sha(rec / 'render_smoke/native.png') == sha(sample.parent / 'native.png')
        save(rec / 'render_smoke/success.json', {'at': time.time(), 'size': scene['source_size'],
                                               'matches_saved_baseline_pixels': parity})
        if not parity:
            raise RuntimeError('Renderer pixels differ from the saved baseline; investigate before comparison')
        threading.Thread(target=beat, daemon=True).start()
        call('style_tests', 'tests/test_style_v2.py')
        call('repair_support', 'repair_support.py', '--run-dir', run, '--configs', 'side96_both', 'side96_line', 'side192_line')
        stage('wait_sam')
        deadline = time.time() + 6 * 3600
        while True:
            try:
                endpoints = services(run, 'sam')
                break
            except (OSError, RuntimeError, KeyError):
                if time.time() > deadline:
                    raise TimeoutError('SAM service did not become ready')
                time.sleep(15)
        stage('sam_ready', endpoints=endpoints)
        call('new_sam', 'sam_tiles.py', '--run-dir', run, '--config', 'side96_both_new')
        save(rec / 'gpu/control.json', {'release_sam': True, 'at': time.time()})
        stage('wait_glm')
        while True:
            try:
                services(run, 'glm')
                break
            except (OSError, RuntimeError, KeyError):
                if time.time() > deadline:
                    raise TimeoutError('GLM service did not become ready')
                time.sleep(15)
        for arm in ('v2_nosam', 'v2', 'v2_newsam'):
            completion = run / 'arms' / arm / 'completion.json'
            if not completion.exists() or read(completion)['failed']:
                call('arm_' + arm, 'run_arm.py', '--run-dir', run, '--arm', arm, '--case-workers', 2)
        for a, b in (('original', 'v2'), ('v2_nosam', 'v2'), ('v2', 'v2_newsam'), ('original', 'v2_newsam')):
            call('evaluate_' + a + '_vs_' + b, 'evaluate.py', '--run-dir', run, '--arms', a, b)
        call('sam_pilot', 'sam_pilot_report.py', '--run-dir', run, '--configs',
             'side96_both_part_owned', 'side96_line_part_owned', 'side192_line_part_owned', 'side96_both_new')
        call('delivery', 'deliver.py', '--run-dir', run)
        stage('completed', wall_seconds=time.time() - started)
    except BaseException as exc:
        stage('failed', error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        stop.set()
        save(rec / 'gpu/STOP.json', {'at': time.time(), 'reason': 'cpu_worker_' + state['stage']})


if __name__ == '__main__':
    main()
