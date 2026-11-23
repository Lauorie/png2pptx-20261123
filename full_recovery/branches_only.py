"""Resume shape/text checkpoints under one case budget, without composition."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from .case_slots import CaseSlots
from .fresh_pipeline import archive_unpublished_requests
from .prepare import read, sha
from .atomic_json import save
from .text_product_validation import verify


def execute(run, *, shape_resume_settings=None, shape_pending_ids=None):
    if os.environ.get('STYLE_CLUSTER_WORKER') != '1':
        raise RuntimeError('Use the CPU rjob worker')
    settings = read(run / 'settings.json')
    if settings['workflow_scope'] != 'shape_text':
        raise ValueError('This entry point never composes a PPT')
    selection = read(run / 'resume_selection.json')
    source = Path(selection['source_run'])
    items = read(run / 'source_manifest.json')['images']
    text_ids = set(selection['pending_text'])
    shape_ids = set(selection['pending_shape'])
    text_items = [item for item in items if item['id'] in text_ids]
    children = []
    mutex = threading.Lock()
    cancel = threading.Event()
    states = {}
    env = {**os.environ, 'PPT_LOCAL_GLM': '1',
           'BOYUE_API_KEY': 'local-glm-without-auth',
           'PYTHONDONTWRITEBYTECODE': '1'}

    def stamp(name, state, **detail):
        with mutex:
            states[name] = {'state': state, 'at': time.time(), **detail}
            save(run / 'branch_progress.json', {'tasks': dict(states), 'at': time.time()})

    def stop(*args):
        cancel.set()
        with mutex:
            snapshot = list(children)
        for child in snapshot:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for child in snapshot:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if args:
            raise SystemExit(143)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def command(name, argv, environment):
        if cancel.is_set():
            raise RuntimeError('Resume cancelled')
        with (run / 'logs' / (name + '.log')).open('a', buffering=1) as log:
            with mutex:
                if cancel.is_set():
                    raise RuntimeError('Resume cancelled')
                child = subprocess.Popen(list(map(str, argv)), cwd=run, env=environment,
                                         stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                children.append(child)
            while child.poll() is None:
                if cancel.wait(0.5):
                    raise RuntimeError('Resume cancelled')
            if child.returncode:
                raise RuntimeError(f'{name} exited {child.returncode}; see its log')

    def text_case(item):
        cid = item['id']
        output = run / 'texts/cases' / cid
        name = 'text_' + cid
        stamp(name, 'queued')
        with CaseSlots(settings['case_slot_config'], 'text').acquire(cid):
            if cancel.is_set():
                raise RuntimeError('Resume cancelled')
            if (output / 'summary.json').is_file():
                verify(item, output)
                stamp(name, 'completed', checkpoint_verified=True)
                return
            stamp(name, 'running')
            archive_unpublished_requests(output)
            argv = [sys.executable, '-B', '-u', run / 'code/full_recovery/text_stage.py',
                    '--backend', run / 'backends/text', '--image', item['original'],
                    '--output', output.resolve(), '--ocr-python', settings['text_ocr_python'],
                    '--models', settings['text_models'],
                    '--recognition-mode', settings['text_recognition_mode'],
                    '--ocr-mode', settings['text_ocr_mode'],
                    '--geometry-backend', settings.get('text_geometry_backend', 'glm'),
                    '--ocr-template-validation', settings['text_ocr_template_validation'],
                    '--threshold', settings['text_threshold'],
                    '--max-score-drop', settings['text_max_score_drop'],
                    '--ocr-max-side', settings['text_ocr_max_side'], '--resume']
            if settings['text_crop_refine']:
                argv.append('--glm-crop-refine')
            try:
                command(name, argv, env)
                save(output / 'completed.json', verify(item, output))
                stamp(name, 'completed')
            except Exception as exc:
                stamp(name, 'failed', error=str(exc))
                raise

    def text_branch():
        errors = {}
        try:
            with ThreadPoolExecutor(max_workers=settings['case_slot_config']['total']) as pool:
                futures = {pool.submit(text_case, item): item['id'] for item in text_items}
                for task in as_completed(futures):
                    try:
                        task.result()
                    except Exception as exc:
                        errors[futures[task]] = str(exc)
            return errors
        finally:
            CaseSlots(settings['case_slot_config'], 'text').drained()

    def shape_branch():
        try:
            if shape_ids:
                command('shape', [sys.executable, '-B', '-u', run / 'shape/code/run_shapes.py',
                                 '--run-dir', source / 'shape',
                                 '--resume-settings', shape_resume_settings or run / 'shape_resume_settings.json',
                                 '--selected-ids', shape_pending_ids or run / 'shape_pending_ids.json',
                                 '--report-root', run / 'shape'],
                        {**env, 'GLM_API_RELAY_URL': settings['glm_url'],
                         'SHAPE_REFERENCE_ROOT': str(run / 'shape/code')})
            return {}
        except Exception as exc:
            return {'shape_controller': str(exc)}
        finally:
            CaseSlots(settings['case_slot_config'], 'shape').drained()

    with ExitStack() as locks:
        # Case directories are reused in place; hold all previous writer locks
        # for the entire continuation, including after a controller failure.
        for relative in ('worker.lock', 'fresh_branches.lock'):
            handle = locks.enter_context((source / relative).open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # The shape subprocess acquires this lock itself and retains it while
        # writing; check it before launching without locking out our child.
        with (source / 'shape/.pipeline.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for branch in ('shape', 'text'):
            (Path(settings['case_slot_config']['root']) / (branch + '_drained.json')).unlink(missing_ok=True)
        for item in items:
            if sha(item['original']) != item['source_sha256']:
                raise ValueError('Original changed: ' + item['id'])
        errors = {}
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(shape_branch), pool.submit(text_branch)]
                while not all(future.done() for future in futures):
                    completed_shape = sum((source / 'shape/geometry/cases' / cid / 'complete.json').is_file()
                                          for cid in shape_ids)
                    with mutex:
                        completed_text = sum(v['state'] == 'completed' for v in states.values())
                        failed_text = sum(v['state'] == 'failed' for v in states.values())
                    save(run / 'progress.json', {'expected': len(shape_ids) + len(text_ids),
                         'completed': completed_shape + completed_text, 'failed': failed_text,
                         'shape_completed_new': completed_shape, 'text_completed_new': completed_text,
                         'composition_enabled': False, 'at': time.time()})
                    if cancel.wait(2):
                        raise RuntimeError('Resume cancelled')
                for future in futures:
                    errors.update(future.result())
            products = []
            for item in items:
                cid = item['id']
                try:
                    text = verify(item, run / 'texts/cases' / cid)
                    root = source / 'shape'
                    shape = read(root / 'geometry/cases' / cid / 'complete.json')
                    if sha(shape['pptx']) != shape['pptx_sha256']:
                        raise ValueError('Shape PPT hash differs')
                    products.append({'id': cid, 'text': text, 'text_root': str(run / 'texts/cases' / cid),
                                     'shape': shape})
                except Exception as exc:
                    errors[cid] = str(exc)
            summary = {'state': 'completed' if not errors else 'completed_with_failures',
                       'expected': len(items), 'exported': len(products), 'errors': errors,
                       'composition_enabled': False, 'arrow_inference_enabled': False,
                       'products': products, 'at': time.time()}
            save(run / 'summary.json', summary)
            if errors:
                raise RuntimeError('Branches retain failed/incomplete checkpoints; see summary.json')
        finally:
            stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    execute(parser.parse_args().run_dir.resolve())


if __name__ == '__main__':
    main()
