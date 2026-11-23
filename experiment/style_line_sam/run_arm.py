"""Run one frozen styling implementation over the frozen 27-case geometry.

Each arm imports its own frozen ``src`` tree, so arms run in separate processes.
Both use the same local GLM transport, budget cap, inputs and options.
"""
import argparse
import fcntl
import inspect
import os
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--ids', nargs='*')
    parser.add_argument('--case-workers', type=int, default=2)
    args = parser.parse_args()
    if args.case_workers < 1:
        parser.error('--case-workers must be positive')
    run = args.run_dir.resolve()
    sys.path[:0] = [str(HERE)]
    from clients import read, save, sha, StyleRuntime, Slots, Endpoints, require_cluster
    require_cluster()
    arm = read(run / 'arms' / (args.arm + '.json'))
    sys.path[:0] = [str(run / arm['code']), str(PROJECT / 'vendor')]
    from shape_integration.styling import options as style_options
    from shape_integration.styling.controller import run_case
    from shape_integration.transport import RequestTimeout

    out = run / 'arms' / args.arm
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save(out / 'execution.json', {'case_workers': args.case_workers, 'pid': os.getpid(),
                                 'host': __import__('socket').gethostname(), 'started_at': time.time()})
    config = read(run / 'config.json')
    config['styling'] = {**config['styling'], **arm.get('styling_overrides', {})}
    opt = style_options(config)
    inputs = read(run / 'inputs.json')['cases']
    if args.ids:
        inputs = [c for c in inputs if c['id'] in args.ids]
    runtime = StyleRuntime(run, arm['runtime'], RequestTimeout,
                           endpoints=Endpoints(run), slots=Slots(arm['runtime']['concurrency']))
    extra = {}
    accepted = inspect.signature(run_case).parameters
    if arm.get('line_evidence'):
        if 'line_evidence' not in accepted:
            raise ValueError('This implementation has no line evidence input')
        extra['line_evidence'] = run / arm['line_evidence']
    save(out / 'arm_identity.json', {'arm': arm, 'options': opt, 'cases': [c['id'] for c in inputs],
                                     'code_files': {str(p.relative_to(run / arm['code'])): sha(p)
                                                    for p in sorted((run / arm['code']).rglob('*.py'))}})
    renders = threading.BoundedSemaphore(opt['render_workers'])
    pixels = threading.BoundedSemaphore(opt['pixel_workers'])
    lock_progress = threading.Lock()
    progress_rows, results = {}, {}

    def progress(ident, stage, done, total):
        with lock_progress:
            progress_rows[ident] = {'stage': stage, 'done': done, 'total': total, 'at': time.time()}
            save(out / 'progress.json', {'cases': progress_rows, 'api': runtime.slots.snapshot(),
                                         'updated_at': time.time()})

    def one(case):
        dest = out / 'cases' / case['id'] / 'styling'
        dest.mkdir(parents=True, exist_ok=True)
        original = dest / 'original.png'
        if not original.exists():
            shutil.copy2(case['original'], original)
        if sha(original) != case['source_sha256'] or sha(case['scene']) != case['scene_sha256']:
            raise ValueError('Frozen input changed: ' + case['id'])
        scene = read(case['scene'])
        assets = read(case['assets'])['assets'] if case['assets'] else []
        started = time.time()
        summary = run_case(scene, original, dest, runtime, opt, pool, renders, assets=assets,
                           pixel_slots=pixels, progress=lambda s, d, t: progress(case['id'], s, d, t), **extra)
        summary = {**summary, 'arm_wall_seconds': time.time() - started}
        save(dest / 'arm_case.json', summary)
        return summary

    # One shared model-task pool, as in the production runner; workers never block on it.
    with ThreadPoolExecutor(max_workers=opt['workers']) as pool, \
            ThreadPoolExecutor(max_workers=args.case_workers) as cases:
        futures = {cases.submit(one, c): c['id'] for c in inputs}
        for future in as_completed(futures):
            ident = futures[future]
            try:
                results[ident] = future.result()
            except Exception as exc:
                results[ident] = {'state': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()}
                print('CASE_FAILED', ident, exc, flush=True)
            print('CASE_DONE', ident, results[ident].get('state'), flush=True)
            save(out / 'summary.json', {'arm': args.arm, 'cases': results, 'finished': len(results),
                                        'total': len(inputs), 'api': runtime.slots.snapshot()})
    failed = [k for k, v in results.items() if v.get('state') == 'failed']
    save(out / 'completion.json', {'arm': args.arm, 'cases': len(inputs), 'failed': failed, 'completed_at': time.time()})
    return int(bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())
