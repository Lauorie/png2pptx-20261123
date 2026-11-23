"""Unattended host-side driver for one prepared run directory.

Waits for the rjob services, smoke-tests them from this host, then runs
  SAM tiles (main config)  ||  arm `original`
  -> release the SAM GPUs for a second GLM replica
  -> arms `v2_nosam` and `v2` (frozen code, same inputs, same GLM transport)
  -> model-free pixel evaluation + blind pairwise GLM review for both comparisons
and finally writes STOP.json so the GPUs are released. Every stage is
checkpointed by its own outputs, so rerunning the driver continues the run.
While alive it refreshes activity.json; the service host frees the GPUs itself
if that heartbeat stops for idle_stop_seconds.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path[:0] = [str(HERE)]
from clients import read, save, post, services, OPENER  # noqa: E402

PYTHON = '/mnt/shared-storage-user/kouyiqun/conda/envs/SVG/bin/python'
ENV = {**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1',
       'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
       'PYTHONPATH': str(PROJECT / '.runtime/shape_layers_py311')}
for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
    ENV.pop(key, None)
MAIN_SAM = 'side96_both'


class Driver:
    def __init__(self, run, args):
        self.run, self.args = run, args
        self.state = {'stage': 'init', 'events': [], 'arms': {}}
        self.stop_beat = threading.Event()
        self.lock = threading.Lock()

    # ---- bookkeeping
    def log(self, message, **fields):
        row = {'at': time.time(), 'message': message, **fields}
        with self.lock:
            self.state['events'].append(row)
            save(self.run / 'driver_state.json', self.state)
        print(time.strftime('%H:%M:%S'), message, fields or '', flush=True)

    def stage(self, name):
        with self.lock:
            self.state['stage'] = name
        self.log('stage ' + name)

    def beat(self):
        while not self.stop_beat.wait(30):
            save(self.run / 'activity.json', {'at': time.time(), 'stage': self.state['stage']})

    # ---- services
    def wait_ready(self):
        submitted = float((self.run / 'jobs/submitted_epoch.txt').read_text()) if (self.run / 'jobs/submitted_epoch.txt').exists() else 0
        deadline = time.time() + self.args.queue_timeout
        while time.time() < deadline:
            path = self.run / 'status.json'
            if path.exists():
                status = read(path)
                if status.get('started_at', 0) >= submitted and status.get('phase') == 'failed':
                    raise RuntimeError('Service host failed: ' + str(status.get('error')))
                try:
                    if status.get('started_at', 0) >= submitted:
                        glm, sam = services(self.run, 'glm'), services(self.run, 'sam')
                        return status, glm, sam
                except RuntimeError:
                    pass
            time.sleep(15)
        raise TimeoutError('Services not ready within the queue timeout')

    def smoke(self, glm, sam):
        out = {'glm': [], 'sam': []}
        for url in glm:
            body = {'model': 'GLM-5.3-Flash', 'messages': [{'role': 'user', 'content': 'Reply with the number 2.'}],
                    'temperature': 0, 'max_tokens': 64, 'chat_template_kwargs': {'reasoning_effort': 'low', 'clear_thinking': True}}
            response = post(url + '/chat/completions', body, 180)
            text = response['choices'][0]['message'].get('content') or ''
            out['glm'].append({'url': url, 'answer': text[:80]})
            if not text.strip():
                raise RuntimeError('GLM smoke returned no content: ' + url)
        for url in sam:
            with OPENER.open(url + '/', timeout=30) as reply:
                out['sam'].append({'url': url, 'status': reply.status})
        save(self.run / 'smoke.json', out)
        return out

    # ---- subprocesses
    def spawn(self, name, argv):
        log = (self.run / 'driver_logs' / (name + '.log')).open('a', buffering=1)
        proc = subprocess.Popen([PYTHON, '-B', '-u', *argv], env=ENV, stdout=log, stderr=subprocess.STDOUT, cwd=str(HERE))
        self.log('started ' + name, pid=proc.pid)
        return proc

    def arm_args(self, arm):
        extra = ['--ids', *self.args.ids] if self.args.ids else []
        return [str(HERE / 'run_arm.py'), '--run-dir', str(self.run), '--arm', arm, *extra]

    def write_arm(self, name, spec):
        save(self.run / 'arms' / (name + '.json'), spec)

    def freeze_v2(self):
        dest = self.run / 'code' / 'v2'
        if not dest.exists():
            subprocess.run(['bash', str(HERE / 'freeze.sh'), str(self.run), 'v2'], check=True)
        runtime = {'concurrency': 16, 'max_tokens': 20000, 'timeout_seconds': 1200, 'seed': 42, 'reasoning_effort': 'high'}
        styling = {'max_tokens': 20000, 'pixel_workers': 2, 'render_workers': 2}
        self.write_arm('v2_nosam', {'description': 'v2 code, contour line measurement, no SAM evidence.', 'code': 'code/v2/src',
                                    'styling_overrides': styling, 'runtime': runtime, 'case_workers': 27})
        self.write_arm('v2', {'description': 'v2 code, contour line measurement + SAM tile line ownership.', 'code': 'code/v2/src',
                              'styling_overrides': styling, 'runtime': runtime, 'case_workers': 27,
                              'line_evidence': f'sam_lines/{MAIN_SAM}'})

    def finished(self, arm):
        path = self.run / 'arms' / arm / 'completion.json'
        return path.exists() and not read(path)['failed']

    # ---- main
    def go(self):
        run = self.run
        (run / 'driver_logs').mkdir(exist_ok=True)
        save(run / 'activity.json', {'at': time.time(), 'stage': 'start'})
        threading.Thread(target=self.beat, daemon=True).start()
        self.stage('wait_services')
        status, glm, sam = self.wait_ready()
        self.log('services ready', ip=status['ip'], glm=glm, sam=sam)
        self.stage('smoke')
        try:
            self.smoke(glm, sam)
        except Exception as exc:
            self.log('smoke failed; keeping services for a bounded debug window', error=str(exc))
            time.sleep(self.args.debug_window)
            raise
        ids = ['--ids', *self.args.ids] if self.args.ids else []
        procs = {}
        self.stage('arm_original+sam')
        if not self.finished('original'):
            procs['original'] = self.spawn('arm_original', self.arm_args('original'))
        if not (run / 'sam_lines' / MAIN_SAM / 'completion.json').exists():
            procs['sam_main'] = self.spawn('sam_main', [str(HERE / 'sam_tiles.py'), '--run-dir', str(run), '--config', MAIN_SAM, *ids])
        if 'sam_main' in procs:
            code = procs['sam_main'].wait()
            self.log('sam_main exit', code=code)
            if code:
                raise RuntimeError('SAM tile precompute failed; see driver_logs/sam_main.log')
        # Diagnostic alternatives on a seeded subset (informational; the main arm is pre-declared).
        for name in self.args.pilot_configs:
            if not (run / 'sam_lines' / name / 'completion.json').exists():
                code = self.spawn('sam_' + name, [str(HERE / 'sam_tiles.py'), '--run-dir', str(run), '--config', name,
                                                  *(ids or ['--pilot', str(self.args.pilot)])]).wait()
                self.log('sam diagnostic exit', config=name, code=code)
        self.stage('release_sam_and_arms')
        save(run / 'control.json', {'release_sam': True, 'at': time.time()})
        self.freeze_v2()
        for arm in ('v2_nosam', 'v2'):
            if not self.finished(arm):
                procs[arm] = self.spawn('arm_' + arm, self.arm_args(arm))
        for name, proc in procs.items():
            if name.startswith('sam'):
                continue
            code = proc.wait()
            self.state['arms'][name] = code
            self.log('arm exit', arm=name, code=code)
        self.stage('evaluate')
        for pair in (('original', 'v2'), ('v2_nosam', 'v2')):
            if all((run / 'arms' / a / 'summary.json').exists() for a in pair):
                code = self.spawn('evaluate_' + '_vs_'.join(pair), [str(HERE / 'evaluate.py'), '--run-dir', str(run), '--arms', *pair, *ids]).wait()
                self.log('evaluate exit', pair=pair, code=code)
        self.stage('done')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--ids', nargs='*')
    parser.add_argument('--queue-timeout', type=float, default=6 * 3600)
    parser.add_argument('--debug-window', type=float, default=2700)
    parser.add_argument('--pilot', type=int, default=5)
    parser.add_argument('--pilot-configs', nargs='*', default=['side96_line', 'side192_line'])
    args = parser.parse_args()
    from clients import require_cluster
    require_cluster()
    run = args.run_dir.resolve()
    driver = Driver(run, args)
    try:
        driver.go()
        save(run / 'driver_done.json', {'state': 'completed', 'at': time.time()})
    except BaseException as exc:
        driver.log('driver failed', error=f'{type(exc).__name__}: {exc}')
        save(run / 'driver_done.json', {'state': 'failed', 'error': str(exc), 'at': time.time()})
        raise
    finally:
        driver.stop_beat.set()
        if (run / 'status.json').exists():
            # Release the GPUs whether the run completed or failed; the idle watchdog is the backstop.
            save(run / 'STOP.json', {'reason': 'driver_finished', 'at': time.time(), 'state': driver.state['stage']})


if __name__ == '__main__':
    main()
