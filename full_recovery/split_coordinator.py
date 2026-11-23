"""Lightweight cluster submission supervisor; all generation runs in rjobs.

Wait for currently occupied case slots, without interrupting their owners.
Holding each released slot prevents that capacity from taking further cases.
Only after every slot is owned here is the old controller stopped. New case
views then have disjoint writers and retain canonical checkpoint locations.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from .atomic_json import save
from .prepare import read, sha
from .partition_resume import validate_partition, prepare_partition, all_shape_products, merge_collection


def acquire_released_slots(root, total, cancel, progress):
    """Return open locked files only after all old case writers have drained."""
    handles = [None] * total
    errors = []
    mutex = threading.Lock()

    def wait_one(index):
        try:
            handle = (Path(root) / f'{index:02d}.lock').open('r')
            fcntl.flock(handle, fcntl.LOCK_EX)
            with mutex:
                handles[index] = handle
                progress(sum(h is not None for h in handles))
        except Exception as exc:
            with mutex: errors.append(exc)

    threads = [threading.Thread(target=wait_one, args=(i,), daemon=True) for i in range(total)]
    for thread in threads: thread.start()
    heartbeat = time.monotonic()
    while not all(handle is not None for handle in handles):
        if errors: raise errors[0]
        if cancel.wait(1): raise InterruptedError('Coordinator cancelled during case drain')
        if time.monotonic() - heartbeat >= 15:
            with mutex:
                progress(sum(h is not None for h in handles))
            heartbeat = time.monotonic()
    return handles


def cluster_command(argv, log):
    result = subprocess.run(argv, text=True, capture_output=True, timeout=60)
    with Path(log).open('a') as stream:
        stream.write(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError('Cluster command failed; inspect ' + str(log))
    return result.stdout + result.stderr


def cluster_state(job, namespace, log):
    out = cluster_command(['rjob', 'get', '--namespace', namespace, job], log)
    match = re.search(r'\(showname=([^)]*)\):\s*(\w+)', out)
    if not match or not match[1]:
        raise RuntimeError('Cluster did not report a nonempty showname')
    ip = re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', out)
    host = re.search(r'([\w.-]+\.host\.h\.pjlab\.org\.cn)', out)
    return {'state': match[2], 'showname': match[1],
            'pod_ip': ip[0] if ip else None, 'host': host[1] if host else None}


def submit_worker(run, label, external):
    run = Path(run)
    if (run / 'jobs/job.id').exists():
        return (run / 'jobs/job.id').read_text().strip()
    script = run / 'jobs/submit.sh'
    env = ('ENV_ARGS=()\nfor name in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do\n'
           '  [[ -n "${!name:-}" ]] && ENV_ARGS+=(-e "${name}=${!name}")\ndone\n') if external else 'ENV_ARGS=()\n'
    remote = ['env']
    if not external:
        for name in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY','http_proxy','https_proxy','all_proxy','no_proxy'):
            remote += ['-u', name]
    entry = run / 'code/partition_worker.py'
    if not entry.exists(): entry = run / 'code/full_recovery/api_worker.py'
    remote += ['SSL_CERT_FILE=' + ('/etc/ssl/certs/ca-certificates.crt' if external else str(run / 'private/cpa-worker.crt')),
               '/opt/shape-cpu/bin/python', '-B', '-u', str(entry), '--run-dir', str(run)]
    script.write_text('#!/usr/bin/env bash\nset -euo pipefail\necho 0 > /proc/self/oom_score_adj\n'
        'RJOB_ARGS=(submit --name=' + shlex.quote(label + '-' + time.strftime('%Y%m%d-%H%M%S')) + '\n'
        ' --namespace ailab-intern10 --gpu=0 --cpu=16 --memory=65536\n'
        ' --charged-group=intern10_proxy --private-machine=group -P 1\n'
        ' --share-host-shm=true --gang-start=true\n'
        ' --image registry.h.pjlab.org.cn/ailab-intern7-intern7_gpu/tianchangyao-workspace:v6\n'
        ' --mount=gpfs://gpfs1/intern7shared:/mnt/shared-storage-user/intern7shared\n'
        ' --mount=gpfs://gpfs1/kouyiqun/conda/envs/SVG:/opt/shape-cpu)\n' + env +
        "printf -v REMOTE_COMMAND '%q ' " + ' '.join(shlex.quote(x) for x in remote) + '\n'
        'rjob "${RJOB_ARGS[@]}" "${ENV_ARGS[@]}" -- bash -lc "${REMOTE_COMMAND}"\n')
    out = cluster_command(['bash', str(script)], run / 'jobs/submit.log')
    match = re.search(r'created[ _]rjob_name["\x27]?\s*[:=]\s*["\x27]?([A-Za-z0-9_.-]+)', out)
    if not match:
        raise RuntimeError('Submission did not return a job ID; do not retry blindly')
    (run / 'jobs/job.id').write_text(match[1] + '\n')
    return match[1]


def validate_running(run, cancel):
    run = Path(run); job = (run / 'jobs/job.id').read_text().strip()
    start = time.time()
    prior = None
    while time.time() - start < 1800:
        state = cluster_state(job, 'ailab-intern10', run / 'jobs/startup_checks.log')
        save(run / 'cluster_status.json', {**state, 'at': time.time()})
        if state['state'] in ('Failed', 'Stopped'):
            raise RuntimeError('Worker startup ended in ' + state['state'])
        if state['state'] in ('Running', 'Succeeded'):
            cluster_command(['rjob', 'events', '--namespace', 'ailab-intern10', job], run / 'jobs/startup_events.log')
            cluster_command(['rjob', 'logs', 'job', job, '--namespace', 'ailab-intern10', '-n', '10'], run / 'jobs/startup_logs.log')
            metrics = read(run / 'api_metrics.json') if (run / 'api_metrics.json').exists() else {}
            status = read(run / 'status.json') if (run / 'status.json').exists() else {}
            if status.get('state') in ('failed', 'paused_budget'):
                raise RuntimeError('Worker preflight failed; inspect ' + str(run / 'status.json'))
            if state['host'] and state['pod_ip'] and metrics.get('completed', 0) >= 1:
                if prior is not None and (metrics.get('upstream_events',0) > prior.get('upstream_events',0)
                        or metrics.get('completed',0) > prior.get('completed',0)
                        or status.get('state') == 'completed'):
                    save(run / 'startup_verified.json', {'cluster': state, 'api': metrics, 'status': status, 'at': time.time()})
                    return
                prior = metrics
        if cancel.wait(10): raise InterruptedError('Coordinator cancelled')
    raise TimeoutError('Worker did not start within 30 minutes')


def prepare_merge(plan):
    from .merge_existing import prepare as prepare_composition
    root = Path(plan['project']); run = Path(plan['merge_run'])
    collection = merge_collection(plan, Path(plan['report']) / 'merge_sources59')
    arrow_root = Path(read(Path(plan['source_run']) / 'external_arrow_inputs.json')['root'])
    args = SimpleNamespace(run_dir=run, shape_run=collection, text_run=collection / 'texts',
        arrow_run=arrow_root, ids=None,
        case_workers=int(plan.get('merge_case_workers', 24)),
        api_concurrency=int(plan.get('merge_api_concurrency', plan.get('discovery_api_concurrency', 96))),
        require_all=True, resume_from=None, reuse_assets_from=None, image_edit_url=None,
        runtime_source=Path(plan['previous_run']), api_key_file=[Path(plan['discovery_key_file'])],
        api_base_url=plan['discovery_base_url'], text_source=None)
    prepare_composition(root, args)
    settings = read(run / 'settings.json')
    settings.update(api_model=plan['discovery_model'], api_force_reasoning_effort='high',
                    api_default_image_detail='auto', api_budget_config=None,
                    composition_enabled=True, partition_coordinator=plan['report'],
                    **plan.get('discovery_rate_settings', {}))
    save(run / 'settings.json', settings)
    manifest = read(run / 'code_manifest.json')
    manifest.update({name: sha(run / name) for name in ('settings.json', 'sources.json', 'source_manifest.json')})
    save(run / 'code_manifest.json', manifest)
    return run


def main():
    Path('/proc/self/oom_score_adj').write_text('0')
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(); plan = read(args.plan); report = Path(plan['report'])
    preflight = read(report / 'probe_v2/preflight_status.json')
    if preflight.get('state') != 'completed' or preflight.get('model') != plan['discovery_model']:
        raise ValueError('Require the successful cluster vision/provider preflight before submission')
    for relative, digest in plan.get('coordinator_code_sha256', {}).items():
        if sha(Path(plan['project']) / relative) != digest:
            raise ValueError('Coordinator implementation changed: ' + relative)
    cancel = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: cancel.set())
    signal.signal(signal.SIGINT, lambda *_: cancel.set())
    owner = (report / 'coordinator.lock').open('a'); fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = report / 'coordinator_state.json'
    state = read(state_path) if state_path.exists() else {'phase': 'draining_active_cases'}
    state.pop('error', None)
    state_mutex = threading.RLock()
    def stamp(phase, **kwargs):
        with state_mutex:
            state.update({'phase':phase, 'at':time.time(), 'pid':os.getpid(),
                          'supervisor_status':'running', **kwargs})
            if phase == 'completed': state['supervisor_status'] = 'completed'
            save(state_path, state)
            save(Path(plan['project']) / 'runs/LATEST_TERRA59_SCHEDULE.json', {
                'at': state['at'], 'phase': phase, 'supervisor_status': state['supervisor_status'],
                'state_file': str(state_path), 'plan': str(args.plan.resolve()),
                'previous_run': plan['previous_run'],
                'shape_runs': {role: plan[role + '_run'] for role in ('cpa', 'discovery')},
                'shape_jobs': state.get('jobs', {}), 'merge_run': plan['merge_run'],
                'merge_job': state.get('merge_job'), 'merge_expected': plan['merge_expected']})
    validate_partition(plan['cpa_ids'] + plan['discovery_ids'],
                       [plan['cpa_ids'], plan['discovery_ids']], plan['protected_active_ids'])
    previous = Path(plan['previous_run']); source = Path(plan['source_run'])
    handles = []
    try:
        if state['phase'] in ('draining_active_cases', 'stopping_drained_controller'):
            if not plan.get('cross_host_case_locks_verified', False):
                raise RuntimeError('Local flock does not prove cluster case ownership; refuse an unverified drain/stop')
        if state['phase'] == 'draining_active_cases':
            settings = read(previous / 'settings.json'); slots = settings['case_slot_config']
            stamp('draining_active_cases', total_slots=slots['total'], acquired_slots=0,
                  current_cases_not_interrupted=True)
            handles = acquire_released_slots(slots['root'], slots['total'], cancel,
                lambda count: stamp('draining_active_cases', acquired_slots=count))
            stamp('stopping_drained_controller')
        if state['phase'] == 'stopping_drained_controller':
            # A restarted supervisor lost its locks. Drain again before any
            # stop request so a newly occupied case is never interrupted.
            if not handles:
                slots = read(previous / 'settings.json')['case_slot_config']
                handles = acquire_released_slots(slots['root'], slots['total'], cancel,
                    lambda count: stamp('stopping_drained_controller', acquired_slots=count))
            cluster_command(['rjob','stop','--namespace','ailab-intern10',plan['previous_job']], report/'old_stop.log')
            for _ in range(30):
                status = cluster_state(plan['previous_job'],'ailab-intern10',report/'old_stop.log')
                if status['state'] in ('Stopped','Succeeded','Failed'): break
                if cancel.wait(2): raise InterruptedError('Coordinator cancelled')
            else: raise RuntimeError('Old controller did not stop')
            for handle in handles: handle.close()
            handles = []
            with ExitStack() as locks:
                for name in ('worker.lock','fresh_branches.lock','shape/.pipeline.lock'):
                    f = locks.enter_context((source/name).open('r')); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            stamp('preparing_partitions')
        if state['phase'] == 'preparing_partitions':
            for relative, digest in plan.get('coordinator_code_sha256', {}).items():
                if sha(Path(plan['project']) / relative) != digest:
                    raise ValueError('Frozen preparation code changed: ' + relative)
            for role in ('cpa','discovery'):
                run = Path(plan[role+'_run'])
                if not (run/'partition_prepared.json').exists():
                    if run.exists(): raise RuntimeError('Partial preparation requires inspection: '+str(run))
                    prepare_partition(plan,role);save(run/'partition_prepared.json',{'at':time.time()})
            stamp('starting_partitions')
        with ExitStack() as locks:
            # Retain canonical ownership while disjoint per-case views run.
            for name in ('worker.lock','fresh_branches.lock','shape/.pipeline.lock'):
                f = locks.enter_context((source/name).open('r'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if state['phase'] == 'starting_partitions':
                jobs={}
                for role in ('cpa','discovery'):
                    jobs[role]=submit_worker(plan[role+'_run'],'terra59-'+role+'-split',role=='discovery')
                stamp('validating_partitions',jobs=jobs)
            if state['phase'] == 'validating_partitions':
                for role in ('cpa','discovery'):validate_running(plan[role+'_run'],cancel)
                stamp('running_partitions')
            if state['phase'] == 'running_partitions':
                last_cluster_check = 0
                while True:
                    completed=[]
                    for role in ('cpa','discovery'):
                        run=Path(plan[role+'_run']);status=read(run/'status.json')
                        if status.get('state') in ('failed','stopped','paused_budget'):
                            raise RuntimeError(role+' worker stopped; checkpoints retained')
                        completed.append(status.get('state')=='completed')
                        if time.time() - last_cluster_check >= 60 and status.get('state') != 'completed':
                            job=(run/'jobs/job.id').read_text().strip()
                            cluster=cluster_state(job,'ailab-intern10',run/'jobs/monitor.log')
                            if cluster['state'] in ('Failed','Stopped','Succeeded'):
                                raise RuntimeError(role+' cluster ended before a completion receipt')
                    if time.time() - last_cluster_check >= 60: last_cluster_check = time.time()
                    stamp('running_partitions',shape_complete_count=sum(
                        (source/'shape/geometry/cases'/cid/'complete.json').is_file()
                        for cid in plan['cpa_ids']+plan['discovery_ids']))
                    if all(completed):
                        if all_shape_products(plan) is None:
                            raise RuntimeError('Workers completed but shape receipts are missing')
                        break
                    if cancel.wait(10):raise InterruptedError('Coordinator cancelled')
                stamp('preparing_merge')
            if state['phase'] == 'preparing_merge':
                for relative, digest in plan.get('coordinator_code_sha256', {}).items():
                    if sha(Path(plan['project']) / relative) != digest:
                        raise ValueError('Frozen composition code changed: ' + relative)
                cluster_command(['rjob','stop','--namespace','ailab-llmmultimodal',plan['sam_job_id']],report/'sam_stop_after_shapes.log')
                run=Path(plan['merge_run'])
                if not (run/'merge_prepared.json').exists():
                    if run.exists():raise RuntimeError('Partial merge preparation requires inspection')
                    prepare_merge(plan);save(run/'merge_prepared.json',{'at':time.time()})
                stamp('starting_merge')
            if state['phase'] == 'starting_merge':
                job=submit_worker(plan['merge_run'],'terra59-discovery-merge',True)
                stamp('validating_merge',merge_job=job)
            if state['phase'] == 'validating_merge':
                validate_running(plan['merge_run'],cancel);stamp('merging')
            if state['phase'] == 'merging':
                run=Path(plan['merge_run'])
                last_cluster_check = 0
                while True:
                    status=read(run/'status.json');stamp('merging',merge_status=status)
                    if status.get('state')=='completed':
                        summary=read(run/'summary.json')
                        if summary['exported']!=plan['merge_expected']:raise RuntimeError('Merge count mismatch')
                        stamp('completed',merged=summary['exported']);break
                    if status.get('state') in ('failed','stopped','paused_budget'):raise RuntimeError('Merge stopped; checkpoints retained')
                    if time.time() - last_cluster_check >= 60:
                        job=(run/'jobs/job.id').read_text().strip()
                        cluster=cluster_state(job,'ailab-intern10',run/'jobs/monitor.log')
                        if cluster['state'] in ('Failed','Stopped','Succeeded'):
                            raise RuntimeError('Merge cluster ended before a completion receipt')
                        last_cluster_check = time.time()
                    if cancel.wait(10):raise InterruptedError('Coordinator cancelled')
    except BaseException as exc:
        stamp(state['phase'], supervisor_status='interrupted' if isinstance(exc,InterruptedError) else 'failed',
              error=str(exc),checkpoints_retained=True)
        raise
    finally:
        for handle in handles:handle.close()


if __name__=='__main__':main()
