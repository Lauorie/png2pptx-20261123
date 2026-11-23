#!/usr/bin/env python3
"""GPU idle work: target 60% measured power; yield to requests and cooldown."""
from __future__ import annotations
import argparse
import fcntl
from pathlib import Path
import signal
import subprocess
import threading
import time
import traceback
from engine import save
from idle_activity import IdleActivity
from low_power_policy import LowPowerPolicy


def read_power(gpu_uuid):
    result = subprocess.run(['nvidia-smi', '-i', gpu_uuid,
        '--query-gpu=power.draw,enforced.power.limit', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True, timeout=10)
    rows = result.stdout.strip().splitlines()
    if len(rows) != 1:
        raise RuntimeError('Expected telemetry for exactly one GPU UUID')
    watts, limit = [float(value.strip()) for value in rows[0].split(',')]
    if watts < 0 or limit <= 0:
        raise ValueError('Invalid GPU power telemetry')
    return watts, limit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--power-pct', type=float, default=60)
    parser.add_argument('--idle-resume-seconds', type=float, default=600)
    parser.add_argument('--low-power-threshold-pct', type=float, default=40)
    parser.add_argument('--activity-dir', type=Path)
    parser.add_argument('--activity-gpu-index', type=int, default=0)
    parser.add_argument('--matrix-size', type=int, default=8192)
    parser.add_argument('--burst', type=int, default=10)
    args = parser.parse_args()
    if not 0 < args.low_power_threshold_pct < args.power_pct <= 100 or args.idle_resume_seconds < 0:
        parser.error('Invalid power target or cooldown')
    if args.matrix_size <= 0 or args.burst <= 0:
        parser.error('--matrix-size and --burst must be positive')
    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.set())
    activity = IdleActivity(args.activity_dir or args.run_dir, cooldown_seconds=args.idle_resume_seconds)
    state_path = args.run_dir / 'power-state.json'
    state = dict(phase='loading', target_pct=args.power_pct, cooldown_seconds=args.idle_resume_seconds)
    save(state_path, state)
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Keepalive requires exactly one allocated CUDA GPU')
        gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
        if not gpu_uuid.startswith('GPU-'):
            gpu_uuid = 'GPU-' + gpu_uuid
        watts, limit = read_power(gpu_uuid)
        state.update(gpu_uuid=gpu_uuid, power_limit_w=limit, target_w=limit * args.power_pct / 100)
        matrix_args = {'device': 'cuda:0', 'dtype': torch.float16}
        a = torch.randn(args.matrix_size, args.matrix_size, **matrix_args)
        b = torch.randn(args.matrix_size, args.matrix_size, **matrix_args)
        c = torch.empty_like(a)
        sleep_s, last_control, last_report = 0.02, 0.0, 0.0
        policy = LowPowerPolicy(args.low_power_threshold_pct, args.idle_resume_seconds)
        with (args.run_dir / 'gpu-work.lock').open('a') as gpu_lock, torch.inference_mode():
            while not stopped.is_set():
                now = time.monotonic()
                if now - last_control >= 0.5:
                    watts, limit = read_power(gpu_uuid)
                    target = limit * args.power_pct / 100
                    if policy.running:
                        sleep_s = min(2.0, max(0.0, sleep_s + 0.0003 * (watts - target)))
                    state.update(power_w=watts, power_limit_w=limit, target_w=target,
                                 measured_pct=watts / limit * 100, sleep_s=sleep_s, updated_at=time.time())
                    last_control = now
                wait_s = 0.1
                with activity.idle_permission(args.activity_gpu_index) as permission:
                    record = activity.read_state()
                    revision = (record.get('boot_id'), record.get('last_request_monotonic'), record.get('event'))
                    request_active = permission['phase'] == 'yielding_to_inference'
                    work_owned = False
                    if not request_active:
                        try:
                            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            work_owned = True
                        except BlockingIOError:
                            pass
                    try:
                        status = policy.observe(watts / limit * 100, revision,
                            request_active=request_active,
                            startup_blocked=not request_active and not work_owned,
                            can_activate=permission['allowed'])
                        state.update(status, cooldown_remaining_s=permission['cooldown_remaining_s'])
                        if policy.running and permission['allowed'] and work_owned:
                            for _ in range(args.burst):
                                if stopped.is_set() or activity.remaining() > 0:
                                    policy.running = False
                                    policy.low_since = None
                                    state.update(phase='yielding_to_inference', idle_load_enabled=False)
                                    break
                                torch.matmul(a, b, out=c)
                                torch.cuda.synchronize()
                            wait_s = sleep_s
                    finally:
                        if work_owned:
                            fcntl.flock(gpu_lock, fcntl.LOCK_UN)
                if now - last_report >= 5:
                    save(state_path, state)
                    print(f"GPU {gpu_uuid}: {watts:.1f}/{state['target_w']:.1f} W "
                          f"target={args.power_pct:g}% phase={state['phase']} "
                          f"below_{args.low_power_threshold_pct:g}_pct="
                          f"{state.get('low_power_elapsed_seconds', 0):.1f}/{args.idle_resume_seconds:g}s", flush=True)
                    last_report = now
                stopped.wait(wait_s)
    except Exception:
        state.update(phase='failed', error=traceback.format_exc(), updated_at=time.time())
        save(state_path, state)
        raise
    finally:
        if state['phase'] != 'failed':
            state.update(phase='stopped', updated_at=time.time())
            save(state_path, state)


if __name__ == '__main__':
    main()
