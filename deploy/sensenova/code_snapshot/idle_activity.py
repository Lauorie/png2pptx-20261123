"""Cross-process request priority and a cooldown after the last response.

Each GPU has its own activity lock so idle controllers can run concurrently.
Requests announce their arrival before waiting for a burst to drain, then hold
shared locks through the entire response. Process death releases these locks.
"""
from contextlib import contextmanager
import asyncio
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time


class IdleActivity:
    def __init__(self, run_dir, gpu_count=1, cooldown_seconds=600, clock=None):
        self.run = Path(run_dir)
        self.run.mkdir(parents=True, exist_ok=True)
        self.gpu_count = int(gpu_count)
        self.cooldown = float(cooldown_seconds)
        if self.gpu_count < 1 or self.cooldown < 0:
            raise ValueError('Invalid GPU count or idle cooldown')
        self.clock = clock or time.monotonic
        self.state_path = self.run / 'keepalive-activity.json'
        self.boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()

    def mark_request(self, event):
        with (self.run / 'keepalive-state.lock').open('a+') as owner:
            fcntl.flock(owner, fcntl.LOCK_EX)
            now = self.clock()
            prior = self.read_state()
            old_deadline = prior.get('resume_after_monotonic', 0) if prior.get('boot_id') == self.boot_id else 0
            deadline = max(old_deadline, now + self.cooldown)
            state = dict(event=event, boot_id=self.boot_id, updated_at=time.time(),
                         last_request_monotonic=now, resume_after_monotonic=deadline,
                         resume_after_unix=time.time() + max(0, deadline - now),
                         cooldown_seconds=self.cooldown)
            fd, temporary = tempfile.mkstemp(prefix='.keepalive-', dir=self.run)
            try:
                with os.fdopen(fd, 'w') as handle:
                    json.dump(state, handle)
                os.replace(temporary, self.state_path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return state

    def read_state(self):
        try:
            return json.loads(self.state_path.read_text())
        except FileNotFoundError:
            return {}

    def remaining(self):
        state = self.read_state()
        if not state:
            return 0.0
        if state.get('boot_id') == self.boot_id:
            return max(0.0, state['resume_after_monotonic'] - self.clock())
        return max(0.0, state['resume_after_unix'] - time.time())

    def announce_request(self):
        self.mark_request('request_arrived')
        return RequestGuard(self)

    @contextmanager
    def request(self):
        guard = self.announce_request()
        try:
            guard.acquire()
            yield guard
        finally:
            guard.finish()

    @contextmanager
    def idle_permission(self, gpu_index=0):
        owner = (self.run / f'keepalive-gpu-{gpu_index}.lock').open('a+')
        try:
            try:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield dict(allowed=False, phase='yielding_to_inference', cooldown_remaining_s=self.remaining())
                return
            remaining = self.remaining()
            yield dict(allowed=remaining <= 0, phase='running' if remaining <= 0 else 'cooldown',
                       cooldown_remaining_s=remaining)
        finally:
            owner.close()


class RequestGuard:
    def __init__(self, activity):
        self.activity = activity
        self.locks = []
        self.finished = False

    def acquire(self):
        for index in range(self.activity.gpu_count):
            owner = (self.activity.run / f'keepalive-gpu-{index}.lock').open('a+')
            self.locks.append(owner)
            fcntl.flock(owner, fcntl.LOCK_SH)

    async def acquire_async(self):
        for index in range(self.activity.gpu_count):
            owner = (self.activity.run / f'keepalive-gpu-{index}.lock').open('a+')
            self.locks.append(owner)
            while True:
                try:
                    fcntl.flock(owner, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    await asyncio.sleep(0.01)

    def finish(self):
        if self.finished:
            return
        self.finished = True
        try:
            self.activity.mark_request('request_finished')
        finally:
            for owner in reversed(self.locks):
                owner.close()
            self.locks.clear()
