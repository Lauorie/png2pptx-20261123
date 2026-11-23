"""Yield idle GPU work at POST arrival and for ten minutes after completion."""
import asyncio
import fcntl
import json
import os
from pathlib import Path
import time
from idle_activity import IdleActivity


class GpuWorkLockMiddleware:
    def __init__(self, app):
        self.app = app
        self.run = Path(os.environ['SENSENOVA_RUN_DIR'])
        config = json.loads((self.run / 'config.json').read_text())
        self.gpu_count = int(config['gpu_count'])
        self.activity = IdleActivity(self.run, self.gpu_count, config.get('idle_resume_seconds', 600))

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('method') != 'POST':
            return await self.app(scope, receive, send)
        activity = self.activity.announce_request()
        locks = []
        started = time.time()
        try:
            await activity.acquire_async()
            for index in range(self.gpu_count):
                lock = (self.run / 'power' / str(index) / 'gpu-work.lock').open('a')
                locks.append(lock)
                while True:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        await asyncio.sleep(0.01)
            self.record('acquired', scope.get('path'), started)
            await self.app(scope, receive, send)
        finally:
            try:
                activity.finish()
            finally:
                for lock in reversed(locks):
                    lock.close()
                self.record('released', scope.get('path'), started)

    def record(self, event, path, started):
        with (self.run / 'evidence' / 'inference-lock.jsonl').open('a') as log:
            log.write(json.dumps(dict(event=event, path=path, time=time.time(), started=started)) + '\n')
