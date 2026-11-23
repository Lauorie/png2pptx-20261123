"""Process-safe case slots shared by shape and text controllers."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time


class CaseSlots:
    def __init__(self, config, branch):
        self.config = config
        self.branch = branch
        if branch not in ('shape', 'text'):
            raise ValueError('Unknown branch')
        self.root = Path(config['root'])
        self.root.mkdir(parents=True, exist_ok=True)
        self.total = int(config['total'])
        self.shape = int(config['shape_reserved'])
        if not 0 < self.shape < self.total:
            raise ValueError('Both branches need an initial allocation')

    def available(self):
        other = 'text' if self.branch == 'shape' else 'shape'
        if (self.root / (other + '_drained.json')).exists():
            return range(self.total)
        return range(self.shape) if self.branch == 'shape' else range(self.shape, self.total)

    @contextmanager
    def acquire(self, case_id):
        handle = None
        try:
            while handle is None:
                for slot in self.available():
                    candidate = (self.root / f'{slot:02d}.lock').open('a+')
                    try:
                        fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        candidate.close()
                        continue
                    handle = candidate
                    handle.seek(0)
                    handle.truncate()
                    json.dump({'branch': self.branch, 'case': case_id,
                               'pid': os.getpid(), 'at': time.time()}, handle)
                    handle.flush()
                    break
                if handle is None:
                    time.sleep(0.2)
            yield
        finally:
            if handle is not None:
                handle.seek(0)
                handle.truncate()
                handle.flush()
                fcntl.flock(handle, fcntl.LOCK_UN)
                handle.close()

    def drained(self):
        path = self.root / (self.branch + '_drained.json')
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps({'at': time.time(), 'pid': os.getpid()}))
        temp.replace(path)
