"""Optional batch-wide file locks; no changes to single-case behavior."""
from contextlib import contextmanager
from functools import wraps
import fcntl
import os
from pathlib import Path
import threading
import time

_state=threading.local()


@contextmanager
def slot(kind):
    root=os.environ.get('SHAPE_BATCH_LIMIT_DIR')
    if not root:
        yield
        return
    key=(str(root),kind)
    held=getattr(_state,'held',set())
    if key in held:
        yield
        return
    count=int(os.environ.get('SHAPE_'+kind.upper()+'_SLOTS','2'))
    if count<1:raise ValueError('Resource slots must be positive')
    folder=Path(root);folder.mkdir(parents=True,exist_ok=True)
    handle=None
    try:
        while handle is None:
            for i in range(count):
                candidate=(folder/f'{kind}_{i:02d}.lock').open('a+b')
                try:fcntl.flock(candidate.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    candidate.close();continue
                handle=candidate;break
            if handle is None:time.sleep(.1)
        _state.held=held|{key}
        yield
    finally:
        _state.held=held
        if handle is not None:
            fcntl.flock(handle.fileno(),fcntl.LOCK_UN);handle.close()


def bounded(kind):
    def decorate(function):
        @wraps(function)
        def wrapped(*args,**kwargs):
            with slot(kind):return function(*args,**kwargs)
        return wrapped
    return decorate
