#!/usr/bin/env python3
"""Keep native renderer concurrency bounded across all isolated stages."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(os.environ['PPT_RENDER_LOCK_DIR']);root.mkdir(parents=True,exist_ok=True)
while True:
    for index in range(2):
        handle=(root/str(index)).open('a')
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:handle.close();continue
        # The outer process owns this slot. Portable renderers may implement
        # the same optional guard; do not make them acquire a second slot.
        environment=dict(os.environ);environment.pop('PPT_RENDER_LOCK_DIR',None)
        raise SystemExit(subprocess.call([os.environ['PPT_LIBREOFFICE_RAW'],*sys.argv[1:]],env=environment))
    time.sleep(.1)
