#!/usr/bin/env python3
"""Portable exact-version renderer with bounded first-profile restarts."""
from pathlib import Path
try: Path('/proc/self/oom_score_adj').write_text('0')
except OSError: pass  # container forbids lowering the OOM score
import fcntl
import json
import os
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parent
    cfg = json.loads((root/'runtime.json').read_text())
    program, libc = Path(cfg['program']), Path(cfg['glibc'])
    if not program.is_absolute(): program=root/program
    if not libc.is_absolute(): libc=root/libc
    cfg['libraries']=cfg['libraries'].replace('{root}',str(root))
    libraries = ':'.join([str(libc), str(program), cfg['libraries'],
                          '/lib/x86_64-linux-gnu', '/usr/lib/x86_64-linux-gnu'])
    env = {**os.environ, 'LD_LIBRARY_PATH': libraries, 'SAL_USE_VCLPLUGIN': 'svp',
           'URE_BOOTSTRAP': 'vnd.sun.star.pathname:'+str(program/'fundamentalrc')}
    if not env.get('FONTCONFIG_FILE') and (root/'fonts.conf').is_file():
        env['FONTCONFIG_FILE'] = str(root/'fonts.conf')
    lock = None
    if env.get('PPT_RENDER_LOCK_DIR'):
        folder = Path(env['PPT_RENDER_LOCK_DIR']); folder.mkdir(parents=True, exist_ok=True)
        while lock is None:
            for index in range(2):
                handle = (folder/str(index)).open('a')
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB); lock = handle; break
                except BlockingIOError:
                    handle.close()
            if lock is None:
                time.sleep(.1)
    argv = [str(libc/'ld-linux-x86-64.so.2'), '--library-path', libraries,
            str(program/'soffice.bin'), *sys.argv[1:]]
    for attempt in range(3):
        code = subprocess.call(argv, env=env)
        if code != 81:
            return code
        print('LibreOffice requested restart (81); retaining the same profile, attempt '+str(attempt+1),
              file=sys.stderr, flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
