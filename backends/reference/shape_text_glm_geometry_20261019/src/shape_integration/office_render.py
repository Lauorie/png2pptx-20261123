"""Isolated local LibreOffice conversion, with shared-storage deliverables."""
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from .common import save


def convert_pdf(source, output, timeout, log_name='libreoffice.log'):
    source = Path(source).resolve()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    env = {k: v for k, v in os.environ.items()
           if k not in ('LD_LIBRARY_PATH', 'LD_PRELOAD')}
    with tempfile.TemporaryDirectory(prefix='shape_office_', dir='/tmp') as scratch:
        local = Path(scratch)
        document = local / source.name
        shutil.copy2(source, document)
        command = ['/usr/bin/libreoffice',
                   '-env:UserInstallation=' + (local / 'profile').as_uri(),
                   '--headless', '--convert-to', 'pdf', '--outdir', str(local), str(document)]
        conversion_started = time.monotonic()
        process = subprocess.Popen(command, cwd=local, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, start_new_session=True)
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
        seconds = time.monotonic() - conversion_started
        log = stdout + stderr + ('\nconversion exit timed out\n' if timed_out else '')
        (output / log_name).write_text(log)
        pdf = local / (source.stem + '.pdf')
        target = output / pdf.name
        # Only a PDF produced in this fresh directory is ever published.
        published = pdf.is_file() and pdf.stat().st_size > 0
        if published:
            staging = output / (pdf.name + '.tmp')
            shutil.copy2(pdf, staging)
            staging.replace(target)
        result = {'returncode': process.returncode, 'timed_out': timed_out,
                  'pdf_published': published, 'conversion_seconds': seconds,
                  'total_seconds': time.monotonic() - started,
                  'policy': 'local_tmp_clean_system_libraries_v1'}
        save(output / 'conversion.json', result)
        return {**result, 'log': log, 'pdf': target}
