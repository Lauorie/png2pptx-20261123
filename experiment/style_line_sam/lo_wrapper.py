#!/usr/bin/env python3
"""Run the preserved LibreOffice entirely on local disk, then publish the PDF."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
environment = {**os.environ, 'LD_LIBRARY_PATH': str(root / 'lib') + ':' + str(root / 'libreoffice/program'),
               'FONTCONFIG_FILE': str(root / 'fonts.conf'), 'SAL_USE_VCLPLUGIN': 'svp'}
argv = sys.argv[1:]
binary = str(root / 'libreoffice/program/soffice')
if '--convert-to' not in argv:
    raise SystemExit(subprocess.call([binary, *argv], env=environment))
out_index = argv.index('--outdir') + 1
destination = Path(argv[out_index])
destination.mkdir(parents=True, exist_ok=True)
source = Path(argv[-1])
with tempfile.TemporaryDirectory(prefix='style-lo-') as tmp:
    local = Path(tmp)
    shutil.copy2(source, local / source.name)
    argv[-1] = str(local / source.name)
    argv[out_index] = str(local)
    argv = [('-env:UserInstallation=' + (local / 'profile').as_uri())
            if a.startswith('-env:UserInstallation=') else a for a in argv]
    result = subprocess.run([binary, *argv], env=environment)
    pdf = local / (source.stem + '.pdf')
    if result.returncode == 0 and pdf.exists():
        shutil.copy2(pdf, destination / pdf.name)
    else:
        raise SystemExit(result.returncode or 1)
