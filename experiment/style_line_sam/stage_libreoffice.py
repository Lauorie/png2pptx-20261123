"""Stage the existing renderer and its shared libraries; no installation/download."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=True)
source = Path('/usr/lib/libreoffice')
shutil.copytree(source, root / 'libreoffice', symlinks=False, dirs_exist_ok=True,
                ignore_dangling_symlinks=True)
shutil.copytree('/etc/libreoffice', root / 'etc', symlinks=False, dirs_exist_ok=True,
                ignore_dangling_symlinks=True)
for tree in (root / 'libreoffice', root / 'etc'):
    for path in tree.rglob('*'):
        if not path.is_file() or path.suffix not in ('.rc', '.rdb', '.xml', '.xcd', '.xcu', '.ini') and not path.name.endswith('rc'):
            continue
        data = path.read_bytes()
        if data.startswith(b'\x7fELF'):
            continue
        patched = data.replace(b'/usr/lib/libreoffice', str(root / 'libreoffice').encode()).replace(b'/etc/libreoffice', str(root / 'etc').encode())
        if patched != data:
            path.write_bytes(patched)
libs = root / 'lib'
libs.mkdir(exist_ok=True)
dependencies = set()
for binary in (source / 'program').glob('*'):
    if not binary.is_file() or ('.so' not in binary.name and binary.name != 'soffice.bin'):
        continue
    result = subprocess.run(['ldd', str(binary)], capture_output=True, text=True)
    for item in re.findall(r'=> (/\S+)', result.stdout):
        dep = Path(item)
        if str(dep).startswith(str(source)) or dep.name.startswith(('libc.so', 'libm.so', 'libpthread', 'libdl.', 'librt.')):
            continue
        dependencies.add(dep)
for dep in dependencies:
    shutil.copy2(dep, libs / dep.name)
for dep in Path('/lib/x86_64-linux-gnu/nss').glob('*'):
    if dep.is_file():
        shutil.copy2(dep, libs / dep.name)
shutil.copytree('/usr/share/fonts/truetype', root / 'fonts', dirs_exist_ok=True)
(root / 'fonts.conf').write_text(f'<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd"><fontconfig><dir>{root}/fonts</dir><cachedir>/tmp/style-lo-font-cache</cachedir></fontconfig>\n')
(root / 'bin').mkdir(exist_ok=True)
wrapper = root / 'bin/libreoffice'
wrapper.write_text('#!/usr/bin/env bash\nset -e\n' +
                   f'export LD_LIBRARY_PATH="{root}/lib:{root}/libreoffice/program"\n' +
                   f'export FONTCONFIG_FILE="{root}/fonts.conf"\n' +
                   'export SAL_USE_VCLPLUGIN=svp\n' +
                   f'exec "{root}/libreoffice/program/soffice" "$@"\n')
wrapper.chmod(0o755)
print('Staged renderer:', root, 'shared dependencies:', len(dependencies))
