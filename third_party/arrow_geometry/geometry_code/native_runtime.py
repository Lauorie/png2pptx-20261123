"""Validate the native renderer once per binary build, with an isolated profile."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading

_LOCK = threading.Lock()
_VERSIONS = {}


def validated_renderer(executable=None):
    executable = executable or os.environ.get('ARROW_LIBREOFFICE') or '/opt/libreoffice26.8/program/soffice'
    resolved = Path(shutil.which(str(executable)) or executable).resolve()
    markers = [resolved] + [p for p in (resolved.parent / 'soffice.bin', resolved.parent / 'versionrc') if p.exists()]
    fingerprint = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in markers)
    with _LOCK:
        if fingerprint not in _VERSIONS:
            # The old --version probe used the user's shared default profile and
            # ran for every preview. Never attach this check to another LO session.
            with tempfile.TemporaryDirectory(prefix='arrow-lo-version-') as profile:
                command = [str(resolved), '-env:UserInstallation=' + Path(profile).as_uri(), '--headless', '--version']
                version = subprocess.run(command, capture_output=True, text=True, check=True, timeout=15).stdout.strip()
            if not version.startswith('LibreOffice 26.8.'):
                raise RuntimeError('Geometry head previews require validated LibreOffice 26.8; got ' + version)
            _VERSIONS[fingerprint] = version
        return resolved, _VERSIONS[fingerprint]
