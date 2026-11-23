from concurrent.futures import ThreadPoolExecutor
import subprocess
from types import SimpleNamespace
from pathlib import Path
import time

import pytest
import native_runtime as runtime


def test_parallel_checks_are_single_flight_and_profile_is_private(tmp_path, monkeypatch):
    binary = tmp_path / 'soffice'; binary.write_text('test executable')
    runtime._VERSIONS.clear(); calls = []
    def probe(command, **kwargs):
        calls.append(command)
        assert '--headless' in command
        uri = next(arg.split('=', 1)[1] for arg in command if arg.startswith('-env:UserInstallation='))
        assert uri.startswith('file://')
        assert Path(uri[7:]).is_dir()
        time.sleep(.02)
        return SimpleNamespace(stdout='LibreOffice 26.8.0.3 test\n')
    monkeypatch.setattr(runtime.subprocess, 'run', probe)
    with ThreadPoolExecutor(16) as pool:
        results = list(pool.map(lambda _: runtime.validated_renderer(binary), range(32)))
    assert len(calls) == 1
    assert len(set(results)) == 1
    binary.write_text('new test executable build')
    runtime.validated_renderer(binary)
    assert len(calls) == 2


def test_timeout_and_old_version_are_not_cached_as_success(tmp_path, monkeypatch):
    binary = tmp_path / 'soffice'; binary.write_text('test')
    runtime._VERSIONS.clear()
    responses = iter([subprocess.TimeoutExpired(['soffice'], 15), 'LibreOffice 7.5', 'LibreOffice 26.8.0.3'])
    def probe(command, **kwargs):
        value = next(responses)
        if isinstance(value, Exception):raise value
        return SimpleNamespace(stdout=value)
    monkeypatch.setattr(runtime.subprocess, 'run', probe)
    with pytest.raises(subprocess.TimeoutExpired):runtime.validated_renderer(binary)
    with pytest.raises(RuntimeError, match='26.8'):runtime.validated_renderer(binary)
    assert runtime.validated_renderer(binary)[1] == 'LibreOffice 26.8.0.3'
