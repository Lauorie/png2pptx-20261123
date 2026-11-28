"""The arrow renderer's version probe survives a slow start on a loaded machine."""
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import pytest

from full_recovery.arrow_renderer_probe import PROBE_TIMEOUTS, install

GEOMETRY_CODE = Path(__file__).resolve().parents[1]/'third_party/arrow_geometry/geometry_code'


def _installed(tmp_path: Path) -> Any:
    """The frozen probe module, copied and patched as a prepared run has it."""
    code = tmp_path/'geometry_code'
    code.mkdir()
    shutil.copy(GEOMETRY_CODE/'native_runtime.py', code/'native_runtime.py')
    install(code)
    spec = importlib.util.spec_from_file_location('native_runtime_' + tmp_path.name, code/'native_runtime.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _renderer(tmp_path: Path) -> Path:
    path = tmp_path/'soffice'
    path.write_text('#!/bin/sh\n')
    return path


def test_a_probe_that_times_out_once_is_tried_again_with_more_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _installed(tmp_path)
    calls: List[float] = []

    def run(command: List[str], **options: Any) -> SimpleNamespace:
        calls.append(options['timeout'])
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(command, options['timeout'])
        return SimpleNamespace(stdout='LibreOffice 26.8.1.2 abc\n')

    monkeypatch.setattr(module.subprocess, 'run', run)
    resolved, version = module.validated_renderer(_renderer(tmp_path))
    assert version.startswith('LibreOffice 26.8.') and resolved.name == 'soffice'
    assert calls == list(PROBE_TIMEOUTS[:2]) and calls[1] > calls[0]


def test_a_renderer_that_never_answers_still_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _installed(tmp_path)
    calls: List[float] = []

    def run(command: List[str], **options: Any) -> SimpleNamespace:
        calls.append(options['timeout'])
        raise subprocess.TimeoutExpired(command, options['timeout'])

    monkeypatch.setattr(module.subprocess, 'run', run)
    with pytest.raises(subprocess.TimeoutExpired):
        module.validated_renderer(_renderer(tmp_path))
    assert calls == list(PROBE_TIMEOUTS)


def test_another_version_is_refused_without_a_second_try(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _installed(tmp_path)
    calls: List[float] = []

    def run(command: List[str], **options: Any) -> SimpleNamespace:
        calls.append(options['timeout'])
        return SimpleNamespace(stdout='LibreOffice 7.3.7.2\n')

    monkeypatch.setattr(module.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='26.8'):
        module.validated_renderer(_renderer(tmp_path))
    assert len(calls) == 1


def test_an_already_patched_or_different_probe_is_refused(tmp_path: Path) -> None:
    _installed(tmp_path)
    with pytest.raises(ValueError):
        install(tmp_path/'geometry_code')


def test_the_frozen_source_itself_is_left_untouched(tmp_path: Path) -> None:
    before = (GEOMETRY_CODE/'native_runtime.py').read_text()
    _installed(tmp_path)
    assert (GEOMETRY_CODE/'native_runtime.py').read_text() == before and 'PROBE_TIMEOUTS' not in before
