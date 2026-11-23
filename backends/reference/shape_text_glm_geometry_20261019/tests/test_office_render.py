import subprocess
from pathlib import Path

from shape_integration import office_render


def test_local_clean_conversion(monkeypatch, tmp_path):
    source = tmp_path / 'input.pptx'
    source.write_bytes(b'input')
    output = tmp_path / 'output'
    seen = {}
    monkeypatch.setenv('LD_LIBRARY_PATH', '/conda/lib')
    monkeypatch.setenv('LD_PRELOAD', 'bad.so')
    monkeypatch.setenv('FONTCONFIG_FILE', 'keep-font-config')

    class Process:
        returncode = 0
        def __init__(self, command, **kwargs):
            seen.update(kwargs)
            assert Path(command[-1]).read_bytes() == b'input'
            assert str(kwargs['cwd']).startswith('/tmp/shape_office_')
            (kwargs['cwd'] / 'input.pdf').write_bytes(b'%PDF-test')
        def communicate(self, timeout):
            return 'converted', ''

    monkeypatch.setattr(office_render.subprocess, 'Popen', Process)
    result = office_render.convert_pdf(source, output, 10)
    assert result['pdf_published']
    assert (output / 'input.pdf').read_bytes() == b'%PDF-test'
    assert 'LD_LIBRARY_PATH' not in seen['env']
    assert 'LD_PRELOAD' not in seen['env']
    assert seen['env']['FONTCONFIG_FILE'] == 'keep-font-config'
    assert not seen['cwd'].exists()
    assert (output / 'conversion.json').is_file()


def test_stale_output_is_not_success(monkeypatch, tmp_path):
    source = tmp_path / 'input.docx'
    source.write_bytes(b'input')
    output = tmp_path / 'output'
    output.mkdir()
    (output / 'input.pdf').write_bytes(b'old')

    class Process:
        returncode = 1
        def __init__(self, *args, **kwargs): pass
        def communicate(self, timeout): return '', 'failed'

    monkeypatch.setattr(office_render.subprocess, 'Popen', Process)
    result = office_render.convert_pdf(source, output, 10)
    assert not result['pdf_published']
    assert (output / 'input.pdf').read_bytes() == b'old'


def test_timeout_kills_only_conversion_group(monkeypatch, tmp_path):
    source = tmp_path / 'input.pptx'
    source.write_bytes(b'input')
    killed = []

    class Process:
        pid = 123456
        returncode = -9
        calls = 0
        def __init__(self, *args, **kwargs): pass
        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1: raise subprocess.TimeoutExpired('libreoffice', timeout)
            return '', 'stopped'

    monkeypatch.setattr(office_render.subprocess, 'Popen', Process)
    monkeypatch.setattr(office_render.os, 'killpg', lambda pid, sig: killed.append(pid))
    result = office_render.convert_pdf(source, tmp_path / 'out', 1)
    assert killed == [123456]
    assert result['timed_out'] and not result['pdf_published']
