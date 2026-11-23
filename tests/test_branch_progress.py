import json
from concurrent.futures import ThreadPoolExecutor
import threading
from unittest.mock import patch

import pytest

from full_recovery.atomic_json import save
from full_recovery.api_worker import read, read_progress


def test_live_readers_never_observe_partial_progress(tmp_path):
    path = tmp_path / 'progress.json'
    save(path, {'completed': 0, 'padding': 'x' * 65536})
    stopped = threading.Event()

    def reader():
        reads = 0
        while not stopped.is_set():
            value = json.loads(path.read_text())
            assert len(value['padding']) == 65536
            assert 0 <= value['completed'] <= 100
            reads += 1
        return reads

    with ThreadPoolExecutor(max_workers=2) as pool:
        readers = [pool.submit(reader) for _ in range(2)]
        try:
            for completed in range(1, 101):
                save(path, {'completed': completed, 'padding': 'x' * 65536})
        finally:
            stopped.set()
        assert all(future.result() > 0 for future in readers)
    assert read(path)['completed'] == 100


def test_interrupted_publication_preserves_previous_document(tmp_path):
    path = tmp_path / 'progress.json'
    save(path, {'completed': 20})

    def interrupted_dump(value, stream, **kwargs):
        stream.write('{"completed":')
        raise OSError('simulated writer interruption')

    with patch('full_recovery.atomic_json.json.dump', interrupted_dump):
        with pytest.raises(OSError):
            save(path, {'completed': 21})
    assert read(path) == {'completed': 20}
    assert not list(tmp_path.glob('*.tmp'))


def test_only_progress_reads_can_fall_back_after_truncated_write(tmp_path):
    path = tmp_path / 'progress.json'
    previous = {'completed': 20, 'failed': 0}
    path.write_text('')
    assert read_progress(path, previous) == (previous, 'JSONDecodeError')
    with pytest.raises(json.JSONDecodeError):
        read(path)
    path.write_text('{"completed":21}')
    assert read_progress(path, previous) == ({'completed': 21}, None)


def test_progress_permissions_remain_readable_and_existing_privacy_is_preserved(tmp_path):
    path=tmp_path/'progress.json'
    save(path,{'completed':1})
    assert path.stat().st_mode & 0o777 == 0o644
    path.chmod(0o600)
    save(path,{'completed':2})
    assert path.stat().st_mode & 0o777 == 0o600
