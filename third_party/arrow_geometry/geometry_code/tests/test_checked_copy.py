from pathlib import Path
import os
import shutil

import pytest
import checked_copy


def test_never_uses_fast_copy_and_keeps_metadata(tmp_path,monkeypatch):
    source=tmp_path/'source.bin';destination=tmp_path/'result.bin'
    source.write_bytes(bytes(range(256))*300+bytes(range(99)))
    os.utime(source,ns=(1700000000123456789,1700000000123456789))
    def forbidden(*args,**kwargs):raise AssertionError('Fast copy must not be called')
    monkeypatch.setattr(shutil,'copyfile',forbidden)
    monkeypatch.setattr(os,'sendfile',forbidden)
    checked_copy.copy_file(source,destination)
    assert destination.read_bytes()==source.read_bytes()
    assert destination.stat().st_mtime_ns==source.stat().st_mtime_ns


def test_checksum_failure_preserves_existing_destination(tmp_path,monkeypatch):
    source=tmp_path/'source';source.write_bytes(b'new data')
    destination=tmp_path/'destination';destination.write_bytes(b'previous result')
    monkeypatch.setattr(checked_copy,'digest',lambda path:'corrupted')
    with pytest.raises(OSError,match='differ'):checked_copy.copy_file(source,destination)
    assert destination.read_bytes()==b'previous result'
    assert not list(tmp_path.glob('*.copy'))


def test_copytree_uses_verified_copy(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'data').write_bytes(b'abc')
    shutil.copytree(source,tmp_path/'target',copy_function=checked_copy.copy_file)
    assert (tmp_path/'target/data').read_bytes()==b'abc'
