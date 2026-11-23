import importlib.util
import multiprocessing
from pathlib import Path
import time

import pytest

from shape_integration.resource_limits import slot, bounded

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('batch_glm_geometry',ROOT/'batch_glm_geometry.py')
batch=importlib.util.module_from_spec(spec);spec.loader.exec_module(batch)


def test_discover_pptbench_uses_only_selected_reference_images(tmp_path):
    (tmp_path/'taskset.txt').write_text('task_0002\ntask_0001\n')
    for ident in ('task_0001','task_0002','task_0003'):
        folder=tmp_path/'tasks'/ident;folder.mkdir(parents=True)
        (folder/'reference.png').write_bytes(b'input')
        (folder/'metadata.json').write_text('DO NOT READ ground truth')
    result=batch.discover(tmp_path)
    assert list(result)==['task_0002','task_0001']
    assert all(p.name=='reference.png' for p in result.values())


@pytest.mark.parametrize('ids',['task_1\ntask_1\n','../outside\n','/absolute\n','missing\n'])
def test_invalid_or_missing_inputs_are_not_silently_skipped(tmp_path,ids):
    (tmp_path/'tasks').mkdir();(tmp_path/'taskset.txt').write_text(ids)
    with pytest.raises(ValueError):batch.discover(tmp_path)


def test_nested_slot_and_exception_release(tmp_path,monkeypatch):
    monkeypatch.setenv('SHAPE_BATCH_LIMIT_DIR',str(tmp_path/'locks'))
    monkeypatch.setenv('SHAPE_CPU_SLOTS','1')
    @bounded('cpu')
    def nested():
        with slot('cpu'):raise RuntimeError('expected')
    with pytest.raises(RuntimeError):nested()
    with slot('cpu'):pass


def test_slots_bound_multiple_processes(tmp_path,monkeypatch):
    monkeypatch.setenv('SHAPE_BATCH_LIMIT_DIR',str(tmp_path/'locks'))
    monkeypatch.setenv('SHAPE_CPU_SLOTS','2')
    ctx=multiprocessing.get_context('fork')
    active=ctx.Value('i',0);peak=ctx.Value('i',0);count=ctx.Value('i',0);guard=ctx.Lock()
    def work():
        with slot('cpu'):
            with guard:
                active.value+=1;peak.value=max(peak.value,active.value)
            time.sleep(.1)
            with guard:active.value-=1;count.value+=1
    workers=[ctx.Process(target=work) for _ in range(6)]
    try:
        for p in workers:p.start()
        for p in workers:p.join(10)
        assert all(p.exitcode==0 for p in workers)
        assert count.value==6 and 1<=peak.value<=2 and active.value==0
    finally:
        for p in workers:
            if p.is_alive():p.terminate();p.join()


def test_resource_limiter_disabled_for_existing_single_case_runs(monkeypatch):
    monkeypatch.delenv('SHAPE_BATCH_LIMIT_DIR',raising=False)
    monkeypatch.setenv('SHAPE_CPU_SLOTS','invalid-unused')
    with slot('cpu'):pass
