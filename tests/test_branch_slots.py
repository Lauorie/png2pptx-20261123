"""Ensure branch controllers share a real capacity limit and lend idle slots."""
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import threading
import time

from full_recovery.case_slots import CaseSlots


def test_capacity_is_shared_and_idle_branch_lends_slots(tmp_path):
    config = {'root': str(tmp_path), 'total': 6, 'shape_reserved': 4}
    release = threading.Event()
    active = {'shape': 0, 'text': 0}
    maximum = {'all': 0}
    mutex = threading.Lock()

    def work(branch, index):
        with CaseSlots(config, branch).acquire(str(index)):
            with mutex:
                active[branch] += 1
                maximum['all'] = max(maximum['all'], sum(active.values()))
            release.wait(5)
            with mutex:
                active[branch] -= 1

    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(work, branch, i) for branch in ('shape', 'text') for i in range(6)]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with mutex:
                ready = active == {'shape': 4, 'text': 2}
            if ready:
                break
            time.sleep(.01)
        assert ready
        release.set()
        for future in futures:
            future.result(timeout=5)
    assert maximum['all'] == 6

    # All six slots must become available to shape after the text queue drains.
    CaseSlots(config, 'text').drained()
    release.clear()
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(work, 'shape', i) for i in range(6)]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with mutex:
                ready = active['shape'] == 6
            if ready:
                break
            time.sleep(.01)
        assert ready
        release.set()
        for future in futures:
            future.result(timeout=5)


def occupy(config, ready):
    with CaseSlots(config, 'shape').acquire('interrupted'):
        ready.set()
        time.sleep(30)


def test_dead_worker_releases_its_case_slot(tmp_path):
    config = {'root': str(tmp_path), 'total': 2, 'shape_reserved': 1}
    context = multiprocessing.get_context('fork')
    ready = context.Event()
    process = context.Process(target=occupy, args=(config, ready))
    process.start()
    try:
        assert ready.wait(5)
        process.terminate()
        process.join(5)
        assert not process.is_alive()
        # A killed owner must not strand capacity after a checkpoint resume.
        with CaseSlots(config, 'shape').acquire('resumed'):
            pass
    finally:
        if process.is_alive():
            process.kill()
            process.join(5)
