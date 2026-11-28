"""The text stage asks independent regions at the same time instead of one by one.

Measured on one dense figure: the four visual-audit regions took 687 s back to
back although no request needs another one's answer before it is sent.
"""
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from full_recovery.text_parallel import audit_in_parallel, crop_executor, with_request_limit

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT/'backends/reference/shape_text_glm_geometry_20261019'


def sequential_audit(inventory, folders, log=None):
    """Stand-in for the backend audit: one request per region, applied in order.

    Like the backend, a region whose request fails leaves the inventory as it was.
    """
    def run(ask):
        rows = list(inventory)
        failures = []
        for folder in folders:
            prompt = {'existing_occurrences': [{'id': row} for row in rows]}
            if log is not None:
                log.append((folder.name, len(rows)))
            try:
                reply = ask('runtime', folder, prompt, [], None, {})
                rows = [r for r in rows if r not in {d['id'] for d in reply['duplicate_ids']}]
                rows += reply['texts']
            except Exception as exc:
                failures.append(f'failed:{folder.name}:{exc}')
        return rows + failures
    return run


def folders(tmp_path, count):
    return [tmp_path/f'R{index:02d}' for index in range(count)]


def test_all_regions_are_asked_at_the_same_time(tmp_path):
    together = threading.Barrier(3, timeout=10)

    def ask(runtime, folder, prompt, images, validator, opt):
        together.wait()  # breaks unless three requests are in flight together
        return {'texts': [folder.name], 'duplicate_ids': []}

    rows = audit_in_parallel(sequential_audit(['T1'], folders(tmp_path, 3)), ask, workers=4)
    assert rows == ['T1', 'R00', 'R01', 'R02']


def test_requests_describe_the_inventory_before_any_reply_is_applied(tmp_path):
    seen = {}

    def ask(runtime, folder, prompt, images, validator, opt):
        seen[folder.name] = [o['id'] for o in prompt['existing_occurrences']]
        return {'texts': [folder.name], 'duplicate_ids': []}

    audit_in_parallel(sequential_audit(['T1'], folders(tmp_path, 2)), ask, workers=4)
    assert seen == {'R00': ['T1'], 'R01': ['T1']}


def test_replies_are_applied_in_region_order_whatever_order_they_arrive_in(tmp_path):
    first_done = threading.Event()

    def ask(runtime, folder, prompt, images, validator, opt):
        if folder.name == 'R00':
            first_done.wait(10)  # the first region answers last
        else:
            first_done.set()
        return {'texts': [folder.name], 'duplicate_ids': []}

    rows = audit_in_parallel(sequential_audit([], folders(tmp_path, 2)), ask, workers=4)
    assert rows == ['R00', 'R01']


def test_a_failed_region_keeps_its_error_and_the_others_still_apply(tmp_path):
    def ask(runtime, folder, prompt, images, validator, opt):
        if folder.name == 'R01':
            raise ValueError('bad reply')
        return {'texts': [folder.name], 'duplicate_ids': []}

    rows = audit_in_parallel(sequential_audit([], folders(tmp_path, 3)), ask, workers=4)
    assert rows == ['R00', 'R02', 'failed:R01:bad reply']


def test_a_retirement_already_made_by_an_earlier_region_is_not_repeated(tmp_path):
    def ask(runtime, folder, prompt, images, validator, opt):
        return {'texts': [], 'duplicate_ids': [{'id': 'T2', 'duplicate_of': 'T1', 'reason': 'same'}]}

    applied = []

    def run(ask_region):
        rows = ['T1', 'T2']
        for folder in folders(tmp_path, 2):
            try:
                reply = ask_region('runtime', folder,
                                   {'existing_occurrences': [{'id': r} for r in rows]}, [], None, {})
            except Exception:
                continue
            applied.append(reply['duplicate_ids'])
            rows = [r for r in rows if r not in {d['id'] for d in reply['duplicate_ids']}]
        return rows

    assert audit_in_parallel(run, ask, workers=4) == ['T1']
    assert [len(items) for items in applied] == [1, 0]


def test_an_audit_without_regions_makes_no_request(tmp_path):
    def ask(*args):
        raise AssertionError('no region, no request')

    assert audit_in_parallel(sequential_audit(['T1'], []), ask, workers=4) == ['T1']


def test_the_images_that_were_sent_stay_on_disk_as_evidence(tmp_path):
    image = tmp_path/'ids.png'
    drawings = iter([b'drawn for the request', b'redrawn while applying'])

    def run(ask_region):
        image.write_bytes(next(drawings))  # the audit redraws the hint image on every pass
        try:
            ask_region('runtime', tmp_path/'R00', {'existing_occurrences': []},
                       [('ids', image)], None, {})
        except Exception:
            pass
        return []

    audit_in_parallel(run, lambda *args: {'texts': [], 'duplicate_ids': []}, workers=4)
    assert image.read_bytes() == b'drawn for the request'


def test_worker_count_must_be_positive():
    with pytest.raises(ValueError):
        audit_in_parallel(lambda ask: [], lambda *args: {}, workers=0)
    with pytest.raises(ValueError):
        crop_executor(0)


def test_crop_review_pool_uses_the_configured_width():
    with crop_executor(12)(max_workers=4) as pool:
        assert pool._max_workers == 12


def test_the_request_limit_is_raised_to_the_crop_pool_width_without_touching_the_input():
    config = {'api_concurrency': 4, 'model': 'm'}
    assert with_request_limit(config, 12) == {'api_concurrency': 12, 'model': 'm'}
    assert config['api_concurrency'] == 4


def test_a_larger_configured_request_limit_is_kept():
    assert with_request_limit({'api_concurrency': 32}, 12)['api_concurrency'] == 32


INSTALL_CHECK = r"""
import sys
from pathlib import Path
root, backend = Path(sys.argv[1]), Path(sys.argv[2])
sys.path[:0] = [str(backend/'src'), str(root)]
from full_recovery import text_diagnostic_repair
from full_recovery.text_parallel import install
from shape_integration.texts import inventory_audit, two_stage_crop
before = inventory_audit.audit, text_diagnostic_repair.ask_inventory
install(audit_workers=4, crop_workers=12)
install(audit_workers=4, crop_workers=12)
assert inventory_audit.audit is not before[0]
assert text_diagnostic_repair.ask_inventory is not before[1]
with two_stage_crop.ThreadPoolExecutor(max_workers=4) as pool:
    assert pool._max_workers == 12
print('INSTALLED')
"""


def test_install_replaces_the_audit_and_widens_the_crop_review_pool():
    # Own interpreter with the text stage's import order: the suite also
    # imports an older shape_integration copy from the repo root.
    proc = subprocess.run([sys.executable, '-c', INSTALL_CHECK, str(ROOT), str(BACKEND)],
                          capture_output=True, text=True, timeout=120)
    assert proc.stdout.strip().endswith('INSTALLED'), proc.stderr[-2000:]
