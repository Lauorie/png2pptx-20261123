"""The SAM service keeps its GPU memory bounded and says how much it really uses."""
import json
import sys
import threading
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deploy/sam/code'))

from gpu_memory import MemoryPolicy, apply_limit, policy_from, settle  # noqa: E402

MIB = 1024 * 1024


class FakeCuda:
    def __init__(self, allocated: int, reserved: int, peak: int, total: int = 32000) -> None:
        self.allocated, self.reserved, self.peak, self.total = allocated * MIB, reserved * MIB, peak * MIB, total * MIB
        self.emptied = 0
        self.fraction = None

    def memory_allocated(self) -> int:
        return self.allocated

    def memory_reserved(self) -> int:
        return self.reserved

    def max_memory_allocated(self) -> int:
        return self.peak

    def reset_peak_memory_stats(self) -> None:
        self.peak = self.allocated

    def empty_cache(self) -> None:
        self.emptied += 1
        self.reserved = self.allocated

    def set_per_process_memory_fraction(self, fraction: float) -> None:
        self.fraction = fraction

    def get_device_properties(self, index: int):
        return type('Properties', (), {'total_memory': self.total})()


class FakeTorch:
    def __init__(self, cuda: FakeCuda) -> None:
        self.cuda = cuda


def test_a_cache_above_the_limit_is_returned_to_the_gpu() -> None:
    cuda = FakeCuda(allocated=9000, reserved=27000, peak=21000)
    report = settle(FakeTorch(cuda), MemoryPolicy(cache_limit_mib=2048))
    assert cuda.emptied == 1
    assert report == {'in_use_mib': 9000, 'reserved_mib': 9000, 'request_peak_mib': 21000, 'cache_released_mib': 18000}


def test_a_small_cache_is_kept_for_the_next_request() -> None:
    cuda = FakeCuda(allocated=9000, reserved=10500, peak=9800)
    report = settle(FakeTorch(cuda), MemoryPolicy(cache_limit_mib=2048))
    assert cuda.emptied == 0 and report['reserved_mib'] == 10500 and report['cache_released_mib'] == 0


def test_each_request_reports_its_own_peak() -> None:
    cuda = FakeCuda(allocated=9000, reserved=9000, peak=21000)
    settle(FakeTorch(cuda), MemoryPolicy(cache_limit_mib=2048))
    assert cuda.peak == cuda.allocated


def test_a_hard_limit_is_given_to_the_allocator_as_a_share_of_the_card() -> None:
    cuda = FakeCuda(0, 0, 0, total=32000)
    assert apply_limit(FakeTorch(cuda), MemoryPolicy(cache_limit_mib=2048, hard_limit_mib=16000)) == 16000
    assert cuda.fraction == pytest.approx(0.5)


def test_no_hard_limit_leaves_the_allocator_alone() -> None:
    cuda = FakeCuda(0, 0, 0)
    assert apply_limit(FakeTorch(cuda), MemoryPolicy(cache_limit_mib=2048)) is None and cuda.fraction is None


def test_the_policy_comes_from_the_service_config_with_a_bounded_default() -> None:
    assert policy_from({}) == MemoryPolicy(cache_limit_mib=2048, hard_limit_mib=None)
    assert policy_from({'gpu_cache_limit_mib': 512, 'gpu_memory_limit_mib': 16000}) == MemoryPolicy(512, 16000)
    with pytest.raises(ValueError):
        policy_from({'gpu_cache_limit_mib': -1})


class FakeEngine:
    def __init__(self, cuda: FakeCuda) -> None:
        self.torch = FakeTorch(cuda)

    def validate(self, request: dict) -> None:
        return None

    def __call__(self, request: dict) -> dict:
        return {'predictions': []}


def test_health_reports_what_the_last_request_used(tmp_path: Path) -> None:
    from server import make_server
    (tmp_path / 'config.json').write_text(json.dumps({'idle_resume_seconds': 0, 'gpu_cache_limit_mib': 2048}))
    cuda = FakeCuda(allocated=9000, reserved=27000, peak=21000)
    state = {'phase': 'ready', 'requests_completed': 0, 'requests_failed': 0}
    server = make_server(FakeEngine(cuda), ('127.0.0.1', 0), tmp_path, state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = 'http://127.0.0.1:%d' % server.server_address[1]
        request = urllib.request.Request(url + '/', json.dumps({'image': '/x.png'}).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=10) as reply:
            assert reply.status == 200
        with urllib.request.urlopen(url + '/health', timeout=10) as reply:
            health = json.loads(reply.read())
    finally:
        server.shutdown()
        server.server_close()
    assert health['gpu_memory'] == {'in_use_mib': 9000, 'reserved_mib': 9000, 'request_peak_mib': 21000, 'cache_released_mib': 18000}
    assert cuda.emptied == 1
