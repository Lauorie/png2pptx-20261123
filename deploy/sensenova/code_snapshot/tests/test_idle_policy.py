import asyncio
import concurrent.futures
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

WORKING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKING))
from idle_activity import IdleActivity
from gpu_lock_middleware import GpuWorkLockMiddleware


class Clock:
    now = 1000.0
    def __call__(self):
        return self.now


class IdlePolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=WORKING / "tests")
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        self.clock = Clock()
        self.gate = IdleActivity(self.run, 2, 600, self.clock)

    def permit(self, index=0):
        with self.gate.idle_permission(index) as status:
            return status['allowed']

    def test_idle_controllers_can_run_on_both_gpus(self):
        with self.gate.idle_permission(0) as one, self.gate.idle_permission(1) as two:
            self.assertTrue(one['allowed'] and two['allowed'])

    def test_arrival_stops_idle_before_waiting_for_locks(self):
        with self.gate.idle_permission(0):
            guard = self.gate.announce_request()
            self.assertFalse(self.permit(1))
        guard.finish()

    def test_long_request_blocks_idle_then_waits_full_ten_minutes(self):
        with self.gate.request():
            self.clock.now += 900
            self.assertFalse(self.permit(0))
            self.assertFalse(self.permit(1))
        self.assertEqual(self.gate.remaining(), 600)
        self.clock.now += 599
        self.assertFalse(self.permit())
        self.clock.now += 1
        self.assertTrue(self.permit())

    def test_overlapping_requests_and_new_request_reset_deadline(self):
        first = self.gate.announce_request(); first.acquire()
        self.clock.now += 20
        second = self.gate.announce_request(); second.acquire(); second.finish()
        self.clock.now += 700
        self.assertFalse(self.permit())
        first.finish()
        self.clock.now += 599
        third = self.gate.announce_request(); third.acquire(); third.finish()
        self.clock.now += 599
        self.assertFalse(self.permit())
        self.clock.now += 1
        self.assertTrue(self.permit())

    def prepare_middleware(self, app):
        (self.run / 'config.json').write_text(json.dumps({'gpu_count': 2, 'idle_resume_seconds': 600}))
        (self.run / 'evidence').mkdir()
        for i in range(2):
            (self.run / 'power' / str(i)).mkdir(parents=True)
        prior = os.environ.get('SENSENOVA_RUN_DIR')
        os.environ['SENSENOVA_RUN_DIR'] = str(self.run)
        try:
            middleware = GpuWorkLockMiddleware(app)
        finally:
            if prior is None: os.environ.pop('SENSENOVA_RUN_DIR', None)
            else: os.environ['SENSENOVA_RUN_DIR'] = prior
        middleware.activity = self.gate
        return middleware

    def test_health_does_not_reset_idle_timer(self):
        async def app(scope, receive, send):
            pass
        middleware = self.prepare_middleware(app)
        asyncio.run(middleware({'type': 'http', 'method': 'GET'}, None, None))
        self.assertFalse(self.gate.state_path.exists())

    def test_stream_lifetime_and_full_cooldown_after_last_chunk(self):
        async def scenario():
            started, finish = asyncio.Event(), asyncio.Event()
            async def app(scope, receive, send):
                await send({'type': 'http.response.body', 'body': b'partial', 'more_body': True})
                started.set()
                await finish.wait()
                await send({'type': 'http.response.body', 'body': b'last', 'more_body': False})
            async def send(message): pass
            middleware = self.prepare_middleware(app)
            task = asyncio.create_task(middleware({'type': 'http', 'method': 'POST', 'path': '/v1/images/edits'}, None, send))
            await asyncio.wait_for(started.wait(), 2)
            self.clock.now += 900
            self.assertFalse(self.permit(0)); self.assertFalse(self.permit(1))
            finish.set(); await task
            self.assertEqual(self.gate.remaining(), 600)
            self.clock.now += 600
            self.assertTrue(self.permit(0)); self.assertTrue(self.permit(1))
        asyncio.run(scenario())

    def test_cancelled_partial_lock_acquisition_releases_every_lock(self):
        async def scenario():
            async def app(scope, receive, send):
                self.fail('Request must be waiting for GPU 1')
            middleware = self.prepare_middleware(app)
            with (self.run / 'power/1/gpu-work.lock').open('a') as blocker:
                fcntl.flock(blocker, fcntl.LOCK_EX)
                task = asyncio.create_task(middleware({'type': 'http', 'method': 'POST'}, None, None))
                await asyncio.sleep(0.03)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError): await task
            self.assertEqual(self.gate.remaining(), 600)
            self.clock.now += 600
            self.assertTrue(self.permit(0)); self.assertTrue(self.permit(1))
            for i in range(2):
                with (self.run / f'power/{i}/gpu-work.lock').open('a') as owner:
                    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(scenario())


class SamHttpTests(unittest.TestCase):
    def test_busy_rejection_health_and_cooldown_without_model_inference(self):
        code = Path(__file__).resolve().parents[3] / 'sam/code'
        sys.path.insert(0, str(code))
        spec = importlib.util.spec_from_file_location('idle60_sam_server', code / 'server.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        entered, finish = threading.Event(), threading.Event()
        class FakeEngine:
            def validate(self, request): pass
            def __call__(self, request):
                entered.set()
                if not finish.wait(3): raise TimeoutError('test engine wait')
                return {'ok': True}
        with tempfile.TemporaryDirectory(dir=WORKING / "tests") as directory:
            run = Path(directory)
            (run / 'config.json').write_text(json.dumps({'idle_resume_seconds': 600}))
            state = dict(phase='ready', requests_completed=0, requests_failed=0)
            server = module.make_server(FakeEngine(), ('127.0.0.1', 0), run, state)
            thread = threading.Thread(target=server.serve_forever); thread.start()
            url = f'http://127.0.0.1:{server.server_port}'
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            def post():
                request = urllib.request.Request(url + '/v1/segment', b'{}', {'Content-Type': 'application/json'})
                with opener.open(request, timeout=3) as response: return response.status
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    first = pool.submit(post)
                    self.assertTrue(entered.wait(2))
                    with opener.open(url + '/health', timeout=2) as response:
                        self.assertTrue(json.load(response)['busy'])
                    with self.assertRaises(urllib.error.HTTPError) as error: post()
                    self.assertEqual(error.exception.code, 503)
                    finish.set(); self.assertEqual(first.result(3), 200)
                activity = IdleActivity(run)
                self.assertGreater(activity.remaining(), 599)
            finally:
                finish.set(); server.shutdown(); thread.join(3); server.server_close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
