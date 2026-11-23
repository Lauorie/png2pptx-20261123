"""Deadline regression: active/stalled streams close, release slots, never retry."""
import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import httpx
import test_pipeline as fixtures
from test_pipeline import FakeRuntime, observed, reviewed
from shape_integration.common import Runtime, RequestTimeout, read
from shape_integration.fitting.scene import assemble, export


class Stream(httpx.AsyncByteStream):
    def __init__(self, mode):
        self.mode = mode; self.closed = False

    async def __aiter__(self):
        # Complete-looking JSON must still be discarded without a finished stream.
        chunk = {'choices': [{'finish_reason': 'stop' if self.mode == 'ok' else None,
                              'delta': {'content': '{"ok":true}'}}]}
        yield ('data: '+json.dumps(chunk)+'\n\n').encode()
        if self.mode == 'ok':
            yield b'data: [DONE]\n\n'
        elif self.mode == 'stall':
            await asyncio.sleep(10)
        else:
            while True:
                await asyncio.sleep(.01)
                yield b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n\n'

    async def aclose(self):
        self.closed = True


class RequestDeadlineTests(unittest.TestCase):
    setUp = fixtures.PipelineTests.setUp
    run_object = fixtures.PipelineTests.run_object

    def transport(self, mode):
        self.stream = Stream(mode); self.sent = []
        async def handle(request):
            self.sent.append(request)
            if mode == 'headers':
                await asyncio.sleep(10)
            return httpx.Response(200, stream=self.stream)
        original = httpx.AsyncClient
        return patch('shape_integration.transport.httpx.AsyncClient',
                     side_effect=lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))

    def test_active_stalled_and_header_timeouts_do_not_retry_even_after_resume(self):
        for mode in ('active', 'stall', 'headers'):
            with self.subTest(mode=mode):
                config = copy.deepcopy(self.config)
                config.update(request_timeout_seconds=.12, shape_schema_retries=3)
                runtime = Runtime(config, 'http://offline.invalid/v1', '')
                root = self.work/mode
                checked = []
                def validate(raw): checked.append(raw); return raw
                with self.transport(mode):
                    started = time.monotonic()
                    for _ in range(2):  # Resume must not relaunch a discarded request.
                        with self.assertRaises(RequestTimeout):
                            runtime.call(root, 'observe', 'system', {}, [], validate, 1000)
                    self.assertLess(time.monotonic()-started, 2)
                self.assertEqual(len(self.sent), 1)
                if mode != 'headers':
                    self.assertTrue(self.stream.closed)
                self.assertEqual(runtime.slots.snapshot()['active'], 0)
                self.assertEqual(checked, [])
                self.assertFalse((root/'observe_00/result.json').exists())
                self.assertFalse((root/'observe_00/parsed_response.json').exists())
                self.assertFalse((root/'observe_01').exists())
                self.assertEqual(read(root/'observe_00/stream_status.json')['state'], 'timed_out')
                self.assertFalse(read(root/'observe_00/error_0.json')['retryable'])

    def test_queue_wait_is_not_part_of_deadline_and_complete_stream_validates(self):
        config = copy.deepcopy(self.config)
        config.update(api_concurrency=1, request_timeout_seconds=.5)
        runtime = Runtime(config, 'http://offline.invalid/v1', '')
        with self.transport('ok'), ThreadPoolExecutor(max_workers=1) as pool:
            with runtime.slots:
                pending = pool.submit(runtime.call, self.work/'queued', 'observe', 'system', {}, [], lambda x: x, 1000)
                time.sleep(.7)
                self.assertEqual(self.sent, [])
            self.assertEqual(pending.result(timeout=2), {'ok': True})
        self.assertTrue(self.stream.closed)
        self.assertEqual(runtime.slots.snapshot()['active'], 0)
        timing = read(self.work/'queued/observe_00/stream_status.json')
        self.assertGreater(timing['queue_seconds'], .5)
        self.assertLess(timing['seconds'], .5)

    def test_refine_timeout_keeps_prior_geometry_and_stop_reason(self):
        calls = []
        def responder(tag, prompt, count):
            calls.append(tag)
            if count == 1: return observed('G1')
            if tag == 'check': return reviewed(prompt, 'revise')
            raise RequestTimeout('deadline; output discarded')
        result = self.run_object(FakeRuntime(self.config, responder))
        self.assertEqual(calls, ['observe', 'check', 'observe'])
        self.assertEqual(result['stop_reason'], 'observation_timeout')
        self.assertEqual(result['selected_revision'], 'revision_00')
        scene = assemble(self.folder, self.item, [self.candidate], {'G1': result}, {}, [])
        self.assertEqual(export(scene, self.folder, self.source)['geometry_final']['native'], 2)

    def test_critic_timeout_keeps_current_geometry(self):
        def responder(tag, prompt, count):
            if tag == 'observe': return observed('G1')
            raise RequestTimeout('deadline; output discarded')
        runtime = FakeRuntime(self.config, responder)
        result = self.run_object(runtime)
        self.assertEqual([tag for tag, _ in runtime.events], ['observe', 'check'])
        self.assertEqual(result['stop_reason'], 'check_timeout')
        self.assertEqual(result['selected_revision'], 'revision_00')


if __name__ == '__main__':
    unittest.main()
