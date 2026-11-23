"""Meaningful wire-contract checks; no network/API credentials or inference."""
import json
from pathlib import Path
import sys
import unittest
import queue
import tempfile
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from full_recovery.api_gateway import assemble,payload_for_api,IncompleteStream,ApiRelay,RequestPacer


class ApiWireTests(unittest.TestCase):
    def test_routed_gpt_model_preserves_route_and_reasoning_wire_contract(self):
        body = {'max_tokens':100000, 'temperature':0, 'seed':42, 'top_p':1,
                'chat_template_kwargs':{'enable_thinking':True},
                'messages':[{'role':'user','content':[
                    {'type':'image_url','image_url':{'url':'data:image/png;base64,test'}}]}]}
        for model in ('gpt-5.6-terra', 'pool104/gpt-5.6-terra'):
            with self.subTest(model=model):
                out = payload_for_api(body, model, 'high', 'priority')
                self.assertEqual(out['model'], model)
                self.assertEqual(out['reasoning_effort'], 'high')
                self.assertEqual(out['service_tier'], 'priority')
                self.assertEqual(out['max_tokens'], 100000)
                self.assertEqual(out['messages'], body['messages'])
                for name in ('thinking','temperature','seed','top_p','chat_template_kwargs'):
                    self.assertNotIn(name, out)

    def test_request_pacing_is_shared_across_concurrent_threads(self):
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor
        pacer = RequestPacer(.04)
        cancel = threading.Event()
        def start():
            pacer.wait(time.monotonic() + 2, cancel)
            return time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as pool:
            times = sorted(pool.map(lambda _: start(), range(4)))
        assert all(b - a >= .03 for a, b in zip(times, times[1:]))
        pacer.limited(.04)
        with self.assertRaises(IncompleteStream):
            pacer.wait(time.monotonic() + .005, cancel)

    def test_rate_limits_wait_and_retry_without_returning_a_shape_failure(self):
        import io
        import threading
        import urllib.error
        class Response:
            headers = {'Content-Type':'text/event-stream'}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __iter__(self):
                yield b'data: {"choices":[{"delta":{"content":"{}"},"finish_reason":"stop"}]}\n'
                yield b'data: [DONE]\n'
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); key = folder/'key'; key.write_text('test-key'); key.chmod(0o600)
            relay = ApiRelay({'api_key_file':str(key), 'api_global_concurrency':8,
                'api_model':'test', 'api_base_url':'https://example.invalid/v1',
                'api_request_start_interval_seconds':.001,
                'api_rate_limit_attempts':5}, folder)
            errors = [urllib.error.HTTPError('https://example.invalid',429,'limited',{},io.BytesIO(b'{}')) for _ in range(3)]
            cancelled = threading.Event()
            output = queue.Queue()
            with patch('full_recovery.api_gateway.urllib.request.build_opener') as build, \
                    patch.object(cancelled, 'wait', return_value=False):
                build.return_value.open.side_effect = [*errors, Response()]
                relay.produce({'messages':[]}, output, cancelled=cancelled)
            assert output.queue[-1][0] == 'complete'
            assert relay.stats['upstream_rate_limited'] == 3
            assert relay.stats['failed'] == 0 and relay.stats['completed'] == 1
            assert relay.stats['active'] == 0 and relay.credentials.qsize() == 8

    def test_provider_image_detail_preserves_images_and_explicit_choices(self):
        body = {'messages':[{'role':'user','content':[
            {'type':'text','text':'Inspect both views'},
            {'type':'image_url','image_url':{'url':'data:first'}},
            {'type':'image_url','image_url':{'url':'data:second','detail':'high'}}]}]}
        out = payload_for_api(body, 'gpt-6.1-terra-azure', 'high', image_detail='auto')
        content = out['messages'][0]['content']
        self.assertEqual(content[1]['image_url'], {'url':'data:first','detail':'auto'})
        self.assertEqual(content[2]['image_url'], {'url':'data:second','detail':'high'})
        self.assertNotIn('detail', body['messages'][0]['content'][1]['image_url'])

    def test_priority_retains_reasoning_and_multimodal_payload(self):
        body = {'max_tokens':100000, 'service_tier':'default',
                'messages':[{'role':'user','content':[
                    {'type':'image_url','image_url':{'url':'data:image/png;base64,test'}}]}]}
        out = payload_for_api(body, 'gpt-5.6-terra', 'high', 'priority')
        self.assertEqual(out['service_tier'], 'priority')
        self.assertEqual(out['reasoning_effort'], 'high')
        self.assertEqual(out['max_tokens'], 100000)
        self.assertEqual(out['messages'], body['messages'])
        self.assertEqual(body['service_tier'], 'default')
        self.assertNotIn('service_tier', payload_for_api({}, 'gpt-5.6-terra', 'high'))

    def test_returned_service_tier_is_preserved_without_inventing_confirmation(self):
        event = {'choices':[{'delta':{'content':'{}'},'finish_reason':'stop'}]}
        self.assertNotIn('service_tier', assemble([json.dumps(event),'[DONE]']))
        event['service_tier'] = 'priority'
        self.assertEqual(assemble([json.dumps(event),'[DONE]'])['service_tier'], 'priority')

    def test_shape_high_and_budget_are_preserved(self):
        body={'model':'local-alias','max_tokens':100000,'temperature':0,'seed':42,
              'chat_template_kwargs':{'reasoning_effort':'high','clear_thinking':True}}
        out=payload_for_api(body,'glm-5.3-flash')
        self.assertEqual(out['model'],'glm-5.3-flash')
        self.assertEqual(out['max_tokens'],100000)
        self.assertEqual(out['reasoning_effort'],'high')
        self.assertNotIn('chat_template_kwargs',out)
        self.assertIn('chat_template_kwargs',body)

    def test_text_local_low_is_not_promoted(self):
        out=payload_for_api({'max_tokens':131072,'chat_template_kwargs':{'reasoning_effort':'low'}},'glm-5.3-flash')
        self.assertEqual(out['reasoning_effort'],'low')
        self.assertEqual(out['max_tokens'],131072)

    def test_arrow_unspecified_effort_remains_unspecified(self):
        out=payload_for_api({'max_tokens':113000,'chat_template_kwargs':{'enable_thinking':True}},'glm-5.3-flash')
        self.assertNotIn('reasoning_effort',out)
        self.assertEqual(out['thinking'],{'type':'enabled'})

    def test_stream_reasoning_usage_and_finish_are_preserved(self):
        events=[json.dumps({'id':'test','choices':[{'delta':{'reasoning_content':'inspect'}}]}),
                json.dumps({'choices':[{'delta':{'content':'{"ok":'}}]}),
                json.dumps({'choices':[{'delta':{'content':'true}'},'finish_reason':'stop'}]}),
                json.dumps({'choices':[],'usage':{'total_tokens':17}}),'[DONE]']
        out=assemble(events)
        self.assertEqual(out['choices'][0]['message']['content'],'{"ok":true}')
        self.assertEqual(out['choices'][0]['message']['reasoning_content'],'inspect')
        self.assertEqual(out['usage']['total_tokens'],17)

    def test_incomplete_output_is_rejected(self):
        for events in [[],['[DONE]'],[json.dumps({'choices':[{'delta':{'content':'partial'},'finish_reason':'stop'}]})]]:
            with self.assertRaises(IncompleteStream):assemble(events)

    def test_length_remains_visible_to_existing_validation(self):
        out=assemble([json.dumps({'choices':[{'delta':{'content':'{'},'finish_reason':'length'}]}),'[DONE]'])
        self.assertEqual(out['choices'][0]['finish_reason'],'length')

    def test_stalled_partial_stream_releases_slot_without_replaying(self):
        class StalledResponse:
            headers = {'Content-Type': 'text/event-stream'}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __iter__(self):
                yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n'
                raise TimeoutError()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            key = folder/'key'; key.write_text('test-credential'); key.chmod(0o600)
            relay = ApiRelay({'api_key_file':str(key), 'api_global_concurrency':1,
                'api_model':'test', 'api_base_url':'https://example.invalid/v1',
                'api_request_timeout_seconds':3600, 'api_idle_timeout_seconds':300}, folder)
            output = queue.Queue()
            with patch('full_recovery.api_gateway.urllib.request.build_opener') as build:
                build.return_value.open.return_value = StalledResponse()
                relay.produce({'messages':[]}, output)
                self.assertEqual(build.return_value.open.call_count, 1)
                self.assertLessEqual(build.return_value.open.call_args.kwargs['timeout'], 300)
            self.assertEqual([kind for kind, value in list(output.queue)], ['event','failed'])
            self.assertEqual(relay.stats['active'], 0)
            self.assertEqual(relay.stats['completed'], 0)
            self.assertEqual(relay.stats['failed'], 1)
            self.assertEqual(relay.credentials.qsize(), 1)
            self.assertTrue(relay.slots.acquire(blocking=False))

    def test_sse_keepalives_do_not_extend_the_generation_idle_budget(self):
        clock = [0.]
        class KeepaliveResponse:
            headers = {'Content-Type': 'text/event-stream'}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __iter__(self):
                for second in (100., 200., 301.):
                    clock[0] = second
                    yield b': keepalive\n'
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);key=folder/'key';key.write_text('test-key');key.chmod(0o600)
            relay=ApiRelay({'api_key_file':str(key),'api_global_concurrency':1,
                'api_model':'test','api_base_url':'https://example.invalid/v1',
                'api_idle_timeout_seconds':300},folder)
            output=queue.Queue()
            with patch('full_recovery.api_gateway.time.monotonic',side_effect=lambda:clock[0]), \
                 patch('full_recovery.api_gateway.urllib.request.build_opener') as build:
                build.return_value.open.return_value=KeepaliveResponse()
                relay.produce({'messages':[]},output)
            self.assertEqual(list(output.queue),[('failed','API stream idle deadline exceeded')])
            self.assertEqual(relay.stats['active'],0)

    def test_consumer_total_deadline_limits_upstream_without_retry(self):
        clock = [0.]
        class ActiveResponse:
            headers = {'Content-Type': 'text/event-stream'}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __iter__(self):
                for second in (.1, .6):
                    clock[0] = second
                    yield b'data: {"choices":[{"delta":{"reasoning_content":"inspect"}}]}\n'
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);key=folder/'key';key.write_text('test-key');key.chmod(0o600)
            relay=ApiRelay({'api_key_file':str(key),'api_global_concurrency':1,
                'api_model':'test','api_base_url':'https://example.invalid/v1',
                'api_request_timeout_seconds':3600},folder)
            output=queue.Queue()
            with patch('full_recovery.api_gateway.time.monotonic',side_effect=lambda:clock[0]), \
                 patch('full_recovery.api_gateway.urllib.request.build_opener') as build:
                build.return_value.open.return_value=ActiveResponse()
                relay.produce({'messages':[]},output,request_timeout=.5)
            assert build.return_value.open.call_count==1
            assert output.queue[-1]==('failed','API response deadline exceeded')
            assert relay.stats['active']==0 and relay.credentials.qsize()==1

    def test_total_deadline_closes_a_blocked_native_socket_and_releases_capacity(self):
        import threading
        import time
        import urllib.request
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        stop=threading.Event()
        class Slow(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(200)
                self.send_header('Content-Type','text/event-stream')
                self.end_headers()
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n')
                self.wfile.flush()
                stop.wait(3)
        upstream=ThreadingHTTPServer(('127.0.0.1',0),Slow)
        thread=threading.Thread(target=upstream.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                folder=Path(directory);key=folder/'key';key.write_text('test-key');key.chmod(0o600)
                relay=ApiRelay({'api_key_file':str(key),'api_global_concurrency':1,
                    'api_model':'test','api_base_url':f'http://127.0.0.1:{upstream.server_address[1]}',
                    'api_request_timeout_seconds':10,'api_idle_timeout_seconds':10},folder)
                output=queue.Queue();started=time.monotonic()
                with patch('full_recovery.api_gateway.urllib.request.build_opener',
                           return_value=urllib.request.build_opener(urllib.request.ProxyHandler({}))):
                    relay.produce({'messages':[]},output,request_timeout=.2)
                assert time.monotonic()-started<1.5
                assert relay.stats['failed']==1 and relay.stats['active']==0 and relay.credentials.qsize()==1
                assert output.queue[-1][0]=='failed'
        finally:
            stop.set();upstream.shutdown();upstream.server_close()


if __name__=='__main__':unittest.main()
