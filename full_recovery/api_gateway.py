"""Loopback-only relay: one authenticated streaming API and one shared request limit.

Existing shape JSON and reference text/arrow SSE clients keep their contracts.
No model runs here. Headers, credentials, image data and prompts are never logged.
"""
import copy
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import queue
import socket
import threading
import time
import urllib.error
import urllib.request
import hashlib
import uuid
from .api_budget import BudgetGuard, BudgetBusy, BudgetPaused


def payload_for_api(body, model, force_effort=None, service_tier=None, image_detail=None):
    payload = copy.deepcopy(body)
    template = payload.pop('chat_template_kwargs', {})
    payload['model'] = model
    if image_detail is not None:
        if image_detail not in ('auto', 'low', 'high'):
            raise ValueError('Invalid default image detail')
        for message in payload.get('messages', []):
            content = message.get('content')
            for part in content if isinstance(content, list) else []:
                if part.get('type') == 'image_url' and isinstance(part.get('image_url'), dict):
                    part['image_url'].setdefault('detail', image_detail)
    if service_tier is not None:
        if service_tier not in ('auto', 'default', 'flex', 'priority'):
            raise ValueError('Invalid API service tier')
        payload['service_tier'] = service_tier
    if 'enable_thinking' in template:
        payload.setdefault('thinking', {'type': 'enabled' if template['enable_thinking'] else 'disabled'})
    else:
        payload.setdefault('thinking', {'type': 'enabled'})
    if template.get('reasoning_effort') is not None:
        payload.setdefault('reasoning_effort', template['reasoning_effort'])
    if force_effort is not None:
        if force_effort not in ('low', 'medium', 'high'):
            raise ValueError('Invalid forced reasoning effort')
        payload['reasoning_effort'] = force_effort
        payload['thinking'] = {'type': 'enabled'}
    if model.rsplit('/', 1)[-1].startswith('gpt-'):
        # OpenAI reasoning routes do not use GLM template/sampling controls.
        for name in ('thinking', 'temperature', 'seed', 'top_p'):
            payload.pop(name, None)
    payload.update(stream=True, stream_options={'include_usage': True})
    return payload


class IncompleteStream(RuntimeError):
    pass


def assemble(events):
    content, reasoning, metadata = [], [], {}
    usage, finish, done = None, None, False
    for raw in events:
        if raw == '[DONE]':
            done = True
            break
        event = json.loads(raw)
        if event.get('error'):
            raise IncompleteStream('API returned a stream error')
        for key in ('id', 'model', 'created', 'system_fingerprint', 'service_tier'):
            if key in event:
                metadata[key] = event[key]
        usage = event.get('usage') or usage
        for choice in event.get('choices', []):
            if choice.get('index', 0) != 0:
                raise IncompleteStream('Only a single completion is supported')
            finish = choice.get('finish_reason') or finish
            delta = choice.get('delta') or {}
            text = delta.get('content') or ''
            think = delta.get('reasoning_content') or delta.get('reasoning') or ''
            if not isinstance(text, str) or not isinstance(think, str):
                raise IncompleteStream('Non-text API delta')
            content.append(text)
            reasoning.append(think)
    if not done or finish is None:
        raise IncompleteStream('Incomplete API stream; partial output discarded')
    if finish not in ('stop', 'length', 'content_filter'):
        raise IncompleteStream('Unsuccessful API stream finish; partial output discarded')
    return {**metadata, 'object': 'chat.completion', 'usage': usage,
            'choices': [{'index': 0, 'finish_reason': finish, 'message': {
                'role': 'assistant', 'content': ''.join(content), 'reasoning_content': ''.join(reasoning)}}]}


class RequestPacer:
    """One start schedule and rate-limit cooldown shared by all relay threads."""
    def __init__(self, interval=0., cooldown=0.):
        self.interval, self.cooldown = max(0., float(interval)), max(0., float(cooldown))
        self.next_start = self.cooldown_until = 0.
        self.lock = threading.Lock()

    def wait(self, deadline, cancelled):
        while True:
            if cancelled.is_set():
                raise IncompleteStream('API consumer disconnected')
            with self.lock:
                now = time.monotonic()
                if now >= deadline:
                    raise IncompleteStream('API rate-limit queue deadline exceeded')
                delay = max(self.next_start, self.cooldown_until) - now
                if delay <= 0:
                    self.next_start = now + self.interval
                    return
            cancelled.wait(min(delay, 1., max(0., deadline - now)))

    def limited(self, retry_after=None):
        try: delay = max(self.cooldown, float(retry_after or 0))
        except (TypeError, ValueError): delay = self.cooldown
        with self.lock:
            self.cooldown_until = max(self.cooldown_until, time.monotonic() + delay)


class ApiRelay:
    def __init__(self, settings, folder):
        self.settings, self.folder = settings, Path(folder)
        self.keys=[]
        for path in settings.get('api_key_files',[settings['api_key_file']]):
            key_file=Path(path)
            if key_file.stat().st_mode & 0o077:
                raise ValueError('API credential file must have mode 0600')
            key=key_file.read_text().strip()
            if not key or any(c.isspace() for c in key):
                raise ValueError('Invalid API credential file')
            self.keys.append(key)
        self.credentials=queue.Queue()
        for index in range(settings['api_global_concurrency']):
            self.credentials.put(self.keys[index%len(self.keys)])
        self.slots = threading.BoundedSemaphore(settings['api_global_concurrency'])
        self.lock = threading.Lock()
        self.stats = dict(requests=0, queued=0, active=0, completed=0, failed=0, peak_active=0,
                          upstream_attempts=0, upstream_events=0, upstream_rate_limited=0)
        self.pacer = RequestPacer(settings.get('api_request_start_interval_seconds', 0),
                                  settings.get('api_rate_limit_cooldown_seconds', 0))
        self.budget = BudgetGuard(settings['api_budget_config']) if settings.get('api_budget_config') else None

    def stamp(self, **changes):
        with self.lock:
            for key, delta in changes.items():
                self.stats[key] += delta
            self.stats['peak_active'] = max(self.stats['peak_active'], self.stats['active'])
            row = {**self.stats, 'at': time.time(), 'phase': 'ready',
                   'requested_service_tier': self.settings.get('api_service_tier'),
                   'model': self.settings['api_model'], 'limit': self.settings['api_global_concurrency'],
                   'request_start_interval_seconds': self.pacer.interval}
            tmp = self.folder/'api_metrics.json.tmp'
            tmp.write_text(json.dumps(row, indent=2)+'\n')
            tmp.replace(self.folder/'api_metrics.json')
            return row

    def produce(self, body, output, request_timeout=None, cancelled=None):
        started=time.monotonic();upstream_status=None;field_hint=None
        self.stamp(requests=1, queued=1)
        cancelled = cancelled or threading.Event()
        budget = self.settings.get('api_request_timeout_seconds', 3600)
        if request_timeout is not None:
            budget = min(budget, request_timeout)
        deadline = time.monotonic()+budget
        acquired = False;credential=None
        events = []
        watchdog = None
        try:
            acquired = self.slots.acquire(timeout=max(.1, deadline-time.monotonic()))
            self.stamp(queued=-1)
            if not acquired:
                raise IncompleteStream('API queue deadline exceeded')
            self.stamp(active=1)
            credential=self.credentials.get_nowait()
            payload = payload_for_api(body, self.settings['api_model'],
                                      self.settings.get('api_force_reasoning_effort'),
                                      self.settings.get('api_service_tier'),
                                      self.settings.get('api_default_image_detail'))
            request = urllib.request.Request(self.settings['api_base_url'].rstrip('/')+'/chat/completions',
                json.dumps(payload).encode(), {'Content-Type': 'application/json', 'Authorization': 'Bearer '+credential})
            opener = urllib.request.build_opener()
            # Retry only before any completion event has reached a consumer.
            attempts = max(3, int(self.settings.get('api_rate_limit_attempts', 3)))
            for attempt in range(attempts):
                if cancelled.is_set():
                    raise IncompleteStream('API consumer disconnected')
                if time.monotonic() >= deadline:
                    raise IncompleteStream('API response deadline exceeded')
                self.pacer.wait(deadline, cancelled)
                reservation=None
                if self.budget:
                    while True:
                        if cancelled.is_set() or time.monotonic() >= deadline:
                            raise IncompleteStream('Budget queue cancelled or deadline exceeded')
                        try:
                            reservation=self.budget.reserve(payload,self.folder)
                            break
                        except BudgetBusy:
                            cancelled.wait(1)
                receipt={'id':reservation or uuid.uuid4().hex, 'at':time.time(),
                    'model':payload['model'],'reasoning_effort':payload.get('reasoning_effort'),
                    'requested_service_tier':payload.get('service_tier'),
                    'max_tokens':payload.get('max_tokens'),
                    'request_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),
                    'attempt':attempt}
                settled=False
                self.stamp(upstream_attempts=1)
                try:
                    # A silent upstream must not occupy a slot for the whole
                    # generation budget. Streaming output still gets the full
                    # deadline; this socket timeout only bounds an idle read.
                    idle = self.settings.get('api_idle_timeout_seconds', 300)
                    with opener.open(request, timeout=max(.1, min(idle, deadline-time.monotonic()))) as response:
                        def close_expired_response():
                            # A blocked idle read must end at the same total
                            # deadline as an actively streaming response.
                            raw=getattr(getattr(response,'fp',None),'raw',None)
                            sock=getattr(raw,'_sock',None)
                            if sock is not None:
                                try:sock.shutdown(socket.SHUT_RDWR)
                                except OSError:pass
                        watchdog=threading.Timer(max(.001,deadline-time.monotonic()),close_expired_response)
                        watchdog.daemon=True;watchdog.start()
                        if 'text/event-stream' not in response.headers.get('Content-Type', ''):
                            raise IncompleteStream('API did not return an SSE stream')
                        last_stamp = time.monotonic()
                        last_event = last_stamp
                        count = 0
                        for line in response:
                            now = time.monotonic()
                            if cancelled.is_set():
                                raise IncompleteStream('API consumer disconnected')
                            if self.budget and self.budget.pause_path.exists():
                                raise BudgetPaused('Task budget paused during generation')
                            if now >= deadline:
                                raise IncompleteStream('API response deadline exceeded')
                            if now-last_event >= idle:
                                raise IncompleteStream('API stream idle deadline exceeded')
                            text = line.decode('utf-8').strip()
                            if not text.startswith('data:'):
                                continue
                            data = text[5:].strip()
                            last_event = now
                            # SSE content is forwarded; never publish an auth value.
                            for secret in self.keys:
                                data=data.replace(secret,'<REDACTED>')
                            events.append(data)
                            output.put(('event', data))
                            count += 1
                            if time.monotonic()-last_stamp >= 5:
                                self.stamp(upstream_events=count)
                                count = 0
                                last_stamp = time.monotonic()
                            if data == '[DONE]':
                                break
                        self.stamp(upstream_events=count)
                    result = assemble(events)
                    receipt.update(returned_model=result.get('model'),usage=result.get('usage'),
                                   returned_service_tier=result.get('service_tier'),
                                   finish_reason=result['choices'][0]['finish_reason'])
                    if self.budget:
                        settled=True
                        self.budget.settle(reservation,result.get('usage'))
                    self.stamp(completed=1)
                    output.put(('complete', result))
                    return
                except urllib.error.HTTPError as exc:
                    status = exc.code
                    upstream_status=status
                    retry_after = exc.headers.get('Retry-After') if exc.headers else None
                    # Classify parameter errors without persisting the remote
                    # body, which may echo credentials or user payloads.
                    try:raw_error=exc.read(4096).decode('utf-8',errors='replace')
                    except (OSError,AttributeError):raw_error=''
                    field_hint=next((field for field in ('reasoning_effort','max_tokens','context_length') if field in raw_error),None)
                    exc.close()
                    if status == 402 and self.budget:
                        self.budget.pause('Provider rejected billing/quota (HTTP 402); new requests disabled')
                    if status == 429:
                        self.stamp(upstream_rate_limited=1)
                        self.pacer.limited(retry_after)
                    if (status not in (408, 429, 500, 502, 503, 504)
                            or attempt >= (attempts - 1 if status == 429 else 2)):
                        raise IncompleteStream('API HTTP '+str(status)) from None
                    cancelled.wait(min(8, 2**attempt, max(0., deadline-time.monotonic())))
                except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                    if events or attempt >= 2:
                        raise IncompleteStream('API network failure: '+type(exc).__name__) from None
                    cancelled.wait(min(8, 2**attempt, max(0., deadline-time.monotonic())))
                finally:
                    if self.budget and not settled:
                        self.budget.settle(reservation)
                    receipt['elapsed_seconds']=time.monotonic()-started
                    with self.lock:
                        with (self.folder/'api_usage.jsonl').open('a') as log:
                            log.write(json.dumps(receipt)+'\n')
                    if watchdog is not None:
                        watchdog.cancel()
                        watchdog=None
            raise IncompleteStream('API transport retry budget exhausted')
        except Exception as exc:
            self.stamp(failed=1)
            # Do not include upstream exception text or HTTP bodies in logs.
            detail = str(exc) if isinstance(exc, (IncompleteStream, BudgetPaused)) else type(exc).__name__
            diagnostic={'at':time.time(),'elapsed_seconds':time.monotonic()-started,
                'upstream_status':upstream_status,'field_hint':field_hint,'error':detail,
                'reasoning_effort':body.get('reasoning_effort',(body.get('chat_template_kwargs') or {}).get('reasoning_effort')),
                'max_tokens':body.get('max_tokens') if type(body.get('max_tokens')) is int else None,'completion_events':len(events)}
            if diagnostic['reasoning_effort'] not in (None,'low','medium','high'):diagnostic['reasoning_effort']='<invalid>'
            with self.lock:
                with (self.folder/'api_failures.jsonl').open('a') as log:log.write(json.dumps(diagnostic)+'\n')
            output.put(('failed', detail))
        finally:
            if watchdog is not None:watchdog.cancel()
            if acquired:
                self.stamp(active=-1)
                if credential is not None:self.credentials.put(credential)
                self.slots.release()

    def server(self, port):
        relay = self
        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *args):
                pass

            def reply(self, status, obj):
                data = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Connection', 'close')
                self.end_headers()
                self.wfile.write(data)
                self.close_connection = True

            def do_GET(self):
                if self.path != '/health':
                    return self.reply(404, {'error': 'Unknown route'})
                self.reply(200, relay.stamp())

            def do_POST(self):
                if self.path != '/v1/chat/completions':
                    return self.reply(404, {'error': 'Unknown route'})
                try:
                    body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                except (ValueError, TypeError, KeyError):
                    return self.reply(400, {'error': 'Invalid JSON request'})
                streaming = bool(body.get('stream'))
                request_timeout = None
                timeout_header = self.headers.get('X-PPT-Request-Timeout-Seconds')
                if timeout_header is not None:
                    try:
                        request_timeout = float(timeout_header)
                        if not math.isfinite(request_timeout) or request_timeout <= 0:
                            raise ValueError()
                    except ValueError:
                        return self.reply(400, {'error': 'Invalid request deadline'})
                messages = queue.Queue()
                cancelled = threading.Event()
                threading.Thread(target=relay.produce, args=(body, messages, request_timeout, cancelled), daemon=True).start()
                if streaming:
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Connection', 'close')
                    self.end_headers()
                try:
                    while True:
                        try:
                            kind, value = messages.get(timeout=10)
                        except queue.Empty:
                            if streaming:
                                self.wfile.write(b': API queue/stream heartbeat\n\n')
                                self.wfile.flush()
                            continue
                        if kind == 'event' and streaming:
                            # Publish DONE only after verifying the complete stream.
                            if value != '[DONE]':
                                self.wfile.write(('data: '+value+'\n\n').encode())
                                self.wfile.flush()
                        elif kind == 'complete':
                            if streaming:
                                self.wfile.write(b'data: [DONE]\n\n')
                                self.wfile.flush()
                            else:
                                self.reply(200, value)
                            return
                        elif kind == 'failed':
                            if not streaming:
                                self.reply(502, {'error': {'message': value}})
                            # Incomplete SSE is rejected by the preserved clients.
                            return
                except (BrokenPipeError, ConnectionResetError):
                    cancelled.set()
                    return
                finally:
                    self.close_connection = True
        server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        server.daemon_threads = True
        return server
