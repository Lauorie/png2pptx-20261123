"""Host-side clients for the rjob GLM/SAM services.

``StyleRuntime`` implements the ``Runtime.for_stage().call()`` contract used by
``shape_integration.styling``: one request per ``{tag}_{attempt}`` folder, cached
``result.json`` replay, ``response_0.json`` with ``finish_reason`` for the style
format-correction logic, and ``RequestTimeout`` for terminal timeouts. Both arms
use the same transport, model, sampling and budget cap.
"""
import base64
import copy
import hashlib
import http.client
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_cluster():
    if os.environ.get('STYLE_CLUSTER_WORKER') != '1':
        raise RuntimeError('Batch style work is restricted to rjob. Use submit_resume_gpu.sh; '
                           'run_cluster.sh sets the worker environment and bounds CPU concurrency.')


OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class TransportError(RuntimeError):
    """No usable service response; never shown to the model as feedback."""


def post(url, body, timeout):
    request = urllib.request.Request(url, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    started = time.monotonic()
    last = None
    for attempt in range(3):
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 1:
            break
        try:
            with OPENER.open(request, timeout=remaining) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            last = f'HTTP {exc.code}: {exc.read()[:500]!r}'
            exc.close()
            if exc.code not in (408, 429, 500, 502, 503, 504):
                raise TransportError(last)
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as exc:
            last = f'{type(exc).__name__}: {exc}'
            if isinstance(exc, TimeoutError) or 'timed out' in str(exc):
                raise TransportError('timeout: ' + last)
        time.sleep(min(8., 2. ** attempt))
    raise TransportError(last or 'deadline exceeded')


def services(run_dir, kind):
    status = read(Path(os.environ.get('STYLE_SERVICE_DIR', run_dir)) / 'status.json')
    if status.get('phase') != 'serving':
        raise RuntimeError('Service host is not serving: ' + str(status.get('phase')))
    if time.time() - status['heartbeat_at'] > 120:
        raise RuntimeError('Service heartbeat is stale')
    rows = [r for r in status['services'][kind] if r['alive'] and r['phase'] == 'ready']
    if not rows:
        raise RuntimeError('No ready ' + kind + ' service')
    return [r['url'] for r in rows]


class Slots(threading.BoundedSemaphore):
    def __init__(self, limit):
        super().__init__(limit)
        self.lock = threading.Lock()
        self.limit, self.active, self.peak = limit, 0, 0

    def __enter__(self):
        self.acquire()
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        return self

    def __exit__(self, *exc):
        with self.lock:
            self.active -= 1
        self.release()
        return False

    def snapshot(self):
        with self.lock:
            return {'limit': self.limit, 'active': self.active, 'peak': self.peak}


class Endpoints:
    """Least-loaded choice among ready GLM replicas, refreshed from status.json."""
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.lock = threading.Lock()
        self.load = {}

    def acquire(self):
        urls = services(self.run_dir, 'glm')
        with self.lock:
            url = min(urls, key=lambda u: self.load.get(u, 0))
            self.load[url] = self.load.get(url, 0) + 1
        return url

    def release(self, url):
        with self.lock:
            self.load[url] -= 1


class StyleRuntime:
    def __init__(self, run_dir, settings, timeout_error, *, endpoints=None, slots=None):
        self.run_dir = Path(run_dir)
        self.settings = settings
        self.timeout_error = timeout_error
        self.endpoints = endpoints or Endpoints(run_dir)
        self.slots = slots or Slots(settings['concurrency'])
        self.config = {'model': 'GLM-5.3-Flash@local-vllm', 'request_timeout_seconds': settings['timeout_seconds'],
                       'shape_schema_retries': 0, 'reasoning_effort': settings['reasoning_effort']}

    def for_stage(self, **overrides):
        stage = copy.copy(self)
        stage.config = {**self.config, **overrides}
        return stage

    def identity(self, system, prompt, images, budget):
        return {'system': system, 'prompt': prompt, 'max_tokens': budget, 'model': 'GLM-5.3-Flash',
                'seed': self.settings['seed'], 'temperature': 0,
                'reasoning_effort': self.settings['reasoning_effort'],
                'images': [{'label': label, 'sha256': sha(path)} for label, path in images]}

    def call(self, root, tag, system, prompt, images, validator, budget, *, retry_builder=None):
        budget = min(budget, self.settings['max_tokens'])
        error = None
        for attempt in range(self.config.get('shape_schema_retries', 0) + 1):
            actual = copy.deepcopy(prompt)
            if error:
                actual.update(validation_error=error,
                              correction='Inspect the same images and return a COMPLETE valid JSON response.')
            folder = Path(root) / f'{tag}_{attempt:02d}'
            identity = self.identity(system, actual, images, budget)
            if (folder / 'request.json').exists() and read(folder / 'request.json') != identity:
                raise ValueError('Request identity changed: ' + str(folder))
            save(folder / 'request.json', identity)
            if (folder / 'result.json').exists():
                return validator(read(folder / 'result.json'))
            if (folder / 'error_0.json').exists():
                saved = read(folder / 'error_0.json')
                if saved.get('type') == 'RequestTimeout':
                    raise self.timeout_error(saved['error'])
                error = saved['error']
                continue
            content = [{'type': 'text', 'text': json.dumps(actual, ensure_ascii=False)}]
            for label, path in images:
                content += [{'type': 'text', 'text': label}, {'type': 'image_url', 'image_url':
                            {'url': 'data:image/png;base64,' + base64.b64encode(Path(path).read_bytes()).decode()}}]
            body = {'model': 'GLM-5.3-Flash', 'messages': [{'role': 'system', 'content': system},
                    {'role': 'user', 'content': content}], 'temperature': 0, 'seed': self.settings['seed'],
                    'max_tokens': budget,
                    'chat_template_kwargs': {'reasoning_effort': self.settings['reasoning_effort'], 'clear_thinking': True}}
            started = time.monotonic()
            with self.slots:
                acquired = time.monotonic()
                url = self.endpoints.acquire()
                try:
                    response = post(url + '/chat/completions', body, self.settings['timeout_seconds'])
                except TransportError as exc:
                    if str(exc).startswith('timeout'):
                        save(folder / 'error_0.json', {'error': str(exc), 'type': 'RequestTimeout'})
                        raise self.timeout_error(str(exc))
                    # Transport failures are not cached; a resumed run reissues the request.
                    raise RuntimeError('transport: ' + str(exc))
                finally:
                    self.endpoints.release(url)
            save(folder / 'response_0.json', response)
            save(folder / 'timing.json', {'queue_seconds': acquired - started,
                                          'service_seconds': time.monotonic() - acquired,
                                          'endpoint': url, 'usage': response.get('usage', {})})
            try:
                choice = response['choices'][0]
                text = choice['message'].get('content') or ''
                (folder / 'content.txt').write_text(text)
                if choice.get('finish_reason') == 'length':
                    raise ValueError('Response truncated; return compact complete JSON')
                if '{' not in text:
                    raise ValueError('No JSON object in response')
                parser = getattr(self, 'parse_response', None)
                raw = parser(text) if parser else json.JSONDecoder().raw_decode(text[text.index('{'):])[0]
                result = validator(raw)
                save(folder / 'result.json', raw)
                return result
            except (ValueError, RuntimeError, KeyError, TypeError) as exc:
                error = f'{type(exc).__name__}: {exc}'
                save(folder / 'error_0.json', {'error': error, 'type': type(exc).__name__})
        raise RuntimeError(error or 'Model call failed')


class SamClient:
    """One in-flight request per SAM service; masks are written by the service."""
    def __init__(self, run_dir, timeout=600):
        import queue
        self.queue = queue.Queue()
        for url in services(run_dir, 'sam'):
            self.queue.put(url)
        self.timeout = timeout

    def __call__(self, image, output, prompts, overlay=False):
        output = Path(output)
        identity = {'image_sha256': sha(image), 'prompts': prompts}
        if (output / 'request.json').exists() and read(output / 'request.json') == identity \
                and (output / 'predictions.json').exists():
            return read(output / 'predictions.json')
        save(output / 'request.json', identity)
        url = self.queue.get()
        try:
            return post(url + '/', {'image': str(image), 'output': str(output), 'prompts': prompts,
                                    'overlay': overlay}, self.timeout)
        finally:
            self.queue.put(url)
