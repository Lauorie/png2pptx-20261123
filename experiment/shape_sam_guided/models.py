"""Checkpointed bounded calls; model failures never erase baseline geometry."""
import base64
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time
import urllib.request
from contextlib import contextmanager
from core import read, save, sha
from request_io import ModelTransportError, request_json


class Models:
    def __init__(self, config):
        self.config = config
        self.api_mode = config.get('glm_transport') == 'api'
        self.slots = threading.BoundedSemaphore(config['glm_concurrency'])
        self.ports = queue.Queue()
        self.external_sam = bool(config.get('sam_urls'))
        for port in config.get('sam_urls') or config['sam_ports']:
            self.ports.put(port)
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.api_opener = urllib.request.build_opener() if self.api_mode else self.opener
        self.sam_cache = {}
        cache = config.get('sam_cache_run')
        if cache:
            checkpoint = read(Path(config['training_run'])/'checkpoints/best.json')
            for request in sorted((Path(cache)/'cases').glob('*/sam/*/masks/request.json')):
                result = request.with_name('predictions.json')
                if result.exists() and read(result).get('best_checkpoint') == checkpoint:
                    key = json.dumps(read(request),sort_keys=True)
                    self.sam_cache[key] = result

    def post(self, url, body, *, trace_path=None, attempts=3, busy_only=False, request_timeout=None):
        timeout=(self.config.get('glm_request_timeout_seconds',1200.)
                 if url.endswith('/chat/completions') else 1200.)
        if request_timeout is not None and url.endswith('/chat/completions'):
            timeout=min(timeout,request_timeout)
        if self.config.get('glm_transport') == 'api' and url.endswith('/chat/completions'):
            path = Path(self.config['glm_api_key_file'])
            if path.stat().st_mode & 0o077:
                raise ValueError('GLM credential file must have mode 0600')
            key = path.read_text().strip()
            if not key:
                raise ValueError('Empty GLM credential file')
            payload = copy.deepcopy(body)
            template = payload.pop('chat_template_kwargs', {})
            payload['model'] = self.config.get('glm_api_model', 'glm-5.3-flash')
            payload.setdefault('thinking', {'type': 'enabled'})
            effort = template.get('reasoning_effort')
            if effort is not None:
                payload.setdefault('reasoning_effort', effort)
            # Preserve semantic request/cache identity while CPU launchers use
            # the project's existing SSE relay for long upstream completions.
            relay = os.environ.get('GLM_API_RELAY_URL')
            destination = relay.rstrip('/')+'/chat/completions' if relay else url
            headers=({'X-PPT-Request-Timeout-Seconds':str(timeout)} if relay else {'Authorization':'Bearer '+key})
            return request_json(self.opener if relay else self.api_opener, destination, payload, trace_path=trace_path,
                                attempts=1 if relay else attempts, timeout=timeout, headers=headers)
        return request_json(self.opener, url, body, trace_path=trace_path, attempts=attempts,timeout=timeout,busy_only=busy_only)

    @contextmanager
    def sam_endpoint_slot(self, endpoint):
        # Per-instance queues cannot serialize separate agent processes using
        # one non-queued external SAM service. The mounted lock is per endpoint.
        root=Path(self.config.get('project_root',Path(__file__).resolve().parents[2]))/'.runtime/sam_endpoint_locks'
        root.mkdir(parents=True,exist_ok=True)
        name=hashlib.sha256(str(endpoint).rstrip('/').encode()).hexdigest()
        with (root/name).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            yield

    def ask(self, folder, system, prompt, images, validator=lambda x: x, budget=16000):
        folder = Path(folder)
        identity = {'system': system, 'prompt': prompt, 'max_tokens': budget,
            'images': [{'label': label, 'sha256': sha(p)} for label, p in images]}
        if self.api_mode:
            identity['api_provider'] = {'base_url': self.config['glm_url'],
                                       'model': self.config.get('glm_api_model', 'glm-5.3-flash')}
        if (folder/'request.json').exists() and read(folder/'request.json') != identity:
            raise ValueError('Cached request changed: '+str(folder))
        save(folder/'request.json', identity)
        if (folder/'result.json').exists():
            return validator(read(folder/'result.json'))
        failure = None
        for attempt in range(3):
            content = [{'type': 'text', 'text': json.dumps(prompt, ensure_ascii=False)}]
            if failure:
                content.append({'type': 'text', 'text': 'Correct the previous response: '+failure[:1800]})
            for label, path in images:
                payload = base64.b64encode(Path(path).read_bytes()).decode()
                content.extend([{'type': 'text', 'text': label}, {'type': 'image_url', 'image_url':
                    {'url': 'data:image/png;base64,'+payload}}])
            body = {'model': 'GLM-5.3-Flash', 'messages': [{'role': 'system', 'content': system},
                {'role': 'user', 'content': content}], 'temperature': 0, 'max_tokens': budget,
                'chat_template_kwargs': {'reasoning_effort': 'high', 'clear_thinking': True}}
            start = time.monotonic()
            try:
                with self.slots:
                    response = self.post(self.config['glm_url']+'/chat/completions', body,
                                         trace_path=folder/f'transport_{attempt}.json')
                save(folder/f'response_{attempt}.json', response)
                choice = response['choices'][0]
                if choice.get('finish_reason') == 'length':
                    raise ValueError('Truncated response; produce compact complete JSON')
                text = choice['message']['content'] or ''
                pos = text.find('{')
                if pos < 0:
                    raise ValueError('No JSON object')
                raw, _ = json.JSONDecoder().raw_decode(text[pos:])
                result = validator(raw)
                save(folder/'result.json', raw)
                save(folder/'timing.json', {'seconds': time.monotonic()-start, 'usage': response.get('usage')})
                return result
            except ModelTransportError:
                # Transport already retried the SAME request. Do not consume a
                # schema attempt or change the prompt because a service failed.
                raise
            except Exception as exc:
                failure = f'{type(exc).__name__}: {exc}'
                save(folder/f'error_{attempt}.json', {'error': failure, 'seconds': time.monotonic()-start})
        raise RuntimeError('Model call failed: '+str(failure))

    def sam(self, image, folder, prompts=None):
        folder = Path(folder)
        identity = {'image_sha256': sha(image), 'threshold': self.config['sam_threshold'], 'prompts': prompts}
        if self.external_sam and self.config.get('external_sam_epoch') is not None:
            identity['external_checkpoint']={'epoch':self.config['external_sam_epoch'],
                'sha256':self.config.get('external_sam_checkpoint_sha256')}
        if (folder/'request.json').exists() and read(folder/'request.json') != identity:
            raise ValueError('SAM cache input/threshold changed')
        save(folder/'request.json', identity)
        if (folder/'predictions.json').exists():
            result=read(folder/'predictions.json')
            return self.verify_external_sam(result) if self.external_sam else result
        # Reuse the exact detector request across runs, even when the fitted
        # scene changed. Target-mask selection is recomputed by the caller.
        output_root=self.config.get('sam_output_root')
        if output_root and folder.is_relative_to(Path(output_root)):
            for reference in self.config.get('sam_reuse_roots',[]):
                previous=Path(reference)/folder.relative_to(Path(output_root))
                if not (previous/'request.json').is_file() or not (previous/'predictions.json').is_file():continue
                if read(previous/'request.json')!=identity:continue
                result=read(previous/'predictions.json')
                if not all(Path(p['mask_path']).is_file() for p in result['predictions']):continue
                if self.external_sam:result=self.verify_external_sam(result)
                save(folder/'predictions.json',result)
                save(folder/'cache_reuse.json',{'source':str(previous/'predictions.json'),
                    'request_identity':identity,'reason':'identical pixels, prompts, threshold and checkpoint'})
                return result
        key = json.dumps(identity,sort_keys=True)
        if key in self.sam_cache:
            result = read(self.sam_cache[key])
            if all(Path(p['mask_path']).is_file() for p in result['predictions']):
                save(folder/'predictions.json', result)
                save(folder/'cache_reuse.json', {'source':str(self.sam_cache[key]),'request_identity':identity})
                return result
        port = self.ports.get()
        try:
            if self.external_sam:
                output=folder/'service_output'
                if output.exists() and not (output/'predictions.json').exists() and self.config.get('sam_attempt_id'):
                    # The old service may finish after its client was stopped.
                    # A new attempt uses a distinct service-owned output path.
                    output=folder/('service_output_'+self.config['sam_attempt_id'])
                if (output/'predictions.json').exists():
                    result=read(output/'predictions.json')
                else:
                    if output.exists():raise RuntimeError('Prior external SAM output is incomplete; do not start a second writer')
                    categories=([p['name'] for p in prompts] if prompts else
                        [p['name'] for p in read(Path(self.config['dataset_root'])/'dataset_manifest.json')['categories']
                         if p['name'] not in {'line','block_arrow'}])
                    with self.sam_endpoint_slot(port):
                        result=self.post(str(port).rstrip('/')+'/',
                            {'image':str(image),'output':str(output),'categories':categories,
                             'threshold':self.config['sam_threshold']},attempts=120,busy_only=True,
                            trace_path=folder/'transport.json')
                result=self.verify_external_sam(result)
                save(folder/'predictions.json',result)
                return result
            # The SAM service writes this output directory itself. A timed-out
            # request may still be writing, so never issue a duplicate writer.
            return self.post(f'http://127.0.0.1:{port}/', {'image': str(image), 'output': str(folder), 'prompts': prompts}, attempts=1)
        finally:
            self.ports.put(port)

    def verify_external_sam(self, result):
        """Accept both service envelopes while checking the actual served weights."""
        checkpoint=result.get('best_checkpoint') or result.get('checkpoint') or {}
        expected=self.config.get('external_sam_epoch')
        if expected is not None and checkpoint.get('epoch')!=expected:
            raise ValueError('External SAM checkpoint epoch differs from the configured model')
        expected_sha=self.config.get('external_sam_checkpoint_sha256')
        actual_sha=checkpoint.get('sha256') or result.get('model',{}).get('checkpoint_sha256')
        if expected_sha and actual_sha!=expected_sha:
            raise ValueError('External SAM checkpoint hash differs from the configured model')
        return {**result,'best_checkpoint':checkpoint}


class OriginalRuntime:
    """Adapter for unchanged original observation and critic entry points."""
    def __init__(self, models, evidence=None):
        self.models, self.evidence = models, evidence

    def for_stage(self, **kwargs):
        return self

    def call(self, folder, name, system, prompt, images, validator, max_tokens, **kwargs):
        images = list(images)
        prompt = copy.deepcopy(prompt)
        if name == 'observe' and self.evidence:
            images.append(('SAM boundary/bbox evidence in the SAME crop coordinates; inspect source first', self.evidence))
            prompt['auxiliary_evidence'] = 'SAM is uncertain. Use its box for identity and complete reliable boundary for observation; correct fragments against the clean source. Never trace annotation text or bounding-box strokes.'
        return self.models.ask(Path(folder)/(name+'_00'), system, prompt, images, validator,
                               min(max_tokens, self.models.config['glm_max_tokens']))
