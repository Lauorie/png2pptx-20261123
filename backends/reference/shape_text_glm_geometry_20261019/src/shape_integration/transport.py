"""Local streaming transport with a cancellable total request deadline.

Keep the relay wire protocol and evidence layout. The deadline starts after
acquiring an API slot and covers headers plus the entire SSE stream, even while
new chunks keep arriving. Cancellation closes the async response before releasing
the slot; incomplete output is evidence only and never reaches a validator.
"""
import asyncio
import base64
import hashlib
import json
import math
from pathlib import Path
import time

import httpx

from png2pptx.api.client import save, parse_json
from png2pptx.images.clients import DetectorClient


class RequestTimeout(RuntimeError):
    """Terminal request failure: discard partial output and do not retry."""


class DeadlineDetectorClient(DetectorClient):
    def __init__(self, *args, request_timeout_seconds=600, write_timeout_seconds=30, **kwargs):
        super().__init__(*args, **kwargs)
        if (not isinstance(request_timeout_seconds, (int, float))
                or not math.isfinite(request_timeout_seconds) or request_timeout_seconds <= 0):
            raise ValueError('request_timeout_seconds must be finite and positive')
        self.request_timeout_seconds = request_timeout_seconds
        if not isinstance(write_timeout_seconds,(int,float)) or not math.isfinite(write_timeout_seconds) or write_timeout_seconds<=0:
            raise ValueError('write_timeout_seconds must be finite and positive')
        self.write_timeout_seconds=min(write_timeout_seconds,request_timeout_seconds)

    def call(self, tag, system, prompt, images, validator=None):
        if self.max_attempts != 1:
            raise ValueError('Streaming geometry experiments require exactly one attempt')
        folder = self.root/tag
        folder.mkdir(parents=True, exist_ok=False)
        content = [{'type': 'text', 'text': json.dumps(prompt, ensure_ascii=False)}]
        records = []
        for label, path in images:
            path = Path(path)
            binary = path.read_bytes()
            content += [{'type': 'text', 'text': label}, {'type': 'image_url', 'image_url': {
                'url': 'data:image/png;base64,'+base64.b64encode(binary).decode()}}]
            records.append({'label': label, 'path': str(path.resolve()), 'sha256': hashlib.sha256(binary).hexdigest()})
        parameters = self.request_parameters()
        wire = {'stream': True, 'stream_options': {'include_usage': True}}
        payload = {**parameters, **wire, 'messages': self.request_messages(system, content)}
        save(folder/'request.json', {'system': system, 'prompt': prompt, 'images': records,
             'parameters': parameters, 'wire_overrides': wire, 'base_url': self.base_url,
             'transport': {'connect_timeout_seconds': 15, 'read_timeout_seconds': self.read_timeout,
                           'write_timeout_seconds':self.write_timeout_seconds,
                           'request_timeout_seconds': self.request_timeout_seconds,
                           'deadline_starts': 'after_api_slot_acquired', 'timeout_policy': 'discard_no_retry', 'max_attempts': 1},
             'auth': 'Bearer key supplied in memory; value deliberately omitted' if self._key else 'none; direct local endpoint'})
        save(folder/'wire_request.json', payload)
        headers = {'Authorization': 'Bearer '+self._key} if self._key else {}
        text_parts, reasoning_parts = [], []
        timing = {'tag': tag, 'attempt': 0, 'first_output_s': None, 'first_content_s': None,
                  'last_output_s': None, 'output_chunks': 0, 'sse_done': False}
        metadata, usage, finish_reason = {}, None, None
        queued = time.monotonic()
        start = queued
        state = 'queued'
        save(folder/'stream_status.json', {**timing, 'state': state})

        async def consume():
            nonlocal usage, finish_reason
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.read_timeout, connect=15, write=self.write_timeout_seconds, pool=15),
                    trust_env=False, follow_redirects=False) as session:
                async with session.stream('POST', self.base_url+'/chat/completions', headers=headers, json=payload) as response:
                    with (folder/'http_0.txt').open('w') as raw_out, (folder/'events_0.jsonl').open('w') as events:
                        timing['headers_s'] = time.monotonic()-start
                        timing['http_status'] = response.status_code
                        if response.status_code != 200:
                            await response.aread()
                            body = self.redact(response.text)
                            raw_out.write(body)
                            raise RuntimeError(f'HTTP {response.status_code}: '+body[:2000])
                        last_status = start
                        async for line in response.aiter_lines():
                            elapsed = time.monotonic()-start
                            # Buffered/incessant chunks need not yield to the event
                            # loop; enforce the same deadline in that case too.
                            if elapsed >= self.request_timeout_seconds:
                                raise asyncio.TimeoutError()
                            line = self.redact(line)
                            raw_out.write(line+'\n')
                            if not line.startswith('data:'):
                                continue
                            body = line[5:].strip()
                            if body == '[DONE]':
                                timing['sse_done'] = True
                                break
                            chunk = json.loads(body)
                            events.write(json.dumps({'elapsed_s': elapsed, 'data': chunk}, ensure_ascii=False)+'\n')
                            if chunk.get('error'):
                                raise RuntimeError(json.dumps(chunk['error'], ensure_ascii=False))
                            for key in ['id', 'created', 'model', 'system_fingerprint']:
                                if key in chunk:
                                    metadata[key] = chunk[key]
                            usage = chunk.get('usage') or usage
                            for choice in chunk.get('choices', []):
                                finish_reason = choice.get('finish_reason') or finish_reason
                                delta = choice.get('delta') or {}
                                text = delta.get('content') or ''
                                reasoning = delta.get('reasoning_content') or delta.get('reasoning') or ''
                                if not isinstance(text, str) or not isinstance(reasoning, str):
                                    raise ValueError('Unexpected non-text stream delta')
                                if text or reasoning:
                                    if timing['first_output_s'] is None:
                                        timing['first_output_s'] = elapsed
                                    timing['last_output_s'] = elapsed
                                    timing['output_chunks'] += 1
                                if text:
                                    if timing['first_content_s'] is None:
                                        timing['first_content_s'] = elapsed
                                    text_parts.append(text)
                                if reasoning:
                                    reasoning_parts.append(reasoning)
                            if time.monotonic()-last_status >= 10:
                                raw_out.flush(); events.flush()
                                save(folder/'stream_status.json', {**timing, 'elapsed_s': elapsed,
                                     'content_chars': sum(map(len, text_parts)), 'reasoning_chars': sum(map(len, reasoning_parts)), 'state': 'running'})
                                last_status = time.monotonic()

        async def bounded_stream():
            await asyncio.wait_for(consume(), timeout=self.request_timeout_seconds)

        try:
            with self.slots:
                start = time.monotonic()
                timing['queue_seconds'] = start-queued
                state = 'running'
                save(folder/'stream_status.json', {**timing, 'state': state})
                try:
                    asyncio.run(bounded_stream())
                    if time.monotonic()-start >= self.request_timeout_seconds:
                        raise asyncio.TimeoutError()
                except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
                    detail = 'total deadline' if isinstance(exc, asyncio.TimeoutError) else type(exc).__name__
                    phase_limit=self.write_timeout_seconds if isinstance(exc,httpx.WriteTimeout) else self.request_timeout_seconds
                    raise RequestTimeout(f'Request timed out ({detail}; phase limit {phase_limit:g}s; total limit {self.request_timeout_seconds:g}s); output discarded; no retry') from None
            timing['seconds'] = time.monotonic()-start
            raw = {**metadata, 'object': 'chat.completion', 'usage': usage,
                   'choices': [{'index': 0, 'finish_reason': finish_reason, 'message': {
                       'role': 'assistant', 'content': ''.join(text_parts), 'reasoning_content': ''.join(reasoning_parts)}}],
                   '_provenance': {'assembled_from': 'events_0.jsonl', 'raw_sse': 'http_0.txt', 'sse_done': timing['sse_done']}}
            save(folder/'response_0.json', raw)
            if not timing['sse_done'] or finish_reason != 'stop':
                raise ValueError('Unfinished stream: '+str(finish_reason))
            parsed = parse_json(''.join(text_parts))
            save(folder/'parsed_response.json', parsed)
            result = validator(parsed) if validator else parsed
            save(folder/'result.json', result)
            event = {'tag': tag, 'attempt': 0, 'seconds': timing['seconds'], 'success': True,
                     'usage': usage, 'response_model': metadata.get('model')}
            with self.lock:
                with (self.root/'calls.jsonl').open('a') as out:
                    out.write(json.dumps(event)+'\n')
            state = 'ended'
            return result
        except Exception as exc:
            state = 'timed_out' if isinstance(exc, RequestTimeout) else 'failed'
            save(folder/'error_0.json', {'type': type(exc).__name__, 'error': self.redact(exc),
                 'seconds': time.monotonic()-start, 'discarded': state == 'timed_out',
                 'retryable': state != 'timed_out'})
            if isinstance(exc, RequestTimeout):
                raise
            raise RuntimeError(self.redact(exc)) from None
        finally:
            (folder/'content.txt').write_text(''.join(text_parts))
            (folder/'reasoning.txt').write_text(''.join(reasoning_parts))
            timing['seconds'] = timing.get('seconds', time.monotonic()-start)
            tokens = (usage or {}).get('completion_tokens')
            timing['completion_tokens'] = tokens
            timing['e2e_tokens_s'] = tokens/timing['seconds'] if tokens is not None else None
            timing['content_chars'] = sum(map(len, text_parts))
            timing['reasoning_chars'] = sum(map(len, reasoning_parts))
            save(folder/'stream_status.json', {**timing, 'state': state, 'finish_reason': finish_reason})
