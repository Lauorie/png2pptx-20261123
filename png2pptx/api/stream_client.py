"""Single-attempt streaming transport with raw SSE evidence and assembled response."""
import base64
import hashlib
import json
from pathlib import Path
import time

import httpx

from .client import save, parse_json
from .relay_client import RelayClient


class StreamingRelayClient(RelayClient):
    def call(self, tag, system, prompt, images, validator=None):
        if self.max_attempts != 1:
            raise ValueError('Streaming geometry experiments require exactly one attempt')
        folder = self.root / tag
        folder.mkdir(parents=True, exist_ok=False)
        content = [{'type': 'text', 'text': json.dumps(prompt, ensure_ascii=False)}]
        records = []
        for label, path in images:
            path = Path(path)
            binary = path.read_bytes()
            content += [{'type': 'text', 'text': label}, {'type': 'image_url', 'image_url': {
                'url': 'data:image/png;base64,' + base64.b64encode(binary).decode()}}]
            records.append({'label': label, 'path': str(path.resolve()), 'sha256': hashlib.sha256(binary).hexdigest()})
        parameters = self.request_parameters()
        wire = {'stream': True, 'stream_options': {'include_usage': True}}
        payload = {**parameters, **wire, 'messages': self.request_messages(system, content)}
        save(folder / 'request.json', {'system': system, 'prompt': prompt, 'images': records,
             'parameters': parameters, 'wire_overrides': wire, 'base_url': self.base_url,
             'transport': {'connect_timeout_seconds': 15, 'read_timeout_seconds': self.read_timeout, 'max_attempts': 1},
             'auth': 'Bearer key supplied in memory; value deliberately omitted' if self._key else 'none; direct local endpoint'})
        save(folder / 'wire_request.json', payload)
        headers = {'Authorization': 'Bearer ' + self._key} if self._key else {}
        text_parts, reasoning_parts = [], []
        timing = {'tag': tag, 'attempt': 0, 'first_output_s': None, 'first_content_s': None,
                  'last_output_s': None, 'output_chunks': 0, 'sse_done': False}
        metadata, usage, finish_reason = {}, None, None
        start = time.monotonic()
        try:
            with self.slots, httpx.Client(timeout=httpx.Timeout(self.read_timeout, connect=15, write=30, pool=15),
                  trust_env=False, follow_redirects=False) as session, session.stream(
                  'POST', self.base_url + '/chat/completions', headers=headers, json=payload) as response, (
                  folder / 'http_0.txt').open('w') as raw_out, (folder / 'events_0.jsonl').open('w') as events:
                timing['headers_s'] = time.monotonic() - start
                timing['http_status'] = response.status_code
                if response.status_code != 200:
                    response.read()
                    body = self.redact(response.text)
                    raw_out.write(body)
                    raise RuntimeError(f'HTTP {response.status_code}: ' + body[:2000])
                last_status = start
                for line in response.iter_lines():
                    elapsed = time.monotonic() - start
                    line = self.redact(line)
                    raw_out.write(line + '\n')
                    if not line.startswith('data:'):
                        continue
                    body = line[5:].strip()
                    if body == '[DONE]':
                        timing['sse_done'] = True
                        break
                    chunk = json.loads(body)
                    events.write(json.dumps({'elapsed_s': elapsed, 'data': chunk}, ensure_ascii=False) + '\n')
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
                    if time.monotonic() - last_status >= 10:
                        raw_out.flush()
                        events.flush()
                        save(folder / 'stream_status.json', {**timing, 'elapsed_s': elapsed,
                             'content_chars': sum(map(len, text_parts)), 'reasoning_chars': sum(map(len, reasoning_parts)), 'state': 'running'})
                        last_status = time.monotonic()
            timing['seconds'] = time.monotonic() - start
            raw = {**metadata, 'object': 'chat.completion', 'usage': usage,
                   'choices': [{'index': 0, 'finish_reason': finish_reason, 'message': {
                       'role': 'assistant', 'content': ''.join(text_parts), 'reasoning_content': ''.join(reasoning_parts)}}],
                   '_provenance': {'assembled_from': 'events_0.jsonl', 'raw_sse': 'http_0.txt', 'sse_done': timing['sse_done']}}
            save(folder / 'response_0.json', raw)
            if not timing['sse_done'] or finish_reason != 'stop':
                raise ValueError('Unfinished stream: ' + str(finish_reason))
            parsed = parse_json(''.join(text_parts))
            save(folder / 'parsed_response.json', parsed)
            result = validator(parsed) if validator else parsed
            save(folder / 'result.json', result)
            event = {'tag': tag, 'attempt': 0, 'seconds': timing['seconds'], 'success': True,
                     'usage': usage, 'response_model': metadata.get('model')}
            with self.lock:
                with (self.root / 'calls.jsonl').open('a') as out:
                    out.write(json.dumps(event) + '\n')
            return result
        except Exception as exc:
            save(folder / 'error_0.json', {'type': type(exc).__name__, 'error': self.redact(exc), 'seconds': time.monotonic() - start})
            raise RuntimeError(self.redact(exc)) from None
        finally:
            (folder / 'content.txt').write_text(''.join(text_parts))
            (folder / 'reasoning.txt').write_text(''.join(reasoning_parts))
            timing['seconds'] = timing.get('seconds', time.monotonic() - start)
            tokens = (usage or {}).get('completion_tokens')
            timing['completion_tokens'] = tokens
            timing['e2e_tokens_s'] = tokens / timing['seconds'] if tokens is not None else None
            timing['content_chars'] = sum(map(len, text_parts))
            timing['reasoning_chars'] = sum(map(len, reasoning_parts))
            save(folder / 'stream_status.json', {**timing, 'state': 'ended', 'finish_reason': finish_reason})
