"""Single-attempt clients with immutable, content-addressed request evidence."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import time

from PIL import Image
import requests

from ..api.relay_client import load_bashrc_credentials
from ..api.stream_client import StreamingRelayClient
from ..scene import save
from .prompts import correction_prompt

EDIT_MODEL = 'models--SenseNova--SenseNova-U1.5-8B-MoT'
EDIT_ENDPOINT = 'http://100.102.14.171:8091'


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    return digest(Path(path).read_bytes())


def credentials():
    base = next((os.environ[k] for k in ('BOYUE_BASE_URL', 'BOYUE_API_BASE', 'OPENAI_BASE_URL', 'OPENAI_API_BASE') if os.environ.get(k)), None)
    key = next((os.environ[k] for k in ('BOYUE_API_KEY', 'OPENAI_API_KEY') if os.environ.get(k)), None)
    if base:
        if not key: raise ValueError('API base set but BOYUE_API_KEY / OPENAI_API_KEY is missing')
        return base.rstrip('/') + ('' if base.rstrip('/').endswith('/v1') else '/v1'), key
    base, key, _ = load_bashrc_credentials()
    return base, key


class DetectorClient(StreamingRelayClient):
    def request_parameters(self):
        # The vision-capable GLM-5.3-Flash route requires enabled thinking; use low effort.
        return {'model': self.model, 'temperature': .1, 'max_tokens': self.max_tokens,
                'thinking': {'type': 'enabled'}, 'reasoning_effort': 'low'}


def make_detector(root, model='glm-5.3-flash', max_tokens=4096):
    base, key = credentials()
    return DetectorClient(root, model, threading.Semaphore(1), base, key,
                          max_tokens=max_tokens, max_attempts=1, thinking='on',
                          thinking_off_strategy='template_flag', read_timeout=180)


def cached_json_call(client, stage, system, prompt, images, validator):
    """Cache every visual planning stage by its actual images and full instructions."""
    identity = {'stage': stage, 'system': system, 'prompt': prompt,
                'images': [{'label': label, 'sha256': file_sha(path)} for label, path in images],
                'parameters': client.request_parameters(), 'endpoint': client.base_url}
    prefix = stage + '_' + digest(identity)
    errors = []; previous_response = None
    # One bounded correction for malformed JSON/transport failures; every attempt is immutable.
    for attempt in range(2):
        tag = prefix + f'_attempt_{attempt:02d}'; folder = client.root/tag
        actual = correction_prompt(prompt, errors[-1] if errors else None, previous_response)
        attempt_identity = {**identity, 'actual_prompt': actual, 'attempt': attempt}
        try:
            if folder.exists():
                if (folder/'result.json').exists():
                    if json.loads((folder/'identity.json').read_text()) != attempt_identity:
                        raise ValueError('Visual-call cache identity mismatch')
                    result = validator(json.loads((folder/'result.json').read_text()))
                    return result, str(folder.resolve())
                error_path = folder/'error_0.json'
                error = json.loads(error_path.read_text()).get('error') if error_path.exists() else 'Prior attempt interrupted'
                raise RuntimeError(error)
            try:
                result = client.call(tag, system, actual, images, validator)
            finally:
                if folder.exists(): save(folder/'identity.json', attempt_identity)
            return result, str(folder.resolve())
        except (RuntimeError, ValueError) as exc:
            errors.append(str(exc))
            content_path = folder/'content.txt'
            previous_response = content_path.read_text() if content_path.exists() else None
    raise RuntimeError('Visual call failed after two attempts: ' + '; '.join(errors))


class SenseNovaClient:
    def __init__(self, root, endpoint=EDIT_ENDPOINT, model=EDIT_MODEL, steps=30, seed=42, timeout=180, image_guidance=3.):
        self.root = Path(root)
        self.endpoint, self.model = endpoint.rstrip('/'), model
        self.steps, self.seed, self.timeout = steps, seed, timeout
        self.image_guidance = image_guidance

    def edit(self, image_path, prompt):
        data = Path(image_path).read_bytes()
        with Image.open(io.BytesIO(data)) as im: width, height = im.size
        parameters = {'model': self.model, 'prompt': prompt, 'size': f'{width}x{height}',
                      'output_format': 'png', 'response_format': 'b64_json',
                      'img_config': {'width': width, 'height': height, 'steps': self.steps,
                                     'seed': self.seed, 'guidance_scale': 4.,
                                     'image_guidance_scale': self.image_guidance, 'image_type': 'png'}}
        identity = {'endpoint': self.endpoint, 'input_sha256': digest(data), 'parameters': parameters}
        folder = self.root / digest(identity)
        if folder.exists():
            result_path = folder / 'result.json'
            if not result_path.exists():
                raise RuntimeError('Previous edit failed/interrupted; evidence retained; choose a new run: ' + str(folder))
            result = json.loads(result_path.read_text())
            if json.loads((folder/'request.json').read_text()) != identity or file_sha(folder/'edit_raw.png') != result['output_sha256']:
                raise ValueError('Edit cache changed: ' + str(folder))
            return folder/'edit_raw.png', {**result, 'cache_hit': True, 'call_dir': str(folder.resolve())}
        folder.mkdir(parents=True, exist_ok=False)
        save(folder/'request.json', identity)
        (folder/'input.png').write_bytes(data)
        payload = {**parameters, 'images': [{'image_url': 'data:image/png;base64,' + base64.b64encode(data).decode()}]}
        save(folder/'wire_request.json', payload)
        start = time.monotonic()
        try:
            with requests.Session() as session:
                session.trust_env = False
                response = session.post(self.endpoint+'/v1/images/edits', json=payload, timeout=(10, self.timeout))
            (folder/'response.json').write_text(response.text)
            response.raise_for_status()
            raw = response.json()
            encoded = raw['data'][0]['b64_json']
            if encoded.startswith('data:'): encoded = encoded.split(',', 1)[1]
            binary = base64.b64decode(encoded, validate=True)
            (folder/'edit_raw.png').write_bytes(binary)
            with Image.open(io.BytesIO(binary)) as im:
                im.load()
                if im.format != 'PNG' or im.size != (width, height):
                    raise ValueError('Editor changed format/size; original placement is no longer valid')
                mode = im.mode
            result = {'output_sha256': digest(binary), 'size': [width, height], 'mode': mode,
                      'elapsed_s': time.monotonic()-start, 'usage': raw.get('usage'), 'request_hash': folder.name}
            save(folder/'result.json', result)
            return folder/'edit_raw.png', {**result, 'cache_hit': False, 'call_dir': str(folder.resolve())}
        except Exception as exc:
            save(folder/'error.json', {'error': str(exc), 'elapsed_s': time.monotonic()-start})
            raise
