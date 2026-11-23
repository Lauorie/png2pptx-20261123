from request_budget import bounded, FileSlots
import hashlib
import json
import threading
import time
from pathlib import Path
from .client import request, TransportError
from .coordinate_precision import normalize_prompt, VERSION as COORDINATE_CONTRACT
from .common import read_json, write_json, sha256
from .prompts import SYSTEM


def close_object_before_array(text):
    """Repair only a missing object closer immediately before its enclosing ].

    No values, coordinates, names, quotes or commas are inferred or changed.
    All other malformed JSON remains an error and goes to format retry.
    """
    stack, output, inserted = [], [], []
    quoted = escaped = False
    for index, char in enumerate(text):
        if quoted:
            output.append(char)
            if escaped: escaped = False
            elif char == '\\': escaped = True
            elif char == '"': quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in '{[':
            stack.append(char)
        elif char in '}]':
            if char == ']' and len(stack) >= 2 and stack[-2:] == ['[', '{']:
                output.append('}'); stack.pop(); inserted.append(index)
            expected = '{' if char == '}' else '['
            if not stack or stack.pop() != expected:
                return text, []
        output.append(char)
    if quoted or stack:
        return text, []
    return ''.join(output), inserted


def parse(content, repair_log=None):
    content = content.strip()
    if content.startswith('```') and content.endswith('```'):
        content = content.split('\n', 1)[1].rsplit('```', 1)[0]
    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        # A provider may echo the complete request before the answer. Consume
        # complete top-level JSON documents, never a nested example inside one.
        decoder=json.JSONDecoder();cursor=0;answers=[]
        while cursor<len(content):
            start=content.find('{',cursor)
            if start<0:break
            try:
                document,end=decoder.raw_decode(content,start)
            except json.JSONDecodeError:
                break
            cursor=end
            if isinstance(document,dict) and not {'task','instructions','response_format'}.intersection(document) and {'arrows','edits','duplicates','joins','crops','routes','groups','targets','regions','line_styles','heads','styles','checks'}.intersection(document):
                answers.append(document)
        if len(answers)==1:
            if repair_log is not None:write_json(repair_log,{'repair':'Ignore complete request echo and Markdown fence; use the unique answer object','values_and_coordinates_unchanged':True})
            return answers[0]
        repaired, inserted = close_object_before_array(content)
        if not inserted:
            raise
        raw = json.loads(repaired)
        if repair_log is not None:
            write_json(repair_log, {'repair': 'insert missing } immediately before enclosing ]',
                                   'insertions_at_original_offsets': inserted, 'values_and_coordinates_unchanged': True})
    # A wrapped answer is allowed. An echoed request containing its example
    # response_format is NOT an answer and must reach validation as-is.
    if isinstance(raw, dict) and len(raw)==1 and not {'arrows', 'regions', 'accept'}.intersection(raw):
        for key in ('output', 'response_format'):
            if isinstance(raw.get(key), dict):
                return raw[key]
    return raw


def reusable_identity(previous, current, allow_lower_limit=False):
    """Only a larger output ceiling may differ, when explicitly enabled.

    This permits reusing complete, validated answers. It never makes a truncated
    stream reusable, and never ignores changed prompts, images or other settings.
    """
    if previous == current:return True
    if not allow_lower_limit:return False
    old=previous.get('inference',{});new=current.get('inference',{})
    a=old.get('max_tokens');b=new.get('max_tokens')
    if type(a) is not int or type(b) is not int or not 0<a<b:return False
    normalized={**previous,'inference':{**old,'max_tokens':b}}
    return normalized==current


class Model:
    def __init__(self, config, run_dir=None, reuse_from=None, system=None):
        self.config = config
        self.system = SYSTEM if system is None else system
        self.run_dir = Path(run_dir) if run_dir else None
        self.reuse_from = Path(reuse_from) if reuse_from else None
        self.slots = (FileSlots(config['request_slot_dir'], config['api_concurrency']) if config.get('request_slot_dir')
                      else threading.BoundedSemaphore(config['api_concurrency']))
        self.blocked = threading.Event()
        self.reason = None

    @bounded
    def ask(self, folder, prompt, images, validator, *, system=None):
        system = self.system if system is None else system
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        from semantic_hint_contract import clean_prompt, VERSION as HINT_CONTRACT
        raw_prompt = prompt
        prompt, hint_changes = clean_prompt(prompt)
        coordinate_source = prompt
        prompt = normalize_prompt(prompt)
        identity = {'system': system, 'prompt': prompt,
                    'images': [(label, sha256(path)) for label, path in images],
                    'inference': {k: self.config[k] for k in ('base_url', 'model', 'temperature', 'max_tokens', 'thinking', 'reasoning_effort')
                                  if k != 'reasoning_effort' or self.config.get(k) is not None}}
        # Exact identity is still required for reuse. Changed upstream results
        # create another request version instead of disabling downstream stages.
        identity = json.loads(json.dumps(identity))
        logical_folder = folder
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if (folder/'identity.json').exists() and read_json(folder/'identity.json') != identity:
            folder = logical_folder/'revisions'/digest
            folder.mkdir(parents=True, exist_ok=True)
        if hint_changes:
            write_json(folder/'raw_prompt.json', {'system': system, 'prompt': raw_prompt})
            write_json(folder/'semantic_hint_normalization.json', {
                'contract': HINT_CONTRACT, 'changes': hint_changes,
                'structured_coordinates_changed': False})
        if prompt != coordinate_source:
            write_json(folder/'coordinate_input_original_prompt.json', {'prompt': coordinate_source})
            write_json(folder/'coordinate_input_precision.json', {
                'contract': COORDINATE_CONTRACT, 'decimal_places': 1,
                'scope': 'model_input_only', 'pipeline_coordinates_unchanged': True})
        write_json(logical_folder/'active_request.json', {'path': str(folder.resolve()), 'identity_sha256': digest})
        if (folder/'identity.json').exists():
            if read_json(folder/'identity.json') != identity:
                raise RuntimeError('Request version identity mismatch: '+str(folder))
            if (folder/'result.json').exists():
                result = validator(read_json(folder/'result.json'))
                write_json(folder/'resume.json', {'same_request': True, 'new_api_call': False})
                return result
        else:
            write_json(folder/'identity.json', identity)
            write_json(folder/'prompt.json', {'system': system, 'prompt': prompt})
        if self.reuse_from is not None:
            old = self.reuse_from/folder.relative_to(self.run_dir)
            previous=read_json(old/'identity.json') if (old/'identity.json').exists() else None
            if previous is not None and reusable_identity(previous,identity,self.config.get('reuse_completed_lower_output_limit',False)):
                # Prefer the original complete answer when the current contract can
                # accept it. Old format retries may have removed legitimate nodes.
                calls = sorted(old.glob('call_*'))
                if (old/'reuse.json').exists():
                    inherited = read_json(old/'reuse.json')
                    source_call = Path(inherited['source'])
                    if (source_call/'content.txt').exists() and sha256(source_call/'content.txt') == inherited['content_sha256']:
                        calls.append(source_call)
                for call in calls:
                    if not (call/'status.json').exists() or not (call/'content.txt').exists():
                        continue
                    status = read_json(call/'status.json')
                    if status.get('state') != 'transport_complete' or not status.get('sse_done') or status.get('finish_reason') != 'stop':
                        continue
                    try:
                        result = validator(parse((call/'content.txt').read_text(), folder/'reuse_syntax_repair.json'))
                    except (ValueError, TypeError, KeyError):
                        continue
                    write_json(folder/'reuse.json', {'source': str(call.resolve()), 'content_sha256': sha256(call/'content.txt'),
                               'same_prompt_images_parameters': previous==identity,
                               'same_prompt_images_model': True,
                               'source_max_tokens': previous['inference']['max_tokens'],
                               'requested_max_tokens': identity['inference']['max_tokens'],
                               'reuse_policy': 'exact_request' if previous==identity else 'validated_complete_answer_with_lower_output_limit',
                               'validated_with_current_contract': True, 'new_api_call': False})
                    write_json(folder/'result.json', result)
                    return result
        actual = prompt
        for fmt in range(self.config['format_retries']+1):
            for retry in range(self.config['transport_retries']+1):
                call = folder/f'call_{len(list(folder.glob("call_*"))):03d}'
                with self.slots:
                    if self.blocked.is_set():
                        raise TransportError('Provider blocked: '+str(self.reason))
                    try:
                        content = request(self.config, system, actual, images, call)
                        break
                    except TransportError as exc:
                        if exc.status_code in (401, 403, 404) or 'model_not_found' in str(exc):
                            self.reason = str(exc)
                            self.blocked.set()
                            raise
                        if retry == self.config['transport_retries'] or not exc.retryable:
                            raise
                time.sleep(min(15, 2**(retry+1)))
            try:
                raw = parse(content, call/'json_syntax_repair.json')
                write_json(folder/f'parsed_{fmt}.json', raw)
                result = validator(raw)
                write_json(folder/'result.json', result)
                return result
            except (ValueError, TypeError, KeyError) as exc:
                write_json(folder/f'validation_error_{fmt}.json', {'error': str(exc)})
                if fmt == self.config['format_retries']:
                    raise ValueError(f'Invalid output at {folder}: {exc}') from None
                actual = {**prompt, 'format_repair': {'error': str(exc), 'previous_output': content,
                          'instruction': 'Return the complete JSON with repaired structure/references. This is a format repair, not a new visual task.'}}
