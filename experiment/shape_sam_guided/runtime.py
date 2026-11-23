"""Same local GLM transport/budget in both arms; preserve stage retry contracts."""
import copy
import json
from pathlib import Path
import time
from core import read, save, sha
from request_io import ModelTransportError


def parse_response(text, tag):
    starts=[index for index in (text.find('{'),text.find('[')) if index>=0]
    if not starts:raise ValueError('No JSON value in response')
    value,_=json.JSONDecoder().raw_decode(text[min(starts):])
    if tag=='global' and isinstance(value,list) and all(isinstance(item,dict) for item in value):
        return {'objects':value,'more_pending':True,
                'uncertain':['Model returned an unwrapped object array; retained as incomplete detection.']}
    return value


def revalidate_response(reference, target, tag, validator):
    """Recheck a complete rejected reply under the current frozen validator.

    The caller first verifies the effective request identity. Historical error
    and response files remain intact; only the new destination gets a result.
    """
    if not (reference/'error_0.json').is_file() or not (reference/'response_0.json').is_file():
        return None
    failure=read(reference/'error_0.json')
    if failure.get('type') not in {'ValueError','RuntimeError'}:
        return None
    response=read(reference/'response_0.json')
    try:
        choice=response['choices'][0]
        if choice.get('finish_reason')!='stop':return None
        raw=parse_response(choice['message'].get('content') or '',tag)
        result=validator(raw)
    except (KeyError, IndexError, TypeError, ValueError, RuntimeError):
        return None
    save(target/'result.json',raw)
    save(target/'response_revalidation.json',{
        'source':str(reference.resolve()),'request_sha256':sha(reference/'request.json'),
        'response_sha256':sha(reference/'response_0.json'),
        'original_error_sha256':sha(reference/'error_0.json'),
        'model_calls':0,'reason':'Identical request; complete response passed current validator.'})
    return result


class ForwardRuntime:
    def __init__(self, models, config, frozen_case, spatial_evidence=None):
        self.models, self.config = models, config
        self.frozen_case = Path(frozen_case) if frozen_case is not None else None
        self.spatial_evidence = spatial_evidence

    def for_stage(self, **overrides):
        out = copy.copy(self); out.config = {**self.config, **overrides}
        if self.config.get('force_reasoning_effort'):
            out.config['reasoning_effort'] = self.config['force_reasoning_effort']
        return out

    def call(self, root, tag, system, prompt, images, validator, budget, *, retry_builder=None):
        root = Path(root)
        if self.frozen_case is not None and tag in {'global', 'layout'}:
            source = self.frozen_case/'shapes'/('global_raw.json' if tag == 'global' else 'plan.json')
            result = validator(read(source))
            save(root/(tag+'_replay.json'), {'source': str(source), 'sha256': sha(source),
                 'reason': 'Identical frozen first-pass discovery/layout in both forward arms; no fitted geometry reused.'})
            return result
        budget = min(budget, self.models.config['glm_max_tokens'])
        error, previous = None, None
        for attempt in range(self.config.get('shape_schema_retries', 1)+1):
            actual, actual_system, actual_images, actual_validator = copy.deepcopy(prompt), system, images, validator
            if error:
                if retry_builder is not None:
                    retry = retry_builder(previous, error)
                    actual_system, actual, actual_images, actual_validator = (retry[k] for k in ('system','prompt','images','validator'))
                else:
                    actual.update(validation_error=error, correction='Return a complete valid JSON response after inspecting the same images.')
            folder = root/f'{tag}_{attempt:02d}'
            identity = {'system': actual_system, 'prompt': actual, 'budget': budget,
                        'model': 'GLM-5.3-Flash', 'seed': self.models.config['seed'],
                        'model_path': self.models.config.get('glm_model'),
                        'reasoning_effort': self.config.get('reasoning_effort', self.models.config.get('reasoning_effort', 'high')),
                        'thinking': self.config.get('thinking', 'on'),
                        'images': [{'label': label, 'sha256': sha(path)} for label, path in actual_images]}
            if self.models.config.get('glm_transport') == 'api':
                identity['api_provider'] = {'base_url': self.models.config['glm_url'],
                    'model': self.models.config.get('glm_api_model', 'glm-5.3-flash')}
                if self.config.get('request_timeout_seconds') is not None:
                    identity['request_timeout_seconds']=self.config['request_timeout_seconds']
            if (folder/'request.json').exists() and read(folder/'request.json') != identity:
                raise ValueError('Forward request identity changed: '+str(folder))
            save(folder/'request.json', identity)
            if (folder/'result.json').exists():
                return actual_validator(read(folder/'result.json'))
            # Controlled incremental ablations can replay unchanged effective
            # requests. Changed prompts/images/budgets always require inference.
            reference_roots = (self.models.config.get('reuse_request_roots')
                               or [self.models.config.get('reuse_request_root')])
            output_root = self.models.config.get('reuse_output_root')
            for target_root, sources in self.models.config.get('reuse_request_map', {}).items():
                if folder.is_relative_to(Path(target_root)):
                    output_root, reference_roots = target_root, sources
                    break
            can_replay = output_root and folder.is_relative_to(Path(output_root))
            replayed_error = False
            for reference_root in reference_roots if can_replay else []:
                if not reference_root:
                    continue
                reference = Path(reference_root) / folder.relative_to(Path(output_root))
                if ((reference/'request.json').exists() and read(reference/'request.json') == identity
                        and not (reference/'result.json').exists()):
                    revalidated=revalidate_response(reference,folder,tag,actual_validator)
                    if revalidated is not None:return revalidated
                cached_failure = (read(reference/'replay.json')
                                  if (reference/'error_0.json').exists() and (reference/'replay.json').exists() else {})
                if (self.models.config.get('replay_schema_failures')
                        and (reference/'request.json').exists() and read(reference/'request.json') == identity
                        and not (reference/'result.json').exists()
                        and (reference/'error_0.json').exists()
                        and ((reference/'response_0.json').exists() or cached_failure.get('schema_failure_replayed'))):
                    failure = read(reference/'error_0.json')
                    if failure.get('type') in {'ValueError', 'RuntimeError'}:
                        for name in ('content.txt', 'parsed_response.json'):
                            if (reference/name).exists():
                                (folder/name).write_bytes((reference/name).read_bytes())
                        save(folder/'error_0.json', {**failure, 'replayed': True})
                        save(folder/'replay.json', {'source': str(reference),
                             'request_sha256': sha(reference/'request.json'),
                             'response_sha256': (sha(reference/'response_0.json') if (reference/'response_0.json').exists()
                                                 else cached_failure['response_sha256']),
                             'reference_usage': (read(reference/'response_0.json').get('usage', {})
                                                 if (reference/'response_0.json').exists() else cached_failure['reference_usage']),
                             'schema_failure_replayed': True,
                             'reason': 'Freeze the completed reference response and retry trajectory for the controlled ablation.'})
                        error, previous, replayed_error = failure['error'], folder, True
                        break
                if ((reference/'request.json').exists() and (reference/'result.json').exists()
                        and read(reference/'request.json') == identity):
                    raw = read(reference/'result.json')
                    result = actual_validator(raw)
                    response = read(reference/'response_0.json') if (reference/'response_0.json').exists() else {}
                    reference_usage = response.get('usage', {})
                    if not reference_usage and (reference/'replay.json').exists():
                        reference_usage = read(reference/'replay.json').get('reference_usage', {})
                    save(folder/'result.json', raw)
                    save(folder/'replay.json', {'source': str(reference),
                         'request_sha256': sha(reference/'request.json'),
                         'result_sha256': sha(reference/'result.json'),
                         'reference_usage': reference_usage,
                         'reason': 'Identical effective request; no output-quality selection.'})
                    return result
            if replayed_error:
                continue
            if (folder/'error_0.json').exists():
                revalidated=revalidate_response(folder,folder,tag,actual_validator)
                if revalidated is not None:return revalidated
                error = read(folder/'error_0.json')['error']; previous = folder
                continue
            import base64
            content = [{'type': 'text', 'text': json.dumps(actual, ensure_ascii=False)}]
            for label, path in actual_images:
                content += [{'type': 'text', 'text': label}, {'type': 'image_url', 'image_url':
                    {'url': 'data:image/png;base64,'+base64.b64encode(Path(path).read_bytes()).decode()}}]
            body = {'model': 'GLM-5.3-Flash', 'messages': [{'role':'system','content':actual_system},
                    {'role':'user','content':content}], 'temperature': 0, 'seed': self.models.config['seed'],
                    'max_tokens': budget, 'chat_template_kwargs': {'reasoning_effort':identity['reasoning_effort'],
                        'enable_thinking':identity['thinking']!='off','clear_thinking':True}}
            started = time.monotonic()
            try:
                with self.models.slots:
                    acquired = time.monotonic()
                    response = self.models.post(self.models.config['glm_url']+'/chat/completions', body,
                                                trace_path=folder/'transport.json',
                                                request_timeout=self.config.get('request_timeout_seconds'))
                save(folder/'response_0.json', response)
                save(folder/'timing.json', {'queue_seconds': acquired-started,
                    'service_seconds': time.monotonic()-acquired, 'usage': response.get('usage',{})})
                choice = response['choices'][0]
                text = choice['message'].get('content') or ''
                (folder/'content.txt').write_text(text)
                if choice.get('finish_reason') == 'length':
                    raise ValueError('Response truncated; return compact complete JSON')
                raw = parse_response(text,tag)
                result = actual_validator(raw)
                save(folder/'result.json', raw)
                return result
            except ModelTransportError:
                # A later resume may retry transport; do not cache a model/schema
                # failure or ask the model to correct a network error.
                raise
            except (ValueError, RuntimeError) as exc:
                error = str(exc); save(folder/'error_0.json', {'error': error, 'type': type(exc).__name__})
                previous = folder
        raise RuntimeError(error or 'Forward response validation exhausted')
