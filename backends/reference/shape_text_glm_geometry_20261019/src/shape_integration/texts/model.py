"""Shared Boyue transport, one saved-response format correction; no visual critic."""
from ..common import save, read, sha
from .prompts import SYSTEM


def ask(runtime, folder, prompt, images, validator, opt):
    from .background import model_fields
    prompt=model_fields(prompt)
    folder.mkdir(parents=True, exist_ok=True)
    save(folder/'prompt.json', {'system': SYSTEM, 'prompt': prompt,
                              'images': [{'label': label, 'path': str(path), 'sha256': sha(path)} for label, path in images]})
    def repair(previous, error):
        response = (previous/'content.txt').read_text() if (previous/'content.txt').exists() else None
        return {'system': SYSTEM, 'prompt': {**prompt, 'format_correction': {
            'error': error, 'previous_response': response,
            'instruction': 'Repair the invalid fields; preserve valid observations. Return one complete JSON object only.'}},
            'images': images, 'validator': validator}
    return runtime.for_stage(shape_schema_retries=opt['protocol_retries']).call(
        folder/'calls', 'request', SYSTEM, prompt, images, validator, opt['max_tokens'], retry_builder=repair)
