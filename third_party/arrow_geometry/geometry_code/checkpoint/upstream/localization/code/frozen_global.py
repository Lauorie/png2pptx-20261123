"""Freeze only GLOBAL outcomes, including failures; never call the model here."""
from pathlib import Path
import contracts as C
import draft_prompts as D
from transport.common import read_json as read, write_json as write, sha256


def freeze(root, cid, source, cfg):
    old = source / 'cases' / cid / 'global_discovery'
    inp = root / 'inputs' / cid
    size = read(inp / 'frames.json')['global_size']
    prompt = {'system': D.DISCOVERY_SYSTEM, 'prompt': D.global_prompt(size)}
    if read(old / 'prompt.json') != prompt:
        raise ValueError('GLOBAL prompt changed; cannot fix its output: ' + cid)
    if sha256(inp / 'global.png') != sha256(source / 'inputs' / cid / 'global.png'):
        raise ValueError('GLOBAL image mismatch: ' + cid)
    prior = read(source / 'config.json')
    for key in ('model', 'base_url', 'temperature', 'thinking', 'max_tokens', 'reasoning_effort'):
        if cfg.get(key) != prior.get(key):
            raise ValueError('GLOBAL inference mismatch: ' + key)
    f = old / 'result.json'
    if f.exists():
        answer = C.discovery_valid(read(f), size)
    else:
        f = old / 'unresolved.json'
        if not f.exists():
            raise ValueError('GLOBAL has no finished outcome: ' + cid)
        answer = {'_failure': read(f)}
    write(root / 'frozen_global' / (cid + '.json'), {
        'source': str(old.resolve()), 'source_file': str(f.resolve()),
        'source_sha256': sha256(f), 'image_sha256': sha256(inp / 'global.png'),
        'prompt': prompt, 'answer': answer, 'new_api_call': False})


def replay(root, cid, size):
    snapshot = root / 'frozen_global' / (cid + '.json')
    saved = read(snapshot)  # Missing evidence is an error, never a fresh GLOBAL request.
    out = root / 'cases' / cid / 'global_discovery'
    answer = saved['answer']
    write(out / 'prompt.json', saved['prompt'])
    write(out / 'reuse.json', {k: v for k, v in saved.items() if k not in ('answer', 'prompt')})
    if '_failure' in answer:
        write(out / 'unresolved.json', answer['_failure'])
    else:
        C.discovery_valid(answer, size)
        write(out / 'result.json', answer)
    return answer
