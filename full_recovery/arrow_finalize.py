"""One whole-route identity decision inside the arrow stage, before final export.

Uses the delivered branch's nomination/validation/application methods. Every
candidate sees original pixels and isolated native PPTs before any deletion.
Surviving geometry/style is never refitted. No separate postprocessing process.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import copy, deepcopy
from pathlib import Path


def profile(model, **overrides):
    """Keep the same request slots and provider failure latch across profiles."""
    child = copy(model)
    child.config = {**model.config, **overrides}
    return child


def endpoint_types(row):
    style = row.get('style', {})
    return [(style.get(end) or {}).get('type') or
            (row.get('heads') or ['none', 'none'])[index]
            for index, end in enumerate(('start', 'end'))]


def protect_endpoints(before, after, audit, pairs):
    """Do not delete a unique head/dot, including after component keeper voting."""
    rows = {r['id']: r for r in before['arrows']}
    if any(r != rows[r['id']] for r in after['arrows']):
        raise ValueError('Identity review changed surviving geometry or style')
    lookup = {p['id']: p for p in pairs}
    removed, protected = [], []
    for edit in audit['removed']:
        loser, keeper = endpoint_types(rows[edit['remove']]), endpoint_types(rows[edit['keep']])
        if lookup[edit['pair']].get('reversed'):
            keeper = keeper[::-1]
        missing = [i for i, (a, b) in enumerate(zip(loser, keeper))
                   if a not in ('none', None, 'unknown') and a != b]
        if missing:
            protected.append({**edit, 'reason': 'unique_endpoint_head_preserved', 'endpoints': missing})
        else:
            removed.append(edit)
    deleted = {edit['remove'] for edit in removed}
    result = deepcopy(before)
    result['arrows'] = [r for r in result['arrows'] if r['id'] not in deleted]
    return result, {**audit, 'removed': removed, 'endpoint_guards': protected}


def evidence(clean, a, b, folder, locks):
    """Same-grid source/A/B crops; endpoint dots in native renders are real ink."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps
    from ownership_dedup import sample
    from lib import native
    folder.mkdir(parents=True, exist_ok=True)
    rendered = []
    for label, row in [('A', a), ('B', b)]:
        path = folder/(label+'.pptx')
        with locks['cpu']:
            native.write(clean.size, [row], [row['style']], path)
            native.render(path, folder/label, clean.size)
        with Image.open(folder/label/'styled.png') as opened:
            rendered.append(opened.convert('RGB'))
    ap, bp = sample(a)[0], sample(b)[0]
    if np.linalg.norm(ap[0]-bp[-1])+np.linalg.norm(ap[-1]-bp[0]) < np.linalg.norm(ap[0]-bp[0])+np.linalg.norm(ap[-1]-bp[-1]):
        bp = bp[::-1]
    points = np.concatenate([ap, bp])
    boxes = [np.r_[points.min(0)-24, points.max(0)+24]]
    boxes += [np.r_[(x+y)/2-24, (x+y)/2+24] for x, y in [(ap[0], bp[0]), (ap[-1], bp[-1])]]
    sheet = Image.new('RGB', (1200, 720), '#e9edf2')
    draw = ImageDraw.Draw(sheet)
    for i, bounds in enumerate(boxes):
        box = tuple(np.rint([max(0, bounds[0]), max(0, bounds[1]),
                              min(clean.width, bounds[2]), min(clean.height, bounds[3])]).astype(int))
        for j, (label, image) in enumerate(zip(('SOURCE', 'A actual PPT', 'B actual PPT'), [clean, *rendered])):
            draw.text((400*j+8, 240*i+5), label+(' route' if i == 0 else ' endpoint '+str(i)), fill='black')
            tile = ImageOps.contain(image.crop(box), (384, 205), Image.Resampling.NEAREST)
            sheet.paste(tile, (400*j+(400-tile.width)//2, 240*i+28+(205-tile.height)//2))
    path = folder/'source_and_native_endpoints.png'
    sheet.save(path)
    return path


def finish(clean, results, out, config, model, locks):
    from ownership_dedup import nominate, validate, apply
    from scene_snapshot import canonical_rows
    from transport.common import write_json as write
    out = Path(out)/'whole_path_identity'
    before = {'size': list(clean.size), 'arrows': canonical_rows(results)}
    pairs = nominate(before['arrows'], endpoint_px=12, p90_px=6, min_length_px=10, length_ratio=.85)
    write(out/'pairs.json', pairs)
    write(out/'before.scene.json', before)
    if not pairs:
        write(out/'audit.json', {'removed': [], 'candidates': 0, 'new_model_calls': 0})
        return results, []
    rows = {r['id']: r for r in before['arrows']}
    glm = profile(model, max_tokens=8192, reasoning_effort='low', request_timeout_seconds=360)
    batches = [pairs[i:i+config.get('identity_review_batch_size', 3)]
               for i in range(0, len(pairs), config.get('identity_review_batch_size', 3))]
    def review(task):
        index, batch = task
        folder = out/f'batch_{index:03d}'
        images = []
        brief = []
        for pair in batch:
            a, b = rows[pair['a']], rows[pair['b']]
            images.append((pair['id']+' source and separate native routes/endpoints',
                           evidence(clean, a, b, folder/pair['id'], locks)))
            brief.append({**pair, 'A': {k: a.get(k) for k in ('id', 'start', 'end', 'style')},
                          'B': {k: b.get(k) for k in ('id', 'start', 'end', 'style')}})
        prompt = {
            'task': 'Judge complete physical source-route identity; select existing survivors only.',
            'pairs': brief,
            'instructions': [
                'SOURCE panels are unmarked original pixels. A/B panels are isolated actual native PPTs, including their real endpoint heads and dots. Never dismiss a SOURCE dot as a locator annotation.',
                'Proximity is only nomination. duplicate requires the same entire source route and attachments with no unique branch, dot, head or other visible content in the removed route. Parallel, crossing, partially shared and opposite-direction routes stay separate.',
                'Choose the existing route whose geometry and style match SOURCE. Preserve source-supported head/dot information. An unsupported or uncertain head difference means keep both; do not silently lose a head.',
                'No geometry/style edits. Use uncertain/both when unresolved. Give only concise visible evidence and the requested JSON.'],
            'output_schema': {'decisions': '[{id,relation:duplicate|distinct|shared|uncertain,keep:A|B|both,fully_redundant:boolean,reason:short source observation}]'},
        }
        return glm.ask(folder/'request', prompt, images, lambda answer: validate(answer, batch))
    try:
        with ThreadPoolExecutor(max_workers=min(4, len(batches))) as pool:
            answers = list(pool.map(review, enumerate(batches)))
        answer = {'decisions': [d for a in answers for d in a['decisions']]}
        after, audit = apply(before, pairs, answer)
        after, audit = protect_endpoints(before, after, audit, pairs)
        write(out/'answer.json', answer)
        write(out/'audit.json', audit)
        write(out/'scene.json', after)
        deleted = {r['remove'] for r in audit['removed']}
        updated = deepcopy(results)
        for group in updated:
            group['rows'] = [r for r in group.get('rows', []) if r['id'] not in deleted]
        return updated, []
    except Exception as exc:
        write(out/'audit.json', {'removed': [], 'retained_all': True, 'error': str(exc)})
        return results, [{'stage': 'whole_path_identity', 'error': str(exc), 'retained_all': True}]
