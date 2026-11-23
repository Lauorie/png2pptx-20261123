"""Whole-image review, image deduplication and discovery queue proposals."""
from PIL import Image

from ..common import read, save, overlay, mapped_box, valid_box
from ..image_conflicts import run_filter
from ..scope import SHAPE_SCOPE
from .control import schema, validate, SYSTEM
from .coordinates import coordinate_mode, model_size, to_pixels, NORM1000_BOX_CONVENTION


def review(scene, item, workspace, folder, runtime, options, existing, assets, threshold, round_index):
    folder.mkdir(parents=True, exist_ok=True)
    size = item['input_size']
    mode = coordinate_mode(options)
    objects = [{**o, 'bbox_2d': mapped_box(o['bbox_2d'], item['source_size'], size)} for o in existing]
    with Image.open(workspace/item['input']) as image:
        overlay(image, objects, folder/'inventory.png')
    # Rendered full-view geometry uses the same image dimensions as inventory.
    with Image.open(folder.parent/'fit_overlay.png') as image:
        image.resize(tuple(size), Image.Resampling.LANCZOS).save(folder/'fit.png')
    candidate_spec = schema({'label': {'type': 'string', 'minLength': 1}, 'kind': {'type': 'string', 'minLength': 1},
                             'bbox_2d': {'type': 'array', 'minItems': 4, 'maxItems': 4, 'items': {'type': 'number'}},
                             'reason': {'type': 'string', 'minLength': 1}})
    pair = schema({'object_ids': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 2, 'maxItems': 2, 'uniqueItems': True},
                   'reason': {'type': 'string', 'minLength': 1}})
    spec = schema({'status': {'enum': ['complete', 'uncertain']},
                   'new_objects': {'type': 'array', 'items': candidate_spec, 'maxItems': options['max_new_candidates_per_round']},
                   'possible_duplicates': {'type': 'array', 'items': pair},
                   'uncertain': {'type': 'array', 'items': {'type': 'string'}}})
    ids = {o['id'] for o in existing}
    prompt = {'version': 'shape_integration.fit_coverage.1', 'scope': SHAPE_SCOPE,
              'image_size': size, 'object_states': [{'id': o['id'], 'state': scene['objects'][o['id']]['state']} for o in existing],
              'task': 'Check missing SHAPE candidates and possible duplicate/parent-child representations. Inventory boxes show existing candidates, including failed/unresolved fits; do NOT rediscover those as new objects. Return only wholly unrepresented objects, one compound candidate for all its similar cells. Never add a parent outline merely because a group has a box. Exclude photos, icons belonging to images, text and standalone arrows under the supplied shape scope. Propose new boxes in THIS full image pixel coordinates. Report possible duplicates as uncertain ownership pairs; no automatic deletion follows this review. Complete means this review found no gaps, not measured recall.',
              'output_schema': spec}
    if mode == 'norm1000':
        prompt.pop('image_size')
        prompt['local_coordinates'] = {'convention': NORM1000_BOX_CONVENTION}
        prompt['task'] = prompt['task'].replace('Propose new boxes in THIS full image pixel coordinates. ', '')
        candidate_spec['properties']['bbox_2d']['items'].update(minimum=0, maximum=1000)
    def checker(raw):
        validate(raw, spec)
        for o in raw['new_objects']:
            if not valid_box(o['bbox_2d'], model_size(size, mode)):
                raise ValueError('Invalid discovery bbox')
        if any(not set(p['object_ids']) <= ids for p in raw['possible_duplicates']):
            raise ValueError('Unknown duplicate candidate ID')
        return raw
    raw = runtime.call(folder/'calls', 'coverage', SYSTEM, prompt,
        [('Clean whole image', workspace/item['input']), ('Existing candidates', folder/'inventory.png'),
         ('Current diagnostic geometry', folder/'fit.png')], checker, options['coverage_max_tokens'])
    if mode == 'norm1000':
        save(folder/'review_wire.json', raw)
        raw = {**raw, 'new_objects': [{**o, 'bbox_2d': to_pixels(o['bbox_2d'], size, mode)} for o in raw['new_objects']]}
    proposed = []
    for i, o in enumerate(raw['new_objects']):
        oid = f'F{round_index+1}_N{i+1:03d}'
        if oid in ids:
            raise ValueError('Discovery ID collision')
        proposed.append({**o, 'id': oid, 'bbox_2d': mapped_box(o['bbox_2d'], size, item['source_size']),
                         'origin': 'fitting_coverage', 'discovery_round': round_index})
    decision = run_filter(proposed, assets,
        {'path': str(workspace/item['original']), 'sha256': item['source_sha256'], 'size': item['source_size']},
        folder/'image_conflict_review', runtime, {**runtime.config, 'iou_threshold': threshold})
    save(folder/'image_dedup.json', decision)
    record = {'round': round_index, 'review': raw, 'new_candidate_ids': [o['id'] for o in decision['kept']],
              'image_removed_ids': [o['id'] for o in decision['removed']],
              'image_conflict_deferred_ids': decision['deferred_ids'],
              'image_conflict_review_calls': decision['model_request_calls'],
              'duplicate_action': 'flag_only', 'new_objects': decision['kept']}
    save(folder/'record.json', record)
    return record
