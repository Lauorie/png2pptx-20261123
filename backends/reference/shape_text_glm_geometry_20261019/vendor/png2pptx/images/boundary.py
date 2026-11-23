"""Independent target-centered bbox review; search and classification stay upstream."""
import copy
from collections import Counter
import json
from pathlib import Path

from PIL import Image, ImageDraw

from ..frames import Crop
from ..scene import save
from .clients import cached_json_call, digest, file_sha
from .detect import source_box, validate_inventory
from .prompts import BOUNDARY_REVIEW_SYSTEM, IMAGE_LABELS, boundary_review_prompt
from .search import native_crop, union_boxes


def review_view(box, size, source_size):
    """A focused view with half a target long edge of context on every side."""
    margin = max(box[2]-box[0], box[3]-box[1]) * .5
    dx, dy = [max(margin, 8*size[i]/source_size[i]) for i in (0, 1)]
    return [max(0., box[0]-dx), max(0., box[1]-dy),
            min(size[0]-1., box[2]+dx), min(size[1]-1., box[3]+dy)]


def validate_review(raw, size, hint):
    if not isinstance(raw, dict) or set(raw) != {'decision', 'bbox', 'edges', 'reason'}:
        raise ValueError('Boundary review requires decision, bbox, edges and reason only')
    decision = raw['decision']
    if decision not in ('accept', 'revise', 'needs_context', 'unresolved'):
        raise ValueError('Invalid boundary review decision')
    edges = raw['edges']
    if (not isinstance(edges, dict) or set(edges) != {'left', 'top', 'right', 'bottom'}
            or any(v not in ('ok', 'clipped', 'loose', 'unclear') for v in edges.values())):
        raise ValueError('Boundary review must inspect all four edges')
    if not isinstance(raw['reason'], str) or not raw['reason'].strip() or len(raw['reason']) > 700:
        raise ValueError('Boundary review requires short visible evidence')
    if decision == 'revise':
        if not isinstance(raw['bbox'], list):
            raise ValueError('Revise requires a numeric bbox array')
        Crop.from_model(raw['bbox'], size)
        if not any(v in ('clipped', 'loose') for v in edges.values()) or 'unclear' in edges.values():
            raise ValueError('Revise requires a visible boundary error; unresolved edges need inspection')
        if all(abs(a-b) < 1e-6 for a, b in zip(raw['bbox'], hint)):
            raise ValueError('Revise must propose a different bbox')
    elif raw['bbox'] is not None:
        raise ValueError('Only revise may change the bbox; other decisions require bbox=null')
    if decision == 'accept' and any(v != 'ok' for v in edges.values()):
        raise ValueError('Accept cannot have clipped, loose or unclear edges')
    return raw


def _overlay(clean, box, path, color):
    marked = clean.copy()
    ImageDraw.Draw(marked).rectangle(box, outline=color, width=3)
    marked.save(path)


def review_boundaries(client, source_path, inventory, size, folder,
                      max_calls=64, calls_per_asset=2):
    """Review each asset once before spending remaining calls on repairs/context.

    A revised box must be seen on a later overlay before model acceptance. Budgets
    count visual stages; cached_json_call separately bounds malformed-answer retries.
    All raw proposals survive, including an unverified revision at the budget limit.
    """
    if max_calls < 0 or calls_per_asset < 1:
        raise ValueError('Boundary budgets require max_calls >= 0 and calls_per_asset >= 1')
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    result = copy.deepcopy(validate_inventory(inventory, size))
    identity = {'source_sha256': file_sha(source_path), 'inventory_sha256': digest(inventory),
                'input_size': list(size), 'max_calls': max_calls, 'calls_per_asset': calls_per_asset,
                'parameters': client.request_parameters() if client else None,
                'endpoint': client.base_url if client else None,
                'code_sha256': {name: file_sha(Path(__file__).with_name(name))
                                for name in ('boundary.py', 'prompts.py', 'search.py')}}
    identity_path = folder/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Boundary review inputs/code changed; choose a new output directory')
    save(identity_path, identity)
    with Image.open(source_path) as opened:
        source = opened.convert('RGB')
    records = {a['id']: {'status': 'pending', 'original_bbox': list(a['bbox']),
                        'bbox': list(a['bbox']), 'calls': [], 'reason': 'awaiting independent review'}
               for a in result['assets']}
    state = {'calls': 0, 'max_calls': max_calls, 'calls_per_asset': calls_per_asset,
             'assets': records, 'visual_accuracy_verified': False}
    views = {a['id']: review_view(a['bbox'], size, source.size) for a in result['assets']}
    purposes = {a['id']: 'initial_review' for a in result['assets']}
    previous_crops = {}

    def checkpoint():
        pending = [i for i, r in records.items() if r['status'] != 'model_review_complete']
        state['status'] = ('not_needed' if not records else
                           'unresolved' if pending else 'model_review_complete')
        state['unresolved_ids'] = pending
        state['decisions'] = dict(Counter(c['decision'] for r in records.values() for c in r['calls']))
        save(folder/'state.json', state)

    checkpoint()
    for round_index in range(calls_per_asset):
        for asset in result['assets']:
            ident = asset['id']; record = records[ident]
            if record['status'] != 'pending':
                continue
            if state['calls'] >= max_calls:
                record.update(status='unresolved', reason='total boundary call budget exhausted')
                checkpoint(); continue
            view = union_boxes([views[ident], asset['bbox']])
            frame, native, metadata = native_crop(source, view, size)
            crop_key = tuple(metadata['source_crop_half_open'])
            if purposes[ident] == 'expanded_context' and crop_key == previous_crops.get(ident):
                record.update(status='unresolved', reason='no larger distinct source view available')
                checkpoint(); continue
            local = folder/ident/f'round_{round_index:02d}'; local.mkdir(parents=True, exist_ok=True)
            native.save(local/'source_native.png')
            clean = native.resize(frame.input_size, Image.Resampling.LANCZOS)
            clean.save(local/'clean.png')
            hint = frame.to_local(asset['bbox'][:2])+frame.to_local(asset['bbox'][2:])
            hint = [max(0., min(v, frame.input_size[i % 2]-1.)) for i, v in enumerate(hint)]
            metadata.update(view_bbox_global=view, bbox_before_global=list(asset['bbox']),
                            bbox_local_hint=hint, purpose=purposes[ident])
            save(local/'frame.json', metadata)
            _overlay(clean, hint, local/'bbox_overlay.png', '#eb00d2')
            source.crop(source_box(asset['bbox'], size, source.size)).save(local/'crop_before.png')
            state['calls'] += 1
            previous_crops[ident] = crop_key
            before = list(asset['bbox'])
            try:
                raw, evidence = cached_json_call(client, f'boundary_{ident}_{round_index:02d}',
                    BOUNDARY_REVIEW_SYSTEM,
                    boundary_review_prompt(asset, frame.input_size, hint, purposes[ident]),
                    [(IMAGE_LABELS['boundary_source'], local/'clean.png'),
                     (IMAGE_LABELS['boundary_overlay'], local/'bbox_overlay.png')],
                    lambda r: validate_review(r, frame.input_size, hint))
                save(local/'review.json', raw)
                entry = {'decision': raw['decision'], 'reason': raw['reason'], 'edges': raw['edges'],
                         'call': evidence, 'frame': str((local/'frame.json').resolve()),
                         'bbox_before_global': before, 'bbox_local_proposal': raw['bbox']}
                record['calls'].append(entry)
                decision = raw['decision']
                if decision == 'accept':
                    record.update(status='model_review_complete', reason=raw['reason'])
                elif decision == 'revise':
                    box = frame.to_global(raw['bbox'][:2])+frame.to_global(raw['bbox'][2:])
                    box = [max(0., min(v, size[i % 2]-1.)) for i, v in enumerate(box)]
                    Crop.from_model(box, size)
                    asset['bbox'] = box
                    # Imported source boxes must not override this new localization downstream.
                    asset['source_box_half_open'] = source_box(box, size, source.size)
                    record.update(bbox=list(box), reason='revised bbox awaits visual verification')
                    purposes[ident] = 'verify_revised_bbox'
                    _overlay(clean, raw['bbox'], local/'proposal_overlay.png', '#05a966')
                elif decision == 'needs_context':
                    actual = [frame.box[0], frame.box[1], frame.box[2]-1, frame.box[3]-1]
                    views[ident] = review_view(union_boxes([actual, view, asset['bbox']]), size, source.size)
                    purposes[ident] = 'expanded_context'
                    record['reason'] = raw['reason']
                else:
                    record.update(status='unresolved', reason=raw['reason'])
                entry['bbox_after_global'] = list(asset['bbox'])
                asset.setdefault('history', []).append({'stage': 'independent_boundary_review', **entry})
                source.crop(source_box(asset['bbox'], size, source.size)).save(local/'crop_after.png')
                save(local/'result.json', entry)
            except Exception as exc:
                record.update(status='unresolved', reason=str(exc))
                save(local/'error.json', {'error': str(exc)})
                if not record['calls'] or record['calls'][-1].get('frame') != str((local/'frame.json').resolve()):
                    record['calls'].append({'decision': 'failed', 'reason': str(exc),
                                            'frame': str((local/'frame.json').resolve())})
            checkpoint()
    for asset in result['assets']:
        record = records[asset['id']]
        if record['status'] == 'pending':
            record.update(status='unresolved', reason='per-asset boundary call budget exhausted; '+record['reason'])
        record['bbox'] = list(asset['bbox'])
        asset['boundary_review'] = {**copy.deepcopy(record), 'visual_accuracy_verified': False}
    checkpoint()
    result['boundary_review'] = {k: copy.deepcopy(v) for k, v in state.items() if k != 'assets'}
    result['boundary_review']['evidence'] = str((folder/'state.json').resolve())
    validate_inventory(result, size)
    save(folder/'inventory.json', result)
    return result
