"""Parallel image boundary reviews with the upstream visual protocol unchanged.

The coordinator owns inventory/state writes and reserves the per-figure budget in
inventory order. Independent objects run on one shared pool; a round barrier keeps
repairs behind every object's first review. API clients retain Runtime.slots.
"""
import copy
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading

from PIL import Image
from png2pptx.frames import Crop
from png2pptx.images import boundary as upstream
from png2pptx.images.clients import cached_json_call, digest, file_sha
from png2pptx.images.detect import source_box, validate_inventory
from png2pptx.images.prompts import BOUNDARY_REVIEW_SYSTEM, IMAGE_LABELS, boundary_review_prompt
from png2pptx.images.search import native_crop, union_boxes
from .common import read, save


class BoundaryScheduler:
    """Share workers across figures, releasing them when the last review finishes."""
    def __init__(self, workers):
        if not isinstance(workers, int) or workers < 1:
            raise ValueError('Boundary workers must be a positive integer')
        self.workers = workers
        self._lock = threading.Lock()
        self._pool = None
        self._users = 0

    def __call__(self, *args, **kwargs):
        with self._lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=self.workers,
                                                thread_name_prefix='image-boundary')
            pool = self._pool
            self._users += 1
        try:
            return review_boundaries(*args, **kwargs, executor=pool, workers=self.workers)
        finally:
            with self._lock:
                self._users -= 1
                if self._users == 0:
                    pool.shutdown(wait=True)
                    self._pool = None


def _review_one(client, source, size, folder, round_index, job):
    """Update only private copies; every worker writes a distinct asset/round dir."""
    asset, record = job['asset'], job['record']
    ident = asset['id']
    frame, native, metadata = job['frame'], job['native'], job['metadata']
    view, next_view, purpose = job['view'], job['next_view'], job['purpose']
    local = folder/ident/f'round_{round_index:02d}'
    local.mkdir(parents=True, exist_ok=True)
    native.save(local/'source_native.png')
    clean = native.resize(frame.input_size, Image.Resampling.LANCZOS)
    clean.save(local/'clean.png')
    hint = frame.to_local(asset['bbox'][:2])+frame.to_local(asset['bbox'][2:])
    hint = [max(0., min(v, frame.input_size[i % 2]-1.)) for i, v in enumerate(hint)]
    metadata.update(view_bbox_global=view, bbox_before_global=list(asset['bbox']),
                    bbox_local_hint=hint, purpose=purpose)
    save(local/'frame.json', metadata)
    upstream._overlay(clean, hint, local/'bbox_overlay.png', '#eb00d2')
    source.crop(source_box(asset['bbox'], size, source.size)).save(local/'crop_before.png')
    before = list(asset['bbox'])
    try:
        raw, evidence = cached_json_call(client, f'boundary_{ident}_{round_index:02d}',
            BOUNDARY_REVIEW_SYSTEM,
            boundary_review_prompt(asset, frame.input_size, hint, purpose),
            [(IMAGE_LABELS['boundary_source'], local/'clean.png'),
             (IMAGE_LABELS['boundary_overlay'], local/'bbox_overlay.png')],
            lambda r: upstream.validate_review(r, frame.input_size, hint))
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
            asset['source_box_half_open'] = source_box(box, size, source.size)
            record.update(bbox=list(box), reason='revised bbox awaits visual verification')
            purpose = 'verify_revised_bbox'
            upstream._overlay(clean, raw['bbox'], local/'proposal_overlay.png', '#05a966')
        elif decision == 'needs_context':
            actual = [frame.box[0], frame.box[1], frame.box[2]-1, frame.box[3]-1]
            next_view = upstream.review_view(union_boxes([actual, view, asset['bbox']]), size, source.size)
            purpose = 'expanded_context'
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
    return asset, record, next_view, purpose


def review_boundaries(client, source_path, inventory, size, folder,
                      max_calls=64, calls_per_asset=2, *, executor, workers):
    if max_calls < 0 or calls_per_asset < 1:
        raise ValueError('Boundary budgets require max_calls >= 0 and calls_per_asset >= 1')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    result = copy.deepcopy(validate_inventory(inventory, size))
    identity = {'source_sha256': file_sha(source_path), 'inventory_sha256': digest(inventory),
                'input_size': list(size), 'max_calls': max_calls, 'calls_per_asset': calls_per_asset,
                'parameters': client.request_parameters() if client else None,
                'endpoint': client.base_url if client else None,
                'scheduling': 'parallel_assets_round_barrier_v1', 'worker_limit': workers,
                'code_sha256': {'image_boundary.py': file_sha(Path(__file__)),
                               **{name: file_sha(Path(upstream.__file__).with_name(name))
                                  for name in ('boundary.py', 'prompts.py', 'search.py')}}}
    identity_path = folder/'identity.json'
    if identity_path.exists() and read(identity_path) != identity:
        raise ValueError('Boundary review inputs/code changed; choose a new output directory')
    save(identity_path, identity)
    with Image.open(source_path) as opened:
        source = opened.convert('RGB')
    records = {a['id']: {'status': 'pending', 'original_bbox': list(a['bbox']),
                        'bbox': list(a['bbox']), 'calls': [], 'reason': 'awaiting independent review'}
               for a in result['assets']}
    state = {'calls': 0, 'max_calls': max_calls, 'calls_per_asset': calls_per_asset,
             'assets': records, 'visual_accuracy_verified': False}
    views = {a['id']: upstream.review_view(a['bbox'], size, source.size) for a in result['assets']}
    purposes = {a['id']: 'initial_review' for a in result['assets']}
    previous_crops = {}
    completed_calls = 0
    pending_jobs = {}

    def checkpoint():
        pending = [i for i, r in records.items() if r['status'] != 'model_review_complete']
        state['status'] = ('not_needed' if not records else
                           'unresolved' if pending else 'model_review_complete')
        state['unresolved_ids'] = pending
        state['decisions'] = dict(Counter(c['decision'] for r in records.values() for c in r['calls']))
        save(folder/'state.json', state)
        save(folder/'scheduling.json', {'mode': identity['scheduling'], 'worker_limit': workers,
             'scheduled_calls': state['calls'], 'completed_calls': completed_calls,
             'pending_asset_ids': [result['assets'][i]['id'] for i in pending_jobs.values()]})

    checkpoint()
    for round_index in range(calls_per_asset):
        # Reserve in inventory order before consuming results. Completion order
        # cannot choose which objects get the remaining visual-stage budget.
        pending_jobs = {}
        for index, asset in enumerate(result['assets']):
            ident = asset['id']
            record = records[ident]
            if record['status'] != 'pending':
                continue
            if state['calls'] >= max_calls:
                record.update(status='unresolved', reason='total boundary call budget exhausted')
                continue
            view = union_boxes([views[ident], asset['bbox']])
            frame, native, metadata = native_crop(source, view, size)
            crop_key = tuple(metadata['source_crop_half_open'])
            if purposes[ident] == 'expanded_context' and crop_key == previous_crops.get(ident):
                record.update(status='unresolved', reason='no larger distinct source view available')
                continue
            state['calls'] += 1
            previous_crops[ident] = crop_key
            job = {'asset': copy.deepcopy(asset), 'record': copy.deepcopy(record),
                   'frame': frame, 'native': native, 'metadata': metadata,
                   'view': view, 'next_view': views[ident], 'purpose': purposes[ident]}
            future = executor.submit(_review_one, client, source, size, folder, round_index, job)
            pending_jobs[future] = index
        checkpoint()
        for future in as_completed(tuple(pending_jobs)):
            index = pending_jobs.pop(future)
            asset, record, view, purpose = future.result()
            result['assets'][index] = asset
            ident = asset['id']
            records[ident], views[ident], purposes[ident] = record, view, purpose
            completed_calls += 1
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
