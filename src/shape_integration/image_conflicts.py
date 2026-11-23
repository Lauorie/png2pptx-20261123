"""Visual batch review of rule-selected shape/image conflicts before fitting."""
from pathlib import Path
import copy

from PIL import Image, ImageDraw, ImageFont
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from .common import FONT, intersect, mapped_box, save, sha
from .dedup import filter_candidates
from png2pptx.images.ownership import IMAGE_OWNERSHIP

VERSION = 'shape_image_conflict_batch.1'
SYSTEM = ('Review selected shape/image conflicts in a scientific figure using visual evidence. '
          'Return concise JSON decisions only. Do not reconstruct geometry or detect new objects.')
COLORS = {'S': '#d10070', 'I': '#008eb8'}


def prompt_for(packets):
    candidates = [{'shape_id': 'S_'+p['shape']['id'],
                   'image_ids': ['I_'+m['asset']['id'] for m in p['overlapping_images']]}
                  for p in packets]
    shape_ids = [c['shape_id'] for c in candidates]
    image_ids = sorted({i for c in candidates for i in c['image_ids']})
    spec = {'type': 'object', 'additionalProperties': False, 'required': ['decisions'], 'properties': {
        'decisions': {'type': 'array', 'minItems': len(candidates), 'maxItems': len(candidates), 'items': {
            'type': 'object', 'additionalProperties': False,
            'required': ['shape_id', 'decision', 'image_id', 'reason'], 'properties': {
                'shape_id': {'enum': shape_ids}, 'decision': {'enum': ['delete', 'keep', 'defer']},
                'image_id': {'enum': image_ids+[None]}, 'reason': {'type': 'string', 'minLength': 1}}}}}}
    return {'version': VERSION,
            'task': 'Review ONLY the listed shape candidates as ONE batch. Decide whether each should be removed from the shape queue because the same object belongs to an image asset.',
            'images': ['Image 1: clean original figure.',
                       'Image 2: the same figure with selected shape boxes (magenta S_) and their matched image boxes (cyan I_).'],
            'candidates': candidates,
            'rules': [
                'The two prefixes indicate detector sources, not confirmed classes. Inspect the clean figure. Boxes and labels are annotations, not object contours or physical occluders; their positions can be inaccurate.',
                IMAGE_OWNERSHIP,
                'delete only when the shape and a listed image identify the SAME in-scope image visual unit. Return its image_id. Keep independent native geometry, and any target containing additional independently owned parts; overlapping boxes are only proposals.',
                'Overlap, enclosure or adjacency alone never justify deletion. A frame around an image can be a separate shape; repeated icons at distinct locations are separate instances. Keep a candidate containing additional independent shape content.',
                'Use defer if identity or image ownership cannot be resolved from the supplied views. For keep/defer set image_id to null. Each supplied shape_id must appear exactly once; do not classify the whole figure, output image-only decisions, add objects, or change boxes.',
                'Give one short reason grounded in visible content for each decision. No coordinates, geometric reconstruction or long reports.'],
            'output_schema': spec}


def validate_response(raw, prompt):
    try:
        Draft202012Validator(prompt['output_schema']).validate(raw)
    except ValidationError as exc:
        raise ValueError(f"{'.'.join(map(str, exc.absolute_path))}: {exc.message}") from exc
    expected = {c['shape_id']: c['image_ids'] for c in prompt['candidates']}
    ids = [row['shape_id'] for row in raw['decisions']]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError('Return each selected shape_id exactly once')
    for row in raw['decisions']:
        if not row['reason'].strip():
            raise ValueError('A visible-evidence reason is required')
        if row['decision'] == 'delete':
            if row['image_id'] not in expected[row['shape_id']]:
                raise ValueError('delete must reference an image paired with this shape')
        elif row['image_id'] is not None:
            raise ValueError('keep/defer must use image_id=null')
    return raw


def render_boxes(clean, rows, path):
    canvas = clean.copy(); draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, 14); occupied = []
    for row in rows:
        draw.rectangle(row['display_bbox'], outline=COLORS[row['origin']], width=2)
    for row in sorted(rows, key=lambda r: (r['display_bbox'][1], r['display_bbox'][0])):
        box = row['display_bbox']; color = COLORS[row['origin']]; label = row['display_id']
        tb = draw.textbbox((0, 0), label, font=font); w, h = tb[2]-tb[0]+4, tb[3]-tb[1]+4
        places = []
        for left in [box[0], box[2]-w]:
            for top in [box[1]-h-2, box[3]+2, box[1]-2*h-4, box[3]+h+4]:
                x = max(0, min(left, canvas.width-w)); y = max(0, min(top, canvas.height-h))
                candidate = [x, y, x+w, y+h]
                score = sum(intersect(candidate, old) for old in occupied)
                score += sum(intersect(candidate, r['display_bbox']) for r in rows)
                places.append((score, candidate))
        chosen = min(places, key=lambda pair: pair[0])[1]; occupied.append(chosen)
        x, y = chosen[:2]; center = (x+w/2, y+h/2)
        anchor = (max(box[0], min(center[0], box[2])), max(box[1], min(center[1], box[3])))
        draw.line([center, anchor], fill=color, width=1)
        draw.rectangle(chosen, fill='white'); draw.text((x+2, y+2-tb[1]), label, font=font, fill=color)
    canvas.save(path)


class VisualBatchReviewer:
    def __init__(self, source, folder, runtime, options, prepare_only=False):
        self.source = source; self.folder = Path(folder); self.runtime = runtime
        self.options = options; self.prepare_only = prepare_only; self.batches = []
        self.clean = None

    def __call__(self, packets):
        if self.clean is None:
            if sha(self.source['path']) != self.source['sha256']:
                raise ValueError('Conflict-review source image changed')
            with Image.open(self.source['path']) as image:
                self.source_size = image.size; clean = image.convert('RGB')
            ratio = 2048/max(clean.size)
            self.clean = clean.resize(tuple(max(1, round(v*ratio)) for v in clean.size), Image.Resampling.LANCZOS)
            self.folder.mkdir(parents=True, exist_ok=True); self.clean.save(self.folder/'clean.png')
        index = len(self.batches); folder = self.folder/f'batch_{index:03d}'; folder.mkdir(exist_ok=True)
        images = {m['asset']['id']: m['asset'] for p in packets for m in p['overlapping_images']}
        rows = []
        for origin, objects in [('S', [p['shape'] for p in packets]), ('I', list(images.values()))]:
            for obj in objects:
                rows.append({'display_id': origin+'_'+obj['id'], 'origin': origin, 'original_id': obj['id'],
                             'original_bbox': obj['bbox_2d'],
                             'display_bbox': mapped_box(obj['bbox_2d'], self.source_size, self.clean.size)})
        render_boxes(self.clean, rows, folder/'candidates.png')
        prompt = prompt_for(packets)
        save(folder/'prompt.json', {'system': SYSTEM, 'prompt': prompt})
        save(folder/'backend.json', {'source': self.source, 'source_size': self.source_size,
             'display_size': self.clean.size, 'rows': rows, 'selected_conflicts': packets,
             'image_sha256': {'clean': sha(self.folder/'clean.png'), 'overlay': sha(folder/'candidates.png')}})
        status = {'batch': index, 'candidate_ids': [p['shape']['id'] for p in packets]}
        save(folder/'status.json', {**status, 'state': 'prepared' if self.prepare_only else 'running'})
        try:
            if self.prepare_only:
                raise RuntimeError('Review pending: prepare-only does not call the model')
            raw = self.runtime.call(self.folder/'calls', f'batch_{index:03d}', SYSTEM, prompt,
                [('Clean original figure', self.folder/'clean.png'), ('Selected conflict boxes', folder/'candidates.png')],
                lambda raw: validate_response(raw, prompt), self.options.get('max_tokens', 100000))
            answers = [{**r, 'shape_id': r['shape_id'][2:],
                        'image_id': r['image_id'][2:] if r['image_id'] is not None else None}
                       for r in raw['decisions']]
            status['state'] = 'completed'
        except (RuntimeError, ValueError) as exc:
            status.update(state='prepared' if self.prepare_only else 'failed', error=str(exc))
            answers = [{'shape_id': p['shape']['id'], 'decision': 'defer', 'image_id': None,
                        'reason': 'Review pending' if self.prepare_only else 'Batch review failed; retain shape. '+str(exc)}
                       for p in packets]
        save(folder/'decisions.json', answers); save(folder/'status.json', status)
        self.batches.append(status)
        return answers


def run_filter(shapes, assets, source, folder, runtime, config, *, prepare_only=False):
    """Shared gate for localization results and newly discovered fitting candidates."""
    if config.get('image_ownership_stage') == 'composition':
        # A registry entry is a proposal, not a committed alternative. The
        # compositor selects a representation after seeing actual native PPT
        # and deletes exact leaves only after its image is realized.
        options = config.get('image_conflict_review', {})
        result = filter_candidates(shapes, assets,
            options.get('candidate_iou_threshold', .4), mode='rule_only', source=source)
        for row in result['decisions']:
            row.update(decision='keep', rule_proposed_delete=False,
                reason='retain_until_image_representation_commits',
                decision_source='composition_ownership')
        for packet in result['review_requests']:
            packet['execution_status'] = 'ownership_deferred_to_composition'
        result.update(kept=copy.deepcopy(shapes), removed=[], mode='composition_ownership',
            deferred_ids=[], reviewer_calls=0, review_batches=[], model_request_calls=0,
            review_failed_batches=0, review_pending=False, ownership_stage='composition')
        save(Path(folder)/'summary.json', {k:result[k] for k in (
            'mode','threshold','review_candidates','reviewer_calls','model_request_calls',
            'review_failed_batches','review_pending','deferred_ids','ownership_stage')})
        return result
    enabled = config.get('model_conflict_review', False)
    options = config.get('image_conflict_review', {})
    reviewer = VisualBatchReviewer(source, folder, runtime, options, prepare_only) if enabled else None
    if enabled and runtime is None and not prepare_only:
        raise ValueError('Model conflict review requires a runtime')
    result = filter_candidates(shapes, assets,
        options.get('candidate_iou_threshold', .4) if enabled else config['iou_threshold'],
        mode='model_review' if enabled else 'rule_only', reviewer=reviewer, source=source,
        review_batch_size=options.get('batch_size', 32))
    result['review_batches'] = reviewer.batches if reviewer else []
    # Batch tasks and wire calls (including retries) are distinct counts.
    result['model_request_calls'] = len(list(Path(folder).glob('calls/*/request.json')))
    result['review_failed_batches'] = sum(b['state'] == 'failed' for b in result['review_batches'])
    result['review_pending'] = any(b['state'] == 'prepared' for b in result['review_batches'])
    save(Path(folder)/'summary.json', {k: result[k] for k in ['mode', 'threshold', 'review_candidates',
        'reviewer_calls', 'model_request_calls', 'review_batches', 'review_failed_batches', 'review_pending', 'deferred_ids']})
    return result
