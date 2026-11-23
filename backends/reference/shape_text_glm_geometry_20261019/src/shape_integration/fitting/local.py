"""Prepare one existing candidate and observe all its parts with one API call."""
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ..common import read, save, sha, now, RequestTimeout
from .adapter import fit_object, transform_fit
from .contracts import MODEL_VERSION, normalize_observation
from .frames import FitFrame
from .observation import SYSTEM, make_prompt, sanitize
from .views import comparison, FONT
from .sizing import crop_size
from .coordinates import coordinate_mode as get_coordinate_mode, model_size, observation_to_pixels


def prepare_candidate(source, candidate, global_size, folder, *, padding=.12, long_edge=None,
                      scale=3., min_long_edge=1080, max_long_edge=2048,
                      crop_bounds_original=None, descriptions=(), input_context=None, feedback=(),
                      coordinate_mode='pixel'):
    """Crop original pixels; bbox and transport stay in backend artifacts."""
    get_coordinate_mode({'coordinate_mode': coordinate_mode})
    if (not math.isfinite(padding) or padding < 0
            or (long_edge is not None and (type(long_edge) is not int or long_edge < 2))):
        raise ValueError('Invalid crop settings')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    w, h = source.size
    box = candidate['bbox_2d']
    if (len(box) != 4 or any(type(v) not in (float, int) or not math.isfinite(v) for v in box)
            or not 0 <= box[0] < box[2] <= w-1 or not 0 <= box[1] < box[3] <= h-1):
        raise ValueError('Candidate bbox must be in original image pixel centers')
    pad = padding*max(box[2]-box[0], box[3]-box[1])
    crop = [max(0, math.floor(box[0]-pad)), max(0, math.floor(box[1]-pad)),
            min(w, math.ceil(box[2]+pad)+1), min(h, math.ceil(box[3]+pad)+1)]
    if crop_bounds_original is not None:
        crop = list(crop_bounds_original)
        if (len(crop) != 4 or any(type(v) is not int for v in crop)
                or not 0 <= crop[0] < crop[2] <= w or not 0 <= crop[1] < crop[3] <= h):
            raise ValueError('Invalid source half-open crop')
    native = source.convert('RGB').crop(crop)
    bounds_global = tuple(v*global_size[i % 2]/source.size[i % 2] for i, v in enumerate(crop))
    extent = (bounds_global[2]-bounds_global[0], bounds_global[3]-bounds_global[1])
    size, sizing = crop_size(extent, scale=scale, min_long_edge=min_long_edge, max_long_edge=max_long_edge)
    if long_edge is not None:  # Explicit compatibility mode for frozen example runners.
        size = tuple(max(2, round(v*long_edge/max(native.size))) for v in native.size)
        sizing = {'policy': 'legacy_fixed_long_edge', 'long_edge': long_edge}
    frame = FitFrame(size, tuple(global_size), source.size, bounds_global)
    meta = {**frame.metadata(), 'source_crop_half_open': crop,
            'sampled_from': 'original_resolution', 'source_crop_size': list(native.size),
            'padding_long_edge_fraction': padding, 'sizing': sizing}
    FitFrame.from_metadata(meta)
    native.save(folder/'native.png')
    clean = native.resize(size, Image.Resampling.LANCZOS)
    clean.save(folder/'clean.png')
    local_box = frame.to_local(box[:2])+frame.to_local(box[2:])
    overlay = clean.copy(); draw = ImageDraw.Draw(overlay)
    edge = max(size)
    font = ImageFont.truetype(FONT, max(14, edge//70))
    draw.rectangle(local_box, outline='#e00090', width=max(2, edge//600))
    x, y = local_box[:2]
    draw.text((max(0, x), max(0, y-edge//45)), candidate['id'], font=font, fill='#e00090', stroke_width=1, stroke_fill='white')
    overlay.save(folder/'overlay.png')
    job = {'object_id': candidate['id'], 'label': candidate['label'], 'proposed_type': candidate['kind'],
           'frame': meta, 'candidate': candidate, 'identity_bbox_local': local_box,
           'identity_descriptions': list(descriptions), 'input_context': input_context or {},
           'revision_feedback': list(feedback), 'observation_protocol': MODEL_VERSION}
    if coordinate_mode != 'pixel':
        job['coordinate_mode'] = coordinate_mode
    save(folder/'job.json', job)
    save(folder/'frame.json', meta)
    save(folder/'candidate.json', candidate)
    prompt = make_prompt(job, descriptions)
    save(folder/'prompt.json', {'system': SYSTEM, 'prompt': prompt})
    save(folder/'description_audit.json', {'original_label': candidate['label'],
                                          'sanitized': sanitize(candidate['label'])})
    job['input_sha256'] = {name: sha(folder/name) for name in ['clean.png', 'overlay.png', 'prompt.json', 'frame.json', 'candidate.json']}
    save(folder/'job.json', job)
    return job


def observe_and_fit(job, folder, runtime, *, namespace, max_tokens):
    """One whole-object observation per revision, with independent protocol retries."""
    folder = Path(folder)
    for name, digest in job['input_sha256'].items():
        if sha(folder/name) != digest:
            raise ValueError('Prepared input changed: '+name)
    frame = FitFrame.from_metadata(job['frame'])
    mode = get_coordinate_mode(job)
    request = read(folder/'prompt.json')
    if request != {'system': SYSTEM, 'prompt': make_prompt(job, job.get('identity_descriptions', ()))}:
        raise ValueError('Prepared prompt differs from current job/code')
    record = {'object_id': job['object_id'], 'started_utc': now(), 'status': 'running'}
    save(folder/'status.json', record)
    received = {}
    def normalize(value):
        audit = {}
        result = normalize_observation(value, job['object_id'], model_size(frame.input_size, mode), audit=audit)
        received.update(raw=value, audit=audit)
        return result
    try:
        raw = runtime.call(folder/'calls', 'observe', request['system'], request['prompt'],
                           [('Image 1: clean crop', folder/'clean.png'),
                            ('Image 2: aligned target box and ID', folder/'overlay.png')],
                           normalize,
                           max_tokens)
        if mode == 'norm1000':
            save(folder/'observation_wire.json', raw)
            raw = observation_to_pixels(raw, frame.input_size, mode)
            save(folder/'coordinate_transform.json', {
                'model_coordinates': mode, 'backend_coordinates': 'image_pixel_centers',
                'input_size_px': list(frame.input_size),
                'scale_xy': [(d-1)/1000 for d in frame.input_size],
                'wire_observation': 'observation_wire.json', 'pixel_observation': 'observation.json'})
        save(folder/'observation.json', raw)
        calls = sorted((folder/'calls').glob('observe_*/result.json'))
        observation_call = str(calls[-1].parent.relative_to(folder)) if calls else None
        if calls and (calls[-1].parent/'parsed_response.json').exists():
            # A resumed call validates its canonical result; recover the original
            # wire object for an accurate normalization audit instead.
            normalize(read(calls[-1].parent/'parsed_response.json'))
        save(folder/'observation_raw.json', received['raw'])
        save(folder/'observation_normalization.json', received['audit'])
        local = fit_object(raw, frame, object_id=job['object_id'], namespace=namespace,
                           provenance={'candidate_sha256': job['input_sha256']['candidate.json'],
                                       'observation_call': observation_call})
        original = transform_fit(local)
        save(folder/'fit_local.json', local)
        save(folder/'fit_original.json', original)
        comparison(folder/'clean.png', raw, local, folder/'comparison.png')
        record.update(status='finished', observation_status=raw['status'], fit_status=local['fit_status'],
                      completeness_status=local['completeness_status'], part_counts=local['part_counts'],
                      kinds=[p['kind'] for p in local['parts']], point_count=len(raw['points']),
                      reason=raw['reason'], uncertain=raw['uncertain'], observation_call=observation_call)
    except (RuntimeError, ValueError) as exc:
        record.update(status='failed', error=str(exc))
        if isinstance(exc, RequestTimeout):
            record.update(failure_kind='request_timeout', output_discarded=True)
    record['finished_utc'] = now()
    save(folder/'status.json', record)
    return record
