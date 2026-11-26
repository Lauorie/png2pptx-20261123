"""Prepare one existing candidate and observe all its parts with one API call."""
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ..common import read, save, sha, now, RequestTimeout
from .adapter import fit_object, transform_fit
from .contracts import MODEL_VERSION, normalize_observation
from .frames import FitFrame
from .observation import system_for, make_prompt, sanitize
from .views import comparison, FONT
from .sizing import crop_size
from .coordinates import coordinate_mode as get_coordinate_mode, model_size, observation_to_pixels


def prepare_candidate(source, candidate, global_size, folder, *, padding=.12, long_edge=None,
                      scale=3., min_long_edge=1080, max_long_edge=2048,
                      crop_bounds_original=None, descriptions=(), input_context=None, feedback=(),
                      coordinate_mode='pixel', spatial_evidence=None, revision_context=None):
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
    # Context-only hint: a zoom/speech pointer can extend outside a body-only
    # detector box. Its topology still has to be observed from source pixels.
    if re.search(r'\b(callout|zoom|inset|speech)\b', candidate.get('label',''), re.I):
        padding = max(padding, .35)
    scope=(input_context or {}).get('coverage_source_extent_original')
    if scope is not None and (len(scope)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in scope)
            or not 0<=scope[0]<scope[2]<=w-1 or not 0<=scope[1]<scope[3]<=h-1):
        raise ValueError('Coverage source extent must use original source pixels')
    crop_box=([min(box[0],scope[0]),min(box[1],scope[1]),max(box[2],scope[2]),max(box[3],scope[3])]
              if scope is not None else box)
    pad = padding*max(crop_box[2]-crop_box[0], crop_box[3]-crop_box[1])
    crop = [max(0, math.floor(crop_box[0]-pad)), max(0, math.floor(crop_box[1]-pad)),
            min(w, math.ceil(crop_box[2]+pad)+1), min(h, math.ceil(crop_box[3]+pad)+1)]
    if crop_bounds_original is not None:
        crop = list(crop_bounds_original)
        if (len(crop) != 4 or any(type(v) is not int for v in crop)
                or not 0 <= crop[0] < crop[2] <= w or not 0 <= crop[1] < crop[3] <= h):
            raise ValueError('Invalid source half-open crop')
    support = None
    if spatial_evidence is not None and hasattr(spatial_evidence, 'prepare'):
        crop, support = spatial_evidence.prepare(candidate, crop, source.size)
    if scope is not None:
        crop=[min(crop[0],math.floor(scope[0])),min(crop[1],math.floor(scope[1])),
              max(crop[2],math.ceil(scope[2])+1),max(crop[3],math.ceil(scope[3])+1)]
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
    hint = scope if scope is not None else support.get('identity_bbox_original', support['bbox_original']) if support else box
    local_box = frame.to_local(hint[:2])+frame.to_local(hint[2:])
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
    if support:
        job['sam_support'] = support
    if revision_context is not None:
        from .revision import reframe_context
        job['revision_context'] = reframe_context(revision_context,meta)
    from .coverage_evidence import annotate
    annotate(overlay,job)
    from .assets import annotate_assets
    registry = folder.parent/'image_assets.json'
    assets = read(registry) if registry.exists() else []
    shown = annotate_assets(overlay, frame, assets)
    if shown:
        job['image_asset_ids'] = [row['id'] for row in shown]
        save(folder/'observer_assets.json', {'assets': shown})
    overlay.save(folder/'overlay.png')
    save(folder/'job.json', job)
    save(folder/'frame.json', meta)
    save(folder/'candidate.json', candidate)
    prompt = make_prompt(job, descriptions)
    save(folder/'prompt.json', {'system': system_for(job), 'prompt': prompt})
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
    if request != {'system': system_for(job), 'prompt': make_prompt(job, job.get('identity_descriptions', ()))}:
        raise ValueError('Prepared prompt differs from current job/code')
    record = {'object_id': job['object_id'], 'started_utc': now(), 'status': 'running'}
    save(folder/'status.json', record)
    received = {}
    def normalize(value, *, canonical=False):
        audit = {}
        if (isinstance(value, dict) and not job.get('image_asset_ids') and any(
                p.get('role') == 'image_content' for p in (value.get('ownership') or {}).get('parts', []))):
            raise ValueError('image_content requires a registered IMAGE footprint in this crop')
        from .observation import source_clipping
        clipping = source_clipping(job)
        if (isinstance(value, dict) and value.get('status') == 'needs_context'
                and clipping and len(clipping['image_edges']) == 4):
            raise ValueError('CLEAN contains the complete original image. Observe its visible target and original clipping; use partial or unsupported when the visible geometry cannot be recovered.')
        context = job.get('revision_context')
        removal_only = (context is not None and not context['editable_ids']
                        and not context['allow_new_parts'] and isinstance(value, dict)
                        and value.get('status') == 'ok' and value.get('points') == [] and value.get('parts') == [])
        wire = {**value, 'status': 'partial', 'reason': 'Removal-only edit'} if removal_only else value
        result = normalize_observation(wire, job['object_id'], model_size(frame.input_size, mode),
                                       audit=audit, recover_parts=True, ownership_required=True,
                                       allow_compound_connections=job.get('proposed_type')=='compound_symbol')
        delegated_only = False
        if context and (audit.get('declared_ownership') or {}).get('delegated_parts'):
            delegated={p['id'] for p in audit['declared_ownership']['delegated_parts']}
            editable=set(context.get('editable_ids',[]))
            if delegated-editable:
                raise ValueError('Ownership declarations cannot retire unrequested protected parts')
            context={**context,'remove_ids':sorted(set(context.get('remove_ids',[]))|delegated),
                     'editable_ids':sorted(editable-delegated)}
            delegated_only = (value.get('status')=='ok' and not result['parts']
                and delegated=={p['id']for p in value['parts']})
        if context:
            from .revision import merge_revision
            # Merging uses pixel coordinates. Runtime caches the resulting
            # observation in its original model units as before.
            pixels = observation_to_pixels(result, frame.input_size, mode)
            if removal_only or delegated_only:
                pixels['status'] = 'ok'
            merged = {}
            pixels = merge_revision(pixels, context, job['object_id'], frame.input_size, audit=merged)
            result = pixels
            if mode == 'norm1000':
                for point in result['points']:
                    point['xy'] = [v * 1000 / (frame.input_size[i] - 1) for i, v in enumerate(point['xy'])]
            audit['revision_merge'] = merged
        received.update(raw=value, audit=audit)
        # Preserve the declared wire response in Runtime caches. Compiled
        # shape geometry is materialized only at this branch boundary.
        return result if canonical else value
    try:
        prompt = request['prompt']
        images = [('Image 1: clean crop', folder/'clean.png'),
                  ('Image 2: aligned target box and ID', folder/'overlay.png')]
        evidence = getattr(runtime, 'spatial_evidence', None)
        if evidence is not None:
            prompt, images = evidence.observe(prompt, images, job, folder)
        raw = runtime.call(folder/'calls', 'observe', request['system'], prompt,
                           images,
                           normalize,
                           max_tokens)
        raw = normalize(raw, canonical=True)
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
        from .ownership import retained_receipt
        local['declared_ownership'] = retained_receipt(received['audit'].get('declared_ownership'),
            (job.get('revision_context') or {}).get('base_ownership'), [p['id']for p in local['parts']])
        if getattr(runtime,'config',{}).get('fitting', {}).get('projected_edge_refinement', False):
            from .projected_edges import refine
            with Image.open(folder/'clean.png') as clean:
                direction_audit=refine(local,job,clean)
            save(folder/'projected_edge_refinement.json',direction_audit)
        if getattr(runtime,'config',{}).get('fitting', {}).get('source_frame_refinement', True):
            from .frame_edges import refine, refine_outline
            with Image.open(folder/'clean.png') as clean:
                edge_audit=refine(local,job,clean,evidence)
                outline_audit=refine_outline(local,job,clean,evidence)
            save(folder/'frame_edge_refinement.json',edge_audit)
            save(folder/'outline_edge_refinement.json',outline_audit)
        if getattr(runtime,'config',{}).get('fitting', {}).get('sam_radius_refinement', False):
            from .rounded_corners import refine
            with Image.open(folder/'clean.png') as clean:
                radius_audit=refine(local,job,clean.convert('RGB'),evidence)
            save(folder/'radius_refinement.json',radius_audit)
        failed_replacements = []
        if job.get('revision_context'):
            from .revision import preserve_fits
            failed_replacements = preserve_fits(local, job['revision_context'])
        original = transform_fit(local)
        save(folder/'fit_local.json', local)
        save(folder/'fit_original.json', original)
        comparison(folder/'clean.png', raw, local, folder/'comparison.png')
        record.update(status='finished', observation_status=raw['status'], fit_status=local['fit_status'],
                      completeness_status=local['completeness_status'], part_counts=local['part_counts'],
                      kinds=[p['kind'] for p in local['parts']], point_count=len(raw['points']),
                      declared_ownership=local.get('declared_ownership'),
                      reason=raw['reason'], uncertain=raw['uncertain'], observation_call=observation_call)
        if job.get('revision_context'):
            record['revision_merge'] = received['audit'].get('revision_merge', {})
            record['failed_replacements'] = failed_replacements
    except (RuntimeError, ValueError) as exc:
        record.update(status='failed', error=str(exc))
        if isinstance(exc, RequestTimeout):
            record.update(failure_kind='request_timeout', output_discarded=True)
        elif type(exc).__name__=='ModelTransportError':
            record.update(failure_kind='model_transport',output_discarded=True)
        elif type(exc).__name__=='ModelRefusedError':
            record.update(failure_kind='model_refused',output_discarded=True)
    record['finished_utc'] = now()
    save(folder/'status.json', record)
    return record
