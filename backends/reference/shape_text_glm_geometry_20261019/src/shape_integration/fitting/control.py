"""Bounded object observations, independent checks and model-selected recrops."""
import math
import re
import shutil
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from PIL import Image, ImageDraw, ImageFont

from ..common import read, save, sha, now, overlay, valid_box, RequestTimeout
from .local import prepare_candidate, observe_and_fit
from .observation import sanitize, sanitize_feedback
from .contracts import protocol_error
from .critic import check_object, revision_feedback
from .paths import sample_commands
from .views import PALETTE, FONT, inspection_style
from .seed import import_seed
from .assets import render_asset_source
from .coordinates import coordinate_mode, model_size, to_pixels, NORM1000_BOX_CONVENTION

SYSTEM = 'Inspect figure geometry using the supplied images. Return only JSON matching the schema. Distinguish source geometry from annotations. Preserve uncertainty.'


def schema(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def validate(raw, spec):
    try:
        Draft202012Validator(spec).validate(raw)
    except ValidationError as exc:
        raise ValueError(protocol_error(exc)) from exc
    return raw


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError('Unsafe case/object ID')
    return value


def prepared(source, candidate, item, folder, options, crop=None, feedback=(), input_context=None):
    """Atomic image preparation; interrupted staging directories are disposable."""
    input_context = input_context or {}
    mode = coordinate_mode(options)
    descriptions = input_context.get('identity_descriptions', [])
    if (folder/'job.json').exists():
        job = read(folder/'job.json')
        if coordinate_mode(job) != mode:
            raise ValueError('Prepared coordinate mode changed')
        if job['candidate'] != candidate or job['identity_descriptions'] != descriptions or job.get('input_context', {}) != input_context or job.get('revision_feedback', []) != list(feedback):
            raise ValueError('Prepared target identity changed')
        return job
    stage = folder.with_name(folder.name+'.preparing')
    if stage.exists():
        shutil.rmtree(stage)
    with Image.open(source) as image:
        job = prepare_candidate(image, candidate, item['input_size'], stage,
            padding=options['crop_padding'], scale=options['crop_scale'],
            min_long_edge=options['crop_min_long_edge'], max_long_edge=options['crop_max_long_edge'],
            crop_bounds_original=crop, descriptions=descriptions, input_context=input_context, feedback=feedback,
            coordinate_mode=mode)
    stage.rename(folder)
    return job


def inspection_images(folder):
    local = read(folder/'fit_local.json')
    with Image.open(folder/'clean.png') as im:
        fit_view = im.convert('RGB'); points_view = im.convert('RGB')
    frame = read(folder/'frame.json') if (folder/'frame.json').exists() else local['frame']
    style = inspection_style(fit_view.size, frame.get('sizing', {}))
    radius = style['point_radius_px']
    fd, pd = ImageDraw.Draw(fit_view), ImageDraw.Draw(points_view)
    font = ImageFont.truetype(FONT, max(12, fit_view.width//90))
    table = {p['id']: p['xy'] for p in local['observation']['points']}
    for i, part in enumerate(local['parts']):
        color = PALETTE[i % len(PALETTE)]
        if part['status'] == 'ok':
            coords = [tuple(p) for p in sample_commands(part['commands'], 64)]
            if len(coords) > 1:
                fd.line(coords, fill=color, width=style['line_width_px'])
                x, y = coords[0]
                fd.text((max(0, min(x+4, fit_view.width-45)), max(0, y-18)), f'F{i+1}',
                        fill=color, font=font, stroke_width=1, stroke_fill='white')
        for pid in part['observed_point_ids']:
            x, y = table[pid]
            box = (x-radius, y-radius, x+radius, y+radius)
            pd.ellipse(box, fill=color)
            fd.ellipse(box, outline=color, width=style['point_outline_width_px'])
    fit_view.save(folder/'fit_overlay.png'); points_view.save(folder/'observed_points.png')
    save(folder/'inspection.json', style)
    render_asset_source(folder)


def recrop(candidate, item, source, input_path, folder, runtime, options, reason):
    """Only this planning request sees the full view; fitting remains local."""
    folder.mkdir(parents=True, exist_ok=True)
    mode = coordinate_mode(options)
    with Image.open(input_path) as image:
        size = image.size
        box = [(v+.5)*size[i % 2]/item['source_size'][i % 2]-.5 for i, v in enumerate(candidate['bbox_2d'])]
        overlay(image, [{**candidate, 'bbox_2d': box}], folder/'target.png')
    spec = schema({'bbox_2d': {'type': 'array', 'minItems': 4, 'maxItems': 4, 'items': {'type': 'number'}},
                   'reason': {'type': 'string', 'minLength': 1}})
    prompt = {'version': 'shape_integration.recrop.1',
              'target': {'id': candidate['id'], 'label': sanitize(candidate['label'])[0]},
              'image_size': list(size), 'context_request': sanitize_feedback(reason),
              'task': 'Choose a crop enclosing the COMPLETE intended target and sufficient context. The magenta box is a fallible identity hint; observe the clean full image. Return inclusive pixel-center crop endpoints in THIS supplied full image. Do not fit parts. Avoid unnecessarily large views.',
              'output_schema': spec}
    if mode == 'norm1000':
        prompt.pop('image_size')
        prompt['local_coordinates'] = {'convention': NORM1000_BOX_CONVENTION}
        prompt['task'] = prompt['task'].replace('Return inclusive pixel-center crop endpoints in THIS supplied full image. ', '')
        spec['properties']['bbox_2d']['items'].update(minimum=0, maximum=1000)
    def check(raw):
        validate(raw, spec)
        if not valid_box(raw['bbox_2d'], model_size(size, mode)):
            raise ValueError('Invalid proposed context crop')
        return raw
    raw = runtime.call(folder/'calls', 'plan', SYSTEM, prompt,
        [('Clean full image', input_path), ('Target identity', folder/'target.png')], check, options['plan_max_tokens'])
    if mode == 'norm1000':
        save(folder/'plan_wire.json', raw)
        raw = {**raw, 'bbox_2d': to_pixels(raw['bbox_2d'], size, mode)}
    # Inclusive centers -> outward pixel edges -> original edges; never expose this transform to fitting.
    b = raw['bbox_2d']; sw, sh = item['source_size']
    crop = [max(0, math.floor(math.floor(b[0])*sw/size[0])),
            max(0, math.floor(math.floor(b[1])*sh/size[1])),
            min(sw, math.ceil((math.ceil(b[2])+1)*sw/size[0])),
            min(sh, math.ceil((math.ceil(b[3])+1)*sh/size[1]))]
    save(folder/'plan.json', {**raw, 'source_crop_half_open': crop})
    return crop


def process_object(root, candidate, item, workspace, runtime, options, *, namespace, prepare_only=False, input_context=None, seed=None, image_assets=()):
    root.mkdir(parents=True, exist_ok=True)
    receipt = root/'done.json'
    if receipt.exists():
        data = read(receipt)
        for name, digest in data['sha256'].items():
            if sha(root/name) != digest:
                raise ValueError('Completed fitting artifact changed: '+name)
        return data['record']
    source, input_path = workspace/item['original'], workspace/item['input']
    save(root/'image_assets.json', [{'id': asset['id'], 'bbox_2d': asset['bbox_2d']} for asset in image_assets])
    input_context = input_context or {}
    seed_job = import_seed(seed, root/'revision_00', candidate, item) if seed else None
    if seed_job:
        input_context = seed_job.get('input_context', {})
    save(root/'input_context.json', input_context)
    observer = runtime.for_stage(shape_schema_retries=options.get('protocol_retries', 1)) if runtime is not None else None
    history, feedback, crop = [], [], None
    if seed_job:
        feedback = seed_job.get('revision_feedback', [])
        crop = seed_job['frame']['source_crop_half_open']
    selected = None; corrections = 0; contexts = 0
    record = {'object_id': candidate['id'], 'state': 'pending', 'selected_revision': None,
              'visual_status': 'unreviewed', 'history': history, 'candidate': candidate}
    for revision in range(1+options['max_revisions']+options['max_context_rounds']):
        folder = root/f'revision_{revision:02d}'
        job = seed_job if revision == 0 and seed_job else prepared(source, candidate, item, folder, options, crop, feedback, input_context)
        if prepare_only:
            save(root/'record.json', record)
            return record
        try:
            reused = revision == 0 and seed is not None and seed['fit_reused']
            status = read(folder/'status.json') if reused else observe_and_fit(job, folder, observer,
                namespace=namespace+f'/revision_{revision:02d}', max_tokens=options['max_tokens'])
            entry = {'revision': revision, 'directory': folder.name, 'observation': status}
            if reused:
                entry['observation_reused_from'] = seed['path']
            history.append(entry)
            if status.get('part_counts', {}).get('succeeded', 0):
                selected = folder.name
            if status['status'] != 'finished':
                record.update(state='unresolved', stop_reason='observation_timeout' if status.get('failure_kind') == 'request_timeout' else 'observation_protocol_exhausted', reason=status.get('error', 'Observation failed'))
                break
            action = 'revise'; reason = 'Reobserve the whole target and return complete geometry.'
            next_feedback = []
            if status['status'] == 'finished':
                action = 'needs_context' if status['observation_status'] == 'needs_context' else 'revise'
                reason = status.get('reason', '') or 'Review all parts and their point references from the clean image.'
                if status['observation_status'] == 'unsupported':
                    record.update(state='unsupported', reason=reason); break
                if status.get('part_counts', {}).get('succeeded', 0) and options['check_enabled']:
                    inspection_images(folder)
                    try:
                        review = check_object(job, folder, runtime, options)
                    except (RuntimeError, ValueError) as exc:
                        entry['check_error'] = str(exc)
                        record.update(state='unresolved', stop_reason='check_timeout' if isinstance(exc, RequestTimeout) else 'check_protocol_exhausted', error=str(exc))
                        break
                    save(folder/'review.json', review); entry['review'] = review
                    action, reason = review['decision'], review['reason']
                    next_feedback = revision_feedback(review)
                    if action == 'unresolved':
                        record.update(state='unresolved', stop_reason='critic_uncertain', reason=reason)
                        break
                    if action == 'accept' and status['fit_status'] == 'ok' and status['observation_status'] == 'ok':
                        record.update(state='accepted_by_model', visual_status='accepted_by_model', reason=reason); break
                    if action == 'accept':
                        action = 'revise'  # A checker cannot erase known incomplete geometry.
                        reason = status.get('reason') or 'Some observed parts are missing or failed numerical fitting.'
                        next_feedback = [sanitize_feedback(reason)]
                    if action == 'reject':
                        record.update(state='rejected_by_model', visual_status='rejected_by_model', reason=reason); break
                elif status['fit_status'] == 'ok' and not options['check_enabled']:
                    record.update(state='fitted_unreviewed'); break
            record.update(state='unresolved', reason=reason)
            if action == 'needs_context':
                if contexts >= options['max_context_rounds']:
                    record['stop_reason'] = 'context_budget_exhausted'; break
                next_crop = recrop(candidate, item, source, input_path, root/f'context_{contexts:02d}', runtime, options,
                                   '\n'.join(next_feedback) if next_feedback else reason)
                if next_crop == job['frame']['source_crop_half_open']:
                    record['stop_reason'] = 'repeated_context'; break
                contexts += 1; crop = next_crop
            else:
                if corrections >= options['max_revisions']:
                    record['stop_reason'] = 'revision_budget_exhausted'; break
                corrections += 1
            feedback = next_feedback or [sanitize_feedback(reason)]
        except Exception as exc:
            record.update(state='unresolved', stop_reason='request_timeout' if isinstance(exc, RequestTimeout) else 'execution_error', error=str(exc)); break
        finally:
            record['selected_revision'] = selected
            save(root/'record.json', record)
    record.update(selected_revision=selected, finished_utc=now(), context_rounds=contexts, corrections=corrections)
    save(root/'record.json', record)
    hashes = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'done.json'}
    save(receipt, {'record': record, 'sha256': hashes})
    return record
