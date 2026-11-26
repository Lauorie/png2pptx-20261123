"""Bounded object observations, independent checks and model-selected recrops."""
import math
import fcntl
from urllib.parse import quote
import shutil
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from PIL import Image, ImageDraw, ImageFont

from ..common import read, save, sha, now, overlay, valid_box, RequestTimeout
from .local import prepare_candidate, observe_and_fit
from .observation import sanitize, sanitize_feedback, source_clipping
from .contracts import protocol_error
from .critic import check_object, revision_feedback
from .paths import sample_commands, command_strokes
from .views import REVIEW_PALETTE as PALETTE, FONT, inspection_style
from .seed import import_seed,checked_job
from .assets import render_asset_source
from .coordinates import coordinate_mode, model_size, to_pixels, NORM1000_BOX_CONVENTION

SYSTEM = 'Inspect figure geometry using the supplied images. Return only JSON matching the schema. Distinguish source geometry from annotations. Preserve uncertainty.'


class ObjectOwnershipError(RuntimeError):
    """A competing worker must not write receipts or errors into this object."""


def needs_more_source(bounds, job):
    """A source-image edge cannot be repaired by requesting a larger crop."""
    if not bounds:return False
    w,h=job['frame']['actual_crop_input_size_px'];eps=max(2.,.005*min(w,h))
    fixed=set((source_clipping(job) or {}).get('image_edges',[]))
    touches=(bounds[0]<=eps,bounds[1]<=eps,bounds[2]>=w-1-eps,bounds[3]>=h-1-eps)
    return any(at_edge and name not in fixed for name,at_edge in zip(('left','top','right','bottom'),touches))


def withheld_source_context(job):
    """A partial fit may omit the very member that crosses its crop edge."""
    support=job.get('sam_support') or {}
    bounds=support.get('context_bbox_original')
    if not support.get('context_expansion_withheld') or not bounds:return False
    crop=job['frame']['source_crop_half_open'];size=job['frame']['original_image_size_px']
    return any((bounds[i]<crop[i] and crop[i]>0) or
               (bounds[i+2]>crop[i+2] and crop[i+2]<size[i]) for i in (0,1))


def schema(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def validate(raw, spec):
    try:
        Draft202012Validator(spec).validate(raw)
    except ValidationError as exc:
        raise ValueError(protocol_error(exc)) from exc
    return raw


def safe_id(value):
    if (not isinstance(value, str) or not value or value in {'.', '..'}
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise ValueError('Unsafe case/object ID')
    return quote(value, safe='-_.~')


def prepared(source, candidate, item, folder, options, crop=None, feedback=(), input_context=None, spatial_evidence=None, revision_context=None):
    """Atomic image preparation; interrupted staging directories are disposable."""
    input_context = input_context or {}
    mode = coordinate_mode(options)
    descriptions = input_context.get('identity_descriptions', [])
    if (folder/'job.json').exists():
        job = read(folder/'job.json')
        if revision_context is not None:
            from .revision import reframe_context
            revision_context=reframe_context(revision_context,job['frame'])
        if coordinate_mode(job) != mode:
            raise ValueError('Prepared coordinate mode changed')
        if (job['candidate'] != candidate or job['identity_descriptions'] != descriptions
                or job.get('input_context', {}) != input_context or job.get('revision_feedback', []) != list(feedback)
                or job.get('revision_context') != revision_context):
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
            coordinate_mode=mode, spatial_evidence=spatial_evidence, revision_context=revision_context)
    stage.rename(folder)
    return job


def inspection_images(folder, *, show_samples=True):
    local = read(folder/'fit_local.json')
    with Image.open(folder/'clean.png') as im:
        fit_view = im.convert('RGB'); points_view = im.convert('RGB')
    frame = read(folder/'frame.json') if (folder/'frame.json').exists() else local['frame']
    style = inspection_style(fit_view.size, frame.get('sizing', {}))
    radius = style['point_radius_px']
    fd, pd = ImageDraw.Draw(fit_view), ImageDraw.Draw(points_view)
    font = ImageFont.truetype(FONT, max(12, fit_view.width//90))
    table = {p['id']: p['xy'] for p in local['observation']['points']}
    labels = []
    for i, part in enumerate(local['parts']):
        color = PALETTE[i % len(PALETTE)]
        if part['status'] == 'ok':
            coords = [tuple(p) for p in sample_commands(part['commands'], 64)]
            if len(coords) > 1:
                for stroke in command_strokes(part['commands'], 64):
                    if len(stroke) > 1:
                        fd.line([tuple(p) for p in stroke], fill=color, width=style['line_width_px'])
                # Shared vertices are common in projected faces. Labels at
                # those vertices overwrite each other and misidentify faces.
                refs = list(dict.fromkeys(part['observed_point_ids']))
                x = sum(table[pid][0] for pid in refs)/len(refs)
                y = sum(table[pid][1] for pid in refs)/len(refs)
                labels.append(((max(0, min(x, fit_view.width-45)), max(0, min(y, fit_view.height-20))),
                               f'F{i+1}', color))
        for pid in part['observed_point_ids']:
            x, y = table[pid]
            box = (x-radius, y-radius, x+radius, y+radius)
            pd.ellipse(box, fill=color)
            if show_samples:
                fd.ellipse(box, outline=color, width=style['point_outline_width_px'])
    for position, label, color in labels:
        fd.text(position, label, fill=color, font=font, stroke_width=1, stroke_fill='white')
    fit_view.save(folder/'fit_overlay.png'); points_view.save(folder/'observed_points.png')
    from .views import geometry_only
    geometry_only(local,fit_view.size,folder/'fit_geometry.png',style['line_width_px'])
    save(folder/'inspection.json', {**style, 'sample_points_shown': show_samples})
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
    else:
        prompt['coordinate_bounds']={'convention':'inclusive pixel centers',
            'x':[0,size[0]-1],'y':[0,size[1]-1],
            'rule':'Image width/height are not valid endpoints. Use these inclusive maxima at image edges.'}
    def check(raw):
        validate(raw, spec)
        if not valid_box(raw['bbox_2d'], model_size(size, mode)):
            limits=model_size(size,mode)
            raise ValueError(f'Invalid proposed context crop {raw["bbox_2d"]}: require xmin<xmax, ymin<ymax; '
                             f'inclusive x=0..{limits[0]-1}, y=0..{limits[1]-1}. '
                             'Width/height are not valid endpoints; correct the out-of-range coordinate.')
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


def context_crop(candidate,item,source,input_path,folder,runtime,options,reason,job):
    crop=recrop(candidate,item,source,input_path,folder,runtime,options,reason)
    if options.get('recover_clipped_partial'):
        previous=job['frame']['source_crop_half_open']
        margin=.25*max(previous[2]-previous[0],previous[3]-previous[1]);sw,sh=item['source_size']
        crop=[max(0,min(crop[0],math.floor(previous[0]-margin))),
              max(0,min(crop[1],math.floor(previous[1]-margin))),
              min(sw,max(crop[2],math.ceil(previous[2]+margin))),
              min(sh,max(crop[3],math.ceil(previous[3]+margin)))]
    return crop


def process_object(root, candidate, item, workspace, runtime, options, *, namespace, prepare_only=False, input_context=None, seed=None, image_assets=(), review_seed=None):
    """One writer owns an object's revisions, including resume and receipts."""
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.object.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ObjectOwnershipError('Another worker owns fitting object: '+str(root)) from exc
        return _process_object(root, candidate, item, workspace, runtime, options,
            namespace=namespace, prepare_only=prepare_only, input_context=input_context,
            seed=seed, image_assets=image_assets, review_seed=review_seed)


def _process_object(root, candidate, item, workspace, runtime, options, *, namespace, prepare_only=False, input_context=None, seed=None, image_assets=(), review_seed=None):
    root.mkdir(parents=True, exist_ok=True)
    receipt = root/'done.json'
    if receipt.exists():
        data = read(receipt)
        if data['record'].get('candidate') != candidate:
            raise ValueError('Completed fitting target identity changed; use a new object checkpoint')
        for name, digest in data['sha256'].items():
            if sha(root/name) != digest:
                raise ValueError('Completed fitting artifact changed: '+name)
        if data['record'].get('stop_reason') not in {'observation_transport_failed','check_transport_failed'}:
            return data['record']
        # Only an explicit resume retries a transient service failure. Keep
        # validated successful calls; no extra geometry/schema retry is spent.
        receipt.unlink()
    source, input_path = workspace/item['original'], workspace/item['input']
    save(root/'image_assets.json', [{'id': asset['id'], 'bbox_2d': asset['bbox_2d']} for asset in image_assets])
    input_context = input_context or {}
    if seed and review_seed:
        from .seed import reviewed_addition_seed
        seed=reviewed_addition_seed(seed,review_seed)
    expanded_source=input_context.get('coverage_source_extent_original') is not None
    expanded_edit=None
    if seed and expanded_source:
        # A coverage scope can already fit inside the available clean crop.
        # Keep its verified geometry and the ordinary partial-edit path.
        old_crop=read(Path(seed['path'])/'job.json')['frame']['source_crop_half_open']
        extent=input_context['coverage_source_extent_original']
        expanded_source=any(extent[i]<old_crop[i] or extent[i+2]>old_crop[i+2]-1 for i in (0,1))
        if not expanded_source:
            save(root/'source_scope_reuse.json',{'requested_extent_original':extent,
                'existing_source_crop_half_open':old_crop,'geometry_reused':bool(seed.get('fit_reused')),
                'reason':'complete_requested_source_scope_already_in_existing_crop'})
    if seed and expanded_source:
        # Local-coordinate geometry cannot be reused after the observation
        # frame changes. Observe the complete source extent in a NEW frame;
        # retain the immutable old fit only as an explicitly unresolved fallback.
        checked_job(seed,candidate,item)
        if (seed.get('fit_reused') and seed.get('source_state')=='accepted_by_model'
                and review_seed and review_seed.get('decision')=='revise'):
            from .revision import plan_revision
            expanded_edit=plan_revision(Path(seed['path']),review_seed)
        seed={**seed,'fit_reused':False}
        save(root/'source_scope_seed.json',seed)
    seed_job = import_seed(seed, root/'revision_00', candidate, item, input_context=input_context) if seed and not expanded_source else None
    save(root/'input_context.json', input_context)
    observer = runtime.for_stage(shape_schema_retries=options.get('protocol_retries', 1)) if runtime is not None else None
    history, feedback, crop, revision_context = [], [], None, None
    if expanded_edit is not None:
        revision_context=expanded_edit
        feedback=revision_feedback(review_seed)
        original_scope=read(Path(seed['path'])/'job.json')['frame']['source_crop_half_open']
        extent=input_context['coverage_source_extent_original']
        crop=[min(original_scope[0],math.floor(extent[0])),min(original_scope[1],math.floor(extent[1])),
              max(original_scope[2],math.ceil(extent[2])+1),max(original_scope[3],math.ceil(extent[3])+1)]
    if seed_job:
        feedback = seed_job.get('revision_feedback', [])
        crop = seed_job['frame']['source_crop_half_open']
    selected = None; corrections = 0; contexts = 0; sam_checked = False
    evidence = getattr(runtime, 'spatial_evidence', None)
    guard_enabled = options.get('geometry_guard_enabled', evidence is not None)
    if expanded_edit is not None:
        # Keep an immutable usable fallback before requesting the bounded
        # patch. Failure in the larger crop cannot erase accepted source parts.
        fallback=root/'source_fallback';fallback.mkdir(exist_ok=True)
        for name in ('fit_local.json','fit_original.json','observation.json','job.json','clean.png','status.json'):
            source_file=Path(seed['path'])/name
            if source_file.exists():shutil.copy2(source_file,fallback/name)
        save(fallback/'reused_revision.json',{**seed,'fit_reused':True,'fallback_only':True,
             'reason':'immutable_source_fallback_for_expanded_bounded_edit'})
        selected=fallback.name
    seen_geometry = {}
    record = {'object_id': candidate['id'], 'state': 'pending', 'selected_revision': None,
              'visual_status': 'unreviewed', 'history': history, 'candidate': candidate}
    for revision in range(1+options['max_revisions']+options['max_context_rounds']):
        folder = root/f'revision_{revision:02d}'
        job = seed_job if revision == 0 and seed_job else prepared(source, candidate, item, folder, options, crop, feedback, input_context,
            spatial_evidence=getattr(runtime, 'spatial_evidence', None), revision_context=revision_context)
        if prepare_only:
            save(root/'record.json', record)
            return record
        try:
            reused = revision == 0 and seed is not None and seed['fit_reused']
            status = read(folder/'status.json') if reused else observe_and_fit(job, folder, observer,
                namespace=namespace+f'/revision_{revision:02d}', max_tokens=options['max_tokens'])
            if reused and not review_seed and options.get('source_frame_refinement',True):
                from .frame_edges import refine
                from .adapter import transform_fit
                fitted=read(folder/'fit_local.json')
                with Image.open(folder/'clean.png') as clean:
                    measured=refine(fitted,job,clean,evidence)
                    from .rounded_corners import refine as refine_radius
                    radii = refine_radius(fitted, job, clean.convert('RGB'), evidence)
                save(folder/'frame_edge_refinement.json',measured)
                save(folder/'radius_refinement.json', radii)
                if any(row['applied'] for row in measured + radii):
                    save(folder/'fit_local.json',fitted)
                    save(folder/'fit_original.json',transform_fit(fitted))
                    status['source_refined_from_seed']=True
                    save(folder/'status.json',status)
            entry = {'revision': revision, 'directory': folder.name, 'observation': status}
            if reused:
                entry['observation_reused_from'] = seed['path']
            history.append(entry)
            if status.get('failed_replacements'):
                record.update(state='unresolved', stop_reason='revision_fit_failed',
                    reason='Previous geometry retained; replacement fitting failed for: ' +
                           ', '.join(status['failed_replacements']))
                break
            if guard_enabled and status['status'] == 'finished' and status.get('part_counts', {}).get('succeeded', 0):
                from .geometry_guard import compare_revision, geometry_signature, VERSION
                fit = read(folder/'fit_original.json')
                signature = geometry_signature(fit)
                complete = fit['fit_status'] == 'ok' and fit['observation_status'] == 'ok'
                guard = {'version': VERSION, 'decision': 'abstain', 'reason': 'initial_geometry'}
                if complete and signature in seen_geometry:
                    guard.update(decision='stop', reason='repeated_geometry',
                                 retained_revision=seen_geometry[signature])
                elif selected:
                    guard = compare_revision(read(root/selected/'fit_original.json'), fit,
                                             source, evidence, candidate)
                    guard['retained_revision'] = selected
                    # An abstaining numeric guard leaves the decision to the
                    # existing post-edit critic, which already sees BEFORE,
                    # AFTER, CLEAN and SAM. No independent vetoing agent.
                save(folder/'geometry_guard.json', guard)
                entry['geometry_guard'] = guard
                if guard['decision'] in {'stop', 'rollback'}:
                    previous_review=history[-2].get('review',{}) if len(history)>1 else {}
                    absent_source=any(r.get('visible_line')=='no' for r in
                                      previous_review.get('independent_source_observations',[]))
                    if (guard['reason']=='repeated_geometry' and absent_source
                            and contexts<options['max_context_rounds']):
                        # The old local crop just reproduced a source-disproved
                        # closure. Spend the existing context allowance once;
                        # do not certify it or start another geometry loop.
                        feedback=revision_feedback(previous_review)
                        reason=('The local edit reproduced a contour independently contradicted by CLEAN. '
                                'Inspect the full image and include the COMPLETE connected source outline; '
                                'the localization box and shape kind are fallible. '+' '.join(feedback))
                        next_crop=context_crop(candidate,item,source,input_path,root/f'context_{contexts:02d}',
                                               runtime,options,reason,job)
                        if next_crop!=job['frame']['source_crop_half_open']:
                            guard.update(decision='recrop',reason='repeated_source_conflict',retained_revision=selected)
                            save(folder/'geometry_guard.json',guard)
                            contexts+=1;crop=next_crop;revision_context=None
                            continue
                    selected = guard['retained_revision']
                    record.update(state='unresolved', stop_reason='geometry_stalled' if guard['decision'] == 'stop'
                                  else 'geometry_regression_blocked',
                                  reason='Previous geometry retained: '+guard['reason'])
                    break  # No extra critic or observation can override this veto.
                if complete:
                    seen_geometry[signature] = folder.name
            if (status.get('part_counts', {}).get('succeeded', 0)
                    or status.get('revision_merge', {}).get('removed_ids') and status['status'] == 'finished'):
                # Keep the first fit as an explicitly unverified baseline.
                # A replacement must pass the existing critic before commit.
                if selected is None or not options['check_enabled']:
                    selected = folder.name
                else:
                    record['unaccepted_revision'] = folder.name
            if status['status'] != 'finished':
                failure=status.get('failure_kind')
                record.update(state='unresolved', stop_reason='observation_timeout' if failure=='request_timeout' else
                    'observation_transport_failed' if failure=='model_transport' else
                    'observation_refused' if failure=='model_refused' else 'observation_protocol_exhausted',
                    reason=status.get('error', 'Observation failed'))
                break
            action = 'revise'; reason = 'Reobserve the whole target and return complete geometry.'
            next_feedback = []; review = None
            if status['status'] == 'finished':
                action = 'needs_context' if status['observation_status'] == 'needs_context' else 'revise'
                reason = status.get('reason', '') or 'Review all parts and their point references from the clean image.'
                removal = status.get('revision_merge', {})
                if (status['observation_status'] == 'unsupported' and removal.get('removed_ids')
                        and not removal.get('preserved_ids') and not removal.get('unrepaired_ids')
                        and not status.get('part_counts', {}).get('observed')):
                    # An explicit removal-only edit is a completed decision.
                    # Falling back to the old usable fit resurrects the rejected object.
                    selected = None
                    record.pop('unaccepted_revision', None)
                    record.update(state='accepted_by_model', visual_status='removed_by_model',
                                  removed_part_ids=removal['removed_ids'], reason=reason)
                    break
                if status['observation_status'] == 'unsupported':
                    record.update(state='unsupported', reason=reason); break
                if status.get('part_counts', {}).get('failed') and not status['part_counts'].get('succeeded'):
                    failed=read(folder/'fit_local.json')['parts']
                    reason='Numerical fitting failed: '+ '; '.join(
                        f"{p['id']} ({p['kind']}): {p.get('error', 'invalid geometry')}" for p in failed)
                    reason+=' Reobserve these contours from CLEAN using the appropriate primitive or path.'
                clipped = False
                if options.get('recover_clipped_partial') and status['observation_status'] == 'partial':
                    fit = read(folder/'fit_local.json')
                    clipped = needs_more_source(fit.get('bbox_2d'), job) or withheld_source_context(job)
                if clipped or status['observation_status']=='needs_context':
                    action = 'needs_context'
                    reason = status.get('reason') or 'Incomplete target touches the crop boundary; inspect the complete target.'
                    next_feedback = [sanitize_feedback(reason)]
                elif status.get('part_counts', {}).get('succeeded', 0) and options['check_enabled']:
                    evidence = getattr(runtime, 'spatial_evidence', None)
                    inspection_images(folder, show_samples=not getattr(evidence, 'clean_fit_view', False))
                    sam_review = None
                    supplied_review = review_seed if revision == 0 and reused else None
                    if not supplied_review and not sam_checked and getattr(evidence, 'precheck_enabled', False):
                        sam_checked = True
                        try:
                            sam_review = check_object(job, folder, runtime, options, sam_first=True)
                            if sam_review is not None:
                                save(folder/'sam_review.json', sam_review)
                                entry['sam_review'] = sam_review
                        except (RuntimeError, ValueError) as exc:
                            entry['sam_check_error'] = str(exc)
                    try:
                        review = supplied_review or (check_object(job, folder, runtime, options, preceding_review=sam_review)
                                  if sam_review is not None else check_object(job, folder, runtime, options))
                    except (RuntimeError, ValueError) as exc:
                        entry['check_error'] = str(exc)
                        record.update(state='unresolved', stop_reason='check_timeout' if isinstance(exc, RequestTimeout) else
                            'check_transport_failed' if type(exc).__name__=='ModelTransportError' else
                            'check_refused' if type(exc).__name__=='ModelRefusedError' else 'check_protocol_exhausted',error=str(exc))
                        break
                    save(folder/'review.json', review); entry['review'] = review
                    action, reason = review['decision'], review['reason']
                    next_feedback = revision_feedback(review)
                    if action == 'unresolved':
                        record.update(state='unresolved', stop_reason='critic_uncertain', reason=reason)
                        break
                    if (action == 'revise' and job.get('revision_context')
                            and status['fit_status'] == 'ok' and status['observation_status'] == 'ok'
                            and review.get('revision_effect', {}).get('outcome') == 'improved'
                            and job['revision_context']['source_revision'] == str(root/selected)):
                        # The same BEFORE/AFTER critic confirms a correction
                        # without new defects. Keep it as an unresolved baseline;
                        # an old residual defect must not undo the corrected edge.
                        selected = folder.name
                        record.pop('unaccepted_revision', None)
                        record.setdefault('retained_improvements', []).append({
                            'revision': folder.name, 'evidence': review['revision_effect']['evidence']})
                    if action == 'accept' and status['fit_status'] == 'ok' and status['observation_status'] == 'ok':
                        selected = folder.name
                        record.pop('unaccepted_revision', None)
                        record.update(state='accepted_by_model', visual_status='accepted_by_model', reason=reason); break
                    if action == 'accept':
                        if not status.get('part_counts', {}).get('failed', 0):
                            # A historical partial flag is not an actionable edit.
                            # Preserve the warning without contradicting an empty
                            # visual accept by asking another agent to redraw all.
                            record.update(state='unresolved', stop_reason='partial_without_actionable_revision',
                                          visual_status='critic_accept_incomplete',
                                          reason=status.get('reason') or 'Partial geometry retained; critic supplied no repair target.')
                            break
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
                next_crop = context_crop(candidate,item,source,input_path,root/f'context_{contexts:02d}',runtime,options,
                                         '\n'.join(next_feedback) if next_feedback else reason,job)
                if next_crop == job['frame']['source_crop_half_open']:
                    record['stop_reason'] = 'repeated_context'; break
                contexts += 1; crop = next_crop
                revision_context = None
            else:
                if corrections >= options['max_revisions']:
                    record['stop_reason'] = 'revision_budget_exhausted'; break
                corrections += 1
                from .revision import plan_revision
                revision_context = plan_revision(folder, review)
                crop = list(job['frame']['source_crop_half_open'])
            feedback = next_feedback or [sanitize_feedback(reason)]
        except Exception as exc:
            record.update(state='unresolved', stop_reason='request_timeout' if isinstance(exc, RequestTimeout) else 'execution_error', error=str(exc)); break
        finally:
            record['selected_revision'] = selected
            save(root/'record.json', record)
    if selected and record.get('stop_reason')=='revision_budget_exhausted' and record.get('unaccepted_revision'):
        # A critic finding on one new part must not erase unrelated additions
        # when every previous native contour was preserved exactly.
        from .partial_additions import select as select_additions
        from .adapter import transform_fit
        attempted=root/record['unaccepted_revision']
        reviewed=read(attempted/'review.json') if (attempted/'review.json').exists() else {}
        attempted_job=read(attempted/'job.json')
        partial=select_additions(read(root/selected/'fit_local.json'),read(attempted/'fit_local.json'),
                                 reviewed,attempted_job.get('revision_context'))
        if partial:
            fitted,audit=partial;target=root/'partial_additions';target.mkdir(exist_ok=True)
            save(target/'fit_local.json',fitted);save(target/'fit_original.json',transform_fit(fitted))
            for name in attempted_job['input_sha256']:
                destination=target/name;destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(attempted/name,destination)
            save(target/'job.json',attempted_job);save(target/'observation.json',fitted['observation'])
            save(target/'status.json',{'status':'finished','observation_status':'partial','fit_status':'partial',
                                      'part_counts':fitted['part_counts'],'partial_additions':audit})
            save(target/'selection.json',{**audit,'baseline_revision':selected,'attempted_revision':record['unaccepted_revision']})
            record.update(partial_additions_commit=audit,selection_reason='retain_reviewed_additions_preserve_old_geometry')
            selected=target.name
    from .seed import reviewed_coverage_recovery
    recovered=reviewed_coverage_recovery(history,selected)
    if recovered:
        record['coverage_recovery_retained']=selected
    if (seed and not seed['fit_reused'] and seed.get('fallback_fit_available') and not recovered
            and record['state'] != 'accepted_by_model'):
        # Fresh observation must be free to change the decomposition. Its
        # failed attempt must not erase the previous usable geometry, however.
        import_seed({**seed, 'fit_reused': True}, root/'revision_baseline', candidate, item)
        record.update(attempted_revision=selected, state='unresolved',
                      selection_reason='fresh_observation_not_accepted_preserve_previous_geometry')
        selected = 'revision_baseline'
    if selected and options.get('source_frame_refinement', True):
        from .source_boxes import register
        from .adapter import transform_fit
        previous = root / selected
        fitted = read(previous / 'fit_local.json'); job = read(previous / 'job.json')
        from .frames import FitFrame
        frame=FitFrame.from_metadata(job['frame'])
        owned_regions=[frame.to_local(a['bbox_2d'][:2])+frame.to_local(a['bbox_2d'][2:])for a in image_assets]
        with Image.open(previous / 'clean.png') as clean:
            registration = register(fitted, job, clean,owned_regions=owned_regions)
        save(root / 'source_registration.json', registration)
        conflicts=[r for r in registration if r.get('reason')in ('compact_source_count_conflict','source_primitive_count_conflict')
            and not r.get('missing_members_recovered')]
        if conflicts:
            # The existing coverage review owns missing content. Preserve the
            # measured disagreement there without inventing another member or
            # launching a second reviewer inside numerical registration.
            record['source_registration_conflicts']=conflicts
        else:
            record.pop('source_registration_conflicts',None)
        geometry_applied=any(r.get('applied') for r in registration)
        metadata_parts=[r['part_id']for r in registration if r.get('metadata_only')]
        if geometry_applied or metadata_parts:
            registered = root / 'source_registered'
            if not registered.exists():
                shutil.copytree(previous, registered)
            save(registered / 'fit_before_registration.json', read(previous / 'fit_original.json'))
            save(registered / 'fit_local.json', fitted)
            save(registered / 'fit_original.json', transform_fit(fitted))
            save(registered / 'observation.json',fitted['observation'])
            status_path=registered/'status.json'
            if status_path.exists():
                source_status=read(status_path)
                source_status.update(part_counts=fitted['part_counts'])
                if geometry_applied:source_status['source_primitive_registration']=True
                if metadata_parts:source_status['declared_ownership_repaired']=True
                save(status_path,source_status)
            save(registered / 'source_registration.json', registration)
            record['source_registration'] = {'source_revision': selected,
                'applied_parts': [r['part_id'] for r in registration if r.get('applied')],
                'evidence': ('one_to_one_source_identity_and_four_stroke_sides'if geometry_applied else
                             'confirmed_source_family_role_metadata'), 'model_acceptance_unchanged': True}
            if metadata_parts:record['source_registration']['metadata_only_parts']=metadata_parts
            selected = 'source_registered'
    record.update(selected_revision=selected, finished_utc=now(), context_rounds=contexts, corrections=corrections)
    save(root/'record.json', record)
    hashes = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'done.json'}
    save(receipt, {'record': record, 'sha256': hashes})
    return record
