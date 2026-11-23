"""Point-first visual review: minimal acceptance, image-grounded corrections."""
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from ..common import read, save
from .contracts import protocol_error
from .observation import sanitize, sanitize_feedback
from .views import PALETTE

SYSTEM = ('Compare the clean source with the model sample points and fitted paths. '
          'Check geometry, not rendering style. Return one short JSON object. '
          'Every reported defect and correction must be grounded in the supplied images; '
          'never invent a mismatch or a hidden object. Correct results need no written report.')
ACTIONABLE = {'missing', 'extra', 'geometry', 'topology', 'occlusion'}
KINDS = sorted(ACTIONABLE | {'ownership', 'context', 'ambiguity', 'wrong_target'})


def _object(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def review_schema(part_ids=None):
    text = {'type': 'string', 'minLength': 1}
    ident = {**text, **({'enum': list(part_ids)} if part_ids is not None else {})}
    issue = _object({'kind': {'enum': KINDS},
                     'parts': {'type': 'array', 'items': ident, 'uniqueItems': True},
                     'evidence': {**text, 'description': 'Locate the visible source feature and its actual point/path mismatch; for hidden geometry state the visible constraints.'},
                     'action': {**text, 'description': 'Specific correction supported by that evidence, or exactly what evidence is needed.'}})
    return _object({'decision': {'enum': ['accept', 'revise', 'needs_context', 'reject', 'unresolved']},
                    'issues': {'type': 'array', 'items': issue}})


def validate_review(raw, part_ids):
    try:
        Draft202012Validator(review_schema(part_ids)).validate(raw)
    except ValidationError as exc:
        raise ValueError(protocol_error(exc)) from exc
    kinds = {i['kind'] for i in raw['issues']}
    for index, issue in enumerate(raw['issues']):
        if not issue['parts'] and issue['kind'] in {'extra', 'geometry', 'topology', 'occlusion'}:
            raise ValueError(f'issues.{index}.parts: identify affected F labels; missing/context/ownership/ambiguity/wrong_target may have no fitted part')
        if any(not issue[k].strip() for k in ('evidence', 'action')):
            raise ValueError(f'issues.{index}: evidence and action must contain concrete information')
    decision = raw['decision']
    if decision == 'accept' and raw['issues']:
        raise ValueError('decision: accept contradicts nonempty issues; resolve the defect or material doubt')
    if decision != 'accept' and not raw['issues']:
        raise ValueError('issues: a non-accept decision requires image evidence and a specific action')
    if decision == 'revise' and not kinds <= ACTIONABLE:
        raise ValueError('decision: revise requires actionable geometry defects; resolve material ownership/context ambiguity first')
    if decision == 'needs_context' and 'context' not in kinds:
        raise ValueError('decision: needs_context requires an actual clipping or insufficient-context issue')
    if decision == 'unresolved' and not kinds & {'ownership', 'ambiguity'}:
        raise ValueError('decision: unresolved requires a material ownership or ambiguity issue')
    if decision == 'reject' and 'wrong_target' not in kinds:
        raise ValueError('decision: reject requires evidence that this is the wrong target')
    if 'wrong_target' in kinds and decision != 'reject':
        raise ValueError('decision: a confirmed wrong_target requires reject, not geometry revision')
    return raw


def part_inventory(local):
    mapping = {f'F{i+1}': p['id'] for i, p in enumerate(local['parts'])}
    inventory = [{'id': f'F{i+1}', 'color': PALETTE[i % len(PALETTE)], 'kind': p['kind']}
                 for i, p in enumerate(local['parts'])]
    owners = {}
    for i, part in enumerate(local['parts']):
        for pid in dict.fromkeys(part['observed_point_ids']):
            owners.setdefault(pid, []).append(f'F{i+1}')
    # Each row is one actual shared vertex; repeated rows are distinct vertices.
    shared = [ids for ids in owners.values() if len(ids) > 1]
    return mapping, inventory, shared


def target_context(job):
    """Keep deduplicated target identity, without speculative neighbor ownership."""
    context = job.get('input_context', {})
    label = sanitize(job['label'])[0]
    descriptions = list(dict.fromkeys(sanitize(x)[0] for x in
        job.get('identity_descriptions', []) + context.get('identity_descriptions', []) if sanitize(x)[0]))
    target = {'label': label}
    if descriptions:
        target['description'] = [x for x in descriptions if x != label]
    return target


def review_prompt(job, local, inspection=None):
    mapping, inventory, shared = part_inventory(local)
    target = target_context(job)
    prompt = {'version': 'shape_integration.fit_check.8', 'target': target,
              'images': ['Image 1: source crop. Establish the target boundary from this image first.',
                         'Image 2: same crop with colored fitted paths, hollow sample-point markers and F part labels. '
                         'Both points and paths are model predictions to check, not ground truth. Marker centers show '
                         'sample locations; path centerlines show fitted geometry. Marker size and line width are annotations. '
                         'Paths are drawn without occlusion masking.'],
              'fitted_parts': inventory,
              'checks': [
                  'Inspect all parts and intrinsic missing parts. Compare sample positions AND fitted paths with the '
                  'clean source, not just with each other. Follow source stroke centerlines (fill edges for unstroked '
                  'objects), including the underlying continuous contour of dashed lines. Check sides, corners, junctions '
                  'and curve shape. For ellipses check contour points; for rounded rectangles pay particular attention '
                  'to the points on each corner arc and its straight-to-arc transitions. Do not accept wrong arc points '
                  'just because the fitted outline looks plausible.',
                  'For projected solids, check closed faces and shared junctions, not only the outer silhouette. '
                  'A usual three-face cuboid has seven projected vertices per solid. shared_vertices lists true shared '
                  'vertices by their F owners; repeated rows are different vertices. Separate instances do not share '
                  'vertices merely because they overlap. Preserve intrinsic cells and shared internal boundaries.',
                  'Infer COMPLETE occluded objects from visible continuation, symmetry and matching instances. '
                  'A hidden point need not lie on a visible edge, and an unmasked hidden path may cross the foreground. '
                  'Require visible evidence of the occluder before calling a discrepancy occlusion; a nearby object '
                  'does not justify extending a boundary. Ordinary hidden-pixel uncertainty alone is not a defect.',
                  'Target descriptions are fallible identity hints. Determine completeness from the source image, '
                  'not from the current fitted-part count or descriptive color words. An intrinsic enclosing frame '
                  'of a compound icon is part of that icon even when its color differs from the interior parts. '
                  'An external scene container does not own all its contents. Do not assume a missing visible part '
                  'was handled elsewhere. Exclude text and external connectors; retain intrinsic symbol strokes. Ignore small '
                  'antialiasing and annotation-style differences.',
                  'In Image 1, cyan boxes labeled IMAGE mark image assets excluded from shape fitting; the boxes '
                  'and labels are annotations, not source geometry, and overlapping target shape boundaries still require review.',
                  'Report only defects you can locate in the images. Each evidence must identify the source feature '
                  'and actual point/path discrepancy at a relative location; each action must say how to correct that '
                  'same discrepancy. Never invent a missing part, displacement, arc error or occluder to justify a '
                  'rejection. If evidence cannot decide, use unresolved with the specific doubt, not a guessed correction. '
                  'Group the same defect across parts; preserve unaffected geometry. No numeric coordinates or replacement geometry.'
              ],
              'decision_rules': 'Correct: {"decision":"accept","issues":[]}. No explanations for passing checks. '
                                'revise: image-supported error with a specific correction. needs_context: actual clipping '
                                'or insufficient image context. reject: wrong target only. unresolved: material ownership '
                                'or geometric ambiguity. Every non-accept needs evidence and action. Use only F labels in '
                                'issues.parts; use [] for a missing feature with no fitted counterpart. Return one JSON, no drafts.',
              'output_schema': review_schema(mapping)}
    if shared:
        prompt['shared_vertices'] = shared
    return prompt


def check_object(job, folder, runtime, options):
    from .critic_retry import retry_builder
    if not (folder/'inspection.json').exists() or not (folder/'critic_source.png').exists():
        from .control import inspection_images
        inspection_images(folder)
    local = read(folder/'fit_local.json')
    mapping, _, _ = part_inventory(local)
    save(folder/'critic_part_map.json', mapping)
    prompt = review_prompt(job, local)
    images = [('Source with image-asset boxes', folder/'critic_source.png'), ('Sample points and fitted paths', folder/'fit_overlay.png')]
    validator = lambda raw: validate_review(raw, mapping)
    checker = runtime.for_stage(shape_schema_retries=options.get('check_retries', 1))
    raw = checker.call(folder/'calls', 'check', SYSTEM, prompt, images, validator, options['check_max_tokens'],
                       retry_builder=retry_builder(SYSTEM, prompt, images, validator))
    # Controller summary is backend-derived; it is not an extra model claim.
    reason = 'No defect reported by critic.' if raw['decision'] == 'accept' else ' '.join(i['evidence'] for i in raw['issues'])
    return {**raw, 'reason': reason, 'reason_origin': 'backend_summary',
            'protocol_version': prompt['version'], 'part_id_map': mapping,
            'issues': [{**i, 'part_ids': [mapping[p] for p in i['parts']]} for i in raw['issues']]}


def revision_feedback(review):
    if review['decision'] == 'accept':
        return []
    if 'issues' not in review:  # Read-only compatibility for historical reports.
        lines = [review['reason']] + [f"{i['feature']}: clean source: {i['source_observation']}; previous fit: "
            f"{i['fitted_observation']}; evidence: {i['evidence']}" for i in review['checks'] if i['comparison'] != 'aligned']
    else:
        lines = ['Address the image-supported issues below; preserve unaffected geometry and supported shared vertices.']
        for issue in review['issues']:
            owners = ', '.join(issue.get('part_ids', issue['parts'])) or 'no existing fitted counterpart'
            evidence = issue.get('evidence') or f"source: {issue['source']}; previous fit: {issue['fit']}"
            lines.append(f"{issue['kind']} ({owners}): evidence: {evidence}; action: {issue['action']}")
    return [value for line in lines if (value := sanitize_feedback(line))]
