"""Point-first visual review: minimal acceptance, image-grounded corrections."""
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from ..common import read, save
from ..scope import COMPOUND_OWNERSHIP, ARROW_OWNERSHIP, PUNCTUATION_OWNERSHIP
from .contracts import protocol_error
from .observation import sanitize, sanitize_feedback, source_clipping, IDENTITY_NOTE_SCOPE
from .views import REVIEW_PALETTE as PALETTE
from .semantics import CONNECTION_EVIDENCE

SYSTEM = ('Compare the clean source with the model sample points and fitted paths. '
          'Check contours and topology; fill, stroke width and dash pattern belong to the later styling stage. '
          'Return one short JSON object. '
          'Every reported defect and correction must be grounded in the supplied images; '
          'never invent a mismatch or a hidden object. Correct results need no written report.')
ACTIONABLE = {'missing', 'extra', 'geometry', 'topology', 'occlusion'}
KINDS = sorted(ACTIONABLE | {'ownership', 'context', 'ambiguity', 'wrong_target'})


def _object(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def review_schema(part_ids=None, observation_ids=(), *, revision=False):
    text = {'type': 'string', 'minLength': 1}
    ident = {**text, **({'enum': list(part_ids)} if part_ids is not None else {})}
    issue = _object({'kind': {'enum': KINDS,
                            'description': 'extra removes an ENTIRE unsupported part. Use geometry/topology to repair segments of a source-supported part.'},
                     'parts': {'type': 'array', 'items': ident, 'uniqueItems': True},
                     'evidence': {**text, 'description': 'Locate the visible source feature and its actual point/path mismatch; for hidden geometry state the visible constraints.'},
                     'action': {**text, 'description': 'Specific correction supported by that evidence, or exactly what evidence is needed.'}})
    properties={'decision': {'enum': ['accept', 'revise', 'needs_context', 'reject', 'unresolved']},
                'issues': {'type': 'array', 'items': issue}}
    if revision:
        properties['revision_effect'] = _object({
            'outcome': {'enum': ['improved', 'unchanged', 'regressed', 'uncertain']},
            'evidence': text})
    if observation_ids:
        relations=['supported','occluded','missing','uncertain']
        if any(ident.startswith('coverage_request_') for ident in observation_ids):relations.append('not_present')
        properties['source_observations']={'type':'array','minItems':len(observation_ids),'maxItems':len(observation_ids),
            'items':_object({'id':{'enum':list(observation_ids)},
                            'source_relation':{'enum':relations},
                            'evidence':text})}
    return _object(properties)


def validate_review(raw, part_ids, observation_ids=(), *, revision=False):
    try:
        Draft202012Validator(review_schema(part_ids,observation_ids,revision=revision)).validate(raw)
    except ValidationError as exc:
        raise ValueError(protocol_error(exc)) from exc
    kinds = {i['kind'] for i in raw['issues']}
    for index, issue in enumerate(raw['issues']):
        if issue['kind']=='extra' and len(issue['parts'])!=1:
            raise ValueError(f'issues.{index}.parts: extra removes exactly one whole part. Identify that unsupported F part; for a family count/layout mismatch use topology and preserve supported members.')
        if not issue['parts'] and issue['kind'] in {'extra', 'geometry', 'topology', 'occlusion'}:
            raise ValueError(f'issues.{index}.parts: identify affected F labels; missing/context/ownership/ambiguity/wrong_target may have no fitted part')
        if any(not issue[k].strip() for k in ('evidence', 'action')):
            raise ValueError(f'issues.{index}: evidence and action must contain concrete information')
    decision = raw['decision']
    if observation_ids:
        observed=raw['source_observations']
        if {r['id'] for r in observed}!=set(observation_ids) or len({r['id'] for r in observed})!=len(observed):
            raise ValueError('source_observations: resolve each flagged source observation exactly once')
        if any(not r['evidence'].strip() for r in observed):raise ValueError('source_observations: concrete source evidence is required')
        if any(r['source_relation']=='not_present' and not r['id'].startswith('coverage_request_') for r in observed):
            raise ValueError('source_observations: not_present only dismisses a disproven coverage request')
        if decision=='accept' and any(r['source_relation'] in ('missing','uncertain') for r in observed):
            raise ValueError('decision: accept contradicts a missing or uncertain source observation')
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


def source_observation_contract(prompt,mapping):
    targets=[]
    for i,request in enumerate(prompt.get('coverage_requested_corrections',{}).get('requests',[])):
        targets.append({'id':f'coverage_request_{i}','parts':list(mapping),
                        'kind':'coverage_requested_source_content','request':request})
    if prompt.get('repeated_structure_check'):
        targets.append({'id':'repeated_structure','parts':list(mapping),
                        'kind':'source_repetition_count_and_layout'})
    if prompt.get('network_connections'):
        targets.append({'id':'connectivity','parts':list(mapping),
                        'kind':'source_connections_not_just_edge_count'})
    if prompt.get('sam_detector_mask_extent_conflict'):
        targets.append({'id':'extent_conflict','parts':list(mapping),
                        'kind':'complete_source_contour_vs_detector_box'})
    if sides:=prompt.get('extent_comparison',{}).get('required_source_sides'):
        targets.append({'id':'matched_extent','parts':list(mapping),'sides':sides,
            'kind':'matched_primitive_extent_disagreement',
            'rule':'Inspect these sides in CLEAN. An internal box edge cannot replace the enclosing '
                   'container boundary. Explain whether the fitted contour follows the actual source '
                   'border, or whether the measurement instead includes a separate object/shadow. '
                   'A measured extent is a fallible locator, never replacement coordinates.'})
    for i,row in enumerate(prompt.get('projected_face_conflicts',{}).get('overlapping_interiors',[])):
        targets.append({'id':f'face_overlap_{i}','parts':row['parts'],'kind':'overlapping_same_solid_faces'})
    for row in prompt.get('source_span_checks',{}).get('spans',[]):
        targets.append({'id':'span_'+row['span'],'parts':[row['part']],'kind':'weak_source_span'})
    ids=[r['id'] for r in targets]
    if ids:
        prompt['version']='shape_integration.fit_check.10'
        prompt['required_source_observations']={'targets':targets,
            'rule':'Resolve EVERY flagged contradiction explicitly against CLEAN, including the enlarged '
                   'source spans when provided. supported: source geometry is visibly present despite '
                   'the numeric flag; identify that source contour. occluded: name the actual visible '
                   'occluder and source-supported continuation; a wholly invented hidden face does not '
                   'qualify. missing: the fitted geometry lacks source support and needs correction. '
                   'uncertain: source evidence cannot decide. An empty accept is incomplete here; '
                   'missing/uncertain cannot accompany accept. Preserve genuine source-supported nesting.'}
        if prompt.get('coverage_requested_corrections'):
            prompt['required_source_observations']['rule'] += (
                ' For coverage_request targets, supported means the requested source content '
                'is actually represented by this fit or a verified independent reconstruction; '
                'merely locating it in CLEAN is insufficient. missing means it remains absent '
                'or incorrectly excluded. A label or containing peer is not a reconstruction witness.')
            prompt['required_source_observations']['rule'] += (
                ' If CLEAN disproves the coverage request itself, return not_present for that '
                'coverage_request ID and describe the actual source evidence. This dismisses a '
                'false coverage alarm; it never authorizes deleting a fitted part. Other flagged '
                'contradictions cannot use not_present.')
        prompt['output_schema']=review_schema(mapping,ids)
    return ids


def part_inventory(local):
    mapping = {f'F{i+1}': p['id'] for i, p in enumerate(local['parts'])}
    inventory = [{'id': f'F{i+1}', 'color': PALETTE[i % len(PALETTE)], 'kind': p['kind'],
                  'closed':bool(p.get('commands') and p['commands'][-1][0]=='Z')}
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
        target['description_scope'] = IDENTITY_NOTE_SCOPE
    return target


def review_prompt(job, local, inspection=None):
    from .semantics import CONTAINER_CONTOUR
    mapping, inventory, shared = part_inventory(local)
    target = target_context(job)
    declared=local.get('declared_ownership') or {}
    prompt = {'version': 'shape_integration.fit_check.8', 'target': target,
              'images': ['Image 1: source crop. Establish the target boundary from this image first.',
                         'Image 2: same crop with colored fitted paths, hollow sample-point markers and F part labels. '
                         'Both points and paths are model predictions to check, not ground truth. Marker centers show '
                         'sample locations; path centerlines show fitted geometry. Marker size and line width are annotations. '
                         'Paths are drawn without occlusion masking.'],
              'fitted_parts': inventory,
              'ownership_scope': COMPOUND_OWNERSHIP,
              'arrow_ownership': ARROW_OWNERSHIP,
              'punctuation_ownership': PUNCTUATION_OWNERSHIP,
              'checks': [
                  '先看原图，再逐一比较拟合轮廓的位置、宽高、弯曲、转角和间距。'
                  '沿描边中心线或无描边色块边界判断；不能把检测框、标注或目标名称当轮廓。',
                  CONTAINER_CONTOUR + '检查固有外框与内部部件是否缺失或重复；数量一致不能证明正确。'
                  '仅按同一peer或IMAGE实际接管的内容排除重复，不因定位类型或外框存在而忽略固有成员。'
                  '分别核对各色块的实际轮廓、负空间、连接端点及共享顶点，不用常见网络模板补全。',
                  '遮挡面按可见延续补齐，不画白色遮罩替代几何。'
                  '真实遮挡可解释隐藏轮廓，但不能凭相邻物体或对称性虚构面、连线或箭头。'
                  '每个离散纯色区域需要独立封闭part；外框加开放分隔线不能代替这些面。'
                  '内部细线仍须恢复；文字排版遮罩由文字合成处理。',
                  '可读文字由文字层恢复，连接归属按arrow_ownership执行。'
                  '职责范围内的独立邻居缺失交coverage登记。'
                  '闭合轮廓用线稿显示；已有闭合面的颜色、透明度、描边宽度和虚实由后续样式阶段判断，不作为几何修订。',
                  '仅报告原图可定位的差异，使用当前F编号和具体修改方向，保留正确部件。'
                  'extra用于整个部件无源图依据或属于其它分支；部分轮廓错误用geometry/topology修改。'
                  '不确定则说明缺少的证据，不为满足检查条目编造错误。'
              ],
              'decision_rules': 'Correct: {"decision":"accept","issues":[]}. No explanations for passing checks. '
                                'Each extra issue names one whole unsupported or out-of-scope F part with source evidence. '
                                'A repeated-family count mismatch does not authorize removing all members; use topology when the unsupported member cannot be identified. '
                                'extra deletes the ENTIRE named part; use it only when that whole part is unsupported or belongs to another branch. '
                                'For a source-supported part with wrong segments use geometry/topology and request an edit. '
                                'Closed faces legitimately share boundary edges; that alone is not an extra face. '
                                'revise: image-supported error with a specific correction. needs_context: actual clipping '
                                'or insufficient image context. reject: wrong target only. unresolved: material ownership '
                                'or geometric ambiguity. Every non-accept needs evidence and action. Use only F labels in '
                                'issues.parts; [] for missing authorizes additions only. If existing spacing, positions or topology must change, '
                                'name those F parts in a separate geometry/topology issue. Return one JSON, no drafts.',
              'output_schema': review_schema(mapping)}
    if declared:
        prompt['declared_branch_roles']={
            'unit':declared['unit'],'roles':{alias:declared.get('retained_roles',{}).get(key,'uncertain')for alias,key in mapping.items()},
            'rule':'The observer declaration is fallible. Check each part against CLEAN and arrow_ownership. '
                   'Use extra for a whole unsupported or out-of-scope part and missing for source-visible in-scope geometry. '
                   'Do not replace delegated routes, raster content or lettering.'}
    if job.get('image_asset_ids'):
        from .assets import IMAGE_OWNERSHIP
        prompt['image_ownership'] = {'rule': IMAGE_OWNERSHIP}
    if shared:
        prompt['shared_vertices'] = shared
    from .connectivity import fitted_connections
    connections=fitted_connections({label:part for label,part in zip(mapping,local['parts'])})
    if connections['routes']:
        prompt['network_connections']={
            'same_endpoint_routes':connections['same_endpoint_routes'],
            'rule':'Trace source connections between actual nodes, not part names or total counts. '
                   'The listed endpoint matches are fallible: use the enlarged CLEAN/fit/SAM views '
                   'to distinguish a wrong endpoint from genuine parallel source connections. '
                   'Repair a misconnection instead of deleting the intended edge.'}
    cylinders=[f'F{i+1}' for i,p in enumerate(local['parts']) if p['kind']=='cylinder']
    if cylinders:
        prompt['native_primitive_contract']={'cylinder_parts':cylinders,
            'rule':'Each cylinder already contains the complete body silhouette AND top rim seam. '
                   'Check the observed top ellipse, extrusion depth, side tangencies and lower front rim against CLEAN. '
                   'Do not request extra side fragments or an invented hidden bottom face. '
                   'Exactly one matching top ellipse shares the body\'s first eight sample IDs and supplies '
                   'the independently styled top face ABOVE the body, including uniform-colour cylinders. '
                   'Style differences do not justify replacing supported primitive geometry. '
                   'Retype to paths only for visible taper, oblique extrusion or another contour the primitive cannot represent.'}
    # The observer's prose is not independent evidence. Replaying its reasons
    # for omissions biases the critic into accepting those same omissions.
    # CLEAN, fitted geometry and the actual peer registry define the scope.
    from .projected_edges import face_conflicts
    if conflicts:=face_conflicts(local,job):
        aliases={value:key for key,value in mapping.items()}
        prompt['projected_face_conflicts']={
            'overlapping_interiors':[{**row,'parts':[aliases[k] for k in row['parts']]} for row in conflicts],
            'rule':'These intersections are fitted-geometry diagnostics, not source truth. '
                   'Check CLEAN: visible faces of one convex solid normally tile the projected '
                   'silhouette, sharing edges without folding over each other. A hidden bottom/back '
                   'face must not be invented to meet a three-face count. Report a source-supported '
                   'topology/geometry/extra correction, or unresolved if source intent is ambiguous. '
                   'Do not justify a same-solid fold merely by a nearby separate occluder. '
                   'Preserve genuine source-supported overlapping layers or transparent geometry.'}
    # Localization names the outer silhouette, not necessarily its contents:
    # a parallelogram may contain many colored cells. Conversely a grid stack
    # may have only three closed planes plus numerous open dividers.
    if len(local['parts'])>=4:
        prompt['repeated_structure_check']={
            'rule':'先从原图观察每个重复单元的真实轮廓、间隙和共享连接，再对照拟合图。'
                   '分别核对形状、位置和数量；不能只数单元后接受整组，也不能把锥形或非矩形单元简化成长条。'
                   '网格外边界不计为内部行列分界；新增分隔线必须定位原图中相应的线迹及两端交点，'
                   '不能仅凭推测的行列数、均分间距或上一轮纠正意见补线。'
                   'source_observations说明实际对应的轮廓与布局。保留原图中的不规则、重叠和留白。'}
    if clipping := source_clipping(job):
        prompt['source_clipping'] = clipping
    if inspection and inspection.get('sample_points_shown') is False:
        prompt['version'] = 'shape_integration.fit_check.11'
        prompt['images'][1] = ('Image 2: fitted geometry on white, same crop coordinates; '
                              'no source pixels. Colored centerlines and F labels identify actual predictions.')
    # Only independently registered contents can explain apparent omissions.
    # An enclosing/nearby object must not hide this target's intrinsic frame.
    nearby = [row for row in job.get('input_context', {}).get('nearby_inventory', [])
              if row.get('relation_hint') == 'inside_target']
    if nearby:
        prompt['localization_context'] = {
            'nearby_inventory': nearby,
            'rule': 'Registry relations nominate independent peers/assets; containment alone does not establish ownership. '
                    'Match the same source feature before omitting it. Retain this target\'s distinct contour and exposed geometry. '
                    'The registry does not certify native peer reconstruction; report material ownership ambiguity as unresolved.'}
    enclosing = job.get('input_context', {}).get('registered_enclosing_shapes', [])
    if enclosing:
        prompt['registered_enclosing_shapes'] = enclosing
        prompt['enclosing_ownership_rule'] = (
            'These enclosing shapes have independent localization IDs. Check the CLEAN source to '
            'distinguish their contours from intrinsic parts of THIS target. Do not request a second '
            'copy of their boundary merely because it is absent from this target fit. Containment '
            'alone never excludes a genuinely distinct intrinsic frame. If contour ownership is '
            'ambiguous, return unresolved; a missing issue must identify a distinct source boundary.')
    peers=job.get('input_context',{}).get('registered_peer_shapes',[])
    if peers:
        prompt['registered_peer_shapes']=peers
        prompt['peer_ownership_rule']=(
            'These are independently localized peers visible in the observation crop. '
            'A peer grouping brace/frame must not be copied into this target. Match '
            'its actual source contour before calling a part extra or missing. '
            'Peer nodes do not cover this target\'s intrinsic wiring, and overlap '
            'alone does not transfer ownership of an internal colored region.')
    if local['fit_status'] != 'ok' or local['observation_status'] != 'ok':
        prompt['incomplete_geometry'] = {
            'observation_status': local['observation_status'],
            'reason': local['observation'].get('reason', ''),
            'failed_parts': [{'id': f'F{i+1}', 'error': p.get('error', '')}
                             for i, p in enumerate(local['parts']) if p['status'] != 'ok'],
            'rule': 'These are recorded omissions/fitting failures, not a complete rendering. '
                    'Identify source-supported missing parts for repair; do not infer completeness '
                    'from the currently visible fitted paths.'}
    if job.get('revision_context'):
        context = job['revision_context']
        prompt['revision_context'] = {
            'previous_issues': context['issues'], 'current_part_ids': mapping,
            'edited_ids': context['editable_ids'], 'removed_ids': context['remove_ids'],
            'preserved_ids': context['preserved_ids'],
            'rule': 'Verify the requested corrections against the source and inspect for regressions. '
                    'Existing protected parts were copied by the backend. Report issues with CURRENT '
                    'F labels only; previous stable part IDs are context, not current display labels.'}
    if prompt.get('network_connections'):
        # A node-link drawing does not need cuboid, cylinder or grid instructions.
        prompt['checks']=[CONNECTION_EVIDENCE,
            'Read CLEAN first: identify the visible nodes, frames and which nodes each line connects. '
            'Compare actual paths, not part names or total counts. Check missing, duplicate and wrong connections.',
            'Inspect node/frame contours and line attachment against CLEAN. Use the enlarged connection views '
            'when present; source pixels decide, colored paths are predictions and SAM contours are fallible '
            'boundary evidence. Preserve source-supported parallel routes and genuine occlusion.',
            'Report a visible discrepancy with affected F labels and a concrete correction. Repair a wrong '
            'endpoint rather than deleting the intended connection; preserve unaffected geometry. '
            'Accept only when supported, otherwise revise or state the unresolved evidence.']
    from .coverage_evidence import augment_prompt
    augment_prompt(prompt,job)
    source_observation_contract(prompt,mapping)
    return prompt


def check_object(job, folder, runtime, options, *, sam_first=False, preceding_review=None):
    from .critic_retry import retry_builder
    if not (folder/'inspection.json').exists() or not (folder/'critic_source.png').exists():
        from .control import inspection_images
        inspection_images(folder, show_samples=not getattr(getattr(runtime, 'spatial_evidence', None), 'clean_fit_view', False))
    local = read(folder/'fit_local.json')
    mapping, _, _ = part_inventory(local)
    save(folder/'critic_part_map.json', mapping)
    inspection = read(folder/'inspection.json')
    prompt = review_prompt(job, local, {**inspection,'sample_points_shown':False})
    clean_paths = True
    if not (folder/'fit_geometry.png').exists():
        from .views import geometry_only
        geometry_only(local,job['frame']['actual_crop_input_size_px'],folder/'fit_geometry.png',inspection.get('line_width_px',3))
    images = [('Source with image-asset boxes', folder/'critic_source.png'),
              ('Fitted geometry ONLY, same coordinates; no source pixels',folder/'fit_geometry.png')]
    prompt['geometry_view_rule']=(
        '先看干净原图，再看白底拟合图；白底图才是实际生成的几何。'
        '叠加图的原图底色不属于恢复结果。逐一比较轮廓位置、宽高、间隔及连接，'
        '数量相同不代表正确。标注框和描述不是几何证据。')
    from .source_spans import inspect as inspect_spans
    spans,span_images=inspect_spans(local,job,folder) if options.get('source_span_checks') else ([],[])
    if spans:
        prompt['source_span_checks']={
            'spans':spans,
            'rule':'The extra CLEAN/overlay pair magnifies a long fitted span with weak nearby source '
                   'edge support. Magenta is a prediction, not a source stroke. Inspect whether '
                   'the span is invented, displaced, occluded, unstroked or a real dash gap. '
                   'No automatic removal follows this numeric flag. If a grouping boundary '
                   'continues outside the crop, request context; do not close it along a guessed '
                   'rectangle side. A material unsupported edge needs revision or unresolved.'}
        images.extend(span_images)
    assess_revision = False
    if job.get('revision_context'):
        from pathlib import Path
        previous = Path(job['revision_context']['source_revision'])
        if (previous/'fit_geometry.png').exists():
            assess_revision = True
            images.append(('Previous geometry ONLY, BEFORE this edit; old F labels only', previous/'fit_geometry.png'))
            prompt['revision_context']['previous_part_ids'] = read(previous/'critic_part_map.json') if (
                previous/'critic_part_map.json').exists() else {}
            prompt['revision_comparison'] = (
                'Compare BEFORE and AFTER against the same clean source. improved requires a visible correction with no new '
                'defect or loss of supported geometry anywhere in the edit; unresolved old defects may remain. '
                'Any new defect is regressed, even if another edge improved. Use uncertain when correspondence or progress is unclear. '
                'Give one brief source-grounded observation; report remaining defects in issues as usual.')
    evidence = getattr(runtime, 'spatial_evidence', None)
    if sam_first:
        prompt, images = evidence.precritic(prompt, images, job, folder)
        if 'sam_auxiliary_evidence' not in prompt:
            return None  # No matched measurement: do not run an empty SAM check.
    elif evidence is not None and hasattr(evidence, 'critic'):
        prompt, images = evidence.critic(prompt, images, job, folder)
    if prompt.get('network_connections'):
        from .connectivity import connection_views
        images.extend(connection_views(local,job,folder,evidence))
    if prompt.get('repeated_structure_check'):
        from .views import repeated_details
        images.extend(repeated_details(local,job,folder))
    # The critic must see the same physical target locator as the observer.
    # CLEAN plus a wrong fit does not identify which repeated source occurrence
    # this worker owns; names and the current prediction cannot replace it.
    locator = folder/'overlay.png'
    if locator.is_file():
        label = 'TARGET OCCURRENCE: original clean crop with magenta ID locator; annotations only'
        images.append((label, locator))
        prompt['target_occurrence'] = {
            'protocol': 'shared_observer_critic_occurrence.1',
            'id': job['object_id'], 'locator_image': label,
            'rule': 'This is the SAME source occurrence locator supplied to the observer, in the SAME crop coordinates. '
                    'The rectangle is a fallible location hint, never a contour or a clipping boundary. '
                    'A source-supported member of this same object or group may extend beyond the locator. '
                    'Being outside it is not extra-part evidence: establish a separate peer/image owner or actual lack of source support. '
                    'If the observation crop cuts that member, request context. '
                    'Establish the intended physical source object from CLEAN and this locator before checking fitted parts. '
                    'Previous names, colors and the fitted paths may be wrong; do not switch to a similarly named neighbor '
                    'or request other independently located rows from this crop. If identity is ambiguous, report unresolved '
                    'instead of treating a neighboring occurrence as a missing part.'}
    observation_ids=source_observation_contract(prompt,mapping)
    prompt['output_schema'] = review_schema(mapping, observation_ids, revision=assess_revision)
    if preceding_review and preceding_review['issues']:
        prompt['preliminary_sam_findings'] = {
            'issues': [{k: issue[k] for k in ('kind', 'parts', 'evidence', 'action')}
                       for issue in preceding_review['issues']],
            'rule': 'These are fallible preliminary findings, not required edits. Verify or dismiss each '
                    'against Image 1 and the fitted paths. Report only defects supported by the source; '
                    'do not copy a SAM boundary or invent a correction to agree with the preliminary check.'}
    def validator(raw):
        validate_review(raw,mapping,observation_ids,revision=assess_revision)
        return raw

    # A critic returns a short decision, not a second full geometry observation.
    # Keep its inference bounded independently of the coordinate-generation budget.
    checker = runtime.for_stage(shape_schema_retries=options.get('check_retries', 1),
                                reasoning_effort=options.get('check_reasoning_effort', 'low'))
    system = SYSTEM.replace('model sample points and fitted paths', 'fitted paths') if clean_paths else SYSTEM
    raw = checker.call(folder/'calls', 'sam_check' if sam_first else 'check', system, prompt, images, validator, min(options['check_max_tokens'], 16384),
                       retry_builder=retry_builder(system, prompt, images, validator))
    independent=[]
    if raw['decision']=='accept' and spans and not sam_first:
        from .source_spans import observe_source
        # A prediction-primed accept cannot silently dismiss absent source ink.
        # Explicitly occluded spans stay with the contextual source review.
        for index,span in enumerate(spans):
            ident='span_'+span['span']
            primary=next((r for r in raw.get('source_observations',[]) if r['id']==ident),None)
            if primary is None or primary['source_relation']!='supported':continue
            observation=observe_source(runtime,folder,span,span_images[2*index][1],options)
            independent.append({'span':span['span'],**observation})
            if observation['visible_line']=='yes':continue
            save(folder/'critic_primary_review.json',raw)
            raw={**raw,'decision':'revise' if observation['visible_line']=='no' else 'unresolved',
                 'issues':[{'kind':'geometry' if observation['visible_line']=='no' else 'ambiguity',
                    'parts':[span['part']],'evidence':observation['source_evidence'],
                    'action':'Reobserve the actual source contour instead of closing a proposed rectangle '
                             'across this unsupported span. Preserve supported geometry; request more '
                             'context if the connected outline leaves the crop.'}],
                 'source_observations':[{**r,'source_relation':('missing' if observation['visible_line']=='no' else 'uncertain'),
                    'evidence':observation['source_evidence']} if r['id']==ident else r for r in raw['source_observations']]}
            validator(raw);break
    # Controller summary is backend-derived; it is not an extra model claim.
    reason = 'No defect reported by critic.' if raw['decision'] == 'accept' else ' '.join(i['evidence'] for i in raw['issues'])
    return {**raw, 'reason': reason, 'reason_origin': 'backend_summary',
            'protocol_version': prompt['version'], 'part_id_map': mapping,
            **({'independent_source_observations':independent} if independent else {}),
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
            lines.append(f"{issue['kind']}; affected part IDs: {owners}; evidence: {evidence}; action: {issue['action']}")
    return [value for line in lines if (value := sanitize_feedback(line))]
