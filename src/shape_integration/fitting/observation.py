"""Sparse local observation prompt extracted from the no-text-bbox experiment.

Sampling clauses remain unchanged; four compact rules cover ownership,
contours, layer completion and output. No history module is imported.
"""
import re
from .contracts import model_observation_schema
from .semantics import GEOMETRY_RULES
from .coordinates import coordinate_mode, model_size, NORM1000_CONVENTION

SYSTEM = 'Reconstruct the requested object from the clean crop. Return only JSON matching the supplied schema.'
EDIT_SYSTEM = ('Edit only the explicitly authorized geometric parts from the clean crop. '
               'The backend preserves other parts. Follow the part-edit scope and text ownership rules. '
               'Return only JSON matching the supplied schema.')


def system_for(job):
    return EDIT_SYSTEM if job.get('revision_context') else SYSTEM


IDENTITY_NOTE_SCOPE = ('描述仅帮助定位，可能写错颜色、形状和数量；以标注位置对应的干净原图为准。'
                       '不要因为原图颜色与旧描述不同就拒绝或移到邻近目标。'
                       '定位框不是裁切边界；同一物体或同一组的源图支持成员可超出框。不能仅因框外而删除，'
                       '排除须有独立peer/image归属或无源图支持的证据；观察裁块截断成员时请求context。'
                       '定位类型不限制职责范围内的固有部件数量；peer/image匹配只约束这些固有部件，连接路线仍按统一箭头职责处理。')

NUMBER = r'-?\d+(?:\.\d+)?'

COORDINATE = re.compile(r'(?i)(?:~|≈)?\b(?:global\s+)?[xy](?:[-\s]+(?:range|span|bounds?|coordinate))?\s*(?:(?:from|to|is|at|=|~|≈|:)\s*)?(?:~|≈)?'+NUMBER+r'(?:\s*(?:-|–|—|to)\s*(?:~|≈)?'+NUMBER+r')?')

GEOMETRY_CLAUSE = re.compile(r'(?i)(?:bbox|bounding\s*box|\bcoordinates?\b|\bpixels?\b|\b\d+(?:\.\d+)?\s*(?:px|times|[x×]\b)|\b(?:upscal\w*|rescal\w*|resize\w*|resolution|scal(?:e|ing))\b|\b[xy]\s*[=~≈]|[\[(]\s*-?\d+(?:\.\d+)?\s*[,，]|\b(?:width|height|radius|radii|diameter)\s*(?:of|is|=|:|~)?\s*\d|\b(?:image\s+boundary|boundary\s+at)\s*\d)')

EXTENT_NUMBER = re.compile(r'(?<![A-Za-z0-9_.])\d{2,}(?:\.\d+)?(?![A-Za-z0-9_.])|\b\d{2,}\s*[x×]\s*\d+\b|\b\d+\s*[x×]\s*\d{2,}\b')

def sanitize(text):
    pieces=re.split(r';|(?<=[.!?])\s+',str(text))
    kept=[];removed=[]
    for piece in pieces:
        original=piece.strip()
        # First remove coordinate-bearing parentheticals without losing the object identity.
        piece=re.sub(r'\([^()]*\)|\[[^\[\]]*\]',lambda m:'' if COORDINATE.search(m[0]) or GEOMETRY_CLAUSE.search(m[0]) or EXTENT_NUMBER.search(m[0]) else m[0],piece)
        piece=COORDINATE.sub('',piece)
        piece=re.sub(r'\s+',' ',piece).strip(' ,;')
        piece=re.sub(r'\s+([.,])',r'\1',piece)
        # Numeric extent-only clauses convey no identity once their coordinates are gone.
        useless=re.fullmatch(r'(?i)(?:node extents span|their proposed boxes span|extents span|span|at)[\s,.-]*',piece)
        if GEOMETRY_CLAUSE.search(piece) or EXTENT_NUMBER.search(piece) or useless or not piece:
            removed.append(original)
        else:
            if piece!=original:removed.append(original)
            kept.append(piece)
    return '; '.join(kept),removed

def sanitize_feedback(text):
    """Keep qualitative critique (including 'pixels'); remove numerical geometry only."""
    value = str(text)
    value = re.sub(r'[\[(][^\[\]()]*[0-9][^\[\]()]*[\])]', '', value)
    value = COORDINATE.sub('', value)
    value = re.sub(r'(?i)\b(?:width|height|radius|diameter|bbox|coordinates?)\s*(?:is|of|at|=|:)?\s*'+NUMBER, '', value)
    value = re.sub(NUMBER+r'\s*(?:px|pixels?|degrees?)\b', '', value, flags=re.I)
    # Numeric estimates must not anchor a fresh observation. Preserve IDs/feature names.
    value = EXTENT_NUMBER.sub('', value)
    return re.sub(r'\s+', ' ', value).strip(' ,;')


def source_clipping(job):
    frame = job['frame']
    crop = frame.get('source_crop_half_open')
    size = frame.get('original_image_size_px')
    if not crop or not size:
        return None
    edges = [name for name, at_edge in zip(('left', 'top', 'right', 'bottom'),
             (crop[0] == 0, crop[1] == 0, crop[2] == size[0], crop[3] == size[1])) if at_edge]
    return {'image_edges': edges, 'rule': (
        'Only these crop edges coincide with the ORIGINAL image boundary; no larger source exists there. '
        'If a filled target is clipped by these image edges, reconstruct its visible region as a closed '
        'fill-only path using the image edge as a clipping boundary, not an inferred hidden contour. '
        'The clipping boundary is NOT a stroked edge. Put source-visible border spans in separate open '
        'parts; an unfilled frame cut by the original image boundary remains open there. All source-visible '
        'geometry represented is ok; explain the source clipping briefly. Do not request more context solely '
        'for these edges. Other crop edges still require context when they cut the target.')} if edges else None


def make_prompt(job, descriptions=()):
    w, h = job['frame']['actual_crop_input_size_px']
    mode = coordinate_mode(job)
    context = job.get('input_context', {})
    identities = list(dict.fromkeys(sanitize(d)[0] for d in
        list(descriptions) + context.get('identity_descriptions', []) if sanitize(d)[0]))
    target = {'id': job['object_id'], 'label': sanitize(job['label'])[0]}
    if identities:
        target['identity_descriptions'] = identities
        target['description_scope'] = IDENTITY_NOTE_SCOPE
    target = {k: v for k, v in target.items() if v}
    prompt = {
        'target': target,
        'images': ['Image 1: clean crop; observe geometry here.',
                   'Image 2: identical crop with magenta target box and ID for identification only; the annotation is not a contour.'],
        'local_coordinates': {'width_px': w, 'height_px': h,
            'convention': 'Top-left pixel center is (0,0); x right, y down. Stay within [0,width_px-1] and [0,height_px-1]. Integers or two decimals are sufficient.'},
        'task': 'Restore the target shape geometry and assign each returned part its source role.',
        'compound_connection_scope_eligible': job.get('proposed_type')=='compound_symbol',
        'rules': dict(GEOMETRY_RULES),
        'part_ownership': ('Declare ownership.unit from CLEAN as descriptive source context, then assign every part by its visible role. '
            'Follow rules.scope: connection means internal headless wiring; external_connection means an external headless route; '
            'directional_connection and directional_head describe headed routes. Use open paths for headless wiring and separate '
            'parts when ownership differs. intrinsic_stroke means a structural stroke of the symbol itself, not an external route. '
            'Return geometry for shape-owned content. External routes and lettering are handled by their '
            'own branches. Assess completeness and context needs for the shape-owned content only. '
            'Use uncertain when CLEAN cannot establish the source role.'),
        'sampling': {
            'cylinder': 'Cylinder BODY: exactly 11 contour samples. First 8 describe the TOP ellipse in the same clockwise order as ellipse. Last 3 are the LOWER front rim: right tangent, frontmost midpoint, left tangent, along the cylinder local axes. Observe actual extrusion depth, not the localization box. Cylinder includes the full body and top seam. ALWAYS follow the body with ONE ellipse part sharing the EXACT first eight point_ids, for the independently styled visible top face. No extra points or second cylinder. This also prevents Office automatic top shading from overriding source colour. Use paths for tapered or irregular bodies.',
            'ellipse': 'Circle or ellipse: use kind ellipse and exactly 8 contour points in clockwise order: topmost, one observed upper-right arc point, rightmost, lower-right arc point, bottommost, lower-left arc point, leftmost, upper-left arc point. Extremes are with respect to crop x/y even when tilted.',
            'rounded_rectangle': 'Rounded rectangle: exactly 16 contour points clockwise: top straight midpoint, top-right arc start, top-right arc middle, top-right arc end, right straight midpoint, bottom-right arc start, bottom-right arc middle, bottom-right arc end, bottom straight midpoint, bottom-left arc start, bottom-left arc middle, bottom-left arc end, left straight midpoint, top-left arc start, top-left arc middle, top-left arc end. Start/end mean where that corner meets the adjacent straight span. For a rotated object follow the same local side order clockwise. The backend fits one common radius for all four corners. Use a closed path when the source corners have different radii or include sharp corners.',
            'rectangle': 'Rectangle: 4 observed corners clockwise starting at its top-left. Do not report virtual sharp corners of a rounded rectangle.',
            'path': 'Use primitives above where suitable, including inside compounds; use path for other polygons/lines/curves. Each line segment references two endpoint IDs; each curve uses at least three ordered contour knots. Segments sharing an end/start ID continue one stroke. In an open path a different start ID begins a separate pen stroke; use this for branches without adding return strokes or invented connectors. A closed path must be continuous and end at its first ID. Keep straight spans as lines and curved spans as curves. Do not force other shapes into 4/8/11/16 samples.',
            'occluded_samples': 'Keep the same sample counts/order on occluded portions, using completion supported by the layers_and_3d rule.'},
        'output_schema': model_observation_schema(model_size((w, h), mode))}
    if mode == 'norm1000':
        prompt['local_coordinates'] = {'convention': NORM1000_CONVENTION}
    if clipping := source_clipping(job):
        prompt['source_clipping'] = clipping
    # Identity is stated once. Preserve the selected neighbor inventory exactly.
    local_context = {k: v for k, v in context.items()
                     if k not in ('identity_descriptions', 'ownership_note','coverage_source_anchors',
                                  'coverage_source_extent_original','coverage_source_requests') and v}
    if local_context:
        prompt['localization_context'] = local_context
    if any('represented_parts' in row for row in context.get('nearby_inventory', [])):
        prompt['coverage_scope'] = ('本次只补目标描述中的缺失部分。peer 的 represented_parts 是已经生成的部件，'
                                    '不按旧名称猜测归属，也不重复画这些部件；空列表表示尚未生成。')
    if context.get('registered_peer_shapes'):
        prompt['peer_ownership_rule'] = (
            'Match registered peer parts to CLEAN before excluding them. '
            'Unregistered intrinsic regions of this target remain in scope; '
            'a peer name or containing box does not establish ownership. '
            'For an enclosing frame, follow its own source stroke and retain visible padding '
            'around inner peers. Do not use an inner box edge as the outer frame edge.')
    feedback = [sanitize_feedback(x) for x in job.get('revision_feedback', [])]
    feedback = [x for x in feedback if x]
    if feedback:
        prompt['revision_feedback'] = feedback
        prompt['revision_policy'] = (
            'No previous fit is shown: reconstruct ALL target parts from the clean image. '
            'Apply feedback only when supported by the image and consistent with the geometry rules.')
    if job.get('revision_context'):
        from .revision import revision_prompt
        prompt['task'] = 'Repair only the requested parts of this target; the backend preserves all other parts.'
        prompt['revision_task'] = revision_prompt(job['revision_context'], (w, h), mode)
        prompt['revision_policy'] = (
            'This is a bounded part edit, not a new decomposition of the whole object. Follow revision_task. '
            'Use its stable part IDs independently of prose feedback. Preserve text/image ownership and '
            'unaffected geometry. Request context only when the clean crop truly clips the intended target.')
        prompt['rules']['output'] = prompt['rules']['output'].replace(
            'ok means complete supported shape geometry', 'ok means all requested edits are supplied')
    if job.get('image_asset_ids'):
        from .assets import IMAGE_OWNERSHIP
        prompt['images'][1] = 'Image 2: identical crop with magenta target locator and cyan IMAGE footprints; annotations are not contours.'
        prompt['part_ownership'] += ' Use image_content only for a part explicitly delegated to a registered IMAGE footprint.'
        prompt['image_ownership'] = {
            'registered_ids': job['image_asset_ids'],
            'rule': IMAGE_OWNERSHIP}
    from .coverage_evidence import augment_prompt
    return augment_prompt(prompt,job)
