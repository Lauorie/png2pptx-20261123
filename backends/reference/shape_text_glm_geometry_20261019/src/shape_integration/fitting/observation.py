"""Sparse local observation prompt extracted from the no-text-bbox experiment.

Sampling clauses remain unchanged; four compact rules cover ownership,
contours, layer completion and output. No history module is imported.
"""
import re
from .contracts import model_observation_schema
from .semantics import GEOMETRY_RULES
from .coordinates import coordinate_mode, model_size, NORM1000_CONVENTION

SYSTEM = 'Reconstruct the requested object from the clean crop. Return only JSON matching the supplied schema.'

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


def make_prompt(job, descriptions=()):
    w, h = job['frame']['actual_crop_input_size_px']
    mode = coordinate_mode(job)
    context = job.get('input_context', {})
    identities = list(dict.fromkeys(sanitize(d)[0] for d in
        list(descriptions) + context.get('identity_descriptions', []) if sanitize(d)[0]))
    target = {'id': job['object_id'], 'label': sanitize(job['label'])[0],
              'proposed_shape': job['proposed_type']}
    if identities:
        target['identity_descriptions'] = identities
    target = {k: v for k, v in target.items() if v}
    prompt = {
        'target': target,
        'images': ['Image 1: clean crop; observe geometry here.',
                   'Image 2: identical crop with magenta target box and ID for identification only; the annotation is not a contour.'],
        'local_coordinates': {'width_px': w, 'height_px': h,
            'convention': 'Top-left pixel center is (0,0); x right, y down. Stay within [0,width_px-1] and [0,height_px-1]. Integers or two decimals are sufficient.'},
        'task': 'Return all target parts in one reconstruction.',
        'rules': dict(GEOMETRY_RULES),
        'sampling': {
            'ellipse': 'Circle or ellipse: use kind ellipse and exactly 8 contour points in clockwise order: topmost, one observed upper-right arc point, rightmost, lower-right arc point, bottommost, lower-left arc point, leftmost, upper-left arc point. Extremes are with respect to crop x/y even when tilted.',
            'rounded_rectangle': 'Rounded rectangle: exactly 16 contour points clockwise: top straight midpoint, top-right arc start, top-right arc middle, top-right arc end, right straight midpoint, bottom-right arc start, bottom-right arc middle, bottom-right arc end, bottom straight midpoint, bottom-left arc start, bottom-left arc middle, bottom-left arc end, left straight midpoint, top-left arc start, top-left arc middle, top-left arc end. Start/end mean where that corner meets the adjacent straight span. For a rotated object follow the same local side order clockwise. The backend fits one common radius for all four corners; report actual observations even if distortion makes them deviate.',
            'rectangle': 'Rectangle: 4 observed corners clockwise starting at its top-left. Do not report virtual sharp corners of a rounded rectangle.',
            'path': 'Use primitives above where suitable, including inside compounds; use path for other polygons/lines/curves. Each line segment references two endpoint IDs; each curve uses at least three ordered contour knots. Adjacent segments share endpoint IDs; a closed path ends at its first ID. Do not force other shapes into 4/8/16 samples.',
            'occluded_samples': 'Keep the same sample counts/order on occluded portions, using completion supported by the layers_and_3d rule.'},
        'output_schema': model_observation_schema(model_size((w, h), mode))}
    if mode == 'norm1000':
        prompt['local_coordinates'] = {'convention': NORM1000_CONVENTION}
    # Identity is stated once. Preserve the selected neighbor inventory exactly.
    local_context = {k: v for k, v in context.items()
                     if k not in ('identity_descriptions', 'ownership_note') and v}
    if local_context:
        prompt['localization_context'] = local_context
    feedback = [sanitize_feedback(x) for x in job.get('revision_feedback', [])]
    feedback = [x for x in feedback if x]
    if feedback:
        prompt['revision_feedback'] = feedback
        prompt['revision_policy'] = (
            'No previous fit is shown: reconstruct ALL target parts from the clean image. '
            'Apply feedback only when supported by the image and consistent with the geometry rules.')
    return prompt
