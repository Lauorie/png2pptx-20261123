"""Shared prompt vocabulary and output schemas. No model calls or image processing."""
import math
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

PROMPT_VERSION = 'geometry_prompt.2'
POINT_ROLES = ['tail', 'tip', 'bend', 'sample', 'junction']
PATH_ROLES = ['arrow', 'trunk', 'branch', 'block_centerline']
KINDS = ['straight', 'polyline', 'curve']

SERIALIZATION = (
    'output_schema is a JSON Schema describing the response, not an example to copy. '
    'Return exactly one data object, without prose or Markdown fences. Coordinates must be finite '
    'JSON numbers, never quoted numbers, strings containing arrays, or point IDs. '
    'IDs and labels are strings; head flags are JSON booleans. Decimals are allowed. '
    'Return empty geometry arrays when no qualifying geometry is visible. State unresolved observations '
    'as short strings in uncertain; do not claim certainty by leaving known problems unreported.')

SCOPE = (
    'Inspect the supplied figure systematically. Include visible short, faint/gray, dashed, '
    'colored, curved, feedback, bidirectional and filled/block arrows, including actual arrow '
    'symbols in legends. Exclude text strokes, ordinary borders, ruler marks, and graph lines '
    'without arrowheads unless they visibly belong to a headed arrow network. A pointed object '
    'box is not automatically an arrow. Identify arrows from visible shaft/head evidence; '
    'source/target labels identify objects but do not establish an unseen connection.')

GEOMETRY = (
    'Trace the full visible SHAFT CENTERLINE. At a headed endpoint use the actual arrowhead '
    'APEX, not its base; a headless network segment terminates at its real tail or junction. '
    'Locate the tail and apex on visible arrow ink independently of the connected objects. '
    'A visible gap between an arrow endpoint and an object is valid: do not extend the arrow '
    'through that gap to a box edge, object center, or label. For filled/block arrows return '
    'the centerline and apex, not polygon vertices. straight: exactly two endpoints; polyline: '
    'endpoints and genuine corners; curve: at least three ordered on-curve samples, usually '
    '5..16, with more samples where curvature changes. Samples lie on the visible route, '
    'not at Bezier control handles or positions assumed from symmetry. Dash gaps do not make '
    'a straight shaft a polyline: follow the route supported by aligned dashes, without a '
    'point at every dash edge. Do not invent a route through an unresolved occlusion.')

NETWORK = (
    'Represent shared geometry once, supporting both splits (one route into several) and '
    'merges (several routes into one). Split routes at real shared junctions when needed to '
    'avoid duplicate shared segments. A geometric crossing alone is not a junction, and '
    'nearby distinct endpoints must remain distinct. role describes geometry: arrow is an '
    'independent route, trunk is a common network segment, branch joins or leaves that segment, '
    'block_centerline is a filled arrow centerline. These roles do not determine head flags. '
    'A trunk may carry an arrowhead; a branch may be headless. A headless segment must use '
    'role=trunk or branch and belong to a visibly connected network containing an arrowhead. '
    'Do not output an isolated headless network or invent a head to satisfy the schema.')

DIRECTION = (
    'start_head and end_head describe visible arrowheads at the first and last point, respectively. '
    'For a one-headed route, order points from its unheaded end to its apex. For a bidirectional '
    'route set both flags true and use role=tip at both endpoints unless an endpoint is a shared '
    'junction. For a headless network segment either traversal direction is valid. At a shared '
    'junction use point role=junction; elsewhere use tip for an actual apex, tail for an unheaded '
    'path endpoint, bend for a real corner, sample for an interior curve sample. Head flags are '
    'authoritative even when a shared point has role=junction.')

BBOX = (
    'bbox_2d=[xmin,ymin,xmax,ymax] is the tight axis-aligned extent of the complete visible '
    'arrow ink, including shaft and all arrowheads. For bent/curved arrows cover the entire '
    'route, not just its endpoints. Estimate each actual ink boundary; do not use a standard '
    'arrowhead height or fixed padding around the shaft. Exclude connected objects, their '
    'borders and labels. Preserve any visible endpoint-to-object gap. Coordinates are minimum '
    'and maximum occupied pixel-center positions (decimals allowed), not half-open crop bounds. '
    'Require xmin<xmax and ymin<ymax. For a shared network, return a box for each complete '
    'visible source-to-head route; boxes may overlap where routes share ink. A bidirectional '
    'route is one arrow. Do not return a separate box for a headless fragment; unresolved '
    'route grouping belongs in uncertain.')


def obj(properties):
    return {'type': 'object', 'properties': properties,
            'required': list(properties), 'additionalProperties': False}


def array(items, *, length=None):
    value = {'type': 'array', 'items': items}
    if length is not None:
        value.update(minItems=length, maxItems=length)
    return value


def enum(values):
    return {'type': 'string', 'enum': list(values)}


def number_array(length):
    return array({'type': 'number'}, length=length)


def validate_response(raw, schema):
    """Enforce the same JSON Schema shown in the prompt, including finite numbers."""
    try:
        Draft202012Validator(schema).validate(raw)
    except ValidationError as exc:
        raise ValueError(f'Invalid response at {list(exc.absolute_path)}: {exc.message}') from exc
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('JSON numbers must be finite')
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)
    finite(raw)
    return raw


def validate_units(size, units):
    if units not in ('absolute', 'normalized'):
        raise ValueError('units must be absolute or normalized')
    if len(size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 2 for v in size):
        raise ValueError('Image width and height must be integers >= 2')


def frame_contract(size, units, *, padding=0, ruler=False, coordinate_max=999):
    """Only coordinate_system differs between matched absolute/normalized arms."""
    validate_units(size, units)
    if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
        raise ValueError('padding must be a nonnegative integer')
    w, h = size
    maximum = [w-1, h-1] if units == 'absolute' else [coordinate_max, coordinate_max]
    frame = {
        'reference': 'source_region_inside_padded_input' if padding else 'entire_supplied_input_image',
        'reference_size_px': {'width': w, 'height': h},
        'reference_bounds_in_input_pixel_centers': [padding, padding, padding+w-1, padding+h-1],
        'origin_in_input_pixels': [padding, padding],
        'axes': 'x increases rightward; y increases downward',
        'meaning': ('The supplied PNG contains a source figure plus an exterior margin. The bounds '
                    'above locate the source region in input pixels; they are metadata, not target '
                    'arrow coordinates. Output uses this inner source region as its reference. '
                    'Canvas pixel positions must first subtract origin_in_input_pixels. '
                    'The margin and enclosing frame line are not source arrow ink.' if padding else
                    'The reference is exactly the entire supplied PNG, including any intrinsic white '
                    'space. If the controller resized or cropped another image to produce this PNG, '
                    'use this PNG only; the controller handles mapping back afterward.'),
    }
    if ruler:
        frame['ruler'] = ('Printed ticks show the requested output units relative to the inner source '
                          'region. Tick labels are data; do not confuse their canvas positions with '
                          'their printed values or include the ruler marks as arrows.')
    coordinates = {
        'units': 'reference_pixel_centers' if units == 'absolute' else f'normalized_0_{coordinate_max}',
        'top_left': [0, 0], 'bottom_right': maximum,
        'limits': {'x': [0, maximum[0]], 'y': [0, maximum[1]]},
        'definition': ('Report positions in reference pixels directly.' if units == 'absolute' else
                       f'Normalize each axis independently across the reference: x_out=x_ref*{coordinate_max}/{w-1}; '
                       f'y_out=y_ref*{coordinate_max}/{h-1}. Here x_ref,y_ref are reference pixel-center '
                       'positions starting at zero. The denominators are width-1 and height-1. '
                       'Final output must use these normalized units, even if you estimated pixels internally.'),
        'precision': 'Use finite JSON numbers. Decimals are allowed; no integer-only grid is required.',
    }
    return {'input_image_size_px': {'width': w+2*padding, 'height': h+2*padding},
            'reference_frame': frame, 'coordinate_system': coordinates}


def graph_schema():
    point = obj({'id': {'type': 'string', 'minLength': 1}, 'xy': number_array(2), 'role': enum(POINT_ROLES)})
    path = obj({'id': {'type': 'string', 'minLength': 1}, 'label': {'type': 'string'}, 'kind': enum(KINDS),
                'point_ids': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 2, 'uniqueItems': True},
                'start_head': {'type': 'boolean'}, 'end_head': {'type': 'boolean'}, 'role': enum(PATH_ROLES)})
    return obj({'points': array(point), 'arrows': array(path),
                'more_pending': {'type': 'boolean'}, 'uncertain': array({'type': 'string'})})


def bbox_schema():
    return obj({'arrows': array(obj({'label': {'type': 'string'}, 'bbox_2d': number_array(4)})),
                'uncertain': array({'type': 'string'})})


def proposal_schema(table):
    points = obj({'xy': number_array(2), 'role': enum(POINT_ROLES)})
    refs = [p['id'] for p in table['points']]
    if refs:
        points = {'oneOf': [points, obj({'ref': enum(refs)})]}
    replacements = array(enum([p['id'] for p in table['paths']])) if table['paths'] else {
        'type': 'array', 'maxItems': 0}
    return obj({'label': {'type': 'string'}, 'kind': enum(KINDS),
                'points': {'type': 'array', 'items': points, 'minItems': 2, 'maxItems': 48},
                'start_head': {'type': 'boolean'}, 'end_head': {'type': 'boolean'}, 'role': enum(PATH_ROLES),
                'replace_ids': replacements, 'evidence': {'type': 'string'}})


def context_schema():
    return array(obj({'anchor': number_array(2),
                      'sides': {'type': 'array', 'items': enum(['left', 'right', 'top', 'bottom']),
                                'minItems': 1, 'uniqueItems': True},
                      'reason': {'type': 'string'}}))


def validate_headed_networks(arrows):
    """Use declared point IDs only; no joining of crossings or nearby coordinates."""
    adjacency, headed = {}, set()
    for arrow in arrows:
        ids = arrow['point_ids']
        for pid in ids:
            adjacency.setdefault(pid, set())
        for left, right in zip(ids, ids[1:]):
            adjacency[left].add(right)
            adjacency[right].add(left)
        if arrow['start_head']:
            headed.add(ids[0])
        if arrow['end_head']:
            headed.add(ids[-1])
    unseen = set(adjacency)
    while unseen:
        pending, component = [unseen.pop()], set()
        while pending:
            pid = pending.pop()
            if pid in component:
                continue
            component.add(pid)
            pending.extend(adjacency[pid] - component)
        unseen.difference_update(component)
        if not component.intersection(headed):
            raise ValueError('Isolated headless network: no declared connection to an arrowhead')
