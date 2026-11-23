"""Region-first line geometry; SAM is optional image evidence, never ownership.

The point/path interface follows the earlier shape arrow core. Numerical
compilation reuses the current line/curve compiler without fitting to masks.
This module is experimental and is not selected by the production pipeline.
"""
import copy
import math
import protocol
import segmented_geometry

SYSTEM = (
    'Recover visible line geometry from CLEAN. Return only the requested JSON. '
    'Colored boxes and mask overlays are annotations, not source ink.'
)


def prompt(size, target_box, description=''):
    return dict(
        task='Recover the visible arrows and connector centerlines in TARGET. Return their geometry directly.',
        size=list(size), target_box=list(target_box), target=description,
        scope=protocol.SCOPE,
        instructions=[
            'CLEAN is the original crop. TARGET locates the observation, not a clipping boundary or a predicted path. '
            'Follow a relevant line to its visible endpoints; unrelated neighbors are context. '
            'If optional MASK evidence is supplied, it may be fragmented, mixed or wrong. '
            'Use it only when useful; do not match or explain mask IDs, and do not let mask count determine path count.',
            'Use current CLEAN pixel centers only. Put headed endpoints at visible arrow tips and headless endpoints where the ink ends. '
            'Do not snap to nearby shape centers or infer all-to-all connections. '
            'If the complete route cannot be judged because it leaves CLEAN, return needs_context instead of guessing.',
            'A line segment joins its points with straight lines. Curve points lie ON the visible centerline. '
            'Separate sharp corners from visibly smooth curves. Dashed lines have one continuous underlying centerline. '
            'Use independent paths for independent strokes; crossings are not junctions. '
            'Only actual shared junctions reuse a point ID; draw a shared trunk once. No forced single traversal.',
            'start_head and end_head say only whether a visible head is present at the ordered first/last endpoint. '
            'Do not classify head shape or estimate color, width or dashes. Broad block arrows require separate shape recovery; '
            'do not turn their filled bodies into centerlines.',
            'Return not_arrow with a brief reason when TARGET has no eligible linework. '
            'For ready, return the complete geometry with any material uncertainty listed briefly. '
            'Do not explain excluded shapes individually.',
        ],
        response_format=dict(
            status='ready|needs_context|not_arrow',
            points='object mapping point ID to [x,y] in CLEAN pixels',
            paths='[{id,segments:[{kind:line|curve,points:[ordered point IDs]}],start_head:boolean,end_head:boolean}]',
            uncertain='list of brief unresolved visual issues',
            reason='brief; required for needs_context or not_arrow',
        ),
    )


def validate(raw, size):
    """Validate references and coordinates, never infer missing geometry."""
    if not isinstance(raw, dict):
        raise ValueError('Geometry must be a JSON object')
    state = raw.get('status')
    if state not in ('ready', 'needs_context', 'not_arrow'):
        raise ValueError('status must be ready, needs_context or not_arrow')
    if state != 'ready':
        if not isinstance(raw.get('reason'), str) or not raw['reason'].strip():
            raise ValueError('Non-ready result needs a brief reason')
        if raw.get('points') or raw.get('paths'):
            raise ValueError('Non-ready result must not silently discard supplied geometry')
        return dict(status=state, points={}, paths=[], reason=raw['reason'],
                    uncertain=raw.get('uncertain', []))
    points, paths = raw.get('points'), raw.get('paths')
    if not isinstance(points, dict) or not points or not isinstance(paths, list) or not paths:
        raise ValueError('ready requires a nonempty point table and paths')
    for name, xy in points.items():
        if not isinstance(name, str) or not name:
            raise ValueError('Point IDs must be nonempty strings')
        if not isinstance(xy, list) or len(xy) != 2:
            raise ValueError('Each point must be [x,y]')
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in xy):
            raise ValueError('Coordinates must be finite numbers')
        if any(not 0 <= xy[i] <= size[i] - 1 for i in (0, 1)):
            raise ValueError('Point outside CLEAN pixel frame: ' + name)
    seen = set()
    for path in paths:
        pid = path.get('id')
        if not isinstance(pid, str) or not pid or '/' in pid or pid in seen:
            raise ValueError('Path IDs must be unique nonempty strings without /')
        seen.add(pid)
        if any(type(path.get(k)) is not bool for k in ('start_head', 'end_head')):
            raise ValueError('Endpoint head-presence flags must be booleans')
        segments = path.get('segments')
        if not isinstance(segments, list) or not segments:
            raise ValueError('Each path requires segments')
        previous = None
        for segment in segments:
            kind, refs = segment.get('kind'), segment.get('points')
            if kind not in ('line', 'curve') or not isinstance(refs, list):
                raise ValueError('Each segment needs line/curve and point references')
            if len(refs) < (2 if kind == 'line' else 3):
                raise ValueError('A line needs >=2 points; a curve needs >=3 on-curve points')
            if any(not isinstance(n, str) or n not in points for n in refs):
                raise ValueError('Undefined point reference')
            if previous is not None and refs[0] != previous:
                raise ValueError('Consecutive segments must share an endpoint ID; independent strokes use separate paths')
            if any(points[a] == points[b] for a, b in zip(refs, refs[1:])):
                raise ValueError('Consecutive points must differ')
            previous = refs[-1]
    uncertain = raw.get('uncertain', [])
    if not isinstance(uncertain, list):
        raise ValueError('uncertain must be a list')
    return dict(status=state, points=copy.deepcopy(points), paths=copy.deepcopy(paths),
                uncertain=copy.deepcopy(uncertain))


def compile_rows(answer, prefix, width_px=1.5):
    """Adapt exact model geometry to existing editable paths; no SAM snapping."""
    if answer['status'] != 'ready':
        return []
    paths = []
    for path in answer['paths']:
        segments = []
        for segment in path['segments']:
            refs = segment['points']
            if segment['kind'] == 'line':
                segments.extend(dict(kind='line', nodes=[a, b]) for a, b in zip(refs, refs[1:]))
            else:
                segments.append(dict(kind='curve', nodes=list(refs)))
        paths.append(dict(id=path['id'], segments=segments,
                          heads=['triangle' if path[k] else 'none' for k in ('start_head', 'end_head')]))
    wire = dict(nodes=[[k, *xy] for k, xy in answer['points'].items()],
                routes=[dict(id='region', paths=paths)])
    rows = segmented_geometry.geometry(wire, {'region': dict(id=prefix)})
    for i, (row, path) in enumerate(zip(rows, answer['paths']), 1):
        row.update(parent_id=row['id'], label=f'A{i}', source_masks=[], source_mask_paths=[],
                   head_presence=[path['start_head'], path['end_head']],
                   head_type_fitted=False, geometry_method='region_glm_direct',
                   width_estimate=width_px, start_head=path['start_head'], end_head=path['end_head'])
        row['style'] = dict(
            color='#000000', width_px=width_px, dash=dict(type='solid', preset='solid'),
            start=dict(type='triangle', width='med', length='med') if path['start_head'] else None,
            end=dict(type='triangle', width='med', length='med') if path['end_head'] else None,
            state='display_only', source='fixed_geometry_comparison',
        )
    return rows
