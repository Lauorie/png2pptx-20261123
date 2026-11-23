"""Endpoint semantics carried with geometry; native med/med diagnostic rendering."""
import copy
from pathlib import Path
from PIL import Image

INSTRUCTION = (
    'For EVERY drawable path return heads=[start_type,end_type], attached to its ordered first/last point. '
    'Use none for no terminal mark, triangle for a directional arrowhead (generic at this geometry stage), '
    'oval for a round terminal mark, and unknown only when evidence is unclear. '
    'A round node or a round line cap is not automatically an endpoint mark. '
    'Keep both heads on a genuinely double-headed path; do not infer direction from point order. '
    'Put an arrow endpoint at its visible tip, and a round terminal endpoint at its center. '
    'Do not draw head wings as separate paths. Sizes are fixed to med/med here; do not estimate head dimensions, color, width or dashes.'
)


def attach(row):
    """Synchronize endpoint flags without changing points or path ownership."""
    import protocol
    heads = row.get('heads')
    if not isinstance(heads, list) or len(heads) != 2 or any(h not in protocol.HEADS for h in heads):
        raise ValueError('Geometry needs two explicit endpoint head types')
    row['geometry_heads_reported'] = True
    row['head_source'] = 'glm_geometry_endpoints'
    for end, head in zip(('start', 'end'), heads):
        row[end + '_head'] = head not in ('none', 'unknown')
        row[end + '_head_type'] = head if head not in ('none', 'unknown') else None
    return row


def head_style(head):
    return None if head in ('none', 'unknown') else dict(type=head, width='med', length='med')


def provisional(style, row):
    """Pixel style estimates cannot overwrite model-owned endpoint semantics."""
    result = copy.deepcopy(style)
    if row.get('geometry_heads_reported'):
        for end, head in zip(('start', 'end'), row['heads']):
            result[end] = head_style(head)
        if row.get('geometry_preview_width_px') is not None:
            result['width_px'] = row['geometry_preview_width_px']
        result['endpoint_source'] = 'glm_geometry_endpoints'
        result['head_size_source'] = 'provisional_med_med'
        result['unknown_endpoints'] = [end for end, head in zip(('start', 'end'), row['heads']) if head == 'unknown']
    return result


def category(head):
    if head in ('none', 'unknown'):
        return head
    return 'round' if head == 'oval' else 'headed'


def render(clean, rows, folder, width_px=2.0, color='#e31a1c'):
    """Fresh native PPT -> PDF -> pixels, never a cached head reference or SVG substitute."""
    import native
    from transport.common import write_json
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    copied = copy.deepcopy(rows)
    for row in copied:
        row['style'] = dict(color=color, width_px=width_px, dash=dict(type='solid', preset='solid'),
                            start=head_style(row['heads'][0]), end=head_style(row['heads'][1]), state='geometry_preview')
    native.write(clean.size, copied, [r['style'] for r in copied], folder/'geometry.pptx')
    native.render(folder/'geometry.pptx', folder/'native', clean.size)
    # The PDF export paints an opaque white page even with alpha=True. Remove
    # only those leading page fills from its native vectors before composition.
    from native_foreground import render_layer
    rgba = render_layer(folder/'native/styled.svg', clean.size, folder/'native')
    overlay = clean.convert('RGBA')
    overlay.alpha_composite(rgba)
    overlay = overlay.convert('RGB')
    # Validate the actual model preview, not merely whether native export worked.
    # Pixels outside the native foreground must retain the unmarked source.
    import numpy as np
    untouched = np.asarray(rgba.getchannel('A')) == 0
    source_rgb = np.asarray(clean.convert('RGB'))
    preserved = np.array_equal(np.asarray(overlay)[untouched], source_rgb[untouched])
    if not untouched.any() or not preserved:
        raise ValueError('Native geometry preview did not preserve the CLEAN background')
    overlay.save(folder/'overlay.png')
    write_json(folder/'preview.json', dict(geometry_only=True, width_px=width_px, head_sizes='med/med',
        endpoint_heads=[dict(id=r['id'], heads=r['heads']) for r in rows],
        source_context_preserved=preserved, source_context_pixels=int(untouched.sum()),
        foreground='native/foreground.json', preview_contract='native_transparent_overlay_v1',
        renderer='native/renderer.json', export='geometry.export.json', new_render=True))
    return overlay
