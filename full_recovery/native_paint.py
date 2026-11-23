"""Read actual native block paint at the composition boundary.

Older arrow products carry geometry without per-part paint metadata. The PPTX
is the paint authority; missing or ambiguous metadata never guesses a fill.
No source product or geometry is modified.
"""
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile

from lxml import etree

from .compose import P, A, matches_name, sha

PROTOCOL = 'native_block_paint.1'


def observed_fill(props):
    if props is None:
        return None
    if props.find('{'+A+'}noFill') is not None:
        return {'fill': None, 'fill_opacity': 1.}
    solid = props.find('{'+A+'}solidFill')
    if solid is None:
        return None
    color = solid.find('{'+A+'}srgbClr')
    if color is None:
        return None
    value = color.get('val', '')
    if len(value) != 6 or any(c not in '0123456789ABCDEFabcdef' for c in value):
        return None
    # Color transforms require theme/color evaluation; do not infer a pigment.
    if any(etree.QName(node).localname != 'alpha' for node in color):
        return None
    alpha = color.find('{'+A+'}alpha')
    opacity = int(alpha.get('val'))/100000 if alpha is not None else 1.
    if not 0 <= opacity <= 1:
        return None
    return {'fill': '#'+value.upper(), 'fill_opacity': opacity}


def bind(scene, pptx):
    result = deepcopy(scene)
    blocks = [row for row in result['arrows'] if row.get('kind') == 'block_arrow']
    if not blocks:
        return result, {'protocol': PROTOCOL, 'bound': [], 'unresolved': []}
    with ZipFile(pptx) as archive:
        root = etree.fromstring(archive.read('ppt/slides/slide1.xml'))
    leaves = []
    for node in root.iter('{'+P+'}sp'):
        ident = node.find('{'+P+'}nvSpPr/{'+P+'}cNvPr')
        if ident is not None:
            leaves.append((ident.get('name', ''), node.find('{'+P+'}spPr')))
    bound, unresolved = [], []
    for row in blocks:
        row.setdefault('style', {}).pop('shape_parts', None)
        native = [(name, observed_fill(props)) for name, props in leaves if matches_name(name, row['id'])]
        if not native or any(fill is None for name, fill in native):
            unresolved.append({'id': row['id'], 'reason': 'native_fill_unavailable_or_transformed'})
            continue
        # The source-occlusion consumer accepts only one uniform opaque face.
        # Preserve all observed leaf fills so it can abstain for mixed faces.
        row.setdefault('style', {})['shape_parts'] = {name: fill for name, fill in native}
        bound.append({'id': row['id'], 'native_leaves': [name for name, fill in native]})
    return result, {'protocol': PROTOCOL, 'pptx_sha256': sha(Path(pptx)),
                    'bound': bound, 'unresolved': unresolved, 'native_geometry_and_paint_unchanged': True}
