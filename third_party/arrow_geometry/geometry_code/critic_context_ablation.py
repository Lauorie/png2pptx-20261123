"""Isolated 2x2 geometry-review ablation; never imported by the main pipeline."""
import copy
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import geometry_critic as critic
from transport.common import read_json as read, write_json as write

VARIANTS = {
    'A_local_current': (False, False),
    'B_context_current': (True, False),
    'C_local_structural': (False, True),
    'D_context_structural': (True, True),
    'E_local_structural_boundary': (False, True),
    'F_local_current_tolerance': (False, False),
}


def structural_prompt(original):
    """Change review scope only, keeping coordinates and output protocol."""
    q = copy.deepcopy(original)
    q['task'] = 'Review visible structural or clearly wrong geometric defects of the indicated paths. Accept an already usable reconstruction.'
    q['instructions'] = [
        'CLEAN is the source. CURRENT red paths and blue labels are annotations. Review only the indicated target; do not judge color, width, dashes or head type.',
        'Refine only a clearly visible wrong connection, missing or spurious bend, missing part of this target, wrong endpoint destination, or unmistakable path displacement. Keep independent crossings separate; do not infer hidden connections.',
        'For dense line bands, check the represented routes and visible connectivity, without enumerating all possible edges or making every endpoint pixel-exact. Small offsets that preserve the same visible attachment do not need refinement. For isolated paths, preserve the actual route and endpoint vicinity; fix clear errors.',
        'Keep usable points and paths unchanged. Do not refine uncertain tiny differences or anti-aliasing. A visible sharp corner must not become a diagonal shortcut or a smooth curve.',
        'All output coordinates are pixels in CLEAN, whose width and height are size; they are not normalized 0-1000. line has two node IDs; curve has at least three ordered points on the curve. Keep route R1; multiple paths are allowed.',
        'Return accept immediately if there is no clear defect requiring correction. For refine, give one short visible reason and complete corrected geometry. Use uncertain only if the target cannot be judged from the available evidence; do not guess.',
    ]
    return q


def global_crop_bbox(frame, view, context_size):
    """Map half-open pixel boundaries, respecting saved pixel-center transforms."""
    left, top, right, bottom = frame['bbox']
    corners = np.array([[left-.5, top-.5, 1], [right-.5, bottom-.5, 1]])
    global_edges = (corners @ np.asarray(view['view_to_global']).T)[:, :2] + .5
    scale = np.asarray(context_size, float) / np.asarray(view['global_size'], float)
    return (global_edges * scale).reshape(-1).tolist()


def prepare_context(full_image, frame, view, out, long_side=1080):
    out = Path(out)
    full = Image.open(full_image).convert('RGB')
    if tuple(view['global_size']) != full.size:
        raise ValueError('Global source size does not match saved transform')
    scale = min(1., long_side/max(full.size))
    size = tuple(max(1, round(x*scale)) for x in full.size)
    image = full.resize(size, Image.Resampling.LANCZOS)
    box = global_crop_bbox(frame, view, size)
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = box
    # Short corner marks keep the global figure readable; no masks or fitted lines.
    arm = max(3., min(14., (x1-x0)/3, (y1-y0)/3))
    for x, y, dx, dy in [(x0,y0,1,1),(x1,y0,-1,1),(x0,y1,1,-1),(x1,y1,-1,-1)]:
        draw.line([(x+dx*arm,y),(x,y),(x,y+dy*arm)], fill='#d800a8', width=2)
    image.save(out/'context.png')
    metadata = dict(size=list(size), clean_bbox=[round(v, 3) for v in box])
    write(out/'context_frame.json', dict(**metadata, original_size=list(full.size), source_view=view))
    return metadata


def build(original, inputs, variant, context):
    include_context, structural = VARIANTS[variant]
    q = structural_prompt(original) if structural else copy.deepcopy(original)
    images = [('CLEAN: unmarked source', Path(inputs)/'clean.png'),
              ('CURRENT: red geometry and blue point IDs on source', Path(inputs)/'geometry.png')]
    if variant == 'E_local_structural_boundary':
        q['instructions'].append(
            'Check target boundaries too: touching strokes may belong to different objects. Remove a segment that visibly belongs to a neighbor, '
            'even when it touches a valid endpoint. Stroke appearance may identify the target; do not estimate style parameters.')
    if variant == 'F_local_current_tolerance':
        q['instructions'].append(
            'Prioritize clearly wrong connections, missing or extra bends/segments, and endpoints belonging to the wrong visible stroke. '
            'Accept small pixel offsets that preserve the same visible route and endpoint attachment. '
            'Check the whole indicated path, including its first and last segment; touching neighboring strokes are not automatically part of this target.')
    if include_context:
        q['context'] = copy.deepcopy(context)
        q['instructions'].append(
            'CONTEXT is the whole source figure. Magenta corner marks locate CLEAN; context.clean_bbox uses CONTEXT pixels. '
            'Use CONTEXT only to understand surrounding connections. Judge detailed ink in CLEAN, and return every point in CLEAN pixels, never CONTEXT coordinates. '
            'CONTEXT does not authorize adding unrelated neighboring targets.')
        images.append(('CONTEXT: whole source; magenta corners locate CLEAN', Path(inputs)/'context.png'))
    return q, images


def render_answer(inputs, answer, out):
    """Use one unchanged view for before/after inspection and compile validation."""
    original = read(Path(inputs)/'current.json')
    result = answer.get('geometry', original)
    result = critic.validate(answer, read(Path(inputs)/'frame.json')['size']).get('geometry', original)
    out = Path(out)
    clean = Image.open(Path(inputs)/'clean.png').convert('RGB')
    frame = dict(bbox=[0,0,*clean.size], size=list(clean.size), scale=[1.,1.])
    critic.views(clean, result, frame, out/'after')
    parent = critic.transform(result, read(Path(inputs)/'frame.json'), inverse=True)
    write(out/'geometry_local.json', result)
    write(out/'geometry_parent.json', parent)
    before_parent = critic.transform(original, read(Path(inputs)/'frame.json'), inverse=True)
    old = {n[0]: np.asarray(n[1:]) for n in before_parent['nodes']}
    new = {n[0]: np.asarray(n[1:]) for n in parent['nodes']}
    moves = {k: float(np.linalg.norm(new[k]-old[k])) for k in old.keys() & new.keys()}
    def counts(doc):
        paths = doc['routes'][0]['paths']
        return dict(nodes=len(doc['nodes']), paths=len(paths), segments=sum(len(p['segments']) for p in paths))
    return dict(before=counts(original), after=counts(result), identical_geometry=result==original,
                matched_node_displacements_parent_pixels=moves,
                note='Node ID displacement is descriptive only; overlay review determines correctness.')
