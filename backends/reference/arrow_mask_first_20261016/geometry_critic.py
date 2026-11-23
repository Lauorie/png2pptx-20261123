"""Bounded geometry-only visual refinement in the existing point/path protocol.

Both GLM and optional mask paths enter through the same adapter and prompt.
No identity text, SAM overlays, styles, or previous reasoning reach the reviewer.
"""
import copy
import hashlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import segmented_geometry as sg
from transport.common import write_json as write, sha256

GLM_SOURCES = {'glm_segmented_points', 'glm_keypoints'}


def eligible(row, config):
    if not config.get('geometry_critic_enabled', False) or row.get('kind') == 'block_arrow':
        return False
    source = row.get('coordinate_source', '')
    return source in GLM_SOURCES or (config.get('geometry_critic_mask_enabled', False) and source.startswith('mask'))


def as_points(rows):
    """Import actual local geometry; accepting a review never recompiles it."""
    nodes, paths, shared = [], [], {}

    def node(xy, key=None):
        if key is not None and key in shared:
            return shared[key]
        name = 'n' + str(len(nodes) + 1)
        nodes.append([name, *map(float, xy)])
        if key is not None:
            shared[key] = name
        return name

    for i, row in enumerate(rows, 1):
        segments = []
        if row.get('model_segments') and row.get('model_nodes'):
            table = {n[0]: n[1:] for n in row['model_nodes']}
            for segment in row['model_segments']:
                refs = [node(table[n], ('model', str(n))) for n in segment['nodes']]
                segments.append(dict(kind=segment['kind'], nodes=refs))
        elif row.get('beziers'):
            last = None
            for bezier in row['beziers']:
                p = np.asarray(bezier, float)
                straight = np.allclose(p[1], p[0] + (p[3]-p[0])/3) and np.allclose(p[2], p[0] + 2*(p[3]-p[0])/3)
                t = np.linspace(0, 1, 2 if straight else 5)[:, None]
                q = (1-t)**3*p[0] + 3*(1-t)**2*t*p[1] + 3*(1-t)*t*t*p[2] + t**3*p[3]
                refs = [last or node(q[0]), *[node(x) for x in q[1:]]]
                segments.append(dict(kind='line' if straight else 'curve', nodes=refs))
                last = refs[-1]
        else:
            points = [row['start'], *row.get('via', []), row['end']]
            refs = []
            for xy in points:
                if not refs or not np.allclose(nodes[int(refs[-1][1:])-1][1:], xy):
                    refs.append(node(xy))
            segments = [dict(kind='line', nodes=[a, b]) for a, b in zip(refs, refs[1:])]
        paths.append(dict(id='p'+str(i), segments=segments))
    if not nodes or any(not p['segments'] for p in paths):
        raise ValueError('No drawable geometry to review')
    return dict(nodes=nodes, routes=[dict(id='R1', paths=paths)])


def frame_for(document, size, scale=2., long_side=1080, min_extent=25., source_view=None):
    """Crop in the parent image, with the minimum span defined in original pixels.

    source_view maps the parent CLEAN to the original figure. Without one, CLEAN
    itself is the original image. The span floor precedes context expansion and
    is independent of both the parent upsampling and the review display size.
    """
    if scale < 1 or long_side < 32 or min_extent <= 0:
        raise ValueError('Invalid geometry review crop configuration')
    original_per_local = np.ones(2, float)
    if source_view is not None:
        if source_view.get('size') is not None and tuple(source_view['size']) != tuple(size):
            raise ValueError('Geometry review source view does not match CLEAN size')
        matrix = np.asarray(source_view['view_to_global'], float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError('Invalid geometry review source transform')
        original_per_local = np.linalg.norm(matrix[:2, :2], axis=0)
        if np.any(original_per_local <= 0):
            raise ValueError('Degenerate geometry review source transform')
    min_extent_local = float(min_extent) / original_per_local
    xy = np.asarray([n[1:] for n in document['nodes']], float)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    center = (lo+hi)/2
    span = np.maximum(hi-lo, min_extent_local)*scale
    left = np.maximum(0, np.floor(center-span/2)).astype(int)
    right = np.minimum(size, np.ceil(center+span/2)+1).astype(int)
    if np.any(right <= left):
        raise ValueError('Geometry bbox is outside the source image')
    native = right-left
    factor = long_side/max(native)
    display = np.maximum(1, np.rint(native*factor)).astype(int)
    return dict(bbox=[*left.tolist(), *right.tolist()], size=display.tolist(),
                scale=(display/native).tolist(), bbox_scale=scale,
                original_size=list(size), min_extent=float(min_extent),
                min_extent_unit='original_image_pixels', min_extent_local=min_extent_local.tolist(),
                local_to_original_pixel_scale=original_per_local.tolist(),
                crop_extent_original_px=(native*original_per_local).tolist())


def transform(document, frame, inverse=False):
    answer = copy.deepcopy(document)
    offset = np.asarray(frame['bbox'][:2], float)
    scale = np.asarray(frame['scale'], float)
    for n in answer['nodes']:
        q = np.asarray(n[1:], float)
        q = (q+.5)/scale-.5+offset if inverse else (q-offset+.5)*scale-.5
        n[1:] = q.tolist()
    return answer


def geometry_rows(document, owner='R1'):
    return sg.geometry(document, {'R1': dict(id=owner, heads=['unknown', 'unknown'], line_style='unknown')})


def views(clean, document, frame, folder):
    """Two images only. No source styles determine the diagnostic geometry ink."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    crop = clean.crop(tuple(frame['bbox'])).resize(tuple(frame['size']), Image.Resampling.LANCZOS)
    crop.save(folder/'clean.png')
    local = transform(document, frame)
    overlay = crop.copy()
    draw = ImageDraw.Draw(overlay)
    rows = geometry_rows(local)
    for row in rows:
        points = [tuple(row['start']), *map(tuple, row.get('via', [])), tuple(row['end'])]
        draw.line(points, fill='#e31a1c', width=2)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    for name, x, y in local['nodes']:
        draw.ellipse((x-3, y-3, x+3, y+3), fill='#2058dc', outline='white')
        pos = (max(0, min(crop.width-28, x+5)), max(0, min(crop.height-16, y+4)))
        draw.text(pos, str(name), font=font, fill='#2058dc', stroke_width=2, stroke_fill='white')
    overlay.save(folder/'geometry.png')
    write(folder/'frame.json', frame)
    write(folder/'current.json', local)
    return local, [('CLEAN: unmarked source', folder/'clean.png'),
                   ('CURRENT: red geometry and blue point IDs on source', folder/'geometry.png')]


def prompt(current, size):
    return dict(task='Check ONLY the current line centerline geometry against CLEAN. Accept it or correct its points and paths.',
        size=list(size), current=current,
        instructions=[
            'CLEAN is source truth. CURRENT has red fitted lines and blue point labels; these annotations are not source ink. Check endpoints, bends and curves of the indicated target only. Do not add neighboring objects or judge color, width, dashes or head shape.',
            'Follow visible strokes. Keep real sharp corners; use curve only for visibly curved ink. Do not invent hidden connections or infer a start at a nearby box. A headed endpoint is at its visible tip.',
            'Use CLEAN pixel coordinates. line has two point IDs; curve has at least three ordered points ON the curve. Parts within one path meet. Multiple paths are allowed; crossings need not connect. Keep route R1. Add/remove/move points as needed for this target.',
            'For accept, return only decision. For refine, return one short reason and the complete corrected geometry. For unclear or clipped evidence, return uncertain and a short reason; do not guess.'],
        output_schema=dict(decision='accept|refine|uncertain', reason='short; omit for accept',
            geometry='only for refine: {nodes:[[id,x,y],...],routes:[{id:R1,paths:[{id,segments:[{kind:line|curve,nodes:[point IDs]}]}]}]}'))


def validate(raw, size):
    if not isinstance(raw, dict) or raw.get('decision') not in ('accept', 'refine', 'uncertain'):
        raise ValueError('Return decision accept, refine or uncertain')
    result = {'decision': raw['decision']}
    if raw['decision'] != 'accept':
        if not isinstance(raw.get('reason'), str) or not raw['reason'].strip():
            raise ValueError('Non-accept decision needs a short reason')
        result['reason'] = raw['reason']
    if raw['decision'] == 'refine':
        result['geometry'] = sg.validate(raw.get('geometry'), {'R1'}, size)
        for route in result['geometry']['routes']:
            if set(route)-{'id','paths','segments'}:
                raise ValueError('Geometry review must not change style or heads')
            for path in sg.drawable_paths(route):
                if set(path)-{'id','segments'}:
                    raise ValueError('Geometry review must not change style or heads')
    elif raw.get('geometry') is not None:
        raise ValueError('accept/uncertain cannot change geometry')
    return result


def refine_target(clean, rows, folder, model, config, cpu_slot=None, source_view=None):
    folder = Path(folder)
    limit = config.get('geometry_critic_max_refinements', 2)
    if type(limit) is not int or not 0 <= limit <= 2:
        raise ValueError('Geometry refinement budget must be 0, 1 or 2')
    source_view = source_view if source_view is not None else rows[0].get('geometry_frame')
    current = as_points(rows)
    write(folder/'before_rows.json', rows)
    write(folder/'before_geometry.json', current)
    state = dict(state='running', revisions=0, calls=0, accepted=False,
                 budget=limit, input_sources=sorted({r.get('coordinate_source', '') for r in rows}))
    write(folder/'state.json', state)
    for step in range(limit):
        round_dir = folder/f'revision_{step:02d}'
        with cpu_slot if cpu_slot is not None else nullcontext():
            frame = frame_for(current, clean.size, config.get('geometry_critic_context_scale', 2.),
                              config.get('geometry_critic_long_side', 1080),
                              config.get('geometry_critic_min_extent_original_px', 25.), source_view)
            local, images = views(clean, current, frame, round_dir/'input')
            q = prompt(local, frame['size'])
            write(round_dir/'input/images.json', [dict(label=l, path=str(p.resolve()), sha256=sha256(p)) for l, p in images])
        try:
            state['calls'] += 1
            answer = model.ask(round_dir/'request', q, images, lambda a: validate(a, frame['size']))
            write(round_dir/'answer.json', answer)
            if answer['decision'] != 'refine':
                state.update(state=answer['decision'], accepted=answer['decision']=='accept', reason=answer.get('reason'))
                break
            candidate = transform(answer['geometry'], frame, inverse=True)
            # Validate the complete candidate before changing the retained state.
            candidate = sg.validate(candidate, {'R1'}, clean.size)
            geometry_rows(candidate)
            current = candidate
            state['revisions'] += 1
            write(round_dir/'after_geometry.json', current)
            with cpu_slot if cpu_slot is not None else nullcontext():
                views(clean, current, frame, round_dir/'after')
            state.update(state='refined', accepted=False)
        except Exception as exc:
            state.update(state='failed', error=str(exc), retained_previous=True)
            write(round_dir/'error.json', dict(error=str(exc), retained_previous=True))
            break
        finally:
            write(folder/'state.json', state)
    else:
        state.update(state='budget_exhausted' if limit else 'disabled', accepted=False)

    after = copy.deepcopy(rows)
    if state['revisions']:
        owner = rows[0].get('geometry_owner_id', rows[0].get('physical_target_id', rows[0]['id']))
        base = copy.deepcopy(rows[0])
        for key in ('style', 'points', 'route_path', 'node_ids', 'shared_nodes', 'model_nodes', 'model_segments'):
            base.pop(key, None)
        base.update(id=owner, heads=['unknown', 'unknown'], line_style='unknown',
                    start_head=False, end_head=False, start_head_type=None, end_head_type=None)
        after = sg.geometry(current, {'R1': base})
        for row in after:
            row.update(geometry_refinement=str(folder.resolve()),
                       geometry_input_sources=state['input_sources'], head_source='deferred_to_fixed_geometry_style')
            refs = list(dict.fromkeys(n for s in row['model_segments'] for n in s['nodes']))
            row['node_ids'] = [owner+':critic:'+str(n) for n in refs]
            row['shared_nodes'] = [owner+':critic:'+str(n) for n in row['shared_nodes']]
    write(folder/'after_geometry.json', current)
    write(folder/'after_rows.json', after)
    write(folder/'state.json', state)
    with cpu_slot if cpu_slot is not None else nullcontext():
        comparison = dict(nodes=as_points(rows)['nodes']+current['nodes'])
        final_frame = frame_for(comparison, clean.size, config.get('geometry_critic_context_scale', 2.),
                                config.get('geometry_critic_long_side', 1080),
                                config.get('geometry_critic_min_extent_original_px', 25.), source_view)
        views(clean, as_points(rows), final_frame, folder/'before')
        views(clean, current, final_frame, folder/'after')
    return after, state


def recover(clean, rows, supports, out, model, config, cpu_slot=None, source_view=None):
    """Common mainline hook, before style estimation and coordinate transport."""
    if not config.get('geometry_critic_enabled', False):
        return rows, supports, [], 0
    owners = {}
    for row in rows:
        owner = row.get('geometry_owner_id', row.get('physical_target_id', row['id']))
        owners.setdefault(owner, []).append(row)
    jobs = {owner: group for owner, group in owners.items() if all(eligible(r, config) for r in group)}
    output, new_supports, issues, calls = [], dict(supports), [], 0

    def work(item):
        owner, group = item
        key = hashlib.sha256(owner.encode()).hexdigest()[:12]
        folder = Path(out)/key
        write(folder/'target.json', dict(owner=owner, input_ids=[r['id'] for r in group]))
        try:
            return refine_target(clean, group, folder, model, config, cpu_slot, source_view)
        except Exception as exc:
            state = dict(state='failed', error=str(exc), retained_previous=True, calls=0)
            write(folder/'state.json', state)
            return copy.deepcopy(group), state

    with ThreadPoolExecutor(max_workers=max(1, config.get('geometry_critic_workers', 8))) as pool:
        results = dict(zip(jobs, pool.map(work, jobs.items())))
    for owner, group in owners.items():
        if owner not in results:
            output.extend(group)
            continue
        revised, state = results[owner]
        calls += state['calls']
        if not state.get('accepted'):
            issues.append(dict(stage='geometry_critic', target=owner, **state))
        available = [supports[r['id']] for r in group if supports.get(r['id']) is not None]
        support = np.logical_or.reduce(available) if available else None
        for row in revised:
            new_supports[row['id']] = support
        output.extend(revised)
    write(Path(out)/'summary.json', dict(targets=len(jobs), calls=calls,
          mask_enabled=config.get('geometry_critic_mask_enabled', False), issues=issues))
    return output, new_supports, issues, calls
