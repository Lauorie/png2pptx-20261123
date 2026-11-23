"""Identity from the original group boxes; inspect masks only after identity."""
import copy
from pathlib import Path

import numpy as np
from PIL import ImageDraw
import contracts
import evidence as ev
import protocol
import semantic_geometry
import target_locator
import visual
from transport.common import write_json as write, sha256 as sha


def build(crop, group, view, context_size, out, config=None):
    out = Path(out)
    config=config or {}
    boxes = target_locator.boxes(group, view)
    annotations=target_locator.configured_draw(crop, boxes, out / 'input/target_boxes.png',config)
    write(out/'input/box_display.json',annotations)
    q = dict(
        task='Identify ONLY the physical arrow/line targets indicated by this existing group of candidate boxes.',
        size=list(crop.size), context_size=list(context_size),
        crop_in_context=ev.crop_in_context(view, context_size), group_boxes=boxes,
        instructions=[
            protocol.SCOPE,
            'CLEAN is the local original. GROUP BOXES are the existing group targets; CONTEXT only explains their surroundings. '
            'Do not discover every arrow in CLEAN or treat the union of boxes as a request for all ink inside it. '
            'Confirm the intended physical targets from visible strokes. Merge duplicate boxes, split a mixed candidate, or exclude non-linework. '
            'Boxes may be partial or overlap unrelated objects; follow the same target to its actual visible ends.',
            'No local mask pixels or numerical axes are supplied at this identity stage. '
            'Do not explain SAM instances or make a one-to-one assignment for every box. '
            'A coherent dense band indicated by this group may be ONE network target; its later geometry may contain many independent paths. '
            'Do not assume all-to-all connections or include unrelated neighboring arrows.',
            'Return a local bbox and short visual locator for each confirmed target, not coordinates of its route. '
            'geometry=mask asks the NEXT stage to inspect local pixel evidence for a simple straight line, smooth curve or block arrow; '
            'it does not claim the mask is accurate. geometry=glm requests direct geometry for a complex polyline/network or ambiguous pixel evidence. '
            'No head types, widths, colors, dashes, point table or network decomposition are needed here.',
            'A followup is only a continuation of a confirmed target or a true shared branch outside CLEAN. '
            'Give context_bbox in CONTEXT pixels. Unrelated neighbors stay for other groups or the later omission pass. '
            'Return status=complete with targets=[] when this group contains only excluded shapes. '
            'Use uncertain only for an unresolved in-scope identity.',
        ],
        output_schema=dict(
            status='complete|uncertain',
            targets='[{id,bbox:[left,top,right,bottom],kind:straight|curve|polyline|network|block_arrow,geometry:mask|glm,locator:short visual identifier}]',
            followups='[{context_bbox:[left,top,right,bottom],description:short visible continuation}] or []',
        ),
    )
    display_note=target_locator.display_instruction(config)
    if display_note:q['instructions'].append(display_note)
    if config.get('identity_neutral_handoff'):
        q['instructions'].append(
            'source_boxes lists the input G IDs that helped locate this target; several boxes may refer to one target, '
            'and one box may contain several targets. This is a reference, not a mask assignment or proof of connectivity. '
            'locator should identify color, nearby label and visible attachment objects only. Do not put guessed coordinates, '
            'horizontal/vertical claims or turn sequences in locator. kind is provisional for routing; the next stage '
            'must determine the complete visible geometry from CLEAN. bbox is approximate; retain the whole visible target.')
        q['output_schema']['targets']='[{id,source_boxes:[G IDs],bbox:[left,top,right,bottom],kind:straight|curve|polyline|network|block_arrow,geometry:mask|glm,locator:short appearance identifier}]'
    images = [('CLEAN: local original', out / 'input/clean.png'),
              ('GROUP BOXES: only the current group targets; neighbors are context', out / 'input/target_boxes.png'),
              ('CONTEXT: whole figure, observation rectangle only', out / 'input/context.png')]
    return q, images


def validate(raw, size, context_size, source_ids=None):
    if not isinstance(raw, dict) or raw.get('status') not in ('complete', 'uncertain'):
        raise ValueError('Identity status must be complete or uncertain')
    targets = raw.get('targets')
    if not isinstance(targets, list):
        raise ValueError('targets must be a list')
    seen = set()
    for target in targets:
        if not isinstance(target, dict):
            raise ValueError('Each identity target must be an object')
        tid = target.get('id')
        if not isinstance(tid, str) or not tid or '/' in tid or tid in seen:
            raise ValueError('Unique nonempty target IDs without / required')
        seen.add(tid)
        contracts.bbox(target.get('bbox'), size)
        if target.get('kind') not in ('straight', 'curve', 'polyline', 'network', 'block_arrow'):
            raise ValueError('Invalid identity kind')
        if target.get('geometry') not in ('mask', 'glm'):
            raise ValueError('geometry must be mask or glm')
        if not isinstance(target.get('locator'), str) or not target['locator'].strip():
            raise ValueError('Short visual target locator required')
        if source_ids is not None:
            refs=target.get('source_boxes')
            if not isinstance(refs,list) or not refs or any(x not in source_ids for x in refs):
                raise ValueError('source_boxes must reference the supplied G candidates')
    followups = raw.get('followups', [])
    if not isinstance(followups, list):
        raise ValueError('followups must be a list')
    for item in followups:
        contracts.bbox(item.get('context_bbox'), context_size)
        if not isinstance(item.get('description'), str) or not item['description'].strip():
            raise ValueError('Followup description required')
    return dict(status=raw['status'], targets=copy.deepcopy(targets), followups=copy.deepcopy(followups))


def mask_prompt(target, size, info, proposals):
    """One already-confirmed target; the model selects usable measurements."""
    return dict(
        task='Choose usable mask geometry for this ONE confirmed target. Do not discover or re-identify its neighbors.',
        size=list(size), target=target,
        masks=[dict(id=m['id'], bbox=m['bbox'], proposal=m.get('proposal')) for m in info],
        measurements=[dict(id=k, masks=p['masks'], kind=p['kind'],
                           points=[p['start'], *p.get('via', []), p['end']]) for k, p in proposals.items()],
        instructions=[
            'CLEAN decides the full target. TARGET is its approximate bbox; MASKS + D shows optional numerical evidence. '
            'Use geometry=mask with a matching proposal only if it represents this complete target. '
            'Otherwise geometry=mask may select supporting masks plus complete endpoint/guide nodes from CLEAN. '
            'Do not shrink a shaft to a head-only mask or join unrelated strokes.',
            'Use geometry=glm, masks=[], proposal=null, nodes=[] when the evidence is unsuitable. '
            'An already-confirmed block_arrow uses geometry=mask with usable filled masks, or geometry=points with ordered closed contour nodes. '
            'Coordinates are local CLEAN pixels. Return exactly one route with the supplied target ID; no new identities.',
            'Keep heads=["unknown","unknown"] and line_style="unknown"; these are estimated later. '
            'For geometry=glm omit point coordinates; the next geometry request sees the confirmed target box.',
        ],
        output_schema=dict(nodes='[[id,x,y],...] or []',
                           routes='[{id:target.id,kind:straight|curve|polyline|block_arrow,geometry:mask|points|glm,masks:[M IDs],proposal:D ID or null,nodes:[IDs]}]'),
    )


def resolve(raw, labels, crop, proposals, candidates, info, out, model, config=None, source_boxes=()):
    """Return the existing routing interface without expanding group ownership."""
    routes, nodes, calls, issues = [], [], 0, []
    out = Path(out)
    config=config or {}
    for target in raw['targets']:
        tid = target['id']
        default = dict(id=tid, kind='polyline' if target['kind'] == 'network' else target['kind'],
                       geometry='glm', masks=[], proposal=None, nodes=[], network=None,
                       locator=target['locator'], target_bbox=target['bbox'],
                       heads=['unknown', 'unknown'], line_style='unknown',
                       sources=list(labels), source_linking='observation_context_only')
        default['identity_kind']=target['kind']
        if target.get('bbox_source'):default['target_bbox_source']=target['bbox_source']
        if target.get('source_boxes'):
            default['source_boxes']=[b for b in source_boxes if b['id'] in target['source_boxes']]
        if target['geometry'] == 'glm' and target['kind'] != 'block_arrow':
            routes.append(default)
            continue
        # Intersection is evidence retrieval, not acceptance or ownership.
        x0, y0, x1, y1 = target['bbox']
        selected = [c for c in candidates if c['mask'][max(0, int(np.floor(y0))):min(crop.height, int(np.ceil(y1))),
                                                        max(0, int(np.floor(x0))):min(crop.width, int(np.ceil(x1)))].any()]
        if not selected and target['kind'] != 'block_arrow':
            routes.append(default)
            continue
        mids = {c['id'] for c in selected}
        measurements = {k: p for k, p in proposals.items() if set(p['masks']) <= mids}
        selected_info = [m for m in info if m['id'] in mids]
        folder = out / 'target_mask_geometry' / f'T{len(routes)+1:03d}'
        (folder / 'input').mkdir(parents=True, exist_ok=True)
        target_locator.configured_draw(crop,[] if target.get('bbox_source')=='original_observation' else [target],folder/'input/target.png',config)
        visual.masks(crop, [(c['id'], c['mask']) for c in selected], folder / 'input/masks.png')
        q = mask_prompt(target, crop.size, selected_info, measurements)
        if target.get('bbox_source')=='original_observation':
            q['instructions'].append('No original candidate box locates this target. Its bbox records the whole observation only; '
                'the TARGET image intentionally has no target rectangle. Use the appearance locator and CLEAN; '
                'do not assign another connector because it lies in the observation.')
        images = [('CLEAN', out / 'input/clean.png'), ('CONFIRMED TARGET', folder / 'input/target.png'),
                  ('MASK evidence for this confirmed target', folder / 'input/masks.png')]

        def check(value):
            if not isinstance(value, dict) or len(value.get('routes', [])) != 1 or value['routes'][0].get('id') != tid:
                raise ValueError('Only the single confirmed target may be fitted')
            a = copy.deepcopy(value)
            r = a['routes'][0]
            r.update(sources=[tid], heads=['unknown', 'unknown'], line_style='unknown',
                     network=None, locator=target['locator'])
            a.update(excluded=[], uncertain=[], followups=[])
            return semantic_geometry.validate(a, crop.size, [tid], measurements, mids, selected)

        calls += 1
        try:
            answer = model.ask(folder / 'request', q, images, check)
        except Exception as exc:
            write(folder / 'error.json', dict(error=str(exc), target=target))
            issues.append(dict(stage='target_mask_geometry', target=tid, error=str(exc)))
            if target['kind'] != 'block_arrow':
                routes.append(default)
            continue
        write(folder / 'answer.json', answer)
        route = answer['routes'][0]
        rename = {str(n[0]): f'{tid}:{n[0]}' for n in answer['nodes']}
        nodes.extend([[rename[str(n)], x, y] for n, x, y in answer['nodes']])
        route['nodes'] = [rename[str(n)] for n in route.get('nodes', [])]
        route.update(sources=list(labels), source_linking='observation_context_only',
                     target_bbox=target['bbox'], locator=target['locator'])
        route['identity_kind']=default['identity_kind']
        if target.get('bbox_source'):route['target_bbox_source']=target['bbox_source']
        if 'source_boxes' in default:route['source_boxes']=default['source_boxes']
        routes.append(route)
        write(folder / 'input/images.json', [dict(label=l, path=str(p.resolve()), sha256=sha(p)) for l, p in images])
    answer = dict(nodes=nodes, routes=routes, followups=raw['followups'],
                  excluded=list(labels) if not routes and raw['status'] == 'complete' else [],
                  uncertain=list(labels) if raw['status'] == 'uncertain' and not routes else [],
                  observation_incomplete=raw['status'] == 'uncertain' or bool(issues),
                  source_linking='observation_context_only', identity_targets=raw['targets'])
    write(out / 'mask_geometry_calls.json', dict(calls=calls, identity_saw_local_masks=False))
    return answer, calls, issues
