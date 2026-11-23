"""Source identities precede branch matching; native paths remain the edit units.

Only shape-led linework is in scope. Source inspection is cached independently
of predictions. A local batch replaces many pairwise votes, and every candidate
receives an explicit receipt, including those left outside the shape scope.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .compose import selection_subset, subset, merge
from .handoff import json_part_id

PROTOCOL = 'source_object_native_ownership.15'


def bounds(row):
    points = [row['points'], *[np.asarray(p) for p in row.get('attachment_bounds', [])]]
    p = np.concatenate(points)
    return np.r_[p.min(0), p.max(0)]


def scopes(rows, size, scene=None):
    """Spatial indexing locates inspection regions, never duplicate identity."""
    shapes = [r for r in rows if r['layer'] == 'shape' and 'segment' not in r]
    span = max(128., min(256., max(size) / 8))
    buckets = {}
    for row in shapes:
        p = bounds(row); center = (p[:2] + p[2:]) / 2
        key = tuple(np.floor(center / span).astype(int))
        buckets.setdefault(key, []).append(row)
    result = []; parts = (scene or {}).get('parts', {})
    for key, members in sorted(buckets.items()):
        boxes = np.asarray([bounds(r) for r in members]); low = boxes[:, :2].min(0); high = boxes[:, 2:].max(0)
        # Context permits inaccurate endpoints to find their actual source
        # occurrence. It does not require two predictions to overlap.
        pad = max(24., min(96., .2 * max(high - low)))
        box = np.r_[np.maximum(0, low - pad), np.minimum(size, high + pad + 1)]
        keys = {key for row in members for key in
                row.get('endpoint_nodes', ()) if key in parts}
        node_boxes = [np.asarray(parts[k]['bbox_2d'], float) for k in keys]
        if node_boxes:
            node_boxes = np.asarray(node_boxes)
            box = np.r_[np.maximum(0, np.minimum(box[:2], node_boxes[:, :2].min(0)-8)),
                        np.minimum(size, np.maximum(box[2:], node_boxes[:, 2:].max(0)+9))]
        result.append({'id': 'scope_%03d' % len(result), 'box': box, 'rows': list(members), 'context_keys': keys})
    # A route touching several inspection regions must be compared with ALL
    # their shape recoveries in one source ledger. Nearest-region assignment
    # would silently hide a competitor on the other side of a bucket boundary.
    parent = list(range(len(result)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    assigned = []; outside = []
    for row in rows:
        if row['layer'] != 'arrows':
            continue
        b = bounds(row); choices = []
        for index, region in enumerate(result):
            q = region['box']
            if np.any(b[:2] > q[2:]) or np.any(q[:2] > b[2:]):
                continue
            # A crossing or a large bounding box is context, not a reason to
            # join disconnected line reviews into one enormous network call.
            # This only locates competitors; source pixels still decide them.
            from .handoff import distances
            related=False
            for shape in region['rows']:
                pair=shape.get('endpoint_nodes')
                if pair and row.get('endpoint_nodes') and set(pair)==set(row['endpoint_nodes']):
                    related=True;break
                tolerance=max(12.,min(32.,.12*shape['length']))
                shape_coverage=float(np.mean(distances(shape['points'],[row['points']])<tolerance))
                arrow_coverage=float(np.mean(distances(row['points'],[shape['points']])<tolerance))
                if min(shape_coverage,arrow_coverage)>=.5 or (max(shape_coverage,arrow_coverage)>=.8 and min(shape_coverage,arrow_coverage)>=.2):
                    related=True;break
            if not related:continue
            choices.append(index)
        if choices:
            for index in choices[1:]:
                parent[find(index)] = find(choices[0])
            assigned.append((row, choices[0]))
        else:
            outside.append(row)
    groups = {}
    for index, region in enumerate(result):
        groups.setdefault(find(index), []).append(region)
    merged = {}
    for ident, (key, regions) in enumerate(sorted(groups.items())):
        boxes = np.asarray([r['box'] for r in regions])
        merged[key] = {'id': 'scope_%03d' % ident,
                       'box': np.r_[boxes[:, :2].min(0), boxes[:, 2:].max(0)],
                       'rows': [row for r in regions for row in r['rows']],
                       'context_keys': set().union(*(r['context_keys'] for r in regions))}
    for row, index in assigned:
        region = merged[find(index)]; region['rows'].append(row); b = bounds(row)
        region['box'] = np.r_[np.maximum(0, np.minimum(region['box'][:2], b[:2]-16)),
                             np.minimum(size, np.maximum(region['box'][2:], b[2:]+17))]
    return list(merged.values()), outside


def validate_inventory(raw, size, assets, *, allow_partial=False):
    if not isinstance(raw, dict) or not isinstance(raw.get('objects'), list):
        raise ValueError('Return source objects and uncertainty')
    seen = set()
    for obj in raw['objects']:
        if not isinstance(obj, dict):
            raise ValueError('Invalid source object')
        ident = obj.get('id'); b = np.asarray(obj.get('bbox', []), float)
        if not isinstance(ident, str) or not ident or ident in seen:
            raise ValueError('Source object IDs must be unique')
        seen.add(ident)
        original_box = b.copy()
        if b.shape == (4,):
            b = np.r_[np.minimum(b[:2], b[2:]), np.maximum(b[:2], b[2:])]
        if (b.shape != (4,) or not np.isfinite(b).all() or np.any(b[:2] < -1)
                or np.any(b[2:] > np.array(size)) or np.any(b[2:] < b[:2]) or not np.any(b[2:] > b[:2])):
            raise ValueError('Source '+str(ident)+' bbox '+str(obj.get('bbox'))+' must use CLEAN pixel centers inside '+str(list(size)))
        clipped = np.clip(b, 0, np.tile(np.asarray(size)-1, 2))
        if not np.array_equal(clipped, original_box):
            reason = 'one_pixel_canvas_edge_convention' if np.array_equal(b, original_box) else 'locator_corner_order'
            obj['source_bbox_normalization'] = {'before': original_box.tolist(), 'after': clipped.tolist(), 'reason': reason}
            obj['bbox'] = clipped.tolist()
        if obj.get('kind') not in ('line', 'arrow', 'bundle', 'frame', 'image_content', 'uncertain'):
            raise ValueError('Unknown source object role')
        if not isinstance(obj.get('description'), str) or not obj['description'].strip():
            raise ValueError('Describe actual visible source content')
        if allow_partial:
            if obj['kind']=='image_content' and obj.get('asset_id') not in assets:
                raw.setdefault('inventory_normalizations',[]).append({'source_id':ident,
                    'proposed_kind':'image_content','proposed_asset_id':obj.get('asset_id'),
                    'reason':'No registered image owner; this source cannot authorize deletion'})
                obj.update(kind='uncertain',asset_id='',visible_count=None,count_status='uncertain',
                    review_status='unregistered_image_owner_preserved')
            count = obj.get('visible_count'); certainty = obj.get('count_status', 'exact')
            members = obj.get('members')
            if isinstance(members, list) and all(isinstance(m, dict) and isinstance(m.get('description'), str) for m in members):
                members = [m['description'] for m in members]; obj['members'] = members
            multiple = obj['kind'] in ('line', 'arrow') and type(count) is int and count > 1
            unverified = (obj['kind'] == 'bundle' and certainty == 'exact' and
                (not isinstance(members, list) or len(members) != count or
                 not all(isinstance(m, str) and m.strip() for m in members) or len(set(members)) != count))
            if multiple or unverified or certainty == 'uncertain' and count is not None:
                obj['source_count_normalization'] = {'proposed_count': count, 'proposed_kind': obj['kind'],
                    'reason': 'Physical members were not independently enumerated; count remains unverified'}
                obj.update(visible_count=None, count_status='uncertain')
                if multiple:
                    obj['kind'] = 'bundle'
        count = obj.get('visible_count'); certainty = obj.get('count_status', 'exact')
        if certainty not in ('exact', 'uncertain'):
            raise ValueError('Source counts must distinguish observed members from uncertainty')
        if certainty == 'uncertain':
            if count is not None:
                raise ValueError('Uncounted source paths require visible_count=null; never infer all-to-all counts')
        elif type(count) is not int or count < 1:
            raise ValueError('Count visible physical paths, not dash fragments')
        if obj['kind'] in ('line', 'arrow') and certainty == 'exact' and count != 1:
            raise ValueError('A line/arrow source ID names one physical connection; use separate IDs for different endpoints')
        if obj['kind'] == 'bundle' and certainty == 'exact':
            members = obj.get('members')
            if (not isinstance(members, list) or len(members) != count or
                    not all(isinstance(m, str) and m.strip() for m in members) or len(set(members)) != count):
                raise ValueError('An exact bundle count requires an explicit source description for every physical member; otherwise return count_status=uncertain and visible_count=null')
        if obj['kind'] == 'image_content' and obj.get('asset_id') not in assets:
            raise ValueError('Image-owned content needs an existing asset ID from '+str(sorted(assets))+'; unregistered native diagram content is not image-owned')
    return raw


def validate_mapping(raw, objects, aliases, asset_ids, prior=None, allow_omissions=False):
    if not isinstance(raw, dict) or not isinstance(raw.get('objects'), list) or not isinstance(raw.get('unassigned'), list):
        raise ValueError('Account for source objects and every native candidate')
    prior = prior or {}
    source = {o['id']: o for o in objects}; seen = set(); accounted = set(); decisions = {}
    def preserve(item,reason):
        raw.setdefault('mapping_normalizations',[]).append({'source_id':item['id'],
            'reason':reason,'proposed_candidate_ids':item.get('candidate_ids'),
            'proposed_keep_ids':item.get('keep_ids'),'proposed_drop_prior_ids':item.get('drop_prior_ids',[])})
        known=[]
        for field in ('candidate_ids','keep_ids'):
            values=item.get(field,[])
            if isinstance(values,list):known.extend(k for k in values if isinstance(k,str) and k in aliases)
        known=list(dict.fromkeys(known))
        item.update(candidate_ids=known,keep_ids=known,drop_prior_ids=[],status='uncertain',
            review_status='invalid_source_decision_preserved',reason=reason+'; native content preserved')
    for item_index,item in enumerate(raw['objects']):
        if not isinstance(item, dict):
            raise ValueError('Invalid source mapping')
        ident = item.get('id')
        if ident not in source or ident in seen:
            raise ValueError('Return each frozen source ID exactly once')
        seen.add(ident)
        candidates = item.get('candidate_ids'); keep = item.get('keep_ids')
        dropped_prior = item.get('drop_prior_ids', [])
        if (not isinstance(dropped_prior, list) or not all(isinstance(k, str) for k in dropped_prior)
                or len(set(dropped_prior)) != len(dropped_prior) or not set(dropped_prior) <= set(prior.get(ident, []))):
            if not allow_omissions:raise ValueError('Only replace existing survivors of the same frozen source ID')
            preserve(item,'Invalid prior replacement cannot authorize deletion')
            candidates=item['candidate_ids'];keep=item['keep_ids'];dropped_prior=[]
        if (not isinstance(candidates, list) or not isinstance(keep, list)
                or not all(isinstance(k,str) for k in candidates+keep)
                or len(set(candidates)) != len(candidates) or len(set(keep)) != len(keep)
                or not set(candidates) <= set(aliases) or not set(keep) <= set(candidates)):
            if not allow_omissions:raise ValueError('Only select listed native candidates assigned to this source object')
            preserve(item,'Invalid native ID relation cannot authorize deletion')
            candidates=item['candidate_ids'];keep=item['keep_ids'];dropped_prior=[]
        if (item.get('status') not in ('resolved', 'incomplete', 'uncertain')
                or not isinstance(item.get('reason'), str) or not item['reason'].strip()):
            raise ValueError('Record unresolved source evidence explicitly')
        if source[ident].get('count_status') == 'uncertain' and item['status'] == 'resolved':
            if not allow_omissions:
                raise ValueError('An uncounted source bundle remains incomplete even when a usable native survivor is selected')
            item.update(status='incomplete', review_status='source_count_unverified')
        if item['status'] == 'uncertain' and set(keep) != set(candidates):
            if not allow_omissions:raise ValueError('Source uncertainty cannot authorize line deletion')
            preserve(item,'Source uncertainty cannot authorize line deletion')
            candidates=item['candidate_ids'];keep=item['keep_ids'];dropped_prior=[]
        if dropped_prior and item['status'] == 'uncertain':
            if not allow_omissions:raise ValueError('Uncertainty cannot delete a prior native survivor')
            preserve(item,'Uncertainty cannot delete a prior native survivor')
            candidates=item['candidate_ids'];keep=item['keep_ids'];dropped_prior=[]
        if source[ident]['kind'] == 'image_content' and isinstance(asset_ids, dict):
            asset = asset_ids[source[ident]['asset_id']]
            q = np.asarray(asset['source_box_half_open'], float)
            for candidate in set(candidates) - set(keep):
                from .image_line_scope import contained
                if not contained(aliases[candidate],asset):
                    if not allow_omissions:raise ValueError('An image cannot consume an external native connection')
                    preserve(item,'An image cannot consume an external native connection')
                    candidates=item['candidate_ids'];keep=item['keep_ids'];dropped_prior=[]
                    break
        image_replacement = source[ident].get('asset_id') in asset_ids and source[ident]['kind']=='image_content'
        if not image_replacement and not keep and not set(prior.get(ident, []))-set(dropped_prior):
            if candidates or dropped_prior:
                if not allow_omissions:raise ValueError('Native deletion requires a surviving representation of the same source')
                preserve(item,'No surviving native representation witnesses this source deletion')
                candidates=item['candidate_ids'];keep=item['keep_ids'];dropped_prior=[]
            elif item['status'] == 'resolved':
                if not allow_omissions:raise ValueError('A visible source connection with no recovery is incomplete')
                item.update(status='incomplete',review_status='visible_source_has_no_survivor')
        for candidate in candidates:
            decision = candidate in keep
            if candidate in decisions and decisions[candidate] != decision:
                if not allow_omissions:raise ValueError('One native object cannot receive contradictory ownership')
                for previous in raw['objects'][:item_index+1]:
                    if candidate in previous.get('candidate_ids',[]):
                        previous['keep_ids']=list(dict.fromkeys([*previous['keep_ids'],candidate]))
                        previous.update(status='incomplete',review_status='shared_native_conflict_preserved')
                decision=True
            decisions[candidate] = decision
        accounted.update(candidates)
    missing_source = set(source)-seen
    if missing_source:
        if not allow_omissions:
            raise ValueError('Missing frozen source objects: '+','.join(sorted(missing_source)))
        raw['objects'].extend({'id': ident, 'candidate_ids': [], 'keep_ids': [], 'status': 'incomplete',
            'reason': 'No explicit mapping receipt was returned for this frozen source object',
            'review_status': 'model_omitted_source'} for ident in sorted(missing_source))
    unassigned = set(); accepted_unassigned = []
    for item in raw['unassigned']:
        if not isinstance(item, dict):
            raise ValueError('Invalid unmatched candidate receipt')
        ident = item.get('candidate_id'); role = item.get('classification')
        if allow_omissions and ident in aliases and (ident in accounted or ident in unassigned):
            raw.setdefault('mapping_normalizations', []).append({'candidate_id': ident,
                'reason': 'Redundant unmatched receipt for an already accounted native candidate', 'receipt': item})
            if ident in accounted and role == 'uncertain':
                for obj in raw['objects']:
                    if ident in obj['candidate_ids']:
                        obj['keep_ids'] = list(dict.fromkeys([*obj['keep_ids'], ident])); obj['status'] = 'incomplete'
            elif ident in unassigned:
                previous = next(u for u in accepted_unassigned if u['candidate_id'] == ident)
                if previous['classification'] != role:
                    previous.update(classification='uncertain', reason='Unmatched native classifications conflict; content preserved')
            continue
        if ident not in aliases or ident in accounted or ident in unassigned:
            raise ValueError('Account for an unmatched native candidate exactly once')
        if (role not in ('context', 'unsupported', 'uncertain', 'image_owned')
                or not isinstance(item.get('reason'), str) or not item['reason'].strip()):
            raise ValueError('Explain the source relation of unmatched native content')
        if role == 'image_owned' and item.get('asset_id') not in asset_ids:
            if not allow_omissions:raise ValueError('Image ownership requires a registered asset')
            raw.setdefault('mapping_normalizations',[]).append({'candidate_id':ident,
                'proposed_classification':role,'proposed_asset_id':item.get('asset_id'),
                'reason':'Unregistered image ownership cannot authorize deletion; content preserved'})
            item.update(classification='uncertain',asset_id='',
                review_status='unregistered_image_owner_preserved',
                reason='No registered image owner; native content preserved. '+item['reason'])
        witness = item.get('duplicate_of')
        if isinstance(witness, dict):
            target = next((obj for obj in raw['objects'] if obj['id'] == witness.get('source_id')), None)
            refs = witness.get('candidate_ids')
            surviving = (set(target['keep_ids']) | (set(prior.get(target['id'], []))-
                         set(target.get('drop_prior_ids', [])))) if target else set()
            if (target and target['status'] != 'uncertain' and isinstance(refs, list) and refs
                    and all(isinstance(ref, str) for ref in refs) and ident not in refs
                    and set(refs) <= surviving):
                target['candidate_ids'].append(ident)
                target.setdefault('duplicate_witnesses', []).append({'candidate_id': ident, **witness})
                accounted.add(ident)
                raw.setdefault('mapping_normalizations', []).append({'candidate_id': ident,
                    'source_id': target['id'], 'reason': 'Explicit surviving native witness binds the duplicate to its source',
                    'duplicate_of': witness})
                continue
        if role == 'unsupported':
            item['review_status']='no_physical_replacement_witness_preserved'
        unassigned.add(ident)
        accepted_unassigned.append(item)
    raw['unassigned'] = accepted_unassigned
    missing_native = set(aliases)-accounted-unassigned
    if missing_native:
        if not allow_omissions:
            raise ValueError('Every supplied native candidate needs a source receipt; missing: '+','.join(sorted(missing_native)))
        raw['unassigned'].extend({'candidate_id': ident, 'classification': 'uncertain',
            'reason': 'Model omitted this native candidate; it remains unreviewed and is preserved',
            'review_status': 'model_omitted_native'} for ident in sorted(missing_native))
    return raw


def summary(row, alias, box, scale):
    p = row['points']; indices = np.unique(np.linspace(0, len(p) - 1, min(9, len(p))).astype(int))
    points = (p[indices] - np.asarray(box[:2]) + .5) * scale - .5
    return {'id': alias, 'branch': row['layer'], 'points': np.round(points, 2).tolist(),
            'heads': row.get('heads'), 'style': {k:v for k,v in row['style'].items()
                if k in ('color','stroke','fill','width_px','dash','fill_opacity')},
            'native_ref': row['refs'], 'closed': bool(row.get('closed')),
            'source_endpoint_context':row.get('source_endpoint_context',[])}


def detail_coordinates(content, crop_box, size):
    """All model-facing geometry uses the pixels of the supplied detail images.

    Canonical source identities and the parent-view ledger remain unchanged.
    Mapping returns IDs only, so no inverse transform is applied to decisions.
    """
    result = deepcopy(content)
    origin = np.asarray(crop_box[:2], float)
    scale = np.asarray(size, float) / (np.asarray(crop_box[2:]) - origin)
    def points(values):
        return ((np.asarray(values, float) - origin + .5) * scale - .5).round(4).tolist()
    def bbox(values):
        return np.asarray(points(np.asarray(values).reshape(2, 2))).reshape(4).tolist()
    for obj in result['source_objects']:
        obj['bbox'] = bbox(obj['bbox'])
    for candidate in result['candidates']:
        candidate['points'] = points(candidate['points'])
    for asset in result.get('image_owned_regions', []):
        asset['bbox'] = bbox(asset['bbox'])
    for obj in result.get('prior_source_selection', []):
        for candidate in obj['native']:
            candidate['points'] = points(candidate['points'])
    result['coordinate_frame'] = {
        'name': 'displayed_detail', 'size': list(size),
        'convention': 'inclusive pixel centers',
        'rule': 'All source bboxes and native points below use the supplied detail image pixels. Return canonical IDs only.'}
    result['detail_transform'] = {
        'parent_crop_box_half_open': [int(v) for v in crop_box], 'scale_xy': scale.tolist(),
        'pixel_center_rule': 'detail = (parent - crop_origin + 0.5) * scale - 0.5'}
    return result


def merge_mapping_receipt(target, row):
    """An uncertain residual batch cannot revoke earlier verified selections."""
    fields=('candidate_ids','keep_ids','drop_prior_ids','status','reason')
    history=target.setdefault('batch_review_statuses',[
        {key:deepcopy(target[key]) for key in fields if key in target}])
    history.append({key:deepcopy(row[key]) for key in fields if key in row})
    target['candidate_ids']=list(dict.fromkeys([*target['candidate_ids'],*row['candidate_ids']]))
    kept=[c for c in target['keep_ids'] if c not in row.get('drop_prior_ids',[])]
    target['keep_ids']=list(dict.fromkeys([*kept,*row['keep_ids']]))
    target.setdefault('duplicate_witnesses',[]).extend(row.get('duplicate_witnesses',[]))
    statuses={entry['status'] for entry in history}
    if statuses=={'resolved'}:
        target['status']='resolved'
    elif 'incomplete' in statuses or set(target['candidate_ids'])!=set(target['keep_ids']):
        # Each batch was already validated, including its surviving witnesses.
        # New uncertainty preserves its own candidates; it does not silently
        # turn a previous explicit replacement into an invalid group decision.
        target['status']='incomplete'
    else:
        target['status']='uncertain'
    target['reason']+='; residual: '+row['reason']


def select_mapping(runtime, local, prompt, images, objects, aliases, assets, box, scale):
    """Bound source targets; spatial indexing cannot silently discard a native.

    A final batch accounts for every unassigned native against ALL frozen
    source IDs. Shared native paths keep a conservative conflict receipt.
    """
    mapper = runtime.for_stage(reasoning_effort='low') if hasattr(runtime, 'for_stage') else runtime
    if len(aliases) <= 64 and len(objects) <= 8:
        return mapper.call(local/'mapping_calls', 'select',
            'Use frozen SOURCE identities. Return only JSON; no architectural inference.', prompt, images,
            lambda d: validate_mapping(d, objects, aliases, assets, allow_omissions=True), 24576), 1
    descriptions = {c['id']: c for c in prompt['candidates']}
    indexed = {}
    for obj in objects:
        b = np.asarray(obj['bbox'], float); pad = max(16., .08*max(b[2:]-b[:2]))
        eligible = []
        for key, row in aliases.items():
            points = (row['points']-np.asarray(box[:2])+.5)*scale-.5
            fraction = np.mean(np.all((points >= b[:2]-pad)&(points <= b[2:]+pad), axis=1))
            if fraction >= .5:
                eligible.append(key)
        indexed[obj['id']] = set(eligible)
    groups = []; current = []; included = set()
    for obj in objects:
        union = included | indexed[obj['id']]
        if current and (len(union) > 64 or len(current) >= 4):
            groups.append((current, included)); current = []; included = set()
        current.append(obj); included |= indexed[obj['id']]
    if current:
        groups.append((current, included))
    def request(index, targets, keys, residual=False, previous=None):
        supplied = dict(list((k, aliases[k]) for k in aliases if k in keys)[:64])
        prior = {k: v['keep_ids'] for k, v in (previous or {}).items()}
        if previous is not None:
            from .handoff import distances
            fresh=[aliases[k] for k in supplied]
            visible_prior={}
            for ident,kept in prior.items():
                ranked=[]
                for candidate in kept:
                    row=aliases[candidate]
                    score=min((float(np.quantile(distances(row['points'],[r['points']]),.6)) for r in fresh),default=float('inf'))
                    if score<=max(16.,.1*row['length']):ranked.append((score,candidate))
                visible_prior[ident]=[c for score,c in sorted(ranked)[:32]]
            prior=visible_prior
        content = {**prompt, 'source_objects': targets,
                   'candidates': [descriptions[k] for k in supplied],
                   'scope_rule': 'Only listed C IDs are supplied in this batch. Other labels in the native context images are context. '
                                 'Account for every supplied C. Other source targets are processed separately; never infer a new source ID.'}
        if previous is not None:
            content['prior_source_selection'] = [
                {'id': k, 'kept_ids': prior.get(k,[]),
                 'native': [{'id': c, 'branch': descriptions[c]['branch'],
                             'points': [descriptions[c]['points'][0], descriptions[c]['points'][-1]],
                             'style': descriptions[c]['style']} for c in prior.get(k,[])]}
                for k, row in previous.items()]
            content['rules'] = [*prompt['rules'],
                'Prior native survivors already recover these SAME frozen source objects. Compare fresh candidates with them; '
                'never add a second version. Return drop_prior_ids per source if a fresh survivor replaces prior IDs. '
                'Keep prior IDs implicitly unless explicitly replaced. A resolved source may keep no fresh IDs when its prior survivor is complete.']
            content['output'] = {**prompt['output'], 'objects': [{**prompt['output']['objects'][0], 'drop_prior_ids': []}]}
        batch_images = images
        if images and targets:
            boxes = np.asarray([o['bbox'] for o in targets], float)
            low, high = boxes[:, :2].min(0), boxes[:, 2:].max(0)
            pad = max(16., .08*max(high-low))
            with Image.open(images[0][1]) as clean:
                crop_box = tuple(np.r_[np.maximum(0, np.floor(low-pad)), np.minimum(clean.size, np.ceil(high+pad+1))].astype(int))
            detail = local/('residual_views' if residual else 'mapping_views')/('%03d' % index)
            detail.mkdir(parents=True, exist_ok=True)
            batch_images = []
            display_size = None
            for view_index, (label, path) in enumerate(images):
                if 'candidate ID labels' in label:continue
                with Image.open(path) as opened:
                    crop = opened.convert('RGB').crop(crop_box)
                ratio = min(3., 1400/max(crop.size))
                size = tuple(max(2, round(v*ratio)) for v in crop.size)
                crop = crop.resize(size, Image.Resampling.LANCZOS)
                if display_size is not None and display_size != size:
                    raise ValueError('Source mapping views have different coordinate grids')
                display_size = size
                target = detail/('%02d.png' % view_index); crop.save(target)
                batch_images.append((label+'; SOURCE-target detail', target))
                branch='shape' if label.startswith('SHAPE:') else 'arrows' if label.startswith('ARROWS:') else None
                if branch:
                    locator=crop.copy();draw=ImageDraw.Draw(locator)
                    for alias,row in supplied.items():
                        if row['layer']!=branch:continue
                        point=np.array(descriptions[alias]['points'][len(descriptions[alias]['points'])//2])
                        display_scale = np.array(size) / (np.array(crop_box[2:])-crop_box[:2])
                        draw.text(tuple((point-np.array(crop_box[:2])+.5)*display_scale-.5),alias,fill='#A00090',stroke_width=1,stroke_fill='white')
                    target=detail/('%02d_ids.png'%view_index);locator.save(target)
                    batch_images.append((branch.upper()+': only supplied candidate ID labels',target))
            if display_size is not None:
                content = detail_coordinates(content, crop_box, display_size)
        def validate_batch(value):
            return validate_mapping(value, targets, supplied, assets, prior=prior, allow_omissions=True)
        result = mapper.call(local/('residual_calls' if residual else 'mapping_batches')/('%03d' % index),
            'select', 'Use frozen SOURCE identities; return concise JSON. Never count a template.', content, batch_images,
            validate_batch, min(12288,max(4096,160*len(supplied))))
        return result
    calls = 0; all_results = []
    with ThreadPoolExecutor(max_workers=max(1,min(2,int(getattr(runtime,'config',{}).get('mapping_workers',2))))) as pool:
        futures = [pool.submit(request, i, targets, keys) for i, (targets, keys) in enumerate(groups)]
        for future in futures:
            all_results.append(future.result()); calls += 1
    table = {o['id']: dict(o) for result in all_results for o in result['objects']}
    assigned = {k for row in table.values() for k in row['candidate_ids']}
    remaining = [k for k in aliases if k not in assigned]
    # Nothing is excluded because it failed a geometric window: the residual
    # batches see the full frozen source inventory with a bounded native list.
    final_unassigned = {}
    for start in range(0, len(remaining), 48):
        result = request(start//48, objects, set(remaining[start:start+48]), residual=True, previous=table); calls += 1
        for row in result['objects']:
            if not row['candidate_ids']:
                continue
            target = table[row['id']]
            merge_mapping_receipt(target,row)
        final_unassigned.update({u['candidate_id']: u for u in result['unassigned']})
        all_results.append(result)
    assigned = {k for row in table.values() for k in row['candidate_ids']}
    unassigned = []
    for key in aliases:
        if key in assigned:
            continue
        if key not in final_unassigned:
            raise ValueError('Missing final native receipt: '+key)
        unassigned.append(final_unassigned[key])
    for key in assigned:
        users = [r for r in table.values() if key in r['candidate_ids']]
        if len({key in r['keep_ids'] for r in users}) > 1:
            for row in users:
                row['keep_ids'] = list(dict.fromkeys([*row['keep_ids'], key]))
                row['status'] = 'incomplete'
                row['reason'] += '; shared native decision conflicted; preserved for source review'
    result = {'objects': list(table.values()), 'unassigned': unassigned}
    validate_mapping(result, objects, aliases, assets)
    return result, calls


def review(scene, styles, rows, source, shape_pptx, arrow_pptx, folder, runtime, *, assets=(), texts=()):
    from shape_integration.common import save
    from shape_integration.styling.export import render_native
    from .quality_choice import apply_choices
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as im:
        clean = im.convert('RGB')
    from .source_ellipsis import observe as observe_punctuation
    punctuation=observe_punctuation(clean,texts)
    rows=[dict(r) for r in rows]
    for row in rows:
        context=[]
        for mark in punctuation:
            xy=np.array([d['center'] for d in mark['dots']]);diam=max(d['diameter'] for d in mark['dots'])
            for side,p in enumerate(row['points'][[0,-1]]):
                if np.all(p>=xy.min(0)-diam) and np.all(p<=xy.max(0)+diam):
                    context.append({'endpoint':side,'role':'source_punctuation','text_id':mark['text_id']})
        if context:row['source_endpoint_context']=context
    from .source_line_bundles import resolve as resolve_pixel_bundles
    all_rows=rows
    from .image_line_scope import resolve as resolve_image_interiors,contained as image_contains
    image_interiors=resolve_image_interiors(rows,assets)
    save(folder/'source_image_interiors.json',{**image_interiors,'claimed_ids':sorted(image_interiors['claimed_ids'])})
    rows=[r for r in rows if r['id'] not in image_interiors['claimed_ids']]
    from .source_straight_routes import resolve as resolve_straight_routes
    straight_routes=resolve_straight_routes(rows,clean)
    save(folder/'source_straight_routes.json',{**straight_routes,'claimed_ids':sorted(straight_routes['claimed_ids'])})
    rows=[r for r in rows if r['id'] not in straight_routes['claimed_ids']]
    pixel_bundles=resolve_pixel_bundles(scene,rows,clean)
    save(folder/'source_pixel_bundles.json',{**pixel_bundles,'claimed_ids':sorted(pixel_bundles['claimed_ids'])})
    rows=[r for r in rows if r['id'] not in pixel_bundles['claimed_ids']]
    regions, outside = scopes(rows, clean.size, scene)
    # Image replacement proposals are candidates, never a pre-review delete.
    # Review only the proposed native users; other arrow-only content remains
    # outside shape/image ownership and travels unchanged into the union.
    for asset_index,asset in enumerate(assets):
        if asset.get('role')=='background':continue
        proposed=set(asset.get('replace_arrows',[]))
        selected=[r for r in outside if r['key'] in proposed or image_contains(r,asset)]
        if not selected:continue
        b=np.array(asset['source_box_half_open'],float)
        rb=np.array([bounds(r) for r in selected])
        q=np.r_[np.maximum(0,np.minimum(b[:2],rb[:,:2].min(0))-16),
                np.minimum(clean.size,np.maximum(b[2:],rb[:,2:].max(0))+17)]
        regions.append({'id':'image_scope_%03d'%asset_index,'box':q,'rows':selected,'context_keys':set()})
        used={r['id'] for r in selected};outside=[r for r in outside if r['id'] not in used]
    for region in regions:
        keys = {key for row in region['rows'] for key in
                row.get('endpoint_nodes', ()) if key in scene['parts']}
        region['context_keys'] = keys
        boxes = [np.asarray(scene['parts'][key]['bbox_2d'], float) for key in keys]
        if boxes:
            boxes = np.asarray(boxes)
            region['box'] = np.r_[np.maximum(0, np.minimum(region['box'][:2], boxes[:, :2].min(0)-8)),
                                    np.minimum(clean.size, np.maximum(region['box'][2:], boxes[:, 2:].max(0)+9))]
    choices = [*image_interiors['choices'],*straight_routes['choices'],*pixel_bundles['choices']]
    receipts = [*image_interiors['receipts'],*straight_routes['receipts'],*pixel_bundles['receipts']]
    unresolved = [{**obj,'status':'incomplete','kind':'native_source_alignment',
        'candidate_ids':[m['candidate_id'] for m in obj['selected_members']],
        'keep_ids':[m['candidate_id'] for m in obj['selected_members']],
        'reason':'Source identity is known; the selected input geometry remains displaced'}
        for obj in pixel_bundles['source_objects'] if obj['native_alignment_status']!='aligned']
    calls = 0
    assets = [a for a in assets if a.get('role') != 'background']
    shared_native={}
    if regions:
        from .native_line_views import render as render_line_views
        try:
            shared_native=render_line_views(scene,styles,all_rows,shape_pptx,arrow_pptx,source,
                                           folder/'shared_native_views',render_native)
        except (ValueError,RuntimeError) as exc:
            save(folder/'shared_native_failure.json',{'error':str(exc),'fallback':'bounded_per_scope_native_views'})
        save(folder/'native_view_reuse.json',{'source_scopes':len(regions),'branch_renders':len(shared_native),
            'native_geometry_rebuilt':False,'source_led_node_context':True})
    def work(region):
        local = folder / region['id']; local.mkdir(parents=True, exist_ok=True)
        box = tuple(np.rint(region['box']).astype(int)); crop = clean.crop(box)
        factor = min(3., 1600 / max(crop.size))
        if max(crop.size) * factor < 900:
            factor = 900 / max(crop.size)
        size = tuple(max(2, round(v * factor)) for v in crop.size)
        scale = np.array(size) / np.array(crop.size)
        crop.resize(size, Image.Resampling.LANCZOS).save(local / 'source.png')
        asset_meta = []
        for asset in assets:
            b = np.asarray(asset['source_box_half_open'], float)
            if np.any(b[:2] > box[2:]) or np.any(np.asarray(box[:2]) > b[2:]):
                continue
            visible = np.clip((b - np.tile(box[:2], 2)) * np.tile(scale, 2), 0, np.tile(np.asarray(size)-1, 2))
            asset_meta.append({'id': asset['id'], 'bbox': visible.round(2).tolist(),
                'ownership_boundary':'measured enclosure interior' if asset.get('native_enclosure') else 'registered source region'})
        asset_ids = {a['id'] for a in asset_meta}
        source_kinds='line|arrow|bundle|frame|uncertain'+('|image_content' if asset_ids else '')
        inventory_prompt = {
            'protocol': PROTOCOL, 'task': 'Define physical SOURCE line/arrow objects in CLEAN before seeing any recovered geometry.',
            'coordinate_frame': {'size': list(size), 'convention': 'inclusive pixel centers'},
            'image_owned_regions': asset_meta,
            'confirmed_source_punctuation':[{'id':m['text_id'],
                'centers':[((np.array(d['center'])-box[:2]+.5)*scale-.5).tolist() for d in m['dots']]}
                for m in punctuation if any(np.all(np.array(d['center'])>=box[:2]) and np.all(np.array(d['center'])<=box[2:]) for d in m['dots'])],
            'rules': [
                ('Image ownership is available only for the listed registered asset IDs.' if asset_ids else
                 'There are NO registered image-owned regions in this view. Allowed kinds are line, arrow, bundle, frame and uncertain. Native diagram pixels are not an image asset; never return image_content or an invented asset ID.'),
                'Read original pixels only. Count and identify visible connections, their actual ends/heads, occlusion and genuinely parallel paths. Do not infer all-to-all links, symmetry or uniform spacing.',
                'Count only physically drawn connections. An ellipsis denotes omitted logical instances, not uncertainty about independently visible wires. Enumerate each visible member separately; one list item must describe one physical connection, never a whole group.',
                'A dashed stroke and visible pieces separated by opaque nodes still belong to their source connection. Crossing does not imply connection.',
                'A bundle contains parallel physical paths between the SAME two source attachments, with their exact visible_count and member order. Different endpoint pairs or different member counts need separate source IDs, even in a repeated network. Never invent uniform counts. Ordinary frames, face boundaries and punctuation are context.',
                'A line/arrow ID represents ONE physical connection. For an exact bundle enumerate members with concrete visible source descriptions. When crossings/occlusion prevent counting, return count_status=uncertain, visible_count=null and members=[]. Never derive a count by multiplying node counts, architectural expectation or symmetry.',
                'Image-owned internal linework stays inside its image asset; identify it as image_content with the supplied asset_id. External connections remain separate and must meet the actual source boundary.',
                'Confirmed punctuation is context, never an omitted neuron. Do not infer paths converging to it from network architecture; inspect whether the original actually has those paths.',
                'A bbox is a locator only. Describe source paths and attachments, not a fitted curve. Mark ambiguity instead of guessing.'],
            'output': {'objects': [{'id': 'S1', 'kind': source_kinds,
                                    'bbox': [0, 0, size[0] - 1, size[1] - 1], 'visible_count': 1,
                                    'count_status': 'exact|uncertain', 'members': [],
                                    'description': 'visible paths, direction, attachments and occlusion', 'asset_id': ''}],
                       'uncertain': []}}
        observer = runtime.for_stage(reasoning_effort='low') if hasattr(runtime, 'for_stage') else runtime
        inventory = observer.call(local / 'source_calls', 'inventory',
            'Observe SOURCE before predictions. Return only the requested JSON.', inventory_prompt,
            [('CLEAN SOURCE: no recovered paths', local / 'source.png')],
            lambda d: validate_inventory(d, size, asset_ids, allow_partial=True), 16384)
        save(local / 'source_objects_initial.json', inventory)
        save(local / 'source_objects.json', inventory)
        members = sorted(region['rows'], key=lambda r: (r['layer'], r['id']))
        aliases = {f'C{i + 1}': row for i, row in enumerate(members)}
        context_keys = [k for k, p in scene['parts'].items()
                        if p.get('status') == 'ok' and p.get('kind') in ('ellipse', 'rectangle', 'rounded_rectangle')
                        and (styles.get(k, {}).get('fill') or k in region['context_keys']) and not (np.any(np.asarray(p['bbox_2d'])[:2] > box[2:])
                        or np.any(np.asarray(box[:2]) > np.asarray(p['bbox_2d'])[2:]))]
        context = None
        if context_keys:
            context = local / 'nodes.pptx'
            subset(shape_pptx, context, [scene['parts'][k]['object_id'] + '/' + json_part_id(k) for k in context_keys])
        images = [('Frozen CLEAN source objects', local / 'source.png')]
        for branch, pptx in [('shape', shape_pptx), ('arrows', arrow_pptx)]:
            selected = [r for r in members if r['layer'] == branch]
            if not selected:
                continue
            if branch in shared_native:
                preview=shared_native[branch]
            else:
                target = local / (branch + '.pptx')
                selection_subset(pptx, target, selected)
                if context is not None:
                    shown = local / (branch + '_with_nodes.pptx')
                    merge(target, [('same_nodes', context)], shown); target = shown
                preview = render_native(target, local / (branch + '_render'), scene['source_size'], 180)
            with Image.open(preview) as im:
                native = im.convert('RGB').crop(box).resize(size, Image.Resampling.LANCZOS)
            native.save(local / (branch + '_native.png'))
            images.append((branch.upper() + ': actual native paths with the same nodes', local / (branch + '_native.png')))
            locator = native.copy(); draw = ImageDraw.Draw(locator)
            for alias, row in aliases.items():
                if row['layer'] != branch:
                    continue
                point = (row['points'][len(row['points']) // 2] - box[:2] + .5) * scale - .5
                draw.text(tuple(point), alias, fill='#A00090', stroke_width=1, stroke_fill='white')
            locator.save(local / (branch + '_ids.png'))
            images.append((branch.upper() + ': same native image with candidate ID labels, annotations only', local / (branch + '_ids.png')))
        prompt = {
            'protocol': PROTOCOL, 'task': 'Match existing native content to each frozen SOURCE object and select its best complete recovery.',
            'source_objects': inventory['objects'], 'image_owned_regions': asset_meta,
            'candidates': [summary(row, alias, box, scale) for alias, row in aliases.items()],
            'rules': [
                ('Image-owned unassigned content requires one of the supplied existing asset IDs.' if asset_ids else
                 'There are NO registered image assets in this scope. Unassigned classification must be context, unsupported or uncertain; never image_owned.'),
                'Source identity was defined before these predictions. A displaced, short, long or differently segmented fit may still recover that SAME source object. Prediction overlap, bbox IoU and endpoint distance do not decide identity.',
                'Account for every C exactly once or as a consistent shared user. Never silently ignore an unmatched fit. Do not change geometry, paint, heads or unrelated nodes.',
                'Select native IDs per actual source connection, not a whole winning branch. A bundle may keep several complementary paths, but cannot retain two versions of the same physical path. State how its retained physical shafts account for visible_count; fragments and heads are not extra parallel paths. Verify every member, not a uniform network template.',
                'Bind shaft, open/filled head and attached dot by source identity regardless of M/C command syntax. Keep complementary fragments together; deleting only a shaft must not leave its head.',
                'Prefer complete source-supported endpoints and proper joins over a shorter partial fit. If all fits are inaccurate record incomplete/uncertain, with the usable survivor, instead of reporting resolved.',
                'Inventory kind and unmatched classification are descriptive hints, never deletion authority. Preserve independent boundaries and punctuation. An unmatched duplicate requires duplicate_of with a frozen source_id and its selected surviving candidate_ids (including supplied prior survivors). Without that physical replacement witness the native content stays unresolved. Image-owned duplicates require a registered asset covering the complete native content; never consume its incoming external connections.',
                'For source objects with no supplied recovery return empty candidate_ids/keep_ids and status incomplete. Return a receipt for every frozen S, including frames and image content.'],
            'output': {'objects': [{'id': 'S1', 'candidate_ids': ['C1', 'C2'], 'keep_ids': ['C2'],
                                    'status': 'resolved|incomplete|uncertain', 'reason': 'source path and unique content'}],
                       'unassigned': [{'candidate_id': 'C3', 'classification': 'context|unsupported|uncertain'+('|image_owned' if asset_ids else ''),
                                       'asset_id': '', 'reason': 'specific source evidence',
                                       'duplicate_of': {'source_id': 'S1', 'candidate_ids': ['C2']}}]}}
        mapping, mapping_calls = select_mapping(runtime, local, prompt, images, inventory['objects'], aliases,
            {a['id']: a for a in assets if a['id'] in asset_ids}, box, scale)
        save(local/'mapping_initial.json',mapping)
        from .source_external_repair import repair as repair_external_sources
        inventory,mapping,external_calls=repair_external_sources(inventory,mapping,aliases,
            [a for a in assets if a['id'] in asset_ids],images,clean,box,scale,local/'external_source_repair',observer,save)
        save(local/'source_objects.json',inventory)
        save(local / 'mapping.json', mapping)
        local_choices = []; local_receipts = []; issues = []
        for item in mapping['objects']:
            candidate_ids = [aliases[k]['id'] for k in item['candidate_ids']]
            keep_ids = [aliases[k]['id'] for k in item['keep_ids']]
            source_object=next(o for o in inventory['objects'] if o['id']==item['id'])
            source_kind=source_object['kind']
            witnesses=[]
            for witness in item.get('duplicate_witnesses',[]):
                removed=witness.get('candidate_id');surviving=witness.get('candidate_ids')
                if (item['status']!='uncertain' and witness.get('source_id')==item['id']
                        and removed in item['candidate_ids'] and removed not in item['keep_ids']
                        and isinstance(surviving,list) and surviving
                        and all(isinstance(k,str) for k in surviving)
                        and set(surviving)<=set(item['keep_ids'])):
                    witnesses.append({'candidate_id':aliases[removed]['id'],
                        'survivor_ids':[aliases[k]['id'] for k in surviving],
                        'source_object_id':region['id']+'/'+item['id']})
            local_choices.append({'candidate_ids': candidate_ids, 'keep_ids': keep_ids, 'reason': item['reason'],
                                  'source_object_id': region['id'] + '/' + item['id'],'source_kind':source_kind,
                                  'source_replacement_witnesses':witnesses,
                                  'source_object_selection':{
                                      'source_object_id':region['id']+'/'+item['id'],
                                      'physical_source_count':source_object.get('visible_count'),
                                      'count_status':source_object.get('count_status','exact'),
                                      'mapping_status':item['status'],'survivor_ids':keep_ids}})
            local_receipts.append({**item, 'id': region['id'] + '/' + item['id'], 'source_kind':source_kind,
                                   'candidate_ids': candidate_ids, 'keep_ids': keep_ids})
            if item['status'] != 'resolved':
                issues.append(local_receipts[-1])
        for item in mapping['unassigned']:
            row = aliases[item['candidate_id']]; role = item['classification']; remove = False
            if role == 'image_owned':
                asset = next(a for a in assets if a['id'] == item['asset_id'])
                remove = image_contains(row,asset)
            local_choices.append({'candidate_ids': [row['id']], 'keep_ids': [] if remove else [row['id']], 'reason': item['reason'],
                **({'source_kind':'image_content','asset_id':item['asset_id']} if role=='image_owned' and remove else {})})
            local_receipts.append({**item, 'candidate_id': row['id'], 'removed': remove})
            if role in ('uncertain','unsupported'):
                issues.append({**local_receipts[-1], 'status': 'uncertain', 'kind': 'unassigned_native_source'})
        save(local / 'receipts.json', local_receipts)
        return local_choices, local_receipts, issues, 1+mapping_calls+external_calls
    workers = max(1, min(4, int(getattr(runtime, 'config', {}).get('source_scope_workers', 2))))
    def safe_work(region):
        try:
            return work(region)
        except Exception as exc:
            local = folder/region['id']; local.mkdir(parents=True, exist_ok=True)
            failure = {'scope_id': region['id'], 'status': 'uncertain', 'kind': 'scope_review_failed',
                       'error': str(exc), 'error_type': type(exc).__name__,
                       'candidate_ids': [r['id'] for r in region['rows']],
                       'reason': 'No valid source ownership result; all native content in this scope is preserved'}
            save(local/'failure.json', failure)
            retained = [{'candidate_ids': [r['id']], 'keep_ids': [r['id']], 'reason': failure['reason']} for r in region['rows']]
            native = [{'candidate_id': r['id'], 'classification': 'uncertain', 'removed': False,
                       'review_status': 'source_review_failed', 'reason': str(exc)} for r in region['rows']]
            calls = sum(1 for _ in local.glob('**/request.json'))
            return retained, native, [failure], calls
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for c, r, u, n in pool.map(safe_work, regions):
            choices.extend(c); receipts.extend(r); unresolved.extend(u); calls += n
    from .source_line_coverage import guard as guard_source_coverage
    native_arrays={}
    for branch,path in shared_native.items():
        with Image.open(path) as opened:native_arrays[branch]=np.asarray(opened.convert('RGB'))
    choices, coverage_guards = guard_source_coverage(all_rows, choices, clean,native_views=native_arrays)
    unresolved.extend({**row,'kind':'replacement_coverage','status':'incomplete'} for row in coverage_guards)
    result = apply_choices(all_rows, choices)
    accounted={key for choice in choices for key in choice['candidate_ids']}
    live={key for choice in choices for key in choice['keep_ids']}
    for receipt in receipts:
        key=receipt.get('candidate_id')
        if key in accounted:
            receipt['proposed_removed']=receipt.get('removed')
            receipt['removed']=key not in live
        if 'candidate_ids' in receipt and 'keep_ids' in receipt:
            receipt['proposed_keep_ids']=list(receipt['keep_ids'])
            receipt['keep_ids']=[key for key in receipt['candidate_ids'] if key in live]
    outside_receipts = [{'candidate_id': r['id'], 'classification': 'outside_shape_scope', 'removed': False} for r in outside]
    outside_receipts += [{'candidate_id': r['id'], 'classification': 'face_boundary_context', 'removed': False}
                         for r in rows if r['layer'] == 'shape' and 'segment' in r]
    result.update(protocol=PROTOCOL, source_object_ownership=True, ordinary_line_owner='source_objects',
                  branch_choices=choices, source_receipts=receipts + outside_receipts,
                  source_only_candidates=[u for u in unresolved if 'id' in u and not u.get('candidate_ids')],
                  unresolved_source_objects=unresolved, model_calls=calls, reviewed_pairs=0,
                  line_ownership_complete=not unresolved and not outside, failed_components=[u for u in unresolved if u.get('kind') == 'scope_review_failed'],
                  source_coverage_guards=coverage_guards,
                  source_pixel_bundles=pixel_bundles['source_objects'],
                  source_straight_routes=straight_routes['source_objects'],
                  source_image_interiors=image_interiors['source_objects'],
                  reviewed_arrow_names=sorted({r['key'] for region in regions for r in region['rows'] if r['layer'] == 'arrows'} |
                      {r['key'] for r in all_rows if r['layer']=='arrows' and r['id'] in pixel_bundles['claimed_ids']|straight_routes['claimed_ids']|image_interiors['claimed_ids']}))
    save(folder / 'audit.json', result)
    return result
