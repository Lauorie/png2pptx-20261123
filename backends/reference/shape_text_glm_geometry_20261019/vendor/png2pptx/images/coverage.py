"""Full-image planning, model-selected local discovery, and bounded coverage repair."""
import copy
from pathlib import Path

from PIL import Image, ImageDraw

from ..frames import Crop
from ..scene import save
from .clients import cached_json_call
from .detect import validate_inventory
from .prompts import (COVERAGE_SYSTEM, IMAGE_LABELS, PENDING_REGION_REASON,
                      SEARCH_REASONS, coverage_audit_prompt, global_detection_prompt, local_detection_prompt)
from .search import contains, union_boxes, gap_regions, native_crop, split_box, coverage_fraction


def check_regions(regions, size, known, required):
    if not isinstance(regions, list): raise ValueError('regions must be an array')
    seen, assigned = set(), []
    for region in regions:
        if not isinstance(region.get('id'), str) or region['id'] in seen:
            raise ValueError('Duplicate/missing region identity')
        seen.add(region['id']); Crop.from_model(region['bbox'], size)
        if not isinstance(region.get('target_ids'), list) or any(x not in known for x in region['target_ids']):
            raise ValueError('Unknown region owner')
        if len(set(region['target_ids'])) != len(region['target_ids']): raise ValueError('Repeated region owner')
        assigned.extend(region['target_ids'])
    if any(assigned.count(ident) != 1 for ident in required):
        raise ValueError('Every new/global asset requires exactly one local inspection region')


def validate_global(raw, size):
    validate_inventory(raw, size)
    if not isinstance(raw.get('more_pending'), bool): raise ValueError('Missing more_pending')
    ids = {a['id'] for a in raw['assets']}
    check_regions(raw['regions'], size, ids, ids)
    return raw


def validate_local(raw, size, known, required):
    if not isinstance(raw.get('more_pending'), bool): raise ValueError('Missing required top-level boolean more_pending (true or false)')
    if not isinstance(raw.get('uncertain'), list): raise ValueError('Missing required top-level array uncertain')
    if not isinstance(raw.get('search_complete'), bool): raise ValueError('Missing search_complete boolean')
    if not isinstance(raw.get('zoom_regions'), list): raise ValueError('Missing zoom_regions array')
    for region in raw['zoom_regions']:
        Crop.from_model(region['bbox'], size)
        if not isinstance(region.get('reason'), str): raise ValueError('Zoom region needs a reason')
    if raw['search_complete'] and raw['zoom_regions']:
        raise ValueError('search_complete cannot be true with pending zoom_regions')
    handled = []
    for i, item in enumerate(raw['assets']):
        owner = item['owner_id']
        if owner is not None and owner not in known: raise ValueError('Unknown local asset owner')
        if owner is not None and owner not in required:
            raise ValueError(f'{owner} is an unassigned peer. Do not change its bbox from a partial view; return only assigned IDs or owner_id=null for a genuinely new complete asset.')
        validate_inventory({'assets': [{**item, 'id': f'I{i+1:04d}'}], 'uncertain': []}, size)
        if owner is not None: handled.append(owner)
    for field in ('excluded', 'needs_context'):
        for record in raw[field]:
            ident = record['id']
            allowed = (ident in required or (field == 'excluded' and ident is None)
                       or (field == 'needs_context' and ident in known))
            if not allowed or not isinstance(record.get('reason'), str):
                raise ValueError(f'Invalid {field} ID {ident!r}. Assigned IDs={sorted(required)}; excluded id=null is allowed only for an unnamed non-asset observation.')
            if ident is not None: handled.append(ident)
    if len(handled) != len(set(handled)) or not set(required).issubset(handled):
        missing = sorted(set(required)-set(handled))
        repeated = sorted({ident for ident in handled if handled.count(ident) > 1})
        raise ValueError(f'Local response must account for each assigned target exactly once; missing={missing}, repeated={repeated}. Use needs_context when the target is cut off.')
    return raw


def duplicate_owners(raw, known):
    """Resolve the whole ownership graph before mutating the inventory."""
    excluded = [record['id'] for record in raw['exclude']]
    drops = [pair['drop_id'] for pair in raw['duplicates']]
    if len(excluded) != len(set(excluded)) or len(drops) != len(set(drops)) or set(excluded) & set(drops):
        raise ValueError('Each removed ID requires one unambiguous resolution')
    direct = {pair['drop_id']: pair['keep_id'] for pair in raw['duplicates']}
    resolved = {}
    for drop, owner in direct.items():
        seen = {drop}
        while owner in direct:
            if owner in seen: raise ValueError('Duplicate ownership cycle')
            seen.add(owner); owner = direct[owner]
        if owner not in known or owner in excluded: raise ValueError('Duplicate points to excluded/unknown owner')
        resolved[drop] = owner
    return resolved


def validate_audit(raw, size, known, historical=()):
    if raw.get('verdict') not in ('complete', 'revise', 'unresolved'):
        raise ValueError('verdict must be complete, revise, or unresolved')
    if not isinstance(raw.get('more_pending'), bool): raise ValueError('Missing required top-level boolean more_pending (true or false)')
    validate_inventory({'assets': raw['new_assets'], 'uncertain': raw['uncertain']}, size)
    new = {a['id'] for a in raw['new_assets']}
    if new & (known | set(historical)): raise ValueError('New audit IDs collide with existing/historical inventory; use a new I0001-style ID')
    check_regions(raw['regions'], size, known | new, new)
    for pair in raw['duplicates']:
        if pair['drop_id'] not in known or pair['keep_id'] not in known or pair['drop_id'] == pair['keep_id']:
            raise ValueError('Invalid duplicate ownership')
    for record in raw['exclude']:
        if record['id'] not in known: raise ValueError('Unknown excluded ID')
    duplicate_owners(raw, known)
    return raw


def expanded_region(region, size, fraction):
    """Add viewing context only; the model still decides the actual target bbox."""
    result = copy.deepcopy(region); b = region['bbox']
    dx, dy = max(8., (b[2]-b[0]+1)*fraction), max(8., (b[3]-b[1]+1)*fraction)
    result['bbox'] = [max(0., b[0]-dx), max(0., b[1]-dy),
                      min(size[0]-1., b[2]+dx), min(size[1]-1., b[3]+dy)]
    return result


def detect_with_coverage(client, image_path, size, folder, repair_rounds=2,
                         source_path=None, max_local_calls=16):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    overview = Image.open(image_path).convert('RGB')
    source = Image.open(source_path or image_path).convert('RGB')
    state = {'assets': {}, 'excluded': {}, 'events': [], 'issues': [], 'audits': [],
             'coverage_verified': False, 'coverage_status': 'running', 'unrefined_ids': [],
             'non_asset_observations': [], 'search_regions': {}, 'local_calls': 0,
             'search_budget': {'max_local_calls': max_local_calls, 'max_depth': repair_rounds}}
    nodes = state['search_regions']
    nodes['root'] = {'id': 'root', 'bbox': [0, 0, size[0]-1, size[1]-1],
                     'children': [], 'parent_id': None, 'status': 'waiting_children', 'depth': -1}
    last_views = {}
    view_history = {}

    def call(stage, prompt, images, validator):
        result, provenance = cached_json_call(client, stage, COVERAGE_SYSTEM, prompt, images, validator)
        state['events'].append({'stage': stage, 'call': provenance})
        return result, provenance

    def add_node(region, parent='root', depth=0, kind='search', prepared_view=None):
        ident = f'R{len(nodes):04d}'
        node = {'id': ident, 'model_region_id': region.get('id'), 'bbox': region['bbox'],
                'target_ids': list(dict.fromkeys(region.get('target_ids', []))),
                'reason': region['reason'], 'parent_id': parent, 'depth': depth, 'kind': kind,
                'status': 'pending', 'children': [], 'examined': False}
        if prepared_view is not None: node['prepared_view'] = prepared_view
        nodes[ident] = node; nodes[parent]['children'].append(ident)
        ancestor = parent
        while ancestor is not None:
            nodes[ancestor]['status'] = 'waiting_children'
            ancestor = nodes[ancestor]['parent_id']
        return node

    def refresh():
        for node in reversed(list(nodes.values())):
            if node.get('kind') == 'context' and node['status'] == 'pending' and not any(
                    ident in state['unrefined_ids'] for ident in node['target_ids']):
                node.update(status='complete', resolution='target resolved by another inspection or exclusion')
            if node['status'] == 'waiting_children' and node['children'] and all(
                    nodes[c]['status'] == 'complete' for c in node['children']):
                node['status'] = 'complete'
        roots = [nodes[c] for c in nodes['root']['children']]
        state['search_coverage'] = {
            'planned_fraction': coverage_fraction([n['bbox'] for n in roots], size),
            'examined_fraction': coverage_fraction([n['view_bbox'] for n in nodes.values() if n.get('examined')], size),
            'completed_fraction': coverage_fraction([n['bbox'] for n in roots if n['status']=='complete'], size),
            'status': 'complete' if nodes['root']['status']=='complete' else 'unresolved',
            'pending_regions': [{'id': n['id'], 'bbox': n['bbox'], 'status': n['status'],
                                 'depth': n['depth'], 'target_ids': n.get('target_ids', [])}
                                for n in nodes.values() if n['id']!='root' and n['status']!='complete'],
            'visual_recall_verified': False}

    def checkpoint():
        refresh()
        save(folder/'state.json', state)
        save(folder/'search_coverage.json', {'summary': state['search_coverage'], 'regions': nodes,
                                           'budget': state['search_budget'], 'local_calls': state['local_calls']})

    def add_context_child(node, ident):
        if ident not in state['assets']: return
        if ident not in state['unrefined_ids']: state['unrefined_ids'].append(ident)
        base = union_boxes([state['assets'][ident]['bbox'], node['view_bbox'],
                            last_views.get(ident, node['view_bbox'])])
        view = expanded_region({'bbox': base}, size, .5)['bbox']
        add_node({'bbox': base, 'target_ids': [ident], 'reason': SEARCH_REASONS['context']},
                 node['id'], node['depth']+1, 'context', view)

    def local_pass(pending, stage):
        for index, node in enumerate(pending):
            refresh()
            if node['status'] != 'pending': continue
            if node['depth'] > repair_rounds or state['local_calls'] >= max_local_calls:
                node['budget_exhausted'] = True; checkpoint(); continue
            assigned = [i for i in node['target_ids'] if i in state['assets']]
            view = node.get('prepared_view') or expanded_region(node, size, .15)['bbox']
            if assigned: view = union_boxes([view]+[state['assets'][i]['bbox'] for i in assigned])
            if node['kind']=='context':
                latest = [last_views[i] for i in assigned if i in last_views]
                if any(not contains(view,b) for b in latest):
                    view = expanded_region({'bbox':union_boxes([view]+latest)},size,.5)['bbox']
            repairing_elsewhere = {i for n in nodes.values() if n['id']!=node['id']
                                   and n.get('kind')=='context' and n['status']=='pending'
                                   for i in n['target_ids']}
            assigned += [i for i,a in state['assets'].items() if i not in assigned
                         and i not in repairing_elsewhere and contains(view,a['bbox'])
                         and (i in state['unrefined_ids'] or contains(node['bbox'],a['bbox']))]
            crop, native, metadata = native_crop(source, view, size)
            # A context retry must show new source pixels, including at image boundaries.
            previous = [last_views[i] for i in assigned if i in last_views]
            source_key = tuple(metadata['source_crop_half_open'])
            if previous and any(source_key in view_history.get(i, set()) for i in assigned):
                view = expanded_region({'bbox': union_boxes([view]+previous)}, size, .5)['bbox']
                crop, native, metadata = native_crop(source, view, size)
                source_key = tuple(metadata['source_crop_half_open'])
                if any(source_key in view_history.get(i, set()) for i in assigned):
                    node.update(status='unresolved', failure='no_larger_distinct_source_view')
                    checkpoint(); continue
            parent = nodes[node['parent_id']]
            if node['kind']=='zoom' and list(source_key)==parent.get('source_crop_half_open'):
                node.update(status='unresolved', failure='zoom_did_not_change_source_view')
                checkpoint(); continue
            # Partial peers remain context only; active context repairs keep their owner.
            node.update(target_ids=assigned, view_bbox=view, source_crop_half_open=list(source_key),
                        input_size=list(crop.input_size), frame_scale=metadata['effective_crop_scale_xy'])
            for ident in assigned:
                if ident not in state['unrefined_ids']: state['unrefined_ids'].append(ident)
                last_views[ident] = view
                view_history.setdefault(ident, set()).add(source_key)
            local_folder = folder/stage/f'{index:03d}'; local_folder.mkdir(parents=True, exist_ok=True)
            native.save(local_folder/'source_native.png')
            native.resize(crop.input_size, Image.Resampling.LANCZOS).save(local_folder/'input_2048.png')
            save(local_folder/'frame.json', metadata)
            save(local_folder/'region.json', {'requested': {k:node[k] for k in ('id','bbox','target_ids','reason')},
                 'with_context': {'bbox': view, 'target_ids': assigned}, 'search_node': node['id']})
            peers = []
            for ident, asset in state['assets'].items():
                b = asset['bbox']
                if ident in assigned or (b[0] <= view[2] and b[2] >= view[0] and b[1] <= view[3] and b[3] >= view[1]):
                    peers.append({'id': ident, 'target': asset['target'], 'assigned': ident in assigned,
                                  'bbox_local_hint': crop.to_local(b[:2])+crop.to_local(b[2:])})
            prompt = local_detection_prompt(stage, crop.input_size, node['reason'], peers, assigned,
                {'id': node['id'], 'depth': node['depth'],
                 'bbox_local': crop.to_local(node['bbox'][:2])+crop.to_local(node['bbox'][2:]),
                 'remaining_local_calls': max_local_calls-state['local_calls']})
            state['local_calls'] += 1
            try:
                raw, evidence = call(stage+f'_{index:03d}', prompt,
                    [(IMAGE_LABELS['local'], local_folder/'input_2048.png')],
                    lambda r: validate_local(r, crop.input_size, set(state['assets']), set(assigned)))
                node.update(examined=True, call=evidence, search_complete_reported=raw['search_complete'])
                for item in raw['assets']:
                    ident = item['owner_id']
                    if ident is None:
                        ident = f'I{1+max([int(x[1:]) for x in set(state["assets"]) | set(state["excluded"])] or [0]):04d}'
                    b = crop.to_global(item['bbox'][:2])+crop.to_global(item['bbox'][2:])
                    b = [max(0., min(v, size[i % 2]-1.)) for i,v in enumerate(b)]
                    previous_asset = state['assets'].get(ident, {})
                    state['assets'][ident] = {**item, 'id': ident, 'bbox': b,
                        'history': previous_asset.get('history', [])+[{'stage': stage, 'call': evidence,
                            'bbox_local': item['bbox'], 'frame': str((local_folder/'frame.json').resolve()),
                            'search_region': node['id']}]}
                    state['assets'][ident].pop('owner_id', None)
                    if ident in state['unrefined_ids']: state['unrefined_ids'].remove(ident)
                for item in raw['excluded']:
                    if item['id'] is None:
                        state['non_asset_observations'].append({**item, 'stage': stage, 'region': index, 'call': evidence})
                    else:
                        state['excluded'][item['id']] = {**state['assets'].pop(item['id']), 'reason': item['reason'], 'call': evidence}
                        if item['id'] in state['unrefined_ids']: state['unrefined_ids'].remove(item['id'])
                for item in raw['needs_context']:
                    add_context_child(node, item['id'])
                for region in raw['zoom_regions']:
                    b = crop.to_global(region['bbox'][:2])+crop.to_global(region['bbox'][2:])
                    b = [max(0., min(v, size[i%2]-1.)) for i,v in enumerate(b)]
                    # Persist model proposals; never turn coordinate validity into visual acceptance.
                    add_node({'bbox': b, 'target_ids': [], 'reason': region['reason']},
                             node['id'], node['depth']+1, 'zoom')
                pending_work = not raw['search_complete'] or raw['more_pending'] or raw['uncertain'] or raw['needs_context']
                if pending_work and not node['children']:
                    for box in split_box(node['bbox']):
                        add_node({'bbox': box, 'target_ids': [], 'reason': SEARCH_REASONS['fallback']},
                                 node['id'], node['depth']+1, 'zoom')
                node['status'] = ('waiting_children' if node['children'] else
                                  'unresolved' if pending_work else 'complete')
                node['unresolved'] = {'needs_context':raw['needs_context'], 'uncertain':raw['uncertain'],
                                      'more_pending':raw['more_pending'], 'zoom_regions':raw['zoom_regions']}
                if pending_work:
                    state['issues'].append({'stage': stage, 'region': node['id'], **node['unresolved']})
            except Exception as exc:
                node.update(status='unresolved', failure=str(exc))
                state['issues'].append({'stage': stage, 'region': node['id'], 'error': str(exc)})
            checkpoint()

    initial, evidence = call('global', global_detection_prompt(size),
        [(IMAGE_LABELS['global'], image_path)], lambda r: validate_global(r, size))
    state['assets'] = {a['id']: {**a, 'history': [{'stage':'global','call':evidence}]} for a in initial['assets']}
    state['unrefined_ids'] = list(state['assets'])
    state['initial_count'] = len(initial['assets']); state['initial_regions'] = initial['regions']
    for region in initial['regions']: add_node(region)
    for box in gap_regions([r['bbox'] for r in initial['regions']], size):
        add_node({'bbox':box, 'target_ids':[], 'reason':SEARCH_REASONS['gap']}, kind='gap')
    checkpoint()
    local_pass([n for n in nodes.values() if n['status']=='pending'], 'local_00')
    for round_index in range(repair_rounds+1):
        overlay = overview.copy(); draw = ImageDraw.Draw(overlay)
        for asset in state['assets'].values():
            draw.rectangle(asset['bbox'], outline='#ff00cc', width=3)
            draw.text(tuple(asset['bbox'][:2]), asset['id'], fill='#ff00cc')
        overlay_path = folder/f'coverage_{round_index:02d}.png'; overlay.save(overlay_path)
        inventory = [{k:a[k] for k in ('id','category','target','bbox')} for a in state['assets'].values()]
        prompt = coverage_audit_prompt(size, inventory, state['issues'][-12:],
                                       state['unrefined_ids'], state['search_coverage'])
        try:
            audit, evidence = call(f'audit_{round_index:02d}', prompt,
                [(IMAGE_LABELS['audit_source'],image_path),(IMAGE_LABELS['audit_overlay'],overlay_path)],
                lambda r: validate_audit(r, size, set(state['assets']), state['excluded']))
        except Exception as exc:
            state['issues'].append({'stage':'audit','error':str(exc)}); break
        state['audits'].append({'response':audit, 'call':evidence})
        duplicates = duplicate_owners(audit, set(state['assets']))
        for record in audit['exclude']:
            state['excluded'][record['id']] = {**state['assets'].pop(record['id']), 'reason':record['reason'], 'call':evidence}
            if record['id'] in state['unrefined_ids']: state['unrefined_ids'].remove(record['id'])
        for drop, owner in duplicates.items():
            state['excluded'][drop] = {**state['assets'].pop(drop),'duplicate_of':owner,'call':evidence}
            if drop in state['unrefined_ids']: state['unrefined_ids'].remove(drop)
        for node in nodes.values():
            if node.get('target_ids'):
                node['target_ids'] = list(dict.fromkeys(duplicates.get(i,i) for i in node['target_ids']
                                          if duplicates.get(i,i) in state['assets']))
        for asset in audit['new_assets']:
            state['assets'][asset['id']] = {**asset, 'history':[{'stage':'audit_discovery','call':evidence}]}
            state['unrefined_ids'].append(asset['id'])
        for region in audit['regions']:
            region = copy.deepcopy(region)
            region['target_ids'] = list(dict.fromkeys(duplicates.get(i,i) for i in region['target_ids']
                                       if duplicates.get(i,i) in state['assets']))
            # Existing context/zoom tasks already own these repairs; enlarge that task if needed.
            queued = next((n for n in nodes.values() if n['status']=='pending' and region['target_ids']
                           and set(region['target_ids']).issubset(n.get('target_ids', []))), None)
            if queued:
                queued['bbox'] = union_boxes([queued['bbox'],region['bbox']])
                if queued.get('prepared_view'):
                    queued['prepared_view'] = union_boxes([queued['prepared_view'],region['bbox']])
            else:
                add_node(region, kind='audit')
        queued_ids = {i for n in nodes.values() if n['status']=='pending' for i in n.get('target_ids', [])}
        for ident in state['unrefined_ids']:
            if ident not in queued_ids:
                base = union_boxes([state['assets'][ident]['bbox'],last_views.get(ident,state['assets'][ident]['bbox'])])
                add_node({'bbox':base,'target_ids':[ident],'reason':PENDING_REGION_REASON}, kind='context',
                         prepared_view=expanded_region({'bbox':base},size,.5)['bbox'])
        checkpoint()
        if (audit['verdict']=='complete' and not audit['more_pending'] and not audit['regions']
            and not audit['new_assets'] and not audit['uncertain'] and not state['unrefined_ids']
            and state['search_coverage']['status']=='complete'):
            state['coverage_status']='model_review_complete'; checkpoint(); break
        if round_index < repair_rounds:
            local_pass([n for n in nodes.values() if n['status']=='pending'], f'local_{round_index+1:02d}')
    if state['coverage_status']=='running': state['coverage_status']='unresolved'
    checkpoint()
    inventory = {'assets':list(state['assets'].values()), 'uncertain':state['issues'],
                 'coverage_status':state['coverage_status'], 'coverage_verified':False,
                 'search_coverage':state['search_coverage'], 'excluded':state['excluded'],
                 'coverage_evidence':str((folder/'state.json').resolve())}
    validate_inventory(inventory,size); save(folder/'inventory.json',inventory)
    return inventory,str((folder/'state.json').resolve())
