"""One case: immutable geometry -> measured lines -> one select -> render -> critic -> one refine -> render.

Phases run as batches. After refinement, review every unit whose crop can see a
changed shape, so a previously accepted neighbour cannot retain stale context.
Pixel measurement along the fixed contour
measures stroke width and proposes line patterns; explicit SAM/RGB visual review
resolves categorical conflicts. The model decides fill, stroke presence and colours.
Model requests are cached by identity, so a
rerun replays completed work and continues at the first missing request.
"""
import copy
import time
from contextlib import nullcontext
from concurrent.futures import as_completed
from pathlib import Path
import numpy as np
from PIL import Image
from shapely.geometry import box as shape_box
from shapely.ops import unary_union
from ..common import read, save, sha, digest, now
from . import rules, occlusion, views, model, grouping, lines
from .geometry import parts, closed, raster, bounds
from .export import export, pattern
from .live import publish
from .native_stroke import NativeFitter, PROTOCOL as NATIVE_PROTOCOL
from .width_limit import limit as limit_width

SCHEDULING_PROTOCOL = 'semantic_then_native_fit_reviewed_commit.21'
LINE_PROTOCOL = NATIVE_PROTOCOL
# Rule reasons that a reliable contour measurement answers without the model.
LINE_REASONS = {'no_separate_stroke_evidence', 'partial_boundary_support', 'width_ambiguous', 'dash_needs_visual_check'}


def unchanged_styles(scene, table, root, assets=()):
    """Recheck existing paint instead of regenerating it for unchanged geometry."""
    if not root:return {},None
    seed=Path(root)/scene['case_id']
    old=read(seed/'fitting/scene.json');document=read(seed/'styling/styles.json')
    if old['source_sha256']!=scene['source_sha256']:raise ValueError('Style seed source differs')
    identity={'geometry':sha(seed/'fitting/scene.json'),'styles':sha(seed/'styling/styles.json')}
    # Geometry equality does not validate an old semantic/numeric policy.
    if (document.get('input',{}).get('style_policy') != style_policy() or
            document.get('input',{}).get('assets_sha256') != digest(list(assets))):
        return {}, {**identity,'reused_parts':0,'reason':'style_policy_changed'}
    previous={(p['object_id'],p['id']):(k,p) for k,p in old['parts'].items()}
    matched={}
    for key,part in table.items():
        entry=previous.get((part['object_id'],part['id']))
        if not entry:continue
        name,p=entry
        if name not in document['parts']:continue
        if document['parts'][name].get('state')!='critic_accept':continue
        if any(part.get(f)!=p.get(f) for f in ('kind','commands','native_parameters_original')):continue
        matched[key]=copy.deepcopy(document['parts'][name]['style'])
    # A changed neighbour or asset can change both visibility and source sampling.
    # Seed only an unchanged scene; exact request caching still works separately.
    old_geometry=sorted((p['object_id'],p['id'],p['kind'],p.get('commands')) for p in old['parts'].values())
    geometry=sorted((p['object_id'],p['id'],p['kind'],p.get('commands')) for p in table.values())
    if old_geometry!=geometry:matched={}
    return matched,{**identity,'reused_parts':len(matched)}


def style_policy():
    return {name:sha(Path(__file__).with_name(name)) for name in
            ('controller.py','model.py','native_stroke.py','occlusion.py','views.py','export.py')}


def fixed_fields(line):
    # Ownership delegates computation; critic feedback can request refitting.
    return ['width_px', 'dash', 'dash_pattern_px']


def resolve_pattern_consensus(styles, checks, evidence):
    """Use an existing measurement only when the last source review agrees."""
    changes={}
    for key,check in checks.items():
        value=check.get('value') or {};expected=value.get('expected_line_pattern')
        line=evidence.get(key,{}).get('line',{})
        if (value.get('verdict')!='revise' or value.get('affected_fields')!=['line_pattern']
                or expected not in ('dash','dot','dash_dot') or not line.get('dash_reliable')
                or line.get('dash')!=expected or styles[key].get('dash')==expected):continue
        lengths=line.get('dash_pattern_px',[])
        if (len(lengths)!=(4 if expected=='dash_dot' else 2)
                or any(not isinstance(v,(int,float)) or not np.isfinite(v) or v<=0 for v in lengths)):continue
        changes[key]={'before':copy.deepcopy(styles[key]),'source_review':copy.deepcopy(value),
                      'measured_pattern':expected,'measured_lengths':list(lengths)}
        styles[key].update(dash=expected,dash_pattern_px=list(lengths))
        changes[key]['after']=copy.deepcopy(styles[key])
    return changes


def commit_refinement(previous, proposed, checks, visibility_evidence=None):
    """A correction is provisional until the existing post-edit critic accepts it."""
    result = copy.deepcopy(proposed)
    rejected = []
    for key, old in previous.items():
        if old == proposed[key]:
            continue
        check=(checks.get(key, {}).get('value') or {})
        # Changed-pixel MAE also measures contour displacement and newly exposed
        # neighbours. Keep it as diagnostics rather than a second paint judge.
        if (check.get('verdict') not in ('accept','revise') or
                check.get('verdict')=='revise' and not check.get('affected_fields')):
            result[key] = copy.deepcopy(old)
            rejected.append(key)
        elif check.get('verdict')=='revise':
            fields={'fill':('fill','fill_opacity'),'gradient':('fill','fill_opacity'),
                    'stroke_presence':('stroke','width_px','dash','dash_pattern_px'),
                    'stroke_color':('stroke',),'stroke_width':('width_px',),
                    'line_pattern':('dash','dash_pattern_px'),'occlusion':()}
            for field in check['affected_fields']:
                # A confirmed category correction must not disappear because
                # its new dash/gap lengths still need refinement. The old
                # category's lengths cannot supply a valid numerical rollback.
                if (field == 'line_pattern'
                        and (check.get('expected_line_pattern') == proposed[key].get('dash')
                             or (old.get('dash') == 'solid'
                                 and check.get('expected_line_pattern') in ('dash','dot','dash_dot')
                                 and proposed[key].get('dash') in ('dash','dot','dash_dot')))
                        and old.get('dash') != proposed[key].get('dash')):
                    # A dot/short-dash dispute still confirms visible gaps.
                    # Preserve that correction instead of reverting to the
                    # contradicted continuous line. The verdict stays revise.
                    continue
                for name in fields[field]:
                    if name in old:result[key][name]=copy.deepcopy(old[name])
                    else:result[key].pop(name,None)
            if result[key]!=proposed[key]:rejected.append(key)
    return result, rejected


def visibility_changes(source,before,after,previous,proposed,geometries):
    """Report transparency-related pixel differences in existing renders.

    Only changed pixels inside a face whose opacity was reduced participate.
    This diagnostic includes contour errors and changes in exposed neighbours;
    the existing source critic decides whether the paint correction commits.
    """
    keys=[k for k,old in previous.items() if isinstance(old.get('fill'),str) and
          (proposed[k].get('fill') is None or proposed[k].get('fill_opacity',1)<old.get('fill_opacity',1))]
    if not keys:return {}
    images=[]
    for path in (source,before,after):
        with Image.open(path) as im:images.append(np.asarray(im.convert('RGB')))
    if any(im.shape!=images[0].shape for im in images):raise ValueError('Visibility comparison grids differ')
    h,w=images[0].shape[:2];result={}
    for key in keys:
        geom=geometries[key]
        if geom.is_empty or geom.area<=0:continue
        b=geom.bounds;x0,y0=max(0,int(b[0])),max(0,int(b[1]));x1,y1=min(w,int(np.ceil(b[2]))+1),min(h,int(np.ceil(b[3]))+1)
        if x1<=x0 or y1<=y0:continue
        clean,old,new=[im[y0:y1,x0:x1].astype(float) for im in images]
        inside=raster(geom,[x0,y0,x1,y1])
        changed=(np.linalg.norm(new-old,axis=2)>24)&inside
        count=int(changed.sum())
        if count<8:continue
        before_error=float(np.linalg.norm(old-clean,axis=2)[changed].mean())
        after_error=float(np.linalg.norm(new-clean,axis=2)[changed].mean())
        result[key]={'changed_pixels':count,'before_error':before_error,'after_error':after_error,
                     'regressed':after_error>before_error*1.25+25}
    return result


def bounded_units(units, limit):
    """Bound reply size, retaining the whole component as visual context."""
    result=[]
    for unit in units:
        if len(unit['keys'])<=limit:
            result.append(unit);continue
        for index,start in enumerate(range(0,len(unit['keys']),limit)):
            result.append({**unit,'id':unit['id']+f'_batch_{index:02d}',
                'root':unit['root']/f'batch_{index:02d}',
                'keys':unit['keys'][start:start+limit],'context_keys':unit['keys']})
    return result


def affected_review_units(units, table, previous, current, previous_order, current_order):
    """Recheck changed targets and actual overlapping paint dependencies.

    Padded whole-object crops are context, not dependencies of every target.
    Keep that full context visible, but request only affected target decisions.
    Shared style groups remain jointly reviewed when any member is affected.
    """
    reordered = {key for index, key in enumerate(previous_order) if current_order[index] != key}
    changed = {key for key in table if previous[key] != current[key] or key in reordered}
    regions = {}
    for key, part in table.items():
        b = bounds(part)
        padding = 2+max(previous[key].get('width_px') or 0, current[key].get('width_px') or 0)/2
        regions[key] = shape_box(b[0]-padding, b[1]-padding, b[2]+padding, b[3]+padding)
    touched = changed | {key for key in table if any(regions[key].intersects(regions[peer]) for peer in changed)}
    result = []
    for unit in units:
        keys = [key for key in unit['keys'] if key in touched]
        if not keys:
            continue
        if unit['kind'] == 'shared_component_style':
            keys = list(unit['keys'])
        result.append({**unit, 'review_keys': keys})
    return result, changed, reordered


def observed_order(table, geoms, relations, key, behind=(), in_front_of=(), origin='critic', *, scene=None):
    """Apply a part observation only to its named parts, including across owners."""
    before=copy.deepcopy((relations.get('internal_model_edges',[]),relations.get('cross_part_model_edges',[])))
    edges=list(relations.get('internal_model_edges',[]))
    for back,front in [(key,p) for p in behind]+[(p,key) for p in in_front_of]:
        if back==front or not geoms[back].intersects(geoms[front]):continue
        back_owner,front_owner=table[back]['object_id'],table[front]['object_id']
        if back_owner!=front_owner:
            cross=[e for e in relations.get('cross_part_model_edges',[])
                   if (e['back'],e['front'])!=(front,back)]
            if not any((e['back'],e['front'])==(back,front) for e in cross):
                cross.append({'back':back,'front':front,'origin':origin})
            relations['cross_part_model_edges']=cross
            continue
        enclosure=table[front]['kind'] in ('rectangle','rounded_rectangle','ellipse')
        if enclosure and geoms[front].area>geoms[back].area and geoms[front].covers(geoms[back]):
            relations.setdefault('rejected_internal_orders',[]).append(
                {'back':back,'front':front,'origin':origin,'reason':'contained_part_behind_own_container'})
            continue
        edges=[e for e in edges if (e['back'],e['front'])!=(front,back)]
        if not any((e['back'],e['front'])==(back,front) for e in edges):
            edges.append({'back':back,'front':front,'origin':origin})
    relations['internal_model_edges']=edges
    return before!=(relations.get('internal_model_edges',[]),relations.get('cross_part_model_edges',[]))


def order_feedback(key, value, order):
    """Use the existing critic's explicit order, irrespective of style verdict."""
    if not value or value.get('verdict')=='uncertain':return value
    rank={k:i for i,k in enumerate(order)}
    pairs=[(key,p) for p in value.get('_behind_parts',())]+[(p,key) for p in value.get('_front_parts',())]
    if not any(a in rank and b in rank and rank[a]>rank[b] for a,b in pairs):return value
    return {**value,'verdict':'revise',
            'affected_fields':list(dict.fromkeys([*value.get('affected_fields',[]),'occlusion'])),
            'rendered_difference':value.get('rendered_difference') or 'Explicit source order differs from the supplied PPT order.',
            'suggestion':value.get('suggestion') or 'Apply the named source order without changing geometry or colour.'}


def run_case(scene, source, folder, runtime, options, pool, render_slots, *, assets=(), selected_objects=None,
             rules_only=False, pixel_slots=None, progress=lambda *a: None, line_evidence=None):
    if options.get('semantic_seed_root'):
        from .repair import run_case as repair_case
        return repair_case(scene,source,folder,runtime,options,pool,render_slots,assets=assets,selected_objects=selected_objects,
                           rules_only=rules_only,pixel_slots=pixel_slots,progress=progress,line_evidence=line_evidence)
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True); table = parts(scene)
    source = Path(source)
    if sha(source) != scene['source_sha256']:
        raise ValueError('Style source pixels do not match geometry')
    line_evidence = line_evidence or options.get('sam_evidence_root')
    fitter = NativeFitter(source, table, line_evidence, scene.get('case_id','unknown'), folder)
    supports = {k: row for k, row in fitter.evidence['parts'].items()
                if k in table and any(t['target'] for t in row['tiles'])}
    support_identity = fitter.identity
    seeded,seed_identity=unchanged_styles(scene,table,options.get('style_seed_root'),assets)
    identity = {'scene_sha256': digest(scene), 'source_sha256': sha(source), 'options': options,
                'style_protocol': model.PROTOCOL, 'component_group_protocol': grouping.PROTOCOL,
                'scheduling_protocol': SCHEDULING_PROTOCOL, 'line_protocol': LINE_PROTOCOL,
                'line_evidence': support_identity, 'pdf_crop_protocol': views.PDF_CROP_PROTOCOL,
                'assets_sha256': digest(list(assets)), 'selected_objects': selected_objects, 'rules_only': rules_only,
                'model': runtime.config['model'] if runtime else None,
                'timeout': runtime.config.get('request_timeout_seconds') if runtime else None,
                'style_seed':seed_identity,'style_policy':style_policy()}
    if (folder / 'identity.json').exists() and read(folder / 'identity.json') != identity:
        raise ValueError('Style case identity changed; create a new run')
    save(folder / 'identity.json', identity); save(folder / 'scene_source.json', scene)
    if (folder / 'summary.json').exists():
        existing = read(folder / 'summary.json')
        if existing['state'] in ('completed', 'completed_with_unresolved', 'rules_only'):
            if existing.get('styles_sha256') != sha(folder / 'styles.json'):
                raise ValueError('Saved styles changed')
            return existing
    started = time.monotonic(); timings = []

    def elapsed(stage, beginning):
        timings.append({'stage': stage, 'wall_seconds': time.monotonic() - beginning})
        save(folder / 'timings.json', {'stages': timings, 'elapsed_wall_seconds': time.monotonic() - started})
    selected = {k for k, p in table.items() if selected_objects is None or p['object_id'] in selected_objects}
    if selected_objects and not set(selected_objects) <= {p['object_id'] for p in table.values()}:
        raise ValueError('Selected object has no exportable geometry')
    model_enabled = runtime is not None and not rules_only
    calls = {'relations': 0, 'group': 0, 'select': 0, 'critic': 0, 'refine': 0,'pattern_observation':0}

    # 1. Whole-component occlusion; unresolved overlaps retain unknown order.
    progress('relations', 0, len(table)); stage_start = time.monotonic()
    geoms, relations = occlusion.discover(table, options)
    selected_components = {table[k]['object_id'] for k in selected}
    overlap = [p for p in relations['pending'] if p['kind'] == 'partial_overlap' and ({p['a'], p['b']} & selected_components)]
    if model_enabled:
        futures = {}
        for index, batch in enumerate(occlusion.relation_batches(overlap, options['relation_batch_pairs'])):
            keys = list(dict.fromkeys(k for p in batch for k in (p['a'], p['b'])))
            mapping = {f'C{i}': k for i, k in enumerate(keys)}; aliases = {k: a for a, k in mapping.items()}
            root = folder / 'relations' / f'batch_{index:03d}'
            try:
                images, units, _ = views.prepare(source, table, mapping, root / 'input', options, assets=assets,
                                                 component_members=relations['component_parts'])
            except Exception as exc:
                save(root / 'input_error.json', {'mapping': mapping, 'error': str(exc)})
                for pair in batch:
                    pair['reason'] = 'input_error: ' + str(exc)
                continue
            pair_table = {f'R{i}': [aliases[p['a']], aliases[p['b']]] for i, p in enumerate(batch)}
            p = model.prompt('relations', [{'id': a, 'object_id': oid,
                               'identity': scene['objects'][oid].get('candidate', {}).get('label', scene['objects'][oid].get('label','')),
                               'part_count': len(relations['component_parts'][oid])}
                              for a, oid in mapping.items()], units,
                             pairs=[{'id': a, 'parts': v} for a, v in pair_table.items()])
            futures[pool.submit(model.ask, runtime, root / 'calls', 'relations', p, images, list(pair_table), {}, options,
                                pairs=pair_table)] = (batch, mapping)
        calls['relations'] = len(futures)
        for f in as_completed(futures):
            batch, mapping = futures[f]
            try:
                result = f.result()
            except Exception as exc:
                result = {'valid': {}, 'errors': {'batch': str(exc)}}
            for i, pair in enumerate(batch):
                row = result['valid'].get(f'R{i}')
                if row and row['front'] is not None:
                    front = mapping[row['front']]; back = pair['b'] if front == pair['a'] else pair['a']
                    relations['edges'].append({'back': back, 'front': front, 'origin': 'model',
                        'front_paint':row.get('front_paint','unknown'),'evidence': row['evidence']})
                    relations['pending'].remove(pair)
                else:
                    pair['reason'] = row['evidence'] if row else result['errors'].get(f'R{i}', result['errors'].get('batch', 'no_relation'))
    # Uncertain visual order remains unknown; smaller area is not occlusion evidence.
    relations = occlusion.order(scene, table, relations); save(folder / 'occlusion.json', relations)
    elapsed('relations', stage_start)

    # 2. Rules for fill evidence and contour measurement for strokes, front to back.
    stage_start = time.monotonic()
    image = Image.open(source).convert('RGB'); _unused = None
    asset_masks = ([] if options.get('image_ownership_stage') == 'composition' else
        [shape_box(*a['bbox_2d']) for a in assets if len(a.get('bbox_2d', [])) == 4])

    def excluded(key, current):
        masks = [occlusion.exclusions(key, geoms, relations, current)]
        # Asset bboxes are conservative unknown regions, never claimed exact opaque masks.
        masks.extend(g for g in asset_masks if geoms[key].intersects(g))
        return unary_union(masks)

    def lookup(geometry):
        if geometry.is_empty:
            return None
        b = [int(v) for v in geometry.bounds]
        box = [max(0, b[0] - 1), max(0, b[1] - 1), min(image.width, b[2] + 2), min(image.height, b[3] + 2)]
        if box[2] <= box[0] or box[3] <= box[1]:
            return None
        mask = raster(geometry, box)

        def blocked(xy):
            x = np.rint(xy[:, 0]).astype(int) - box[0]; y = np.rint(xy[:, 1]).astype(int) - box[1]
            ok = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
            out = np.zeros(len(xy), bool); out[ok] = mask[y[ok], x[ok]]
            return out
        return blocked

    styles = {}; evidence = {}; records = {}
    for index, key in enumerate(reversed(relations['flat_order'])):
        state = scene['objects'][table[key]['object_id']]['state']
        try:
            with pixel_slots or nullcontext():
                ex = excluded(key, styles)
                e = rules.estimate(image, table[key], ex, options)
                line = {'measured': False, 'numeric_owner': NATIVE_PROTOCOL}
            e['line'] = line
            e['remaining_reasons'] = list(e['reasons'])
            e['needs_model'] = bool(e['remaining_reasons'])
            style = dict(e['style'])
            # Let the visual role choose colour before fitting any width. Gray
            # shadow pixels must never be converted into a wide coloured border.
            e['needs_model'] = bool(e['needs_model'] or style.get('stroke'))
            styles[key] = style; evidence[key] = e
            records[key] = {'origin': 'rules_then_native_fit', 'history': [{'stage': 'rules', 'style': copy.deepcopy(styles[key])}],
                            'selected_for_model': key in selected, 'geometry_state': state}
        except Exception as exc:
            styles[key] = {'fill': None, 'stroke': '#D000C0', 'width_px': 1.2, 'dash': 'solid'}
            evidence[key] = {'style': styles[key], 'closed': closed(table[key]), 'needs_model': True,
                             'reasons': ['rule_error: ' + str(exc)], 'remaining_reasons': ['rule_error: ' + str(exc)],
                             'rule_error': str(exc), 'line': {'measured': False, 'error': str(exc)},
                             'candidates': {'fills': [None], 'strokes': [None]}}
            records[key] = {'origin': 'diagnostic_fallback', 'history': [{'stage': 'rules', 'error': str(exc)}],
                            'selected_for_model': key in selected, 'geometry_state': state}
        if index % 10 == 0:
            progress('rules', index + 1, len(table))
    for key,style in seeded.items():
        styles[key]=style
        records[key]['origin']='unchanged_geometry_style_seed'
        records[key]['history'].append({'stage':'style_seed','style':copy.deepcopy(style)})
    save(folder / 'rules.json', evidence)
    elapsed('rules_and_lines', stage_start)

    # 3. Component groups and whole-object units (unchanged protocol).
    stage_start = time.monotonic()
    group_audit = grouping.discover(scene, source, table, selected, folder / 'component_groups',
                                    runtime if model_enabled else None, options, pool, assets)
    calls['group'] = len(group_audit['proposals']) if model_enabled else 0
    elapsed('component_groups', stage_start)
    grouped = {k for group in group_audit['groups'] for k in group['members']}
    work_units = []
    for obj in scene['objects'].values():
        keys = [k for k in obj['part_keys'] if k in table and k in selected]
        if keys and not grouped.intersection(keys):
            work_units.append({'id': obj['id'], 'keys': keys, 'root': folder / 'objects' / obj['id'], 'kind': 'object'})
    for group in group_audit['groups']:
        keys = [group['representative']] + [k for k in group['members'] if k != group['representative']]
        work_units.append({'id': group['id'], 'keys': keys, 'root': folder / 'component_groups' / group['id'],
                           'kind': 'shared_component_style'})
        for key in keys:
            records[key]['style_group'] = group['id']
    work_units=bounded_units(work_units,options['max_target_parts'])
    save(folder / 'style_units.json', [{**u, 'root': str(u['root'].relative_to(folder))} for u in work_units])

    def floor(keys, stage):
        minimum = options['minimum_stroke_width_px']; applied = {}
        for key in keys:
            style = styles[key]
            if style['stroke'] is not None and style['width_px'] < minimum:
                styles[key] = {**style, 'width_px': minimum}
                applied[key] = {'before_width_px': style['width_px'], 'after_width_px': minimum}
                records[key]['history'].append({'stage': 'minimum_stroke_width', 'revision': stage, **applied[key]})
        return applied
    floor(list(table), 'rules_baseline')

    def request(unit, mode, revision, snapshot, feedback_by_key=None):
        keys = unit.get('review_keys', unit['keys']); shared = unit['kind'] == 'shared_component_style'
        # Stable IDs for the whole compound: a batch owns its answers, but
        # may refer to a foreground face in another batch.
        mapping = {f'P{i}': k for i, k in enumerate(unit.get('context_keys', unit['keys']))}
        # Carry the existing relation decision into the style task that must
        # implement it. Show only intersecting neighbour parts, not an entire
        # large network that would shrink this target in the comparison.
        target_owners={table[k]['object_id'] for k in keys}
        overlaps=[e for e in relations.get('effective_edges',[])
                  if e['origin']=='model' and {e['back'],e['front']}&target_owners]
        neighbours={oid for e in overlaps for oid in (e['back'],e['front'])}-target_owners
        for k,p in table.items():
            if p['object_id'] in neighbours and any(geoms[k].intersects(geoms[t]) for t in keys):
                mapping.setdefault(f'P{len(mapping)}',k)
        # Batch repeated objects together, but never broadcast one member's
        # semantic colour/fill to its peers. Each displayed ID owns one answer.
        output_mapping = {a: [k] for a, k in mapping.items() if k in keys}
        if mode == 'refine':
            # Accepted neighbours remain visible context, never rewrite targets.
            # Numeric/order findings are handled by their existing owners below.
            output_mapping = {a: members for a, members in output_mapping.items()
                              if members[0] in (feedback_by_key or {}) and not numeric_only(members[0])}
        root = unit['root'] / revision / f'{mode}_000'
        native_pdf = snapshot['root'] / 'native.pdf' if snapshot else None
        base = snapshot['styles'] if snapshot else styles
        try:
            images, units, frame = views.prepare(source, table, mapping, root / 'input', options, native_pdf=native_pdf,
                                                 assets=assets, with_stroke_reference=False,context_keys=unit.get('context_keys'))
            focus={a:m[0] for a,m in output_mapping.items()}
            detail,detail_targets=views.target_details(source,table,focus,root/'detail',
                frame['source_to_crop_scale'],native_pdf=native_pdf)
            images.extend(detail)
            with pixel_slots or nullcontext():
                ranked = rules.ranked_source_colors(image, {k: table[k] for k in keys},
                                                    lambda k: excluded(k, base), options['color_candidate_limit'],
                                                    separate_roles=options.get('color_roles_enabled', False)) if mode == 'select' else None
        except Exception as exc:
            save(root / 'input_error.json', {'mapping': mapping, 'error': str(exc)})
            return {k: {'value': None, 'error': 'input_error: ' + str(exc), 'path': str(root.relative_to(folder))} for k in keys}
        if ranked is not None:
            save(root / 'color_candidates.json', ranked)
        sam_views={}
        if mode=='critic' and options.get('sam_visual_pattern'):
            # Reuse existing tiles in the existing critic, not another model
            # round. Bound image count; remaining ambiguous parts retain the
            # existing conditional observation path.
            uncertain=[k for k in keys if base[k].get('stroke') and k in supports and
                (not evidence[k]['line'].get('dash_reliable') or
                 evidence[k]['line'].get('pattern_sam_rgb_conflicts',0))]
            for key in uncertain[:3]:
                alias=next(a for a,k in mapping.items() if k==key)
                extra,audit=views.sam_pattern_views(fitter.evidence,key,root/'sam_views'/alias,limit=1)
                images.extend([(alias+': '+label,path) for label,path in extra])
                if audit:sam_views[key]=digest(audit)
        scale = frame['source_to_crop_scale']; targets = []
        table_view = {a: table[k] for a, k in mapping.items()}
        for alias, members in output_mapping.items():
            key = members[0]; line = evidence[key]['line']
            target = {'id': alias, 'closed': closed(table[key]), 'kind': table[key]['kind'],
                      'contour':views.contour_identity(table[key]),'source_bbox': list(bounds(table[key]))}
            if alias in detail_targets:target['detail_card']=detail_targets[alias]
            if table[key]['kind']=='cylinder':
                target['native_primitive']='cylinder'
                target['native_geometry_scope']='Complete body and top rim seam; a separate ellipse, if present, is the independently styled top face.'
            if mode!='critic' and target['closed'] and evidence[key].get('fill_candidates'):
                # References for selection/correction only. The critic judges
                # source/render pixels independently of this fallible sampler.
                target['source_interior_rgb'] = {
                    'colors': evidence[key]['fill_candidates'][:3],
                    'purity': evidence[key].get('evidence', {}).get('fill_purity'),
                    'rule': 'Fallible samples inside fitted geometry, possibly including foreground '
                            'or background. Decide fill from visible CLEAN regions of this part; '
                            'neither RGB rank nor SAM mask overrides visual ownership.'}
            if ranked and key in ranked.get('part_colors',{}):
                target['source_color_roles']=ranked['part_colors'][key]
            if shared:
                target['similar_components'] = [a for a in mapping if a != alias]
            if mode in ('select', 'refine'):
                fixed = fixed_fields(line)
                target['numeric_owner'] = {'method': NATIVE_PROTOCOL, 'fields': fixed,
                                           'critic_can_request_refit': True}
                table_view[alias] = {**table[key], '_fixed': fixed}
            if mode == 'refine':
                shown = copy.deepcopy(base[key]); effective = pattern(shown)
                if effective:
                    shown['dash_pattern_px'] = effective
                target['previous_style'] = {k: shown[k] for k in ('fill', 'stroke')}
                if 'fill_opacity' in shown:target['previous_style']['fill_opacity']=shown['fill_opacity']
                target['numeric_refit_feedback'] = (feedback_by_key or {}).get(key, [])
            if mode == 'critic':
                # Parameters without lengths: the critic judges pixels, not numbers.
                target['current_style'] = {k: base[key][k] for k in ('fill', 'stroke', 'dash')}
                if 'fill_opacity' in base[key]:target['current_style']['fill_opacity']=base[key]['fill_opacity']
                # Sample agreement describes the estimator's source profiles,
                # not whether the PPT matches. Keep it in line diagnostics;
                # exposing it here caused visible defects to be dismissed.
            gradients = [f for f in evidence[key]['candidates']['fills'] if isinstance(f, dict)]
            if mode == 'select' and gradients:
                target['gradient_candidates'] = gradients
            targets.append(target)
        feedback = None
        if mode == 'refine':
            feedback = [{'id': alias, 'problems': list({digest(p): p for k in members for p in (feedback_by_key or {}).get(k, [])}.values())}
                        for alias, members in output_mapping.items() if any((feedback_by_key or {}).get(k) for k in members)]
        colors = [entry['color'] for entry in ranked['colors']] if ranked else []
        p = model.prompt(mode, targets, units, feedback=feedback, color_candidates=colors)
        p['overlap_context']=[{'back':e['back'],'front':e['front'],
            'back_parts':[a for a,k in mapping.items() if table[k]['object_id']==e['back']],
            'front_parts':[a for a,k in mapping.items() if table[k]['object_id']==e['front']]}
            for e in overlaps]
        p['context_ids'] = list(mapping)
        if mode=='critic' and unit.get('review_keys') is not None:
            p['read_only_context_targets']=[alias for alias,key in mapping.items() if key not in keys]
            p['context_rule']='All original source and complete object geometry stay visible. Only targets need verdicts; other IDs remain read-only context.'
        if sam_views:p['sam_visual_targets']=[a for a,k in mapping.items() if k in sam_views]
        if ranked and 'fill_colors' in ranked:
            p['source_color_roles'] = {k: ranked[k] for k in ('fill_colors','stroke_colors','role_rule')}
        save(root / 'job.json', {'mapping': mapping, 'output_mapping': output_mapping, 'unit_kind': unit['kind'], 'prompt': p,
                                 'geometry_digest': digest({k: table[k] for k in keys}),
                                 'render_snapshot': str(snapshot['root'].relative_to(folder)) if snapshot else None,
                                 'render_sha256': frame['rendered_source']['sha256'] if native_pdf else None,
                                 'rendered_source': frame['rendered_source'], 'frame_sha256': sha(root / 'input/frame.json')})
        try:
            res = model.ask(runtime, root / 'calls', mode, p, images, list(output_mapping),
                            table_view or {a: table[m[0]] for a, m in output_mapping.items()}, options,
                            source_to_crop_scale=scale)
        except Exception as exc:
            res = {'valid': {}, 'errors': {a: str(exc) for a in output_mapping}}
        for alias,value in res['valid'].items():
            key=mapping[alias]
            if value.get('expected_line_pattern') and key in sam_views:
                value['sam_visual_evidence']=sam_views[key]
            if 'behind' in value:value['_behind_parts']=[mapping[a] for a in value.pop('behind')]
            if 'in_front_of' in value:value['_front_parts']=[mapping[a] for a in value.pop('in_front_of')]
        return {k: {'value': copy.deepcopy(res['valid'].get(a)), 'error': res['errors'].get(a),
                    'path': str(root.relative_to(folder))} for a, members in output_mapping.items() for k in members}

    def batch(units, mode, revision, snapshot=None, feedback=None):
        futures = {pool.submit(request, u, mode, revision, snapshot, feedback): u for u in units}
        calls[mode] += len(futures); done = 0; results = {}
        for future in as_completed(futures):
            unit = futures[future]
            try:
                results[unit['id']] = future.result()
            except Exception as exc:
                results[unit['id']] = {k: {'value': None, 'error': str(exc)} for k in unit['keys']}
            for key, result in results[unit['id']].items():
                records[key]['history'].append({'stage': mode, 'revision': revision, **result})
            done += 1; progress(mode + '_' + revision, done, len(futures))
        return results

    confirmed_patterns = {}
    visual_pattern_feedback = {}
    source_patterns = {}

    def fit_key(key, value, feedback=None):
        # A coarse rule proposal has not established a broken stroke. Only a
        # confirmed numerical measurement or explicit critic category survives
        # an inconclusive refit; unknown does not promote the rule's guess.
        feedback=list(feedback or [])
        if key in visual_pattern_feedback and not any(r.get('sam_visual_evidence') for r in feedback):
            feedback.append(visual_pattern_feedback[key])
        previous_confirmed=key in confirmed_patterns
        previous=confirmed_patterns.get(key, {'dash':'solid'})
        value={**value,'dash':previous['dash']}
        value.pop('dash_pattern_px',None)
        if previous.get('dash_pattern_px'):
            value['dash_pattern_px']=previous['dash_pattern_px']
        if feedback and value.get('width_px') is None and any(row.get('expected_stroke_width') for row in feedback):
            value['width_px']=styles[key].get('width_px')
        with pixel_slots or nullcontext():
            hidden=excluded(key, styles)
            fitted, line = fitter.fit(key, value, feedback=feedback, excluded=lookup(hidden),
                visibility_id=digest(hidden.wkb_hex),source_pattern=source_patterns.get(key))
        evidence[key]['line'] = line
        if line.get('dash_reliable') or line.get('pattern_origin') in (
                'critic_category_with_unknown_numeric_pattern','glm_sam_rgb_visual_observation','critic_source_render_observation',
                'source_style_with_conflicting_numeric_pattern'):
            confirmed_patterns[key]={field:copy.deepcopy(fitted[field]) for field in ('dash','dash_pattern_px') if field in fitted}
        line['pattern_prior_origin']='confirmed_result' if previous_confirmed else 'unconfirmed_default'
        return fitted

    def apply(unit, results, origin, feedback=None):
        """Shared semantics, independent per-member numerical fits."""
        before = {k: copy.deepcopy(styles[k]) for k in unit['keys']}
        previous_order=copy.deepcopy((relations.get('internal_model_edges',[]),relations.get('edges',[])))
        for key, result in results.items():
            value = result['value']
            if value is None:
                continue
            value=copy.deepcopy(value)
            if origin in ('model_selection','model_refine') and value.get('dash') in ('solid','dash','dot','dash_dot'):
                source_patterns[key]=value['dash']
            behind=value.pop('_behind_parts',None)
            if behind is not None and origin=='model_selection':
                observed_order(table,geoms,relations,key,behind=behind,origin=origin,scene=scene)
            if value.get('issue'):
                records[key].setdefault('issues', []).append(value['issue'])
            with pixel_slots or nullcontext():
                value,audit=fitter.supported_stroke_colour([key],value,excluded=lookup(excluded(key,styles)),paint=styles)
            if audit:
                records[key]['history'].append({'stage':'source_stroke_colour_validation','origin':origin,**audit})
            styles[key] = fit_key(key, value, (feedback or {}).get(key))
            records[key]['origin'] = origin
        floor(unit['keys'], origin)
        return any(before[k] != styles[k] for k in unit['keys']) or previous_order!=(relations.get('internal_model_edges',[]),relations.get('edges',[]))

    snapshots = []

    def render(tag):
        stage_start = time.monotonic()
        # Apply before both rendering and critic; the saved style is exactly
        # what PowerPoint displays. A cap leaves every thinner width untouched.
        for key in styles:
            styles[key], capped = limit_width(styles[key], image.width, options)
            if capped:
                records[key]['history'].append({'stage': 'maximum_stroke_width', 'revision': tag, **capped})
        relations.update(occlusion.order_internal_by_paint(scene, table, relations, styles, geoms,options['geometry_tolerance_px'],source=image))
        save(folder / 'occlusion.json', relations)
        snapshot = {'root': folder / 'snapshots' / f'snapshot_{len(snapshots):02d}_{tag}_{digest(styles)[:12]}',
                    'styles': copy.deepcopy(styles),'order':list(relations['flat_order']),
                    'internal_model_edges':copy.deepcopy(relations.get('internal_model_edges',[]))}
        save(snapshot['root'] / 'styles.json', snapshot['styles'])
        export(scene, snapshot['styles'], relations, snapshot['root'], source, options, render_slots)
        save(snapshot['root'] / 'calibration.json', {'enabled': False, 'reason': 'replaced_by_line_measurement'})
        publish(folder, snapshot['root'], len(snapshots)); snapshots.append(snapshot)
        elapsed('render_' + tag, stage_start)
        return snapshot

    checks = {}; selection_errors = {}
    # 4. One select pass for units whose evidence the measurements do not settle.
    if model_enabled:
        stage_start = time.monotonic()
        overlap_fronts={e['front'] for e in relations.get('effective_edges',[])
                        if e.get('front_paint') in ('translucent','unfilled')}
        need = [u for u in work_units if not all(k in seeded for k in u['keys']) and
                (len(u['keys']) > 1 or any(evidence[k]['needs_model'] or
                table[k]['object_id'] in overlap_fronts for k in u['keys']))]
        need=[{**u,'context_keys':u.get('context_keys',u['keys']),
               'keys':[k for k in u['keys'] if k not in seeded]} for u in need]
        results = batch(need, 'select', 'revision_00')
        for unit in need:
            apply(unit, results[unit['id']], 'model_selection')
            for key, result in results[unit['id']].items():
                if result['value'] is None:
                    selection_errors[key] = [{'selection_error': result['error']}]
        elapsed('select', stage_start)
    # 5. Render all final first-pass styles once, then critic every unit on it.
    # All semantic styles and observed face orders are now available. Resolve
    # their attached dividers before measuring, using the same order as render.
    relations.update(occlusion.order_internal_by_paint(scene, table, relations, styles, geoms,options['geometry_tolerance_px'],source=image))
    progress('native_fit', 0, len(table))
    stage_start = time.monotonic()
    for i, key in enumerate(table):
        if key not in seeded:styles[key] = fit_key(key, styles[key])
        if i % 10 == 0:
            progress('native_fit', i + 1, len(table))
    save(folder / 'rules.json', evidence)
    elapsed('native_fit', stage_start)
    first = render('initial')
    if model_enabled and options.get('critic_enabled', True):
        stage_start = time.monotonic()
        results = batch(work_units, 'critic', 'revision_00', first)
        for unit in work_units:
            checks.update(results[unit['id']])
        elapsed('critic', stage_start)
        # 6. One refine for units with an actionable critic verdict or a failed select.
        feedback = {}
        for key, result in checks.items():
            value = order_feedback(key, result['value'], first['order'])
            if value and value['verdict'] == 'revise':
                feedback.setdefault(key, []).append(copy.deepcopy(value))
        for key, rows in selection_errors.items():
            feedback.setdefault(key, []).extend(rows)
        # Reuse the first actual render. Only unresolved numerical patterns
        # without an existing categorical correction get this small observation.
        def inspect_pattern(key):
            root=folder/'pattern_observations'/digest(key)[:16]
            images,_,_=views.prepare(source,table,{'P0':key},root/'input',options,
                native_pdf=first['root']/'native.pdf',assets=assets,with_stroke_reference=False)
            audit=[]
            if options.get('sam_visual_pattern'):
                extra,audit=views.sam_pattern_views(fitter.evidence,key,root/'sam_views')
                images.extend(extra)
            result=model.check_uncertain_pattern(runtime,root/'calls',images,
                styles[key]['dash'],evidence[key]['line'],options)
            return {**result,**({'sam_visual_evidence':digest(audit)} if audit else {})}
        ambiguous=[k for k in selected if not any(r.get('sam_visual_evidence') for r in feedback.get(k,[])) and
            (checks.get(k,{}).get('value') or {}).get('verdict')!='accept' and
            (any(set(r.get('affected_fields',[]))&{'stroke_width','line_pattern'} for r in feedback.get(k,[])) or
             (checks.get(k,{}).get('value') or {}).get('verdict')=='uncertain') and
            styles[k].get('stroke') and (evidence[k]['line'].get('measured') or
                options.get('sam_visual_pattern') and k in supports) and
            ((not evidence[k]['line'].get('dash_reliable') and
              not any(row.get('expected_line_pattern') for row in feedback.get(k,[]))) or
              options.get('sam_visual_pattern') and (
                  (evidence[k]['line'].get('pattern_sam_rgb_conflicts') or 0)>0 or
                  any(row.get('expected_line_pattern') for row in feedback.get(k,[]))))]
        pattern_jobs={pool.submit(inspect_pattern,k):k for k in ambiguous} if options['refine_rounds'] and options.get('conditional_pattern_critic') else {}
        calls['pattern_observation']=len(pattern_jobs)
        for future in as_completed(pattern_jobs):
            key=pattern_jobs[future]
            try:observation=future.result()
            except Exception as exc:observation={'source_pattern':'unknown','error':str(exc)}
            records[key]['pattern_observation']=observation
            category=observation['source_pattern']
            direction=observation.get('expected_stroke_width')
            weight=observation.get('stroke_weight','unknown')
            primary={r['expected_stroke_width'] for r in feedback.get(key,[])
                     if r.get('expected_stroke_width')}
            # This helper exists to resolve missing pattern evidence, not to
            # silently reverse an explicit finding from the actual-PPT critic.
            if primary:
                if weight!='unknown' and (weight=='match' or direction not in primary):
                    records[key]['width_observation_conflict']={
                        'critic':sorted(primary),'auxiliary':weight,'resolution':'retain_primary_critic'}
                direction=None
            if not primary and weight in ('match','increase','decrease'):
                previous=feedback.get(key,[])
                for row in previous:
                    row.pop('expected_stroke_width',None)
                    if weight=='match' and 'stroke_width' in row.get('affected_fields',[]):
                        row['affected_fields']=[field for field in row['affected_fields'] if field!='stroke_width']
                remaining=[row for row in previous if row.get('affected_fields') or row.get('selection_error')]
                if remaining:feedback[key]=remaining
                else:
                    feedback.pop(key,None)
                    if weight=='match' and previous:
                        records[key]['width_critic_resolution']={'previous':checks[key]['value'],'source_observation':observation}
                        checks[key]={**checks[key],'value':{'verdict':'accept'},'resolution':'source_observation'}
            if direction:
                feedback.setdefault(key,[]).append({'verdict':'revise','affected_fields':['stroke_width'],
                    'expected_stroke_width':direction,'source_evidence':observation['source_evidence'],
                    'rendered_difference':'Rendered stroke weight differs from CLEAN.',
                    'suggestion':'Select the source-supported numerical width mode in the observed direction.'})
            if category=='unknown':continue
            if observation.get('sam_visual_evidence'):
                visual_pattern_feedback[key]={'verdict':'revise','affected_fields':['line_pattern'],
                    'expected_line_pattern':category,'sam_visual_evidence':observation['sam_visual_evidence'],
                    'source_evidence':observation['source_evidence']}
            if category==styles[key]['dash']:
                confirmed_patterns[key]={k:copy.deepcopy(styles[key][k]) for k in ('dash','dash_pattern_px') if k in styles[key]}
                evidence[key]['line']['pattern_confirmation']='source_observation'
                continue
            feedback.setdefault(key,[]).append({'verdict':'revise','affected_fields':['line_pattern'],
                'expected_line_pattern':category,'source_evidence':observation['source_evidence'],
                **({'sam_visual_evidence':observation['sam_visual_evidence']} if observation.get('sam_visual_evidence') else {}),
                'rendered_difference':'Current line pattern differs from source observation.',
                'suggestion':'Use the source-confirmed categorical line pattern.'})
        targets = [u for u in work_units if options['refine_rounds'] and any(k in feedback for k in u['keys'])]
        proposed={(key,p) for key,rows in feedback.items() for row in rows for p in row.get('_behind_parts',())}
        proposed.update((p,key) for key,rows in feedback.items() for row in rows for p in row.get('_front_parts',()))
        opposed={pair for pair in proposed if pair[::-1] in proposed}
        relations['order_feedback_conflicts']=[list(pair) for pair in sorted(opposed)]
        for key,rows in feedback.items():
            for row in rows:
                if '_behind_parts' in row:row['_behind_parts']=[p for p in row['_behind_parts'] if (key,p) not in opposed]
                if '_front_parts' in row:row['_front_parts']=[p for p in row['_front_parts'] if (p,key) not in opposed]
        stage_start = time.monotonic()
        # Moving an unfilled face in front cannot hide any rear paint. Let the
        # existing semantic correction inspect its fill; never invent white.
        paint_dependencies=[]
        for key,rows in feedback.items():
            if styles[key].get('fill') is not None or geoms[key].area<=0:continue
            for row in rows:
                peers=[p for p in row.get('_front_parts',())
                       if p in geoms and geoms[key].intersects(geoms[p])]
                if row.get('verdict')=='revise' and 'occlusion' in row.get('affected_fields',[]) and peers:
                    row['affected_fields']=list(dict.fromkeys([*row['affected_fields'],'fill']))
                    paint_dependencies.append({'part':key,'rear_parts':peers,
                        'reason':'unfilled_foreground_requires_source_paint_review'})
        if paint_dependencies:save(folder/'paint_dependencies.json',paint_dependencies)
        # Width/pattern-only findings belong to the existing numerical fitter.
        # Do not send an additional semantic-colour request that cannot fix them.
        numeric_only=lambda key:bool(feedback.get(key)) and all(
            row.get('affected_fields') and set(row['affected_fields'])<={'stroke_width','line_pattern','occlusion'}
            for row in feedback[key])
        semantic=[u for u in targets if any(k in feedback and not numeric_only(k) for k in u['keys'])]
        results = batch(semantic, 'refine', 'revision_01', first, feedback)
        order_changed=set()
        for unit in targets:
            for key in unit['keys']:
                for row in feedback.get(key,[]):
                    if row.get('verdict')=='revise' and 'occlusion' in row.get('affected_fields',[]):
                        if observed_order(table,geoms,relations,key,row.get('_behind_parts',()),
                                          row.get('_front_parts',()),'critic_source_order',scene=scene):
                            order_changed.add(unit['id'])
        relations.update(occlusion.order_internal_by_paint(scene,table,relations,styles,geoms,options['geometry_tolerance_px'],source=image))
        changed=[]
        for unit in targets:
            if unit in semantic:
                modified=apply(unit,results[unit['id']],'model_refine',feedback)
            else:
                before={k:copy.deepcopy(styles[k]) for k in unit['keys']}
                for key in unit['keys']:
                    if any(set(r.get('affected_fields',[]))&{'stroke_width','line_pattern'} for r in feedback.get(key,[])):
                        styles[key]=fit_key(key,styles[key],feedback[key])
                floor(unit['keys'],'numeric_refine')
                modified=any(before[k]!=styles[k] for k in unit['keys'])
            if unit in semantic:
                # The same unit may also contain width/pattern-only findings.
                for key in unit['keys']:
                    if numeric_only(key) and any(set(r.get('affected_fields',[]))&{'stroke_width','line_pattern'} for r in feedback.get(key,[])):
                        before_numeric=copy.deepcopy(styles[key])
                        styles[key]=fit_key(key,styles[key],feedback[key])
                        modified=modified or before_numeric!=styles[key]
            if modified or unit['id'] in order_changed:changed.append(unit)
        elapsed('refine', stage_start)
        if changed:
            second = render('refined')
            stage_start = time.monotonic()
            affected,changed_parts,reordered=affected_review_units(work_units,table,first['styles'],styles,
                                                                  first['order'],second['order'])
            save(folder/'context_recheck.json',{'changed_parts':sorted(changed_parts),
                'reordered_parts':sorted(reordered),'reviewed_units':[u['id'] for u in affected],
                'reviewed_targets':{u['id']:u['review_keys'] for u in affected},
                'dependency_scope':'changed_target_or_actual_paint_overlap; whole_object_context_retained',
                'render_snapshot':str(second['root'].relative_to(folder))})
            results = batch(affected, 'critic', 'revision_01', second)
            for unit in affected:
                checks.update(results[unit['id']])
                for key in unit.get('review_keys',unit['keys']):
                    if unit['id'] in {u['id'] for u in changed}:
                        records[key]['selected_style_revision'] = str((unit['root'] / 'revision_01').relative_to(folder))
            elapsed('critic_refined', stage_start)
            visibility=visibility_changes(source,first['root']/'native.png',second['root']/'native.png',
                                          first['styles'],styles,geoms)
            pattern_consensus=resolve_pattern_consensus(styles,checks,evidence)
            for key,change in pattern_consensus.items():
                records[key]['history'].append({'stage':'final_pattern_consensus',**change,
                    'extra_model_calls':0,'final_render_reviewed':False})
            committed, rejected = commit_refinement(first['styles'], styles, checks,visibility)
            save(folder/'visibility_commit.json',visibility)
            # A remaining colour/one-pair defect does not invalidate every
            # previously observed order involving that part. Apply only the
            # explicit pair corrections from this same render review.
            reviewed_edges=copy.deepcopy((relations.get('internal_model_edges',[]),relations.get('edges',[])))
            for key,check in checks.items():
                value=check.get('value') or {}
                if value.get('verdict') not in ('accept','revise'):continue
                observed_order(table,geoms,relations,key,value.get('_behind_parts',()),
                               value.get('_front_parts',()),origin='render_review',scene=scene)
            order_changed=reviewed_edges!=(relations.get('internal_model_edges',[]),relations.get('edges',[]))
            if rejected or order_changed or pattern_consensus:
                styles.clear(); styles.update(committed)
                save(folder/'refinement_commit.json', {'rolled_back_parts':rejected,
                     'order_rollback':False,'order_corrected':order_changed,'reason':'commit_explicit_render_findings',
                     'pattern_consensus':pattern_consensus,
                     'visibility_evidence':visibility,'extra_model_calls':0})
                for key in rejected:
                    records[key]['history'].append({'stage':'rollback_unaccepted_refinement',
                                                   'rejected_critic':checks.get(key), 'style':copy.deepcopy(styles[key]),
                                                   'reason':'restore_disputed_fields_preserve_other_reviewed_changes'})
                    # This final mixture was not reviewed; never report acceptance.
                    checks[key]={'value':{'verdict':'uncertain'}, 'error':'unaccepted_refinement_rolled_back'}
                render('committed')
    latest = snapshots[-1]
    if latest['styles'] != styles:
        raise AssertionError('Published snapshot does not hold the final styles')

    for key, record in records.items():
        value = checks.get(key, {}).get('value')
        state = ('context_only' if key not in selected else 'rules_only' if not model_enabled else
                 'styled_unreviewed' if not options.get('critic_enabled', True) else
                 'critic_' + value['verdict'] if value else 'unresolved')
        if state=='critic_accept' and not evidence[key]['line'].get('dash_reliable') and record.get('pattern_observation',{}).get('source_pattern')=='unknown':
            state='critic_uncertain'
        record.update(state=state, style=copy.deepcopy(styles[key]), critic=checks.get(key, {}),
                      line=copy.deepcopy(evidence[key]['line']))
        record.setdefault('selected_style_revision', 'revision_00')
        record['invisible_geometry'] = styles[key]['fill'] is None and styles[key]['stroke'] is None
    final = str(latest['root'].relative_to(folder))
    save(folder / 'styles.json', {'schema_version': 'shape_integration.styles.2', 'input': identity,
                                  'length_unit': 'original_source_pixels',
                                  'minimum_stroke_width_px': options['minimum_stroke_width_px'], 'state': 'finished',
                                  'maximum_stroke_width_pt': options.get('maximum_stroke_width_pt'),
                                  'selected_revision': final, 'parts': records,
                                  'style_groups': [{**g, 'style': styles[g['representative']]} for g in group_audit['groups']],
                                  'visual_accuracy_verified': False})
    unresolved = [k for k in selected if records[k]['state'] != 'critic_accept']
    order_conflicts=relations['order_conflicts']+relations.get('internal_style_conflicts',[])+relations.get('order_feedback_conflicts',[])
    summary = {'state': 'rules_only' if not model_enabled else
               'completed_with_unresolved' if unresolved or relations['pending'] or order_conflicts else 'completed',
               'parts': len(table), 'targeted_parts': len(selected),
               'critic_accepted': sum(records[k]['state'] == 'critic_accept' for k in selected),
               'unresolved_parts': len(unresolved), 'relation_unresolved': len(relations['pending']),
               'order_conflicts': len(order_conflicts),
               'gradients': sum(isinstance(styles[k]['fill'], dict) for k in selected),
               'line_measured_parts': sum(bool(evidence[k]['line'].get('present')) for k in table),
               'line_fixed_width': 0, 'line_fixed_pattern': 0,
               'native_fitted_strokes': sum(bool(evidence[k]['line'].get('measured')) and styles[k]['stroke'] is not None for k in table),
               'filled_compact_parts': sum(evidence[k]['line'].get('role') == 'filled_compact' for k in table),
               'internal_paint_edges': len(relations.get('internal_style_edges', [])),
               'source_colour_corrections':sum(any(e.get('stage')=='source_stroke_colour_validation' and e.get('changed') for e in records[k]['history']) for k in table),
               'sam_supported_parts': len(supports), 'model_calls': calls, 'renders': len(snapshots),
               'component_style_groups': len(group_audit['groups']), 'grouped_components': len(grouped),
               'grouping_errors': len(group_audit['errors']), 'scheduling_protocol': SCHEDULING_PROTOCOL,
               'completed_units': len(work_units), 'selected_revision': final,
               'styles_sha256': sha(folder / 'styles.json'), 'updated_utc': now(), 'visual_accuracy_verified': False}
    if digest(scene) != identity['scene_sha256']:
        raise AssertionError('Style recovery mutated geometry')
    save(folder / 'summary.json', summary)
    lines_md = ['# 样式恢复结果', '', f'状态：`{summary["state"]}`。模型接受不等于视觉准确率。', '',
                f'目标部件 {len(selected)}，critic 接受 {summary["critic_accepted"]}，未决 {len(unresolved)}；几何状态不变。',
                f'模型调用 {calls}；整图渲染 {len(snapshots)} 次。', '',
                '[样式与状态](styles.json) · [组件分组](component_groups/groups.json) · [遮挡关系](occlusion.json) · '
                '[可编辑 PPTX](styled.pptx) · [SVG](styled.svg) · [矢量 PDF](native.pdf)', '',
                '![原图与实际 PPT 渲染](comparison.png)', '', '| 对象/部件 | 样式状态 | 来源 |', '| --- | --- | --- |']
    for k in relations['flat_order']:
        if k in selected:
            lines_md.append(f'| {table[k]["object_id"]}/{table[k]["id"]} | {records[k]["state"]} | {records[k]["origin"]} |')
    (folder / 'REPORT.zh-CN.md').write_text('\n'.join(lines_md) + '\n')
    progress('completed', len(selected), len(selected))
    return summary
