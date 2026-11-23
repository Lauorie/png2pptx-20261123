#!/usr/bin/env python3
"""Trial bbox separation and transactional parent/child reconstruction ownership."""
import argparse
import copy
import fcntl
import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from core import area, intersection, inside, union, gap, crop_box
from evidence import frame, local_box, save
from pipeline_frames import Crop
from protocol import KINDS, box as valid_box
from transport.common import read_json as read, write_json as write, sha256
from transport.model import Model
from visual_labels import render

SYSTEM = ('Inspect actual scientific figure pixels. Completeness comes before deduplication. '
          'Identify complete physical arrows before assigning reconstruction ownership. '
          'All bbox coordinates use absolute TARGET pixels; backend performs conversions. Return compact JSON.')


def now(): return datetime.now(timezone.utc).isoformat()


def iou(a, b):
    inter = intersection(a, b)
    return inter/max(1e-9, area(a)+area(b)-inter)


def local_view(size, bounds, cfg):
    w, h = bounds[2]-bounds[0], bounds[3]-bounds[1]
    edge = max(cfg['local_min_edge'], min(cfg['local_max_edge'], round(max(w, h)*cfg['local_scale'])))
    return Crop(tuple(size), tuple(bounds), tuple(max(2, round(v*edge/max(w, h))) for v in (w, h)))


def put_view(root, cid, folder, bounds, cfg):
    native = Image.open(root/'inputs'/cid/'source.png').convert('RGB')
    view = local_view(native.size, bounds, cfg)
    save(view.image(native), folder/'target.png')
    write(folder/'view.json', view.metadata(native.size))
    return view


def spatial_tiles(parent_bbox, size, edge=96, padding=24):
    """Disjoint ownership cores with overlapping observation windows."""
    x0, y0 = max(0, math.floor(parent_bbox[0])), max(0, math.floor(parent_bbox[1]))
    x1, y1 = min(size[0], math.ceil(parent_bbox[2])+1), min(size[1], math.ceil(parent_bbox[3])+1)
    nx, ny = max(1, math.ceil((x1-x0)/edge)), max(1, math.ceil((y1-y0)/edge))
    out = []
    for iy in range(ny):
        for ix in range(nx):
            core = [round(x0+(x1-x0)*ix/nx), round(y0+(y1-y0)*iy/ny),
                    round(x0+(x1-x0)*(ix+1)/nx), round(y0+(y1-y0)*(iy+1)/ny)]
            crop = [max(0, core[0]-padding), max(0, core[1]-padding),
                    min(size[0], core[2]+padding), min(size[1], core[3]+padding)]
            out.append({'id': f'T{len(out)+1:03d}', 'core': core, 'crop': crop})
    return out


def in_core(bbox, core):
    x, y = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
    return core[0] <= x < core[2] and core[1] <= y < core[3]


def nearby_existing(parent, baseline, limit):
    """Only settled identities can own a region; unresolved parent cycles are forbidden."""
    b = parent['bbox_native']
    candidates = [o for o in baseline if o['id'] != parent['id'] and o['kind'] != 'dense_region'
                  and intersection(b, o['bbox_native']) > 0]
    candidates.sort(key=lambda o: (-max(iou(b, o['bbox_native']), inside(o['bbox_native'], b)*.6), o['id']))
    return candidates[:limit], candidates[limit:]


def ask(model, folder, prompt, images, validator):
    started = time.time()
    result = model.ask(folder/'request', prompt, images, validator)
    write(folder/'result.json', result)
    write(folder/'timing.json', {'seconds_this_execution': time.time()-started, 'updated_at': now()})
    return result


def validate_plan(a):
    if a.get('action') not in {'split_spatial', 'split_routes', 'keep_group', 'uncertain'}:
        raise ValueError('Invalid partition action')
    if a.get('bbox_separability') not in {'easy', 'hard', 'uncertain'}:
        raise ValueError('bbox_separability required')
    if not all(isinstance(a.get(k), str) and a[k].strip() for k in ('reason', 'target_scope')):
        raise ValueError('reason and target_scope required')
    if a['action'].startswith('split') and a['bbox_separability'] != 'easy':
        raise ValueError('Only easy bbox separability permits splitting')
    return a


def plan_region(root, cid, parent, cfg, model):
    folder = root/'cases'/cid/'regions'/parent['display_id']/'plan'
    size = Image.open(root/'inputs'/cid/'source.png').size
    cb = crop_box([parent['bbox_native']], size, 32)
    view = put_view(root, cid, folder, cb, cfg)
    render(folder/'target.png', [parent], folder/'boxes.png', view, mode='request')
    q = {'task': '判断这一个待拆区域是否方便用bbox分开，以及需要重建的具体对象范围。',
         'rules': [
             '完整性第一，减少重复第二。框容易分别定位的独立箭头必须拆，不得只因数量多或方向各异就保留整片。',
             '空间重复小单元内的不同方向箭头应进入split_spatial，由后台小块扫描。不要在这里枚举全部箭头。',
             '少数独立长连线若完整bbox仍能清楚区分，用split_routes。允许少量框重叠。',
             '曲线/虚线相互交叉且各框覆盖几乎同一片复杂笔画、不容易用bbox分开的，用keep_group。不要强拆或截成线段。',
             '真正共享主干网络、局部密集线束、紧凑难分复合符号可以保留。图型是神经网络本身不能作为合组理由。',
             '父框和旧描述是待核验线索，不是语义真值。只根据实际像素说明target_scope，排除穿过框的无关连线和形状边。',
             '本步只决定策略。原图GLOBAL辅助看完整起终点。所有坐标使用TARGET像素。'],
         'parent': {'kind': parent['kind'], 'bbox': local_box(parent['bbox_native'], view), 'locator_hint': parent['locator']},
         'target_size_px': list(view.input_size),
         'output_schema': {'action': 'split_spatial|split_routes|keep_group|uncertain',
                           'bbox_separability': 'easy|hard|uncertain', 'target_scope': '短目标范围描述', 'reason': '短可见依据'}}
    result = ask(model, folder, q, [('GLOBAL', root/'inputs'/cid/'G/target.png'),
                 ('TARGET', folder/'target.png'), ('BOXES', folder/'boxes.png')], validate_plan)
    print({'case': cid, 'parent': parent['display_id'], 'plan': result}, flush=True)
    return result


def validate_inventory(a, size, aliases):
    if a.get('decision') not in {'replace', 'keep_group', 'uncertain'}:
        raise ValueError('inventory decision required')
    if not isinstance(a.get('objects'), list) or type(a.get('complete_inventory')) is not bool:
        raise ValueError('objects and complete_inventory required')
    if not isinstance(a.get('reason'), str) or not a['reason'].strip():
        raise ValueError('reason required')
    ids, used_refs = set(), set()
    for o in a['objects']:
        if not isinstance(o.get('id'), str) or not o['id'] or o['id'] in ids:
            raise ValueError('Unique output IDs required')
        ids.add(o['id'])
        ref = o.get('existing_id')
        if ref is not None and (ref not in aliases or ref in used_refs):
            raise ValueError('Use each supplied existing ID at most once')
        if ref is not None: used_refs.add(ref)
        if o.get('kind') not in KINDS or o.get('extent') not in {'complete', 'partial'}:
            raise ValueError('kind and extent required')
        valid_box(o.get('bbox'), size)
        if not isinstance(o.get('locator'), str) or not o['locator'].strip():
            raise ValueError('Short visual object identity required')
    return a


def inventory(root, cid, parent, plan, spec, baseline, cfg, model):
    folder = root/'cases'/cid/'regions'/parent['display_id']/'inventory'/spec['id']
    is_spatial = plan['action'] == 'split_spatial'
    neighbors, deferred = nearby_existing(parent, baseline, cfg['existing_context_max'])
    if is_spatial:
        neighbors = [o for o in neighbors if intersection(o['bbox_native'], spec['crop']) > 0]
    bounds = spec['crop']
    if not is_spatial and neighbors:
        size = Image.open(root/'inputs'/cid/'source.png').size
        bounds = crop_box([parent['bbox_native']]+[o['bbox_native'] for o in neighbors], size, 32)
    view = put_view(root, cid, folder, bounds, cfg)
    focus = {'id': 'REGION', 'kind': 'dense_region', 'bbox_native': spec['core']}
    render(folder/'target.png', [focus], folder/'boxes.png', view, context=neighbors, mode='request')
    aliases = {f'r{i+1}': o for i, o in enumerate(neighbors)}
    q = {'task': '在指定区域发现完整身份，并核对它是否已由只读小框负责。',
         'target_scope': plan['target_scope'], 'mode': 'spatial_tile' if is_spatial else 'complete_routes',
         'parent_locator_hint': parent['locator'], 'target_size_px': list(view.input_size),
         'focus_core_bbox': local_box(spec['core'], view),
         'rules': [
             '完整性第一。每支普通独立箭头输出完整杆与头；不同方向/不同路径即使相邻或同一起终点也不能自动并为一支。',
             '空间重复单元中的箭头必须逐支定位，封闭圆形/方形单元边界不是箭头；不同颜色只是像素区分线索，不推算数量或方向。',
             'BOXES的1仅是本次观察责任区域，不是一个真实对象。r开头为只读已有身份。旧框、旧描述可有误，按像素核对。',
             '对核心区域内的真实对象检查完整范围；核心边界外的同一对象也必须框全。TARGET裁掉对象则partial，不能把残段说成完整。',
             '后台根据完整bbox中心归属核心窗口；请输出核心内及其边缘相交的候选，不做局部到全图换算。',
             '只处理target_scope描述的对象。不能把矩形框内碰巧经过的其他连线、形状边、文字笔画也算入父区域责任。',
             '同一物理箭头已有r身份时引用existing_id，不再新增；bbox包含/相交不证明同一对象，必须核对完整起终点和实际线路。',
             '便于用框分别表示才decision=replace。多数框互相覆盖同一复杂线束、难以辨别时keep_group，不强行拆成残段。',
             '若只读身份已完整承担父区域全部对象，可以replace并只引用已有ID；这表示消除重复父任务。',
             '找不全或不确定时complete_inventory=false。不要为凑数量补线，不输出颜色/宽度/路径点。'],
         'existing_readonly': [{'id': k, 'kind': o['kind'], 'bbox': local_box(o['bbox_native'], view),
                               'locator_hint': o['locator']} for k, o in aliases.items()],
         'output_schema': {'decision': 'replace|keep_group|uncertain', 'complete_inventory': True, 'reason': '短依据',
             'objects': [{'id': 'O1', 'existing_id': None, 'kind': 'single|serial_chain|shared_network|dense_network',
                          'extent': 'complete|partial', 'bbox': [0, 0, 10, 10], 'locator': '完整起终点/对象短描述'}]}}
    answer = ask(model, folder, q, [('GLOBAL', root/'inputs'/cid/'G/target.png'),
                  ('TARGET', folder/'target.png'), ('BOXES', folder/'boxes.png')],
                  lambda a: validate_inventory(a, view.input_size, aliases))
    objects, outside, partial = [], [], []
    for o in answer['objects']:
        b = view.to_global(o['bbox'][:2])+view.to_global(o['bbox'][2:])
        item = {'id': parent['display_id']+'/'+spec['id']+'/'+o['id'], 'kind': o['kind'], 'bbox_native': b,
                'locator': o['locator'], 'extent': o['extent'], 'uncertain': False,
                'existing_id': aliases[o['existing_id']]['id'] if o.get('existing_id') is not None else None,
                'source_region_id': parent['id'], 'source_tile': spec['id'], 'source_ids': parent['source_ids'],
                'geometry_ready': False, 'handoff_stage': 'discovery_only_not_geometry_validated'}
        if is_spatial and not in_core(b, spec['core']):
            outside.append(item)
            continue
        if item['extent'] != 'complete': partial.append(item)
        objects.append(item)
    result = {'spec': spec, 'view': view.metadata(view.global_size), 'answer': answer,
              'objects': objects, 'outside_core': outside, 'partial': partial,
              'readonly_ids': [o['id'] for o in neighbors], 'deferred_context_ids': [o['id'] for o in deferred]}
    write(folder/'materialized.json', result)
    render(folder/'target.png', objects, folder/'overlay.png', view)
    print({'case': cid, 'parent': parent['display_id'], 'tile': spec['id'], 'decision': answer['decision'],
           'objects': len(objects), 'partial': len(partial)}, flush=True)
    return result


def validate_coverage(a, ids):
    if type(a.get('complete')) is not bool or type(a.get('duplicates')) is not bool:
        raise ValueError('complete and duplicates must be booleans')
    if not isinstance(a.get('reason'), str) or not a['reason'].strip():
        raise ValueError('Visible completeness reason required')
    if not isinstance(a.get('missing'), list) or not isinstance(a.get('problem_ids'), list):
        raise ValueError('missing and problem_ids required')
    if not set(a['problem_ids']) <= set(ids):
        raise ValueError('Only supplied object IDs may be referenced')
    if a['complete'] and (a['missing'] or a['problem_ids']):
        raise ValueError('Complete contradicts unresolved missing/problem objects')
    return a


def coverage_neighbors(primary, all_children, core, margin=3):
    ids={o['id'] for o in primary}
    b=[core[0]-margin,core[1]-margin,core[2]+margin,core[3]+margin]
    return [o for o in all_children if o['id'] not in ids and intersection(o['bbox_native'],b)>0]


def coverage(root, cid, parent, plan, inv, objects, baseline_index, cfg, model, all_children=None):
    if plan['action'] != 'split_spatial':
        # Only the spatial seam contract changed. Preserve exact route-review
        # prompts and pixels so already complete checks can be reused honestly.
        from legacy_region_partition_v17 import coverage as route_coverage
        return route_coverage(root,cid,parent,plan,inv,objects,baseline_index,cfg,model)
    spec = inv['spec']
    folder = root/'cases'/cid/'regions'/parent['display_id']/'coverage'/spec['id']
    view = frame(inv['view'])
    source_target = root/'cases'/cid/'regions'/parent['display_id']/'inventory'/spec['id']/'target.png'
    save(Image.open(source_target).convert('RGB'), folder/'target.png')
    neighbors=coverage_neighbors(objects,all_children or objects,spec['core'])
    display = []
    for o in objects+neighbors:
        r = copy.deepcopy(baseline_index[o['existing_id']] if o.get('existing_id') else o)
        r['id'] = o['id']
        r.pop('display_id', None)
        display.append(r)
    primary_display=display[:len(objects)];neighbor_display=display[len(objects):]
    render(folder/'target.png', primary_display, folder/'boxes.png', view, context=neighbor_display, mode='request')
    aliases=[str(i+1) for i in range(len(primary_display))]+[f'r{i+1}' for i in range(len(neighbor_display))]
    write(folder/'ownership_context.json',{'primary_ids':[o['id'] for o in objects],
        'readonly_neighbor_ids':[o['id'] for o in neighbors], 'one_global_owner_per_identity':True})
    q = {'task': '独立复核：父区域的完整重建责任是否已由给定子身份全部承担？',
         'target_scope': plan['target_scope'], 'focus_core_bbox': local_box(spec['core'], view),
         'completeness_scope': '仅检查本核心区域及边界的可见目标是否有完整负责身份。r开头是邻窗已有身份，同样可以证明覆盖。不要重算中心归属或清点整个父区域总数。',
         'rules': [
             '以TARGET像素为准。检查每支应有箭头是否有完整的杆、头、真实起终点对应；有框不等于正确覆盖。',
             '只审核target_scope和核心范围内的对象。对象跨出核心但完整显示在TARGET时仍需完整子身份；别把旁边其他区域的对象算成遗漏。',
             '后台已经分配唯一重建责任。数字框与r邻居框都是真实负责者，邻居只是只读显示，不会多重建一次。不判断0.x像素的中心归属。',
             '边界处的箭头若由数字框或r框完整负责，就已经覆盖；只有两类身份均未负责才报告missing。不要把相邻窗口负责的对象误报为遗漏。',
             'target_scope内可能有规划阶段旧坐标，不能当作本轮坐标；按对象语义和本轮TARGET像素判断。只做可见覆盖核对，不逐像素测量。',
             '子身份可引用已有对象；实际使用的原bbox已画出。检查是否把子身份错误对应到旁边另一条线。',
             '不同颜色/方向/线路的普通箭头逐支检查。圆/矩形轮廓、文字等不属于目标。',
             '所有对象均完整且没有漏项才complete=true。有疑点返回false，父区域将继续保留重建责任。',
             '若同一真实连线出现多个子身份，duplicates=true，指出problem_ids；框相交本身不说明重复。',
             '不用重做坐标或路径拟合；missing仅写短位置描述，reason简短。'],
         'objects': [{'id': aliases[i], 'kind': o['kind'], 'bbox': local_box(o['bbox_native'], view),
                      'role': 'primary' if i<len(objects) else 'readonly_neighbor',
                      'locator_hint': o.get('locator', '')} for i, o in enumerate(display)],
         'output_schema': {'complete': True, 'duplicates': False, 'missing': [], 'problem_ids': [], 'reason': '短像素依据'}}
    return ask(model, folder, q, [('TARGET', folder/'target.png'), ('BOXES', folder/'boxes.png')],
               lambda a: validate_coverage(a, aliases))


def can_commit(plan, inventories, checks, conflicts):
    if not inventories or conflicts or len(checks) != len(inventories):
        return False
    if any(x['answer']['decision'] != 'replace' or x['partial'] for x in inventories):
        return False
    if plan['action'] in ('keep_group', 'uncertain') and any(not o.get('existing_id') for x in inventories for o in x['objects']):
        return False
    return all(c['complete'] and not c['duplicates'] and not c['missing'] and not c['problem_ids'] for c in checks)


def apply_replacement(active, parent_id, children, commit):
    """Atomic ownership switch: retain old parent on any unresolved condition."""
    if not commit:
        return copy.deepcopy(active), []
    index = {o['id']: o for o in active}
    if parent_id not in index or not children:
        raise ValueError('Replacement requires an active parent and nonempty owner inventory')
    for child in children:
        if child.get('existing_id') == parent_id or child['id'] == parent_id:
            raise ValueError('Parent cannot own itself after retirement')
        if child.get('existing_id') and child['existing_id'] not in index:
            raise ValueError('Referenced existing owner is unavailable')
    new = [copy.deepcopy(o) for o in active if o['id'] != parent_id]
    owners = []
    for child in children:
        oid = child.get('existing_id') or child['id']
        owners.append(oid)
        if not child.get('existing_id'):
            new.append(copy.deepcopy(child))
    if len({o['id'] for o in new}) != len(new):
        raise ValueError('Duplicate active reconstruction ID')
    return new, sorted(set(owners))


def seam_pairs(children):
    """BBox similarity only nominates a semantic check; no unconditional merge."""
    return [(a, b) for i, a in enumerate(children) for b in children[i+1:]
            if a['source_tile'] != b['source_tile'] and iou(a['bbox_native'], b['bbox_native']) >= .40]


def reconcile_spatial(root, cid, parent, inventories, cfg, model):
    children = [copy.deepcopy(o) for inv in inventories for o in inv['objects']]
    folder = root/'cases'/cid/'regions'/parent['display_id']/'seams'
    size = Image.open(root/'inputs'/cid/'source.png').size
    decisions = []
    for number, (a, b) in enumerate(seam_pairs(children), 1):
        out = folder/f'P{number:03d}'
        cb = crop_box([a['bbox_native'], b['bbox_native']], size, 20)
        view = put_view(root, cid, out, cb, cfg)
        render(out/'target.png', [a, b], out/'boxes.png', view, mode='request')
        q = {'task': '只判断两个跨分块候选是否为同一支物理箭头。',
             'rules': ['TARGET是干净局部，BOXES给出两个候选。请按实际杆、头、方向判断。',
                       '同一箭头的偏框为same；不同红蓝方向、不同头、相邻平行箭头为different。相交或同起点不能证明same。',
                       '不重新估计坐标，不枚举其他箭头。无法确认用uncertain。'],
             'objects': [{'id': str(i+1), 'bbox': local_box(o['bbox_native'], view), 'locator_hint': o['locator']} for i, o in enumerate((a, b))],
             'output_schema': {'relation': 'same|different|uncertain', 'reason': '一句像素依据'}}
        def validate(x):
            if x.get('relation') not in {'same','different','uncertain'} or not str(x.get('reason','')).strip():
                raise ValueError('relation and reason required')
            return x
        answer = ask(model, out, q, [('TARGET', out/'target.png'), ('BOXES', out/'boxes.png')], validate)
        decisions.append({'a': a['id'], 'b': b['id'], **answer})
    current = {o['id']: o for o in children}
    aliases = {k: k for k in current}
    groups = {k: {k} for k in current}
    different = {frozenset((d['a'],d['b'])) for d in decisions if d['relation']=='different'}
    conflicts, merged = [], []
    for d in decisions:
        if d['relation']=='uncertain':
            conflicts.append({'type':'uncertain_seam_relation', **d})
        if d['relation']!='same':continue
        a, b = aliases[d['a']], aliases[d['b']]
        if a==b:continue
        if any(frozenset((x,y)) in different for x in groups[a] for y in groups[b]):
            conflicts.append({'type':'seam_transitive_conflict', **d});continue
        if current[a].get('existing_id') and current[b].get('existing_id') and current[a]['existing_id']!=current[b]['existing_id']:
            conflicts.append({'type':'different_existing_owners', **d});continue
        if current[b].get('existing_id') and not current[a].get('existing_id'):a,b=b,a
        current[a]['bbox_native'] = union([current[a]['bbox_native'],current[b]['bbox_native']])
        current[a].setdefault('merged_observation_ids', []).extend([b]+current[b].get('merged_observation_ids',[]))
        current[a].setdefault('seam_decisions', []).append(d)
        for old in groups[b]:aliases[old]=a
        groups[a]|=groups.pop(b);del current[b];merged.append(d)
    # Complete observations near a seam can fall outside both predicted ownership
    # centers. Recover only candidates not geometrically represented anywhere;
    # final independent pixel coverage checks still decide whether to commit.
    recovered=[];represented=[]
    for inv in inventories:
        for o in inv['outside_core']:
            if o['extent']!='complete' or o['kind']!='single' or not in_core(o['bbox_native'],parent['bbox_native']):continue
            near=[p for p in current.values() if iou(p['bbox_native'],o['bbox_native'])>=.35]
            if near:
                represented.append({'observation':o['id'],'candidate_ids':[p['id'] for p in near]});continue
            current[o['id']]=copy.deepcopy(o);recovered.append(o['id'])
    repartitioned=copy.deepcopy(inventories)
    for inv in repartitioned:
        inv['objects']=[copy.deepcopy(o) for o in current.values() if in_core(o['bbox_native'],inv['spec']['core'])]
        inv['partial']=[o for o in inv['objects'] if o['extent']!='complete']
    all_assigned=[o['id'] for inv in repartitioned for o in inv['objects']]
    if set(all_assigned)!=set(current) or len(all_assigned)!=len(set(all_assigned)):
        conflicts.append({'type':'seam_assignment_not_total_or_unique'})
    write(folder/'audit.json',{'input_core_count':len(children),'output_count':len(current),
        'decisions':decisions,'merged':merged,'recovered_complete_observations':recovered,
        'outside_observations_geometrically_represented':represented,'conflicts':conflicts})
    return repartitioned, conflicts


def process_region(root, cid, parent, baseline, cfg, model):
    folder = root/'cases'/cid/'regions'/parent['display_id']
    if (folder/'summary.json').exists(): return read(folder/'summary.json')
    plan = plan_region(root, cid, parent, cfg, model)
    size = Image.open(root/'inputs'/cid/'source.png').size
    if plan['action'] in {'keep_group','uncertain'}:
        result={'parent_id':parent['id'],'parent_display_id':parent['display_id'],'plan':plan,
                'children':[],'inventory_count':0,'inventory_summaries':[],'coverage_checks':[],
                'conflicts':[],'commit':False,'parent_role':'active_reconstruction_group',
                'retention_reason':'User policy: no forced enumeration when bbox separation is hard or uncertain',
                'retention_is_not_a_completeness_certificate':True,
                'completeness_semantically_proven':False,'completed_at':now()}
        write(folder/'summary.json',result)
        inp=root/'inputs'/cid;gv=frame(read(inp/'views.json')['views'][0]['transform'])
        render(inp/'G/target.png',[parent],folder/'retained_group.png',gv)
        render(inp/'G/target.png',[],folder/'candidate_children.png',gv)
        print({'case':cid,'parent':parent['display_id'],'decision':'retain_group_without_forced_enumeration'},flush=True)
        return result
    if plan['action'] == 'split_spatial':
        specs = spatial_tiles(parent['bbox_native'], size, cfg['spatial_core_native_px'], cfg['spatial_overlap_native_px'])
    else:
        specs = [{'id': 'WHOLE', 'core': parent['bbox_native'], 'crop': crop_box([parent['bbox_native']], size, 32)}]
    write(folder/'tile_plan.json', specs)
    invs, failures = [], []
    with ThreadPoolExecutor(max_workers=cfg['task_workers']) as pool:
        jobs = {pool.submit(inventory, root, cid, parent, plan, spec, baseline, cfg, model): spec for spec in specs}
        for job in as_completed(jobs):
            try: invs.append(job.result())
            except Exception as exc: failures.append({'stage': 'inventory', 'tile': jobs[job]['id'], 'error_type': type(exc).__name__, 'error': str(exc)})
    invs.sort(key=lambda x: x['spec']['id'])
    if cfg.get('seam_review_enabled') and plan['action']=='split_spatial' and not failures:
        invs, seam_conflicts = reconcile_spatial(root,cid,parent,invs,cfg,model)
        failures.extend(seam_conflicts)
    children = [o for inv in invs for o in inv['objects']]
    conflicts = copy.deepcopy(failures)
    if not children: conflicts.append({'type': 'empty_inventory'})
    if plan['action'] == 'split_spatial' and any(o['kind'] != 'single' for o in children):
        conflicts.append({'type': 'spatial_split_still_contains_group', 'reason': 'Separable independent arrows must not remain grouped'})
    base_index = {o['id']: o for o in baseline}
    for i, a in enumerate(children):
        if a.get('existing_id') == parent['id']:
            conflicts.append({'type': 'parent_self_reference', 'id': a['id']})
        for b in children[:i]:
            if a.get('existing_id') and a.get('existing_id') == b.get('existing_id'):
                conflicts.append({'type': 'repeated_existing_owner', 'ids': [a['id'], b['id']]})
            elif a['source_tile'] != b['source_tile'] and iou(a['bbox_native'], b['bbox_native']) > .85:
                conflicts.append({'type': 'possible_tile_duplicate', 'ids': [a['id'], b['id']]})
        if not a.get('existing_id'):
            for old in baseline:
                if old['kind'] != 'dense_region' and iou(a['bbox_native'], old['bbox_native']) > .85:
                    conflicts.append({'type': 'unresolved_existing_duplicate', 'new': a['id'], 'old': old['id']})
    checks = []
    # An inventory's uncertainty is not overridden silently. Independent local
    # pixel coverage is still required and can confirm or reject its proposal.
    # This also handles a local model declining to count the entire global array.
    eligible = all(x['answer']['decision']=='replace' and not x['partial'] for x in invs)
    if eligible and not conflicts and (plan['action'].startswith('split') or all(o.get('existing_id') for o in children)):
        with ThreadPoolExecutor(max_workers=cfg['task_workers']) as pool:
            jobs = {pool.submit(coverage, root, cid, parent, plan, inv, inv['objects'], base_index, cfg, model, children): inv for inv in invs}
            for job in as_completed(jobs):
                try: checks.append(job.result())
                except Exception as exc: conflicts.append({'stage': 'coverage', 'tile': jobs[job]['spec']['id'], 'error_type': type(exc).__name__, 'error': str(exc)})
    commit = can_commit(plan, invs, checks, conflicts)
    result = {'parent_id': parent['id'], 'parent_display_id': parent['display_id'], 'plan': plan,
              'children': children, 'inventory_count': len(invs), 'inventory_summaries':
              [{'tile': x['spec']['id'], 'decision': x['answer']['decision'], 'complete_inventory': x['answer']['complete_inventory'],
                'count': len(x['objects']), 'reason': x['answer']['reason']} for x in invs],
              'coverage_checks': checks, 'conflicts': conflicts, 'commit': commit,
              'parent_role': 'context_only' if commit else 'active_reconstruction_group',
              'completeness_semantically_proven': False, 'completed_at': now()}
    write(folder/'summary.json', result)
    inp = root/'inputs'/cid
    global_view = frame(read(inp/'views.json')['views'][0]['transform'])
    render(inp/'G/target.png', children, folder/'candidate_children.png', global_view)
    print({'case': cid, 'parent': parent['display_id'], 'commit': commit, 'children': len(children), 'checks': len(checks), 'conflicts': len(conflicts)}, flush=True)
    return result


def execute_case(root, cid, cfg, model):
    c = root/'cases'/cid
    if (c/'finished.json').exists(): return read(c/'result.json')
    imported_root=Path(cfg['completed_result_import_run']) if cfg.get('completed_result_import_run') else None
    if imported_root and (imported_root/'cases'/cid/'finished.json').exists():
        original=imported_root/'cases'/cid
        if sha256(original/'baseline_result.json')!=sha256(c/'baseline_result.json'):
            raise ValueError('Completed-result import baseline differs')
        if sha256(imported_root/'inputs'/cid/'source.png')!=sha256(root/'inputs'/cid/'source.png'):
            raise ValueError('Completed-result import image differs')
        provenance=[]
        for f in original.rglob('*'):
            if not f.is_file() or f.name=='baseline_result.json':continue
            rel=f.relative_to(original);dest=c/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(f,dest);provenance.append({'relative_path':str(rel),'source':str(f),'sha256':sha256(f)})
        write(c/'completed_result_import.json',{'source_run':str(imported_root),
              'policy':'Retain a complete validated result unchanged; the new keep-group terminal branch applies only to unfinished cases',
              'same_baseline_and_source_pixels':True,'new_model_calls':0,'files':provenance})
        publish(root)
        return read(c/'result.json')
    baseline = read(c/'baseline_result.json')['objects']
    active = copy.deepcopy(baseline)
    parents = [o for o in baseline if o.get('needs_split') or o['kind']=='dense_region']
    ledgers = []
    for parent in parents:
        summary = process_region(root, cid, parent, baseline, cfg, model)
        active, owners = apply_replacement(active, parent['id'], summary['children'], summary['commit'])
        ledgers.append({**summary, 'active_owner_ids': owners if summary['commit'] else [parent['id']]})
    next_id = max((int(o['display_id'][1:]) for o in baseline),default=0)+1
    old_index = {o['id']: o for o in baseline}
    for o in active:
        if o['id'] not in old_index:
            o['display_id'] = f'I{next_id:04d}'
            next_id += 1
        else:
            assert o == old_index[o['id']]
    result = {'case_id': cid, 'baseline_count': len(baseline), 'output_identity_count': len(active),
              'objects': active, 'region_ledger': ledgers,
              'committed_regions': sum(x['commit'] for x in ledgers),
              'retained_regions': sum(not x['commit'] for x in ledgers),
              'new_identities': sum(o['id'] not in old_index for o in active),
              'existing_identities_unchanged': True, 'formal_accuracy_measured': False,
              'geometry_handoff_executed': False, 'completed_at': now()}
    write(c/'result.json', result)
    inp = root/'inputs'/cid
    render(inp/'G/target.png', active, c/'overlay.png', frame(read(inp/'views.json')['views'][0]['transform']))
    write(c/'finished.json', {'completed_at': now()})
    publish(root)
    return result


def prepare(root, source, ids, report, replay=None, completed_import=None):
    if root.exists(): raise RuntimeError('Fresh run required')
    root.mkdir(parents=True)
    cfg = read(source/'config.json')
    cfg.update(source_run=str(source), review_output_dir=str(report), protocol_version='region_split_ownership_v21',
               api_concurrency=4, task_workers=4, case_workers=1, existing_context_max=4,
               spatial_core_native_px=96, spatial_overlap_native_px=24, seam_review_enabled=True,
               inventory_replay_run=str(replay) if replay else None,
               wait_for_replay_case=False,
               completed_result_import_run=str(completed_import) if completed_import else None)
    assert cfg['max_tokens']==113000 and 'reasoning_effort' not in cfg
    write(root/'config.json', cfg);write(root/'cases.json', ids)
    shutil.copytree(Path(__file__).parent, root/'code', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(source/'constraints_snapshot', root/'constraints_snapshot')
    (root/'constraints_snapshot/user_bbox_separability_20261025.txt').write_text(
        '方便用bbox区分的独立、方向各异箭头必须拆；曲线或虚线跳连若确实难以用框区分，可以不拆。'
        '试验0183、0077、0346。0170父框包含子框时完整性最重要，减少重复其次；不得仅因包含删除。\n')
    sources=[]
    for cid in ids:
        if not (source/'cases'/cid/'finished.json').exists(): raise ValueError('Finished baseline required')
        shutil.copytree(source/'inputs'/cid, root/'inputs'/cid)
        c=root/'cases'/cid;c.mkdir(parents=True)
        f=source/'cases'/cid/'result.json';shutil.copy2(f,c/'baseline_result.json')
        sources.append({'case_id':cid,'source':str(f),'sha256':sha256(f)})
    write(root/'baseline_sources.json',sources)
    (root/'PROTOCOL.md').write_text('# bbox可分性与完整归属试验\n\n'
        '固定v15的4图。对needs_split区域先判断空间可分、路径可分、保留组或不确定。空间可分区域使用96原图像素核心、24像素重叠；输入长边1080–2048。'
        '每个请求最多4个已有身份只读参考。容易分别定位的独立箭头必须拆，曲线/虚线难框分则保留。\n\n'
        '先产生候选子身份，再独立检查核心内完整性与重复；所有窗口返回完整、没有冲突且coverage复核通过，才将父框改为context_only。'
        '同一连线已经有已有小框负责时引用该ID；不重新新增。任何遗漏、失败或归属不确定均保留原父组，候选子框只作试验展示。'
        '父框不是渲染对象，active objects才是后续重建清单。当前未运行几何/PPT，只验证发现层与交接账本。\n\n'
        '跨窗高重叠候选仅提名两框语义审核，same才合并；补回未被任何现有候选表示的完整边界观察，再分配唯一核心责任并独立复核。'
        '相同prompt/图片/参数且完整SSE的请求可精确复用。初始inventory的完整性自评仅作证据，独立复核只评价本核心内责任，不能要求局部图清点整个父区域总数。'
        '覆盖复核同时显示跨边界的邻窗身份作为只读负责者，后台保持每个身份唯一归属，不让模型重算中心在哪个窗。'
        '已确认bbox难分或不确定时直接终止拆分并保留原组，不再强迫枚举。此前完整结果可按源结果/像素哈希原样导入，不冒充新调用。'
        '若独立复核仍发现任何缺失，继续保留父组。默认thinking、113000；GLM4、图1。完整性优先，模型coverage通过不能视为视觉真值，最终仍需看图。所有源结果冻结不改写。\n')
    files=[p for directory in ('code','inputs','constraints_snapshot','cases') for p in (root/directory).rglob('*') if p.is_file()]
    files += [root/'config.json',root/'cases.json',root/'baseline_sources.json',root/'PROTOCOL.md']
    write(root/'manifest.json',{'files':{str(p.relative_to(root)):sha256(p) for p in files},'created_at':now()})
    (root/'RESULTS.md').write_text('# 拆分与父子归属试验\n\n准备完成，模型结果待验证。\n')
    write(root/'progress.json',{'state':'prepared','total':len(ids),'finished':0})
    publish(root)


def publish(root):
    if read(root/'config.json').get('integrated_mode'):return
    cfg=read(root/'config.json');dest=Path(cfg['review_output_dir']);(dest/'images').mkdir(parents=True,exist_ok=True)
    lines=['# bbox可分性与父子归属：4图试验','',
           '完整性优先；候选拆分只有通过独立覆盖检查才替换父任务。既有小框可直接复用，父框仅作为上下文保留。', '',
           '[方案]('+str(root/'PROTOCOL.md')+') · [结果]('+str(root/'RESULTS.md')+')','']
    summary=[]
    for cid in read(root/'cases.json'):
        c=root/'cases'/cid
        before=dest/'images'/f'{cid}_before.png'
        if not before.exists():shutil.copy2(Path(cfg['source_run'])/'cases'/cid/'overlay.png',before)
        lines+=['## '+cid,'']
        if (c/'result.json').exists():
            d=read(c/'result.json');shutil.copy2(c/'overlay.png',dest/'images'/f'{cid}_after.png')
            summary.append({k:d[k] for k in ('case_id','baseline_count','output_identity_count','committed_regions','retained_regions','new_identities')})
            lines += [f'原身份 {d["baseline_count"]} → 当前责任清单 {d["output_identity_count"]}；已切换父区域 {d["committed_regions"]}，保留组 {d["retained_regions"]}，新增身份 {d["new_identities"]}。','',
                      '| 原结果 | 试验后生效清单 |','|---|---|',f'| ![](images/{cid}_before.png) | ![](images/{cid}_after.png) |','']
            for region in d['region_ledger']:
                pid=region['parent_display_id'];name=f'{cid}_{pid}_children.png'
                preview=c/'regions'/pid/('retained_group.png' if not region['children'] and (c/'regions'/pid/'retained_group.png').exists() else 'candidate_children.png')
                shutil.copy2(preview,dest/'images'/name)
                label='保留组范围' if not region['children'] else '候选拆分图'
                lines += [f'- {pid}：{region["plan"]["action"]}；父任务状态 {region["parent_role"]}；候选子身份 {len(region["children"])}。'
                          f' [{label}](images/{name}) · [详细账本]({c}/regions/{pid}/summary.json)']
            lines+=['']
        else:
            lines+=['进行中；此图尚无最终试验结论。','',f'![](images/{cid}_before.png)','']
    (dest/'拆分与父子归属对照_4例.md').write_text('\n'.join(lines))
    write(root/'summary.json',{'cases':summary,'finished':len(summary),'total':len(read(root/'cases.json')),'updated_at':now()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    p.add_argument('--source',type=Path);p.add_argument('--cases',nargs='+');p.add_argument('--report-dir',type=Path)
    p.add_argument('--replay-from',type=Path)
    p.add_argument('--import-results-from',type=Path)
    args=p.parse_args();root=args.run_dir.resolve()
    if args.prepare:return prepare(root,args.source.resolve(),args.cases,args.report_dir.resolve(),args.replay_from.resolve() if args.replay_from else None,args.import_results_from.resolve() if args.import_results_from else None)
    if Path(__file__).resolve().parent != root/'code':raise RuntimeError('Run frozen code only')
    lock=(root/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for rel,digest in read(root/'manifest.json')['files'].items():
        if sha256(root/rel)!=digest:raise RuntimeError('Frozen file changed: '+rel)
    cfg=read(root/'config.json');replay=Path(cfg['inventory_replay_run']) if cfg.get('inventory_replay_run') else None
    model=Model(cfg,run_dir=root,reuse_from=replay,system=SYSTEM)
    write(root/'process.json',{'pid':os.getpid(),'started_at':now()});records=[]
    for cid in args.cases or read(root/'cases.json'):
        if replay and cfg.get('wait_for_replay_case',True):
            while not (replay/'cases'/cid/'finished.json').exists():
                if (replay/'cases'/cid/'failed.json').exists():break
                write(root/'progress.json',{'state':'waiting_for_frozen_replay_case','current_case':cid,'updated_at':now()})
                time.sleep(5)
        write(root/'progress.json',{'state':'running','current_case':cid,'finished':len(records),'updated_at':now()})
        try:
            result=execute_case(root,cid,cfg,model);records.append({'case_id':cid,'state':'finished','committed_regions':result['committed_regions']})
        except Exception as exc:
            import traceback
            err={'case_id':cid,'state':'failed','error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()}
            write(root/'cases'/cid/'failed.json',err);records.append(err)
    write(root/'progress.json',{'state':'finished','records':records,'updated_at':now()});publish(root)


if __name__=='__main__':main()
