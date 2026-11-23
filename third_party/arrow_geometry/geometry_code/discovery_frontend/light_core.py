"""Loss-aware candidate compression; no model calls and no case-specific rules."""
import copy
from collections import Counter
from core import area, intersection, inside, union, gap, crop_box

NETWORKS = {'serial_chain', 'shared_network', 'dense_network'}


def iou(a, b):
    inter = intersection(a, b)
    return inter / max(1e-6, area(a) + area(b) - inter)


def duplicate_reason(a, b, protect_splits=False):
    """Compare original observations, not growing unions (avoids chain collapse)."""
    if protect_splits:
        # Provenance overlap across separate calls does not prove an explicit split.
        if set(a.get('split_families',[])) & set(b.get('split_families',[])):return None
        if not (a.get('reviewed_by_glm') or b.get('reviewed_by_glm')) and set(a.get('source_ids',[])) & set(b.get('source_ids',[])):return None
    if a['kind'] != b['kind']:
        return None
    x, y = a['bbox_native'], b['bbox_native']
    score = iou(x, y)
    same_view = a.get('source_view') and a.get('source_view') == b.get('source_view')
    if same_view and a['id'] != b['id'] and score < .95:
        return None
    if protect_splits and a['kind']=='single' and a.get('reviewed_by_glm') and b.get('reviewed_by_glm'):
        # Refined bent/nested paths can have overlapping bboxes but different ink.
        if score < .90:return None
    if score >= .65:
        return 'high_iou'
    small, large = (a, b) if area(x) <= area(y) else (b, a)
    ratio = area(small['bbox_native']) / max(1, area(large['bbox_native']))
    containment = inside(small['bbox_native'], large['bbox_native'])
    if containment >= .90 and (ratio >= .40 or small['extent'] == 'partial' or a['kind'] in NETWORKS):
        return 'contained_duplicate_or_partial'
    # Thin boxes can be shifted by several pixels across scales. Only combine
    # aligned overlapping shafts, preserving nearby parallel or serial arrows.
    if a['kind'] == 'single':
        dims = [[max(1, z[2]-z[0]), max(1, z[3]-z[1])] for z in [x,y]]
        axes = [0 if w/h >= 4 else 1 if h/w >= 4 else -1 for w,h in dims]
        if axes[0] == axes[1] and axes[0] >= 0:
            axis = axes[0]; cross = 1-axis
            along = max(0, min(x[axis+2], y[axis+2])-max(x[axis], y[axis]))
            aligned = along / max(1, min(dims[0][axis], dims[1][axis])) >= .85
            delta = abs((x[cross]+x[cross+2]-y[cross]-y[cross+2])/2)
            if aligned and delta <= .45 * max(dims[0][cross], dims[1][cross]):
                return 'aligned_thin_duplicate'
    return None


def compress(rows, protect_splits=False, semantic_only=False):
    """One union box per observation cluster; retain every original source."""
    groups = []; audit = []
    def rank(r):
        return (r['extent'] != 'complete', r.get('source_view') != 'G', -area(r['bbox_native']), r['id'])
    for row in sorted(rows, key=rank):
        choices = []
        for i, group in enumerate(groups):
            anchor = group[0]
            why = None if semantic_only else duplicate_reason(anchor, row, protect_splits)
            if why:
                # A third view must not join two distinct complete candidates
                # already distinguished in another view.
                clash = any(m.get('source_view') == row.get('source_view') and
                            m.get('source_view') and m['id'] != row['id'] and
                            m['extent'] == row['extent'] == 'complete' and
                            iou(m['bbox_native'], row['bbox_native']) < .95 for m in group)
                if not clash:
                    choices.append((iou(anchor['bbox_native'], row['bbox_native']), i, why))
        if choices:
            _, i, why = max(choices)
            groups[i].append(row)
            audit.append({'member': row['id'], 'anchor': groups[i][0]['id'], 'reason': why})
        else:
            groups.append([row])
    out = []
    for i, members in enumerate(groups, 1):
        best = members[0]; row = copy.deepcopy(best)
        row.update(id=f'C{i:04d}', bbox_native=union([m['bbox_native'] for m in members]),
                   extent='complete' if any(m['extent']=='complete' for m in members) else 'partial',
                   source_ids=sorted({s for m in members for s in m.get('source_ids', [m['id']])}),
                   observation_ids=[m['id'] for m in members],
                   member_count=sum(m.get('member_count', 1) for m in members),
                   bbox_source=best.get('bbox_source','union') if len(members)==1 else 'union',
                   reason=best.get('reason','GLM discovery candidate retained') if len(members)==1 else 'High-overlap observations unioned; original sources retained',
                   uncertain=any(m.get('uncertain', False) for m in members), sam_ids=[])
        row['union_growth'] = area(row['bbox_native']) / max(1, area(best['bbox_native']))
        row['member_reasons']=[m.get('reason',m['locator']) for m in members]
        row['split_families']=sorted({s for m in members for s in m.get('split_families',[])})
        out.append(row)
    # Dense neural wiring remains a group. Completely interior short edges are
    # observations of that group, while boundary-crossing outputs remain single.
    absorbed = []; removed = set()
    for child in out:
        if semantic_only:break
        if child['kind'] != 'single':
            continue
        owners = []
        for parent in out:
            if parent['kind'] != 'dense_network' or parent['extent'] != 'complete':
                continue
            b=parent['bbox_native']; c=child['bbox_native']
            if inside(c,b) >= .98 and c[0] >= b[0]+2 and c[1] >= b[1]+2 and c[2] <= b[2]-2 and c[3] <= b[3]-2:
                owners.append(parent)
        if len(owners)==1:
            owner=owners[0];owner['source_ids']=sorted(set(owner['source_ids'])|set(child['source_ids']))
            owner.setdefault('interior_member_observations', []).append(copy.deepcopy(child))
            removed.add(child['id']);absorbed.append({'member':child['id'],'owner':owner['id'],
                'reason':'fully interior observation of GLM dense network; geometric discovery heuristic'})
    out=[r for r in out if r['id'] not in removed]
    assert {s for r in rows for s in r.get('source_ids',[r['id']])} == {s for r in out for s in r['source_ids']}
    return out, {'raw_count':len(rows),'compressed_count':len(out),'merges':audit,'network_members':absorbed,
                 'not_physical_ground_truth':True,'transitive_union_chaining':False}


def classify(rows):
    """Only ambiguity, residual offsets and incomplete extents need GLM repair."""
    out=[]
    for row in rows:
        r=copy.deepcopy(row); flags=[]
        if r.get('network_region'):flags.append('network_region_inventory')
        if r['kind'] in {'dense_region','sam_proposal'}:flags.append(r['kind'])
        if r['extent']=='partial':flags.append('partial_extent')
        if r.get('uncertain'):flags.append('uncertain')
        if r.get('union_growth',1)>1.55:flags.append('large_union_growth')
        for other in rows:
            if other['id']==r['id']:continue
            a,b=r['bbox_native'],other['bbox_native']
            if iou(a,b)>=.12 or (inside(a,b)>.90 and other['kind'] in NETWORKS):
                flags.append('remaining_overlap');break
            if r['kind']==other['kind']=='single' and intersection(a,b)>0:
                flags.append('crossing_or_offset');break
        r['review_reasons']=list(dict.fromkeys(flags));out.append(r)
    return out


def repair_batches(rows, size, cfg):
    remaining=sorted([r for r in rows if r['review_reasons']],key=lambda r:(r['bbox_native'][1],r['bbox_native'][0],r['id']))
    out=[]
    while remaining:
        group=[remaining.pop(0)]
        def complex_object(r):
            b=r['bbox_native']
            return r['kind'] in NETWORKS|{'dense_region'} or max(b[2]-b[0],b[3]-b[1])>cfg.get('repair_single_large_px',320)
        # A network/large extent is one task. Small packets must form one local
        # problem: all pairs close, bounded union and bounded overlap count.
        while remaining and len(group)<min(3,cfg['repair_batch_max']) and not any(complex_object(r) for r in group):
            candidates=[]
            for j,row in enumerate(remaining):
                if complex_object(row):continue
                subset=group+[row];candidate=union([r['bbox_native'] for r in subset])
                if max(candidate[2]-candidate[0],candidate[3]-candidate[1])>min(320,cfg['repair_span_px']):continue
                if any(gap(a['bbox_native'],b['bbox_native'])>min(48,cfg['repair_neighbor_gap']) for i,a in enumerate(subset) for b in subset[i+1:]):continue
                overlaps=sum(intersection(a['bbox_native'],b['bbox_native'])/max(1,min(area(a['bbox_native']),area(b['bbox_native'])))>.15 for i,a in enumerate(subset) for b in subset[i+1:])
                if overlaps>2:continue
                candidates.append((area(candidate),j))
            if not candidates:break
            _,j=min(candidates);group.append(remaining.pop(j))
        partial=any(r['extent']=='partial' for r in group)
        box=union([r['bbox_native'] for r in group]);padding=max(48,min(160,.75*max(box[2]-box[0],box[3]-box[1])))
        cb=crop_box([box],size,padding)
        out.append({'batch_id':f'B{len(out)+1:04d}','focus_ids':[r['id'] for r in group],
                    'context_ids':[],'crop':cb,'complexity':{'focus_count':len(group),'context_count':0,
                    'network_or_large_singleton':complex_object(group[0])}})
    return out


def select_clean_sam(row, masks):
    """Tighten a GLM single only with a unique, close, unmixed mask bbox."""
    candidates=[];b=row['bbox_native']
    for m in masks:
        mb=m['bbox_native'];ratio=area(mb)/max(1,area(b));score=iou(b,mb)
        if score>=.55 and .55<=ratio<=1.4 and inside(mb,b)>=.80:
            candidates.append((score,m))
    candidates.sort(key=lambda x:x[0],reverse=True)
    if not candidates:return None,'no_unique_matching_mask'
    best=candidates[0][1]
    if any(iou(best['bbox_native'],m['bbox_native'])<.85 for _,m in candidates[1:]):
        return None,'competing_masks'
    return best,'unique_close_local_sam_bbox'


def compact_prompt(scope, size, focus, context, masks, region):
    return {'task':'依据干净像素定位focus附近的完整真实箭头/独立连线。优先发现完整范围，框稍松可接受，不做精细测量。短JSON回答。',
      'scope':scope,
      'rules':[
        '坐标是TARGET绝对像素。GLOBAL仅提供语义上下文；后台转换坐标。',
        'BOXES图和本请求使用相同短编号：focus为1、2等，邻居为r1、r2等；标签通过细引线关联框，标签位置不是目标位置。',
        '每个focus必须有objects/removed/unresolved去向。source_ids仅引用focus。context只读。',
        '旧框、旧完整性不是事实。对照像素重新确定整条线，尤其点线/虚线在形状遮挡前后的可见延续；不能只框旧框内的半段。不得仅因旧框偏空白就删除，先在附近定位真实线。',
        'focus已将高重叠/包含的跨视图观察取并集，不需逐条重复比对；框可能过宽或有相邻箭头，需要时拆开。',
        '普通可区分箭头一支一框，完整包含杆与头；真实共享分叉可shared_network；密集神经网络dense_network整体。',
        '仅难以拆清的区域用dense_region。非箭头框边/字形排除，分子只发现外部连线。',
        'SAM只作可选视觉证据，不选择mask ID，不推断精确几何。输出TARGET绝对像素bbox即可。',
        '无SAM不能删除；偏移框有真实对应对象则纠正。超出TARGET保持partial，不编造图外坐标。',
        '重复观察通过objects.source_ids合并，不在removed中再列这些来源。不能以属于其他批次/图外候选为由删除当前真实箭头。removed只排除明确非箭头形状/文字/无真实线内容。理由一短句。'],
      'target_size_px':list(size),'target_region_in_global_px':region,'focus':focus,'context':context,'sam_evidence':masks,
      'output_schema':{'objects':[{'id':'O1','source_ids':['1'],'kind':'single|shared_network|dense_network|dense_region',
        'locator':'短描述','extent':'complete|partial','bbox':[0,0,10,10],
        'uncertain':False,'reason':'短理由'}],
        'removed':[{'source_id':'1','reason':'非范围理由'}],'unresolved':[{'source_id':'1','reason':'不确定原因'}]}}
