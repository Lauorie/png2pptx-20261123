"""Small, explicit repair units. Spatial matches nominate work, not identity."""
import copy
from collections import Counter
from core import area, inside, union, crop_box
from pipeline_frames import Crop
from protocol import validate_refine, validate_discovery


def local_view(size, bounds, cfg):
    w, h = bounds[2]-bounds[0], bounds[3]-bounds[1]
    edge = max(cfg.get('local_min_edge',1080),
               min(cfg.get('local_max_edge',2048), round(max(w,h)*cfg.get('local_scale',3))))
    dims = tuple(max(2,round(v*edge/max(w,h))) for v in (w,h))
    return Crop(tuple(size),tuple(bounds),dims,cfg.get('local_scale',3),cfg.get('local_max_edge',2048))


def expanded(b, margin):
    return [b[0]-margin,b[1]-margin,b[2]+margin,b[3]+margin]


def network_units(rows):
    """Nominate one semantic region task per global network, hiding its fragments.

    No member is discarded here: all original candidates and sources are stored.
    The model re-inventories this bounded region, including independent connectors.
    """
    rows=copy.deepcopy(rows)
    parents=[r for r in rows if r['kind']=='dense_network' and r['extent']=='complete'
             and any(s.startswith('G/') for s in r['source_ids'])]
    choices={}; audits=[]
    for child in rows:
        if child in parents: continue
        matches=[]
        for parent in parents:
            b=parent['bbox_native']; c=child['bbox_native']
            margin=max(12,min(48,.08*max(b[2]-b[0],b[3]-b[1])))
            if area(c)>area(b)*1.1 or inside(c,expanded(b,margin))<.85:continue
            if any(s.startswith('G/') for s in child['source_ids']) and child['extent']=='complete':continue
            grown=union([b,c])
            if area(grown)>area(b)*1.8:continue
            matches.append((inside(c,b),-area(grown)/max(1,area(b)),parent['id']))
        matches.sort(reverse=True)
        if matches and (len(matches)==1 or matches[0][0]-matches[1][0]>.15):
            choices[child['id']]=matches[0][2]
    index={r['id']:r for r in rows}
    for parent in parents:
        members=[parent]+[index[k] for k,v in choices.items() if v==parent['id']]
        originals=copy.deepcopy(members)
        parent['bbox_native']=union([r['bbox_native'] for r in members])
        parent['source_ids']=sorted({s for r in members for s in r['source_ids']})
        parent['region_members']=originals
        parent['network_region']=True
        parent['review_reasons']=['network_region_inventory']
        audits.append({'unit_id':parent['id'],'member_ids':[r['id'] for r in originals],
                       'candidate_count':len(originals),'bbox_native':parent['bbox_native'],
                       'identity_confirmed':False})
    return [r for r in rows if r['id'] not in choices],audits


def link_fragments(rows):
    """Give incomplete observations a possible full extent to inspect together."""
    rows=copy.deepcopy(rows);links=[]
    for child in rows:
        if child['extent']!='partial' or child.get('network_region'):continue
        candidates=[];c=child['bbox_native']
        for parent in rows:
            if parent['id']==child['id'] or parent['extent']!='complete' or parent.get('network_region'):continue
            if parent['kind']!=child['kind'] and not (parent['kind']=='serial_chain' and child['kind']=='single'):continue
            b=parent['bbox_native']
            if area(c)>area(b)*.65:continue
            coverage=inside(c,expanded(b,8))
            if coverage>=.70:candidates.append((coverage,-area(b),parent))
        if not candidates:continue
        parent=max(candidates,key=lambda x:x[:2])[2]
        child['possible_parent_id']=parent['id']
        parent['review_reasons']=list(dict.fromkeys(parent.get('review_reasons',[])+['possible_fragment']))
        links.append({'child':child['id'],'possible_parent':parent['id'],'confirmed_duplicate':False})
    return rows,links


def packet_plan(rows,size,cfg,base_planner):
    pending=copy.deepcopy(rows);index={r['id']:r for r in pending};out=[];used=set()
    # Parent and fragments are examined in one packet; at most three boxes.
    for parent in pending:
        children=[r for r in pending if r.get('possible_parent_id')==parent['id']]
        if not children:continue
        group=[parent]+children[:2];used.update(r['id'] for r in group)
        out.append({'focus_ids':[r['id'] for r in group],'context_ids':[],
                    'crop':crop_box([r['bbox_native'] for r in group],size,48),
                    'complexity':{'focus_count':len(group),'fragment_with_full_candidate':True}})
    rest=base_planner([r for r in pending if r['id'] not in used],size,cfg)
    out+=rest
    for i,p in enumerate(out,1):
        p['batch_id']=f'B{i:04d}'
        if any(index[k].get('network_region') for k in p['focus_ids']):
            p['network_region']=True
            p['crop']=crop_box([index[k]['bbox_native'] for k in p['focus_ids']],size,24)
    return out


def normalize_answer(answer,size,aliases,restore):
    """Keep unsolicited discoveries separate without retrying valid focus work."""
    ans=copy.deepcopy(answer)
    if not isinstance(ans,dict):raise ValueError('Answer object required')
    # Omitting an empty disposition list cannot hide a missing focus: the
    # complete-accounting validator below still checks every supplied ID.
    for key in ('removed','unresolved'):ans.setdefault(key,[])
    for o in ans.get('objects',[]):
        if o.get('uncertain') in ('true','false'):
            o['uncertain']=o['uncertain']=='true'
    extras=list(ans.get('extras',[]));focus=[]
    for o in ans.get('objects',[]):
        if o.get('source_ids')==[]: extras.append(o)
        else: focus.append(o)
    ans['objects']=focus
    for o in focus:o.update(bbox_source='model',original_id=None,sam_ids=[],complete_support=False)
    restored=restore(ans,aliases)
    validate_refine(restored,size,list(aliases.values()),[])
    valid=[];rejected=[]
    for i,o in enumerate(extras):
        try:
            validate_discovery({'objects':[{**o,'id':f'E{i+1}'}]},size)
            valid.append(o)
        except (ValueError,TypeError,KeyError) as exc:rejected.append({'object':o,'error':str(exc)})
    ans['extras']=valid;ans['rejected_extras']=rejected
    return ans


def repair_prompt(size,focus,network=False,global_context=False,sam=False):
    rules=[
      '本阶段只发现完整范围，不做精细几何。框稍松、稍长、稍宽均可；直接给完整粗框即可，不逐像素测量、不计算角度/圆心/线宽、不反复校核边界。',
      '只校正编号focus对应的真实对象；旧框可偏移，附近有对应线就调正。不主动扫描或枚举其他对象，extras默认空。无关线不要挂在focus来源下。',
      '普通箭头一支一框，含整段杆与头。一个头部残段不是另一支箭头。合并同一箭头的重复focus来源。不同并行线、嵌套跳连分别保留。',
      '一条连续连接上有多个箭头头时，发现阶段允许用一个serial_chain整体框，包含整条杆、所有头、转弯和末端无头尾段；后续几何重建再按头分段。本阶段不要仅框一个头、末端折角或半条线。不同并行/嵌套连接仍分别框出。',
      '神经网络按一层间连线或一个明确子网络整体框出；密集平行示意线束按一束整体框出。组内边/头/短线不再单列。网络外的普通连接独立。',
      '独立无头连线、虚线、坐标轴、分隔线、引线、括号和块箭头属于范围；非箭头闭合形状轮廓、格边、文字笔画、照片纹理排除；分子仅外部连线。',
      '可见虚线跨遮挡的同一连接须框全。裁剪截断则partial，后台补视野；不要凭空补线。所有坐标均为TARGET绝对像素，后台变换。',
      '每个focus在objects/removed/unresolved中有去向；objects.source_ids只能引用focus短编号。removed仅用于明确非范围内容，重复通过合并来源表达。理由一短句。']
    if network:
        rules.insert(0,'这是一个网络/线束区域任务，框内候选碎片已由后台暂收集，不逐条编号。group_scope仅帮助确认是哪一层/哪一束，其中旧坐标和数量不作事实。依据像素输出这层/这束的完整组；框内确有独立外连箭头才另列。邻层/邻束出现在crop边缘不意味着本组partial，不要把邻层纳入本组。')
    return {'task':'校正给定身份和完整bbox，输出简短JSON。','target_size_px':list(size),
        'images':'TARGET干净局部；BOXES同坐标短编号。'+('GLOBAL只供语义参考。' if global_context else '')+('SAM只供可选像素证据，不输出mask编号。' if sam else ''),
        'rules':rules,'focus':focus,
        'output_schema':{'objects':[{'id':'O1','source_ids':['1'],'kind':'single|serial_chain|shared_network|dense_network|dense_region',
           'bbox':[0,0,10,10],'locator':'短对象描述','extent':'complete|partial','uncertain':False,'reason':'短依据'}],
           'removed':[{'source_id':'1','reason':'非范围依据'}],'unresolved':[{'source_id':'1','reason':'歧义'}],
           'extras':[]}}
