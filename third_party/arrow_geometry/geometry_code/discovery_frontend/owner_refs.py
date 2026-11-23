#!/usr/bin/env python3
"""Identity-preserving post-discovery trial. Bboxes never imply ownership."""
import argparse
import copy
import fcntl
import math
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from core import area, intersection, inside, union, crop_box
from evidence import frame, local_box, native_box
from protocol import box as valid_box
from region_partition import local_view, put_view, ask, now, iou
from transport.common import read_json as read, write_json as write, sha256
from transport.model import Model
from visual_labels import render

SYSTEM = ('Inspect actual scientific figure pixels. Preserve complete physical arrows and their identity. '
          'A bbox is a locator, never an ownership or exclusion mask. Completeness before deduplication. '
          'Use absolute TARGET pixel coordinates. Return short JSON.')
GROUPS = {'dense_region','dense_network','shared_network','serial_chain'}
POLICY = [
    '只做一次粗粒度身份判读，不反复猜测。看不清就立即uncertain并保留候选，这是正常结果。不要为了给确定答案而反复比较。',
    'bbox稍宽、稍长都可，禁止为几像素边界反复核对；原框已完整可直接复用original。起终点用节点/位置文字，不计算端点坐标。',
    'GLOBAL只帮助认上下文，绝对不要在GLOBAL和TARGET间心算坐标映射；只给TARGET整数粗框。',
    '完整性第一，允许冗余。独立箭头即使位于大bbox内，仍保留自己的完整身份。',
    'bbox相交、包含、同色、同起点不证明同一箭头。确实是同一物理连接的重复观察才合并。',
    '起点、终点和可见路径可稳定指认的直线/曲线/虚线分别处理，允许bbox高度重叠。不要为分框把一支箭头切成残段。',
    '真正共享主干/分支网络或难以辨认的局部可保留组；图叫神经网络本身不是合组理由。',
    'r编号蓝框是其他任务负责的确定身份，仅该身份不重建。蓝框矩形不是禁绘区；穿过蓝框的其他目标线仍需保留。',
    '旧框、旧文字和complete都只是线索，不是事实。检查实际杆、头、端点与方向；不因旧框偏空白就丢掉附近真线。',
    '范围只需完整粗框，不拟合几何、不逐像素测量。非箭头闭合框边、格边、文字不是连线；块箭头属于范围。',
]
LOCK = threading.Lock()


def compact(o, view, alias):
    return {'id':alias,'kind':o['kind'],'bbox':[round(x) for x in local_box(o['bbox_native'],view)],
            'original_id':o['id'],'source_focus_ids':o.get('input_ids',[]),
            'locator_hint':o.get('locator',''),'extent_hint':o.get('extent','partial'),
            'start_hint':o.get('start_hint',''),'end_hint':o.get('end_hint',''),
            'route_hint':o.get('route_hint','')}


def refs_images(folder, target, view, focus, refs, page_size=12):
    """All refs shown, in pages; aliases agree across images and prompt."""
    refs=[{**r,'visual_color':'#1565c0'} for r in refs]
    mapping={r['id']:str(i+1) for i,r in enumerate(focus)}
    mapping.update({r['id']:'r'+str(i+1) for i,r in enumerate(refs)})
    images=[]
    for i in range(max(1,math.ceil(len(refs)/page_size))):
        page=refs[i*page_size:(i+1)*page_size]
        p=folder/f'boxes_{i+1:02d}.png'
        render(target,focus,p,view,context=page,mode='request',label_map=mapping)
        images.append((f'BOXES_{i+1}',p))
    write(folder/'aliases.json',mapping)
    return images


def absorbed_candidates(objects):
    found=[];seen=set()
    def visit(o, parent):
        for field in ('interior_member_observations','residual_members'):
            for child in o.get(field,[]):
                if o['kind']==child['kind']=='single' and child.get('history_relation',{}).get('relation')=='same':
                    # A saved duplicate is an observation of its existing owner.
                    # Preserve its full evidence under the owner, without a second task.
                    continue
                if child['kind']=='single':
                    key=(tuple(child['bbox_native']),tuple(sorted(child.get('source_ids',[]))))
                    if key not in seen:
                        seen.add(key);c=copy.deepcopy(child)
                        c.update(id=f'REC{len(found)+1:04d}',recovered_from_parent=parent,
                                 certainty='candidate',verified_independent=False)
                        found.append(c)
                visit(child,parent)
    for o in objects:visit(o,o['id'])
    return found


def risky_targets(objects, recovered, verified_ids):
    groups=[o for o in objects if o['kind'] in GROUPS]
    targets=[]
    for o in objects:
        if o['kind']!='single' or o['id'] in verified_ids:continue
        b=o['bbox_native']
        risk=(o.get('uncertain') or o.get('extent')!='complete' or
              any(intersection(b,g['bbox_native'])>0 for g in groups) or
              any(x['id']!=o['id'] and x['kind']=='single' and
                  (iou(b,x['bbox_native'])>=.30 or max(inside(b,x['bbox_native']),inside(x['bbox_native'],b))>=.85)
                  for x in objects))
        if risk:targets.append(copy.deepcopy(o))
    return targets+recovered


def batches(targets, size):
    remaining=sorted(targets,key=lambda o:(o['bbox_native'][1],o['bbox_native'][0],o['id']))
    result=[]
    while remaining:
        chunk=[remaining.pop(0)]
        while remaining and len(chunk)<4:
            b=union([o['bbox_native'] for o in chunk]);choices=[]
            for i,o in enumerate(remaining):
                c=union([b,o['bbox_native']])
                # Do not combine separate panels merely to fill a packet.
                close=intersection(b,o['bbox_native'])>0 or (abs((b[1]+b[3]-o['bbox_native'][1]-o['bbox_native'][3])/2)<70 and c[2]-c[0]<450)
                if close and c[3]-c[1]<max(size[1]*.42,200):choices.append((area(c),i))
            if not choices:break
            _,i=min(choices);chunk.append(remaining.pop(i))
        result.append(chunk)
    return result


def validate_objects(a,size,aliases,group=False):
    if not isinstance(a.get('objects'),list):raise ValueError('objects list required')
    used=set();ids=set()
    for o in a['objects']:
        if not isinstance(o.get('id'),str) or o['id'] in ids:raise ValueError('unique id required')
        ids.add(o['id']);valid_box(o.get('bbox'),size)
        if o.get('identity') not in {'independent','uncertain','shared','non_target'}:raise ValueError('identity required')
        if o.get('extent') not in {'complete','partial'}:raise ValueError('extent required')
        if not all(isinstance(o.get(k),str) and o[k].strip() for k in ['locator','start_hint','end_hint','route_hint','reason']):raise ValueError('identity hints required; use unknown explicitly')
        if not group:
            if not o.get('source_ids') or any(str(s) not in aliases for s in o['source_ids']):raise ValueError('valid focus source IDs required')
            used.update(str(s) for s in o['source_ids'])
    if not group and used!=set(aliases):raise ValueError('Every input must be accounted for; uncertainty may retain it')
    if group:
        if len(a['objects'])>6:raise ValueError('At most six separable routes per group, retain rest')
        if not isinstance(a.get('remaining_scope'),str):raise ValueError('remaining scope required')
        valid_box(a.get('complete_group_bbox'),size)
        if type(a.get('claims_no_remainder')) is not bool:raise ValueError('claims_no_remainder boolean required')
    return a


SCHEMA={'id':'o1','source_ids':['1'],'identity':'independent|shared|uncertain|non_target',
        'bbox':'original 或 [x1,y1,x2,y2]整数完整粗框','extent':'complete|partial','locator':'短身份',
        'start_hint':'起点节点及哪侧，未知写unknown','end_hint':'终点及箭头头，未知写unknown',
        'route_hint':'区分同端点邻线的可见路径特征','reason':'一句依据'}


def proposals(root,cid,key,focus,cfg,model,refs=(),group=False,prior=None,full=False):
    folder=root/'cases'/cid/('groups' if group else 'identity')/key/('expanded' if full else 'proposal')
    size=Image.open(root/'inputs'/cid/'source.png').size
    bounds=[0,0,*size] if full else crop_box([o['bbox_native'] for o in focus],size,160 if group else 96)
    view=put_view(root,cid,folder,bounds,cfg)
    shown=[o for o in refs if intersection(o['bbox_native'],bounds)>0]
    aliases={str(i+1):o['id'] for i,o in enumerate(focus)}
    q={'task':('核对组的完整责任范围，提取能够稳定指认的少量独立完整路线，其余明确保留为残余组。' if group else
               '核实编号候选的完整物理身份。只处理focus，不扫描其他箭头。重复focus确认同一条才合并source_ids。'),
       'rules':POLICY+(['最多列6条容易指认的独立路线；真正密集网络不必枚举所有边。剩余线必须留在remaining_scope。',
                        '本请求新增独立路线需要自己的整数粗框，不能使用original占位；不确定就不提取该路线，留在remaining_scope。',
                        '不要再次列出r编号身份。不能因为r框覆盖此区域，就认为区域内其他线也有人负责。',
                        'complete_group_bbox必须包含整个目标的可见延伸，不限于旧框；GLOBAL帮助追踪裁剪外端点。'] if group else []),
       'target_size_px':list(view.input_size),'focus':[compact(o,view,str(i+1)) for i,o in enumerate(focus)],
       'readonly':[compact(o,view,'r'+str(i+1)) for i,o in enumerate(shown)],
       'previous_partial_evidence':prior,
       'output_schema':{'objects':[SCHEMA], **({'complete_group_bbox':[0,0,10,10],'remaining_scope':'尚未交给独立任务的完整连线范围；无则空字符串','claims_no_remainder':False,'reason':'短依据'} if group else {})}}
    images=[('TARGET',folder/'target.png')]+refs_images(folder,folder/'target.png',view,focus,shown)
    if not full and any(o.get('extent')=='partial' or max(o['bbox_native'][2]-o['bbox_native'][0],o['bbox_native'][3]-o['bbox_native'][1])>450 for o in focus):
        images.insert(0,('GLOBAL',root/'inputs'/cid/'source.png'))
    def normalize(a):
        a=copy.deepcopy(a)
        for o in a.get('objects',[]):
            if o.get('bbox')=='original':
                if group:raise ValueError('New group child requires its own coarse bbox')
                source=[f for f in focus if f['id'] in [aliases.get(str(i)) for i in o.get('source_ids',[])]]
                if not source:raise ValueError('original bbox requires source IDs')
                o['bbox']=[max(0,min(round(x),view.input_size[i%2]-1)) for i,x in enumerate(local_box(union([f['bbox_native'] for f in source]),view))]
                o['bbox_reused_from_sources']=True
        return validate_objects(a,view.input_size,aliases,group)
    answer=ask(model,folder,q,images,normalize)
    rows=[]
    for i,o in enumerate(answer['objects']):
        b=native_box(o['bbox'],view);b=[max(0,min(float(x),size[j%2]-1)) for j,x in enumerate(b)]
        touch=any([o['bbox'][0]<=2 and bounds[0]>0,o['bbox'][1]<=2 and bounds[1]>0,
                   o['bbox'][2]>=view.input_size[0]-3 and bounds[2]<size[0],o['bbox'][3]>=view.input_size[1]-3 and bounds[3]<size[1]])
        ids=[focus[0]['id']] if group else [aliases[str(x)] for x in o['source_ids']]
        rows.append({**o,'id':f'{key}/{o["id"]}','kind':'single','bbox_native':b,'input_ids':ids,
                     'source_ids':sorted({s for f in focus if f['id'] in ids for s in f.get('source_ids',[f['id']])}),
                     'extent':'partial' if touch else o['extent'],'uncertain':True,'verified_independent':False,
                     'proposal_trace':str(folder/'request'),'crop_boundary_touch':bool(touch)})
    group_box=native_box(answer['complete_group_bbox'],view) if group else None
    if not full and any(o['extent']=='partial' for o in rows):
        previous={'answer':answer,'frame':view.metadata(size),'instruction':'前轮发现截断。保持同一物理目标，沿已见延伸方向补全，不能换邻近线或只框头部。'}
        return proposals(root,cid,key,focus,cfg,model,refs,group,previous,True)
    write(folder/'mapped.json',{'objects':rows,'group_bbox_native':group_box})
    return rows,answer,group_box


def validate_identity_decisions(a,expected,readonly_count):
    a=copy.deepcopy(a)
    ds=a.get('decisions')
    if not isinstance(ds,list) or {str(x.get('id')) for x in ds}!=expected or len(ds)!=len(expected):raise ValueError('One decision per candidate required')
    for d in ds:
        normalizations=[]
        for key in ['independent_complete','source_identity_preserved']:
            if key in d and d[key] is None:
                d[key]=False
                normalizations.append({'field':key,'from':None,'to':False,'meaning':'unknown is not an accepted identity'})
        if d.get('same_as_readonly') is False:
            d['same_as_readonly']=None
            normalizations.append({'field':'same_as_readonly','from':False,'to':None})
        if normalizations:d['format_normalizations']=normalizations
        if type(d.get('independent_complete')) is not bool or type(d.get('source_identity_preserved')) is not bool or not d.get('reason'):raise ValueError('Complete checks required')
        if d.get('same_as_readonly') is not None and d['same_as_readonly'] not in {'r'+str(i+1) for i in range(readonly_count)}:raise ValueError('Invalid readonly reference')
    return a


def verify(root,cid,key,rows,focus,cfg,model,refs=(),group=False):
    if not rows:return {}
    folder=root/'cases'/cid/'checks'/key
    size=Image.open(root/'inputs'/cid/'source.png').size
    bounds=crop_box([o['bbox_native'] for o in rows+focus],size,96)
    view=put_view(root,cid,folder,bounds,cfg)
    shown=[o for o in refs if intersection(o['bbox_native'],bounds)>0]
    q={'task':'独立检查候选是否为可稳定指认且完整的一支箭头，核实端点和完整线身；检查它是否错误换成邻线。',
       'rules':POLICY+['只认可像素中完整存在的物理对象。模型文字不是证据；未看清起点或终点、追踪不清、换了候选身份，均不要通过。',
                        '候选与r编号是否同一物理连接？若是，写same_as_readonly；仅框相交不算。',
                        '无需精细坐标，输出每个候选一条简短判断。'],
       'candidate':[compact(o,view,str(i+1)) for i,o in enumerate(rows)],
       'original_hints':[compact(o,view,'old'+str(i+1)) for i,o in enumerate(focus)],
       'readonly':[compact(o,view,'r'+str(i+1)) for i,o in enumerate(shown)],
       'target_size_px':list(view.input_size),
       'output_schema':{'decisions':[{'id':'1','independent_complete':True,'source_identity_preserved':True,'same_as_readonly':None,'reason':'可见依据'}]}}
    def valid(a):
        return validate_identity_decisions(a,{str(i+1) for i in range(len(rows))},len(shown))
    images=[('TARGET',folder/'target.png')]+refs_images(folder,folder/'target.png',view,rows,shown)
    a=ask(model,folder,q,images,valid);decisions={}
    for d in a['decisions']:
        r=rows[int(d['id'])-1];decision={**d}
        decision['same_as_id']=shown[int(d['same_as_readonly'][1:])-1]['id'] if d.get('same_as_readonly') else None
        decision['accepted']=d['independent_complete'] and d['source_identity_preserved'] and r['identity']=='independent' and r['extent']=='complete'
        decisions[r['id']]=decision
    write(folder/'mapped_decisions.json',decisions)
    return decisions


def run_identity_batch(root,cid,index,focus,cfg,model):
    key=f'A{index:03d}'
    prior=Path(cfg['completed_check_import_run']) if cfg.get('completed_check_import_run') else None
    if prior and sha256(prior/'cases'/cid/'baseline_result.json')==sha256(root/'cases'/cid/'baseline_result.json'):
        check=prior/'cases'/cid/'checks'/key/'mapped_decisions.json'
        packet=prior/'cases'/cid/'identity'/key
        mapped=packet/'expanded/mapped.json' if (packet/'expanded/mapped.json').exists() else packet/'proposal/mapped.json'
        if check.exists() and mapped.exists():
            rows=read(mapped)['objects'];ds=read(check)
            if set(i for o in rows for i in o['input_ids'])=={o['id'] for o in focus} and set(ds)=={o['id'] for o in rows}:
                result={'key':key,'focus_ids':[o['id'] for o in focus],'objects':rows,'decisions':ds,
                        'imported_complete_check':{'source_run':str(prior),'mapped_sha256':sha256(mapped),'check_sha256':sha256(check),
                        'same_baseline':True,'same_prompt_claimed':False,'new_calls':0}}
                write(root/'cases'/cid/'identity'/key/'completed_check_import.json',result)
                return result
    rows,answer,_=proposals(root,cid,key,focus,cfg,model)
    decisions=verify(root,cid,key,rows,focus,cfg,model)
    return {'key':key,'focus_ids':[o['id'] for o in focus],'objects':rows,'decisions':decisions}


def accept_identity(active,recovered,results):
    active=copy.deepcopy(active);index={o['id']:o for o in active};candidates={o['id']:o for o in active+recovered}
    additions=[];covered=set();audit=[]
    from collections import Counter
    accepted_counts=Counter(i for r in results for o in r['objects'] if r['decisions'].get(o['id'],{}).get('accepted') for i in o['input_ids'])
    for result in results:
        for o in result['objects']:
            d=result['decisions'].get(o['id'],{})
            if not d.get('accepted'):continue
            x=copy.deepcopy(o);x.update(verified_independent=True,uncertain=False,certainty='independently_checked',identity_check=d)
            existing=[i for i in x['input_ids'] if i in index]
            split_source=any(accepted_counts[i]>1 for i in x['input_ids'])
            chosen=existing[0] if existing and not split_source else x['id']
            x['id']=chosen
            if chosen in index:
                x['display_id']=index[chosen].get('display_id')
                x['readonly_component_ids']=copy.deepcopy(index[chosen].get('readonly_component_ids',[]))
            additions.append(x)
            if not split_source:covered.update(x['input_ids'])
            else:
                for old in active:
                    if old['id'] in x['input_ids']:old.update(uncertain=True,certainty='source_split_pending')
            audit.append({'owner_id':chosen,'input_ids':x['input_ids'],'status':'checked_complete_identity'})
    active=[o for o in active if o['id'] not in covered]+additions
    for o in recovered:
        if o['id'] not in covered:
            active.append({**copy.deepcopy(o),'uncertain':True,'certainty':'recovered_pending','verified_independent':False})
    return active,audit


def group_job(root,cid,parent,refs,cfg,model):
    key='G_'+parent['id'].replace('/','_')
    size=Image.open(root/'inputs'/cid/'source.png').size
    neighborhood=crop_box([parent['bbox_native']],size,160)
    refs=[r for r in refs if intersection(r['bbox_native'],neighborhood)>0]
    rows,answer,gb=proposals(root,cid,key,[parent],cfg,model,refs,True,full=True)
    ds=verify(root,cid,key,rows,[parent],cfg,model,refs,True)
    accepted=[];aliases=[]
    for o in rows:
        d=ds.get(o['id'],{})
        if d.get('accepted') and not d.get('same_as_id'):
            accepted.append({**o,'verified_independent':True,'uncertain':False,'certainty':'independently_checked','identity_check':d})
        elif d.get('accepted') and d.get('same_as_id'):aliases.append({'candidate':o['id'],'owner':d['same_as_id']})
    # Responsibility of the original group survives. No visual absence claim is used to delete it.
    updated=copy.deepcopy(parent)
    updated['unverified_scope_proposal']={'bbox_native':gb,'remaining_scope':answer['remaining_scope'],'claims_no_remainder':answer['claims_no_remainder']}
    updated.update(residual_scope='原身份涉及的完整连接中，除已明确分配的只读子身份以外的部分。原身份：'+parent.get('locator',''),
                   ownership_policy='all remaining target identities except explicitly assigned independent references',
                   group_extent_model_claim=answer.get('reason',''),extent_verified=False,
                   uncertain=True,certainty='residual_group_requires_coverage',
                   proposed_no_remainder=answer['claims_no_remainder'])
    return {'parent':updated,'children':accepted,'candidate_children':rows,'aliases':aliases,'checks':ds}


def attach_recovery_links(active, recovered):
    """Keep historical owners visible to the relation stage after proposal IDs change."""
    parents={o['id']:o.get('recovered_from_parent') for o in recovered}
    rows=copy.deepcopy(active)
    for o in rows:
        links=set(o.get('identity_parent_candidates',[]))
        for source in [o['id']]+o.get('input_ids',[])+o.get('confirmed_duplicate_ids',[]):
            if parents.get(source):links.add(parents[source])
        if o.get('recovered_from_parent'):links.add(o['recovered_from_parent'])
        links.discard(o['id'])
        o['identity_parent_candidates']=sorted(links)
    return rows


def relation_pairs(active):
    singles=[o for o in active if o['kind']=='single']
    pairs=[]
    for i,a in enumerate(singles):
        for b in singles[i+1:]:
            if a.get('certainty')==b.get('certainty')=='prior_partition_checked':continue
            linked=(a['id'] in b.get('identity_parent_candidates',[]) or b['id'] in a.get('identity_parent_candidates',[]))
            if not linked and not (a.get('verified_independent') or b.get('verified_independent')):continue
            overlap=iou(a['bbox_native'],b['bbox_native']);contain=max(inside(a['bbox_native'],b['bbox_native']),inside(b['bbox_native'],a['bbox_native']))
            if linked or overlap>=.55 or contain>=.90:pairs.append((a,b))
    return sorted(pairs,key=lambda p:-iou(p[0]['bbox_native'],p[1]['bbox_native']))


def pair_job(root,cid,i,pair,cfg,model):
    folder=root/'cases'/cid/'relations'/f'P{i:03d}'
    size=Image.open(root/'inputs'/cid/'source.png').size;view=put_view(root,cid,folder,crop_box([o['bbox_native'] for o in pair],size,48),cfg)
    q={'task':'判断两个候选的物理身份关系，包括整体与已指认子连接。','rules':POLICY+[
       'same仅用于同一完整物理身份的重复观察；仅bbox相交、包含不能判same或成员关系。',
       'a_contains_b表示1号实际由多条连接组成，2号是其中一条完整子身份；b_contains_a相反。两者都保留，不把整体与子身份合并。',
       '一支箭头的残段不是完整子身份；端点或路线不清用uncertain，不依靠旧complete文字猜测。'],
       'focus':[compact(o,view,str(j+1)) for j,o in enumerate(pair)],'target_size_px':list(view.input_size),
       'output_schema':{'relation':'same|different|a_contains_b|b_contains_a|uncertain','reason':'依据起终点、杆与头说明'}}
    def valid(a):
        if a.get('relation') not in {'same','different','a_contains_b','b_contains_a','uncertain'} or not a.get('reason'):raise ValueError('relation required')
        return a
    a=ask(model,folder,q,[('TARGET',folder/'target.png')]+refs_images(folder,folder/'target.png',view,pair,[]),valid)
    return {'a':pair[0]['id'],'b':pair[1]['id'],**a}


def reconcile(active,decisions):
    index={o['id']:copy.deepcopy(o) for o in active};aliases={i:i for i in index};members={i:{i} for i in index};conflicts=[]
    different={frozenset((d['a'],d['b'])) for d in decisions if d['relation'] in {'different','a_contains_b','b_contains_a'}}
    applied=[]
    for d in decisions:
        if d['relation']!='same':continue
        a,b=aliases[d['a']],aliases[d['b']]
        if a==b:continue
        if any(frozenset((x,y)) in different for x in members[a] for y in members[b]):conflicts.append(d);continue
        # Preserve stable original identity IDs whenever possible.
        if index[b].get('certainty')=='prior_partition_checked' and index[a].get('certainty')!='prior_partition_checked':a,b=b,a
        elif not index[a].get('display_id') and index[b].get('display_id'):a,b=b,a
        if not index[a].get('verified_independent') and index[b].get('verified_independent'):
            for key in ['verified_independent','uncertain','certainty','identity_check','extent','locator','start_hint','end_hint','route_hint','proposal_trace']:
                if key in index[b]:index[a][key]=copy.deepcopy(index[b][key])
            index[a]['identity_evidence_source_id']=b
        index[a]['identity_parent_candidates']=sorted(set(index[a].get('identity_parent_candidates',[]))|set(index[b].get('identity_parent_candidates',[])))
        index[a]['readonly_component_ids']=sorted(set(index[a].get('readonly_component_ids',[]))|set(index[b].get('readonly_component_ids',[])))
        index[a]['bbox_native']=union([index[a]['bbox_native'],index[b]['bbox_native']])
        index[a]['source_ids']=sorted(set(index[a].get('source_ids',[]))|set(index[b].get('source_ids',[])))
        index[a]['input_ids']=sorted(set(index[a].get('input_ids',[]))|set(index[b].get('input_ids',[])))
        index[a].setdefault('confirmed_duplicate_ids',[]).extend([b]+index[b].get('confirmed_duplicate_ids',[]))
        for x in members[b]:aliases[x]=a
        members[a]|=members.pop(b);del index[b];applied.append({**d,'owner':a,'alias':b})
    resolved=dict(aliases)
    for o in index.values():
        for source in o.get('input_ids',[]):resolved.setdefault(source,o['id'])
    edges=set()
    for o in index.values():
        for child in o.get('readonly_component_ids',[]):
            target=resolved.get(child,child)
            if target in index and target!=o['id']:edges.add((o['id'],target))
    for d in decisions:
        if d['relation'] not in {'a_contains_b','b_contains_a'}:continue
        a,b=aliases[d['a']],aliases[d['b']]
        parent,child=(a,b) if d['relation']=='a_contains_b' else (b,a)
        if parent==child:conflicts.append({**d,'conflict':'component_collapsed_into_parent'});continue
        edges.add((parent,child))
    # Whole/part cycles are contradictory evidence; do not produce reciprocal exclusions.
    def reaches(start,target,skip):
        seen=set();todo=[start]
        while todo:
            node=todo.pop()
            if node==target:return True
            if node in seen:continue
            seen.add(node);todo.extend(b for a,b in edges if a==node and (a,b)!=skip)
        return False
    bad={edge for edge in edges if reaches(edge[1],edge[0],edge)}
    for a,b in bad:conflicts.append({'a':a,'b':b,'conflict':'component_cycle'})
    for o in index.values():
        o['readonly_component_ids']=sorted(b for a,b in edges-bad if a==o['id'])
        if o['readonly_component_ids']:
            o.update(verified_independent=False,uncertain=True,certainty='component_parent_requires_coverage')
    return list(index.values()),applied,conflicts


def export_handoff(root,cid,active,cfg):
    c=root/'cases'/cid;size=Image.open(root/'inputs'/cid/'source.png').size
    verified=[o for o in active if o.get('verified_independent')]
    tasks=[];gallery=[]
    for o in active:
        component_ids=set(o.get('readonly_component_ids',[]))
        refs=[r for r in verified if r['id']!=o['id'] and (r['id'] in component_ids or intersection(r['bbox_native'],o['bbox_native'])>0)]
        # Uncertain singles do not exclude overlapping other singles: avoid reciprocal guessing.
        if o['kind'] not in GROUPS and o.get('certainty')!='source_split_pending':refs=[r for r in refs if r['id'] in component_ids]
        folder=c/'handoff'/o['display_id']
        bounds=[0,0,*size] if o['kind'] in GROUPS or o.get('extent')=='partial' else crop_box([o['bbox_native']],size,96)
        view=put_view(root,cid,folder,bounds,cfg)
        images=refs_images(folder,folder/'target.png',view,[o],refs)
        task={'target_id':o['id'],'display_id':o['display_id'],'kind':o['kind'],
              'bbox_native':o['bbox_native'],'identity_hint':o.get('locator',''),
              'residual_scope':o.get('residual_scope'), 'certainty':o.get('certainty','baseline_asserted'),
              'extent_verified':o.get('verified_independent',False),
              'readonly_identities':[compact(r,view,'r'+str(i+1))|{'owner_id':r['id']} for i,r in enumerate(refs)],
              'target_image':str(folder/'target.png'),'reference_images':[str(p) for _,p in images],
              'view':view.metadata(size),'bbox_is_hint_not_pixel_clip':True,
              'readonly_rectangles_are_not_exclusion_masks':True,
              'instruction':'重建目标完整身份或残余组。蓝色r编号只读，由指定owner负责；仅跳过该箭头身份，不清空其矩形，不删穿过蓝框的其他线。不能辨别的交叉保持待确认，不能猜测删除。',
              'render_owner':True,'geometry_executed':False,
              'requires_review':bool(o.get('uncertain') or o.get('extent')!='complete'),
              'task_role':'coverage_review' if component_ids or o['kind'] in GROUPS else 'candidate_review' if o.get('uncertain') else 'independent_owner'}
        write(folder/'task.json',task);tasks.append(task)
        if refs:gallery.append({'target':o['display_id'],'readonly_count':len(refs),'image':str(images[0][1]),'pages':len(images)})
    write(c/'reconstruction_handoff.json',{'case_id':cid,'tasks':tasks,'gallery':gallery,'geometry_executed':False,'physical_completeness_measured':False})
    return gallery


def execute_case(root,cid,cfg,model):
    c=root/'cases'/cid
    if (c/'finished.json').exists():return read(c/'result.json')
    baseline=read(c/'baseline_result.json');active=copy.deepcopy(baseline['objects'])
    committed={i for x in baseline.get('region_ledger',[]) if x['commit'] for i in x['active_owner_ids']}
    for o in active:
        o['verified_independent']=o['id'] in committed and o['kind']=='single'
        o['certainty']='prior_partition_checked' if o['verified_independent'] else 'baseline_asserted'
    recovered=absorbed_candidates(active);size=Image.open(root/'inputs'/cid/'source.png').size
    targets=risky_targets(active,recovered,committed);packets=batches(targets,size)
    write(c/'identity_plan.json',{'recovered':recovered,'targets':[o['id'] for o in targets],'batches':[[o['id'] for o in p] for p in packets]})
    print({'case':cid,'stage':'identity','batches':len(packets),'recovered':len(recovered)},flush=True)
    results=[];errors=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        fs={pool.submit(run_identity_batch,root,cid,i,p,cfg,model):i for i,p in enumerate(packets,1)}
        for f in as_completed(fs):
            try:results.append(f.result())
            except Exception as e:errors.append({'stage':'identity','batch':fs[f],'error':str(e)})
            write(c/'status.json',{'stage':'identity','done':len(results)+len(errors),'total':len(packets),'errors':len(errors)})
    active,identity_audit=accept_identity(active,recovered,results)
    target_ids={o['id'] for o in targets}
    for o in active:
        if o['id'] in target_ids and not o.get('verified_independent'):
            o['uncertain']=True
            if o.get('certainty')!='source_split_pending':o['certainty']='identity_check_pending'
    write(c/'after_identity.json',{'objects':active,'identity_audit':identity_audit,'errors':errors})
    refs=[o for o in active if o.get('verified_independent')]
    parents=[o for o in active if o['kind'] in GROUPS and (o.get('uncertain') or o['kind']=='dense_region' or
             any(intersection(o['bbox_native'],r['bbox_native'])>0 for r in refs))]
    group_results=[]
    print({'case':cid,'stage':'groups','groups':len(parents),'verified':len(refs)},flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        fs={pool.submit(group_job,root,cid,p,refs,cfg,model):p['id'] for p in parents}
        for f in as_completed(fs):
            try:group_results.append(f.result())
            except Exception as e:errors.append({'stage':'groups','parent':fs[f],'error':str(e)})
    replaced={r['parent']['id'] for r in group_results}
    active=[o for o in active if o['id'] not in replaced]+[r['parent'] for r in group_results]+[o for r in group_results for o in r['children']]
    active=attach_recovery_links(active,recovered)
    pairs=relation_pairs(active);decisions=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        fs={pool.submit(pair_job,root,cid,i,p,cfg,model):p for i,p in enumerate(pairs,1)}
        for f in as_completed(fs):
            try:decisions.append(f.result())
            except Exception as e:errors.append({'stage':'relations','pair':[o['id'] for o in fs[f]],'error':str(e)})
    decisions.sort(key=lambda d:(d['a'],d['b']))
    active,applied,conflicts=reconcile(active,decisions)
    deferred=[]
    write(c/'relations_audit.json',{'decisions':decisions,'applied':applied,'conflicts':conflicts,'deferred':deferred})
    next_id=max((int(o['display_id'][1:]) for o in baseline['objects']),default=0)+1
    for o in active:
        if not o.get('display_id'):o['display_id']=f'I{next_id:04d}';next_id+=1
    original_ids={o['id'] for o in baseline['objects']};final_ids={o['id'] for o in active};resolved={}
    for o in active:
        for old in o.get('input_ids',[])+o.get('confirmed_duplicate_ids',[]):resolved[old]=o['id']
    assert all(i in final_ids or i in resolved for i in original_ids)
    result={'case_id':cid,'baseline_count':len(baseline['objects']),'objects':active,
            'output_identity_count':len(active),'recovered_candidate_count':len(recovered),
            'verified_independent_count':sum(o.get('verified_independent',False) for o in active),
            'pending_identity_count':sum(o.get('uncertain',False) or o.get('extent')!='complete' for o in active),
            'pending_relation_count':len(deferred)+len(conflicts)+sum(d['relation']=='uncertain' for d in decisions),
            'identity_audit':identity_audit,'group_audit':group_results,'aliases':resolved,'errors':errors,
            'region_ledger':copy.deepcopy(baseline.get('region_ledger',[])),
            'state':'completed_with_issues' if errors else 'completed','completed_at':now(),
            'formal_accuracy_measured':False,'geometry_executed':False,
            'policy':'No bbox-only merges or containment absorption; independently checked identities are read-only references for enclosing group tasks.'}
    write(c/'result.json',result)
    inp=root/'inputs'/cid;gv=frame(read(inp/'views.json')['views'][0]['transform'])
    render(inp/'G/target.png',active,c/'overlay.png',gv)
    render(inp/'G/target.png',[o for o in active if o.get('verified_independent')],c/'independent.png',gv)
    gallery=export_handoff(root,cid,active,cfg)
    write(c/'finished.json',{'completed_at':now(),'reference_tasks':len(gallery)})
    print({'case':cid,'stage':'finished','baseline':len(baseline['objects']),'now':len(active),'verified':result['verified_independent_count'],'errors':len(errors)},flush=True)
    with LOCK:publish(root)
    return result


def prepare(root,base,partition,sam,ids,report,imports=None):
    if root.exists():raise ValueError('Fresh run required')
    root.mkdir(parents=True);cfg=read(partition/'config.json')
    cfg.update(api_concurrency=16,case_workers=8,pair_review_budget=None,review_output_dir=str(report),
               protocol_version='owner_refs_v25_unlimited_relations',inventory_replay_run=None,completed_result_import_run=None,
               completed_check_import_run=str(imports) if imports else None,
               request_timeout_seconds=1200,transport_retries=0)
    cfg.pop('reasoning_effort',None)
    assert cfg['max_tokens']==113000 and cfg['thinking']['type']=='enabled'
    write(root/'config.json',cfg);write(root/'cases.json',ids)
    shutil.copytree(Path(__file__).parent,root/'code',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(partition/'constraints_snapshot',root/'constraints_snapshot')
    (root/'constraints_snapshot/user_owner_refs_20261025.txt').write_text('\n'.join(POLICY)+'\n独立身份即使完全被大框包围也保留。大框任务可视化蓝色小框只读；允许冗余、不能缺少。增加图数验证。\n')
    sources=[]
    for cid in ids:
        source=partition if (partition/'cases'/cid/'finished.json').exists() else sam if (sam/'cases'/cid/'finished.json').exists() else base
        shutil.copytree(source/'inputs'/cid,root/'inputs'/cid)
        c=root/'cases'/cid;c.mkdir(parents=True)
        shutil.copy2(source/'cases'/cid/'result.json',c/'baseline_result.json')
        shutil.copy2(source/'cases'/cid/'overlay.png',c/'baseline.png')
        sources.append({'case_id':cid,'source_run':str(source),'result_sha256':sha256(c/'baseline_result.json'),'source_sha256':sha256(root/'inputs'/cid/'source.png')})
    write(root/'baseline_sources.json',sources)
    (root/'PROTOCOL.md').write_text('# 独立身份与只读参考16图\n\n后置对照试验，保留既有v21拆分和v16补漏结果，其余来自v15。原发现与SAM证据固定。\n\n取消本轮bbox包含吸收；找回旧结果中被吸收的single候选，核实身份、起终点和完整路径，经独立复核通过才用作蓝色只读身份。不确定候选保留。\n\n大组重新检查完整范围，允许提取至多6条清楚的独立路线，其余保留残余组。整组不因模型声称为空而删除。蓝框只排除已指定身份，不能裁掉蓝框矩形内其他曲线。\n\n同一箭头仅在语义确认same后合并；不限制每图提名关系数，新核验候选与已有single及其历史owner均比较；冲突与不确定全部记录为pending。普通孤立箭头走原结果；v21已验收空间拆分保留。新增模型调用default thinking，113000，API16/图4。\n\n本轮验证发现、完整身份及下游交接输入，不执行几何/PPT，模型接受不能当作正式召回率。原发现阶段早期误并未全部回放，因此保留原始来源供后续审计。\n')
    files=[p for sub in ['code','inputs','constraints_snapshot','cases'] for p in (root/sub).rglob('*') if p.is_file()]
    with (root/'PROTOCOL.md').open('a') as f:
        f.write('\n\nv23补充：v22出现简单线条反复像素推敲，人工停止并保存23个已完成传输及12个中止传输。v23将提示坐标取整数、允许直接复用原粗框，明确一次判读不清立即保留不确定；局部核验只送TARGET与BOXES，必要的扩展才另送GLOBAL。默认thinking与113000不变。单次总时限300秒、传输自动重试0，超时保留候选并记录，不将超时计作非箭头。可导入v22完整的独立核验结果，须验证baseline哈希/目标集合，保存单独来源记录，不宣称同prompt复用。\n')
    files += [root/'config.json',root/'cases.json',root/'baseline_sources.json',root/'PROTOCOL.md']
    write(root/'manifest.json',{'files':{str(p.relative_to(root)):sha256(p) for p in files},'created_at':now()})
    write(root/'progress.json',{'state':'prepared','total':len(ids)})
    (root/'RESULTS.md').write_text('# 独立身份与只读参考16图\n\n准备完成，结果待验证。用户指定单次总时限1200秒。\n')
    publish(root)


def publish(root):
    if read(root/'config.json').get('integrated_mode'):return
    cfg=read(root/'config.json');dest=Path(cfg['review_output_dir']);dest.mkdir(parents=True,exist_ok=True)
    lines=['# 独立箭头保留与只读参考：16图','','蓝色 r 编号表示其他任务负责的箭头身份；其矩形内部仍可有当前任务需要重建的线。',
           '',f'[方案]({root}/PROTOCOL.md) · [运行结果]({root}/RESULTS.md)','','计数为候选/任务数，不代表真实箭头召回率；未执行几何或PPT。','']
    records=[]
    for cid in read(root/'cases.json'):
        c=root/'cases'/cid;lines += [f'## {cid}','']
        if not (c/'finished.json').exists():lines+=['处理中。',''];continue
        r=read(c/'result.json');h=read(c/'reconstruction_handoff.json')
        lines += [f'原{r["baseline_count"]} → 当前{r["output_identity_count"]}；独立身份已核验{r["verified_independent_count"]}；待确认身份{r["pending_identity_count"]}，关系{r["pending_relation_count"]}。',
                  '', '| 原结果 | 当前全部任务 |','|---|---|',f'| ![baseline]({c}/baseline.png) | ![current]({c}/overlay.png) |','',f'[单独看已核验身份 PNG]({c}/independent.png) · [完整交接清单]({c}/reconstruction_handoff.json)','']
        if h['gallery']:
            g=max(h['gallery'],key=lambda x:x['readonly_count'])
            lines += [f'目标 {g["target"]} 的只读参考示例（共{g["readonly_count"]}个参考身份，{g["pages"]}页；这里只展示第一页）：','',f'![readonly]({g["image"]})','']
        records.append({k:r[k] for k in ['case_id','baseline_count','output_identity_count','verified_independent_count','pending_identity_count','pending_relation_count','state']})
    (dest/'独立身份与只读参考_16例.md').write_text('\n'.join(lines)+'\n')
    write(dest/'summary.json',{'cases':records,'finished':len(records),'total':len(read(root/'cases.json'))})


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    p.add_argument('--base',type=Path);p.add_argument('--partition',type=Path);p.add_argument('--sam',type=Path);p.add_argument('--report',type=Path);p.add_argument('--cases',nargs='+');p.add_argument('--import-checks',type=Path)
    args=p.parse_args();root=args.run_dir.resolve()
    if args.prepare:return prepare(root,args.base,args.partition,args.sam,args.cases,args.report,args.import_checks)
    if Path(__file__).resolve().parent!=root/'code':raise RuntimeError('Run frozen code only')
    lock=(root/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for rel,h in read(root/'manifest.json')['files'].items():
        if sha256(root/rel)!=h:raise RuntimeError('Frozen file changed '+rel)
    cfg=read(root/'config.json');assert 'reasoning_effort' not in cfg and cfg['max_tokens']==113000
    model=Model(cfg,run_dir=root,system=SYSTEM);records=[]
    write(root/'process.json',{'pid':os.getpid(),'started_at':now()})
    with ThreadPoolExecutor(max_workers=cfg['case_workers']) as pool:
        fs={pool.submit(execute_case,root,cid,cfg,model):cid for cid in read(root/'cases.json')}
        for f in as_completed(fs):
            try:r=f.result();records.append({'case_id':r['case_id'],'state':r['state']})
            except Exception as e:
                import traceback
                err={'case_id':fs[f],'state':'failed','error':str(e),'traceback':traceback.format_exc()}
                write(root/'cases'/fs[f]/'failed.json',err);records.append(err)
            write(root/'progress.json',{'state':'running','finished':len(records),'total':len(fs),'records':records,'updated_at':now()})
    write(root/'progress.json',{'state':'finished','records':records,'updated_at':now()});publish(root)


if __name__=='__main__':main()
