"""At most two boxes per request, only for residual high-overlap identities."""
import copy
from concurrent.futures import ThreadPoolExecutor,as_completed
from PIL import Image
from core import area,inside,union,crop_box
from light_core import iou
from repair_policy import local_view
from evidence import save,local_box
from visual_labels import render
from transport.common import write_json as write


def preserve_chain_scope(objects,observations):
    out=copy.deepcopy(objects);audit=[]
    for o in out:
        if o['kind']!='serial_chain':continue
        candidates=[r for r in observations if r['id'] in o['source_ids'] and r.get('source_view')=='G'
                    and r['extent']=='complete' and r['kind'] in {'single','serial_chain'}]
        ranked=sorted([(iou(o['bbox_native'],r['bbox_native']),r['id'],r) for r in candidates],key=lambda t:t[:2],reverse=True)
        if not ranked or ranked[0][0]<.45:continue
        if len(ranked)>1 and ranked[0][0]-ranked[1][0]<.15:continue
        ref=ranked[0][2];before=o['bbox_native'];o['bbox_native']=union([before,ref['bbox_native']])
        o['scope_envelope']={'model_bbox_native':before,'global_source_id':ref['id'],
            'reason':'Retain the coarse complete discovery extent; geometry is not estimated here'}
        audit.append({'id':o['id'],'before':before,'after':o['bbox_native'],'source':ref['id']})
    return out,audit


def nominees(objects,limit=12):
    pairs=[]
    for i,a in enumerate(objects):
        for b in objects[i+1:]:
            if set(a.get('split_families',[]))&set(b.get('split_families',[])):continue
            x,y=a['bbox_native'],b['bbox_native'];contain=max(inside(x,y),inside(y,x));score=iou(x,y)
            if contain<.70 and score<.55:continue
            if a['kind']!=b['kind'] and not ({'dense_network','shared_network','serial_chain'} & {a['kind'],b['kind']}):continue
            pairs.append({'a':a['id'],'b':b['id'],'score':max(score,contain*.8)})
    pairs.sort(key=lambda p:(-p['score'],p['a'],p['b']))
    for i,p in enumerate(pairs,1):p['id']=f'P{i:03d}'
    return (pairs,[]) if limit is None else (pairs[:limit],pairs[limit:])


def reconcile(root,cid,objects,model):
    folder=root/'cases'/cid/'residual_review';index={o['id']:o for o in objects}
    pairs,deferred=nominees(objects,model.config.get('residual_pair_limit',12))
    write(folder/'plan.json',{'pairs':pairs,'deferred':deferred,'max_boxes_per_request':2})
    native=Image.open(root/'inputs'/cid/'source.png').convert('RGB')
    def review(pair):
        out=folder/pair['id'];focus=[index[pair[k]] for k in ['a','b']]
        cb=crop_box([r['bbox_native'] for r in focus],native.size,32);view=local_view(native.size,cb,model.config)
        save(view.image(native),out/'target.png')
        render(out/'target.png',focus,out/'boxes.png',view,mode='request')
        q={'task':'只判断这两个框是否重复。不要重新定位或估算精确坐标。',
           'rules':['TARGET是干净局部，BOXES同坐标，1和2仅是候选身份。',
             '同一完整物理身份的重复观察才是same；不同的完整箭头、同一网络的完整子成员不能按same合并。成员关系用a_in_b或b_in_a，不能仅依据bbox包含。',
             '独立并行箭头、上下嵌套跳连、不同网络层或仅bbox相交的对象为distinct；只凭框包含不能认定重复。',
             '不确定用uncertain，不删除任何线。连续多头连接允许整体框，几何阶段再分段。'],
           'focus':[{'id':str(i+1),'kind':r['kind'],'extent':r['extent'],
                'bbox':[round(v) for v in local_box(r['bbox_native'],view)]} for i,r in enumerate(focus)],
           'target_size_px':list(view.input_size),
           'output_schema':{'relation':'same|a_in_b|b_in_a|distinct|uncertain','reason':'一句可见依据'}}
        def validate(ans):
            if ans.get('relation') not in {'same','a_in_b','b_in_a','distinct','uncertain'} or not str(ans.get('reason','')).strip():raise ValueError('relation and reason required')
            return ans
        ans=model.ask(out/'request',q,[('TARGET',out/'target.png'),('BOXES',out/'boxes.png')],validate)
        write(out/'result.json',ans);return {**pair,**ans}
    decisions=[];errors=[]
    with ThreadPoolExecutor(max_workers=min(12,max(1,len(pairs)))) as fallback:
        submit=lambda p:model.task_pool.submit(cid,review,p) if hasattr(model,'task_pool') else fallback.submit(review,p)
        jobs={submit(p):p for p in pairs}
        for f in as_completed(jobs):
            try:decisions.append(f.result())
            except Exception as exc:errors.append({**jobs[f],'error':str(exc)})
    current=copy.deepcopy(index);aliases={k:k for k in index};groups={k:{k} for k in index};applied=[];conflicts=[]
    distinct={frozenset((d['a'],d['b'])) for d in decisions if d['relation'] in {'distinct','a_in_b','b_in_a'}}
    for d in sorted(decisions,key=lambda d:d['id']):
        if d['relation'] not in {'same','a_in_b','b_in_a'}:continue
        a,b=aliases[d['a']],aliases[d['b']]
        if a==b:continue
        if d['relation'] in {'a_in_b','b_in_a'}:
            owner,child=(b,a) if d['relation']=='a_in_b' else (a,b)
            current[owner].setdefault('readonly_component_ids',[]).append(child)
            current[owner].setdefault('component_relation_evidence',[]).append(d)
            continue
        if any(frozenset((x,y)) in distinct for x in groups[a] for y in groups[b]):
            conflicts.append({**d,'reason':'Transitive merge conflicts with explicit distinct judgement'});continue
        left,right=current[a],current[b]
        if d['relation']=='a_in_b':owner,child=b,a
        elif d['relation']=='b_in_a':owner,child=a,b
        else:
            owner,child=sorted([a,b],key=lambda k:(current[k]['extent']!='complete',-area(current[k]['bbox_native']),k))
        if d['relation']!='same' and current[owner]['kind'] not in {'dense_network','shared_network','serial_chain'}:
            conflicts.append({**d,'reason':'Claimed member owner is not a network/chain'});continue
        keep=current[owner];drop=current[child]
        keep['bbox_native']=union([keep['bbox_native'],drop['bbox_native']])
        keep['source_ids']=sorted(set(keep['source_ids'])|set(drop['source_ids']))
        duplicate=copy.deepcopy(drop)
        duplicate['history_relation']={'relation':'same','owner_id':owner,'pair_id':d['id'],'trace':str(folder/d['id']/'request')}
        keep.setdefault('residual_members',[]).append(duplicate)
        keep['extent']='complete' if keep['extent']=='complete' or (d['relation']=='same' and drop['extent']=='complete') else 'partial'
        keep['reason']='Two-box semantic review: '+d['reason']
        for k in groups[child]:aliases[k]=owner
        groups[owner]|=groups.pop(child);del current[child];applied.append(d)
    result={'decisions':decisions,'applied':applied,'conflicts':conflicts,'errors':errors,'deferred':deferred}
    write(folder/'audit.json',result)
    return list(current.values()),result
