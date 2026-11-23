"""Model-requested scope changes and bounded post-edit review. No semantic clustering."""
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image, ImageDraw
import contracts as C
from frames import global_frame, visible_global_bounds
from local_review import (batches, frame_for, global_answer, local_rows, overlap,
                          pair_key, prompt, signature, unresolved_pairs, validate)
from presentation import latest, overlay
from transport.common import write_json as write


def resolve_ids(ids, aliases, live):
    """Follow committed replacements, never reinterpret spatial ownership."""
    def expand(key, seen):
        if key in live:
            return [key]
        if key in seen:
            raise ValueError('Cyclic edit lineage')
        return [v for child in aliases.get(key, []) for v in expand(child, seen | {key})]
    return list(dict.fromkeys(v for key in ids for v in expand(key, set())))


def edit_lineage(focus, mapped, namespace):
    refs = {r['id']: [] for r in focus}
    for key in mapped['keep_ids']:
        refs[key] = [key]
    for i, patch in enumerate(mapped['replacements'], 1):
        targets = [f'{namespace}_{i}_{j}' for j in range(1, len(patch['regions']) + 1)]
        for key in patch['source_ids']:
            refs[key] = targets
    return refs


def choose_jobs(current, pending, aliases, checked, limit, first):
    """Reserve non-overlapping editable IDs, preserving model-requested scopes intact."""
    live = {r['id']: r for r in current}
    used, jobs, deferred, seen = set(), [], [], set()
    for item in pending:
        ids = resolve_ids(item['ids'], aliases, live)
        key = (item['kind'], tuple(sorted(ids)))
        if not ids or key in seen:
            continue
        seen.add(key)
        task = {**item, 'ids': ids}
        if used.intersection(ids):
            deferred.append(task)
        else:
            jobs.append(task); used.update(ids)
    remaining = [r for r in current if r['id'] not in used]
    for group in batches(remaining, limit, None if first else checked):
        jobs.append({'ids': [r['id'] for r in group], 'kind': 'initial' if first else 'boundary'})
    return jobs, deferred


def local_prompt(size, focus, neighbors, kind):
    q = prompt(size, focus, kind == 'boundary')
    q['task'] = ('复查上次拆分或合并后的区域：对照CLEAN检查可见连接是否遗漏、重复，分组是否反而增加恢复难度。必要时允许合回或重拆。'
                 if kind == 'change_check' else
                 '对照CLEAN修正本批定位与粒度。孤立箭头分开，可控线束/分支可整体恢复；必要时先请求加入周边相关候选。')
    q['nearby_candidates'] = [{k: r[k] for k in ('id', 'bbox', 'label')} for r in neighbors]
    q['instructions'][0] = ('CLEAN是判断依据；BOXES中蓝/橙框为本次候选，灰色角标为附近候选，均不是原图笔画。'
        '附近候选只显示在当前裁剪内的部分，不能据此裁短原候选。若必须共同处理，先用include请求其ID；后台会给完整共同视野。')
    q['instructions'][-1] = 'edit、expand、include三种决策选一种直接返回，不加外层包装。include/expand时不提交任何框修改。'
    q['instructions'].append('include只请求与当前目标需要共同判断的候选，不因在同一裁剪内就全部加入；候选数量不是拆分箭头的理由。')
    q['output_schema']['include'] = {'frame': 'LOCAL', 'decision': 'include', 'include_ids': ['附近候选ID'], 'reason': '为何需要一起判断'}
    return q


def validate_scope(answer, size, focus, neighbors):
    if isinstance(answer, dict) and answer.get('decision') == 'include':
        ids = answer.get('include_ids')
        if answer.get('frame') != 'LOCAL':
            raise ValueError('Explicit LOCAL frame required')
        if (not isinstance(ids, list) or not ids or any(not isinstance(x, str) for x in ids)
                or len(set(ids)) != len(ids) or not set(ids) <= {r['id'] for r in neighbors}):
            raise ValueError('include_ids must name distinct nearby candidate IDs')
        if not isinstance(answer.get('reason'), str) or not answer['reason'].strip():
            raise ValueError('Include requires visible reason')
        return {k: answer[k] for k in ('frame', 'decision', 'include_ids', 'reason')}
    return validate(answer, size, focus)


def draw_context(clean, focus, neighbors, path):
    overlay(clean, focus, path)
    with Image.open(path) as src:
        im = src.convert('RGB')
    draw = ImageDraw.Draw(im)
    for r in neighbors:
        x0, y0, x1, y1 = r['bbox']
        arm = min(7, (x1-x0)/3, (y1-y0)/3)
        for x, y, dx, dy in [(x0,y0,1,1),(x1,y0,-1,1),(x0,y1,1,-1),(x1,y1,-1,-1)]:
            draw.line([(x+dx*arm,y),(x,y),(x,y+dy*arm)], fill='#999999', width=1)
        draw.text((x0, max(0,y0-12)), r['id'], fill='#777777')
    im.save(path)


def run(root, cid, rows, model, cfg, state):
    out = root / 'cases' / cid / 'local_review'
    with Image.open(root / 'inputs' / cid / 'source.png') as src:
        source = src.convert('RGB')
    view = global_frame(source.size)
    current, pending, aliases = copy.deepcopy(rows), [], {}
    checked, issues, history = set(), [], []
    scheduled = 0
    for wave in range(cfg.get('local_review_rounds', 3)):
        jobs, deferred = choose_jobs(current, pending, aliases, checked,
                                     cfg.get('local_review_batch_size', 6), wave == 0)
        if not jobs:
            pending = deferred
            break
        folder = out / f'round_{wave+1:02d}'
        write(folder / 'plan.json', {'groups': [j['ids'] for j in jobs], 'jobs': jobs,
            'deferred': deferred, 'input_count': len(current),
            'policy': 'ID reservations only; grouping and scope extensions decided by model'})
        state(f'local_bbox_review_round_{wave+1}')
        snapshot = copy.deepcopy(current)
        index = {r['id']: r for r in snapshot}

        def one(number, job, part=''):
            focus = [index[k] for k in job['ids']]
            base = folder / (f'B{number:03d}' + part)
            namespace = f'LR{wave+1}B{number}{part}'
            try:
                for expansion in range(cfg.get('local_review_expansions', 1)+1):
                    target = base / f'view_{expansion:02d}'
                    target.mkdir(parents=True, exist_ok=True)
                    frame = frame_for(focus, view, cfg.get('local_review_context_scale',1.25), expansion)
                    local, before = local_rows(focus, view, frame)
                    bounds = visible_global_bounds(frame, view)
                    neighbors = [r for r in snapshot if r['id'] not in job['ids'] and overlap(r['bbox'], bounds)>0]
                    nearby, neighbor_audit = local_rows(neighbors, view, frame)
                    frame.image(source).save(target/'clean.png')
                    draw_context(target/'clean.png', local, nearby, target/'boxes.png')
                    write(target/'frame.json', {'crop': frame.metadata(source.size),
                        'global_view': view.metadata(source.size), 'local_bounds_in_global': bounds,
                        'input_conversion': before, 'neighbor_conversion': neighbor_audit})
                    answer = model.ask(target/'request', local_prompt(frame.input_size,local,nearby,job['kind']),
                        [('CLEAN: sole LOCAL output coordinates',target/'clean.png'),
                         ('BOXES: editable candidates and gray nearby candidate corners',target/'boxes.png')],
                        lambda a: validate_scope(a,frame.input_size,local,nearby))
                    if answer['decision']=='expand':
                        continue
                    if answer['decision']=='include':
                        return {'focus': focus, 'replacement': focus, 'lineage': {}, 'changed': False,
                            'pending': [{'ids': job['ids']+answer['include_ids'], 'kind': job['kind'] if job['kind']=='change_check' else 'included_scope',
                                         'reason': answer['reason']}], 'issue': None, 'scope_deferred': True}
                    mapped, audit = global_answer(answer,frame,view)
                    C.review_valid(mapped,view.input_size,focus,False)
                    write(target/'mapped_answer.json',mapped); write(target/'output_conversion.json',audit)
                    replacement = C.apply_review(focus,mapped,namespace)
                    changed = bool(mapped['replacements'])
                    recheck = []
                    if job['kind']!='change_check':
                        for i, patch in enumerate(mapped['replacements'],1):
                            if len(patch['source_ids'])!=1 or len(patch['regions'])!=1:
                                recheck.append({'ids':[f'{namespace}_{i}_{j}' for j in range(1,len(patch['regions'])+1)],
                                    'kind':'change_check','reason':'复查本次拆分/合并，没有预设必须再修改'})
                    return {'focus': focus,'replacement':replacement,'lineage':edit_lineage(focus,mapped,namespace),
                            'changed':changed,'pending':recheck,'issue':None,'scope_deferred':False}
                raise ValueError('Model still requests expansion after allowed local expansion')
            except Exception as exc:
                write(base/'failure.json',{'error':str(exc),'focus_ids':job['ids']})
                if len(focus)>1 and not part:
                    mid=len(focus)//2
                    left=one(number,{**job,'ids':job['ids'][:mid]},'a')
                    right=one(number,{**job,'ids':job['ids'][mid:]},'b')
                    return {'focus':focus,'replacement':left['replacement']+right['replacement'],
                        'lineage':{**left['lineage'],**right['lineage']},'pending':left['pending']+right['pending'],
                        'issue':left['issue'] or right['issue'],'changed':left['changed'] or right['changed'],
                        'scope_deferred':left.get('scope_deferred') or right.get('scope_deferred'), 'split_retry':True}
                return {'focus':focus,'replacement':[{**r,'extent':'uncertain'} for r in focus],
                        'lineage':{},'pending':[],'changed':False,'issue':{'stage':'local_bbox_review',
                         'focus_ids':job['ids'],'error':str(exc),'retained_pending':True}}

        answers = {}
        with ThreadPoolExecutor(max_workers=cfg.get('local_workers',4)) as pool:
            futures={pool.submit(one,i,j):i for i,j in enumerate(jobs,1)}
            for f in as_completed(futures):
                answers[futures[f]]=f.result()
                partial=copy.deepcopy(snapshot)
                for i in sorted(answers):
                    a=answers[i]; ids={r['id'] for r in a['focus']}
                    partial=[r for r in partial if r['id'] not in ids]+a['replacement']
                latest(root,cid,partial,f'local_review_{wave+1:02d}_partial')
        pending=deferred
        for i in sorted(answers):
            a=answers[i]; ids={r['id'] for r in a['focus']}
            current=[r for r in current if r['id'] not in ids]+a['replacement']
            aliases.update({k:v for k,v in a['lineage'].items() if v!=[k]})
            pending.extend(a['pending'])
            if a['issue']:
                issues.append(a['issue'])
            elif not a.get('scope_deferred') and not a.get('split_retry'):
                for members in [a['focus']]+([a['replacement']] if not a['pending'] else []):
                    checked.update(pair_key(x,y) for n,x in enumerate(members) for y in members[n+1:])
            history.append({'round':wave+1,'batch':i,'kind':jobs[i-1]['kind'],'focus_ids':sorted(ids),
                'output_ids':[r['id'] for r in a['replacement']],'pending':a['pending'],'issue':a['issue']})
        scheduled+=len(jobs)
        C.regions_valid(current,view.input_size)
        write(folder/'result.json',{'regions':current,'pending':pending,'unchecked_overlaps':unresolved_pairs(current,checked)})
        latest(root,cid,current,f'local_review_{wave+1:02d}')
    live={r['id'] for r in current}
    pending=[{**j,'ids':resolve_ids(j['ids'],aliases,live)} for j in pending]
    pending=[j for j in pending if j['ids']]
    if pending:
        uncertain={k for j in pending for k in j['ids']}
        current=[{**r,'extent':'uncertain'} if r['id'] in uncertain else r for r in current]
        issues.append({'stage':'local_bbox_review','error':'Local round budget ended with pending scope/post-edit checks',
                       'pending':pending,'retained_pending':True})
    write(out/'audit.json',{'history':history,'scheduled_batches':scheduled,'pending':pending,
        'unchecked_overlaps':unresolved_pairs(current,checked),'issues':issues,
        'all_geometry_decisions_by_model':True,'global_correction_calls':0})
    return current,issues
