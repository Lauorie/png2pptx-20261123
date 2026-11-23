"""Review ONLY new missing-stage regions; baseline localization is immutable."""
import argparse
import copy
import fcntl
import json
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image, ImageDraw
import experiment as E
import contracts as C
import draft_prompts as D
from containment_filter import contains
from pipeline_frames import Crop
from review_spatial import batches, frame_for, pair_key, unresolved_pairs
from transport.model import Model
from supplement_protocol import ReviewProtocol
from addition_scope_review import review as review_addition_scope

SYSTEM='只审核本轮新增的箭头/线条定位框。干净原图是唯一视觉依据，原始框只读。所有输出坐标使用当前CLEAN局部图的绝对像素。只输出指定JSON。'
FIELDS=('id','bbox','render_type','grouping','label','extent')
TERMINAL=('completed','completed_with_issues','failed')
REPORT_LOCK=threading.RLock()


def containment_filter(rows, baseline):
    kept=[];removed=[]
    for row in rows:
        refs=[r['id'] for r in baseline if contains(r['bbox'],row['bbox'])]
        if refs:removed.append(dict(candidate=copy.deepcopy(row),contained_by=refs,reason='complete_native_bbox_containment'))
        else:kept.append(copy.deepcopy(row))
    return kept,removed


def context_frame(focus, baseline, image_size, scale, expansion=0):
    # One hop only: include complete original boxes intersecting editable additions.
    # This enlarges observation context; it never merges reconstruction tasks.
    refs=[r for r in baseline if any(E.overlap(r['bbox'],f['bbox'])>0 for f in focus)]
    native=Crop(tuple(image_size),(0,0,*image_size),tuple(image_size))
    return frame_for(focus+refs,native,scale,expansion)


def visual_rows(rows, view):
    out=[];audit=[]
    for row in E.nearby(rows,view):
        exact=E.local_box(row['bbox'],view);shown=E.visible_box(exact,view.input_size)
        out.append(dict(id=row['id'],bbox=shown,clipped=exact!=shown,
            render_type=row['render_type'],grouping=row['grouping']))
        audit.append(dict(id=row['id'],native_bbox=row['bbox'],local_exact=exact,local_visible=shown))
    return out,audit


def review_prompt(size, focus, readonly, expanded=False):
    return dict(task='整理本轮补漏新增框：验证实际笔画，修正完整范围，去掉重复和误检。只能编辑editable_additions；original_readonly始终保持不变。',
        input_size=list(size),coordinate_frame='LOCAL绝对像素，原点左上，x向右/y向下，0..W-1与0..H-1。后台换算回原图。',
        rules=[E.SCOPE,D.DISCOVERY_RULES,
            'CLEAN是干净原图；BOXES同坐标，蓝框/ADD或RV编号为待审核新增框，橙色B编号为只读原框。框边与编号不是笔画。',
            '不沿用前一步的描述或猜想。先确认框内/附近确有原图笔画；空白框、文字、形状边框等误检直接discard，不按流程常识补出连线。',
            'SAM可能包含多个不连通片段；一个SAM候选框跨过两段线不意味着中间有线。不可把两段线之间的空白补成连接。',
            '只整理给定新增候选，不重新扫描其他遗漏。keep_ids保留正确完整框；replacements可修框、合并或拆分，但source_ids只能引用本批新增ID，不能为空。discard只删除新增候选。',
            '对照原始框判断是否已覆盖：原始框是后续几何恢复范围，不是已恢复线条。若同一可见目标已完整纳入原始范围，删除重复新增候选。不同目标仅bbox部分重叠不是删除理由。',
            '原框未框全时可另加完整目标框；原框仍不变。沿真实笔画看到当前连接的起终点，不能只留下伸出的箭头尖角、端头或一截残线。不推断遮挡下的走向。',
            '只读框若clipped=true，表示输入视野截断了参考范围，不表示旧框实际就在裁剪边缘结束。不得把裁剪窗口边当完整目标边界。',
            ('若无法看到完整目标，decision=expand，暂不提交修改。' if not expanded else
             '这是最后一次扩图；仍看不全或无法确认的候选decision=uncertain，后台单独待审，不冒充已审核结果。'),
            'decision=edit时，每个editable_additions的ID恰好出现于keep_ids、一次replacements.source_ids或一次discard.ids。不可引用原始B编号来修改原框。只返回简短可见依据。'],
        editable_additions=focus,original_readonly=readonly,expanded=expanded,
        output_schema={'decision':'edit|expand|uncertain','reason':'扩图/待审原因；edit可为空',
            'keep_ids':['本批新增ID'],
            'replacements':[{'source_ids':['本批新增ID'],'regions':[C.REGION],'reason':'实际笔画依据'}],
            'discard':[{'ids':['本批新增ID'],'reason':'无笔画/重复/非线条等实际依据'}]})


def validate(a,size,focus):
    return ReviewProtocol(size,focus)(a)


def commit(focus,answer,view,prefix,baseline):
    contract = ReviewProtocol(view.input_size,focus)
    answer = contract.validate_saved(answer) if '_protocol' in answer else contract(answer)
    index={r['id']:r for r in focus}
    rows=[copy.deepcopy(index[k]) for k in answer['keep_ids']]
    for row in rows:
        row['extent']='complete'
        row['candidate_status']='reviewed'
    conversions=[];pending=[copy.deepcopy(index[k]) for k in answer.get('_protocol',{}).get('pending_focus_ids',[])]
    for i,patch in enumerate(answer['replacements'],1):
        lineage=sorted({x for k in patch['source_ids'] for x in index[k].get('review_source_ids',[k])})
        for j,r in enumerate(patch['regions'],1):
            raw=list(r.get('raw_model_bbox',r['bbox']));bounded=list(r['bbox'])
            exact=E.native_box(bounded,view);visible=E.visible_box(exact,view.global_size)
            row={k:copy.deepcopy(r[k]) for k in FIELDS}
            row['candidate_status']='reviewed' if row['extent']=='complete' else 'pending'
            row.update(id=f'{prefix}_{i}_{j}',bbox=visible,operation='add_only',
                review_source_ids=lineage,review_reason=patch['reason'],raw_local_bbox=raw,
                effective_local_bbox=bounded,mapped_exact_bbox=exact)
            conversions.append(dict(id=row['id'],raw_local_bbox=raw,effective_local_bbox=bounded,
                                    mapped_exact_bbox=exact,native_bbox=visible))
            if row['extent']=='complete':rows.append(row)
            else:pending.append(row)
    for r in rows:
        r['review_confirmed']=True;r['operation']='add_only'
    C.regions_valid(rows,view.global_size)
    kept,contained=containment_filter(rows,baseline)
    # Reuse immutable-baseline guard, including rejecting old ID collisions.
    E.append_only(baseline,kept,view.global_size)
    return kept,pending,contained,conversions


def review_one(root,cid,focus,baseline,image,model,cfg,round_index,batch_index):
    folder=root/'cases'/cid/'review'/f'round_{round_index:02d}'/f'B{batch_index:03d}'
    prefix=f'RV{round_index}B{batch_index}'
    try:
        for expansion in range(cfg['review_context_expansions']+1):
            view=context_frame(focus,baseline,image.size,cfg['review_context_scale'],expansion)
            d=folder/f'view_{expansion:02d}'
            new,na=visual_rows(focus,view);old,oa=visual_rows(baseline,view)
            if len(new)!=len(focus) or any(r['clipped'] for r in new):raise ValueError('Editable candidates must be fully visible')
            clean=view.image(image);E.save(clean,d/'clean.png')
            boxes=clean.copy();draw=ImageDraw.Draw(boxes)
            for rows,color in [(old,'#d37811'),(new,'#1679d3')]:
                for row in rows:E.corners(draw,row['bbox'],color,row['id'])
            E.save(boxes,d/'boxes.png')
            E.write(d/'frame.json',dict(crop=view.metadata(image.size),editable_conversion=na,readonly_conversion=oa))
            q=review_prompt(view.input_size,new,old,bool(expansion))
            a=model.ask(d/'request',q,[('CLEAN: sole LOCAL coordinates',d/'clean.png'),('BOXES: blue editable; orange original read-only',d/'boxes.png')],
                ReviewProtocol(view.input_size,focus),system=SYSTEM)
            if a['decision']=='expand':continue
            if a['decision']=='uncertain':raise ValueError('Model uncertain: '+a['reason'])
            rows,pending,contained,audit=commit(focus,a,view,prefix,baseline)
            E.write(d/'coordinate_audit.json',audit)
            E.write(folder/'transaction.json',dict(editable_before=focus,answer=a,editable_after=rows,
                pending=pending,contained_after_review=contained,baseline_modified=False))
            return dict(focus=focus,rows=rows,pending=pending,contained=contained,
                issues=[dict(stage='review',pending_ids=[r['id'] for r in pending])] if pending else [])
        raise ValueError('Context expansion budget exhausted; candidate held pending')
    except Exception as exc:
        error=dict(stage='review',focus_ids=[r['id'] for r in focus],error=str(exc),pending=True)
        E.write(folder/'failure.json',error)
        return dict(focus=focus,rows=[],pending=copy.deepcopy(focus),contained=[],issues=[error])


def snapshot(root,cid,image,baseline,current,unreviewed,pending):
    # Only reviewed additions enter current.png/handoff. Pending/raw survive separately.
    final,audit=E.append_only(baseline,current,image.size)
    E.write(root/'cases'/cid/'reviewed_additions.json',current)
    E.write(root/'cases'/cid/'unreviewed_additions.json',unreviewed)
    E.write(root/'cases'/cid/'pending_additions.json',pending)
    E.write(root/'cases'/cid/'baseline_preservation.json',audit)
    E.write(root/'cases'/cid/'current_native.json',final)
    E.plot(image,final,root/'cases'/cid/'current.png')
    return final


def run_case(root,cid,cfg,model):
    source=Path(cfg['source_run']);out=root/'cases'/cid
    baseline=E.read(root/'inputs'/cid/'baseline_native.json')
    baseline_hash=E.rows_digest(baseline)
    image=Image.open(root/'inputs'/cid/'source.png').convert('RGB')
    raw=E.read(source/'cases'/cid/'additions.json',[])
    source_status=E.read(source/'cases'/cid/'status.json',{})
    source_result=E.read(source/'cases'/cid/'result.json',{})
    E.write(out/'source_snapshot.json',dict(source=str((source/'cases'/cid).resolve()),
        source_status=source_status,additions=raw,additions_sha256=E.rows_digest(raw),captured_at=E.now(),
        source_issues=source_result.get('issues',[])))
    E.plot(image,baseline+raw,out/'before.png')
    current,removed=containment_filter(raw,baseline)
    E.write(out/'containment_before.json',removed)
    issues=[];pending=[];checked=set();history=[];confirmed=[]
    try:
        for wave in range(1,cfg['review_rounds']+1):
            groups=batches(current,cfg['review_batch_size'],None if wave==1 else checked)
            if not groups:break
            E.write(out/'status.json',dict(state='running',stage=f'additions_review_{wave}',updated_at=E.now(),raw_additions=len(raw),reviewed_additions=len(confirmed)))
            E.write(out/'review'/f'round_{wave:02d}'/'plan.json',dict(groups=[[r['id'] for r in g] for g in groups],input_count=len(current)))
            answers={}
            with ThreadPoolExecutor(max_workers=cfg['review_workers']) as pool:
                jobs={pool.submit(review_one,root,cid,g,baseline,image,model,cfg,wave,i):i for i,g in enumerate(groups,1)}
                for future in as_completed(jobs):
                    answers[jobs[future]]=future.result()
                    done={r['id'] for result in answers.values() for r in result['focus']}
                    remaining=[r for r in current if r['id'] not in done]
                    visible=[r for r in remaining if r.get('review_confirmed')]+[r for i in sorted(answers) for r in answers[i]['rows']]
                    snapshot(root,cid,image,baseline,visible,[r for r in remaining if not r.get('review_confirmed')],pending+[r for v in answers.values() for r in v['pending']])
            done={r['id'] for a in answers.values() for r in a['focus']}
            current=[r for r in current if r['id'] not in done]
            for i in sorted(answers):
                a=answers[i];current+=a['rows'];pending+=a['pending'];issues+=a['issues']
                if not a['issues']:
                    for members in [a['focus'],a['rows']]:
                        checked.update(pair_key(x,y) for j,x in enumerate(members) for y in members[j+1:])
                history.append(dict(round=wave,batch=i,input_ids=[r['id'] for r in a['focus']],output_ids=[r['id'] for r in a['rows']],issues=a['issues'],contained_removed=a['contained']))
            confirmed=current
        E.write(out/'status.json',dict(state='running',stage='sam_addition_scope_review',updated_at=E.now(),raw_additions=len(raw),reviewed_additions=len(current)))
        current,scope_audit=review_addition_scope(root,cid,current,baseline,image,model,cfg)
        issues+=scope_audit['issues']
        overlaps=unresolved_pairs(current,checked)
        if overlaps:issues.append(dict(stage='review',reason='unreviewed_overlap_pairs_after_budget',pairs=overlaps))
        if source_status.get('state')!='completed':
            issues.append(dict(stage='upstream_missing',state=source_status.get('state'),source_issues_count=len(source_result.get('issues',[]))))
        if E.rows_digest(baseline)!=baseline_hash:raise ValueError('Original baseline mutation detected')
        final=snapshot(root,cid,image,baseline,current,[],pending)
        global_rows,handoff,audit=E.export_additive(cid,final,root/'inputs'/cid/'source.png')
        E.write(out/'reconstruction_handoff.json',handoff);E.write(out/'coordinate_audit.json',audit)
        E.write(out/'review_history.json',history)
        state='completed_with_issues' if issues else 'completed'
        E.write(out/'result.json',dict(case_id=cid,state=state,native_regions=final,regions=global_rows,
            baseline_count=len(baseline),raw_additions_count=len(raw),contained_before_review=len(removed),
            reviewed_additions_count=len(current),pending_count=len(pending),issues=issues,
            addition_scope_review=scope_audit,
            baseline_preserved=True,geometry_executed=False,formal_accuracy_measured=False))
        E.write(out/'status.json',dict(state=state,stage='finished',updated_at=E.now(),raw_additions=len(raw),reviewed_additions=len(current),pending=len(pending)))
    except Exception as exc:
        E.write(out/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()))
        E.write(out/'status.json',dict(state='failed',stage='failed',error=str(exc),updated_at=E.now()))
    publish(root)


def publish(root):
    with REPORT_LOCK:
        cfg=E.read(root/'config.json');records=[];sections=[]
        for cid in E.read(root/'cases.json'):
            out=root/'cases'/cid;s=E.read(out/'status.json',dict(state='waiting_source',stage='waiting_source'))
            upstream=E.read(Path(cfg['source_run'])/'cases'/cid/'status.json',{})
            records.append(dict(case=cid,**s,source_state=upstream.get('state')))
            before='before.png' if (out/'before.png').exists() else 'baseline.png'
            sections += [f'## {cid}','',f"上游 {upstream.get('state','等待')}；整理 {s['state']}/{s['stage']}；新增 {s.get('raw_additions','—')} → 审核保留 {s.get('reviewed_additions','—')}；待审 {s.get('pending','—')}。",'',
                f'[原图](../inputs/{cid}/source.png) · [整理trace](../cases/{cid}/) · [上游新增结果]({Path(cfg["source_run"])/"cases"/cid/"current.png"})','',
                '| 原框＋整理前新增 | 原框＋已审核新增（未审核/失败候选单独保留） |','|---|---|',
                f'| [![整理前](../cases/{cid}/{before})](../cases/{cid}/{before}) | [![当前](../cases/{cid}/current.png)](../cases/{cid}/current.png) |','']
        finished=sum(r['state'] in TERMINAL for r in records)
        E.write(root/'progress.json',dict(updated_at=E.now(),finished=finished,total=len(records),cases=records))
        dest=root/'reports';dest.mkdir(exist_ok=True)
        text=['# SAM新增框整理：24图对照','',f'更新 {E.now()}；结束 {finished}/{len(records)}。','',
            '原始627个定位框只读。只对补漏新增框进行包含过滤和模型整理；重新从原图取上下文。蓝色可编辑新增、橙色只读原框。未执行几何。',
            '左栏是完整原始新增快照；右栏只加入已审核候选。待审/请求失败的新增保留JSON，不能把框减少当成召回或准确率提升。', '',
            '[方案](../README.md) · [配置](../config.json) · [提示词](../PROMPT.md)','']+sections
        tmp=dest/'24图_整理前后对照.md.tmp';tmp.write_text('\n'.join(text)+'\n');tmp.replace(dest/'24图_整理前后对照.md')
        (root/'RESULTS.md').write_text(f'# SAM新增框整理\n\n更新 {E.now()}；结束 {finished}/{len(records)}。\n\n[24图整理前后](reports/24图_整理前后对照.md) · [方案](README.md)\n\n原框保持，新增质量需看图，未执行几何。\n')


def run(root):
    lock=(root/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name,digest in E.read(root/'manifest.json')['files'].items():
        if E.sha256(root/name)!=digest:raise ValueError('Frozen file changed: '+name)
    cfg=E.read(root/'config.json');model=Model(cfg,run_dir=root)
    E.write(root/'process.json',dict(pid=os.getpid(),started_at=E.now()))
    E.write(root/'execution.json',dict(state='running',started_at=E.now()))
    stop=threading.Event()
    def watch():
        while not stop.wait(15):publish(root)
    watcher=threading.Thread(target=watch,daemon=True);watcher.start()
    submitted=set();done=set();ids=E.read(root/'cases.json')
    try:
        with ThreadPoolExecutor(max_workers=cfg['review_case_workers']) as pool:
            active={}
            while len(done)<len(ids):
                for cid in ids:
                    if cid in submitted or len(active)>=cfg['review_case_workers']:continue
                    s=E.read(Path(cfg['source_run'])/'cases'/cid/'status.json',{})
                    if s.get('state') in TERMINAL:
                        active[pool.submit(run_case,root,cid,cfg,model)]=cid;submitted.add(cid)
                for future in list(active):
                    if future.done():future.result();done.add(active.pop(future))
                if len(done)<len(ids):time.sleep(5)
        E.write(root/'execution.json',dict(state='finished',finished_at=E.now()))
    finally:stop.set();watcher.join();publish(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','report']);parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();(run if args.action=='run' else publish)(args.run_dir.resolve())
