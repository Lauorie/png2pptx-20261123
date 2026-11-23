"""Isolated whole-instance line trial; reuses frozen masks, never skeletonizes.

Existing identities scope the experiment. Numerical lines are proposals, GLM
confirms their physical endpoints and connections. No discovery or SAM calls.
"""
import argparse, copy, json, shutil, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import pipeline, protocol, straight_mask, visual
from transport.model import Model
from transport.common import read_json as read, write_json as write, sha256


def propose(decision, candidates, gid):
    selected = {c['id']: c for c in candidates if c.get('selected') and not c.get('represented_by')}
    owners = {}
    for i, target in enumerate(decision['targets'], 1):
        for mid in target.get('filled_masks', []):
            owners.setdefault(mid, []).append(i)
    rows, supports, skipped = [], {}, []
    for i, target in enumerate(decision['targets'], 1):
        tid = f'T{i}'
        explicit = target.get('filled_masks', [])
        mids = explicit or [k for k, c in selected.items() if c.get('best_global') in target['sources']]
        # Association ambiguity is left unresolved; it is not a geometry guess.
        ambiguous = any(k not in selected or len(owners.get(k, [i])) > 1 for k in mids)
        if not explicit:
            ambiguous |= any(sum(c['best_global'] in t['sources'] for t in decision['targets']) > 1
                             for k, c in selected.items() if k in mids)
        if not mids or ambiguous or target.get('repair'):
            skipped.append(dict(id=tid, reason='unresolved mask ownership or pending geometry repair')); continue
        mask = np.logical_or.reduce([selected[k]['mask'] for k in mids])
        row = straight_mask.estimate(mask)
        if row is None:
            skipped.append(dict(id=tid, reason='mask does not support one straight centerline')); continue
        row.update(id=tid, parent_id=f'{gid}.{tid}', identity_index=i, globals=target['sources'],
                   source_masks=mids, label=tid, node_ids=[tid+'.start', tid+'.end'])
        rows.append(row); supports[tid] = mask
    return rows, supports, skipped


def prompt(rows):
    return dict(task='Check only these proposed whole straight arrows/lines and label their TWO physical ends.',
        head_types=protocol.HEADS,
        line_styles={'solid':'continuous stroke','dashed':'separate elongated strokes',
                     'dotted':'separate dots','unknown':'cannot determine from original'},
        targets=[dict(id=r['id'], masks=r['source_masks'], start=np.round(r['start'], 1).tolist(),
                      end=np.round(r['end'], 1).tolist()) for r in rows],
        instructions=[protocol.SCOPE,
            'CLEAN is the original crop. OVERLAY shows raw masks and proposed straight centerlines on CLEAN. Colors are arbitrary. The full-figure image supplies context.',
            'The proposal joins collinear dash marks into one geometric axis; visible dash gaps remain style evidence. A triangular head is an endpoint decoration, not another line or two arrowheads on its inner strokes.',
            'For each T return keep if it represents one physical straight arrow/line, exclude if it is not in scope, or review if wrong/incomplete/mixed. Mask ownership and fitted geometry are suggestions, not ground truth. Do not add unrelated arrows or coordinates.',
            'heads is [start type, end type]. In the overlay T1a marks T1.start and T1b marks T1.end (likewise for other T labels). Judge both ends from CLEAN, not from mask thickness.',
            'Give line_style from CLEAN, excluding the arrowhead. Ignore gaps caused only by occlusion. Do not estimate widths or dash lengths; numerical fitting measures those parameters.',
            'Only when endpoints truly meet, put their names (T1.end etc.) together in junctions. Ordinary crossings do not share a node. Internal junction endpoints have no arrowhead.'],
        output_schema={'targets':'[{id:T ID,status:keep|exclude|review,heads:[start type,end type],line_style:solid|dashed|dotted|unknown,reason:brief text}]',
                       'junctions':'[[endpoint name, endpoint name, ...], ...] or []'})


def validate(raw, rows):
    expected = {r['id'] for r in rows}
    if not isinstance(raw, dict) or not isinstance(raw.get('targets'), list): raise ValueError('targets list required')
    targets = raw['targets']
    if len(targets) != len(expected) or {r.get('id') for r in targets} != expected:
        raise ValueError('Declare every T exactly once')
    for r in targets:
        if r.get('status') not in ('keep', 'exclude', 'review'): raise ValueError('Unknown status')
        if not isinstance(r.get('heads'), list) or len(r['heads']) != 2 or any(h not in protocol.HEADS for h in r['heads']):
            raise ValueError('Two endpoint head enums required')
        if r.get('line_style') not in ('solid','dashed','dotted','unknown'):
            raise ValueError('line_style required: solid, dashed, dotted, or unknown')
    ends = {r['id']+'.'+e for r in targets if r['status']=='keep' for e in ('start', 'end')}
    seen = set()
    if not isinstance(raw.get('junctions'), list): raise ValueError('junctions list required')
    for j in raw['junctions']:
        if not isinstance(j, list) or len(j)<2 or len(set(j))!=len(j) or not set(j)<=ends or seen & set(j):
            raise ValueError('Junctions must use distinct kept physical endpoints')
        seen.update(j)
    return raw


def materialize(rows, decision):
    choices = {r['id']: r for r in decision['targets']}
    kept = {r['id']: copy.deepcopy(r) for r in rows if choices[r['id']]['status']=='keep'}
    junction_log = []
    for j in decision['junctions']:
        normals, offsets, ends = [], [], []
        for ref in j:
            tid, end = ref.split('.'); row = kept[tid]
            a, b = np.array(row['start']), np.array(row['end'])
            direction = (b-a)/np.linalg.norm(b-a); normal = np.array([-direction[1], direction[0]])
            normals.append(normal); offsets.append(normal@a); ends.append(row[end])
        # Closest common point on measured axes, anchored to the participating
        # endpoints in rank-deficient (collinear) cases. Topology comes from GLM.
        anchor = np.mean(ends, axis=0); matrix = np.array(normals)
        point = anchor + np.linalg.lstsq(matrix, np.array(offsets)-matrix@anchor, rcond=None)[0]
        jid = 'J'+str(len(junction_log)+1)
        junction_log.append(dict(id=jid, endpoints=j, xy=point.tolist(), before=ends))
        for ref in j:
            tid, end = ref.split('.'); kept[tid][end] = point.tolist()
            kept[tid]['node_ids'][0 if end=='start' else 1] = jid
    for tid, row in kept.items():
        row['heads'] = choices[tid]['heads']
        row['line_style'] = choices[tid].get('line_style','unknown')
        for end, h in zip(('start', 'end'), row['heads']):
            row[end+'_head'] = h not in ('none', 'unknown')
            row[end+'_head_type'] = h if h not in ('none', 'unknown') else None
        a, b = np.array(row['start']), np.array(row['end'])
        row['route_path'] = np.linspace(a,b,max(2,int(np.linalg.norm(b-a))+1)).tolist()
    return list(kept.values()), junction_log


def draw_proposals(clean, rows, supports, path):
    visual.masks(clean, [(r['id'], supports[r['id']]) for r in rows], path)
    image = Image.open(path).convert('RGB'); d = ImageDraw.Draw(image)
    for i, r in enumerate(rows):
        a, b = np.array(r['start']), np.array(r['end']); tint = visual.color(i)
        d.line([tuple(a),tuple(b)], fill=tint, width=1)
        direction=(b-a)/np.linalg.norm(b-a); normal=np.array([-direction[1],direction[0]])
        for name, p, sign in [('start',a,-1),('end',b,1)]:
            d.ellipse([p[0]-3,p[1]-3,p[0]+3,p[1]+3],outline=tint,width=2)
            visual.label(d,p-direction*sign*42+normal*18,r['id']+('a' if name=='start' else 'b'),tint,image)
    image.save(path)


def offline_decision(rows, before, candidates):
    """Preview only: preserve unambiguous cached heads; never invent new ones."""
    targets=[]
    for row in rows:
        old=[r for r in before if r['parent_id']==row['parent_id'] and r['kind']!='block_arrow']
        heads=['unknown','unknown']; source='pending GLM: cached fragment endpoints cannot be mapped safely'
        if old and all(r.get('heads')==['none','none'] for r in old):
            heads=['none','none']; source='cached model: no heads on this identity'
        else:
            chosen=[c for c in candidates if c['id'] in row['source_masks']]
            mapped=straight_mask.refit_confirmed(old,chosen)
            if len(mapped)==1 and mapped[0].get('geometry_method')=='direct_straight_mask':
                heads=mapped[0]['heads']; source='cached model heads at safely mapped physical endpoints'
        targets.append(dict(id=row['id'],status='keep',heads=heads,line_style='unknown',reason=source))
    return dict(targets=targets,junctions=[],preview_only=True,new_model_decision=False,
                note='keep means numerical preview only; no new semantic acceptance or shared-node judgment')


def trial(spec, source, baseline, out, model):
    cid, gid = spec.split('/'); src=source/'cases'/cid/'groups'/gid
    folder=out/'cases'/cid/'groups'/gid; folder.mkdir(parents=True,exist_ok=True)
    snapshot=folder/'input'; snapshot.mkdir(exist_ok=True); hashes={}
    candidates=read(src/'candidates.json')
    files=['decision.json','candidates.json','input/clean.png','input/context.png','input/transform.json']
    files += ['masks/'+c['id']+'.png' for c in candidates if c.get('selected') and not c.get('represented_by')]
    for f in files:
        dst=snapshot/f; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src/f,dst); hashes[f]=sha256(dst)
    old_path=baseline/'cases'/cid/'groups'/gid/'rows_local.json'
    shutil.copy2(old_path,snapshot/'before_rows.json'); hashes['before_rows.json']=sha256(old_path)
    write(folder/'provenance.json',dict(source=str(src),baseline=str(old_path),hashes=hashes))
    for c in candidates:
        if c.get('selected') and not c.get('represented_by'):
            c['mask']=np.asarray(Image.open(snapshot/'masks'/f'{c["id"]}.png').convert('L'))>0
    clean_path=snapshot/'input/clean.png'; clean=Image.open(clean_path).convert('RGB')
    rows,supports,skipped=propose(read(snapshot/'decision.json'),candidates,gid)
    write(folder/'proposals.json',dict(rows=rows,skipped=skipped))
    if not rows: return dict(case=cid,group=gid,state='not_applicable',skipped=skipped)
    visual.masks(clean,[(r['id'],supports[r['id']]) for r in rows],folder/'masks.png',white=True)
    draw_proposals(clean,rows,supports,folder/'overlay.png')
    before=read(snapshot/'before_rows.json'); pipeline.viz.render(clean_path,before,folder/'before')
    images=[
        ('CONTEXT: full figure with old target hints, context only',snapshot/'input/context.png'),
        ('CLEAN: original local image',clean_path),
        ('OVERLAY: mask evidence and whole-line endpoint proposals',folder/'overlay.png')]
    if model is None:
        answer=offline_decision(rows,before,candidates)
        write(folder/'head_request/prompt.json',dict(system=protocol.SYSTEM,prompt=prompt(rows),not_sent=True))
    else:
        answer=model.ask(folder/'head_request',prompt(rows),images,lambda a:validate(a,rows))
    write(folder/'decision.json',answer); after,junctions=materialize(rows,answer)
    write(folder/'junctions.json',junctions)
    for r in after:
        pipeline.style_rows(clean,[r],supports[r['id']],adaptive_heads=True); r['id']=gid+':'+r['id']
    pending=[r for r in after if r['style']['dash']['type']=='unknown']
    write(folder/'pending_style.json',pending)
    if pending:
        # Keep unresolved candidates and their pixels for review, but do not
        # replace the baseline with a newly asserted solid line.
        pending_parents={r['parent_id'] for r in pending}
        after=[r for r in after if r['parent_id'] not in pending_parents]
    else:pending_parents=set()
    replaced={gid+'.'+r['id'] for r in answer['targets'] if r['status'] in ('keep','exclude')}
    replaced-=pending_parents
    final=after+[r for r in before if r['parent_id'] not in replaced]
    write(folder/'rows_local.json',final); pipeline.viz.render(clean_path,final,folder/'result')
    pipeline.native.write(clean.size,final,[r['style'] for r in final],folder/'arrows.pptx')
    view=read(snapshot/'input/transform.json'); global_rows=[pipeline.fit.transform(r,view) for r in final]
    for r in global_rows:r.pop('route_path',None)
    write(folder/'rows_global.json',global_rows)
    report=dict(case=cid,group=gid,state='offline_preview' if model is None else 'completed',before_objects=len(before),after_objects=len(final),
                decisions=answer,skipped=skipped,pending_style=[r['id'] for r in pending],
                styles=[dict(id=r['id'],style=r['style']) for r in after])
    write(folder/'summary.json',report)
    (folder/'README.md').write_text(f'# {cid} / {gid}\n\n复用同一份局部 SAM；整实例直线拟合，不做骨架化。GLM 判断物理两端头型、共享节点和虚实线类别；像素测量样式参数。\n\n'
        '| 原图 | 原始 mask | 修正前 | 修正后 |\n|---|---|---|---|\n'
        '| ![](input/input/clean.png) | ![](masks.png) | ![](before/styled.png) | ![](result/styled.png) |\n\n'
        '实际模型输入：原图、全图上下文以及下方的 mask/直线端点叠图。\n\n![](overlay.png)\n\n'
        '[prompt](head_request/prompt.json) · [模型决定](decision.json) · [测量/样式](summary.json) · [可编辑 PPT](arrows.pptx)\n')
    if model is None:
        p=folder/'README.md'; text=p.read_text().replace('GLM 重新判断物理两端的头型和共享节点。',
            '**这是离线数值预览，尚未经过新 GLM 复核。** 只复用能安全对应到物理端点的旧头型；unknown 端点暂不绘制箭头头部。共享节点暂未合并。')
        text=text.replace('实际模型输入：','已准备但未发送的模型输入：').replace('[模型决定]','[离线头型来源]');p.write_text(text)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--groups',nargs='+',required=True)
    p.add_argument('--offline-preview',action='store_true');p.add_argument('--reuse-model-from',type=Path);a=p.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True);root=Path(__file__).resolve().parent
    config=read(root/'config.json');model=None if a.offline_preview else Model(config,run_dir=out,
        reuse_from=a.reuse_model_from.resolve() if a.reuse_model_from else None,system=protocol.SYSTEM)
    code=out/'code';code.mkdir(exist_ok=True);hashes={}
    for src in [*root.glob('*.py'),*root.glob('lib/*.py'),*root.glob('transport/*.py')]:
        dst=code/src.relative_to(root);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);hashes[str(src.relative_to(root))]=sha256(dst)
    write(out/'code_manifest.json',hashes);write(out/'config.json',config)
    write(out/'inputs.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
    results=[];start=time.time()
    with ThreadPoolExecutor(max_workers=min(6,len(a.groups))) as pool:
        jobs={pool.submit(trial,g,a.source.resolve(),a.baseline.resolve(),out,model):g for g in a.groups}
        for f in as_completed(jobs):
            try:r=f.result()
            except Exception as exc:
                cid,gid=jobs[f].split('/');r=dict(case=cid,group=gid,state='failed',error=str(exc))
                write(out/'cases'/cid/'groups'/gid/'error.json',dict(error=str(exc),traceback=traceback.format_exc()))
            results.append(r);write(out/'progress.json',dict(done=len(results),total=len(jobs),results=results,seconds=time.time()-start))
            print(r['case'],r['group'],r['state'],flush=True)
    for cid in sorted({r['case'] for r in results}):
        folder=out/'cases'/cid;folder.mkdir(exist_ok=True);src=a.baseline/'cases'/cid
        shutil.copy2(src/'clean.png',folder/'clean.png');scene=read(src/'scene.json');rows=scene['arrows']
        pipeline.viz.render(folder/'clean.png',rows,folder/'before')
        for r in results:
            if r['case']==cid and r['state'] in ('completed','offline_preview'):
                gid=r['group'];rows=[x for x in rows if not x['parent_id'].startswith(gid+'.')]+read(folder/'groups'/gid/'rows_global.json')
        # Preserve existing A labels so reviewers can compare the same targets.
        labels={r['parent_id']:r['label'] for r in scene['arrows']}
        for r in rows:r['label']=labels.get(r['parent_id'],r.get('label',r['id']))
        write(folder/'scene.json',dict(size=scene['size'],arrows=rows,scope='replace pilot groups only; all other baseline issues retained'))
        pipeline.viz.render(folder/'clean.png',rows,folder/'result')
        pipeline.native.write(scene['size'],rows,[r['style'] for r in rows],folder/'arrows.pptx')
    write(out/'summary.json',dict(results=results,seconds=time.time()-start,new_sam_calls=0))
    lines=['# 整实例直线拟合小实验','','沿用原身份和原始局部 mask；仅数值几何表示及物理端点语义重新处理。未重跑 SAM、补漏或其余几何。','',
           '| case / group | 原图 | 修正前 | 修正后 |','|---|---|---|---|']
    for r in sorted(results,key=lambda r:(r['case'],r['group'])):
        b=f'cases/{r["case"]}/groups/{r["group"]}'
        lines.append(f'| [{r["case"]} / {r["group"]}]({b}/README.md) | ![]({b}/input/input/clean.png) | ![]({b}/before/styled.png) | '+(f'![]({b}/result/styled.png)' if r['state'] in ('completed','offline_preview') else r['state'])+' |')
    lines+=['','整图对照（其他问题保留原状）：','','| case | 修正前 | 修正后 |','|---|---|---|']
    for cid in sorted({r['case'] for r in results}):lines.append(f'| {cid} | ![](cases/{cid}/before/instances.png) | ![](cases/{cid}/result/instances.png) |')
    if a.offline_preview:
        lines.insert(2,'**离线预览：新 GLM 头型复核遇到 401，未完成。025 / G004 的头型为 unknown，暂不画头；不能把它当作完整修复。其余复用无歧义的旧头型。**')
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
