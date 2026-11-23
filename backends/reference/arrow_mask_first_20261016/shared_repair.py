"""Targeted continuation with the current cross-group geometry in view.

No per-group critic. Identity/coverage nominate regions; each repair can add,
replace or leave unchanged. Only the model chooses edits and mask associations.
"""
import copy,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'lib'))
import cv2
import numpy as np
from PIL import Image,ImageDraw
import contracts, evidence as ev, local_sampling, protocol, segmented_geometry as sg
import semantic_geometry, geometry_routing, mask_vectors, style_windows, visual, retained_evidence, head_semantics, observation_pool
from lib import fit,viz,style
from transport.common import read_json as read,write_json as write,sha256 as sha


def transform(points,view):
    p=np.asarray(points,float)
    return (np.c_[p,np.ones(len(p))]@np.asarray(view['global_to_view']).T)[:,:2]


def visible(row,box):
    """Viewport retrieval only: a path's large bounding box is not coverage."""
    p=fit.points(row)-np.asarray(box[:2]);size=np.ceil(np.asarray(box[2:])-box[:2]).astype(int)
    rect=(0,0,int(size[0])+1,int(size[1])+1)
    return any(cv2.clipLine(rect,tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)))[0] for a,b in zip(p,p[1:]))


def existing_rows(rows,view):
    # Stable within a request; the saved ID map owns all later replacements.
    return [{**copy.deepcopy(r),'short_id':f'E{i+1}'} for i,r in enumerate(sorted(rows,key=lambda r:r['id'])) if visible(r,view['bbox_global'])]


def inventory(rows,view):
    result=[]
    for r in rows:
        p=transform(fit.points(r),view)
        q=cv2.approxPolyDP(p.astype('float32'),.4,False).reshape(-1,2)
        result.append(dict(id=r['short_id'],kind=r['kind'],heads=r.get('heads',['none','none']),
            points=[[round(float(x),1),round(float(y),1)] for x,y in q]))
    return result


def draw_current(crop,rows,view,path,labels=True):
    from coverage_visual import render
    render(crop,rows,view,path,labels)


def prompt(size,context_size,task,masks,existing,allow_expand,allow_relocation=False,retain_locator_text=False):
    # Region nomination is a locator, never a prior answer about route counts,
    # sources or network connectivity. Keep the original task in audit files.
    focus={k:v for k,v in task.items() if k in ('bbox','regions')}
    if 'regions' in focus:focus['regions']=[{'bbox':r['bbox']} for r in focus['regions']]
    if retain_locator_text and task.get('description'):focus['description']=task['description']
    return dict(task='Check the suspected missing lines near FOCUS against CLEAN. Return only supported additions or replacements using the existing cross-group paths.',
        size=list(size),context_size=list(context_size),focus=focus,existing=existing,mask_measurements=masks,
        instructions=[protocol.SCOPE,protocol.OCCLUSION,
            ('FOCUS is approximate and may be shifted or incomplete. You may correct its location to a nearby clearly visible missing connector in CLEAN; it is not a clipping boundary. Check CURRENT to avoid rebuilding an existing neighbor. A blank rectangle alone does not prove the nearby intended object is absent. Do not invent a line to fill the rectangle.' if allow_relocation else 'FOCUS only locates an unverified candidate; it does not prescribe route counts, connections or shape. Do not invent a line to fill a blank region.'),
            'FOCUS is an unverified suggestion, not proof that a line is missing or that nearby paths form one network. CLEAN determines visible lines. CURRENT shows recovered paths in deep blue over a faded gray source, including known heads, dashes and filled contours; thin shafts are thickened only for visibility. Gray source ink is not reconstruction. Blue results may be wrong. MASKS are fallible pixel evidence, including earlier local SAM when the new crop has no candidates. Evidence associated with an existing path may also contain an omitted branch.',
            'Nearby or collinear paths may be independent. Preserve blank gaps between separate terminations. Add a continuation only when source strokes, a visible dash pattern, or an actual occluder support it; alignment and existing endpoints alone are not evidence. True junctions can share nodes; crossings and shape borders do not establish a junction. Use the visible attachments without inferring the diagram architecture.'
            ,'Keep correct existing paths. Add genuinely missing portions of the focused connection, including its visible short branches. Ignore unrelated neighbors and minor style differences. replace lists only E IDs to remove/rebuild; supply their complete replacements. Never copy kept paths. Return complete with empty edits when nothing needs changing or focus is out of scope.',
            'For status=repair choose visible target geometry from CLEAN, not from earlier descriptions. A target may contain multiple independently drawable paths. Sharp corners use line parts; visible curves use on-curve points. Parts within one path are consecutive, but separate paths need not form a single traversal. Do not insert a connector or backtrack to satisfy the format. A visibly shared location may reuse a node ID; no network decomposition is required. Existing coordinates are attachment references, not mandatory fits.',
            'A wide filled block arrow stays kind=block_arrow even without SAM. Give contour=[ordered node IDs] for a closed filled outline, or contour=[] with supporting masks to use their outline; segments=[] in either case. Never replace a filled arrow with a stroked outline or force it into a thick shaft plus a head. Ordinary shafts plus heads remain line/curve routes. Coordinates are local CLEAN pixels, one decimal. A complete replacement may reuse EXACT supplied existing coordinates outside CLEAN; new estimates must be inside or request expansion. Heads are classified separately for centerlines. line_style=solid/dashed/dotted/unknown. Masks are optional style evidence; do not estimate widths or dash lengths.',
            ('If the continuation/endpoints cannot be judged in CLEAN, return status=expand and context_bbox in CONTEXT coordinates. The same repair will continue there with current paths, without another planning request.' if allow_expand else 'This is the last expanded view. If still unresolvable return status=unresolved, leaving current paths unchanged; do not invent geometry.')],
        output_schema=dict(status='complete|repair|expand|unresolved',replace='[existing E IDs]',nodes='[[id,x,y],...]',
            routes='[{id,kind:line|block_arrow,line_style,masks:[M IDs],contour:[node IDs] for block_arrow,paths:[{id:path_ID,segments:[{kind:line|curve,nodes:[IDs]}]}] for line}]',
            context_bbox='[x0,y0,x1,y1] for expand; otherwise null',reason='brief'))


def validate(answer,size,context_size,existing,mids,allow_expand,reference_points=None):
    if not isinstance(answer,dict) or answer.get('status') not in ('complete','repair','expand','unresolved'):raise ValueError('Invalid repair status')
    a=copy.deepcopy(answer);replace=contracts.ids(a.get('replace',[]))
    if not set(replace)<=set(existing):raise ValueError('Unknown existing replacement ID')
    a['replace']=replace
    if a['status']=='expand':
        if not allow_expand:raise ValueError('Context expansion already used')
        contracts.bbox(a.get('context_bbox'),context_size)
    if a['status']!='repair':
        if replace or a.get('nodes',[]) or a.get('routes',[]):raise ValueError('Non-repair decisions must not contain edits')
        return a
    if not isinstance(a.get('routes'),list):raise ValueError('routes list required')
    targets={};seen=set()
    for r in a['routes']:
        if not isinstance(r.get('id'),str) or r['id'] in seen:raise ValueError('Unique string route IDs required')
        seen.add(r['id'])
        r.setdefault('heads',['unknown','unknown'])
        if not isinstance(r.get('heads'),list) or len(r['heads'])!=2 or any(h not in protocol.HEADS for h in r['heads']):raise ValueError('Two head enums required')
        if r.get('line_style') not in protocol.LINE_STYLES:raise ValueError('line_style required')
        r['masks']=contracts.ids(r.get('masks',[]))
        if not set(r['masks'])<=set(mids):raise ValueError('Unknown supporting M mask')
        if r.get('kind','line') not in ('line','block_arrow'):raise ValueError('kind=line or block_arrow')
        if r.get('kind')=='block_arrow':
            contour=r.get('contour',[])
            # Equivalent explicit closed line-chain outline from older outputs.
            # This normalization changes representation only, never coordinates.
            if not contour and r.get('segments'):
                segments=r['segments'];chain=[]
                for seg in segments:
                    if seg.get('kind')!='line' or not isinstance(seg.get('nodes'),list) or len(seg['nodes'])<2:break
                    if chain and chain[-1]!=seg['nodes'][0]:break
                    chain.extend(seg['nodes'] if not chain else seg['nodes'][1:])
                else:
                    table={n:[x,y] for n,x,y in a['nodes']}
                    if len(chain)>=4 and table.get(chain[0])==table.get(chain[-1]):
                        contour=chain[:-1];r['segments']=[];r['contour']=contour
            if r.get('segments',[]) or r.get('paths',[]):raise ValueError('Block arrow needs a closed contour, not centerline paths')
            if contour:
                table={n:[x,y] for n,x,y in a['nodes']}
                if not isinstance(contour,list) or len(contour)<3 or any(n not in table for n in contour):raise ValueError('Block contour needs at least three ordered defined nodes')
                if len({tuple(table[n]) for n in contour})<3:raise ValueError('Block contour needs three distinct vertices')
            elif not r['masks']:raise ValueError('Block arrow needs contour node IDs or contour mask evidence')
        else:targets[r['id']]={}
    if not seen and not replace:raise ValueError('An empty patch uses status=complete')
    centerline={**a,'routes':[r for r in a['routes'] if r.get('kind')!='block_arrow']}
    preserved=[p for eid in replace for p in (reference_points or {}).get(eid,[])]
    checked=sg.validate(centerline,targets,size,reference_points=preserved)
    byid={r['id']:r for r in checked['routes']}
    a['nodes']=checked['nodes']
    a['routes']=[byid.get(r['id'],r) for r in a['routes']]
    return a


def block_geometry(route,nodes,index):
    """Render the model's explicit filled contour, including its head, once."""
    if route.get('contour'):
        table={n:[x,y] for n,x,y in nodes};p=[table[n] for n in route['contour']]
        return dict(kind='block_arrow',start=p[0],via=p[1:-1],end=p[-1],contours=[p],
            closed=True,beziers=[],width_estimate=None,geometry_method='model_closed_contour',coordinate_source='glm_keypoints')
    support=np.logical_or.reduce([index[m]['mask'] for m in route['masks']])
    return {**mask_vectors.contour(support),'coordinate_source':'mask_no_skeleton'}


def apply_patch(rows,answer,newrows,mapping):
    removed={mapping[k] for k in answer.get('replace',[])}
    retained=[r for r in rows if r['id'] not in removed]
    ids=[r['id'] for r in retained+newrows]
    if len(ids)!=len(set(ids)):raise ValueError('Patch creates colliding route IDs')
    return retained+newrows


def recover(clean,task,current,pid,out,config,model,locks,evidence_pool=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    original=list(current);requested=task['bbox'];attempts=[];answer=None
    for step in range(2):
        folder=out/('local' if step==0 else 'expanded')
        cfg={**config,'crop_context_scale':config.get('repair_context_scale',3.) if step==0 else 1.,
             'display_side':config.get('repair_display_side',1536)}
        with locks['cpu']:crop,view=local_sampling.prepare(clean,[],folder,cfg,requested)
        selected,all_candidates=local_sampling.sample(crop,view,[],[],{},pid,folder,cfg,locks['sam'])
        existing=existing_rows(current,view)
        reused=[]
        if not selected and config.get('repair_reuse_cached_masks',True):
            with locks['cpu']:reused=retained_evidence.load(existing,view,folder/'retained_masks')
            selected=reused
        write(folder/'mask_evidence.json',dict(fresh_selected=len(selected)-len(reused),retained_selected=len(reused),ids=[c['id'] for c in selected]))
        # All masks stay available in this bounded repair. None are automatically
        # promoted to a target; the response explicitly selects evidence per route.
        with locks['cpu']:proposals,_=semantic_geometry.prepare(crop,[],selected,folder)
        shared=None
        if evidence_pool is not None:
            with locks['cpu']:shared=observation_pool.overlay(crop,view,selected,pid,evidence_pool,folder,layer='masks.png')
        index={c['id']:c for c in selected}
        measured=geometry_routing.measurements(list(index),index,proposals)
        for item in measured:
            candidate=index[item['id']]
            if candidate.get('origin')=='retained_local_sam':
                item['origin']='earlier local SAM'
                item['associated_existing']=candidate['retained_from']['existing_short_ids']
        inv=inventory(existing,view)
        mapping={r['short_id']:r['id'] for r in existing}
        with locks['cpu']:draw_current(crop,existing,view,folder/'input/current.png')
        context_size=Image.open(folder/'input/context.png').size
        focus_box=task.get('focus_bbox',task['bbox'])
        focus_points=(np.asarray(focus_box)-np.tile(view['bbox_global'][:2],2))*np.tile(view['scale'],2)
        retain_text=config.get('repair_retain_locator_text',False)
        q=prompt(crop.size,context_size,dict(description=task['description'],bbox=[round(float(x),1) for x in focus_points]),measured,inv,step==0,config.get('repair_focus_relocation',False),retain_text)
        if task.get('focus_regions'):
            q['focus']['regions']=[dict(bbox=np.round((np.asarray(f['bbox'])-np.tile(view['bbox_global'][:2],2))*np.tile(view['scale'],2),1).tolist()) for f in task['focus_regions']]
            if retain_text:
                for f,source in zip(q['focus']['regions'],task['focus_regions']):f['description']=source['description']
            q['instructions'].append('The focus regions are grouped only to avoid simultaneous edits to nearby evidence. They may contain independent networks. Do not connect them merely because they share this request; check each against CLEAN.')
        if shared is not None:
            q['shared_evidence']=dict(additional_mask_pixels=shared['additional_pixels'],native_observations=shared['observations'])
            q['instructions'].append('Pale cyan is additional native SAM coverage from other observations. It is optional pixel evidence, NOT an existing recovered path, a new target, or a required connection. CURRENT is the actual reconstruction. Use CLEAN to distinguish missed source strokes from segmentation mistakes.')
        q['crop_in_context']=ev.crop_in_context(view,context_size)
        q['instructions'].append('crop_in_context is the exact CLEAN rectangle in CONTEXT pixels, shown by the magenta annotation. Use this correspondence directly.')
        images=[('CLEAN: original local pixels',folder/'input/clean.png'),('MASKS: local SAM evidence on CLEAN',folder/'input/masks.png'),
                ('CURRENT: existing geometry, may be incomplete or wrong',folder/'input/current.png'),('CONTEXT: whole figure, crop marked',folder/'input/context.png')]
        write(folder/'prompt.json',q);write(folder/'existing_map.json',mapping)
        write(folder/'input/images.json',[dict(label=l,path=str(p.resolve()),sha256=sha(p)) for l,p in images])
        answer=model.ask(folder/'request',q,images,lambda a:validate(a,crop.size,context_size,mapping,index,step==0,{r['id']:r['points'] for r in inv}))
        write(folder/'answer.json',answer)
        attempts.append(dict(folder=folder.name,status=answer['status'],bbox_global=view['bbox_global'],masks=len(selected)))
        if answer['status']!='expand':break
        requested=(np.asarray(answer['context_bbox'])*np.tile(np.asarray(clean.size)/context_size,2)).tolist()
    newrows=[];issues=[];head_calls=0;review_calls=0
    if answer['status']=='repair':
        metadata={r['id']:dict(id=pid+':'+r['id'],heads=r['heads'],line_style=r['line_style'],
            source_masks=r['masks'],source_mask_paths=[index[m]['path'] for m in r['masks']],globals=[],
            identity_index=i+1,geometry_request=str(folder/'request'),local_input=str(folder/'input/clean.png')) for i,r in enumerate(answer['routes'])}
        with locks['cpu']:
            centerline={**answer,'routes':[r for r in answer['routes'] if r.get('kind')!='block_arrow']}
            fitted=sg.geometry(centerline,metadata);local=[]
            for r in answer['routes']:
                if r.get('kind')=='block_arrow':
                    row={**metadata[r['id']],**block_geometry(r,answer['nodes'],index),
                         'heads':['none','none'],'semantic_heads':r['heads'],'shared_nodes':[]}
                    owned=[row]
                else:
                    owned=[row for row in fitted if row['geometry_target_id']==r['id']]
                for row in owned:
                    nodes=['contour:'+r['id']] if r.get('kind')=='block_arrow' else list(dict.fromkeys(n for s in row['model_segments'] for n in s['nodes']))
                    local.append(row)
                    row['node_ids']=[pid+':'+str(n) for n in nodes];row['shared_nodes']=[pid+':'+str(n) for n in row['shared_nodes']]
                    for end,h in zip(('start','end'),row['heads']):row[end+'_head']=h not in ('none','unknown');row[end+'_head_type']=h if h not in ('none','unknown') else None
        support_by_id={row['id']:np.logical_or.reduce([index[m]['mask'] for m in row['source_masks']]) if row['source_masks'] else None for row in local}
        if config.get('geometry_critic_enabled',False):
            import geometry_critic
            local,support_by_id,review_issues,review_calls=geometry_critic.recover(
                crop,local,support_by_id,out/'geometry_critic',model,config,locks['cpu'],source_view=view)
            issues.extend(review_issues)
        if config.get('separate_style',False):
            head_issues=[];head_calls=0
        else:local,head_issues,head_calls=head_semantics.recover(crop,local,support_by_id,out/'heads',model,full_image=clean,view=view,cpu_slot=locks['cpu'])
        issues.extend(head_issues)
        with locks['cpu']:
            for row in local:
                if config.get('separate_style',False):
                    import style_contract
                    row['style']=style_contract.provisional(crop,support_by_id[row['id']],row)
                    row['geometry_frame']=view
                else:
                    row['style']=style_windows.recover(np.asarray(crop),support_by_id[row['id']],row)
                    if style.pending(row):issues.append(dict(stage='style_pending',route=row['id']))
            mask_vectors.assign_parents(local,pid)
            newrows=[fit.transform(r,view) for r in local]
            write(out/'rows_local.json',local);write(out/'rows_global.json',newrows)
            viz.render(folder/'input/clean.png',local,out/'patch_result')
    after=apply_patch(current,answer,newrows,mapping)
    write(out/'patch.json',dict(status=answer['status'],replace=[mapping[k] for k in answer.get('replace',[])],added=newrows))
    write(out/'attempts.json',attempts)
    with locks['cpu']:
        draw_current(crop,existing_rows(original,view),view,out/'before.png',False)
        draw_current(crop,existing_rows(after,view),view,out/'after.png',False)
    summary=dict(group=pid,state='completed_with_issues' if issues or answer['status']=='unresolved' else 'completed',
        rows=newrows,issues=issues,status=answer['status'],replace=[mapping[k] for k in answer.get('replace',[])],
        semantic_calls=0,repair_calls=len(attempts),geometry_calls=len(attempts),head_calls=head_calls,task=task)
    if config.get('geometry_critic_enabled',False):summary['geometry_critic_calls']=review_calls
    write(out/'completed.json',summary)
    (out/'README.md').write_text('# 相关连接补全\n\n'+task['description']+'\n\n'+
        '| 原图 | SAM 证据 | 修复前实际路径 | 修复后实际路径 |\n|---|---|---|---|\n'+
        f'| ![]({folder.name}/input/clean.png) | ![]({folder.name}/input/masks.png) | ![](before.png) | ![](after.png) |\n\n'+
        f'[实际请求]({folder.name}/prompt.json) · [模型决定]({folder.name}/answer.json) · [补丁](patch.json)\n'+
        ('\n[端点头型的实际输入](heads/sources.json) · [端点头型请求](heads/request/prompt.json)\n' if head_calls else ''))
    return after,summary


def merge_results(results,patch):
    """Apply explicit model edits without resetting other groups or identities."""
    result=copy.deepcopy(results);removed=set(patch['replace'])
    for group in result:group['rows']=[r for r in group.get('rows',[]) if r['id'] not in removed]
    result.append(copy.deepcopy(patch));return result
