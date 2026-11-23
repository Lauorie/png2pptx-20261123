"""Per-image coordinator; whole arrow objects progress independently in one pool."""
import copy
import shutil
import time
from concurrent.futures import as_completed
from pathlib import Path
from PIL import Image
from shapely.geometry import LineString
from shapely.ops import unary_union
from ..common import save,read,sha,digest,now,usage,intersect
from ..fitting.paths import sample_commands
from . import PROTOCOL
from . import geometry as geo,contracts,prompts,model,export


def discover(source,folder,runtime,opt,pool):
    folder.mkdir(parents=True,exist_ok=True)
    gf=geo.frame(source,None,folder/'global/input',opt,full=True);errors=[]
    clean=folder/'global/input/clean.png'
    try:
        raw=model.ask(runtime,folder/'global',prompts.discover(gf['size'],layout=True),[('CLEAN full figure',clean)],
                      lambda v:contracts.discovery(v,gf['size'],layout=True),opt)
    except Exception as exc:
        raw={'objects':[],'regions':[],'uncertain':['Global discovery failed: '+str(exc)]};errors.append(raw['uncertain'][0])
    objects=[];history=[]
    def append(rows,f,origin):
        for row in rows:
            if row['same_as']:
                history.append({'origin':origin,'action':'already_covered','same_as':row['same_as'],'description':row['description']});continue
            b=geo.box(row['bbox_2d'],f,True);b=[max(0,min(v,f['source_size'][i%2]-1)) for i,v in enumerate(b)]
            o={'id':f'A{len(objects)+1:03d}','bbox_2d':b,'description':row['description'],'origin':origin}
            objects.append(o);history.append({'action':'add',**o})
    append(raw['objects'],gf,'global')
    save(folder/'global.json',{'objects':objects,'raw':raw});geo.draw_boxes(clean,objects,gf,folder/'global_overlay.png')
    regions=raw['regions']
    if not regions:
        w,h=[v-1 for v in gf['size']]
        # A failed full-image call must not force another equally large search.
        regions=([{'bbox_2d':[x0,y0,x1,y1],'description':'Coarse fallback region after failed global discovery'}
                  for y0,y1 in ((0,h/2),(h/2,h)) for x0,x1 in ((0,w/2),(w/2,w))] if errors else
                 [{'bbox_2d':[0,0,w,h],'description':'Full-image fallback'}])
    save(folder/'regions.json',regions)
    snapshot=copy.deepcopy(objects);results={};concerns=list(raw['uncertain'])
    def local(i,region):
        root=folder/'regions'/f'R{i:02d}';b=geo.box(region['bbox_2d'],gf,True)
        f=geo.frame(source,b,root/'input',opt);visible=[o for o in snapshot if intersect(o['bbox_2d'],f['box'])>0]
        geo.draw_boxes(root/'input/clean.png',visible,f,root/'input/target.png')
        ids=[o['id'] for o in visible]
        value=model.ask(runtime,root,prompts.discover(f['size'],ids),
            [('CLEAN region',root/'input/clean.png'),('Existing complete-arrow candidates, visual IDs only',root/'input/target.png')],
            lambda v:contracts.discovery(v,f['size'],ids),opt)
        return value,f
    jobs={pool.submit(local,i,r):i for i,r in enumerate(regions,1)}
    for fut in as_completed(jobs):
        i=jobs[fut]
        try:results[i]=fut.result()
        except Exception as exc:errors.append(f'Region R{i:02d}: {exc}')
    for i in sorted(results):
        value,f=results[i];append(value['objects'],f,f'R{i:02d}');concerns.extend(value['uncertain'])
    # A full view catches long paths that no individual region contains.
    review=folder/'coverage';review.mkdir(exist_ok=True)
    geo.draw_boxes(clean,objects,gf,review/'existing.png');ids=[o['id'] for o in objects]
    try:
        value=model.ask(runtime,review,prompts.discover(gf['size'],ids,concerns=concerns),
                        [('CLEAN full figure; inspect complete cross-region routes',clean),('Existing candidate boxes, visual IDs only',review/'existing.png')],
                        lambda v:contracts.discovery(v,gf['size'],ids),opt)
        append(value['objects'],gf,'coverage');concerns.extend(value['uncertain'])
    except Exception as exc:errors.append('Coverage: '+str(exc))
    result={'objects':objects,'uncertain':concerns,'errors':errors,'history':history}
    save(folder/'candidates.json',result);geo.draw_boxes(clean,objects,gf,folder/'final_overlay.png')
    return result


def fit_object(source,candidate,folder,runtime,opt,slots,*,initial_full_context=False):
    folder.mkdir(parents=True,exist_ok=True);save(folder/'candidate.json',candidate)
    record={'id':candidate['id'],'candidate':candidate,'state':'running','history':[],'started_utc':now()}
    def status(stage,**extra):
        record.update(stage=stage,updated_utc=now(),**extra);save(folder/'record.json',record)
    current=None;previous=None;feedback=None;f=None;context_round=0
    full_view=initial_full_context;fit_box=candidate['bbox_2d'];revision=0;refine_round=0
    try:
        while True:
            root=folder/'geometry'/f'revision_{revision:02d}'
            oldf=f
            f=geo.frame(source,fit_box,root/'input',opt,full=full_view)
            if current and oldf!=f:
                previous=geo.mapped(geo.mapped(current,oldf,True),f)
                previous.pop('uncertain',None)  # prose may contain old-view pixel positions
            else:previous=current
            geo.draw_boxes(root/'input/clean.png',[{**candidate,'bbox_2d':fit_box}],f,root/'input/target.png')
            save(root/'input/target.json',{'id':candidate['id'],'bbox_source':fit_box,'original_candidate_bbox':candidate['bbox_2d']})
            images=[('CLEAN target crop',root/'input/clean.png'),('Target candidate box for identity only',root/'input/target.png')]
            if previous:
                geo.draw_geometry(root/'input/clean.png',previous,root/'input/previous.png')
                images.append(('Previous geometry, aligned with previous_geometry text',root/'input/previous.png'))
            status('geometry',geometry_attempt=revision)
            value=model.ask(runtime,root/'observe',prompts.geometry(f['size'],candidate['description'],previous,feedback),images,
                            lambda v:contracts.geometry(v,f['size']),opt)
            save(root/'geometry_local.json',value)
            if value['status']=='not_arrow':status('finished',state='excluded',reason=value['reason']);return record
            if value['status']=='needs_context':
                record['history'].append({'stage':'context','reason':value['reason']})
                if not initial_full_context and not full_view and context_round<opt['context_expansions']:
                    context_round+=1;full_view=True;revision+=1
                    # Keep the verbatim reason in history, not in a different
                    # view where its old pixel coordinates would be misleading.
                    feedback='The previous crop did not contain the complete target route. The new CLEAN shows the full figure; recover this complete target using only the current image and its visual identity box.'
                    continue
                break
            current=value;original=geo.mapped(value,f,True)
            record.update(geometry=original,geometry_state='unreviewed',geometry_revision=revision,
                          geometry_source_view='full_context' if full_view else 'local')
            save(root/'geometry_source.json',original);geo.draw_geometry(root/'input/clean.png',current,root/'input/fitted.png')
            panels=([('PREVIOUS',previous)] if previous else [])+[('CURRENT',current)]
            geo.compare_geometry(root/'input/clean.png',panels,root/'comparison.png')
            if full_view:
                # The full view reacquires the route; final tip placement and its
                # critic need an enlarged view, just like ordinary local fitting.
                fit_box=geo.bounds(original);full_view=False;revision+=1
                record['history'].append({'stage':'reframe','reason':'Recovered full route; continue geometry fitting in its enlarged local view.'})
                feedback='The full-figure view recovered a tentative complete route. Refine its centerline and visible extreme tips in this enlarged CLEAN crop; previous_geometry has been transformed into this crop.'
                continue
            status('geometry_critic')
            verdict=model.ask(runtime,root/'critic',prompts.critic(current),
                   [('CLEAN source',root/'input/clean.png'),('Current points and fitted centerlines; rings at headed tips',root/'input/fitted.png')],lambda v:contracts.geometry_review(v,current),opt)
            record['history'].append({'stage':'geometry_critic','revision':revision,**verdict})
            record['geometry_state']=verdict['verdict'];save(folder/'record.json',record)
            if verdict['verdict']=='accept' or refine_round>=opt['geometry_refines']:break
            feedback=verdict;refine_round+=1;revision+=1
    except Exception as exc:
        record['history'].append({'stage':'geometry_error','error':str(exc)})
        record['geometry_state']='unresolved'
    if not record.get('geometry'):
        status('finished',state='unresolved');return record
    return fit_style(source,folder,runtime,opt,slots,record)


def fit_style(source,folder,runtime,opt,slots,record):
    """Style one complete network on frozen geometry; also used by stage replay."""
    record=copy.deepcopy(record)
    def status(stage,**extra):
        record.update(stage=stage,updated_utc=now(),**extra);save(folder/'record.json',record)
    # Every later stage uses the selected original geometry and its own crop.
    g=record['geometry'];f=geo.frame(source,geo.bounds(g),folder/'style/input',opt)
    local=geo.mapped(g,f);geo.draw_geometry(folder/'style/input/clean.png',local,folder/'style/input/target.png',labels=False,terminals=True)
    clean=folder/'style/input/clean.png';target=folder/'style/input/target.png'
    guide=None
    if any(p['start_head'] or p['end_head'] for p in g['paths']):
        guide=folder/'style/input/head_types.png'
        shutil.copy2(Path(__file__).parent/'assets/head_types.png',guide)
    current_style={p['id']:{'stroke':'#000000','width_px':max(.8,opt['minimum_stroke_width_px']),'dash':'solid',
                         'start':{'type':'triangle','width':'med','length':'med'} if p['start_head'] else None,
                         'end':{'type':'triangle','width':'med','length':'med'} if p['end_head'] else None} for p in g['paths']}
    record.update(styles=copy.deepcopy(current_style),style_state='fallback_unresolved')
    feedback=None;last_render=None
    for revision in range(opt['style_refines']+1):
        root=folder/'style'/f'revision_{revision:02d}';root.mkdir(parents=True,exist_ok=True)
        before=copy.deepcopy(current_style);before_local=geo.local_styles(before,f)
        try:
            status('shaft',style_revision=revision)
            images=[('CLEAN target crop',clean),('Entire network path IDs',target)]
            if last_render:images.append(('Previous ACTUAL PPT render, matching previous_style',last_render))
            shaft=model.ask(runtime,root/'shaft',prompts.style('shaft',local,before_local if revision else None,feedback,opt['minimum_stroke_width_px']*f['scale'],size=f['size']),
                            images,lambda v:contracts.styles(v,local['paths'],'shaft'),opt)
            body=geo.source_styles(shaft['styles'],f,opt['minimum_stroke_width_px'])
            for pid,s in body.items():s.update(start=before[pid]['start'] if revision else None,end=before[pid]['end'] if revision else None)
            save(root/'shaft_source.json',body)
            # Preserve the successfully estimated shaft even if the head request fails.
            current_style={pid:{**copy.deepcopy(s),'start':before[pid]['start'],'end':before[pid]['end']} for pid,s in body.items()}
            record['styles']=copy.deepcopy(current_style)
            if any(p['start_head'] or p['end_head'] for p in g['paths']):
                # First pass shows the shaft only, avoiding a synthetic head-type
                # anchor. Refinement shows previous heads with the updated shaft.
                body_obj={'id':record['id'],'geometry':g,'styles':body}
                body_render=export.render_object(body_obj,source,f,root/'shaft_render',slots)
                status('heads')
                heads=model.ask(runtime,root/'heads',prompts.style('heads',local,geo.local_styles(body,f),feedback,size=f['size']),
                                [('CLEAN source',clean),('Path IDs and observed headed terminals',target),
                                 ('ACTUAL PPT matching shaft_style; first pass has no heads, refinement keeps previous heads',body_render),(prompts.HEAD_GUIDE_LABEL,guide)],
                                lambda v:contracts.styles(v,local['paths'],'heads'),opt)
                for row in heads['heads']:body[row['id']].update(start=row['start'],end=row['end'])
            current_style=body;record['styles']=copy.deepcopy(body);record['style_state']='unreviewed'
            save(root/'styles_source.json',body)
            obj={'id':record['id'],'geometry':g,'styles':body}
            last_render=export.render_object(obj,source,f,root/'render',slots)
            status('style_critic')
            critic_images=[('CLEAN source',clean),('ACTUAL PPT render, matching current_style',last_render),('Path IDs',target)]
            if guide:critic_images.append((prompts.HEAD_GUIDE_LABEL,guide))
            verdict=model.ask(runtime,root/'critic',prompts.critic(local,geo.local_styles(body,f),size=f['size']),
                            critic_images,contracts.review,opt)
            record['history'].append({'stage':'style_critic','revision':revision,**verdict})
            record['style_state']=verdict['verdict'];save(folder/'record.json',record)
            if verdict['verdict']=='accept':break
            feedback=verdict
        except Exception as exc:
            record['history'].append({'stage':'style_error','revision':revision,'error':str(exc)})
            record['style_state']='unresolved';break
    state='completed' if record['geometry_state']=='accept' and record['style_state']=='accept' else 'completed_with_unresolved'
    status('finished',state=state);return record


def duplicate(a,b,tolerance):
    """Same complete network only; crossings and partially shared trunks survive."""
    def signature(obj):
        g=obj['geometry'];lines=[];tips=[]
        for p in g['paths']:
            lines.append(LineString(sample_commands(geo.commands(g,p),160)))
            if p['start_head']:tips.append(g['points'][p['segments'][0]['points'][0]])
            if p['end_head']:tips.append(g['points'][p['segments'][-1]['points'][-1]])
        return unary_union(lines),tips
    ga,ta=signature(a);gb,tb=signature(b)
    if len(ta)!=len(tb) or ga.hausdorff_distance(gb)>tolerance:return False
    remaining=list(tb)
    for tip in ta:
        if not remaining:return False
        distances=[sum((a-b)**2 for a,b in zip(tip,q))**.5 for q in remaining]
        best=min(range(len(distances)),key=distances.__getitem__)
        if distances[best]>tolerance:return False
        remaining.pop(best)
    return True


def run_case(source,folder,runtime,opt,pool,slots,base=None,progress=lambda *a:None,*,frozen=None,retry_missing_geometry=False):
    folder.mkdir(parents=True,exist_ok=True)
    identity={'source_sha256':sha(source),'protocol':PROTOCOL,'options':opt,'model':runtime.config['model'],'base_sha256':sha(base) if base else None}
    if frozen:
        identity['geometry_source']={'path':str(frozen),'scene_sha256':sha(frozen/'scene.json')}
        identity['retry_missing_geometry']=retry_missing_geometry
    if (folder/'identity.json').exists() and read(folder/'identity.json')!=identity:raise ValueError('Arrow identity changed; use a new run')
    save(folder/'identity.json',identity)
    if (folder/'summary.json').exists():return read(folder/'summary.json')
    started=time.monotonic();records={};last=0
    stage=('recovery_and_style' if retry_missing_geometry else 'style_replay') if frozen else 'geometry_and_style'
    if frozen:
        if read(frozen/'summary.json')['state'] not in ('completed','completed_with_unresolved'):raise ValueError('Replay requires a terminal arrow case')
        scene=read(frozen/'scene.json')
        if scene['source_sha256']!=sha(source):raise ValueError('Frozen geometry source image differs')
        found=read(frozen/'detection/candidates.json');save(folder/'detection/candidates.json',found)
        for name in ('final_overlay.png','global_overlay.png','regions.json'):
            if (frozen/'detection'/name).exists():shutil.copy2(frozen/'detection'/name,folder/'detection'/name)
        save(folder/'detection/provenance.json',{'source':str(frozen/'detection'),'candidates_sha256':sha(frozen/'detection/candidates.json')})
        jobs={};candidates_by_id={c['id']:c for c in found['objects']}
        for oid,old in scene['records'].items():
            record=copy.deepcopy(old);target=folder/'objects'/oid
            record['source_record']={'path':str(frozen/'objects'/oid/'record.json'),'scene_sha256':sha(frozen/'scene.json')}
            if retry_missing_geometry and not old.get('geometry') and old['state']!='excluded':
                save(target/'source_record.json',record)
                candidate=old.get('candidate') or candidates_by_id[oid]
                jobs[pool.submit(fit_object,source,candidate,target,runtime,opt,slots,initial_full_context=True)]=candidate
                continue
            if not old.get('geometry') or old['state']=='excluded':
                records[oid]=record;save(target/'record.json',record);continue
            record.update(state='running',history=[],started_utc=now())
            save(target/'source_geometry.json',old['geometry']);save(target/'candidate.json',old['candidate'])
            jobs[pool.submit(fit_style,source,target,runtime,opt,slots,record)]=old['candidate']
    else:
        progress('discovery',0,0)
        found=discover(source,folder/'detection',runtime,opt,pool)
        jobs={pool.submit(fit_object,source,c,folder/'objects'/c['id'],runtime,opt,slots):c for c in found['objects']}
    total=len(found['objects'])
    save(folder/'progress.json',{'stage':stage,'done':len(records),'total':total})
    progress(stage,len(records),total)
    def publish(final=False):
        values=[records[k] for k in sorted(records) if records[k].get('geometry') and records[k].get('styles')]
        save(folder/'scene.json',{'source_sha256':sha(source),'objects':values,'records':records,'state':'finished' if final else 'running'})
        if values:
            with Image.open(source) as im:size=im.size
            temp=folder/'arrows.partial.pptx';export.write(values,size,temp);temp.replace(folder/'arrows.pptx')
            if base:
                temp=folder/'combined.partial.pptx';export.write(values,size,temp,base);temp.replace(folder/'combined.pptx')
        save(folder/'progress.json',{'stage':'finished' if final else stage,'done':len(records),'total':total,'updated_utc':now()})
    for fut in as_completed(jobs):
        c=jobs[fut]
        try:records[c['id']]=fut.result()
        except Exception as exc:records[c['id']]={'id':c['id'],'state':'unresolved','error':str(exc)};save(folder/'objects'/c['id']/'runner_error.json',records[c['id']])
        progress(stage,len(records),total)
        if time.monotonic()-last>10:publish();last=time.monotonic()
    publish(True)
    result=finalize_case(source,folder,records,found,opt,slots,base,started=started,frozen=frozen)
    progress('finished',total,total);return result


def finalize_case(source,folder,records,found,opt,slots,base=None,*,started=None,frozen=None):
    """One deterministic exporter for live runs and frozen experimental replays."""
    if started is None:started=time.monotonic()
    total=len(found['objects'])
    active=[];duplicates=[]
    for key in sorted(records):
        row=records[key]
        if not row.get('geometry') or not row.get('styles'):continue
        match=next((r for r in active if duplicate(row,r,opt['duplicate_tolerance_px'])),None)
        if match:duplicates.append({'id':key,'same_as':match['id'],'method':'whole centerline Hausdorff + headed-tip matching','tolerance_source_px':opt['duplicate_tolerance_px']})
        else:active.append(row)
    save(folder/'duplicates.json',duplicates)
    save(folder/'scene.json',{'source_sha256':sha(source),'objects':active,'records':records,'state':'finished'})
    counts=export.final(active,source,folder,slots,base)
    unresolved=sum(r['state'] not in ('completed','excluded') for r in records.values())
    result={'state':'completed_with_unresolved' if unresolved or found['errors'] or found['uncertain'] else 'completed','candidates':total,'objects':len(active),'unresolved':unresolved,
            'excluded':sum(r['state']=='excluded' for r in records.values()),'duplicates':len(duplicates),'counts':counts,
            'wall_seconds':time.monotonic()-started,'detection_errors':found['errors'],'visual_accuracy_verified':False}
    save(folder/'summary.json',result);save(folder/'usage.json',usage(folder))
    if frozen:result['geometry_source']=str(frozen);save(folder/'summary.json',result)
    lines=['# 细线箭头恢复','',f'状态：{result["state"]}。候选 {total}，输出对象 {len(active)}，未决 {unresolved}。',
           '', '[箭头 PPT](arrows.pptx) · [Scene](scene.json) · [定位](detection/final_overlay.png)', '', '![原图与原生箭头](comparison.png)']
    if base:lines+=['','[组合 PPT](combined.pptx)','','![Shape 与箭头组合](combined_comparison.png)']
    lines+=['','| 对象 | 状态 | 几何 | 样式 | 点线对比 |','| --- | --- | --- | --- | --- |']
    for k,r in sorted(records.items()):
        revision=r.get('geometry_revision');preview=f'objects/{k}/geometry/revision_{revision:02d}/comparison.png' if revision is not None else None
        link=f'[查看]({preview})' if preview and (folder/preview).exists() else '—'
        lines.append(f'| [{k}](objects/{k}/record.json) | {r["state"]} | {r.get("geometry_state","—")} | {r.get("style_state","—")} | {link} |')
    (folder/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n');return result
