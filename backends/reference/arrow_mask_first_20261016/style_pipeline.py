"""Reusable fixed-geometry style recovery and real PPT critic/refine.

The same entry is used by full runs and saved-geometry replay. Failures retain
the last valid style/geometry and are explicit issues, never deleted paths.
"""
import copy,hashlib,json,traceback,threading
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import evidence as ev
import style_contract as contract
import block_shapes,shape_render,visual
from lib import fit,native,viz
from transport.common import write_json as write,read_json as read,sha256 as sha


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def binding(crop,rows,out,view,source_size):
    out.mkdir(parents=True,exist_ok=True);crop.save(out/'clean.png')
    locator=crop.copy();draw=ImageDraw.Draw(locator)
    for i,r in enumerate(rows):
        pts=fit.points(r);c=visual.color(i)
        if r['kind']=='block_arrow' and r.get('shape_parts'):
            from shape_core.fitting.paths import sample_commands
            from shape_core.styling.geometry import bounds
            for j,part in enumerate(r['shape_parts']):
                samples=sample_commands(part['commands'])
                draw.line([tuple(p) for p in samples],fill=visual.color(i+j),width=2)
                b=bounds(part);center=[(b[0]+b[2])/2,(b[1]+b[3])/2]
                visual.label(draw,center,r['id']+'/'+part['id'],visual.color(i+j),locator)
            continue
        draw.line([tuple(p) for p in pts],fill=c,width=2)
        for end,p in [('start',pts[0]),('end',pts[-1])]:
            visual.label(draw,p,r['id']+':'+end,c,locator)
    locator.save(out/'locators.png')
    write(out/'geometry.json',contract.geometry_signature(rows))
    write(out/'frame.json',dict(view=view,source_size=list(source_size),style_unit='local CLEAN pixels'))


def render(rows,out,source_size,view,locks):
    """Render at full-slide physical scale, then rasterize PDF on CLEAN's grid."""
    import fitz
    out.mkdir(parents=True,exist_ok=True)
    global_rows=[fit.transform(r,view) for r in rows]
    with locks['cpu']:
        native.write(source_size,global_rows,[r['style'] for r in global_rows],out/'styled.pptx')
        native.render(out/'styled.pptx',out/'native',source_size)
        with native.PDF_LOCK,fitz.open(out/'native/styled.pdf') as src,fitz.open() as local:
            size=view['size'];page=local.new_page(width=size[0],height=size[1]);m=view['global_to_view']
            dx=m[0][2]+.5-.5*m[0][0];dy=m[1][2]+.5-.5*m[1][1]
            rect=fitz.Rect(dx,dy,source_size[0]*m[0][0]+dx,source_size[1]*m[1][1]+dy)
            if src[0].get_contents():page.show_pdf_page(rect,src,0,keep_proportion=False)
            pix=page.get_pixmap(alpha=False)
            image=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
            if image.size!=tuple(size):raise ValueError('Native crop grid mismatch')
            image.save(out/'render.png')
    write(out/'render_binding.json',dict(geometry_sha256=digest(contract.geometry_signature(rows)),
        styles=[contract.wire(r) for r in rows],pptx_sha256=sha(out/'styled.pptx'),
        render_sha256=sha(out/'render.png'),view=view,source_size=list(source_size)))
    return out/'render.png'


def apply_styles(rows,answer):
    table={v['id']:v for v in answer['styles']};result=copy.deepcopy(rows)
    for row in result:
        row['style']=contract.validate_style(table[row['id']],row)
        row['style']['needs_review']=True
        if row['kind']!='block_arrow':
            row['heads']=[row['style'][e]['type'] if row['style'][e] else 'none' for e in ('start','end')]
            for end in ('start','end'):
                row[end+'_head']=row['style'][end] is not None
                row[end+'_head_type']=row['style'][end]['type'] if row['style'][end] else None
            row['line_style']='dashed' if row['style']['dash']['pattern_px'] else 'solid'
    if contract.geometry_signature(rows)!=contract.geometry_signature(result):raise ValueError('Style changed geometry')
    return result


def recover(crop,rows,supports,out,config,model,locks,view=None,source_size=None,prepare_shapes=False):
    """Local style API, also usable offline from saved geometry and optional masks."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=copy.deepcopy(rows);issues=[]
    view=view or ev.identity_view(crop.size);source_size=source_size or crop.size
    # identity_view historically omitted size; style coordinates are explicit.
    view={**view,'size':list(crop.size)}
    write(out/'input_rows.json',rows)
    for row in rows:
        row['style']=row.get('style') or contract.provisional(crop,supports.get(row['id']),row)
    render(rows,out/'before',source_size,view,locks)
    for i,row in enumerate(rows):
        if row['kind']!='block_arrow':continue
        if not prepare_shapes or row.get('shape_parts'):
            rows[i]['shape_parts']=row.get('shape_parts') or shape_render.mask_parts(row)
            continue
        try:
            rows[i]=block_shapes.recover(crop,row,supports.get(row['id']),out/'shape_geometry'/f'S{i:03d}',model)
        except Exception as exc:
            rows[i]['shape_parts']=row.get('shape_parts') or shape_render.mask_parts(row)
            support=supports.get(row['id'])
            if support is not None and support.any():
                import mask_vectors
                rows[i]['mask_shape_parts']=shape_render.mask_parts(mask_vectors.contour(support))
            issues.append(dict(stage='block_shape_geometry',id=row['id'],error=str(exc),retained='input geometry'))
    binding(crop,rows,out/'input',view,source_size)
    images=[('CLEAN reference',out/'input/clean.png'),('LOCATORS annotations only',out/'input/locators.png')]
    pixel_evidence=None
    if config.get('style_pixel_evidence',False):
        import style_evidence
        pixel_evidence=style_evidence.measurements(crop,rows,supports,out/'input')
    # Preserved masks let the model make the requested shape-vs-mask choice.
    blocks=[(r['id'],supports[r['id']]) for r in rows if r['kind']=='block_arrow' and supports.get(r['id']) is not None]
    if blocks:
        visual.masks(crop,blocks,out/'input/block_masks.png')
        images.append(('BLOCK MASKS optional fallback evidence',out/'input/block_masks.png'))
    previous=copy.deepcopy(rows);history=[];feedback=None
    max_refines=config.get('style_refine_rounds',1)
    for revision in range(max_refines+1):
        folder=out/f'rev_{revision:02d}';folder.mkdir(parents=True,exist_ok=True)
        try:
            review_existing=revision==0 and config.get('style_entry')=='review_existing'
            if review_existing:
                # Validate saved parameters, but retain their actual values.
                from lib.style import pending
                if any(pending(r) for r in rows):review_existing=False
                else:
                    for row in rows:contract.validate_style(contract.wire(row),row)
            if review_existing:candidate=copy.deepcopy(rows)
            else:
                q=contract.prompt(rows,crop.size,'estimate' if revision==0 else 'refine',feedback)
                if pixel_evidence is not None:
                    q['pixel_measurements']=pixel_evidence
                    q['instructions'].append('Pixel measurements are optional estimates on the final geometry, in CLEAN pixels. Prefer a supported measurement over guessing a different value; reject it when the mask or sampling visibly includes a neighbor/background. Do not copy measurements through a geometry error.')
                estimate_images=images+([('RENDER previous revision',last_render)] if revision else [])
                answer=model.ask(folder/'estimate',q,estimate_images,lambda a:contract.validate(a,rows))
                candidate=apply_styles(rows,answer)
            image=render(candidate,folder,source_size,view,locks)
            rows=candidate;last_render=image
            write(folder/'rows.json',rows)
            critic_prompt=contract.prompt(rows,crop.size,'critic')
            critic_images=images+[('RENDER actual PPT',image)]
            if pixel_evidence is not None:critic_prompt['pixel_measurements']=pixel_evidence
            if config.get('style_detail_views',False):
                import style_evidence
                critic_images+=style_evidence.details(crop,rows,image,folder/'details')
                critic_prompt['instructions'].append('Use the paired SOURCE/PPT details to check shaft thickness and head extent, not just presence/type. Both sides use identical boxes and scale. Identify a visible mismatch before accepting; revise only the named style.')
            review=model.ask(folder/'critic',critic_prompt,critic_images,
                             lambda a:contract.checks(a,rows))
            write(folder/'review.json',review);feedback=review['checks']
            history.append(dict(revision=revision,entry='review_existing' if review_existing else 'estimate',render=str(image.resolve()),checks=feedback))
            byid={c['id']:c for c in feedback};changed_geometry=False
            for i,row in enumerate(rows):
                check=byid[row['id']]
                row['style']['state']='reviewed' if check['verdict']=='accept' else check['verdict']
                row['style']['needs_review']=check['verdict']!='accept'
                row['style']['review']=check
                if check.get('geometry_action')=='mask' and revision<max_refines:
                    rows[i]=block_shapes.use_parts(row,row['mask_shape_parts'],'model_selected_mask_shape')
                    rows[i]['style'].pop('shape_parts',None);changed_geometry=True
                    history[-1].setdefault('geometry_replacements',[]).append(row['id'])
                elif check.get('geometry_action')=='mask':
                    row['style'].update(state='geometry_issue',needs_review=True)
                    issues.append(dict(stage='shape_mask_selection_pending',id=row['id'],reason='Revision limit reached; retained the rendered geometry and style'))
            if changed_geometry:
                updated_input=folder/'mask_geometry_input'
                binding(crop,rows,updated_input,view,source_size)
                images=[('CLEAN reference',updated_input/'clean.png'),('LOCATORS annotations only',updated_input/'locators.png')]+images[2:]
                # Previous render belongs to old geometry; explicitly re-render
                # the newly selected contour before requesting its style.
                last_render=render(rows,folder/'mask_geometry',source_size,view,locks)
            if not changed_geometry and not any(c['verdict']=='revise' for c in feedback):break
        except Exception as exc:
            issues.append(dict(stage='style_revision',revision=revision,error=str(exc),retained='last valid candidate'))
            write(folder/'error.json',dict(error=str(exc),traceback=traceback.format_exc()))
            break
    unresolved=[r['id'] for r in rows if r['style'].get('state')!='reviewed']
    if unresolved:issues.append(dict(stage='style_unresolved',ids=unresolved))
    # A final image always exists, including explicit provisional results.
    final_image=render(rows,out/'final',source_size,view,locks)
    write(out/'rows.json',rows);write(out/'history.json',history)
    write(out/'status.json',dict(state='completed_with_issues' if issues else 'completed',issues=issues,
         style_revisions=len(history),paths=len(rows),geometry_unchanged=contract.geometry_signature(previous)==contract.geometry_signature(rows),
         geometry_comparison_scope='fixed geometry entering style; excludes the earlier shape geometry stage',
         visual_accuracy_verified=False))
    (out/'RESULTS.md').write_text('# 固定几何的样式恢复\\n\\n| 原图 | 原样式 | 最终实际 PPT |\\n|---|---|---|\\n'
        '| ![](input/clean.png) | ![](before/render.png) | ![](final/render.png) |\\n\\n'
        '[逐轮记录](history.json) · [状态和未解决问题](status.json) · [PPT](final/styled.pptx)\\n'.replace('\\n','\n'))
    return rows,issues


def frame_for(row,clean):
    if row.get('geometry_frame') and row.get('local_input'):
        return Image.open(row['local_input']).convert('RGB'),row['geometry_frame']
    if row.get('local_input'):
        path=Path(row['local_input']);transform=path.parent/'transform.json'
        if path.exists() and transform.exists():return Image.open(path).convert('RGB'),read(transform)
    return clean.copy(),ev.identity_view(clean.size)


def finish_case(clean,results,out,config,model,locks):
    results=copy.deepcopy(results);rows=[r for group in results for r in group.get('rows',[])];batches={}
    for row in rows:
        key=(row.get('local_input') or '__full_figure__')
        batches.setdefault(key,[]).append(row)
    updates={};issues=[];out=Path(out)
    def work(key,items):
        crop,view=frame_for(items[0],clean);view={**view,'size':list(crop.size)}
        local=[fit.transform(r,view,to_global=False) for r in items];supports={}
        for r in local:
            masks=[ev.mask(p) for p in r.get('source_mask_paths',[]) if Path(p).exists()]
            masks=[m for m in masks if m.shape==(crop.height,crop.width)]
            supports[r['id']]=np.logical_or.reduce(masks) if masks else None
        folder=out/'styles'/hashlib.sha256(key.encode()).hexdigest()[:12]
        styled,errors=recover(crop,local,supports,folder,config,model,locks,view,clean.size)
        return [fit.transform(r,view) for r in styled],errors
    with ThreadPoolExecutor(max_workers=config.get('task_workers',8)) as pool:
        jobs={pool.submit(work,key,items):items for key,items in batches.items()}
        for f in as_completed(jobs):
            try:
                new,errors=f.result();updates.update({r['id']:r for r in new});issues.extend(errors)
            except Exception as exc:
                issues.append(dict(stage='style_batch',ids=[r['id'] for r in jobs[f]],error=str(exc)))
    for group in results:group['rows']=[updates.get(r['id'],r) for r in group.get('rows',[])]
    write(out/'style_results.json',results);write(out/'style_issues.json',issues)
    return results,issues


def prepare_case_geometry(clean,results,out,config,model,locks):
    """Finish closed shape geometry BEFORE the fixed-geometry style checkpoint."""
    results=copy.deepcopy(results);updates={};issues=[]
    blocks=[r for group in results for r in group.get('rows',[]) if r['kind']=='block_arrow' and not r.get('shape_parts')]
    def work(row):
        crop,view=frame_for(row,clean);local=fit.transform(row,view,False)
        masks=[ev.mask(p) for p in row.get('source_mask_paths',[]) if Path(p).exists()]
        masks=[m for m in masks if m.shape==(crop.height,crop.width)]
        support=np.logical_or.reduce(masks) if masks else None
        folder=Path(out)/'shape_geometry'/hashlib.sha256(row['id'].encode()).hexdigest()[:12]
        return fit.transform(block_shapes.recover(crop,local,support,folder,model),view)
    with ThreadPoolExecutor(max_workers=config.get('task_workers',8)) as pool:
        jobs={pool.submit(work,r):r for r in blocks}
        for f in as_completed(jobs):
            row=jobs[f]
            try:updates[row['id']]=f.result()
            except Exception as exc:
                retained=copy.deepcopy(row);retained['shape_parts']=shape_render.mask_parts(row);updates[row['id']]=retained
                issues.append(dict(stage='block_shape_geometry',id=row['id'],error=str(exc),retained='input geometry'))
    for group in results:group['rows']=[updates.get(r['id'],r) for r in group.get('rows',[])]
    write(Path(out)/'shape_geometry_issues.json',issues)
    return results,issues
