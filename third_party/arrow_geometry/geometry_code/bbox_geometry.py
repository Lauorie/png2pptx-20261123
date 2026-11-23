"""Import saved discovery boxes directly into geometry. No discovery/identity/SAM calls."""
import argparse,copy,fcntl,json,math,os,sys,threading,time,traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'lib')]
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import evidence as ev
import geometry_critic as critic
import geometry_heads as heads
import native
import segmented_geometry as sg
import target_locator
from lib import fit
from transport.common import read_json as read,write_json as write,sha256
from transport.model import Model


def make_frame(source,bbox,folder,config,expansion=0):
    bbox=np.asarray(bbox,float)
    if bbox.shape!=(4,) or not np.isfinite(bbox).all() or np.any(bbox[2:]<=bbox[:2]):raise ValueError('Invalid native discovery bbox')
    center=(bbox[:2]+bbox[2:])/2
    extent=np.maximum(bbox[2:]-bbox[:2],config['pilot_min_extent'])*config['crop_context_scale']*(2**expansion)
    cfg={**config,'display_side':config['pilot_display_side'],'min_crop_side':config['pilot_min_extent'],
         'display_long_side_range':config.get('geometry_input_long_side_range')}
    crop,frame=ev.frame(source,[np.r_[center-extent/2,center+extent/2]],folder,cfg)
    return crop,frame


def local_box(bbox,frame):
    points=np.asarray([[bbox[0],bbox[1],1],[bbox[2],bbox[3],1]])@np.asarray(frame['global_to_view']).T
    return points[:,:2].reshape(-1).tolist()


def geometry_prompt(obj,variant,frame,context_size,expanded=False):
    target=dict(id='R1',bbox=[round(x,2) for x in local_box(obj['bbox_native'],frame)])
    if variant=='bbox_locator':target['description']=obj.get('locator','')
    q=sg.prompt([target],frame['size'],locator_mode='boxes',with_heads=True,linework_only=True)
    q['task']='Recover the visible arrow/line geometry located by this ONE discovery box from CLEAN. Return geometry directly, without an identity or architecture explanation.'
    q['instructions']=[s.replace('already confirmed R targets','candidate R regions') for s in q['instructions']]
    q['instructions'][:0]=[
        'The box is a discovery locator, not validated identity or a clipping boundary. Follow the visible strokes to their actual ends. One region may contain several distinct arrows; draw each real arrow once. Crossings and repeated patterns do not imply missing all-to-all edges.',
        'CLEAN defines output pixel coordinates. BOX has only one R1 corner annotation, expanded 5% for visibility. CONTEXT locates CLEAN in the whole figure. Do not treat annotation edges as lines. Any supplied description is an unverified appearance hint, never a required connection.',
        'Exclude non-arrow shape/container borders, text glyphs and image texture. If no eligible target is supported, return status=empty with a brief reason and empty nodes/routes.',
        ('If the target is cut off by CLEAN, return status=needs_context with a short reason and no geometry. The same target gets one larger source view.' if not expanded else 'This is the expanded view. If still clipped or ambiguous, return status=needs_context and no geometry; do not invent the missing part.')]
    q['context_size']=list(context_size);q['crop_in_context']=ev.crop_in_context(frame,context_size)
    q['output_schema']['status']='draw|empty|needs_context'
    return q


def validate(value,size):
    if not isinstance(value,dict):raise ValueError('Return a JSON object')
    status=value.get('status','draw' if value.get('routes') else None)
    if status not in ('draw','empty','needs_context'):
        raise ValueError('This is a linework request: use draw with shaft centerlines and endpoint heads, empty, or needs_context. Shape routing is selected before this call.')
    if status!='draw':
        if value.get('nodes') or value.get('routes'):raise ValueError('No geometry for empty/needs_context')
        if not isinstance(value.get('reason'),str) or not value['reason'].strip():raise ValueError('Give a brief reason')
        return dict(status=status,reason=value['reason'],nodes=[],routes=[])
    result=sg.validate(value,{'R1'},size,require_heads=True);result['status']='draw';return result



def preview_style(row,width):
    row['style']=dict(color='#666666',width_px=width,dash=dict(type='solid',preset='solid'),
        start=heads.head_style(row.get('heads',['none','none'])[0]),end=heads.head_style(row.get('heads',['none','none'])[1]),state='geometry_preview')
    return row

def prepare_target(source,obj,folder,config,expansion=0):
    folder=Path(folder);cfg={**config,'pilot_min_extent':config.get('bbox_min_extent',25),
        'pilot_display_side':config.get('bbox_display_side',1080),'crop_context_scale':config.get('bbox_context_scale',2.)}
    crop,frame=make_frame(source,obj['bbox_native'],folder,cfg,expansion)
    target_locator.draw(crop,[dict(id='R1',bbox=local_box(obj['bbox_native'],frame))],folder/'target_boxes.png',.05,'corners')
    context=source.copy();context.thumbnail((config.get('context_side',768),)*2,Image.Resampling.LANCZOS)
    ImageDraw.Draw(context).rectangle(ev.crop_in_context(frame,context.size),outline='#c00080',width=2);context.save(folder/'context.png')
    q=geometry_prompt(obj,'bbox_locator',frame,context.size,bool(expansion))
    readonly=[]
    for r in obj.get('readonly',[]):
        b=local_box(r['bbox_native'],frame)
        if b[2]<0 or b[3]<0 or b[0]>=crop.width or b[1]>=crop.height:continue
        readonly.append(dict(id=r.get('id'),owner_id=r.get('owner_id'),bbox=[round(x,1) for x in b],
                             locator=r.get('locator',''),already_assigned=r.get('exclusion_verified',False)))
    if readonly:
        q['readonly_identities']=readonly
        q['instructions'].append('Readonly identities with already_assigned=true have separate reconstruction owners: skip ONLY that identified arrow, never its rectangle. Preserve other target ink crossing a readonly box. already_assigned=false is unverified context and MUST NOT justify dropping visible ink. Do not prove ownership or infer a network architecture.')
        q['residual_hint']=obj.get('residual_hint','')
    images=[('CLEAN: source pixels, output coordinates',folder/'clean.png'),
            ('TARGET: one R1 discovery box, annotations only',folder/'target_boxes.png'),
            ('CONTEXT: source whole image, crop marked',folder/'context.png')]
    if obj.get('missing_only'):
        from coverage_visual import render as render_current
        render_current(crop,obj.get('existing_geometry',[]),frame,folder/'current.png',labels=False)
        images.append(('CURRENT: blue already recovered geometry on gray source',folder/'current.png'))
        q['instructions'].insert(0,'This is append-only omission recovery. Recover only genuinely missing visible portions near TARGET. CURRENT shows existing geometry; do not copy it or repair minor offsets. Existing paths cannot be deleted/replaced. No supported missing ink means status=empty.')
    # Numeric reference boxes and short locators preserve ownership without a stack
    # of opaque colored boxes over the target's original pixels.
    write(folder/'prompt_preview.json',q);write(folder/'readonly.json',readonly)
    return crop,frame,q,images

def block_mask(crop,base,folder,model,config,locks):
    from sam_client import SamClient
    import visual,mask_vectors
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);crop.save(folder/'clean.png')
    with locks['sam']:SamClient(config['sam_url'],config.get('sam_timeout_seconds',900)).segment(folder/'clean.png',folder/'sam',['block_arrow'],config.get('score_threshold',.5))
    predictions=read(folder/'sam/predictions.json')['predictions'];masks={}
    for i,p in enumerate(predictions,1):
        mask=ev.mask(p['mask_path'])
        if mask.shape!=(crop.height,crop.width):raise ValueError('Block SAM coordinate frame mismatch')
        if mask.any():masks['M'+str(i)]=(mask,p['mask_path'])
    if not masks:return None,[]
    visual.masks(crop,[(k,v[0]) for k,v in masks.items()],folder/'masks.png')
    def check(v):
        if not isinstance(v,dict) or not isinstance(v.get('masks'),list) or not set(v['masks'])<=set(masks):raise ValueError('Select known masks or []')
        return v
    q=dict(task='Select only masks covering the ONE target block arrow; return [] when unreliable. Do not add identities or include neighboring shapes.',
           target_bbox=base['target_bbox'],target_description=base.get('identity_locator',''),output_schema={'masks':['M_ID']})
    a=model.ask(folder/'selection',q,[('CLEAN',folder/'clean.png'),('MASK evidence',folder/'masks.png')],check)
    if not a['masks']:return None,[]
    support=np.logical_or.reduce([masks[k][0] for k in a['masks']])
    return support,[masks[k][1] for k in a['masks']]

def recover_target(source,obj,out,model,config,locks,prepare_only=False):
    import block_shapes,shape_render
    from transport.common import write_json as write
    out=Path(out);issues=[];max_expand=config.get('bbox_max_expansions',1)
    for expansion in range(max_expand+1):
        folder=out/('input' if not expansion else 'expanded_input_'+str(expansion))
        with locks['cpu']:crop,frame,q,images=prepare_target(source,obj,folder,config,expansion)
        if prepare_only:return dict(group=obj['id'],state='prepared',rows=[],issues=[])
        if obj.get('render_type')=='block_arrow':
            # Localization already chose the closed-shape pipeline. Do not ask
            # the line-geometry endpoint to classify it again: a format retry
            # there can turn a valid block decision into an unfilled polyline.
            request=out/'block_geometry'/'request'
            answer=dict(status='block',reason='Localization render_type=block_arrow')
            write(out/'dispatch.json',dict(source='localization.render_type',
                render_type='block_arrow',route='closed_shape',line_geometry_requested=False))
        else:
            request=out/'geometry'/('F%03d'%(expansion+1))/'request'
            answer=model.ask(request,q,images,lambda value:validate(value,crop.size));write(out/'initial.json',answer)
        if answer['status']!='needs_context':break
    if answer['status'] in ('empty','needs_context'):
        status=answer['status'];result=dict(group=obj['id'],state='completed' if status=='empty' else 'unresolved',rows=[],issues=[],
            decision=status,reason=answer['reason'],discovery_task={k:v for k,v in obj.items() if k!='existing_geometry'})
        if status!='empty':result['issues'].append(dict(stage='bbox_geometry',target=obj['id'],reason=answer['reason']))
        write(out/'result.json',result);return result
    base=dict(id=obj['id'],parent_id=obj['id'],label=obj.get('display_id',obj['id']),heads=['unknown','unknown'],
        source_discovery_id=obj['id'],source_display_id=obj.get('display_id'),source_masks=[],source_mask_paths=[],
        geometry_request=str(request),local_input=str(folder/'clean.png'),geometry_frame=frame,context_image=str(folder/'context.png'),
        identity_locator=obj.get('locator',''),discovery_task_role=obj.get('task_role'),readonly_targets=q.get('readonly_identities',[]))
    if answer['status']=='block':
        b=local_box(obj['bbox_native'],frame)
        base.update(kind='block_arrow',start=b[:2],via=[[b[2],b[1]],b[2:] ],end=[b[0],b[3]],beziers=[],heads=['none','none'],
                    coordinate_source='bbox_shape_locator',target_bbox=b,locator_only=True)
        support=None
        if config.get('bbox_block_mask_evidence',False):
            try:
                support,paths=block_mask(crop,base,out/'block_evidence',model,config,locks);base['source_mask_paths']=paths
            except Exception as exc:issues.append(dict(stage='block_mask_evidence',error=str(exc),fallback='shape_without_mask'))
        try:rows=[block_shapes.recover(crop,base,support,out/'block_geometry',model)]
        except Exception as exc:
            if support is None and config.get('bbox_block_sam_fallback',True):
                try:support,paths=block_mask(crop,base,out/'block_fallback',model,config,locks);base['source_mask_paths']=paths
                except Exception as sam_exc:issues.append(dict(stage='block_fallback',error=str(sam_exc)))
            if support is not None:
                import mask_vectors
                parts=shape_render.mask_parts(mask_vectors.contour(support))
                rows=[block_shapes.use_parts(base,parts,'model_selected_mask_shape')];rows[0]['mask_shape_parts']=parts
                issues.append(dict(stage='block_geometry',error=str(exc),retained='model_selected_mask'))
            else:raise
        if config.get('block_shape_review_enabled',False):
            import block_shape_review
            rows[0],review_issues=block_shape_review.recover(crop,rows[0],support,out/'block_shape_review',model,config,locks,
                observation_path=out/'block_geometry/answer.json')
            issues.extend(review_issues)
    else:
        rows=sg.geometry(answer,{'R1':base})
        for row in rows:row['geometry_preview_width_px']=config.get('geometry_preview_width_original_px',1.2)*frame['scale'][0]
        rows,_,errors,_=critic.recover(crop,rows,{},out/'geometry_critic',model,config,locks['cpu'],source_view=frame)
        issues.extend(errors)
    for row in rows:preview_style(row,config.get('geometry_preview_width_original_px',1.2)*frame['scale'][0])
    result=dict(group=obj['id'],state='completed_with_issues' if issues else 'completed',
                rows=[fit.transform(r,frame) for r in rows],issues=issues,discovery_task={k:v for k,v in obj.items() if k!='existing_geometry'})
    write(out/'result.json',result);return result




