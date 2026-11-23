"""Closed block-arrow geometry through the existing shape fitting protocol."""
import copy
from pathlib import Path
import numpy as np
from PIL import ImageDraw
from shape_core.fitting import contracts,adapter
from shape_core.fitting.paths import sample_commands
import mask_vectors,shape_render,visual
from transport.common import write_json as write


def use_parts(row,parts,source):
    row=copy.deepcopy(row);row['shape_parts']=parts
    contours=[sample_commands(p['commands'],32) for p in parts]
    points=max(contours,key=len)
    row.update(start=points[0],via=points[1:-1],end=points[-1],contours=contours,
               beziers=[],heads=['none','none'],start_head=False,end_head=False,
               start_head_type=None,end_head_type=None,geometry_method=source,coordinate_source=source)
    return row


def recover(crop,row,support,out,model,*,feedback=()):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);row=copy.deepcopy(row)
    if support is not None and support.any():
        row['mask_shape_parts']=shape_render.mask_parts(mask_vectors.contour(support))
    else:row['mask_shape_parts']=[]
    visual.masks(crop,[(row['id'],support if support is not None else np.zeros((crop.height,crop.width),bool))],out/'mask.png')
    crop.save(out/'clean.png')
    points=np.array([row['start'],*row.get('via',[]),row['end']])
    box=row.get('target_bbox',[*points.min(axis=0).tolist(),*points.max(axis=0).tolist()])
    unlocated=row.get('target_bbox_source')=='original_observation'
    image=crop.copy()
    if not unlocated:ImageDraw.Draw(image).rectangle(box,outline='#C00080',width=2)
    image.save(out/'locator.png')
    from block_shape_protocol import make_prompt,VERSION
    prompt=make_prompt(crop.size,box,row,feedback)
    if row.get('readonly_targets'):
        prompt['readonly_identities']=row['readonly_targets']
        prompt['instructions'].append('Skip only the identified arrows with already_assigned=true; their boxes are NOT exclusion masks. Keep this target shape where it crosses those boxes. Unverified references cannot justify removal.')
    if unlocated:
        prompt.pop('bbox');prompt['task']='Recover ONLY the closed arrow shape identified by target_description using the shape geometry protocol.'
        prompt['instructions'].append('No original target box is available; TARGET BBOX is intentionally unmarked. Locate only the described shape, not all neighbors.')
    images=[('CLEAN',out/'clean.png'),('TARGET BBOX',out/'locator.png'),('MASK evidence',out/'mask.png')]
    if row.get('context_image') and Path(row['context_image']).exists():images.append(('CONTEXT whole figure',Path(row['context_image'])))
    answer=model.ask(out/'request',prompt,images,
                     lambda a:contracts.normalize_observation(a,row['id'],crop.size))
    write(out/'answer.json',answer)
    if answer['status'] not in ('ok','partial') or not answer['parts']:
        raise ValueError('Shape geometry unresolved: '+answer.get('reason',''))
    table={p['id']:p['xy'] for p in answer['points']}
    parts=[dict(id=p['id'],**adapter.fit_part(p,table)) for p in answer['parts']]
    if any(p['commands'][-1][0]!='Z' for p in parts):raise ValueError('Block arrow parts must be closed')
    result=use_parts(row,parts,'glm_shape_geometry')
    result['shape_observation_protocol']=VERSION
    result['shape_observation_status']=answer['status']
    result.pop('locator_only',None)
    write(out/'geometry.json',result)
    return result
