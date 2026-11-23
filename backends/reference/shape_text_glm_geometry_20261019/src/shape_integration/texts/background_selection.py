"""Choose white/transparent native frame fill using actual, contextual CPU renders.

Never infer missing non-text geometry from the source. A text-only render on a
white slide cannot identify these two paints; it must keep the transparent tie.
"""
from copy import deepcopy
import math
import subprocess

import numpy as np
from PIL import Image

from ..common import save, sha


def roi(row, size):
    """Include both glyph bounds and the entire rotated native paint footprint."""
    x,y,w,h=row['frame_px'];cx=x+w/2;cy=y+h/2
    angle=math.radians(row.get('rotation_deg',0))
    dx=(abs(w*math.cos(angle))+abs(h*math.sin(angle)))/2
    dy=(abs(w*math.sin(angle))+abs(h*math.cos(angle)))/2
    a,b,c,d=row['bbox_2d']
    return [max(0,math.floor(min(a,cx-dx)-3)),max(0,math.floor(min(b,cy-dy)-3)),
            min(size[0],math.ceil(max(c,cx+dx)+3)),min(size[1],math.ceil(max(d,cy+dy)+3))]


def batches(rows, size):
    """Disjoint supports can be evaluated in the same render without interactions."""
    groups=[]
    for row in rows:
        box=roi(row,size)
        for group in groups:
            if all(min(box[2],other[2])<=max(box[0],other[0]) or
                   min(box[3],other[3])<=max(box[1],other[1]) for _,other in group):
                group.append((row['id'],box));break
        else:groups.append([(row['id'],box)])
    return groups


def score_pair(source, transparent, white, box):
    arrays=[np.asarray(im.crop(box).convert('RGB'),dtype=np.float64)/255
            for im in (source,transparent,white)]
    ref,a,b=arrays
    if ref.size==0:raise ValueError('Empty background scoring ROI')
    sa=1-float(np.mean((ref-a)**2));sb=1-float(np.mean((ref-b)**2))
    return {'transparent_score':sa,'white_score':sb,
            'selected':'white' if sb>sa+1e-12 else 'transparent',
            'different_pixels':int(np.any(a!=b,axis=2).sum()),'roi':box,
            'metric':'1-MSE_RGB_0_1_same_ROI','tie_policy':'transparent'}


def select(rows, source, folder, slots, opt, base=None):
    from .export import write
    from .equations import render_preview
    selected=deepcopy(rows)
    if opt.get('background_policy') == 'transparent_only':
        folder.mkdir(parents=True,exist_ok=True)
        for row in selected:
            row.update(background_fill=None,background_evidence='',background_authority='transparent_default')
            row['background_selection']={'policy':'transparent_only','cpu_selection':False}
        save(folder/'selection.json',{'policy':'transparent_only','renders':0,
            'items':[{'id':r['id'],'background_fill':None} for r in selected]})
        return selected
    if opt.get('background_policy') == 'vlm_two_stage':
        from .background import validate
        folder.mkdir(parents=True,exist_ok=True)
        for row in selected:
            row.update(validate(row))
            row['background_selection']={'policy':'vlm_geometry_stage','cpu_selection':False}
        save(folder/'selection.json',{'policy':'vlm_geometry_stage','renders':0,
            'items':[{'id':r['id'],'background_fill':r.get('background_fill'),
                      'background_evidence':r.get('background_evidence','')} for r in selected]})
        return selected
    for row in selected:
        row.update(background_fill=None,background_evidence='')
        row.pop('background_selection',None)
    folder.mkdir(parents=True,exist_ok=True)
    with Image.open(source) as im:reference=im.convert('RGB')
    size=list(reference.size);audit=[]
    if base is None:
        # Both native candidates flatten to exactly the same white-slide image.
        for row in selected:
            record={'id':row['id'],'selected':'transparent','identifiable':False,
                    'reason':'white_and_transparent_identical_without_nontext_base',
                    'transparent_score':None,'white_score':None}
            row['background_selection']=record;audit.append(record)
        save(folder/'selection.json',{'policy':'cpu_white_or_transparent_v1','renders':0,
             'source_sha256':sha(source),'base_sha256':None,'items':audit})
        return selected
    if len({r['id'] for r in selected})!=len(selected):raise ValueError('Duplicate text IDs')
    by_id={r['id']:r for r in selected};render_count=0
    for index,group in enumerate(batches(selected,size)):
        root=folder/f'batch_{index:03d}'
        try:
            images=[]
            for label in ('transparent','white'):
                trial=deepcopy(selected);ids={ident for ident,_ in group}
                for row in trial:
                    if row['id'] in ids:
                        row.update(background_fill='#FFFFFF' if label=='white' else None,
                                   background_evidence='CPU candidate' if label=='white' else '')
                target=root/(label+'.pptx')
                write(trial,size,target,base)
                with slots:render_preview(target,root/label/'native_render',size,opt['render_timeout_seconds'])
                render_count+=1
                with Image.open(root/label/'native.png') as im:images.append(im.convert('RGB'))
            for ident,box in group:
                record={'id':ident,'batch':index,**score_pair(reference,*images,box)}
                record['identifiable']=record['different_pixels']>0
                if record['selected']=='white':
                    by_id[ident].update(background_fill='#FFFFFF',
                        background_evidence='CPU native-render 1-MSE exceeds transparent on the same source ROI')
                by_id[ident]['background_selection']=record;audit.append(record)
        except (ValueError,RuntimeError,OSError,TimeoutError,subprocess.TimeoutExpired) as exc:
            # A paint comparison must not invalidate usable text geometry/content.
            for ident,_ in group:
                record={'id':ident,'batch':index,'selected':'transparent','identifiable':False,
                        'reason':'candidate_render_failed','error':str(exc)}
                by_id[ident]['background_selection']=record;audit.append(record)
    save(folder/'selection.json',{'policy':'cpu_white_or_transparent_v1','renders':render_count,
         'source_sha256':sha(source),'base_sha256':sha(base),'items':audit})
    return selected
