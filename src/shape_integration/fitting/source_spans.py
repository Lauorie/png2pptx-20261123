"""Bounded source crops for long unsupported straight spans in one outline.

These are observations for the existing critic, never automatic deletions.
Real dash gaps, source occlusion and unstroked boundaries remain source judgements.
"""
from pathlib import Path
import math
import cv2
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt,map_coordinates
from ..common import save,read,sha
from .paths import command_strokes


def inspect(local,job,folder,*,limit=1):
    parts=local['parts']
    if (len(parts)!=1 or parts[0]['status']!='ok' or not parts[0]['commands'] or
            parts[0]['commands'][-1][0]!='Z' or job.get('proposed_type')=='projected_3d'):
        return [],[]
    folder=Path(folder)
    with Image.open(folder/'clean.png') as opened:clean=opened.convert('RGB')
    rgb=np.asarray(clean);edge=np.maximum.reduce([cv2.Canny(rgb[:,:,i].copy(),20,60) for i in range(3)])
    distance=distance_transform_edt(edge==0)
    f=job['frame'];crop=f.get('source_crop_half_open');scale=max(f['actual_crop_input_size_px'][i]/(crop[i+2]-crop[i]) for i in (0,1)) if crop else 1.
    rows=[];position=None;start=None
    for cmd in parts[0]['commands']:
        endpoint=cmd[1:3] if cmd[0]=='L' else start if cmd[0]=='Z' else None
        if endpoint is not None and position is not None:
            a=np.asarray(position,float);b=np.asarray(endpoint,float);length=float(np.linalg.norm(b-a))
            if length>=30*scale:
                t=np.linspace(0,1,max(3,math.ceil(length/scale)));xy=a[None]+t[:,None]*(b-a)
                missing=map_coordinates(distance,[xy[:,1],xy[:,0]],order=1,mode='constant',cval=0)>3*scale
                changes=np.flatnonzero(np.diff(np.r_[False,missing,False].astype(int))).reshape(-1,2)
                for lo,hi in changes:
                    run=(hi-lo)*length/max(1,len(t)-1)
                    if run>=max(18*scale,.1*length):rows.append((run,xy[lo],xy[hi-1],run/length))
        if cmd[0]=='M':position=start=cmd[1:3]
        elif cmd[0]=='L':position=cmd[1:3]
        elif cmd[0]=='C':position=cmd[-2:]
        elif cmd[0]=='Z':position=start
    images=[];checks=[]
    for i,(run,a,b,fraction) in enumerate(sorted(rows,key=lambda r:-r[0])[:limit]):
        pad=12*scale;lo=np.maximum(0,np.floor(np.minimum(a,b)-pad)).astype(int)
        hi=np.minimum(clean.size,np.ceil(np.maximum(a,b)+pad+1)).astype(int)
        if np.any(hi<=lo):continue
        box=(*lo,*hi);source=clean.crop(box);overlay=source.copy();draw=ImageDraw.Draw(overlay)
        for stroke in command_strokes(parts[0]['commands']):
            draw.line([(x-lo[0],y-lo[1]) for x,y in stroke],fill='#d10070',width=max(1,round(scale)))
        size=tuple(max(2,round(v*min(3.,768/max(source.size)))) for v in source.size)
        for label,image in [('CLEAN',source),('fitted contour annotation',overlay)]:
            path=folder/f'source_span_{i}_{"clean" if label=="CLEAN" else "fit"}.png'
            image.resize(size,Image.Resampling.LANCZOS).save(path);images.append((f'F1 source span S{i}: {label}',path))
        delta=b-a
        direction=('vertical' if abs(delta[0])<.2*abs(delta[1]) else
                   'horizontal' if abs(delta[1])<.2*abs(delta[0]) else
                   'downward-right diagonal' if delta[0]*delta[1]>0 else 'upward-right diagonal')
        checks.append({'part':'F1','span':f'S{i}','direction':direction,
                       'unsupported_fraction_of_segment':round(fraction,3)})
    save(folder/'source_span_checks.json',{'checks':checks,'automatic_geometry_changes':False})
    return checks,images


def observe_source(runtime,folder,span,source,options):
    """One independent CLEAN-only observation per object, without fit priming."""
    folder=Path(folder);receipt=folder.parent/'source_only_observation.json';identity={'image':sha(source),'direction':span['direction']}
    if receipt.exists():
        old=read(receipt)
        if old['identity']==identity:return old['observation']
        return {'visible_line':'uncertain','source_evidence':'The one source-only observation for this object was already used on another span.'}
    def validate(value):
        if not isinstance(value,dict) or value.get('visible_line') not in ('yes','no','uncertain') or not isinstance(value.get('source_evidence'),str) or not value['source_evidence'].strip():
            raise ValueError('Source observation requires visible_line and brief concrete source_evidence')
        return {k:value[k] for k in ('visible_line','source_evidence')}
    prompt={'task':'Inspect only the supplied CLEAN source crop. Is there a repeated or continuous '
        +span['direction']+' stroke or visible fill boundary along the CENTRAL band through the middle half '
        'of this crop? Crossing strokes in other directions and marks only near its ends do not count. '
        'Do not complete a diagram from a shape name, infer a hidden continuation, or imagine dash gaps. '
        'Report only visibly painted source pixels. No prediction overlay is supplied.',
        'output':{'visible_line':'yes|no|uncertain','source_evidence':'one brief visible-pixel observation'}}
    try:
        observation=runtime.for_stage(shape_schema_retries=0).call(folder/'source_only_calls','observe',
            'Observe the CLEAN image independently of any reconstruction. Return concise JSON only.',
            prompt,[('CLEAN source crop, no fitted paths',source)],validate,min(8192,options['check_max_tokens']))
    except (ValueError,RuntimeError) as exc:
        observation={'visible_line':'uncertain','source_evidence':'Independent source observation unavailable: '+str(exc)}
    save(receipt,{'identity':identity,'observation':observation,'max_model_calls':1})
    return observation
