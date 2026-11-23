"""Conservative source-ink registration of independently rendered text glyphs."""
from copy import deepcopy
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ..common import save


def best_match(ink, template):
    """Soft Dice at every translation; foreground-only, not white-background score."""
    h,w=template.shape
    if h>ink.shape[0] or w>ink.shape[1] or template.sum()<3:return None
    dot=cv2.matchTemplate(ink,template,cv2.TM_CCORR)
    integral=cv2.integral(ink*ink)
    mass=integral[h:,w:]-integral[:-h,w:]-integral[h:,:-w]+integral[:-h,:-w]
    score=2*np.maximum(dot,0)/(float((template*template).sum())+mass+1e-6)
    y,x=np.unravel_index(np.argmax(score),score.shape)
    return float(score[y,x]),int(x),int(y)


def supported_cross_axis_repair(row,candidate):
    """Allow an oversized OCR box across the baseline, not a shrunken phrase."""
    return (row.get('kind')!='formula' and candidate.get('score',0)>=.80
            and .75<=candidate.get('flow_coverage',0)<=1.30
            and candidate.get('cross_axis_ratio',1)<.65)


def supported_flow_axis_repair(row,candidate):
    """A complete word template may occupy much less than an overlong box."""
    if (row.get('kind')=='formula' or len(row.get('text','').strip())<3
            or any(r.get('script','normal')!='normal' for r in row.get('runs',[]))
            or candidate.get('score',0)<.82
            or not .30<=candidate.get('flow_coverage',0)<.75
            or not .65<=candidate.get('cross_axis_ratio',0)<=1.30):return False
    a=row['bbox_2d'];b=candidate['bbox']
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0])+1)*max(0,min(a[3],b[3])-max(a[1],b[1])+1)
    return intersection/((b[2]-b[0]+1)*(b[3]-b[1]+1))>=.98


def _register_raw(row, measurement, image, *, min_score=None):
    if min_score is not None and (not math.isfinite(min_score) or not 0<=min_score<=1):
        raise ValueError('Invalid template match threshold')
    path=measurement.get('ink_template') or measurement.get('equation_preview')
    if not path or not Path(path).exists():return row,measurement,{'status':'no_template'}
    box=row['bbox_2d'];bw=box[2]-box[0]+1;bh=box[3]-box[1]+1
    # Search stays local to this occurrence, not elsewhere in the diagram.
    pad=min(12,max(5,min(bw,bh)*.45))
    # Rotated adjacent words are sometimes assigned to opposite columns.
    # Search across one neighboring column, but not far along the text flow.
    px=py=pad
    if 60<=abs(row['rotation_deg'])<=120:px=min(44,max(pad,bw*1.5+6))
    x0=max(0,math.floor(box[0]-px));y0=max(0,math.floor(box[1]-py))
    x1=min(image.width,math.ceil(box[2]+px+1));y1=min(image.height,math.ceil(box[3]+py+1))
    rgb=np.asarray(image.crop((x0,y0,x1,y1)).convert('RGB'),dtype=np.float32)
    from .foreground import contrast_ink
    # Estimate the local background near the proposed glyphs, rather than from
    # a white margin outside their colored node. Do not use a guessed text hue
    # as the foreground direction: font/geometry evidence must be independent.
    ink=contrast_ink(rgb,[math.floor(box[0]-x0)-2,math.floor(box[1]-y0)-2,
                          math.ceil(box[2]-x0)+3,math.ceil(box[3]-y0)+3])
    with Image.open(path) as im:
        template=im.convert('RGBA').getchannel('A')
    angles=[row['rotation_deg']]
    # Explicitly test the opposite reading direction for rotated text only.
    if abs(row['rotation_deg'])>=20:angles.append((row['rotation_deg']+360)%360-180)
    a=math.radians(angles[0]);c=abs(math.cos(a));s=abs(math.sin(a))
    base=min(bw/(c*template.width+s*template.height),bh/(s*template.width+c*template.height))
    candidates=[]
    # Tiny scripted formulas must not shrink to fit one stroke inside a glyph.
    # They may need to grow when OCR omitted a superscript from the rough box.
    # Font height must remain supported by the OCR glyph extent. Searching down
    # to 0.4 made almost any phrase match a few strokes of an unrelated word.
    scales=np.linspace(.70,1.80,56) if row.get('kind')=='formula' else np.linspace(.75,1.25,26)
    for angle in angles:
        best=None
        for multiplier in scales:
            scale=base*multiplier
            w=max(2,round(template.width*scale));h=max(2,round(template.height*scale))
            view=template.resize((w,h),Image.Resampling.LANCZOS).rotate(-angle,expand=True,resample=Image.Resampling.BICUBIC)
            values=np.asarray(view,dtype=np.float32)/255
            narrow=min(values.shape)<.65*min(bw,bh)
            if narrow and row.get('kind')=='formula':continue
            match=best_match(ink,values)
            if match is not None and (best is None or match[0]>best['score']):
                best={'score':match[0],'bbox':[x0+match[1],y0+match[2],x0+match[1]+view.width-1,y0+match[2]+view.height-1],
                      'angle':angle,'scale_multiplier':float(multiplier),
                      'narrow_extent':narrow,'flow_coverage':w/(c*bw+s*bh),
                      'cross_axis_ratio':h/(s*bw+c*bh)}
        if best:candidates.append(best)
    if not candidates:return row,measurement,{'status':'no_fit'}
    best=max(candidates,key=lambda r:r['score']);audit={'candidates':candidates,'before_bbox':box,'before_angle':row['rotation_deg']}
    minimum=min_score if min_score is not None else (.58 if row.get('kind')!='formula' else .65)
    if best['score']<minimum:return row,measurement,{**audit,'status':'low_score'}
    supported=supported_cross_axis_repair(row,best) or supported_flow_axis_repair(row,best)
    if best['narrow_extent'] and not supported:
        return row,measurement,{**audit,'status':'insufficient_cross_axis_support'}
    area=(best['bbox'][2]-best['bbox'][0]+1)*(best['bbox'][3]-best['bbox'][1]+1)
    if area/(bw*bh)<.45 and not supported:return row,measurement,{**audit,'status':'insufficient_extent'}
    if len(candidates)>1:
        scores=sorted((r['score'] for r in candidates),reverse=True)
        if scores[0]-scores[1]<.10:return row,measurement,{**audit,'status':'ambiguous_direction'}
    out=deepcopy(row);metric=deepcopy(measurement)
    out['bbox_2d']=best['bbox'];out['rotation_deg']=best['angle']
    if best['angle']!=row['rotation_deg'] and 'rotated_ink_bounds' in metric:
        l,t,r,b=metric['rotated_ink_bounds'];metric['rotated_ink_bounds']=[-r,-b,-l,-t]
    return out,metric,{**audit,'status':'accepted','after':best}


def _register_at(row, measurement, image, *, min_score=None):
    # Legacy OCR template validation only. The GLM path never calls this,
    # and source-pixel bounding-box tightening has been disconnected.
    return _register_raw(row,measurement,image,min_score=min_score)


def register(row, measurement, image):
    """Compare local-review geometry with its original occurrence anchor.

    A local OCR proposal must not overwrite better pixel-supported geometry.
    The prior is the SAME object ID, never another matching word in the figure.
    """
    current=_register_at(row,measurement,image)
    prior=row.get('geometry_prior_bbox')
    if prior and prior!=row['bbox_2d']:
        candidate=_register_at({**row,'bbox_2d':prior},measurement,image)
        score=lambda record:record.get('after',{}).get('score',0.)
        if (candidate[2]['status']=='accepted' and score(candidate[2])>=.70
                and score(candidate[2])>=score(current[2])+.04):
            out,metric,audit=candidate
            current=(out,metric,{**audit,'anchor':'original_occurrence',
                                'rejected_local':current[2]})
    out,metric,audit=current
    out=deepcopy(out);out.pop('glyph_registration',None)
    if audit['status']=='accepted':
        out['glyph_registration']={'status':audit['status'],
                                   'score':audit['after']['score'],
                                   'anchor':audit.get('anchor','local_review')}
    return out,metric,audit


def refine(rows, measurements, source, folder):
    output=[];metrics=dict(measurements);audit=[]
    cv2.setNumThreads(1)
    with Image.open(source) as image:
        for row in rows:
            updated,metric,record=register(row,metrics[row['id']],image)
            output.append(updated);metrics[row['id']]=metric;audit.append({'id':row['id'],**record})
    # Separate OCR occurrences must not converge onto the same glyph pixels.
    from .local_content import collisions
    conflicts={ident for pair in collisions(output) for ident in pair}
    for i,row in enumerate(rows):
        if row['id'] in conflicts and audit[i]['status']=='accepted':
            output[i]=row;metrics[row['id']]=measurements[row['id']]
            audit[i]['status']='rejected_identity_collision'
    save(folder/'registration.json',audit)
    return output,metrics
