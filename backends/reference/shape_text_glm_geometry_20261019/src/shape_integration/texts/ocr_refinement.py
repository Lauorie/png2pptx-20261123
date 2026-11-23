"""Conservative source-ink tightening of matched OCR occurrences.

This is localization evidence, not a text/background segmenter. Edge-touching
ink is ambiguous (clipped glyph or crossing graphic) and is deferred, not cut.
No connected component is removed: dots, accents and punctuation must survive.
"""
from copy import deepcopy
import math
import unicodedata
from difflib import SequenceMatcher

import numpy as np
import cv2

from .foreground import contrast_ink


def normalize(text):
    return ''.join(unicodedata.normalize('NFKC', text).split())


def content_match(text, observed, approximate=True):
    """Association evidence only. Never return replacement transcription."""
    target=normalize(text);ocr=normalize(observed)
    if not target or not ocr:return None
    if target==ocr:return {'mode':'exact','edits':0}
    if not approximate:return None
    a=target.casefold();b=ocr.casefold()
    if a==b:return {'mode':'casefold','edits':0}
    # Short labels and mathematical symbols must not match by one-character
    # substitutions. Longer ordinary labels tolerate a small OCR error budget.
    if min(len(a),len(b))<3 or max(len(a),len(b))<4:return None
    if not all(c.isalpha() for c in a+b):return None
    previous=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        current=[i]
        for j,cb in enumerate(b,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(ca!=cb)))
        previous=current
    edits=previous[-1];limit=2 if max(len(a),len(b))>=10 else 1
    if edits>limit or edits/max(len(a),len(b))>.2:return None
    return {'mode':'approximate','edits':edits}


def ordinary(row):
    return row.get('kind')!='formula' and all(r.get('script','normal')=='normal' for r in row.get('runs',[]))


def association(row,detection):
    found=content_match(row['text'],detection['text'],ordinary(row))
    if found:return found
    if row.get('kind')=='formula':
        from .formula_protocol import simple_structure
        try:
            if simple_structure(detection['text'].strip('$'))==simple_structure(row['latex']):
                return {'mode':'formula_structure','edits':0}
        except (ValueError,TypeError,KeyError):pass
    return None


def validate_match(row, detection, size, min_confidence=None):
    match=association(row,detection) if detection else None
    if not match:
        raise ValueError('OCR does not reliably associate with the target transcription')
    if min_confidence is None:min_confidence=.55 if row.get('kind')=='formula' else .8
    confidence = detection['confidence']
    if not math.isfinite(confidence) or confidence < min_confidence:
        raise ValueError('Insufficient OCR recognition confidence')
    b = detection['source_bbox']
    if (len(b) != 4 or not all(math.isfinite(v) for v in b)
            or not 0 <= b[0] < b[2] < size[0]
            or not 0 <= b[1] < b[3] < size[1]):
        raise ValueError('Invalid source-coordinate OCR bbox')
    old = row['bbox_2d']
    distance = math.hypot((b[0]+b[2]-old[0]-old[2])/2,
                          (b[1]+b[3]-old[1]-old[3])/2)
    limit=max(32,min(.15*min(size),max(2.5*(old[3]-old[1]),1.5*math.hypot(old[2]-old[0],old[3]-old[1]))))
    if distance > limit:
        raise ValueError('OCR is too far from the original occurrence')
    if min(b[2],old[2])<=max(b[0],old[0]) or min(b[3],old[3])<=max(b[1],old[1]):
        horizontal=max(0,min(b[2],old[2])-max(b[0],old[0]))
        vertical=max(0,min(b[3],old[3])-max(b[1],old[1]))
        aligned=(horizontal>=.5*min(b[2]-b[0],old[2]-old[0]) or
                 vertical>=.5*min(b[3]-b[1],old[3]-old[1]))
        if match['mode']!='exact' or confidence<.95 or not aligned:
            raise ValueError('Displaced OCR needs unique exact content and axis alignment')


def detection_indices(detection):
    return detection.get('indices',[detection['index']])


def joined_candidates(row,detections):
    """Bounded same-baseline joins, plus attached formula script fragments."""
    target=normalize(row['text']);items=sorted(detections,key=lambda d:d['source_bbox'][0]);out=[]
    def walk(parts,text):
        if len(parts)>=6 or len(out)>=128:return
        a=parts[-1]['source_bbox'];height=a[3]-a[1]+1
        for d in items:
            if d in parts:continue
            b=d['source_bbox'];h=b[3]-b[1]+1
            if b[0]<=a[0] or b[0]-a[2]>max(4,.8*max(height,h)):continue
            if a[2]-b[0]>.35*min(a[2]-a[0]+1,b[2]-b[0]+1):continue
            overlap=max(0,min(a[3],b[3])-max(a[1],b[1]))
            same_line=overlap>=.5*min(height,h)
            script=(row.get('kind')=='formula' and abs((b[1]+b[3]-a[1]-a[3])/2)<=max(height,h))
            if not same_line and not script:continue
            joined=text+normalize(d['text'])
            if not target.startswith(joined):continue
            new=parts+[d]
            if joined==target:
                ids=[i for p in new for i in detection_indices(p)]
                out.append({'index':'join:'+':'.join(map(str,ids)),'indices':ids,'text':joined,
                            'confidence':min(p['confidence'] for p in new),'rotation':0,
                            'source_bbox':[min(p['source_bbox'][0] for p in new),min(p['source_bbox'][1] for p in new),
                                           max(p['source_bbox'][2] for p in new),max(p['source_bbox'][3] for p in new)],
                            'components':new,'join_policy':'same_line_or_attached_script'})
            else:walk(new,joined)
    for d in items:
        text=normalize(d['text'])
        if text and target.startswith(text) and text!=target and d.get('rotation',0)==0:walk([d],text)
    return out


def select_match(row,detections,size,used=()):
    matches=[]
    old=row['bbox_2d'];pad=max(64,3*(old[3]-old[1]))
    nearby=[d for d in detections if not set(detection_indices(d)).intersection(used)
            and old[0]-pad<=(d['source_bbox'][0]+d['source_bbox'][2])/2<=old[2]+pad
            and old[1]-pad<=(d['source_bbox'][1]+d['source_bbox'][3])/2<=old[3]+pad]
    for detection in [*nearby,*joined_candidates(row,nearby)]:
        angle=abs(row.get('rotation_deg',0))%180
        if (45<angle<135)!=(detection.get('rotation',0)%180==90):continue
        try:validate_match(row,detection,size)
        except (ValueError,TypeError,KeyError):continue
        matches.append(detection)
    # Never use recognition confidence to break occurrence ambiguities.
    if len(matches)!=1:raise ValueError(f'Expected one OCR occurrence, got {len(matches)}')
    return matches[0]


def localize(row,detection,source,vertical_bounds=None):
    """Exact OCR uses old ink logic; inexact OCR uses bounded color support.

    Missing glyphs may lie outside the detector box. Expand the search using
    alignment, then locate source pixels near the authoritative text color.
    This is a candidate only: the caller must render complete VLM text and
    apply its atomic CPU score gate. Ambiguous/clipped support is not accepted.
    """
    match=association(row,detection)
    if not match:raise ValueError('Unassociated OCR text')
    from .ocr_ink_support import locate
    support,mask=locate(row,detection,source,vertical_bounds)
    if support['status']=='located':return support,mask
    if detection.get('confidence',1)<.8:return support,mask
    if match['mode'] in ('exact','formula_structure'):
        if match['mode']=='formula_structure':return support,mask
        return tighten(source,detection['source_bbox'])
    b=list(detection['source_bbox']);height=b[3]-b[1]+1;width=b[2]-b[0]+1
    target=normalize(row['text']).casefold();observed=normalize(detection['text']).casefold()
    vertical=abs(row.get('rotation_deg',0))%180
    if 45<vertical<135:
        return {'status':'uncertain','reason':'inexact_rotated_ocr_requires_oriented_support','bbox_2d':None},np.zeros((1,1),bool)
    char_width=width/max(1,len(observed));left=right=0
    for tag,i,j,u,v in SequenceMatcher(None,target,observed,autojunk=False).get_opcodes():
        if tag=='delete':
            if i==0:left+=j-i
            elif j==len(target):right+=j-i
            else:left+=.5*(j-i);right+=.5*(j-i)
    parent=row.get('parent_geometry_prior_bbox',row['bbox_2d'])
    pad=max(3,height*.25)
    search=[max(0,parent[0]-pad,b[0]-3-left*char_width*1.5),
            max(0,parent[1]-pad,b[1]-pad),
            min(source.width-1,parent[2]+pad,b[2]+3+right*char_width*1.5),
            min(source.height-1,parent[3]+pad,b[3]+pad)]
    if vertical_bounds is not None:
        search[1]=max(search[1],vertical_bounds[0])
        search[3]=min(search[3],vertical_bounds[1])
    roi=[math.floor(search[0]),math.floor(search[1]),math.ceil(search[2])+1,math.ceil(search[3])+1]
    colors={r.get('color',row.get('color','#000000')) for r in row.get('runs',[])} or {row.get('color','#000000')}
    record={'status':'uncertain','bbox_2d':None,'association':match,'ocr_bbox':b,'roi':roi,
            'text_authority':'vlm','reason':'unverified_color_support'}
    if len(colors)!=1:
        record['reason']='inexact_multicolor_text_requires_separate_support'
        return record,np.zeros((1,1),bool)
    color=next(iter(colors));rgb=np.asarray(source.crop(roi).convert('RGB'),dtype=np.float32)
    target_color=np.array([int(color[i:i+2],16) for i in (1,3,5)],dtype=np.float32)
    distance=np.linalg.norm(rgb-target_color,axis=2)
    seeds=distance<=80
    # Grow only seeded glyph components to include antialias edges; using
    # dark cores alone would shrink the bbox and therefore the fitted font.
    _,labels=cv2.connectedComponents((distance<=180).astype(np.uint8),8)
    ids=np.unique(labels[seeds]);ids=ids[ids!=0]
    mask=np.isin(labels,ids)
    record.update(color_seed_distance=80,color_support_distance=180)
    yy,xx=np.where(mask)
    if len(xx)<3:record['reason']='insufficient_color_support'
    elif mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any():
        record['reason']='color_support_touches_search_boundary'
    else:
        bounds=[int(xx.min()+roi[0]),int(yy.min()+roi[1]),int(xx.max()+roi[0]),int(yy.max()+roi[1])]
        # Do not squeeze full text into a clipped OCR prefix. This conservative
        # length check complements (not replaces) the actual render score gate.
        required=width*len(target)/max(1,len(observed))*.8
        if len(target)>len(observed) and bounds[2]-bounds[0]+1<required:
            record['reason']='incomplete_width_support'
        else:record.update(status='located',bbox_2d=bounds,reason='bounded_text_color_support_requires_render_gate')
    return record,mask


def tighten(source, box, threshold=.20):
    """Return integer pixel-center ink bounds and an auditable support mask."""
    if not 0 < threshold < 1:
        raise ValueError('Invalid ink threshold')
    if (len(box) != 4 or not all(math.isfinite(v) for v in box)
            or not 0 <= box[0] < box[2] < source.width
            or not 0 <= box[1] < box[3] < source.height):
        raise ValueError('Invalid OCR bounds')
    # Include every pixel center inside the continuous OCR box, no added pad.
    roi = [math.ceil(box[0]), math.ceil(box[1]),
           math.floor(box[2])+1, math.floor(box[3])+1]
    rgb = np.asarray(source.crop(roi).convert('RGB'), dtype=np.float32)
    if min(rgb.shape[:2]) < 3:
        raise ValueError('OCR crop too small for reliable localization')
    ink = contrast_ink(rgb, [0, 0, rgb.shape[1], rgb.shape[0]])
    mask = ink >= threshold
    yy, xx = np.where(mask)
    record = {'ocr_bbox':list(box), 'roi':roi, 'threshold':threshold,
              'ink_pixels':int(mask.sum()), 'status':'uncertain', 'bbox_2d':None}
    if len(xx) < 3:
        record['reason'] = 'insufficient_source_ink'
    elif (mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any()):
        record['reason'] = 'ink_touches_ocr_boundary_possible_clipping_or_graphic'
    else:
        record.update(status='located', reason='all_source_contrast_components_preserved',
                      bbox_2d=[int(xx.min()+roi[0]), int(yy.min()+roi[1]),
                               int(xx.max()+roi[0]), int(yy.max()+roi[1])])
    return record, mask


def proposal(row, box, audit):
    result = deepcopy(row)
    for key in ('layout', 'font_size_px', 'frame_px', 'text_insets_px',
                'glyph_registration', 'resolved_font'):
        result.pop(key, None)
    result['bbox_2d'] = list(box)
    result['ocr_refinement'] = deepcopy(audit)
    color=audit.get('proposed_color')
    if color:
        if row.get('kind')=='formula':
            from .omml import latex_to_omml
            import re
            tree=latex_to_omml(row['latex'],color=row['color'])
            palette=set(tree.xpath('.//*[local-name()="srgbClr"]/@val'))
            if len(palette)>1:raise ValueError('Cannot flatten multicolor formula')
            result['latex']=re.sub(r'#[0-9A-Fa-f]{6}',lambda m:color,result['latex'])
        if len({r.get('color',row['color']) for r in row['runs'] if r['text'].strip()})>1:
            raise ValueError('Cannot flatten multicolor runs')
        result['color']=color
        for run in result['runs']:run['color']=color
    return result
