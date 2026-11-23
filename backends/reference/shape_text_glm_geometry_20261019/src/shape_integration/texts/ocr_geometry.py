"""OCR boxes are geometric proposals, never a second transcription authority.

Search bounded unions of adjacent detector boxes; validate the complete native
glyph template against source pixels. Confidence/text can be wrong. Keep runs,
colors, LaTeX and occurrence ownership unchanged. No model requests here.
"""
from copy import deepcopy
import math

from .registration import _register_at
from .ocr_refinement import detection_indices
from ..resource_limits import bounded

OCR_TEMPLATE_MIN_SCORE = .50


def box_union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def iou(a,b):
    area=lambda q:max(0,q[2]-q[0]+1)*max(0,q[3]-q[1]+1)
    inter=max(0,min(a[2],b[2])-max(a[0],b[0])+1)*max(0,min(a[3],b[3])-max(a[1],b[1])+1)
    return inter/max(1,area(a)+area(b)-inter)


def distance(a,b):
    return math.hypot((a[0]+a[2]-b[0]-b[2])/2,(a[1]+a[3]-b[1]-b[3])/2)


def proposals(row, detections, size, used=()):
    old=row['bbox_2d'];diagonal=math.hypot(old[2]-old[0],old[3]-old[1])
    radius=max(32,min(.15*min(size),2*diagonal))
    vertical=45<abs(row.get('rotation_deg',0))%180<135
    near=[]
    for d in detections:
        b=d.get('source_bbox',[])
        if len(b)!=4 or not all(isinstance(v,(float,int)) and math.isfinite(v) for v in b):continue
        if not (0<=b[0]<b[2]<size[0] and 0<=b[1]<b[3]<size[1]):continue
        if set(detection_indices(d)).intersection(used) or distance(old,b)>radius:continue
        if (d.get('rotation',0)%180==90)!=vertical:continue
        near.append(d)
    near=sorted(near,key=lambda d:distance(old,d['source_bbox']))[:12]
    candidates=[dict(d,indices=detection_indices(d)) for d in near]
    seen={tuple(c['indices']) for c in candidates}
    axis=1 if vertical else 0;cross=1-axis
    short=max(4,min(old[2]-old[0]+1,old[3]-old[1]+1))
    for c in candidates:
        if len(candidates)>=96:break
        if len(c['indices'])>=4:continue
        a=c['source_bbox']
        for d in near:
            ids=tuple(sorted(set(c['indices'])|set(detection_indices(d))))
            if len(ids)==len(c['indices']) or ids in seen:continue
            b=d['source_bbox']
            # No transcript-prefix requirement: subscript detector boxes may
            # overlap the base glyph and still belong to the same expression.
            gap=max(0,max(a[axis],b[axis])-min(a[axis+2],b[axis+2]))
            overlap=max(0,min(a[cross+2],b[cross+2])-max(a[cross],b[cross]))
            same_line=overlap>=.35*min(a[cross+2]-a[cross],b[cross+2]-b[cross])
            scripts=row.get('kind')=='formula' and distance(a,b)<2*short
            if gap>.8*short or not (same_line or scripts):continue
            united=box_union([a,b])
            if united[axis+2]-united[axis]>max(12,1.8*(old[axis+2]-old[axis])):continue
            seen.add(ids)
            candidates.append({'index':ids[0],'indices':list(ids),'text':'',
                'confidence':min(c.get('confidence',0),d.get('confidence',0)),
                'rotation':d.get('rotation',0),'source_bbox':united,
                'proposal_type':'adjacent_detector_union'})
            if len(candidates)>=96:break
    # Missing OCR is explicit. A prior-only template hypothesis is NOT reported
    # as an OCR success and is subject to the same complete-glyph checks.
    candidates.append({'index':-1,'indices':[],'text':'','confidence':0,
        'rotation':90 if vertical else 0,'source_bbox':old,'proposal_type':'vlm_prior_fallback'})
    return candidates


def select_without_template(row,detections,source,used=(),neighbors=()):
    """Use local OCR geometry/ink, not a font-template acceptance threshold.

    OCR transcription is an association hint only; full VLM content and style
    stay unchanged. A proposal still needs the caller's native-render gate.
    """
    from .ocr_refinement import association,tighten
    old=row['bbox_2d'];diagonal=max(1,math.hypot(old[2]-old[0],old[3]-old[1]))
    candidates=[]
    for candidate in proposals(row,detections,source.size,used):
        if not candidate['indices']:continue
        box=candidate['source_bbox'];own=distance(old,box)
        if any(n['id']!=row['id'] and distance(n['bbox_2d'],box)+5<own for n in neighbors):continue
        overlap=iou(old,box)
        match=association(row,candidate)
        # A displaced box needs transcription support; a bad OCR transcript
        # may still provide geometry when it overlaps this known occurrence.
        if overlap<.15 and not match:continue
        score=overlap-.25*own/diagonal
        candidates.append((bool(match),score,candidate))
    candidates.sort(key=lambda c:(c[0],c[1]),reverse=True)
    if candidates:
        matched,score,candidate=candidates[0]
        different=[c for c in candidates[1:] if c[0]==matched and iou(c[2]['source_bbox'],candidate['source_bbox'])<.60]
        if different and score-different[0][1]<.08:
            raise ValueError('Ambiguous OCR geometry without template; retained prior')
        box=candidate['source_bbox'];ids=list(candidate['indices'])
        proposal_type=candidate.get('proposal_type','ocr_detector')
    else:
        candidate={'text':''};box=old;ids=[];proposal_type='prior_ink_only_no_ocr'
    support,_=tighten(source,box)
    if support['status']=='located':box=support['bbox_2d']
    elif not ids or support.get('ink_pixels',0)<3:
        raise ValueError('No usable OCR/ink geometry without template')
    for d in detections:
        if iou(box,d['source_bbox'])>.80:ids.extend(detection_indices(d))
    return {'status':'located','bbox_2d':list(box),'text_authority':'vlm',
        'reason':'geometric_proposal_without_font_template','template_validation':False,
        'detection_indices':sorted(set(ids)), 'proposal_type':proposal_type,
        'ocr_text_hint':candidate['text'],'preserve_runs_and_colors':True,
        'source_ink_support':support,'requires_native_render_gate':True}


@bounded('cpu')
def select(row, detections, source, used=(), neighbors=(), *, template_validation=True):
    if not template_validation:return select_without_template(row,detections,source,used,neighbors)
    metric=row.get('layout',{}).get('measurement',{})
    if not (metric.get('ink_template') or metric.get('equation_preview')):
        raise ValueError('No independently rendered full glyph template')
    verified=[];reasons=[]
    for candidate in proposals(row,detections,source.size,used):
        fitted,_,audit=_register_at({**row,'bbox_2d':candidate['source_bbox']},metric,source,
                                   min_score=OCR_TEMPLATE_MIN_SCORE)
        if audit['status']!='accepted':
            reasons.append(audit['status']);continue
        best=audit['after'];box=best['bbox']
        if fitted['rotation_deg']!=row['rotation_deg']:continue
        # Use the same floor inside registration and here. Geometry/identity
        # checks and the caller's final native-render gate remain independent.
        if best['score']<OCR_TEMPLATE_MIN_SCORE:continue
        own=distance(row['bbox_2d'],box)
        if any(n['id']!=row['id'] and distance(n['bbox_2d'],box)+max(5,.2*(box[3]-box[1]))<own
               for n in neighbors):continue
        record={'status':'located','bbox_2d':box,'text_authority':'vlm',
            'reason':'complete_native_template_at_geometric_proposal',
            'template_score':best['score'],'template_min_score':OCR_TEMPLATE_MIN_SCORE,
            'detection_indices':candidate['indices'],
            'proposal_type':candidate.get('proposal_type','ocr_detector'),
            'ocr_text_hint':candidate['text'],'ocr_confidence_not_a_gate':True,
            'preserve_runs_and_colors':True,'registration':audit}
        verified.append(record)
    verified.sort(key=lambda r:r['template_score'],reverse=True)
    unique=[]
    for record in verified:
        if not any(iou(record['bbox_2d'],p['bbox_2d'])>=.75 for p in unique):unique.append(record)
    if not unique:
        raise ValueError('No complete glyph-supported geometry; '+','.join(sorted(set(reasons))))
    if len(unique)>1 and unique[0]['template_score']-unique[1]['template_score']<.06:
        raise ValueError('Multiple pixel-supported occurrences; no confident ownership')
    result=unique[0]
    # Consume every detector fragment mostly inside the selected glyph box,
    # including duplicate resolutions, not only the winning union's IDs.
    consumed=set(result['detection_indices'])
    for d in detections:
        b=d['source_bbox'];q=result['bbox_2d']
        inter=max(0,min(b[2],q[2])-max(b[0],q[0]))*max(0,min(b[3],q[3])-max(b[1],q[1]))
        if inter/max(1,(b[2]-b[0])*(b[3]-b[1]))>.65:consumed.update(detection_indices(d))
    result['detection_indices']=sorted(consumed)
    # Uniform-color correction needs independent full-glyph source support.
    # Mixed-color runs/expressions remain intact; geometry can still improve.
    from .ocr_ink_support import locate
    support,_=locate(row,{'source_bbox':result['bbox_2d'],'text':row['text']},source)
    if (support.get('status')=='located' and support.get('proposed_color')
            and iou(result['bbox_2d'],support['bbox_2d'])>=.70):
        result['proposed_color']=support['proposed_color']
        result['color_support']=support
        result['preserve_runs_and_colors']=False
    return result
