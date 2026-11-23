"""Estimate residual translation of actual rendered text against source pixels.

No resizing, font replacement or OCR bbox is used as ground truth. This is a
local-registration estimate, not a human-annotated localization error. Weak,
ambiguous, clipped and neighborhood-contaminated matches are excluded.
"""
import argparse
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'.deps'),str(ROOT/'src'),str(ROOT/'vendor')]
import cv2
import numpy as np
from PIL import Image
from shape_integration.common import read,save,sha
from shape_integration.texts.foreground import contrast_ink


def translation(ink,template,anchor,radius):
    """Return translation of template TO source; fixed scale and orientation."""
    h,w=template.shape
    if h>ink.shape[0] or w>ink.shape[1] or template.sum()<4:return None
    dot=cv2.matchTemplate(ink.astype(np.float32),template.astype(np.float32),cv2.TM_CCORR)
    integral=cv2.integral(ink*ink)
    mass=integral[h:,w:]-integral[:-h,w:]-integral[h:,:-w]+integral[:-h,:-w]
    scores=np.clip(2*np.maximum(dot,0)/(float((template*template).sum())+mass+1e-6),0,1)
    yy,xx=np.indices(scores.shape)
    valid=(np.abs(xx-anchor[0])<=radius)&(np.abs(yy-anchor[1])<=radius)
    scores[~valid]=-1
    y,x=np.unravel_index(np.argmax(scores),scores.shape)
    if scores[y,x]<0:return None
    distinct=valid&((xx-x)**2+(yy-y)**2>9)
    second=float(scores[distinct].max()) if distinct.any() else -1.
    dx,dy=int(x-anchor[0]),int(y-anchor[1])
    return {'dx_to_source':dx,'dy_to_source':dy,'distance_px':math.hypot(dx,dy),
        'fixed_scale_dice':float(scores[y,x]),'alternative_margin':float(scores[y,x])-second,
        'search_boundary':abs(dx)>=radius or abs(dy)>=radius}


def measure(row,rows,source,rendered):
    record={'id':row['id'],'text':row['text'],'kind':row.get('kind','text'),'trusted':False}
    if abs(row.get('rotation_deg',0))>1e-6:return {**record,'reason':'rotated_object_not_supported'}
    b=row['bbox_2d'];x,y,w,h=row['frame_px']
    # A small extraction margin catches antialiasing outside the target bbox.
    box=[max(0,math.floor(min(b[0],x))-2),max(0,math.floor(min(b[1],y))-2),
         min(source.width,math.ceil(max(b[2]+1,x+w))+2),
         min(source.height,math.ceil(max(b[3]+1,y+h))+2)]
    rgb=np.asarray(rendered.crop(box).convert('RGB'),dtype=np.float32)
    pred=contrast_ink(rgb,[0,0,rgb.shape[1],rgb.shape[0]])
    ys,xs=np.where(pred>.20)
    if len(xs)<4:return {**record,'reason':'insufficient_rendered_ink'}
    tight=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]
    if tight[0]==0 or tight[1]==0 or tight[2]==rgb.shape[1] or tight[3]==rgb.shape[0]:
        return {**record,'reason':'rendered_ink_touches_extraction_boundary'}
    bounds=[tight[0]+box[0],tight[1]+box[1],tight[2]+box[0],tight[3]+box[1]]
    for other in rows:
        if other['id']==row['id']:continue
        q=other['bbox_2d']
        if min(bounds[2],q[2])-max(bounds[0],q[0])>0 and min(bounds[3],q[3])-max(bounds[1],q[1])>0:
            return {**record,'reason':'overlapping_neighbor_could_contaminate_template','neighbor':other['id']}
    template=pred[tight[1]:tight[3],tight[0]:tight[2]]
    radius=min(32,max(8,math.ceil(template.shape[0]*.75)))
    roi=[max(0,bounds[0]-radius),max(0,bounds[1]-radius),
         min(source.width,bounds[2]+radius),min(source.height,bounds[3]+radius)]
    raw=np.asarray(source.crop(roi).convert('RGB'),dtype=np.float32)
    anchor=[bounds[0]-roi[0],bounds[1]-roi[1]]
    ink=contrast_ink(raw,[max(0,anchor[0]-2),max(0,anchor[1]-2),
        anchor[0]+template.shape[1]+2,anchor[1]+template.shape[0]+2])
    found=translation(ink,template,anchor,radius)
    if found is None:return {**record,'reason':'no_matching_window'}
    reason=('weak_glyph_match' if found['fixed_scale_dice']<.70 else
            'ambiguous_translation' if found['alternative_margin']<.02 else
            'search_boundary' if found['search_boundary'] else 'reliable_local_translation')
    return {**record,**found,'trusted':reason=='reliable_local_translation','reason':reason,
        'rendered_ink_bbox':bounds,'search_radius_px':radius,
        'pred_minus_source_dx':-found['dx_to_source'],'pred_minus_source_dy':-found['dy_to_source']}


def summarize(records):
    accepted=[r for r in records if r['trusted']]
    out={'total':len(records),'trusted':len(accepted),'excluded':len(records)-len(accepted)}
    if accepted:
        dist=[r['distance_px'] for r in accepted]
        out.update(mean_distance_px=float(np.mean(dist)),median_distance_px=float(np.median(dist)),
            p90_distance_px=float(np.percentile(dist,90)),max_distance_px=max(dist),
            mean_abs_dx=float(np.mean([abs(r['pred_minus_source_dx']) for r in accepted])),
            mean_abs_dy=float(np.mean([abs(r['pred_minus_source_dy']) for r in accepted])))
    return out


def ocr_center_offsets(rows,records,run):
    """Independent detector-box center proxy; does NOT gate by font similarity."""
    from shape_integration.texts.ocr_refinement import normalize,validate_match
    from shape_integration.texts.ocr_geometry import iou
    detections=[]
    for job in read(run/'ocr_manifest.json')['jobs']:
        path=Path(job['output'])/'report.json'
        if path.exists():detections.extend(read(path)['detections'])
    by_id={r['id']:r for r in records};result=[]
    for row in rows:
        item={'id':row['id'],'text':row['text'],'trusted':False}
        if row.get('kind')!='text':continue
        bounds=by_id[row['id']].get('rendered_ink_bbox')
        if not bounds:
            result.append({**item,'reason':'No isolated rendered ink bbox'});continue
        candidates=[]
        old=row['bbox_2d'];center=lambda b:((b[0]+b[2])/2,(b[1]+b[3])/2)
        for d in detections:
            if normalize(d['text'])!=normalize(row['text']):continue
            try:validate_match(row,d,read(run/'final/scene.json')['source_size'],min_confidence=.90)
            except ValueError:continue
            candidates.append(d)
        candidates.sort(key=lambda d:math.dist(center(old),center(d['source_bbox'])))
        if not candidates:
            result.append({**item,'reason':'No nearby high-confidence exact OCR transcription'});continue
        first=candidates[0]['source_bbox']
        duplicates=[d['source_bbox'] for d in candidates if iou(first,d['source_bbox'])>.5]
        others=[d for d in candidates if iou(first,d['source_bbox'])<=.5]
        if others and math.dist(center(old),center(others[0]['source_bbox']))-math.dist(center(old),center(first))<10:
            result.append({**item,'reason':'Ambiguous OCR occurrence'});continue
        ref=np.median(duplicates,axis=0).tolist()
        # OCR boxes use inclusive pixel coordinates; actual ink bounds are exclusive.
        actual=[(bounds[0]+bounds[2]-1)/2,(bounds[1]+bounds[3]-1)/2]
        target=center(ref);dx,dy=actual[0]-target[0],actual[1]-target[1]
        result.append({**item,'trusted':True,'distance_px':math.hypot(dx,dy),
            'pred_minus_source_dx':dx,'pred_minus_source_dy':dy,'ocr_bbox':ref,
            'rendered_ink_bbox':bounds,'reason':'Exact OCR content and unique local occurrence'})
    return {'note':'Detector bbox center vs rendered ink bbox center; OCR padding is an uncertainty. No glyph-similarity gate.',
            'ordinary_text':summarize(result),'objects':result}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('image','run','output'):p.add_argument('--'+arg,type=Path,required=True)
    args=p.parse_args()
    with Image.open(args.image) as im:source=im.convert('RGB')
    output={'method':'fixed_scale_whole_rendered_glyph_translation_v1',
        'note':'Estimated residual translation, not GT error; no resizing. Equal weight per reliable object.',
        'source_size':list(source.size),'source_sha256':sha(args.image),'results':{}}
    for stage in ('baseline','final'):
        rows=read(args.run/stage/'scene.json')['texts']
        with Image.open(args.run/stage/'native.png') as im:rendered=im.convert('RGB')
        if rendered.size!=source.size:raise ValueError('Source/render pixel grids differ')
        records=[measure(r,rows,source,rendered) for r in rows]
        output['results'][stage]={'all_text_layer':summarize(records),
            'ordinary_text':summarize([r for r in records if r['kind']=='text']),
            'formulas':summarize([r for r in records if r['kind']=='formula']),'objects':records}
    common={r['id'] for r in output['results']['baseline']['objects'] if r['trusted']} & {
            r['id'] for r in output['results']['final']['objects'] if r['trusted']}
    output['paired_reliable']={stage:summarize([r for r in output['results'][stage]['objects'] if r['id'] in common])
                              for stage in ('baseline','final')}
    output['ocr_center_proxy']=ocr_center_offsets(read(args.run/'final/scene.json')['texts'],
                                                 output['results']['final']['objects'],args.run)
    save(args.output,output)
    import json
    print(json.dumps({s:{k:v for k,v in result.items() if k!='objects'} for s,result in output['results'].items()},indent=2))
    print('PAIRED',json.dumps(output['paired_reliable']))
    print('OCR_CENTER_PROXY',json.dumps(output['ocr_center_proxy']['ordinary_text']))
    print('DIGITS',json.dumps([r for r in output['results']['final']['objects'] if r['text'] in ('0','3','5')]))


if __name__=='__main__':main()
