"""Match one ordinary text occurrence to complete, ordered OCR baselines.

VLM owns transcription; approximate OCR provides localization only. Spatial anchors prevent distant
repeated labels from completing a sequence. Formula scripts are never split.
The caller must render and score the entire replacement as one atomic update.
"""
from copy import deepcopy
import math
import re

from .ocr_refinement import normalize, localize, proposal, content_match
from .reflow import slice_runs


def ordered_lines(a,b):
    x0,y0,x1,y1=a;u0,v0,u1,v1=b
    ah=y1-y0+1;bh=v1-v0+1
    if (v0+v1)/2 <= (y0+y1)/2:return False
    overlap=max(0,min(y1,v1)-max(y0,v0))
    if overlap>.25*min(ah,bh) or v0-y1>max(16,2*max(ah,bh)):return False
    if max(ah,bh)>2.5*min(ah,bh):return False
    horizontal=max(0,min(x1,u1)-max(x0,u0))
    return horizontal>=.2*min(x1-x0+1,u1-u0+1)


def match_lines(row,detections,size,used=()):
    if row.get('kind')=='formula' or any(r.get('script','normal')!='normal' for r in row['runs']):
        return None,{'status':'ineligible_math'}
    if abs(row.get('rotation_deg',0))>5:
        return None,{'status':'unsupported_rotated_multiline'}
    tokens=list(re.finditer(r'\S+',row['text']))
    if not 2<=len(tokens)<=80:return None,{'status':'ineligible_word_count'}
    expected=row.get('visual_line_count',1)
    box=row['bbox_2d'];pad=max(8,min(box[2]-box[0]+1,box[3]-box[1]+1)*.5)
    pool=[]
    for detection in detections:
        if detection['index'] in used or detection.get('rotation',0)!=0:continue
        if not math.isfinite(detection['confidence']) or detection['confidence']<.8:continue
        b=detection['source_bbox']
        if (len(b)!=4 or not all(math.isfinite(v) for v in b)
                or not 0<=b[0]<b[2]<size[0] or not 0<=b[1]<b[3]<size[1]):continue
        cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2
        if not box[0]-pad<=cx<=box[2]+pad or not box[1]-pad<=cy<=box[3]+pad:continue
        if min(b[2],box[2])<=max(b[0],box[0]) or min(b[3],box[3])<=max(b[1],box[1]):continue
        for start in range(len(tokens)):
            for end in range(start+1,len(tokens)+1):
                lo=tokens[start].start();hi=tokens[end-1].end()
                association=content_match(row['text'][lo:hi],detection['text'])
                if association:
                    pool.append({'start':start,'end':end,'span':[lo,hi],'detection':detection,'association':association})
    paths=[];visited=0
    def walk(position,parts):
        nonlocal visited
        visited+=1
        if visited>4096:return
        if position==len(tokens):
            if len(parts)>=2 and (expected<=1 or len(parts)==expected):paths.append(parts)
            return
        if len(parts)>=min(16,len(tokens)) or expected>1 and len(parts)>=expected:return
        for candidate in pool:
            if candidate['start']!=position:continue
            if any(p['detection']['index']==candidate['detection']['index'] for p in parts):continue
            if parts and not ordered_lines(parts[-1]['detection']['source_bbox'],candidate['detection']['source_bbox']):continue
            walk(candidate['end'],parts+[candidate])
    walk(0,[])
    if visited>4096:return None,{'status':'ambiguous_search_budget'}
    if not paths:return None,{'status':'no_complete_ordered_ocr_lines','candidate_segments':len(pool)}
    # Distinct physical solutions must not be resolved by confidence alone.
    solutions={tuple((tuple(round(v,1) for v in p['detection']['source_bbox']),tuple(p['span'])) for p in path):path for path in paths}
    if len(solutions)!=1:return None,{'status':'ambiguous_ocr_line_assignment','solutions':len(solutions)}
    selected=next(iter(solutions.values()))
    return selected,{'status':'matched_multiline','expected_lines':expected,'matched_lines':len(selected),
                     'detection_indices':[p['detection']['index'] for p in selected]}


def propose_lines(row,detections,source,used=()):
    parts,audit=match_lines(row,detections,source.size,used)
    if not parts:return None,audit
    children=[];details=[]
    for index,part in enumerate(parts,1):
        detection=part['detection']
        lo,hi=part['span']
        child=deepcopy(row)
        child.update(id=f"{row['id']}.L{index:02d}",parent_id=row['id'],
                     semantic_unit_id=row.get('semantic_unit_id',row['id']),
                     text=row['text'][lo:hi],runs=slice_runs(row['runs'],lo,hi),
                     visual_line_count=1,line_layout_status='split_baseline',
                     parent_geometry_prior_bbox=list(row['bbox_2d']))
        # Expansion for a missing glyph must never consume the adjacent line.
        upper=0 if index==1 else (parts[index-2]['detection']['source_bbox'][3]+detection['source_bbox'][1])/2
        lower=source.height-1 if index==len(parts) else (detection['source_bbox'][3]+parts[index]['detection']['source_bbox'][1])/2
        record,_=localize(child,detection,source,(upper,lower))
        if record['status']!='located' and part['association']['mode']!='exact':
            return None,{**audit,'status':'inexact_ocr_support_unresolved','line':index,'localization':record}
        # Only exact complete transcriptions may retain a raw detector box.
        bounds=record['bbox_2d'] if record['status']=='located' else detection['source_bbox']
        detail={'character_span':[lo,hi],'ocr_detection':detection,'ink_refinement':record,
                'association':part['association'],'text_authority':'vlm',
                'bbox_policy':'tight_ink' if record['status']=='located' else 'raw_ocr_requires_render_gate'}
        if record.get('proposed_color'):detail['proposed_color']=record['proposed_color']
        children.append(proposal(child,bounds,detail));details.append(detail)
    if normalize(''.join(r['text'] for r in children))!=normalize(row['text']):
        raise ValueError('Multiline split changed content')
    return children,{**audit,'lines':details,'atomic_parent_id':row['id']}


def regroup_fitted(updates,fitted):
    """Keep parent -> child-list mapping; never turn split IDs into parent IDs."""
    by_id={r['id']:r for r in fitted}
    if len(by_id)!=len(fitted):raise ValueError('Duplicate fitted IDs')
    wanted={r['id'] for parts in updates.values() for r in parts}
    if set(by_id)!=wanted:raise ValueError('Missing or extra fitted children')
    return {parent:[by_id[r['id']] for r in parts] for parent,parts in updates.items()}


def conflict_parents(updates,pairs):
    owners={r['id']:parent for parent,parts in updates.items() for r in parts}
    return {owners[i] for pair in pairs for i in pair if i in owners}
