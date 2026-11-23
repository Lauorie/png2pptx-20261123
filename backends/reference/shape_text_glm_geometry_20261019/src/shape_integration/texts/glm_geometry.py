"""Replace the OCR detector with ID-bound GLM geometry proposals.

Content and style stay frozen after crop review. Model coordinates are only
candidates: the caller performs native rendering and CPU score checks. Model
boxes are used verbatim; no source-ink or font-template tightening is applied.
No PaddleOCR imports, fabricated model confidence or source-image edits.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from PIL import Image

from ..common import read, save, sha, valid_box
from ..styling.views import resample_crop
from . import model
from .controller import local_hints, mapped_box
from .multiline_retry import validate as validate_lines
from .ocr_geometry import distance
from .ocr_refinement import proposal
from .reflow import slice_runs


def validate(raw, rows, size):
    if not isinstance(raw,dict) or set(raw)!={'items'} or not isinstance(raw['items'],list):
        raise ValueError('Return items array only')
    by_id={r['id']:r for r in rows};items=raw['items']
    if (any(not isinstance(i,dict) for i in items) or len(items)!=len(rows)
            or {i.get('id') for i in items}!=set(by_id)):
        raise ValueError('Every supplied ID must occur exactly once')
    result=[]
    for item in items:
        if set(item)!={'id','status','reason','lines'}:
            raise ValueError('Only id/status/reason/lines may change; no styles or new content')
        if item['status'] not in ('located','ambiguous') or not isinstance(item['reason'],str):
            raise ValueError('Invalid status/reason')
        if item['status']=='ambiguous':
            if item['lines']!=[]:raise ValueError('Ambiguous targets require lines=[]')
            result.append(deepcopy(item));continue
        row=by_id[item['id']]
        expected=1 if row.get('kind')=='formula' else row.get('visual_line_count',1)
        lines=item['lines']
        if not isinstance(lines,list) or len(lines)!=expected:
            raise ValueError('Return exactly one bbox per known visual baseline; never split formulas')
        # Validate the original string partition, not a second transcription.
        checked=validate_lines({'parent_id':row['id'],'lines':lines},
                               {**row,'visual_line_count':expected},size)
        result.append({**deepcopy(item),'spans':[l['span'] for l in checked['lines']]})
    return {'items':result}


def geometry_prompt(rows, frame, scores):
    return {'protocol':'glm_frozen_text_geometry.1','image_size':frame['size'],
        'task':'Locate the SOURCE glyph ink for each fixed text ID. Return geometry, not a keep decision.',
        'targets':[{'id':r['id'],'text':r['text'],'kind':r.get('kind','text'),
                    'runs':r['runs'],'latex':r.get('latex',''),'rotation_deg':r['rotation_deg'],
                    'visual_line_count':1 if r.get('kind')=='formula' else r.get('visual_line_count',1),
                    'rough_bbox_2d':mapped_box(r['bbox_2d'],frame),
                    'cpu_similarity_percent':round(100*scores[r['id']]['before']['score'],2)} for r in rows],
        'rules':[
            'SOURCE is the original enlarged crop. HINTS adds fallible boxes/IDs. PPT is the current '
            'text-only rendering at exactly the SAME crop coordinates and scale. The CPU score is '
            'a foreground similarity proxy, not ground truth: non-text background objects can lower it.',
            'Return a fresh tight bbox for EVERY identifiable target, even when it already looks correct. '
            'Do not output keep or simply repeat the hint box. Inspect actual source glyph ink.',
            'Coordinates are pixel centers [xmin,ymin,xmax,ymax] in SOURCE. Tight means the complete '
            'visible glyph ink including dots, accents, punctuation and underline, with NO extra whitespace. '
            'Exclude enclosing rectangles, arrows, fills and adjacent words belonging to another ID.',
            'The latest transcription, font, bold/italic/underline, color, scripts and LaTeX are frozen. '
            'Copy the supplied text exactly in lines; do not correct spelling or infer missing content. '
            'Style differences may explain scores but cannot be fixed by stretching the bbox beyond source ink.',
            'One ordinary baseline has one lines entry. A known multiline ordinary target has one entry '
            'per visible baseline in reading order; their text must partition the original string. '
            'Never split a word. A formula always has ONE bbox containing all its scripts/fraction/root.',
            'Associate each ID with its anchored physical occurrence, not a matching word elsewhere. '
            'For unidentifiable/clipped targets return status=ambiguous and lines=[]. '
            'Reasons are brief visible observations, not chains of reasoning.'],
        'response_format':{'items':[{'id':'supplied ID','status':'located/ambiguous',
            'reason':'brief observed source evidence','lines':[
                {'text':'exact supplied text or its complete visual line','bbox_2d':[10,20,120,40]}]}]}}


def mapped_detections(checked, rows, frame):
    by_id={r['id']:r for r in rows};detections=[]
    for item in checked['items']:
        if item['status']!='located':continue
        row=by_id[item['id']]
        for line,span in zip(item['lines'],item['spans']):
            box=mapped_box(line['bbox_2d'],frame,True)
            if not valid_box(box,frame['source_size']):raise ValueError('Mapped geometry outside source')
            detections.append({'index':len(detections),'target_id':row['id'],
                'character_span':span,'text':row['text'][span[0]:span[1]],
                'rotation':90 if 45<abs(row['rotation_deg'])%180<135 else 0,
                'source_bbox':box,'proposal_type':'glm_geometry',
                'reason':item['reason'],'geometry_backend':'glm'})
    return detections


def run_jobs(jobs, rows, source, rendered, scores, runtime, opt, progress):
    by_id={r['id']:r for r in rows}
    stage=runtime.for_stage(reasoning_effort=opt.get('geometry_reasoning_effort','low'))
    def job(spec):
        root=Path(spec['output']);root.mkdir(parents=True,exist_ok=True)
        crop=Path(spec['crop_dir']);frame=read(crop/'frame.json')
        selected=[by_id[i] for i in spec['ids']]
        local_hints(crop/'clean.png',selected,frame,root/'hints.png')
        with Image.open(rendered) as image:
            resample_crop(image.convert('RGB'),frame['box'],frame['size'],frame['scale'],frame['offset']).save(root/'rendered.png')
        prompt=geometry_prompt(selected,frame,scores)
        images=[('SOURCE clean enlarged original',crop/'clean.png'),
                ('HINTS current bbox and ID; fallible annotations',root/'hints.png'),
                ('PPT current text-only render, same coordinates',root/'rendered.png')]
        digests={name:sha(crop/name) for name in ('clean.png','frame.json')}
        fingerprint=hashlib.sha256(json.dumps([prompt,digests,sha(root/'rendered.png')],sort_keys=True).encode()).hexdigest()
        try:
            def validator(raw):
                validate(raw,selected,frame['size'])
                return raw
            raw=model.ask(stage,root/fingerprint[:16],prompt,images,validator,opt)
            checked=validate(raw,selected,frame['size'])
            detections=mapped_detections(checked,selected,frame)
            report={'backend':'glm','input_sha256':digests,'detections':detections,
                    'target_results':checked['items'],'frame':frame,'geometry_failed':False}
        except Exception as exc:
            report={'backend':'glm','input_sha256':digests,'detections':[],
                    'geometry_failed':True,'error':str(exc),'fallback':'keep prior geometry unchanged'}
        save(root/'report.json',report)
        return {'group':spec['group'],'state':'failed' if report['geometry_failed'] else 'completed'}
    results=[]
    with ThreadPoolExecutor(max_workers=min(4,max(1,len(jobs)))) as pool:
        for future in as_completed([pool.submit(job,j) for j in jobs]):
            results.append(future.result());progress('glm_geometry',done=len(results),total=len(jobs))
    return results


def select_geometry(row, detections, size, neighbors=()):
    """Select an ID-bound model box without measuring/moving its ink edges."""
    parts=[d for d in detections if d.get('target_id')==row['id']]
    if len(parts)!=1:raise ValueError('Missing or ambiguous GLM bbox; retain prior geometry')
    chosen=parts[0];box=chosen.get('source_bbox')
    if not valid_box(box,size):raise ValueError('Invalid GLM source bbox')
    if chosen.get('text')!=row['text']:
        raise ValueError('GLM bbox must cover the complete frozen transcription')
    if chosen.get('character_span',[0,len(row['text'])])!=[0,len(row['text'])]:
        raise ValueError('GLM bbox has only a partial character span')
    diagonal=math.hypot(row['bbox_2d'][2]-row['bbox_2d'][0],row['bbox_2d'][3]-row['bbox_2d'][1])
    own=distance(row['bbox_2d'],box)
    if own>max(32,min(.15*min(size),2*diagonal)):
        raise ValueError('GLM bbox too far from its original occurrence')
    if any(n['id']!=row['id'] and distance(n['bbox_2d'],box)+5<own for n in neighbors):
        raise ValueError('GLM bbox closer to another occurrence')
    return {'status':'located','bbox_2d':list(box),'text_authority':'vlm',
            'reason':'use_glm_bbox_verbatim_no_cpu_tightening',
            'bbox_policy':'model_bbox_without_cpu_tightening_v1',
            'detection_indices':[chosen['index']], 'proposal_type':'glm_geometry',
            'preserve_runs_and_colors':True,'requires_native_render_gate':True}


def propose_lines(row, detections, source, neighbors=()):
    """Keep a known multiline target atomic through the final rendered gate."""
    count=row.get('visual_line_count',1)
    if row.get('kind')=='formula' or count<=1:
        return None,{'status':'single_unit'}
    parts=[d for d in detections if d.get('target_id')==row['id']]
    if len(parts)!=count:return None,{'status':'glm_multiline_unresolved'}
    children=[]
    for i,d in enumerate(parts,1):
        lo,hi=d['character_span'];box=d['source_bbox']
        own=distance(row['bbox_2d'],box)
        if any(n['id']!=row['id'] and distance(n['bbox_2d'],box)+5<own for n in neighbors):
            return None,{'status':'glm_multiline_wrong_owner'}
        child=deepcopy(row)
        child.update(id=f'{row["id"]}.L{i:02d}',parent_id=row['id'],
                     text=row['text'][lo:hi],runs=slice_runs(row['runs'],lo,hi),
                     semantic_unit_id=row.get('semantic_unit_id',row['id']),
                     visual_line_count=1,line_layout_status='split_baseline',
                     parent_geometry_prior_bbox=list(row['bbox_2d']))
        if not valid_box(box,source.size):return None,{'status':'glm_multiline_invalid_bbox'}
        children.append(proposal(child,box,{'geometry_backend':'glm','character_span':[lo,hi],
            'bbox_policy':'model_bbox_without_cpu_tightening_v1','requires_native_render_gate':True}))
    return children,{'status':'glm_multiline_proposal','detection_indices':[d['index'] for d in parts]}
