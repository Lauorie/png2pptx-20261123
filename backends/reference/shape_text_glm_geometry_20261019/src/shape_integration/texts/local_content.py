"""Joint local transcription and geometry review with explicit identity evidence."""
from concurrent.futures import as_completed
from copy import deepcopy
import hashlib
import json
import re
import unicodedata

from PIL import Image, ImageDraw, ImageFont
from ..common import read, save, intersect, valid_box, sha, FONT
from . import contracts
from .prompt_protocol import SYSTEM, review
from .line_evidence import line_bands
from .numeric_routing import route
from .local_geometry import compact_groups, union, direction_board, validate as geometry_validate

FIELDS = ('text','runs','kind','latex','bbox_2d','rotation_deg')
STYLE_FIELDS = ('font','color','bold','italic','underline')


def highlight_review(clean, rows, target_ids, frame, output):
    """Keep clean pixels separate; red targets, gray read-only context, no box fill."""
    from .controller import mapped_box
    with Image.open(clean) as im:view=im.convert('RGB')
    draw=ImageDraw.Draw(view);font=ImageFont.truetype(FONT,max(12,round(max(view.size)/130)))
    for row in sorted(rows,key=lambda r:r['id'] in target_ids):
        box=mapped_box(row['bbox_2d'],frame)
        box=[max(0,min(v,view.size[i%2]-1)) for i,v in enumerate(box)]
        color='#E02040' if row['id'] in target_ids else '#888888'
        draw.rectangle(box,outline=color,width=2)
        label=row['id'];b=draw.textbbox((0,0),label,font=font)
        x=max(0,min(box[0],view.width-b[2]-4));y=max(0,box[1]-b[3]-4)
        draw.rectangle((x,y,x+b[2]+4,y+b[3]+4),fill='white')
        draw.text((x+2,y),label,font=font,fill=color)
    view.save(output)


def collisions(rows, threshold=.65):
    pairs=[]
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            boxes=[a['bbox_2d'],b['bbox_2d']]
            area=min((v[2]-v[0])*(v[3]-v[1]) for v in boxes)
            if intersect(*boxes)/max(area,1e-9)>=threshold:pairs.append([a['id'],b['id']])
    return pairs


def validate(raw, group, size):
    if isinstance(raw,list):raw={'items':raw}
    # Accept an equivalent envelope, never infer or drop target identities.
    if isinstance(raw,dict) and set(raw)=={'targets'} and isinstance(raw['targets'],list):
        raw={'items':raw['targets']}
    # Some providers return an ID-keyed object instead of an items list. It is
    # losslessly equivalent only when all and exactly the expected IDs occur.
    if isinstance(raw,dict) and set(raw)=={r['id'] for r in group}:
        if not all(isinstance(v,dict) and v.get('id',k)==k for k,v in raw.items()):
            raise ValueError('ID-keyed response disagrees with explicit item ID')
        raw={'items':[{**v,'id':k} for k,v in raw.items()]}
    if not isinstance(raw,dict) or not isinstance(raw.get('items'),list):raise ValueError('Expected items array')
    by_id={r['id']:r for r in group};items=raw['items']
    if len(items)!=len(group) or {r.get('id') for r in items}!=set(by_id):raise ValueError('Every target ID exactly once')
    output=[]
    for item in items:
        if item.get('status')=='resolved':
            units=item.get('units')
            if not isinstance(units,list) or not 1<=len(units)<=64:
                raise ValueError('resolved requires 1..64 complete visible units')
            if len(units)==1:
                item={**units[0],'id':item['id'],'status':'located',
                      'reason':units[0].get('reason',item.get('reason',''))}
            else:
                item={'id':item['id'],'status':'split','parts':units,'reason':item.get('reason','')}
        elif 'units' in item:
            if item['units']!=[]:raise ValueError('Unresolved/excluded targets require units=[]')
            item={**item,'bbox_2d':None,'rotation_deg':None,'direction_status':'ambiguous'}
        if item.get('status')=='split':
            parts=item.get('parts')
            if not isinstance(parts,list) or not 2<=len(parts)<=64:
                raise ValueError('split requires 2..64 complete visible lines or independent labels')
            if not isinstance(item.get('reason'),str) or not item['reason'].strip():
                raise ValueError('split requires visible layout evidence')
            children=[];sources=[]
            for index,part in enumerate(parts,1):
                ident=f'{item["id"]}.L{index:02d}'
                children.append({**part,'id':ident,'status':'located'})
                sources.append({**by_id[item['id']],'id':ident})
            checked=validate({'items':children},sources,size)['items']
            output.append({**item,'parts':checked});continue
        if item.get('status')=='excluded_image_text':
            if item.get('bbox_2d') is not None or item.get('rotation_deg') is not None:
                raise ValueError('Excluded image text must have null bbox and rotation')
            if not isinstance(item.get('reason'),str) or not item['reason'].strip():
                raise ValueError('Image-text exclusion requires visible asset evidence')
            output.append(deepcopy(item));continue
        item=deepcopy(item)
        def number(value):
            if isinstance(value,str) and re.fullmatch(r'-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?',value):
                return float(value)
            return value
        if isinstance(item.get('bbox_2d'),list):item['bbox_2d']=[number(v) for v in item['bbox_2d']]
        if 'rotation_deg' in item:item['rotation_deg']=number(item['rotation_deg'])
        geometry_validate({'items':[{k:item.get(k) for k in ('id','status','bbox_2d','rotation_deg','direction_status','reason')}]},[item['id']],size)
        clean=deepcopy(item)
        clean.pop('background_fill',None)
        clean.pop('background_evidence',None)
        if item['status']=='located':
            if item.get('kind')=='text' and item.get('visual_line_count',1)!=1:
                raise ValueError('Multiple plain-text baselines require status=split and one part per line')
            if item['direction_status']!='confirmed':raise ValueError('Use ambiguous status if reading direction uncertain')
            for field in ('runs','kind','latex'):
                if field not in item:raise ValueError('Explicit observed '+field+' required')
            # The newest colored runs are the sole plain-text authority. Do
            # not require a redundant matching transcription from the model.
            item['text']=''.join(r['text'] for r in item['runs'])
            candidate={**by_id[item['id']],**{k:item[k] for k in FIELDS},'replaces':[]}
            candidate.update({k:item[k] for k in STYLE_FIELDS if k in item})
            # Unicode display scripts (g₁) and structured runs (g + subscript 1)
            # are equivalent. Normalize only when NFKC agrees, never repair a
            # genuinely different transcription (e.g. Mask vs Attention).
            run_text=''.join(r.get('text','') for r in candidate['runs'])
            normalize=lambda s:''.join(unicodedata.normalize('NFKC',s).split())
            left,right=normalize(run_text),normalize(candidate['text'])
            if candidate['kind']=='formula':
                # Display summaries may use ^{...}/_{...} while runs already
                # carry those scripts structurally. Complete LaTeX remains the
                # authority for formula export, never this plain text summary.
                left=''.join(c for c in left if c not in '^_{}')
                right=''.join(c for c in right if c not in '^_{}')
            if left==right:candidate['text']=run_text
            candidate=route(candidate,'crop_reply_before_formula')
            routing_audit=candidate.get('routing_audit')
            if candidate['kind']=='formula':
                from .formula_state import adopt
                candidate=adopt(candidate,{'id':item['id'],'status':'resolved','latex':candidate['latex']})
            candidate=contracts.recognition({'texts':[candidate]},size)['texts'][0]
            clean.update({k:candidate[k] for k in FIELDS})
            clean.update({k:candidate[k] for k in STYLE_FIELDS})
            if routing_audit:clean['routing_audit']=routing_audit
        output.append(clean)
    located=[part for r in output for part in (r['parts'] if r['status']=='split' else [r] if r['status']=='located' else [])]
    duplicates=collisions(located)
    if duplicates:raise ValueError('Different IDs assigned overlapping glyphs; assign distinct physical occurrences or mark ambiguous: '+str(duplicates))
    return {'items':output}


def conflict_groups(rows,pairs):
    components=[]
    for pair in pairs:
        connected=set(pair)
        for old in list(components):
            if old & connected:connected |= old;components.remove(old)
        components.append(connected)
    by_id={r['id']:r for r in rows}
    return [[by_id[key] for key in sorted(ids)] for ids in components]


def source_box(box, frame):
    """Map valid local pixel centers, clipping only half-pixel border support.

    Upsampling puts the outer local pixel centers inside the outer source
    pixel's footprint, possibly outside its center. This is not an out-of-image
    prediction; larger excursions and invalid local boxes are still rejected.
    """
    from .controller import mapped_box
    if not valid_box(box,frame['size']):raise ValueError('Invalid local bbox')
    mapped=mapped_box(box,frame,True)
    if any(v<-.5-1e-7 or v>frame['source_size'][i%2]-.5+1e-7 for i,v in enumerate(mapped)):
        raise ValueError('mapped bbox outside source')
    bounded=[max(0,min(v,frame['source_size'][i%2]-1)) for i,v in enumerate(mapped)]
    if not valid_box(bounded,frame['source_size']):raise ValueError('mapped bbox outside source')
    return bounded


def validate_pixel_lines(checked, group, frame, source_image):
    """Apply the same baseline evidence check to every resolved leaf unit."""
    from .controller import mapped_box
    by_id={r['id']:r for r in group}
    for parent in checked['items']:
        proposals=parent['parts'] if parent['status']=='split' else [parent]
        for proposal in proposals:
            if proposal['status']!='located':continue
            candidate={**by_id[parent['id']],**proposal,
                       'bbox_2d':source_box(proposal['bbox_2d'],frame)}
            bands=line_bands(candidate,source_image)
            if bands:
                raise ValueError(f'{proposal["id"]}: source pixels show multiple separated text bands; '
                                 'return one complete unit per visible line or mark the parent ambiguous. '
                                 f'Crop band hints: {[mapped_box(b,frame) for b in bands]}')
    return checked


def refine(rows, source, folder, runtime, opt, pool, progress=lambda *a:None,
           *, target_ids=None, rendered=None, scores=None, collision_passes=2):
    from .controller import frame, mapped_box, local_hints
    folder.mkdir(parents=True,exist_ok=True)
    with Image.open(source) as im:
        size=list(im.size);source_image=im.convert('RGB')
    image_sha=sha(source)
    selected=set(target_ids) if target_ids is not None else {r['id'] for r in rows}
    if not selected<={r['id'] for r in rows}:raise ValueError('Unknown crop target IDs')
    groups=compact_groups([r for r in rows if r['id'] in selected],size,opt.get('geometry_max_targets',6))
    save(folder/'input_rows.json',rows)
    save(folder/'plan.json',{'groups':[[r['id'] for r in g] for g in groups],
         'strategy':'single_cpu_gated_low_score_crop' if rendered is not None else 'joint_content_geometry_with_context_and_collision_review',
         'content_frozen':False,'allowed_updates':list(FIELDS+STYLE_FIELDS),
         'target_ids':sorted(selected),'max_collision_passes':collision_passes,
         'highlight':'red=target, gray=read-only context','rendered_comparison':rendered is not None})
    stage=runtime.for_stage(reasoning_effort=opt.get('geometry_reasoning_effort','low'),shape_schema_retries=1)

    def job(name,group,context):
        root=folder/name;root.mkdir(exist_ok=True)
        short=max(min(r['bbox_2d'][2]-r['bbox_2d'][0]+1,r['bbox_2d'][3]-r['bbox_2d'][1]+1) for r in group)
        f=frame(source,root/'input',{**opt,'crop_min_padding_px':max(64 if rendered else 24,short*2.5),'crop_padding':0,'crop_scale':100},union(group))
        visible=[r for r in context if intersect(r['bbox_2d'],f['box'])>0]
        ids={r['id'] for r in group}
        highlight_review(root/'input/clean.png',visible,ids,f,root/'input/ids.png')
        images=[('CLEAN source crop; all bbox coordinates refer to this.',root/'input/clean.png'),
                ('Fallible OCR anchors, including read-only neighboring IDs. Labels are not source text.',root/'input/ids.png')]
        if rendered is not None:
            from pathlib import Path
            from .render_feedback import render_evidence
            with Image.open(rendered) as image:
                render_evidence(Path(rendered).with_suffix('.pdf'),image.convert('RGB'),f,root/'input')
            images.append(('PPT actual text render, SAME crop coordinates and scale.',root/'input/render.png'))
        elif direction_board(source,group,root/'input/directions.png'):
            images.append(('Optional rotation evidence; do NOT use these panel coordinates.',root/'input/directions.png'))
        targets=[{k:r.get(k) for k in ('id','text','kind','latex','runs','rotation_deg',*STYLE_FIELDS)} |
                     {'rough_bbox_2d':mapped_box(r['bbox_2d'],f),
                      'pixel_line_bands_hint':[mapped_box(b,f) for b in line_bands(r,source_image)]} for r in group]
        context_items=[{'id':r['id'],'text':r['text'],'latex':r.get('latex',''),
                       'rough_bbox_2d':mapped_box(r['bbox_2d'],f)} for r in visible if r['id'] not in ids]
        prompt=review(f['size'],targets,context_items)
        if scores is not None:
            for target in prompt['targets']:
                target['cpu_similarity_percent']=round(scores[target['id']]['before']['score']*100,2)
            prompt['rules'] += [] if prompt.get('review_rules_version') == 'user_r1_r21_20261021' else [
                'This is the ONLY text-review pass. Only supplied low-score target IDs may change; '
                'gray neighboring IDs are read-only context. Red boxes are current predictions, '
                'not correct boundaries. Read glyphs from CLEAN, not annotation strokes.',
                'PPT is the actual current text-only reconstruction at the SAME crop coordinates '
                'and scale. Compare CLEAN and PPT for content, font, weight, slant, underline, '
                'foreground colors, orientation and source bbox. Correct these jointly in the units.',
                'The per-target CPU score is a pixel-similarity proxy. Missing non-text shapes, '
                'lines and images may lower it. A low score does not by itself prove a text error. '
                'Return complete supported units; use ambiguous if the source cannot resolve them. '
                'Do not invent unseen content or change high-score context to improve a score.']
        # Cache is tied to source, input fields, prompt and geometry protocol.
        fingerprint=hashlib.sha256(json.dumps([image_sha,prompt,f,
            sha(root/'input/render.png') if rendered is not None else None],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        calls=root/'calls'/fingerprint[:16]
        def validator(value):
            checked=validate(value,group,f['size'])
            return validate_pixel_lines(checked,group,f,source_image)
        previous_prompt=read(root/'prompt.json') if (root/'prompt.json').exists() else None
        recover_previous=previous_prompt==prompt
        save(root/'prompt.json',prompt)
        try:
            if (root/'result.json').exists() and (root/'cache.json').exists() and read(root/'cache.json').get('sha256')==fingerprint:
                result=validator(read(root/'result.json'))
            else:
                # Recover complete responses rejected solely by a representational
                # schema issue, without spending another API call.
                recovered=None
                for parsed in sorted(calls.glob('review_*/parsed_response.json'),reverse=True):
                    if not recover_previous:break
                    status_path=parsed.parent/'stream_status.json'
                    if not status_path.exists():continue
                    status=read(status_path)
                    if status.get('finish_reason')!='stop' or not status.get('sse_done'):continue
                    try:recovered=validator(read(parsed));break
                    except ValueError:continue
                if recovered is not None:
                    save(root/'result.json',recovered);save(root/'cache.json',{'sha256':fingerprint})
                    save(root/'recovery.json',{'source':str(parsed),'policy':'validated completed response; Unicode script normalization','api_calls':0})
                    return name,group,recovered,f,None
                def repair(previous,error):
                    return {'system':SYSTEM,'prompt':{**prompt,'repair_error':error,
                         'previous':(previous/'content.txt').read_text() if (previous/'content.txt').exists() else ''},
                         'images':images,'validator':validator}
                if opt.get('retry_failed_calls') and (root/'error.json').exists():
                    attempt=1
                    while (calls/f'explicit_retry_{attempt:02d}').exists():attempt+=1
                    calls=calls/f'explicit_retry_{attempt:02d}'
                result=stage.call(calls,'review',SYSTEM,prompt,images,validator,opt['max_tokens'],retry_builder=repair)
                save(root/'result.json',result);save(root/'cache.json',{'sha256':fingerprint})
            return name,group,result,f,None
        except Exception as exc:
            save(root/'error.json',{'error':str(exc)})
            return name,group,None,f,str(exc)

    output=deepcopy(rows);by_id={r['id']:r for r in output};audit=[];errors=[];failed=[];excluded=set();split_parents=set()
    def update_target(target,proposal,f,name):
        box=source_box(proposal['bbox_2d'],f)
        if not valid_box(box,size):raise ValueError('mapped bbox outside source')
        if not target.get('parent_id'):
            target.setdefault('geometry_prior_bbox',deepcopy(target['bbox_2d']))
        target.update({k:deepcopy(proposal[k]) for k in FIELDS+STYLE_FIELDS if k in proposal});target['bbox_2d']=box
        target.update(visual_line_count=1,line_layout_status='native_formula' if target.get('kind')=='formula' else 'single_baseline')
        if target.get('kind')=='formula':
            target.update(formula_status='resolved',formula_authority='latest_valid_crop_reply')
            target.pop('formula_fallback',None)
        elif proposal.get('routing_audit'):
            target.update(formula_status='not_applicable',routing_audit=deepcopy(proposal['routing_audit']))
        target.setdefault('observations',[]).append({'origin':'joint_local_review','group':name,'local':deepcopy(proposal)})
    def apply(jobs):
        results=[]
        for done,future in enumerate(as_completed(jobs),1):
            results.append(future.result());progress('local_content',done,len(jobs))
        for name,group,result,f,error in sorted(results,key=lambda r:r[0]):
            if error:errors.append(name+': '+error);failed.append(name);continue
            for proposal in result['items']:
                target=by_id[proposal['id']];before={k:deepcopy(target.get(k)) for k in FIELDS+STYLE_FIELDS}
                record={'id':target['id'],'group':name,'before':before,'proposal':proposal}
                if proposal['status']=='located':
                    candidate=deepcopy(target)
                    try:update_target(candidate,proposal,f,name)
                    except (ValueError,TypeError,KeyError) as exc:
                        errors.append(name+': '+str(exc));record['unresolved']=True
                    else:target.update(candidate)
                elif proposal['status']=='split':
                    children=[]
                    try:
                        for part in proposal['parts']:
                            child={**deepcopy(target),'id':part['id'],'parent_id':target['id'],
                                   'parent_geometry_prior_bbox':deepcopy(target.get('geometry_prior_bbox',target['bbox_2d']))}
                            update_target(child,part,f,name);children.append(child)
                    except (ValueError,TypeError,KeyError) as exc:
                        errors.append(name+': '+str(exc));record['unresolved']=True
                    else:
                        split_parents.add(target['id']);record['child_ids']=[c['id'] for c in children]
                        output.extend(children);by_id.update({c['id']:c for c in children})
                elif proposal['status']=='excluded_image_text':
                    excluded.add(target['id']);record['excluded_image_text']=True
                else:record['unresolved']=True
                record['after']={k:deepcopy(target.get(k)) for k in FIELDS};audit.append(record)
    snapshot=deepcopy(rows)
    apply([pool.submit(job,f'C{i:03d}',g,snapshot) for i,g in enumerate(groups)])
    output=[r for r in output if r['id'] not in excluded|split_parents]
    for iteration in range(collision_passes):
        pairs=collisions(output)
        if not pairs:break
        snapshot=deepcopy(output)
        conflicts=conflict_groups(snapshot,pairs)
        apply([pool.submit(job,f'X{iteration}_{i:03d}',g,snapshot) for i,g in enumerate(conflicts)])
        output=[r for r in output if r['id'] not in excluded|split_parents]
    unresolved_pairs=collisions(output)
    latest={r['id']:r for r in audit}
    unresolved_ids=[r['id'] for r in output if r['id'] in selected and
        ((r['id'] not in latest and not r.get('parent_id')) or latest.get(r['id'],{}).get('unresolved'))]
    if unresolved_pairs:errors.append('Unresolved overlapping text IDs: '+str(unresolved_pairs))
    if unresolved_ids:errors.append('Unresolved local transcription: '+str(unresolved_ids))
    initial={r['id']:r for r in rows}
    changed=lambda fields:sum(r['id'] not in initial or any(r.get(k)!=initial[r['id']].get(k) for k in fields) for r in output)
    save(folder/'audit.json',audit)
    save(folder/'summary.json',{'groups':len(groups),'targets':len(selected),'total_texts':len(rows),'bbox_updated':changed(['bbox_2d']),
         'retained_targets':len(output),'excluded_image_text_ids':sorted(excluded),
         'split_parent_ids':sorted(split_parents),
         'angles_changed':changed(['rotation_deg']),'content_changed':changed(['text','latex','kind','runs']),
         'failed_groups':len(failed),'errors':errors,'unresolved_ids':unresolved_ids,
         'unresolved_collisions':unresolved_pairs,'content_unchanged':changed(['text','latex','kind','runs'])==0})
    return output,errors
