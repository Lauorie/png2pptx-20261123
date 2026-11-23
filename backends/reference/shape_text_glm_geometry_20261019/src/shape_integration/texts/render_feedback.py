"""Bounded, occurrence-preserving review of actual PPT text-layer renders.

Pixel scores are local source-contrast proxies, not ground-truth text metrics.
They gate proposed edits conservatively; no score establishes OCR correctness.
"""
from copy import deepcopy
import math
import unicodedata

import numpy as np
from PIL import Image

from ..common import intersect, valid_box
from .controller import mapped_box
from .foreground import contrast_ink
from .local_content import FIELDS, STYLE_FIELDS, validate, validate_pixel_lines, collisions
from .local_geometry import compact_groups, union
from .prompt_protocol import RECOVERY_SCOPE, coordinates, unit_schema
from . import FONTS

CPU_SCORE_THRESHOLD = 0.80


def render_evidence(pdf, rendered, frame, output):
    from ..styling.export import render_pdf_crop
    from ..styling.views import resample_crop
    from ..common import save
    if pdf.is_file():
        image, metadata=render_pdf_crop(pdf,frame['source_size'],frame['box'],frame['size'],frame['scale'],frame['offset'])
    else:
        image=resample_crop(rendered,frame['box'],frame['size'],frame['scale'],frame['offset'])
        metadata={'method':'resampled_baseline_png','reason':'native PDF unavailable'}
    image.save(output/'render.png')
    save(output/'render_provenance.json',metadata)


def window(rows, size, padding=4):
    box=union(rows)
    return [max(0,math.floor(box[0]-padding)),max(0,math.floor(box[1]-padding)),
            min(size[0],math.ceil(box[2]+padding+1)),min(size[1],math.ceil(box[3]+padding+1))]


def evidence(source, before, after, old_rows, new_rows=()):
    """Score both renders on the SAME source-derived support and background.

    Foreground is never selected separately for the two candidate renders.
    The source contains non-text graphics; low proxy scores are only flags.
    """
    roi=window([*old_rows,*new_rows],source.size)
    src=np.asarray(source.crop(roi).convert('RGB'),dtype=np.float32)
    ref=contrast_ink(src,[0,0,src.shape[1],src.shape[0]])
    mass=float(ref.sum())
    def score(render):
        rgb=np.asarray(render.crop(roi).convert('RGB'),dtype=np.float32)
        pred=contrast_ink(rgb,[0,0,rgb.shape[1],rgb.shape[0]])
        denom=float((ref*ref).sum()+(pred*pred).sum())
        dice=float(2*(ref*pred).sum()/max(denom,1e-9))
        overlap=np.minimum(ref,pred)
        distance=np.linalg.norm(src-rgb,axis=2)/(255*math.sqrt(3))
        agreement=float(((1-distance)*overlap).sum()/max(float(overlap.sum()),1e-9))
        return {'mask_dice_proxy':dice,'color_agreement_proxy':agreement,
                'score':dice*(.75+.25*agreement),'render_mass':float(pred.sum())}
    return {'roi':roi,'source_mass':mass,'before':score(before),'after':score(after),
            'reliable':mass>=8,'metric':'same_roi_source_contrast_soft_dice_color_proxy'}


def select_groups(rows, source, rendered, max_groups=6, max_targets=6):
    anomalies=[]
    for row in rows:
        metric=evidence(source,rendered,rendered,[row])
        reasons=[]
        reg=row.get('glyph_registration',{})
        if reg.get('status')!='accepted' or reg.get('score',0)<.80:
            reasons.append('weak_or_missing_glyph_registration')
        if not absolute_gate(metric)[0]:reasons.append('cpu_score_not_above_threshold')
        if metric['before']['render_mass']<8:reasons.append('little_visible_rendered_ink')
        if reasons:anomalies.append({'id':row['id'],'reasons':reasons,'evidence':metric})
    by_id={r['id']:r for r in rows};by_flag={r['id']:r for r in anomalies}
    groups=compact_groups([by_id[r['id']] for r in anomalies],source.size,max_targets) if anomalies else []
    groups.sort(key=lambda g:min(by_flag[r['id']]['evidence']['before']['score'] for r in g))
    return groups[:max_groups],{'anomalies':anomalies,'selected_ids':[[r['id'] for r in g] for g in groups[:max_groups]],
           'deferred_ids':[r['id'] for g in groups[max_groups:] for r in g],
           'policy':'low pixel support is a review trigger, never an automatic edit'}


def prompt(size, targets, context):
    from .background import model_fields
    targets=model_fields(targets);context=model_fields(context)
    return {'protocol':'ppt_text_render_feedback.1',
        'task':'Compare source text with the actual rendered editable text layer; propose only necessary corrections.',
        'coordinates':coordinates(size),'fonts':{'allowed':list(FONTS)},
        'targets':targets,'context_read_only':context,
        'evidence':{'SOURCE':'Original clean crop; the only coordinate and content authority.',
                    'RENDER':'Actual PPT text-layer rendering at the SAME crop/scale. Missing graphics/backgrounds are expected.',
                    'IDS':'Source crop with fallible occurrence labels; do not transcribe annotations.',
                    'formula_preview':'Native Office Math may use a same-OMML compatibility image in LibreOffice; this is not PowerPoint verification.'},
        'rules':[*RECOVERY_SCOPE,
            'Inspect all target occurrences and their neighboring text jointly. Compare glyph family, weight, size, baseline, color, wrapping and direction. Repeated labels may differ legitimately; do not enforce uniformity without source evidence.',
            'Return exactly one item per target ID. Context is read-only. Do not add unrelated text, remove targets, or restore pictures/lines/shapes absent from the text-only render.',
            'Use decision=keep when the target is faithful or no supported correction is available. Use decision=uncertain when evidence cannot establish a correction. Both use units=[].',
            'Use decision=revise only for a visible discrepancy, with complete replacement units. One unit per ordinary baseline; keep a complete formula including scripts/radicals in one unit. Do not return a partial word or lose content when splitting.',
            'For font, weight, color, geometry, rotation or wrapping edits, concatenating all replacement units must preserve the entire target transcription, ignoring whitespace only. A line split must return ALL lines including any line outside the old rough bbox. Never drop a suffix because it does not have a separate target ID. A true character correction requires issue_types to include content and an explicit source-supported explanation.',
            'Describe the observed discrepancy briefly, not a chain of reasoning. issue_types may contain content,font,weight,color,geometry,rotation,wrapping.',
            'Return tight SOURCE glyph bounds and source styles, not a font-size multiplier or rendered-image coordinates. CPU measurements will determine size and placement. Do not crop glyphs to make a string fit.',
            'The current reconstruction and automatic mismatch hints are fallible. A small score is not evidence that the text is wrong. Preserve correct neighboring occurrences.'],
        'response_format':{'items':[{'id':'supplied ID','decision':'keep/revise/uncertain',
            'issue_types':['font'],'reason':'brief visible discrepancy or preservation reason','units':[unit_schema()]}]}}


def validate_review(raw, group, frame, source):
    if not isinstance(raw,dict) or not isinstance(raw.get('items'),list):
        raise ValueError('Expected items array')
    items=raw['items'];ids={r['id'] for r in group}
    if len(items)!=len(ids) or any(not isinstance(r,dict) for r in items) or {r.get('id') for r in items}!=ids:
        raise ValueError('Every target ID exactly once')
    allowed={'content','font','weight','color','geometry','rotation','wrapping'}
    revised=[];decisions=[]
    for item in items:
        decision=item.get('decision');reason=item.get('reason');issues=item.get('issue_types',[])
        if decision not in ('keep','revise','uncertain') or not isinstance(reason,str) or not reason.strip():
            raise ValueError('Valid decision and visible-evidence reason required')
        if not isinstance(issues,list) or any(v not in allowed for v in issues):raise ValueError('Invalid issue_types')
        if decision!='revise':
            if item.get('units')!=[]:raise ValueError('Keep/uncertain must not change units')
        else:
            if not issues:raise ValueError('A revision needs an observed issue')
            units=item.get('units')
            if not isinstance(units,list) or not units or any(not isinstance(u,dict) or not isinstance(u.get('text'),str) for u in units):
                raise ValueError('Revision requires complete replacement units')
            if 'content' not in issues:
                original=next(r for r in group if r['id']==item['id'])
                normalize=lambda s:''.join(unicodedata.normalize('NFKC',s).split())
                if normalize(''.join(u['text'] for u in units))!=normalize(original['text']):
                    raise ValueError(item['id']+': style/layout/wrapping edits must preserve ALL original characters across replacement units; return every baseline, or explicitly justify a content correction')
            revised.append({'id':item['id'],'status':'resolved','reason':reason,'units':item.get('units')})
        decisions.append({k:deepcopy(item[k]) for k in ('id','decision','reason')})
    chosen=[r for r in group if r['id'] in {v['id'] for v in revised}]
    checked=validate_pixel_lines(validate({'items':revised},chosen,frame['size']),chosen,frame,source) if revised else {'items':[]}
    return {'decisions':decisions,'revisions':checked['items']}


def proposals(checked, group, frame):
    by_id={r['id']:r for r in group};updates={}
    for item in checked['revisions']:
        original=by_id[item['id']];parts=item['parts'] if item['status']=='split' else [item]
        rows=[]
        for part in parts:
            row={k:deepcopy(original[k]) for k in ('id','legibility') if k in original}
            row.update({k:deepcopy(part[k]) for k in (*FIELDS,*STYLE_FIELDS)})
            row.update(background_fill=None,background_evidence='')
            row.update(id=part['id'],bbox_2d=mapped_box(part['bbox_2d'],frame,True))
            if not valid_box(row['bbox_2d'],frame['source_size']):raise ValueError('Mapped bbox outside source')
            if item['status']=='split':row['parent_id']=original['id']
            rows.append(row)
        # Occurrences stay local: crossing into a distant repeated label is not permitted.
        anchor=window([original],frame['source_size'],padding=max(24,min(original['bbox_2d'][2]-original['bbox_2d'][0],original['bbox_2d'][3]-original['bbox_2d'][1])*2))
        # A previously collapsed multiline child can have a tiny fitted box.
        # Only its recorded enclosing parent may extend a split's anchor.
        parent=original.get('parent_geometry_prior_bbox')
        if len(rows)>1 and original.get('parent_id') and valid_box(parent,frame['source_size']):
            old=original['bbox_2d']
            if parent[0]<=old[0] and parent[1]<=old[1] and parent[2]>=old[2] and parent[3]>=old[3]:
                anchor=[min(anchor[0],parent[0]),min(anchor[1],parent[1]),max(anchor[2],parent[2]),max(anchor[3],parent[3])]
        for row in rows:
            b=row['bbox_2d'];cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2
            if not (anchor[0]<=cx<=anchor[2] and anchor[1]<=cy<=anchor[3]):raise ValueError('Revision moved to a different occurrence')
        updates[item['id']]=rows
    return updates


def replace(rows, updates):
    return [deepcopy(part) for row in rows for part in updates.get(row['id'],[row])]


def absolute_gate(metric):
    """Strict absolute CPU floor; improvement alone cannot certify quality."""
    if not metric['reliable']:return False,'insufficient_source_ink'
    after=metric['after']
    if after['render_mass']<8:return False,'empty_or_invisible_candidate'
    if not math.isfinite(after['score']) or after['score']<=CPU_SCORE_THRESHOLD:
        return False,'cpu_score_below_threshold'
    return True,'cpu_score_meets_threshold'


def gate(metric, minimum_gain=.015):
    passed,reason=absolute_gate(metric)
    if not passed:return False,reason
    before=metric['before'];after=metric['after']
    if after['mask_dice_proxy']<before['mask_dice_proxy']-.01:return False,'glyph_overlap_regression'
    if after['score']<before['score']+minimum_gain:return False,'no_clear_proxy_improvement'
    return True,'improved_on_fixed_source_roi'


def audit_cpu_quality(source, rendered, rows):
    checks=[]
    for row in rows:
        metric=evidence(source,rendered,rendered,[row])
        passed,reason=absolute_gate(metric)
        checks.append({'id':row['id'],'text':row['text'],'cpu_passed':passed,
                       'reason':reason,'score':metric['after']['score'],'evidence':metric})
    return {'threshold':CPU_SCORE_THRESHOLD,'comparison':'>',
            'model_approval_implied':False,'checks':checks,
            'cpu_failed_ids':[r['id'] for r in checks if not r['cpu_passed']]}


def new_collisions(before, after):
    old={tuple(sorted(p)) for p in collisions(before)}
    return [p for p in collisions(after) if tuple(sorted(p)) not in old]


VERIFY_CRITERIA=('content', 'font_and_weight', 'size_and_position', 'color',
                 'wrapping_and_direction', 'neighbor_preservation')


def verification_prompt(size, targets):
    """Read-only acceptance of the actual combined render, not proposed styles."""
    return {
        'protocol':'ppt_text_render_acceptance.1',
        'task':'Independently accept or reject actual rendered text corrections against SOURCE.',
        'coordinates':coordinates(size), 'targets':targets,
        'rules':[
            *RECOVERY_SCOPE,
            'SOURCE is the content and geometry authority. BEFORE and AFTER are actual PPT text-only renders on exactly the same crop and scale. Missing non-text graphics and backgrounds are expected, not errors.',
            'IDS supplies fallible original occurrence labels only. Inspect the complete occurrence, including ALL replacement lines and adjacent unchanged text.',
            'Judge actual visible AFTER glyphs, not the proposed metadata or presumed improvements. Compare AFTER with SOURCE and BEFORE. No numerical CPU scores or proposer verdicts establish correctness.',
            'For each criterion return pass, fail or uncertain. Pass requires source-supported faithfulness and no new defect. A partly improved but still incorrect changed occurrence is not a pass. Unreadable or ambiguous evidence is uncertain.',
            'Content includes every character, punctuation, scripts and mathematical structure. Font and weight include italic style. Size and position include baseline and spacing. Wrapping and direction include all lines and rotations. Neighbor preservation includes overlap, lost text and unwanted changes.',
            'Return pass only when ALL six criteria pass and the modification visibly improves at least one source discrepancy. Otherwise return fail or uncertain. Do not propose further edits.',
            'Native formulas may use same-OMML compatibility images in LibreOffice. This does not establish native PowerPoint equation appearance; if that limits a verdict, return uncertain.',
            'Return exactly one item per supplied target ID, a brief visual-evidence reason, and no replacement units.'],
        'response_format':{'items':[{'id':'supplied ID','decision':'pass/fail/uncertain',
            'checks':{k:'pass/fail/uncertain' for k in VERIFY_CRITERIA},
            'improvement':'pass/fail/uncertain','reason':'brief visible evidence'}]}}


def validate_verification(raw, ids):
    items=raw.get('items') if isinstance(raw,dict) else None
    if not isinstance(items,list) or len(items)!=len(ids) or any(not isinstance(i,dict) for i in items) or {i.get('id') for i in items}!=set(ids):
        raise ValueError('Every verification target ID exactly once')
    for item in items:
        if set(item)!={'id','decision','checks','improvement','reason'}:
            raise ValueError('Verification is read-only; exact verdict fields required')
        checks=item['checks'];allowed={'pass','fail','uncertain'}
        if not isinstance(checks,dict) or set(checks)!=set(VERIFY_CRITERIA) or any(v not in allowed for v in checks.values()):
            raise ValueError('All verification criteria required')
        if item['decision'] not in allowed or item['improvement'] not in allowed or not isinstance(item['reason'],str) or not item['reason'].strip():
            raise ValueError('Explicit verification verdict and evidence required')
        if item['decision']=='pass' and (item['improvement']!='pass' or any(v!='pass' for v in checks.values())):
            raise ValueError('Acceptance requires all criteria and improvement to pass')
    return {'items':deepcopy(items)}


def dual_gate(cpu_checks, model_items, expected_ids):
    """Fail closed, including missing, duplicated and uncertain verdicts."""
    expected=set(expected_ids)
    if not expected:return False
    if len(cpu_checks)!=len(expected) or {r['id'] for r in cpu_checks}!=expected:return False
    if len(model_items)!=len(expected) or {r['id'] for r in model_items}!=expected:return False
    try:validate_verification({'items':model_items},expected)
    except (ValueError,TypeError,KeyError):return False
    return all(r['passed'] is True for r in cpu_checks) and all(r['decision']=='pass' for r in model_items)
