"""CPU routes low-scoring text to a bounded bbox/color-only model review."""
from copy import deepcopy
import math
import re

from ..common import valid_box
from .controller import mapped_box
from .local_geometry import compact_groups
from .render_feedback import evidence,window
from .prompt_protocol import coordinates

THRESHOLD=0.80
DEFAULT_MAX_SCORE_DROP=0.05  # Five absolute points on the 0–100 display scale.
POLICY='cpu_gt80_else_source_ink_localization_with_explicit_score_tolerance'
TEXT_CHECKS=('size','position','color','content','font_style','orientation')


def nonregression_gate(metric,max_score_drop=DEFAULT_MAX_SCORE_DROP):
    if not math.isfinite(max_score_drop) or not 0<=max_score_drop<=1:
        raise ValueError('Invalid permitted score drop')
    if metric.get('neighbor_regressions'):return False,'protected_neighbor_cpu_score_decreased'
    before=metric['before']['score'];after=metric['after']['score']
    if not math.isfinite(before) or not math.isfinite(after):
        return False,'invalid_cpu_score'
    if metric.get('reliable') is False:return False,'insufficient_source_evidence'
    if after<=0 or metric['after'].get('render_mass',1)<=0:
        return False,'no_visible_render_evidence'
    if after>=before:return True,'cpu_score_not_decreased'
    if before-after<=max_score_drop+1e-12:return True,'cpu_drop_within_explicit_tolerance'
    return False,'cpu_score_decreased'


def retain_nonregressing(updates,evaluate,max_score_drop=DEFAULT_MAX_SCORE_DROP):
    """Re-render after rollbacks until every edit is within the absolute tolerance.

    evaluate renders the exact supplied combination and returns same-ROI scores
    against the unchanged baseline. Removed edits are never reintroduced.
    """
    remaining=deepcopy(updates);history=[]
    while True:
        metrics=evaluate(remaining)
        if set(metrics)!=set(remaining):raise ValueError('Every candidate needs a CPU comparison')
        records=[];rejected=[]
        for ident,metric in metrics.items():
            ok,reason=nonregression_gate(metric,max_score_drop)
            records.append({'id':ident,'accepted':ok,'reason':reason,'evidence':metric})
            if not ok:rejected.append(ident)
        history.append({'round':len(history)+1,'checks':records,'rejected_ids':rejected})
        if not rejected:return remaining,history
        for ident in rejected:remaining.pop(ident)


def select(rows,source,rendered,max_groups=0):
    low=[];scores={};automatic=[]
    for r in rows:
        m=evidence(source,rendered,rendered,[r]);scores[r['id']]=m
        if math.isfinite(m['before']['score']) and m['before']['score']>THRESHOLD:
            automatic.append(r['id'])
        else:low.append(r)
    groups=compact_groups(low,source.size,6) if low else []
    groups.sort(key=lambda g:min(scores[r['id']]['before']['score'] for r in g))
    selected=groups if not max_groups else groups[:max_groups]
    return selected,{'policy':POLICY,'threshold':THRESHOLD,'comparison':'>',
        'automatic_ids':automatic,'selected_ids':[[r['id'] for r in g] for g in selected],
        'deferred_ids':[r['id'] for g in groups[len(selected):] for r in g],
        'scores':scores}


def prompt(size,targets,context):
    return {'protocol':'source_ink_bbox_color.5','coordinates':coordinates(size),
        'task':'Locate tight original glyph bounds and foreground color for EVERY target from SOURCE first. Do not decide whether the existing text is close enough to keep.',
        'targets':targets,'context_read_only':context,
        'evidence':{'SOURCE':'Clean original crop including non-text elements.',
                    'PPT':'Actual text-only PPT rendering; exactly the same crop, pixel coordinates and scale.',
                    'REFERENCE':'SOURCE with predicted bbox/ID highlights. Annotations are fallible, not glyph boundaries.'},
        'rules':[
            'FIRST return bbox_2d and color for EVERY identifiable target, even if the existing text looks correct or background explains the low score. bbox_2d means source glyph ink bounds, NOT a PPT textbox or layout region. There is no keep decision. Similarity judgments must not suppress localization.',
            'A crop can contain several target IDs. Localize EACH ID independently; never return the union of neighboring texts or the enclosing module. Crop padding is context only and must not become bbox padding.',
            'Use the smallest axis-aligned rectangle enclosing the complete visible glyph strokes, with no intentional outer padding, line-height margin or safety space. Include dots, accents, descenders, punctuation and superscripts/subscripts. Preserve internal spaces and grouping. Tight means complete and precise, not necessarily smaller: expand an old box that clips glyphs.',
            'Set left=min x, top=min y, right=max x, bottom=max y of this target\'s visible glyph ink in SOURCE. Each of the four edges must touch at least one outermost glyph stroke; no removable empty strip is permitted between an edge and all target strokes. No extra margin on any side. This does not require filling the entire rectangle: internal word spaces, character holes and empty corners are normal. Do not cut strokes to eliminate those spaces.',
            'Before returning, check each edge against SOURCE glyphs, not against the old colored box, a rounded rectangle, an arrow or an adjacent label. If moving an edge inward would remove only outer blank space, move it inward until it touches target ink; if it clips a glyph, move it outward to include the entire glyph. For vertical text use its visible image-axis bounds. In localization_evidence briefly identify the target glyph strokes defining left, top, right and bottom.',
            'localization_status=located requires reliable SOURCE glyph bounds and foreground color. Give brief visible boundary evidence in localization_evidence. Use uncertain only if SOURCE localization cannot be established, such as severe occlusion, unreadable boundaries or ambiguous occurrence identity. Uncertain SOURCE/PPT similarity alone must not suppress localization.',
            'AFTER localization complete text_checks for size, position, color, content, font_style and orientation, each with status=match/mismatch/uncertain and a concrete comparative observation. Diagnostics do not control whether a located bbox is returned. Report font/content/orientation errors without changing those properties or distorting SOURCE bounds to compensate. Give observable findings, not reasoning traces.',
            'For size compare actual glyph height and word extent in the shared pixel scale; for position compare glyph centers and baseline; for color compare glyph foreground; for content compare characters and scripts; for font_style compare family appearance, weight and italics; for orientation compare reading direction. A correct word or a shared center does not imply a correct size. For rotated text compare the same rotated glyph extents, not an unrotated textbox.',
            'REFERENCE highlights are the CURRENT predicted text bounds, not an annotation of correct source text. A box encompassing an entire shape/container is not evidence that its enclosed text should fill that container. Judge SOURCE versus PPT glyphs before using the reference to locate an ID.',
            'Each target states: the similarity between SOURCE and PPT within this target evaluation ROI is XX/100. Read cpu_similarity_score_0_to_100 and score_roi_on_crop. This is NOT a whole-crop score. Scores <=80 trigger localization, not automatic acceptance; CPU renders each proposal and rejects score decreases. This is a pixel proxy, not OCR accuracy.',
            'Two possible contributors to a low score are (1) non-text elements in SOURCE, such as arrows, boundaries, images or background colors, absent from the text-only PPT; (2) inaccurate target text appearance, especially bbox-derived size, position or foreground color. Both can coexist. Do not assume either explanation from the score alone.',
            'Determine the actual complete glyph bounds from clean SOURCE, using neighboring words as context. Check the highlighted predicted bbox for clipped first/last characters, dots, ascenders, descenders, accents and scripts. Expand, shrink or move it as necessary; never use the predicted box as ground truth.',
            'Exclude arrows, connector strokes, shape outlines, illustrations and surrounding whitespace from the text bbox. Do not include an entire crossing arrow. At occlusions use visible glyph evidence; if complete bounds cannot be established return null rather than hallucinating.',
            'Return causes=[1] for background-only interference, causes=[2] for incorrect target text appearance, or causes=[1,2] when both are present. Do not use background interference to excuse an independently visible text error.',
            'For cause 1, background_elements must identify the specific visible objects and their relation to the target: e.g. a diagonal gray connector crossing the middle of the word, or a colored rectangle behind it. Do not merely repeat background interference. For cause 2, text_issues must identify the concrete difference from SOURCE: e.g. rightmost letters outside the predicted bbox, glyphs too short, shifted baseline, or wrong gray level. These are brief observations, not chain-of-thought.',
            'Any mismatching text_check requires cause 2 with a corresponding text_issues entry. All checks matching, causes=[1], or uncertain comparative checks still require bbox/color when SOURCE localization is possible. causes may be empty if no cause is supported. Do not invent a discrepancy to justify returning a bbox.',
            'Return only id, localization_status, bbox_2d, color, localization_evidence, text_checks, causes, background_elements and text_issues, in that order. Content, formula structure, font properties, rotation and grouping are frozen. CPU calculates size from ink bounds and constructs its own PPT frame. Do not deform the text to match background or raise the CPU score.',
            'color is a #RRGGBB foreground color. For mixed-color runs, color may instead be an array of #RRGGBB values, exactly one per supplied run in unchanged order. A single color changes the base color and matching inherited run colors while preserving explicitly different run colors.',
            'Return each supplied target ID exactly once, not context IDs. bbox_2d uses local SOURCE pixel centers. If localization_status=uncertain, return bbox_2d=null and color=null and identify the localization obstacle.'],
        'response_format':{'items':[{'id':'supplied ID',
            'localization_status':'located/uncertain','bbox_2d':[10,20,100,45],'color':'#A0A0A0',
            'localization_evidence':'visible first/last glyph strokes and top/bottom ink limits, or localization obstacle',
            'text_checks':{key:{'status':'match/mismatch/uncertain','evidence':'specific SOURCE versus PPT glyph comparison'} for key in TEXT_CHECKS},
            'causes':[1,2],
            'background_elements':['specific interfering element and its position'],
            'text_issues':['specific text size/position/color mismatch']}]}}


def validate(raw,group,frame):
    # A bare array is equivalent to the requested items envelope; validate every
    # item identically rather than spending another model call on JSON wrapping.
    if isinstance(raw,list):raw={'items':raw}
    items=raw.get('items') if isinstance(raw,dict) else None;by_id={r['id']:r for r in group}
    if not isinstance(items,list) or len(items)!=len(by_id) or any(not isinstance(r,dict) for r in items) or {r.get('id') for r in items}!=set(by_id):
        raise ValueError('Return every target ID exactly once')
    for item in items:
        if set(item)!={'id','localization_status','localization_evidence','text_checks','causes','background_elements','text_issues','bbox_2d','color'}:
            raise ValueError('Return the exact diagnosis, decision, bbox and color fields')
        checks=item['text_checks']
        if not isinstance(checks,dict) or set(checks)!=set(TEXT_CHECKS):
            raise ValueError('All six independent text checks are required')
        for check in checks.values():
            if (not isinstance(check,dict) or set(check)!={'status','evidence'}
                or check['status'] not in ('match','mismatch','uncertain')
                or not isinstance(check['evidence'],str) or not check['evidence'].strip()):
                raise ValueError('Each text check needs status and concrete comparative evidence')
        causes=item['causes'];status=item['localization_status']
        if status not in ('located','uncertain'):raise ValueError('Return located or uncertain, never keep')
        if not isinstance(item['localization_evidence'],str) or not item['localization_evidence'].strip():
            raise ValueError('Source boundary evidence or localization obstacle is required')
        if not isinstance(causes,list) or any(type(c) is not int or c not in (1,2) for c in causes) or len(causes)!=len(set(causes)):
            raise ValueError('causes must contain unique 1 and/or 2')
        for number,field in [(1,'background_elements'),(2,'text_issues')]:
            details=item[field]
            if not isinstance(details,list) or any(not isinstance(s,str) or not s.strip() for s in details):
                raise ValueError('Specific visible evidence strings required')
            if bool(details)!=(number in causes):raise ValueError('Each diagnosed cause needs matching concrete evidence')
        mismatches={key for key,value in checks.items() if value['status']=='mismatch'}
        if mismatches and 2 not in causes:raise ValueError('Text mismatch must be diagnosed as cause 2')
        if 2 in causes and not mismatches:raise ValueError('Cause 2 needs an explicit mismatching text check')
        box=item['bbox_2d'];color=item['color']
        if status=='uncertain':
            if box is not None or color is not None:raise ValueError('Uncertain localization must not change bbox/color')
            continue
        if not valid_box(box,frame['size']):raise ValueError('Invalid local bbox')
        colors=color if isinstance(color,list) else [color]
        if not colors or any(not isinstance(c,str) or re.fullmatch(r'#[0-9a-fA-F]{6}',c) is None for c in colors):
            raise ValueError('Expected #RRGGBB color or per-run colors')
        old=by_id[item['id']]
        if isinstance(color,list) and len(color)!=len(old['runs']):raise ValueError('Run color count must match frozen runs')
        mapped=mapped_box(box,frame,True)
        if not valid_box(mapped,frame['source_size']):raise ValueError('Mapped bbox outside source')
        b=old['bbox_2d'];pad=max(64,2*min(b[2]-b[0],b[3]-b[1]))
        anchor=window([old],frame['source_size'],padding=pad)
        cx=(mapped[0]+mapped[2])/2;cy=(mapped[1]+mapped[3])/2
        if not(anchor[0]<=cx<=anchor[2] and anchor[1]<=cy<=anchor[3]):raise ValueError('Moved to a distant occurrence')
    return deepcopy(raw)


def proposals(result,group,frame):
    by_id={r['id']:r for r in group};updates={}
    for item in validate(result,group,frame)['items']:
        if item['localization_status']!='located':continue
        old=by_id[item['id']]
        row={k:deepcopy(old[k]) for k in ('id','text','runs','kind','latex','font','bold','italic','underline','rotation_deg','color','legibility','parent_id','background_fill','background_evidence') if k in old}
        row['bbox_2d']=mapped_box(item['bbox_2d'],frame,True)
        color=item['color']
        if isinstance(color,list):
            row['color']=color[0].upper()
            for run,c in zip(row['runs'],color):run['color']=c.upper()
        else:
            row['color']=color.upper()
            for run in row['runs']:
                if run.get('color',old['color']).upper()==old['color'].upper():run['color']=row['color']
        updates[item['id']]=[row]
    return updates
