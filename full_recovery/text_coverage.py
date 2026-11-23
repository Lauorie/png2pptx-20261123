"""Source-led coverage for isolated typographic marks omitted from text input.

Source components only nominate locations. One batched visual decision names
actual missing text and rejects diagram marks. Geometry comes from original
ink, and the existing native text metric fitter places every added glyph.
"""
from pathlib import Path
import math
import unicodedata
import requests

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROTOCOL = 'source_text_coverage.6'


def is_symbol(text):
    return bool(text and text.strip()) and all(c.isspace() or (unicodedata.category(c).startswith(('P','S'))
        and 'ARROW' not in unicodedata.name(c,'')) for c in text)


def candidates(source, rows, native=None):
    rgb = np.asarray(source.convert('RGB')); height, width = rgb.shape[:2]
    ink = (rgb.mean(2) < 160) & (np.ptp(rgb.astype(int), axis=2) < 80)
    _, labels, stats, centers = cv2.connectedComponentsWithStats(ink.astype('uint8'), 8)
    heights = [r['bbox_2d'][3]-r['bbox_2d'][1]+1 for r in rows if len(r.get('bbox_2d', [])) == 4]
    typical = max(6., float(np.median(heights)) if heights else .05*max(source.size))
    known = np.zeros((height, width), 'uint8')
    for row in rows:
        b = row.get('bbox_2d', [])
        if len(b) == 4:
            x0, y0 = np.maximum(0, np.floor(np.asarray(b[:2])-2)).astype(int)
            x1, y1 = np.minimum([width, height], np.ceil(np.asarray(b[2:])+3)).astype(int)
            known[y0:y1, x0:x1] = 1
    rendered = None
    if native is not None:
        rendered = np.asarray(native.convert('RGB'))
        if rendered.shape != rgb.shape:
            raise ValueError('Text coverage requires aligned original/native canvases')
    result = []
    for ident, (x, y, w, h, area) in enumerate(stats[1:], 1):
        if (area < 5 or min(w, h) < 2 or max(w, h) > 2.6*typical
                or not .12 < w/h < 8 or area/(w*h) < .08):
            continue
        component = labels[y:y+h, x:x+w] == ident
        if np.mean(known[y:y+h, x:x+w][component]) > .25:
            continue
        colour = np.median(rgb[y:y+h, x:x+w][component], axis=0).astype(int)
        coverage = 0.
        if rendered is not None:
            x0, y0 = max(0, x-2), max(0, y-2)
            x1, y1 = min(width, x+w+2), min(height, y+h+2)
            # A coloured/grey node behind black text is not recovered glyph
            # ink. Compare the source component's colour, not any dark area.
            matching = (np.linalg.norm(rendered[y0:y1, x0:x1].astype(float)-colour, axis=2) < 45).astype('uint8')
            covered = cv2.dilate(matching, np.ones((5, 5), 'uint8')) > 0
            coverage = float(np.mean(covered[y-y0:y-y0+h, x-x0:x-x0+w][component]))
        if coverage >= .65:
            continue
        result.append({'bbox_2d': [int(x), int(y), int(x+w-1), int(y+h-1)], 'ink_pixels': int(area),
                       'native_coverage': coverage, 'color': '#'+''.join('%02X' % int(v) for v in colour)})
    result.sort(key=lambda r: (-(1-r['native_coverage'])*r['ink_pixels'], r['bbox_2d']))
    repeated = repeated_labels(rgb, rows, known, rendered)
    # Complete source word matches nominate text before small dashed-line
    # fragments consume the existing bounded review budget. Their content is
    # still decided by the same visual request, never by template matching.
    return repeated+[r for r in result if not any(
        a['bbox_2d'][0]<=r['bbox_2d'][0]<=r['bbox_2d'][2]<=a['bbox_2d'][2]
        and a['bbox_2d'][1]<=r['bbox_2d'][1]<=r['bbox_2d'][3]<=a['bbox_2d'][3]
        for a in repeated)]


def repeated_labels(rgb, rows, known, rendered):
    """Nominate complete repeated labels, including ink touching a node frame."""
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);height,width=gray.shape
    seen=set();result=[]
    for row in rows:
        text=row.get('text','').strip()
        if (not 2<=len(text)<=8 or not text.isalnum() or row.get('kind','text')!='text'
                or any(r.get('script','normal')!='normal' for r in row.get('runs',[]))):continue
        identity=(text,row.get('rotation_deg',0),row.get('font'),row.get('bold'),row.get('italic'))
        if identity in seen:continue
        seen.add(identity)
        bounds=np.asarray(row['bbox_2d'],float)
        lo=np.maximum(0,np.floor(bounds[:2])).astype(int)
        hi=np.minimum([width,height],np.ceil(bounds[2:]+1)).astype(int)
        kernel=rgb[lo[1]:hi[1],lo[0]:hi[0]];h,w=kernel.shape[:2]
        if min(h,w)<3 or h>=height or w>=width:continue
        border=np.concatenate([kernel[0],kernel[-1],kernel[:,0],kernel[:,-1]])
        background=np.median(border,axis=0)
        contrast=np.linalg.norm(kernel.astype(float)-background,axis=2)
        if contrast.max()<8:continue
        foreground=contrast>.5*contrast.max()
        if foreground.sum()<5:continue
        score=cv2.matchTemplate(gray,cv2.cvtColor(kernel,cv2.COLOR_RGB2GRAY),cv2.TM_CCOEFF_NORMED)
        ys,xs=np.nonzero((score>=.90)&(score>=cv2.dilate(score,np.ones((3,3),'uint8'))))
        peaks=sorted(zip(ys,xs),key=lambda p:-float(score[p]))
        for y,x in peaks:
            if np.mean(known[y:y+h,x:x+w][foreground])>.25:continue
            b=[int(x),int(y),int(x+w-1),int(y+h-1)]
            if any(max(0,min(b[2],r['bbox_2d'][2])-max(b[0],r['bbox_2d'][0])+1)
                   *max(0,min(b[3],r['bbox_2d'][3])-max(b[1],r['bbox_2d'][1])+1)>.5*w*h for r in result):continue
            pigment=np.median(rgb[y:y+h,x:x+w][foreground],axis=0)
            coverage=0.
            if rendered is not None:
                x0,y0=max(0,x-2),max(0,y-2);x1,y1=min(width,x+w+2),min(height,y+h+2)
                match=(np.linalg.norm(rendered[y0:y1,x0:x1].astype(float)-pigment,axis=2)<45).astype('uint8')
                covered=cv2.dilate(match,np.ones((5,5),'uint8'))>0
                coverage=float(np.mean(covered[y-y0:y-y0+h,x-x0:x-x0+w][foreground]))
                if coverage>=.65:continue
            result.append({'bbox_2d':b,'ink_pixels':int(foreground.sum()),'native_coverage':coverage,
                'color':'#'+''.join('%02X'%round(v) for v in pigment),
                'complete_label_reference':{'source_text_id':row['id'],'text':text,
                    'source_bbox_2d':row['bbox_2d'],'rotation_deg':row.get('rotation_deg',0),
                    'source_pattern_similarity':round(float(score[y,x]),4)}})
    result.sort(key=lambda r:(-r['complete_label_reference']['source_pattern_similarity'],r['bbox_2d']))
    return result


def validate(value, aliases, fonts, *, allow_omissions=False):
    if not isinstance(value, dict) or not all(isinstance(value.get(k), list) for k in ('missing_text', 'other')):
        raise ValueError('Account for every isolated source mark as missing text or another diagram mark')
    seen = set()
    accepted = []; rejected = []
    for item in value['missing_text']:
        if not isinstance(item, dict):
            raise ValueError('Invalid missing text receipt')
        ids = item.get('candidate_ids', [])
        if not isinstance(ids, list) or not ids or not all(isinstance(k, str) for k in ids) or not set(ids) <= set(aliases) or len(set(ids)) != len(ids) or set(ids) & seen:
            raise ValueError('Each source mark needs exactly one disposition')
        text = item.get('text')
        if (not isinstance(text, str) or not 1 <= len(text) <= 8 or any(unicodedata.category(c).startswith('C') for c in text)
                or item.get('font') not in fonts or type(item.get('bold')) is not bool or type(item.get('italic')) is not bool
                or not isinstance(item.get('reason'), str) or not item['reason'].strip()):
            raise ValueError('Name visible text and its existing font style; do not invent an annotation')
        seen.update(ids)
        complete=all('complete_label_reference' in aliases[k] for k in ids)
        standalone=(len(text)==1 and text.isalnum() and item.get('unit')=='standalone_label')
        if not is_symbol(text) and not standalone and not (complete and len(ids)==1):
            rejected.extend({'candidate_id': ident, 'classification':'uncertain',
                'reason':'Only complete source labels and isolated typography can be recovered',
                'rejected_proposal':text} for ident in ids)
        else:
            accepted.append(item)
    for item in value['other']:
        if not isinstance(item, dict):
            raise ValueError('Invalid source mark receipt')
        ident = item.get('candidate_id')
        if ident not in aliases or ident in seen or item.get('classification') not in ('diagram', 'already_recovered', 'uncertain') or not isinstance(item.get('reason'), str) or not item['reason'].strip():
            raise ValueError('Explain each rejected or uncertain source mark')
        seen.add(ident)
    if seen != set(aliases):
        if not allow_omissions:
            raise ValueError('Missing source mark receipts: '+','.join(sorted(set(aliases)-seen)))
        value['other'].extend({'candidate_id': ident, 'classification': 'uncertain',
            'reason': 'Model did not classify this source mark; no text is added',
            'review_status': 'model_omitted_source_mark'} for ident in sorted(set(aliases)-seen))
    value['missing_text'] = accepted; value['other'].extend(rejected)
    return value


def observe(source, native, rows, folder, runtime, save, *, maximum=64):
    proposed = candidates(source, rows, native)
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    save(folder/'candidates.json', {'protocol': PROTOCOL, 'candidates': proposed,
        'considered': min(maximum, len(proposed)), 'unreviewed': max(0, len(proposed)-maximum)})
    fonts = sorted({r.get('font', 'Arial') for r in rows} | {'Arial'})
    additions = []; receipts = []; calls = 0; used = {r['id'] for r in rows}
    for start in range(0, min(maximum, len(proposed)), 16):
        batch = proposed[start:min(start+16,maximum)]; local = folder/('batch_%03d' % (start//16)); local.mkdir(exist_ok=True)
        aliases = {'C%d' % (i+1): row for i, row in enumerate(batch)}
        panel = Image.new('RGB', (960, math.ceil(len(batch)/2)*210), 'white'); draw = ImageDraw.Draw(panel)
        for i, (ident, row) in enumerate(aliases.items()):
            b = np.asarray(row['bbox_2d']); span = b[2:]-b[:2]+1
            pad = max(20., 1.2*max(span))
            box = tuple(np.r_[np.maximum(0, np.floor(b[:2]-pad)), np.minimum(source.size, np.ceil(b[2:]+pad+1))].astype(int))
            left = (i%2)*480; top = (i//2)*210
            for column, (label, im) in enumerate((('CLEAN', source), ('NATIVE without text', native))):
                crop = im.crop(box); crop.thumbnail((230, 178), Image.Resampling.LANCZOS)
                # Scale up too: an isolated glyph must remain readable.
                ratio = min(230/crop.width, 178/crop.height)
                crop = crop.resize((max(1, round(crop.width*ratio)), max(1, round(crop.height*ratio))), Image.Resampling.LANCZOS)
                panel.paste(crop, (left+column*240, top+24))
                sx,sy=crop.width/(box[2]-box[0]),crop.height/(box[3]-box[1])
                target=[left+column*240+(b[0]-box[0]-2)*sx,top+24+(b[1]-box[1]-2)*sy,
                        left+column*240+(b[2]-box[0]+3)*sx,top+24+(b[3]-box[1]+3)*sy]
                draw.rectangle(target,outline='#D020D0',width=1)
                draw.text((left+column*240+3, top+4), ident+' '+label, fill='black')
        panel.save(local/'source_native.png')
        try:
            value = runtime.call(local/'calls', 'coverage', 'Inspect original typography. Return only JSON.',
            {'protocol': PROTOCOL, 'task': 'Read missing isolated typography or a complete repeated label at the CENTRAL source marks.',
             'candidates': aliases, 'available_fonts': fonts,
             'rules': ['Original pixels establish existence; native absence alone is insufficient.',
                'The magenta box marks the nominated ink component. Read that component; surrounding pixels provide context and may contain clipped neighbouring words.',
                'Distinguish mathematical/textual operators and labels from arrowheads, crossings, node boundaries, legend samples and decorative marks. Never repair or recognize arrows here.',
                'Read each mark in context. Recover complete readable text or an independent label; leave incomplete or ambiguous content out.',
                'Existing text source boxes were already excluded. Do not retranscribe those labels or duplicate text inside images. If a source mark is ambiguous return uncertain.',
                'A multi-dot symbol may combine adjacent candidate_ids. Treat each physical glyph exactly once. Name the actual visible character, not a visually similar Latin letter or an invented formula.',
                'Use source font style and one supplied font family. Geometry and size are measured independently from original pixels; do not return fitted coordinates.'],
             'output': {'missing_text': [{'candidate_ids': ['C1'], 'text': 'visible glyph', 'font': fonts[0],
                        'bold': False, 'italic': False, 'unit': 'typography|standalone_label', 'reason': 'source typography and missing native ink'}],
                        'other': [{'candidate_id': 'C2', 'classification': 'diagram|already_recovered|uncertain', 'reason': 'source evidence'}]}},
            [('CLEAN source context and aligned native comparison', local/'source_native.png')],
                lambda v: validate(v, aliases, fonts, allow_omissions=True), 8192)
        except (RuntimeError, ValueError, requests.RequestException) as error:
            # Optional additions cannot prevent export of the supplied text
            # product or independently verified source geometry.
            value = {'missing_text': [], 'other': [
                {'candidate_id': ident, 'classification': 'uncertain',
                 'reason': 'Source-symbol review failed; no text is added'} for ident in aliases],
                'review_error': {'type': type(error).__name__, 'message': str(error)[:500]}}
        calls += 1; receipts.append(value)
        for item in value['missing_text']:
            members = [aliases[k] for k in item['candidate_ids']]
            boxes = np.asarray([r['bbox_2d'] for r in members]); b = [*boxes[:, :2].min(0), *boxes[:, 2:].max(0)]
            index = len(additions)+1; ident = 'SOURCE_TEXT_%04d' % index
            while ident in used:
                index += 1; ident = 'SOURCE_TEXT_%04d' % index
            used.add(ident)
            additions.append({'id': ident, 'text': item['text'], 'kind': 'text', 'bbox_2d': [float(v) for v in b],
                'font': item['font'], 'bold': item['bold'], 'italic': item['italic'], 'color': members[0]['color'],
                'rotation_deg': members[0].get('complete_label_reference',{}).get('rotation_deg',0.),
                'layout_mode': 'single_line', 'runs': [{'text': item['text'], 'script': 'normal'}],
                'source_coverage': {'protocol': PROTOCOL, 'reason': item['reason'], 'candidate_ids': item['candidate_ids'],
                    **({'complete_label_reference':members[0]['complete_label_reference']}
                       if len(members)==1 and 'complete_label_reference' in members[0] else {})}})
    audit = {'protocol': PROTOCOL, 'candidates': len(proposed), 'model_calls': calls, 'added_text': additions,
             'unreviewed_candidates': max(0, len(proposed)-maximum), 'receipts': receipts, 'arrow_geometry_changed': False}
    audit['uncertain_source_marks'] = sum(u['classification']=='uncertain' for v in receipts for u in v['other'])
    save(folder/'coverage.json', audit)
    return additions, audit


def verify_glyphs(source, rendered, rows):
    """A new glyph must explain original ink, including its empty interior."""
    original = np.asarray(source.convert('RGB')); probe = np.asarray(rendered.convert('RGB'))
    kept = []; audit = []
    for row in rows:
        b = np.asarray(row['bbox_2d'], float)
        lo = np.maximum(0, np.floor(b[:2]-2)).astype(int)
        hi = np.minimum(source.size, np.ceil(b[2:]+3)).astype(int)
        pigment = np.array([int(row['color'][i:i+2],16) for i in (1,3,5)])
        masks = [(np.linalg.norm(im[lo[1]:hi[1],lo[0]:hi[0]].astype(float)-pigment,axis=2)<85)
                 for im in (original,probe)]
        a,c = masks; intersection = int((a&c).sum()); union = int((a|c).sum())
        iou = intersection/max(1,union); precision = intersection/max(1,int(c.sum())); recall = intersection/max(1,int(a.sum()))
        # Raster phase and substituted fonts can move a thin stroke by one
        # source pixel. Require support in both directions within that pixel;
        # a blank rendering or unrelated mark still cannot explain the ink.
        from scipy.ndimage import distance_transform_edt
        near_precision=float(np.mean(distance_transform_edt(~a)[c]<=1)) if a.any() and c.any() else 0.
        near_recall=float(np.mean(distance_transform_edt(~c)[a]<=1)) if a.any() and c.any() else 0.
        # The pixel gate distinguishes an uncertain isolated mark from a
        # diagram crossing. It must not discard an independently reread whole
        # label just because the installed font approximates its source face.
        reference=row.get('source_coverage',{}).get('complete_label_reference',{})
        source_identity=bool(reference and reference.get('text')==row.get('text'))
        accepted = source_identity or (iou>=.55 and precision>=.7 and recall>=.65) or (near_precision>=.85 and near_recall>=.7)
        audit.append({'id':row['id'],'text':row['text'],'accepted':accepted,'source_ink_iou':round(iou,4),
                      'precision':round(precision,4),'recall':round(recall,4),
                      'one_pixel_precision':round(near_precision,4),'one_pixel_recall':round(near_recall,4),
                      'complete_source_identity_confirmed':source_identity,
                      'font_approximation_reported_without_retiring_source_text':source_identity})
        if accepted:kept.append(row)
    return kept,audit
