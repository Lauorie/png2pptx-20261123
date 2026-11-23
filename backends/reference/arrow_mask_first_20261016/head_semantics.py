"""Simple text-defined head decisions with whole-target image context."""
import copy
from pathlib import Path
import numpy as np
from PIL import ImageDraw, ImageFont
import protocol
from transport.common import write_json as write, sha256 as sha

TASK = 'For each target line, decide whether each end has an arrowhead and classify its type.'


def validate(answer, ids):
    if isinstance(answer, dict) and set(answer) == set(ids):
        answer = dict(heads=answer)
    values = answer.get('heads') if isinstance(answer, dict) else None
    if not isinstance(values, dict) or set(values) != set(ids):
        raise ValueError('Return heads for all and only supplied R IDs')
    if any(not isinstance(v, list) or len(v) != 2 or any(h not in protocol.HEADS for h in v) for v in values.values()):
        raise ValueError('Each R needs [start type,end type] from head_types')
    return answer


def recover(crop, rows, supports, out, model, full_image=None, view=None, cpu_slot=None, context_description=''):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    result = copy.deepcopy(rows)
    targets = [r for r in result if r['kind'] != 'block_arrow']
    if not targets:
        return result, [], 0
    inputs = out / 'input'; inputs.mkdir(exist_ok=True)
    clean = inputs / 'clean.png'; crop.save(clean)
    marked = crop.copy(); draw = ImageDraw.Draw(marked)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', max(12, min(20, round(max(crop.size)/70))))
    global_marked = full_image.copy() if full_image is not None else None
    global_draw = ImageDraw.Draw(global_marked) if global_marked is not None else None
    records = []; prompt_targets = []; need_global = False
    for i, row in enumerate(targets, 1):
        rid = f'R{i}'; ends = []
        for end, suffix in [('start', 'a'), ('end', 'b')]:
            point = np.asarray(row[end], float); frame = 'local'; available = True
            painter = draw; image = marked
            if not (0 <= point[0] < crop.width and 0 <= point[1] < crop.height):
                if full_image is not None and view is not None:
                    point = (np.asarray(view['view_to_global']) @ np.r_[point, 1])[:2]
                    frame = 'global'; painter = global_draw; image = global_marked; need_global = True
                else:
                    available = False
            if available:
                available = bool(0 <= point[0] < image.width and 0 <= point[1] < image.height)
            if available:
                x, y = point
                painter.ellipse((x-2,y-2,x+2,y+2),outline='#0070c0',width=1)
                painter.text((max(0,min(image.width-45,x+7)),max(0,min(image.height-22,y+7))),rid+suffix,
                             font=font,fill='#0070c0',stroke_width=2,stroke_fill='white')
            ends.append(dict(end=end,label=rid+suffix,frame=frame,available=available,
                             point=np.round(point,1).tolist()))
        records.append(dict(id=rid,route=row['id'],ends=ends,image=str(inputs/'targets.png')))
        prompt_targets.append(dict(id=rid,ends=ends))
    path = inputs/'targets.png'; marked.save(path)
    images = [('CLEAN: original local context',clean),('TARGETS: same image with approximate endpoint labels only',path)]
    if need_global:
        path = inputs/'global_context.png'; global_marked.save(path)
        images.append(('GLOBAL: context for endpoints outside the local crop',path))
    q = dict(task=TASK,targets=prompt_targets,head_types=protocol.HEADS,
        instructions=[
            'Use CLEAN and the endpoint locators to follow each line. a/b mean its geometric first/last ends, not tail/tip. Labels are approximate; identify the visible head belonging to that line end even if its fitted point lies slightly inside an attached object.',
            'First decide if a head exists. A box edge, separate node/circle, or junction is not an arrowhead. Use none for headless ends and unknown when unclear. Types are definitions, not a required assortment. Return only head types; do not change geometry.'
        ],output_schema=dict(heads={r['id']:['a type','b type'] for r in records}))
    write(out/'sources.json',records)
    write(inputs/'images.json',[dict(label=l,path=str(p.resolve()),sha256=sha(p)) for l,p in images])
    issues = []
    try:
        answer = model.ask(out/'request',q,images,lambda a:validate(a,[r['id'] for r in records]))
        write(out/'answer.json',answer)
    except Exception as exc:
        write(out/'error.json',dict(error=str(exc)))
        answer = dict(heads={r['id']:['unknown','unknown'] for r in records})
        issues.append(dict(stage='head_types',error=str(exc),fallback='geometry retained; endpoint types unresolved'))
    for row, record in zip(targets, records):
        row['previous_head_hints'] = row.get('heads'); row['heads'] = answer['heads'][record['id']]
        row['head_source'] = 'glm_context_text'; row['head_request'] = str(out/'request')
        for end, h in zip(['start','end'],row['heads']):
            row[end+'_head'] = h not in ('none','unknown')
            row[end+'_head_type'] = h if h not in ('none','unknown') else None
        if 'unknown' in row['heads']:
            issues.append(dict(stage='head_types',route=row['id'],state='unresolved'))
    return result, issues, 1
