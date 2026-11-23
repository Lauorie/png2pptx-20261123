"""Re-request only malformed ordinary multiline occurrences before layout.

Keep the exact original object on any failure. Successful children inherit
content and colored runs by slicing, not by accepting new style predictions.
"""
from copy import deepcopy
import re

from ..common import save, valid_box, intersect
from . import model
from .controller import frame, mapped_box
from .reflow import slice_runs
from .response_validation import numeric_box


def fallback_lines(row):
    """Render-only, word-preserving line partition; not recovered source geometry."""
    text=row['text'];words=list(re.finditer(r'\S+',text))
    count=min(max(1,int(row.get('visual_line_count',1))),len(words))
    if count<=1:return [deepcopy(row['runs'])]
    # Balanced partition with no word splits. Runs/content in the scene remain
    # untouched; only the approximate renderer gets these line spans.
    states={(0,0):(0.,[])};target=len(text)/count
    for line in range(1,count+1):
        for end in range(line,len(words)+1):
            choices=[]
            for start in range(line-1,end):
                previous=states.get((line-1,start))
                if previous is None:continue
                lo=words[start].start();hi=words[end-1].end()
                choices.append((previous[0]+(hi-lo-target)**2,previous[1]+[(lo,hi)]))
            if choices:states[(line,end)]=min(choices,key=lambda x:x[0])
    return [slice_runs(row['runs'],lo,hi) for lo,hi in states[(count,len(words))][1]]


def validate(raw, row, size):
    if not isinstance(raw, dict) or raw.get('parent_id') != row['id']:
        raise ValueError('Return only the supplied parent_id')
    lines=raw.get('lines')
    if not isinstance(lines,list) or len(lines)!=row['visual_line_count']:
        raise ValueError('Return one line record per visual baseline')
    text=row['text'];cursor=0;checked=[]
    for line in lines:
        if not isinstance(line,dict) or set(line)-{'text','bbox_2d'}:
            raise ValueError('Each line has text and bbox_2d only')
        value=line.get('text')
        if not isinstance(value,str) or not value.strip() or '\n' in value or '\r' in value:
            raise ValueError('Each line needs one nonempty visible baseline')
        box=numeric_box(line.get('bbox_2d'))
        if not valid_box(box,size):raise ValueError('Invalid per-line bbox')
        while cursor<len(text) and text[cursor].isspace():cursor+=1
        # Tolerate space normalization, but never accept changed characters,
        # punctuation, reordered/repeated lines or omitted content.
        pattern=r'\s+'.join(re.escape(w) for w in value.split())
        match=re.compile(pattern).match(text,cursor)
        if not match:raise ValueError('Line text must partition the original transcription in order')
        end=match.end()
        if end<len(text) and not text[end].isspace() and row.get('layout_mode')!='multiline':
            raise ValueError('Do not split a word across text boxes')
        checked.append({'span':[cursor,end],'bbox_2d':box})
        cursor=end
    if text[cursor:].strip():raise ValueError('Line partition omitted trailing text')
    for i,a in enumerate(checked):
        for b in checked[i+1:]:
            box=a['bbox_2d'];other=b['bbox_2d']
            area=lambda q:(q[2]-q[0])*(q[3]-q[1])
            if intersect(box,other)>.5*min(area(box),area(other)):
                raise ValueError('Line boxes overlap excessively; return distinct baselines')
    return {'parent_id':row['id'],'lines':checked}


def repair(inventory, source, folder, runtime, opt, progress=lambda *a,**k:None):
    result=deepcopy(inventory);output=[];audit=[];warnings=[]
    reserved={r['id'] for r in inventory['texts']}
    for row in inventory['texts']:
        if row.get('kind')=='formula' or row.get('visual_line_count',1)<=1:
            output.append(deepcopy(row));continue
        root=folder/row['id']
        progress('multiline_target_retry',id=row['id'],lines=row['visual_line_count'])
        try:
            f=frame(source,root/'input',{**opt,'crop_min_padding_px':24},row['bbox_2d'])
            prompt={'task':'Split this ONE ordinary text occurrence into its visible baselines.',
                'parent_id':row['id'],'text':row['text'],'expected_lines':row['visual_line_count'],
                'image_size':f['size'],'rough_bbox':mapped_box(row['bbox_2d'],f),
                'rotation_deg':row['rotation_deg'],
                'rules':['Return ONLY this occurrence; neighboring text is context, not new targets.',
                    'Return one tight complete ink bbox per visible line, in reading order. '
                    'Coordinates are CLEAN crop pixel centers [xmin,ymin,xmax,ymax].',
                    'Partition the supplied text; preserve every character and punctuation. '
                    'Do not reword, correct content, split words or output colors/font/LaTeX.',
                    'If an explicitly confirmed multiline transcription accidentally joins the end of one '
                    'visible baseline directly to the start of the next without a space, split at the '
                    'ACTUAL visible line boundary. Do not insert/delete characters or split within a '
                    'word that is physically on the same baseline.',
                    'Do not return one combined bbox for multiple lines. '
                    'If the supplied content or line count cannot be located, return lines=[]; do not guess.'],
                'response_format':{'parent_id':row['id'],'lines':[
                    {'text':'first visible line','bbox_2d':[10,10,100,30]},
                    {'text':'second visible line','bbox_2d':[10,40,100,60]}]}}
            reply=model.ask(runtime,root,prompt,[('CLEAN enlarged target crop',root/'input/clean.png')],
                lambda value:validate(value,row,f['size']),opt)
            children=[]
            for index,line in enumerate(reply['lines'],1):
                lo,hi=line['span'];box=mapped_box(line['bbox_2d'],f,True)
                if not valid_box(box,f['source_size']):raise ValueError('Mapped line bbox outside source')
                ident=f"{row['id']}.L{index:02d}"
                if ident in reserved:raise ValueError('Child ID already exists')
                child=deepcopy(row)
                child.update(id=ident,parent_id=row['id'],semantic_unit_id=row.get('semantic_unit_id',row['id']),
                    text=row['text'][lo:hi],runs=slice_runs(row['runs'],lo,hi),bbox_2d=box,
                    visual_line_count=1,line_layout_status='split_baseline',
                    layout_mode='single_line',multiline_render_fallback=False,
                    multiline_retry={'status':'split','parent_id':row['id'],'character_span':[lo,hi]})
                if ''.join(r['text'] for r in child['runs'])!=child['text']:
                    raise ValueError('Original runs do not agree with original text')
                children.append(child)
            output.extend(children);reserved.update(c['id'] for c in children)
            audit.append({'id':row['id'],'status':'split','children':children})
        except Exception as exc:
            # User policy: failure retains bbox/content/style exactly. Audit is
            # separate so no synthetic geometry or silent text deletion occurs.
            output.append(deepcopy(row))
            record={'id':row['id'],'status':'kept_original','reason':str(exc),'bbox_2d':row['bbox_2d']}
            audit.append(record);warnings.append({'stage':'multiline_target_retry',**record})
    result['texts']=output
    result['regions_needing_review']=result.get('regions_needing_review',[])+warnings
    save(folder/'audit.json',{'targets':len(audit),'split':sum(r['status']=='split' for r in audit),
        'kept_original':len(warnings),'records':audit})
    return result
