"""Short visual labels with collision-free placement; detection boxes unchanged."""
import copy
import json
import math
from pathlib import Path
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def intersection(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))


def expand(b,p):return [b[0]-p,b[1]-p,b[2]+p,b[3]+p]


def labels_for(rows,context=(),mode='report'):
    """Report I0063 -> 63; request aliases 1..N and r1..M are backend mapped."""
    result={};used=set()
    for i,row in enumerate(rows,1):
        match=re.fullmatch(r'I0*(\d+)',str(row.get('display_id',''))) if mode=='report' else None
        label=str(int(match.group(1))) if match else str(i)
        if label in used:raise ValueError('Duplicate display number')
        result[row['id']]=label;used.add(label)
    for i,row in enumerate(context,1):
        if row['id'] in result:raise ValueError('Context and focus must be disjoint')
        result[row['id']]='r'+str(i)
    return result


def short_request_rows(focus,context=()):
    """Use the same aliases in image and prompt; restore IDs after validation."""
    mapping=labels_for(focus,context,mode='request')
    def convert(rows):
        out=[]
        for r in rows:
            x=copy.deepcopy(r);x['id']=mapping[r['id']];out.append(x)
        return out
    return convert(focus),convert(context),{v:k for k,v in mapping.items()}


def restore_answer_ids(answer,alias_to_id):
    answer=copy.deepcopy(answer)
    for o in answer.get('objects',[]):
        o['source_ids']=[alias_to_id[str(k)] for k in o['source_ids']]
        if o.get('original_id') is not None:o['original_id']=alias_to_id[str(o['original_id'])]
    for kind in ['removed','unresolved']:
        for o in answer.get(kind,[]):o['source_id']=alias_to_id[str(o['source_id'])]
    return answer


def _nearest(point,box):
    return [min(box[2],max(box[0],point[0])),min(box[3],max(box[1],point[1]))]


def _integral(array):
    return np.pad(array.astype('float32').cumsum(0).cumsum(1),((1,0),(1,0)))


def _sum(integral,b):
    x0,y0,x1,y1=map(int,b)
    return float(integral[y1,x1]-integral[y0,x1]-integral[y1,x0]+integral[y0,x0])


def render(clean,rows,path,view,context=(),mode='report',show_labels=True,label_map=None,draw_box_outlines=True):
    if isinstance(clean,Image.Image):image=clean.convert('RGB')
    else:
        with Image.open(clean) as source:image=source.convert('RGB')
    width,height=image.size;original_size=image.size
    font_size=max(11,min(16,round(max(width,height)/150)))
    font=ImageFont.truetype(FONT,font_size)
    labels=label_map or labels_for(rows,context,mode)
    draw=ImageDraw.Draw(image);items=[]
    for row,is_context in [(r,False) for r in rows]+[(r,True) for r in context]:
        native=row['bbox_native'];b=view.to_local(native[:2])+view.to_local(native[2:])
        b=[max(0,min(float(v),(width-1,height-1)[i%2])) for i,v in enumerate(b)]
        if b[0]>=b[2] or b[1]>=b[3]:continue
        color='#777777' if is_context else '#8e24aa' if row.get('uncertain') or row.get('extent')=='partial' else '#1565c0' if row['kind'] in {'serial_chain','shared_network','dense_network','dense_region'} else '#d34b00'
        items.append({'source_id':row['id'],'display_id':row.get('display_id'),'label':labels[row['id']],
            'bbox_image_px':b,'bbox_native':list(native),'context':is_context,'color':row.get('visual_color',color)})
    # Source edges discourage tags on text or actual strokes. Compact target
    # boxes receive an additional protection cost; large L-shaped boxes do not
    # reserve their entire empty interior.
    gray=np.asarray(image.convert('L'),dtype='float32')
    edge=np.zeros((height,width),dtype='float32')
    edge[:,1:]+=np.abs(gray[:,1:]-gray[:,:-1])/255
    edge[1:,:]+=np.abs(gray[1:,:]-gray[:-1,:])/255
    edge_integral=_integral(edge)
    protected=np.zeros((height,width),dtype='float32')
    for item in items:
        b=item['bbox_image_px']
        if min(b[2]-b[0],b[3]-b[1])<=font_size*2.5 and (b[2]-b[0])*(b[3]-b[1])<width*height*.025:
            x0,y0,x1,y1=[int(v) for v in [max(0,b[0]-2),max(0,b[1]-2),min(width,b[2]+3),min(height,b[3]+3)]]
            protected[y0:y1,x0:x1]=1
    protected_integral=_integral(protected)
    placed=[]
    def crowd(item):
        b=expand(item['bbox_image_px'],40)
        return sum(intersection(b,x['bbox_image_px'])>0 for x in items)
    for item in sorted(items,key=lambda x:(x['context'],-crowd(x),x['bbox_image_px'][1],x['bbox_image_px'][0],x['label'])):
        if not show_labels:break
        label=item['label'];tb=draw.textbbox((0,0),label,font=font);tw=tb[2]-tb[0]+6;th=tb[3]-tb[1]+6
        b=item['bbox_image_px'];x0,y0,x1,y1=b;cx=(x0+x1)/2;cy=(y0+y1)/2
        candidates=[];seen=set()
        def add(x,y):
            x=int(round(max(0,min(x,width-tw))));y=int(round(max(0,min(y,height-th))))
            if (x,y) not in seen:seen.add((x,y));candidates.append([x,y,x+tw,y+th])
        for distance in [3,th+6,2*th+8,3*th+10,5*th+12]:
            for x in [cx-tw/2,x0,x1-tw]:add(x,y0-th-distance);add(x,y1+distance)
            for y in [cy-th/2,y0,y1-th]:add(x0-tw-distance,y);add(x1+distance,y)
        def cost(rect):
            if any(intersection(expand(rect,2),p['label_rect'])>0 for p in placed):return None
            center=[(rect[0]+rect[2])/2,(rect[1]+rect[3])/2];anchor=_nearest(center,b)
            distance=math.hypot(center[0]-anchor[0],center[1]-anchor[1])
            ink=_sum(edge_integral,rect)/(tw*th);target=_sum(protected_integral,rect)/(tw*th)
            return distance+ink*180+target*95
        valid=[(cost(c),c) for c in candidates];valid=[x for x in valid if x[0] is not None]
        if not valid:
            # Guaranteed collision avoidance on very crowded plots. Prefer a
            # visible tag farther away to hidden/overprinted IDs.
            for y in range(0,max(1,height-th+1),th+3):
                for x in range(0,max(1,width-tw+1),tw+3):
                    c=[x,y,x+tw,y+th];score=cost(c)
                    if score is not None:valid.append((score,c))
        if not valid:raise ValueError('No non-overlapping label position; split the visualization')
        _,rect=min(valid,key=lambda x:(x[0],x[1][1],x[1][0]))
        center=[(rect[0]+rect[2])/2,(rect[1]+rect[3])/2];anchor=_nearest(center,b);tag_point=_nearest(anchor,rect)
        item.update(label_rect=rect,leader=[tag_point,anchor],text_offset=[3-tb[0],3-tb[1]])
        placed.append(item)
    # Paint outlines first, all label badges last, so labels cannot be overdrawn.
    layer=Image.new('RGBA',image.size,(0,0,0,0));ld=ImageDraw.Draw(layer)
    if draw_box_outlines:
        for item in items:
            c=tuple(bytes.fromhex(item['color'][1:]))
            ld.rectangle(item['bbox_image_px'],outline=(*c,205),width=max(1,round(max(width,height)/1800)))
    for item in placed:
        if math.dist(*item['leader'])>2:
            c=tuple(bytes.fromhex(item['color'][1:]));ld.line([tuple(p) for p in item['leader']],fill=(*c,150),width=1)
    image=Image.alpha_composite(image.convert('RGBA'),layer).convert('RGB');draw=ImageDraw.Draw(image)
    for item in placed:
        b=item['label_rect'];draw.rounded_rectangle(b,radius=2,fill='white',outline=item['color'],width=1)
        draw.text((b[0]+item['text_offset'][0],b[1]+item['text_offset'][1]),item['label'],font=font,fill=item['color'])
    collisions=[(a['label'],b['label']) for i,a in enumerate(placed) for b in placed[i+1:] if intersection(a['label_rect'],b['label_rect'])>0]
    assert not collisions and image.size==original_size
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name('.'+path.name+'.tmp');image.save(tmp,format='PNG');tmp.replace(path)
    audit={'image_size_px':list(image.size),'font_size':font_size,'label_count':len(placed),'label_collisions':collisions,
        'detection_coordinates_unchanged':True,'mode':mode,'items':items}
    path.with_suffix('.labels.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    return audit
