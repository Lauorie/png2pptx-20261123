"""Whole-target locators retain global provenance; local masks are measurements."""
import numpy as np
from PIL import ImageDraw
import visual


def boxes(group, view):
    result=[]
    matrix=np.asarray(view['global_to_view'],float)
    for candidate in group:
        if 'box' not in candidate:
            raise ValueError('A target bbox requires the original global candidate, not its local mask')
        x0,y0,x1,y1=candidate['box']
        p=np.array([[x0,y0,1],[x1,y1,1]])@matrix.T
        result.append(dict(id=candidate['id'],bbox=np.round(p[:,:2].reshape(-1),1).tolist(),
                           source='global_candidate',extent='approximate; may be partial'))
    return result


def display_box(box, size, padding_fraction=0.):
    """Expand annotations only; bbox coordinates in model records stay unchanged."""
    x0,y0,x1,y1=map(float,box)
    dx=(x1-x0)*padding_fraction;dy=(y1-y0)*padding_fraction
    return [max(0.,x0-dx),max(0.,y0-dy),min(size[0]-1.,x1+dx),min(size[1]-1.,y1+dy)]


def draw(clean, targets, path, padding_fraction=0., mode='rectangle'):
    if mode not in ('rectangle','corners'):raise ValueError('Unknown locator display mode')
    if not 0 <= padding_fraction < 1:raise ValueError('Invalid locator padding')
    image=clean.copy();d=ImageDraw.Draw(image)
    annotations=[];placed=[]
    for i,target in enumerate(targets):
        box=display_box(target['bbox'],image.size,padding_fraction);color=visual.color(i)
        x0,y0,x1,y1=box
        if mode=='rectangle':
            d.rectangle(box,outline=color,width=2)
        else:
            # Short corner marks avoid painting a fake long shaft over thin ink.
            leg=min(14.,(x1-x0)/4,(y1-y0)/4)
            for x,y,sx,sy in ((x0,y0,1,1),(x1,y0,-1,1),(x0,y1,1,-1),(x1,y1,-1,-1)):
                d.line([(x+sx*leg,y),(x,y),(x,y+sy*leg)],fill=color,width=2)
        if padding_fraction==0 and mode=='rectangle':
            visual.label(d,(x0,y0-visual.font(image).size-7),target['id'],color,image)
            label_box=None
        else:
            # Place the identifier outside the box in the least ink-covered
            # available position, with a collision penalty for earlier labels.
            f=visual.font(image);bb=d.textbbox((0,0),target['id'],font=f)
            w,h=bb[2]-bb[0]+4,bb[3]-bb[1]+4
            positions=[(x0,y0-h-3),(x1-w,y0-h-3),(x0,y1+3),(x1-w,y1+3),
                       (x0-w-3,y0),(x1+3,y0)]
            gray=np.asarray(clean.convert('L'))
            def candidate(p):
                x=int(max(0,min(image.width-w,p[0])));y=int(max(0,min(image.height-h,p[1])))
                rect=[x,y,x+w,y+h]
                overlap=sum(max(0,min(rect[2],b[2])-max(x,b[0]))*max(0,min(rect[3],b[3])-max(y,b[1])) for b in placed)
                ink=float(np.mean(gray[y:y+h,x:x+w]<180))
                return ink+overlap/(w*h),rect
            _,label_box=min(map(candidate,positions),key=lambda item:item[0])
            placed.append(label_box)
            d.text((label_box[0]+2,label_box[1]+2-bb[1]),target['id'],font=f,fill=color,
                   stroke_width=1,stroke_fill='white')
        annotations.append(dict(id=target['id'],bbox=list(target['bbox']),display_bbox=box,label_bbox=label_box))
    image.save(path)
    return annotations


def configured_draw(clean, targets, path, config=None):
    config=config or {}
    return draw(clean,targets,path,config.get('bbox_display_padding',0.),
                config.get('bbox_display_mode','rectangle'))


def display_instruction(config):
    if not config.get('bbox_display_padding') and config.get('bbox_display_mode','rectangle')=='rectangle':return ''
    return ('Colored box marks and labels are annotations, not source strokes. The marks are expanded outward '
            f"by {100*config.get('bbox_display_padding',0.):g}% of the original width/height on EACH side for visibility; "
            'the numeric bbox remains the original evidence extent. Neither annotation edges nor bbox edges are route geometry. '
            'Use CLEAN to see unobscured strokes; CONTEXT and crop_in_context locate the same image region.')


def apply(prompt, images, group, view, clean, folder):
    """Replace one locator image rather than increasing image/request count."""
    targets=boxes(group,view);path=folder/'input/target_boxes.png';draw(clean,targets,path)
    images=list(images)
    index=next(i for i,(name,_) in enumerate(images) if name.startswith('GLOBAL HINTS'))
    images[index]=('TARGET BOXES: approximate global candidate extents in CLEAN pixels',path)
    prompt['target_boxes']=targets
    prompt['instructions'].insert(1,
        'TARGET BOXES locate the supplied G candidates on CLEAN. These global candidate bounds can be more complete than a local M mask, but are not exact endpoints or proof of a real arrow. Use the original pixels to merge, split, exclude, or correct the target extent. A local mask may contain only the head; do not shrink the whole target to it. Box overlap, touching boxes and nearby same-colored strokes do not establish connectivity.')
    return images
