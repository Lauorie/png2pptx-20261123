"""Scale-aware, non-semantic endpoint guides for fixed-geometry style review.

Only the locator image changes. Source coordinates and source/render pixels stay
unchanged. S/E denote physical start/end, never an inferred arrow direction.
"""
import math
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
COLORS = ['#c62f37', '#2850b8', '#24763e', '#9736a6', '#b75c08', '#157b85']
INSTRUCTION = ('LOCATORS is annotation only: numbered tags such as 1S and 1E mean the physical start and end of path 1 in locator_paths. '
               'Corner brackets center on the exact endpoint; dashed leaders connect displaced labels to it. '
               'Several tags sharing one bracket refer to the same physical position. '
               'Colored paths, brackets, labels and leaders are added guides, not source terminal marks. '
               'Read endpoint appearance from CLEAN at the corresponding coordinates. Return the original full path IDs.')


def points(row):
    return row.get('points') or [row['start'], *row.get('via', []), row['end']]


def integral(mask):
    return np.pad(np.cumsum(np.cumsum(mask.astype(float), axis=0), axis=1), ((1, 0), (1, 0)))


def area(ii, box):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]


def intersects(a, b, pad=0):
    return a[0] < b[2]+pad and a[2] > b[0]-pad and a[1] < b[3]+pad and a[3] > b[1]-pad


def dashed(draw, a, b, fill, width, unit):
    length = math.dist(a, b)
    if not length: return
    for t in np.arange(0, length, max(2, 5*unit)):
        q = min(t + max(1, 2*unit), length)
        p0 = tuple(a[k] + (b[k]-a[k])*t/length for k in (0, 1))
        p1 = tuple(a[k] + (b[k]-a[k])*q/length for k in (0, 1))
        draw.line([p0, p1], fill=fill, width=width)


def render(source, rows):
    source = source.convert('RGB')
    w, h = source.size
    # Relative to a model-view long edge of ~1080; no arbitrary font-size cap
    # that would make a high-resolution image's text disappear after resizing.
    unit = max(w, h)/1080
    fs = max(12, round(16*unit))
    font = ImageFont.truetype(FONT, fs)
    pad = max(2, round(3*unit))
    gap = max(2, round(4*unit))
    stroke = max(1, round(unit))
    groups = []
    mapping = {str(i+1): r['id'] for i, r in enumerate(rows)}
    for i, row in enumerate(rows):
        for endpoint, p in [('S', points(row)[0]), ('E', points(row)[-1])]:
            group = next((g for g in groups if math.dist(g['point'], p) < .25), None)
            if group is None:
                group = dict(point=list(p), tags=[]); groups.append(group)
            group['tags'].append(dict(text=f'{i+1}{endpoint}', path_id=row['id'], end=endpoint, color=COLORS[i%len(COLORS)]))
    scratch = ImageDraw.Draw(source.copy())
    protected = Image.new('L', (w,h)); pd = ImageDraw.Draw(protected)
    for g in groups:
        x,y=g['point']
        nearest=min((math.dist(g['point'], other['point']) for other in groups if other is not g), default=max(w,h))
        g['radius'] = max(1.5, min(max(3, 4*unit), nearest/3))
        clearance=max(g['radius']+gap, 8*unit)
        pd.ellipse((x-clearance,y-clearance,x+clearance,y+clearance),fill=255)
        g['text'] = ' '.join(t['text'] for t in g['tags'])
        bb = scratch.textbbox((0,0),g['text'],font=font)
        g['size'] = (bb[2]-bb[0]+2*pad, bb[3]-bb[1]+2*pad)
    # Favor flat background over letters, diagram boundaries and linework.
    rgb=np.asarray(source).astype(float)
    edge=np.zeros((h,w)); edge[:,1:]=np.max(np.abs(rgb[:,1:]-rgb[:,:-1]),axis=2)
    edge[1:]=np.maximum(edge[1:],np.max(np.abs(rgb[1:]-rgb[:-1]),axis=2))
    # Keep text away from the interiors immediately beside diagram boundaries,
    # not just from a one-pixel contour (e.g. a label inside a small round node).
    edge_mask=Image.fromarray(((edge>25)*255).astype('uint8'))
    dilation=max(3,round(9*unit)//2*2+1)
    edge_ii=integral(np.asarray(edge_mask.filter(ImageFilter.MaxFilter(dilation)))>0)
    protect_ii=integral(np.asarray(protected)>0)
    pathmask=Image.new('L',(w,h)); pathdraw=ImageDraw.Draw(pathmask)
    for r in rows:pathdraw.line([tuple(p) for p in points(r)],fill=255,width=max(3,round(5*unit)))
    path_ii=integral(np.asarray(pathmask)>0)
    occupied=[]
    # Place crowded/shared positions first. All coordinates remain in source pixels.
    order=sorted(groups,key=lambda g:(-len(g['tags']),min((math.dist(g['point'],o['point']) for o in groups if o is not g),default=1e9)))
    for g in order:
        x,y=g['point']; bw,bh=g['size']; candidates=set()
        for distance in [12,20,32,48,72,104,152,216,320,480,720,1080]:
            d=max(g['radius']+gap+max(bw,bh)/2, distance*max(.6,unit))
            for a in range(0,360,15):
                rad=math.radians(a)
                left=round(min(max(pad,x+math.cos(rad)*d-bw/2),w-bw-pad))
                top=round(min(max(pad,y+math.sin(rad)*d-bh/2),h-bh-pad))
                if left<0 or top<0 or left+bw>w or top+bh>h:continue
                candidates.add((left,top,left+bw,top+bh))
        scored=[]
        for box in candidates:
            collisions=sum(intersects(box,o,gap) for o in occupied)
            protect=area(protect_ii,box)
            cx=(box[0]+box[2])/2; cy=(box[1]+box[3])/2
            distance=math.hypot(cx-x,cy-y)/max(.6,unit)
            box_area=bw*bh
            # Density rather than raw pixel counts keeps behavior stable across
            # resolutions. Keep labels local instead of preferring distant white
            # space and drawing long leaders across neighboring endpoints.
            cost=(collisions*1e8 + protect*1e6 + area(path_ii,box)/box_area*200
                  + area(edge_ii,box)/box_area*150 + distance*2.5)
            scored.append((cost,box))
        if not scored:raise ValueError(f'Image {source.size} has no space for endpoint tag {g["text"]}')
        _,box=min(scored)
        g['box']=list(box);g['overlapping_prior_labels']=sum(intersects(box,o,gap) for o in occupied)
        g['protected_pixels_covered']=int(area(protect_ii,box)); occupied.append(box)
        anchor=(min(max(x,box[0]),box[2]),min(max(y,box[1]),box[3]))
        d=math.dist((x,y),anchor); r=g['radius']+max(1,unit)
        g['leader']=[list((x+(anchor[0]-x)*r/d,y+(anchor[1]-y)*r/d)) if d else [x,y],list(anchor)]
    image=source.copy();draw=ImageDraw.Draw(image)
    for i,row in enumerate(rows):draw.line([tuple(p) for p in points(row)],fill=COLORS[i%len(COLORS)],width=stroke)
    for g in groups:
        color=g['tags'][0]['color'] if len(g['tags'])==1 else '#555555'
        dashed(draw,*g['leader'],color,stroke,max(.6,unit))
    for g in groups:
        x,y=g['point'];r=g['radius'];k=max(1,min(r*.6,3*unit))
        color=g['tags'][0]['color'] if len(g['tags'])==1 else '#555555'
        # Four separated corners leave the endpoint's center unpainted.
        for sx in (-1,1):
            for sy in (-1,1):
                corner=(x+sx*r,y+sy*r)
                draw.line([(corner[0]-sx*k,corner[1]),corner,(corner[0],corner[1]-sy*k)],fill=color,width=stroke)
    for g in groups:
        box=g['box'];draw.rectangle(box,fill='white',outline='#a6a6a6',width=stroke)
        x=box[0]+pad
        for tag in g['tags']:
            bb=draw.textbbox((0,0),tag['text'],font=font)
            draw.text((x,box[1]+pad-bb[1]),tag['text'],font=font,fill=tag['color'])
            x+=draw.textlength(tag['text']+' ',font=font)
    return image, dict(size=[w,h],font_px=fs,scale_unit=unit,locator_paths=mapping,groups=groups,
                       label_overlap_count=sum(g['overlapping_prior_labels'] for g in groups),
                       protected_pixels_covered=sum(g['protected_pixels_covered'] for g in groups))


def attach_prompt(prompt, layout):
    """Bind short image tags to full IDs, preserving all model output IDs."""
    if layout is None:
        return prompt
    mapping = layout['locator_paths']
    prompt['locator_paths'] = mapping
    reverse = {path_id: tag for tag, path_id in mapping.items()}
    for target in prompt.get('targets', []):
        if target['id'] in reverse:
            tag = reverse[target['id']]
            target['locator_tags'] = {'start': tag+'S', 'end': tag+'E'}
    prompt['instructions'].append(INSTRUCTION)
    return prompt
