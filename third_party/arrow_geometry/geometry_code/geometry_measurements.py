"""Optional numeric mask evidence for GLM; never a final geometry/identity gate."""
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def line_hints(masks, size, maximum=24):
    """Hough support fragments, not complete arrows or implied junctions."""
    if not masks:return []
    union=np.logical_or.reduce(list(masks.values())).astype('uint8')*255
    side=max(size)
    raw=cv2.HoughLinesP(union,1,np.pi/360,max(12,round(.015*side)),
        minLineLength=max(12,round(.02*side)),maxLineGap=max(6,round(.025*side)))
    if raw is None:return []
    lines=[]
    for value in raw[:,0]:
        a,b=np.array(value[:2],float),np.array(value[2:],float)
        d=b-a;length=np.linalg.norm(d)
        if length<1:continue
        if tuple(a)>tuple(b):a,b=b,a;d=-d
        lines.append((length,a,b,d/length))
    kept=[]
    for length,a,b,d in sorted(lines,key=lambda x:-x[0]):
        duplicate=False
        for old in kept:
            p,q=np.array(old['points']);axis=(q-p)/np.linalg.norm(q-p);normal=np.array([-axis[1],axis[0]])
            u=np.array([(a-p)@axis,(b-p)@axis])
            if abs(axis@d)>math.cos(math.radians(3)) and max(abs((a-p)@normal),abs((b-p)@normal))<3 and u.min()>-4 and u.max()<np.linalg.norm(q-p)+4:
                duplicate=True;break
        if duplicate:continue
        samples=a+np.linspace(0,1,max(8,round(length)))[:,None]*(b-a)
        px=np.clip(np.rint(samples[:,0]).astype(int),0,size[0]-1);py=np.clip(np.rint(samples[:,1]).astype(int),0,size[1]-1)
        support=[]
        for mid,mask in masks.items():
            near=cv2.dilate(mask.astype('uint8'),np.ones((5,5),np.uint8))
            if float(near[py,px].mean())>.25:support.append(mid)
        kept.append(dict(id='L'+str(len(kept)+1),points=np.round([a,b],1).tolist(),support_masks=support))
        if len(kept)>=maximum:break
    return kept


def evidence(targets, metadata, candidates, proposals, mask_arrays, size, include_lines):
    used={m for row in metadata.values() for m in row['source_masks']}
    masks=[{k:c[k] for k in ('id','bbox','components') if k in c} for c in candidates if c['id'] in used]
    data=dict(coordinate_frame='local CLEAN pixels; all numbers rounded to one decimal',
        masks=masks,route_masks={rid:row['source_masks'] for rid,row in metadata.items()},
        direct_proposals=[p for p in proposals if p['mask'] in used])
    if include_lines:data['line_fragments']=line_hints({m:a for m,a in mask_arrays.items() if m in used},size)
    return data


def draw(clean, hints, path):
    image=Image.open(clean).convert('RGB');d=ImageDraw.Draw(image)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    for h in hints:
        a,b=h['points'];d.line([tuple(a),tuple(b)],fill='#0079bc',width=2)
        for p in [a,b]:d.ellipse((p[0]-2,p[1]-2,p[0]+2,p[1]+2),fill='#d00076')
        center=(np.array(a)+np.array(b))/2
        d.text(tuple(center+2),h['id'],font=font,fill='#0079bc',stroke_width=1,stroke_fill='white')
    image.save(path)
