"""Local color/geometry candidates verified against a complete text template.

OCR provides an occurrence anchor, not transcription. Large fills, crossing
same-color graphics, mixed-color expressions and weak template evidence defer.
Final acceptance still requires actual native PPT rendering and CPU scoring.
"""
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def locate(row,detection,source,vertical_bounds=None):
    record={'status':'uncertain','bbox_2d':None,'reason':'no_verified_glyph_support',
            'text_authority':'vlm','ocr_bbox':detection['source_bbox']}
    empty=np.zeros((1,1),bool)
    colors={r.get('color',row.get('color','#000000')).upper() for r in row.get('runs',[])}
    if row.get('kind')=='formula':
        from .omml import latex_to_omml
        tree=latex_to_omml(row['latex'],color=row['color'])
        colors.update('#'+c.upper() for c in tree.xpath('.//*[local-name()="srgbClr"]/@val'))
    if len(colors)>1:
        return {**record,'reason':'multicolor_structure_preserved'},empty
    try:
        if row.get('kind')=='formula':
            path=row.get('layout',{}).get('measurement',{}).get('equation_preview')
            if not path or not Path(path).is_file():return record,empty
            with Image.open(path) as im:template=im.convert('RGBA').getchannel('A').copy()
        else:
            from .reflow import template as text_template
            from .layout import resolve_font
            if not row.get('font'):return record,empty
            font=resolve_font(row['font'],row.get('bold',False),row.get('italic',False))['path']
            template=text_template(row['text'],font)
    except (ValueError,OSError):return record,empty
    box=template.getbbox()
    if not box:return record,empty
    template=template.crop(box)
    b=detection['source_bbox'];h=b[3]-b[1]+1;w=b[2]-b[0]+1
    # Small textline context plus bounded missing-suffix space; not the enclosing
    # graphical container. Adjacent baselines impose hard upper/lower limits.
    pad=max(3,h*.3)
    expected=''.join(row['text'].split());observed=''.join(detection['text'].strip('$').split())
    missing=max(0,len(expected)-len(observed))
    extra=min(h,missing*w/max(1,len(observed))*1.5)
    roi=[max(0,math.floor(b[0]-pad-extra)),max(0,math.floor(b[1]-pad)),
         min(source.width,math.ceil(b[2]+pad+extra)+1),min(source.height,math.ceil(b[3]+pad)+1)]
    if vertical_bounds:
        roi[1]=max(roi[1],math.ceil(vertical_bounds[0]))
        roi[3]=min(roi[3],math.floor(vertical_bounds[1])+1)
    if roi[2]-roi[0]<3 or roi[3]-roi[1]<3:return record,empty
    rgb=np.asarray(source.crop(roi).convert('RGB'),dtype=np.float32)
    edge=np.concatenate((rgb[0],rgb[-1],rgb[:,0],rgb[:,-1]))
    keys,counts=np.unique((edge//16).astype(int),axis=0,return_counts=True)
    key=keys[counts.argmax()];background=np.median(edge[np.all((edge//16).astype(int)==key,axis=1)],axis=0)
    pixels=rgb.reshape(-1,3);keys,counts=np.unique((pixels//16).astype(int),axis=0,return_counts=True)
    candidates=[];seen=[];peaks=[]
    for index in np.argsort(counts)[::-1][:12]:
        if counts[index]<4:continue
        color=np.median(pixels[np.all((pixels//16).astype(int)==keys[index],axis=1)],axis=0)
        separation=float(np.linalg.norm(color-background))
        if separation>=65:peaks.append((separation,color))
    # Antialias grays are mixtures with the background, not competing font
    # colors. Prefer the strongest supported color along each color direction.
    for separation,color in sorted(peaks,key=lambda x:x[0],reverse=True):
        direction=(color-background)/separation
        if any(np.linalg.norm(color-c)<45 or float(direction@v)>.98 for c,v in seen):continue
        seen.append((color.copy(),direction))
        distance=np.linalg.norm(rgb-color,axis=2)
        strong=distance<=30;weak=distance<=min(180,separation*.55)
        _,labels=cv2.connectedComponents(weak.astype(np.uint8),8)
        ids=np.unique(labels[strong]);ids=ids[ids!=0];mask=np.isin(labels,ids)
        border_ids=np.unique(np.concatenate((labels[0],labels[-1],labels[:,0],labels[:,-1])))
        cx0=max(0,math.ceil(b[0])-roi[0]);cy0=max(0,math.ceil(b[1])-roi[1])
        cx1=min(labels.shape[1],math.floor(b[2])+1-roi[0]);cy1=min(labels.shape[0],math.floor(b[3])+1-roi[1])
        core_ids=set(np.unique(labels[cy0:cy1,cx0:cx1]))
        # Only discard disconnected boundary components completely OUTSIDE the
        # OCR text box. Never strip an edge component that also contains glyphs.
        removable=[i for i in border_ids if i and i not in core_ids]
        if removable:mask[np.isin(labels,removable)]=False
        yy,xx=np.where(mask)
        if len(xx)<4 or mask.mean()>.55:continue
        if mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any():continue
        x0,y0,x1,y1=xx.min(),yy.min(),xx.max()+1,yy.max()+1
        width=x1-x0;height=y1-y0
        if width<max(2,w*.3) or height<max(3,h*.3):continue
        aspect=(width/height)/(template.width/template.height)
        if not .5<=aspect<=2:continue
        glyph=np.asarray(template.resize((width,height),Image.Resampling.LANCZOS),dtype=float)/255
        target=mask[y0:y1,x0:x1].astype(float)
        score=float(2*(glyph*target).sum()/max(1e-9,(glyph*glyph).sum()+target.sum()))
        if score<(.60 if row.get('kind')=='formula' else .50):continue
        if color.max()<24:color[:]=0
        if color.min()>239:color[:]=255
        hexcolor='#%02X%02X%02X'%tuple(int(round(v)) for v in color)
        bounds=[int(x0+roi[0]),int(y0+roi[1]),int(x1-1+roi[0]),int(y1-1+roi[1])]
        candidates.append((score,hexcolor,bounds,mask))
    if not candidates:return {**record,'roi':roi},empty
    candidates.sort(key=lambda c:c[0],reverse=True)
    best=candidates[0]
    if len(candidates)>1 and best[0]-candidates[1][0]<.05:
        return {**record,'roi':roi,'reason':'ambiguous_text_color_support'},empty
    return {**record,'status':'located','bbox_2d':best[2],'roi':roi,
            'reason':'complete_glyph_template_and_source_color','proposed_color':best[1],
            'template_score':best[0],'ink_pixels':int(best[3].sum()),
            'requires_native_render_gate':True},best[3]
