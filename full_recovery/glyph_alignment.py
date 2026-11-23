"""Translate confidently matched native glyphs within the same source occurrence."""
from copy import deepcopy
from pathlib import Path
import math


def install(backend):
    import hashlib
    path=Path(backend)/'src/shape_integration/texts/two_stage_crop.py'
    source=path.read_text();before=hashlib.sha256(source.encode()).hexdigest()
    marker='    from .typography_audit import audit as audit_typography'
    if source.count(marker)!=1:raise ValueError('Reference final text stage changed')
    source=source.replace(marker,
        "    from full_recovery.glyph_alignment import align as align_native_glyphs\n"
        "    chosen,glyph_translations=align_native_glyphs(chosen,source)\n"
        "    save(output/'glyph_translations.json',glyph_translations)\n"+marker)
    path.write_text(source)
    return {'protocol':'local_translation_only.1','installed':True,
            'before_sha256':before,'after_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'font_size_content_and_styles_unchanged':True,'extra_model_stages':0}


def align(rows, source):
    import cv2
    import numpy as np
    from PIL import Image
    from shape_integration.texts.foreground import contrast_ink
    from shape_integration.texts.layout_integrity import overlap

    cv2.setNumThreads(1)
    with Image.open(source) as opened:image=opened.convert('RGB')
    output=deepcopy(rows);audit=[]
    for before,after in zip(rows,output):
        m=before.get('layout',{}).get('measurement',{})
        record={'id':before['id'],'applied':False};audit.append(record)
        path=m.get('ink_template')
        if (m.get('method')!='native_ppt_atlas' or not path or not Path(path).is_file()
                or abs(before.get('rotation_deg',0))>1e-6):
            record['reason']='no_unrotated_native_glyph_template';continue
        factor=before['font_size_px']/m['reference_size_px']
        w,h=max(2,round(m['width']*factor)),max(2,round(m['height']*factor))
        box=before['bbox_2d'];cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
        pad=min(12,max(4,math.ceil(min(w,h)*.9)))
        region=[max(0,math.floor(cx-w/2)-pad),max(0,math.floor(cy-h/2)-pad),
                min(image.width,math.ceil(cx+w/2)+pad+1),min(image.height,math.ceil(cy+h/2)+pad+1)]
        rgb=np.asarray(image.crop(region),np.float32)
        local=[(math.floor if i<2 else math.ceil)(box[i]-region[i%2]) for i in range(4)]
        ink=contrast_ink(rgb,local).astype(np.float32)
        with Image.open(path) as opened:
            template=np.asarray(opened.convert('RGBA').getchannel('A').resize((w,h),Image.Resampling.LANCZOS),np.float32)/255
        if h>ink.shape[0] or w>ink.shape[1] or (template*template).sum()<6:
            record['reason']='insufficient_template';continue
        dot=cv2.matchTemplate(ink,template,cv2.TM_CCORR)
        integral=cv2.integral(ink*ink)
        mass=integral[h:,w:]-integral[:-h,w:]-integral[h:,:-w]+integral[:-h,:-w]
        scores=2*np.maximum(dot,0)/(float((template*template).sum())+mass+1e-6)
        y,x=np.unravel_index(np.argmax(scores),scores.shape)
        ax=int(np.clip(round(cx-(w-1)/2-region[0]),0,scores.shape[1]-1))
        ay=int(np.clip(round(cy-(h-1)/2-region[1]),0,scores.shape[0]-1))
        yy,xx=np.indices(scores.shape);outside=(xx-x)**2+(yy-y)**2>=max(2,min(w,h)*.35)**2
        second=float(scores[outside].max()) if outside.any() else 0.
        best,current=float(scores[y,x]),float(scores[ay,ax])
        delta=[float(x+region[0]+(w-1)/2-cx),float(y+region[1]+(h-1)/2-cy)]
        record.update(score=best,current_score=current,distinct_peak_margin=best-second,translation_px=delta)
        if best<.84 or best-current<.08 or best-second<.04:
            record['reason']='no_unique_supported_improvement';continue
        moved=[v+delta[i%2] for i,v in enumerate(box)]
        if not (0<=moved[0]<moved[2]<image.width and 0<=moved[1]<moved[3]<image.height):
            record['reason']='outside_source';continue
        after['bbox_2d']=moved
        after['frame_px']=[before['frame_px'][0]+delta[0],before['frame_px'][1]+delta[1],*before['frame_px'][2:]]
        after['glyph_registration']={'mode':'local_translation_only.1','score':best,'translation_px':delta}
        record.update(applied=True,reason='same_native_glyphs_unique_local_source_match')
    rejected=set()
    for i,a in enumerate(output):
        for j in range(i+1,len(output)):
            now=overlap(a['bbox_2d'],output[j]['bbox_2d'])
            if now>.20 and now-overlap(rows[i]['bbox_2d'],rows[j]['bbox_2d'])>.10:
                rejected.update(k for k in (i,j) if audit[k]['applied'])
    for i in rejected:
        output[i]=deepcopy(rows[i]);audit[i].update(applied=False,reason='new_text_collision')
    return output,audit
