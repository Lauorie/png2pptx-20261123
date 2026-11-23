"""CPU-selected native numerical-vector typography; source boxes never move."""
from copy import deepcopy
import math

import numpy as np
from PIL import Image

from ..common import save
from . import equations
from .omml import latex_to_omml


def eligible(row, metric):
    if abs(row.get('rotation_deg',0))>1e-6 or metric.get('preview_degraded'):return False
    b=row['bbox_2d'];w=b[2]-b[0]+1;h=b[3]-b[1]+1
    scale=min(w/metric['width'],h/metric['height'])*.98
    if metric['width']*scale/w>=.90:return False
    baseline=latex_to_omml(row['latex'])
    vector=latex_to_omml(row['latex'],numeric_vectors=True)
    return len(vector.findall('.//m:m',equations.NS))>len(baseline.findall('.//m:m',equations.NS))


def preview_score(row, metric, source):
    """Score fixed-box native glyphs, never register/shift/shrink the source box."""
    from .layout import fit
    from .foreground import contrast_ink
    fitted=fit(row,metric)
    b=row['bbox_2d'];roi=[max(0,math.floor(b[0])-4),max(0,math.floor(b[1])-4),
                        min(source.width,math.ceil(b[2])+5),min(source.height,math.ceil(b[3])+5)]
    ref=np.asarray(source.crop(roi),dtype=np.float32)
    ink=contrast_ink(ref,[0,0,ref.shape[1],ref.shape[0]])
    canvas=Image.new('RGB',(roi[2]-roi[0],roi[3]-roi[1]),'white')
    with Image.open(metric['equation_preview']) as asset:
        x,y,w,h=fitted['frame_px']
        asset=asset.convert('RGBA').resize((max(1,round(w)),max(1,round(h))),Image.Resampling.LANCZOS)
        canvas.paste(asset,(round(x-roi[0]),round(y-roi[1])),asset)
    rgb=np.asarray(canvas,dtype=np.float32)
    pred=contrast_ink(rgb,[0,0,rgb.shape[1],rgb.shape[0]])
    dice=float(2*(ink*pred).sum()/max(float((ink*ink).sum()+(pred*pred).sum()),1e-9))
    overlap=np.minimum(ink,pred)
    agreement=float(((1-np.linalg.norm(ref-rgb,axis=2)/(255*math.sqrt(3)))*overlap).sum()/max(float(overlap.sum()),1e-9))
    return {'score':dice*(.75+.25*agreement),'source_mass':float(ink.sum()),
            'predicted_ink_size_px':fitted['layout']['predicted_ink_size_px']}


def select(rows, measurements, source, folder, slots, opt):
    """Only unambiguous sparse numerical vectors receive bounded candidates.

    Mathematical symbols/scripts keep their math fonts. The second candidate
    tests Arial only for numeric matrix cells. Ordinary words, intervals,
    fractions, radicals and existing multidimensional matrices are untouched.
    """
    candidates=[];groups={};audit=[]
    for row in rows:
        try:qualifies=eligible(row,measurements[row['id']])
        except (ValueError,TypeError):qualifies=False
        if not qualifies:continue
        group=[]
        for index,font in enumerate((None,'Arial')):
            profile={'numeric_vectors':True,'vector_gap_em':1.0,'numeric_font_family':font}
            candidate={**deepcopy(row),'id':row['id']+f'.V{index}', 'native_formula_layout':profile}
            candidates.append(candidate);group.append(candidate)
        groups[row['id']]=group
    if not candidates:
        save(folder/'selection.json',[]);return rows,measurements
    from .equation_recovery import measure_resilient
    metrics,recovery=measure_resilient(candidates,folder/'candidates',slots,opt['render_timeout_seconds'])
    revised=[];measurements=dict(measurements)
    with Image.open(source) as image:
        image=image.convert('RGB')
        for row in rows:
            if row['id'] not in groups:revised.append(row);continue
            before=preview_score(row,measurements[row['id']],image)
            tested=[]
            for candidate in groups[row['id']]:
                metric=metrics[candidate['id']]
                if metric.get('preview_degraded'):continue
                tested.append({**preview_score(candidate,metric,image),'id':candidate['id'],
                               'profile':candidate['native_formula_layout']})
            best=max(tested,key=lambda m:m['score'],default=None)
            # This is proposal ranking, not quality certification. An extra
            # absolute floor rejects genuine improvements on difficult glyphs;
            # the actual combined PPT render still has a zero-regression gate.
            accepted=bool(best and before['source_mass']>=8
                          and best['score']>=before['score']+.02)
            if accepted:
                row={**row,'native_formula_layout':deepcopy(best['profile'])}
                measurements[row['id']]=metrics[best['id']]
            revised.append(row)
            audit.append({'id':row['id'],'bbox_unchanged':True,'before':before,'candidates':tested,
                          'accepted':accepted,'chosen':best['id'] if accepted else 'baseline',
                          'policy':'fixed-box native glyph comparison; minimum gain .02; no bitmap stretching'})
    save(folder/'selection.json',audit)
    return revised,measurements
