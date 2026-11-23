"""Measure style on representative shaft sections; geometry remains unchanged."""
import copy
import numpy as np
from lib import style


def recover(image,mask,row):
    if row['kind']=='block_arrow':return style.filled(image,mask if mask is not None and mask.any() else None,row)
    if 'shaft_sample' in row and row['shaft_sample']!='auto':
        import shaft_sampling
        return shaft_sampling.recover(image,row)
    effective=mask if mask is not None and mask.any() else None
    result=style.recover(image,effective,row,adaptive_heads=True)
    if row.get('line_style') not in ('dashed','dotted'):return result
    p=style.points(row);lengths=np.linalg.norm(np.diff(p,axis=0),axis=1);cum=np.r_[0,np.cumsum(lengths)]
    candidates=[]
    # Straight legs supply local dash evidence even when another leg is damaged.
    if row['kind']=='polyline':
        for i,(a,b) in enumerate(zip(p,p[1:])):
            if lengths[i]<=0:continue
            part=copy.deepcopy(row);part.update(kind='straight',start=a.tolist(),via=[],end=b.tolist(),beziers=[])
            if i:part['start_head']=False;part['start_head_type']=None
            if i!=len(p)-2:part['end_head']=False;part['end_head_type']=None
            candidates.append((float(cum[i]),part))
    # Half-route windows avoid requiring a uniform good mask along the full arrow.
    if len(p)>1 and cum[-1]>0:
        for lo,hi in [(0,.5),(.25,.75),(.5,1)]:
            s=np.linspace(lo*cum[-1],hi*cum[-1],max(2,round((hi-lo)*cum[-1]/2)))
            q=np.c_[np.interp(s,cum,p[:,0]),np.interp(s,cum,p[:,1])]
            part=copy.deepcopy(row);part.update(start=q[0].tolist(),via=q[1:-1].tolist(),end=q[-1].tolist(),beziers=[])
            if lo>0:part['start_head']=False;part['start_head_type']=None
            if hi<1:part['end_head']=False;part['end_head_type']=None
            candidates.append((float(s[0]),part))
    measured=[(0.,result)] if result['dash'].get('mark_px') else []
    for offset,part in candidates:
        local=style.recover(image,effective,part,adaptive_heads=True)
        if local['dash'].get('mark_px') and local['dash'].get('gap_px'):measured.append((offset,local))
    if measured:
        offset,best=max(measured,key=lambda pair:(pair[1]['dash'].get('pattern_agreement',0),pair[1]['dash'].get('complete_marks',0)))
        result['dash']=copy.deepcopy(best['dash'])
        if result['dash'].get('period_px') and 'phase_px' in result['dash']:
            result['dash']['phase_px']=(result['dash']['phase_px']-offset)%result['dash']['period_px']
        if best.get('width_px') is not None:result['width_px']=best['width_px'];result['color']=best['color']
        result.setdefault('evidence',{})['style_window']=dict(start_distance_px=offset,
            method='representative local shaft; no whole-arrow mask completeness requirement',
            width_samples=best['evidence'].get('width_samples'))
        result['review_reasons']=[x for x in result.get('review_reasons',[]) if x not in ('dash_parameters_use_preset','dash_or_occlusion_ambiguous')]
        result['needs_review']=bool(result['review_reasons'])
    return result
