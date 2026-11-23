"""Calibrate native dash lengths from source pixels after semantic style recovery.

V2 fixed dash vs dash-dot but guessed lengths poorly despite image-size metadata.
This pass measures repeated source runs along unchanged geometry. It never fits or
moves points. Robust-run thresholds are generic; no image/shape-specific patches.
"""
import numpy as np
from .rules import poly, rgb, profile_visibility

from scipy.ndimage import map_coordinates, median_filter
import argparse

def measure(image,part,style,excluded):
    q=poly(part);ll=np.linalg.norm(np.diff(q,axis=0),axis=1);q=q[np.r_[True,ll>1e-5]]
    ll=np.linalg.norm(np.diff(q,axis=0),axis=1);cum=np.r_[0,np.cumsum(ll)]
    n=min(25000,max(80,int(cum[-1]/.4)));step=cum[-1]/n;t=np.arange(n)*step
    ix=np.clip(np.searchsorted(cum,t,side='right')-1,0,len(ll)-1)
    frac=(t-cum[ix])/ll[ix];pts=q[ix]+frac[:,None]*(q[ix+1]-q[ix]);tangent=(q[ix+1]-q[ix])/ll[ix,None]
    normals=np.c_[-tangent[:,1],tangent[:,0]];offsets=np.arange(-14,14.01,.5)
    coords=pts[:,None,:]+normals[:,None,:]*offsets[None,:,None]
    arr=np.asarray(image,dtype=float)
    prof=np.stack([map_coordinates(arr[:,:,k],[coords[:,:,1],coords[:,:,0]],order=1,mode='nearest') for k in range(3)],axis=-1)
    valid=profile_visibility(coords,image.size,excluded)
    support=(np.min(np.linalg.norm(prof-rgb(style['stroke']),axis=-1),axis=1)<45)&valid
    widths=[]
    # Integrate antialias coverage across a narrow connected band. Different colors
    # on each side of the boundary get separate background estimates.
    for row in prof[support][::max(1,int(support.sum()/1000))]:
        peak=int(np.argmin(np.linalg.norm(row-rgb(style['stroke']),axis=1)))
        bg0=np.median(row[:4],axis=0);bg1=np.median(row[-4:],axis=0)
        bg=np.where((np.arange(len(offsets))<=peak)[:,None],bg0,bg1)
        delta=bg-rgb(style['stroke']);den=np.sum(delta*delta,axis=1)
        alpha=np.clip(np.sum((bg-row)*delta,axis=1)/np.maximum(den,1),0,1)
        a=peak;b=peak
        while a>0 and alpha[a-1]>.08:a-=1
        while b+1<len(alpha) and alpha[b+1]>.08:b+=1
        if a>0 and b<len(alpha)-1 and (b-a)*.5<12 and alpha[peak]>.65:
            widths.append(float(alpha[a:b+1].sum()*.5))
    support=median_filter(support.astype(np.uint8),size=3,mode='wrap').astype(bool)
    cuts=np.where(np.diff(np.r_[False,support,False].astype(int)))[0].reshape(-1,2)
    on=np.array([b-a for a,b in cuts[1:-1] if valid[max(0,a-1):min(len(valid),b+1)].all()])*step
    off=np.array([cuts[i+1,0]-cuts[i,1] for i in range(1,len(cuts)-2) if valid[cuts[i,0]:cuts[i+1,1]].all()])*step
    result={'sample_step_px':float(step),'normal_search_px':14,'color_tolerance_rgb':45,
            'observed_on_px':np.round(on,2).tolist(),'observed_off_px':np.round(off,2).tolist(),
            'valid_fraction':float(valid.mean()),'support_fraction':float(support[valid].mean()) if valid.any() else 0,'apply':False}
    if len(widths)>=20:
        wm=float(np.median(widths));wmad=float(np.median(np.abs(np.array(widths)-wm))/max(wm,.5))
        result.update(width_samples=len(widths),measured_width_px=round(wm,2),width_mad_ratio=wmad)
        if .5<=wm<=12 and wmad<.2:result['width_px']=round(wm,2)
    if len(on)<6 or len(off)<5:return {**result,'reason':'too_few_repeated_runs'}
    # Exclude long occlusion/misalignment gaps; retain short dots as a separate mode.
    gm=np.median(off);off=off[(off>.25*gm)&(off<2.5*gm)]
    gm=np.median(off);gmad=np.median(np.abs(off-gm))/max(gm,.5)
    lo,hi=np.quantile(on,[.25,.75]);pattern=None;kind=style['dash'];stable=False
    if style['dash']=='dash_dot' and hi>lo*1.9:
        a=on[on<(lo+hi)/2];b=on[on>=(lo+hi)/2]
        a=a[a<2.5*np.median(a)];b=b[b<2.5*np.median(b)]
        am,bm=np.median(a),np.median(b)
        mad_a=np.median(abs(a-am))/max(am,.5);mad_b=np.median(abs(b-bm))/max(bm,.5)
        stable=len(a)>=3 and len(b)>=3 and mad_a<.25 and mad_b<.25 and gmad<.28
        pattern=[bm,gm,am,gm];kind='dash_dot'
        result['mode_mad_ratios']=[float(mad_a),float(mad_b),float(gmad)]
    else:
        if style['dash']=='dash_dot':return {**result,'reason':'expected_dot_mode_not_observed'}
        m=np.median(on);on=on[(on>.4*m)&(on<2.5*m)];m=np.median(on)
        mad=np.median(abs(on-m))/max(m,.5)
        stable=mad<.2 and gmad<.25
        pattern=[m,gm];kind=style['dash']
        result['mode_mad_ratios']=[float(mad),float(gmad)]
    if not stable or not all(.5<=v<=150 for v in pattern):return {**result,'reason':'irregular_or_contaminated_runs'}
    result.update(apply=True,dash=kind,dash_pattern_px=[round(float(v),2) for v in pattern],reason='stable_repeated_source_runs')
    return result


def calibrate(image,part,style,excluded):
    result=dict(style)
    if not style.get('stroke'): return result,{'apply':False,'reason':'no_stroke'}
    m=measure(image,part,style,excluded)
    if m.get('valid_fraction',0)<.25:
        return result,{**m,'apply':False,'reason':'insufficient_visible_boundary'}
    width=m.get('width_px')
    if width and m.get('support_fraction',0)>=.65 and m.get('width_mad_ratio',1)<.15:
        result['width_px']=width
    if style['dash']!='solid' and m.get('apply'):
        result['dash_pattern_px']=m['dash_pattern_px']
    m['changed']=result!=style
    return result,m
