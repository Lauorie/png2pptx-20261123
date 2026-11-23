"""Robust one-direction RGB trend candidates; no semantic/image-ID conditions."""
import math
import numpy as np
from .geometry import gradient_position


def rgb(c): return np.array([int(c[i:i+2],16) for i in (1,3,5)],float)
def hexcolor(c): return '#'+''.join(f'{int(v):02X}' for v in np.clip(np.rint(c),0,255))


def estimate(part, xy, pixels, options):
    audit={'samples':len(pixels),'supported':False}
    if len(pixels)<options['gradient_min_samples']: return None,{**audit,'reason':'too_few_visible_samples'}
    # Deterministic thinning bounds cost, without depending on source identity.
    stride=max(1,len(pixels)//5000);xy=np.asarray(xy,float)[::stride];y=np.asarray(pixels,float)[::stride]
    center=xy.mean(axis=0); scale=np.maximum(np.ptp(xy,axis=0),1)
    x=np.c_[np.ones(len(xy)),(xy-center)/scale]
    keep=np.ones(len(y),bool)
    for _ in range(3):
        beta=np.linalg.lstsq(x[keep],y[keep],rcond=None)[0]
        errors=np.linalg.norm(y-x@beta,axis=1)
        keep=errors<=max(6.,float(np.quantile(errors,.8)))
    slopes=beta[1:]/scale[:,None]
    u,s,_=np.linalg.svd(slopes,full_matrices=False)
    if s[0]<1e-5: return None,{**audit,'reason':'no_color_trend'}
    direction=u[:,0]; angle=math.degrees(math.atan2(direction[1],direction[0]))%360
    t=gradient_position(part,angle,xy); design=np.c_[1-t,t]
    colors=np.linalg.lstsq(design[keep],y[keep],rcond=None)[0]
    pred=design@colors
    error=float(np.median(np.linalg.norm(y-pred,axis=1)))
    solid_error=float(np.median(np.linalg.norm(y-np.median(y,axis=0),axis=1)))
    difference=float(np.linalg.norm(colors[1]-colors[0]));span=float(np.quantile(t,.95)-np.quantile(t,.05))
    # Confirm a smooth trend across bands rather than fitting a two-color boundary.
    occupied=[];band_errors=[]
    for lo,hi in zip(np.linspace(0,1,9)[:-1],np.linspace(0,1,9)[1:]):
        hit=(t>=lo)&(t<hi)
        if hit.sum()>=4:
            occupied.append((lo+hi)/2)
            band_errors.append(float(np.linalg.norm(np.median(y[hit],axis=0)-np.median(pred[hit],axis=0))))
    valid=(span>=options['gradient_min_span'] and difference>=options['gradient_min_color_difference']
           and error<=options['gradient_max_error'] and error<solid_error*options['gradient_error_ratio']
           and len(occupied)>=4 and max(band_errors,default=999)<options['gradient_max_error']*1.5
           and s[-1]/s[0]<.35 and colors.min()>=-20 and colors.max()<=275)
    audit.update(supported=bool(valid),reason='supported_linear_trend' if valid else 'insufficient_linear_evidence',
                 median_error=error,solid_median_error=solid_error,visible_span=span,
                 color_difference=difference,band_errors=band_errors)
    if not valid:return None,audit
    return {'type':'linear_gradient','start_color':hexcolor(colors[0]),'end_color':hexcolor(colors[1]),
            'angle_deg':round(angle,3)},audit
