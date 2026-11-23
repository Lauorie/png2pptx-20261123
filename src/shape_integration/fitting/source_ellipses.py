"""Measure a visible ellipse from SOURCE pixels, independently of model points.

An isolated flat-fill component nominates identity. The full source perimeter
must support an ellipse before its center, axes or border can be exported.
"""
import cv2
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter1d
from scipy.optimize import least_squares


def measure(clean, mask, pigment, scale=1.):
    contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if len(contours)!=1 or len(contours[0])<16:return None
    contour=contours[0][:,0,:].astype(float)
    (cx,cy),(w,h),angle=cv2.fitEllipse(contour.astype('float32'))
    if min(w,h)<6*scale or max(w,h)>3*min(w,h):return None
    angle=np.radians(angle);rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    center=np.array([cx,cy]);axes=np.array([w,h])/2
    q=(contour-center)@rot
    fill_error=np.abs(np.sqrt((q*q/(axes**2)).sum(1))-1)*min(axes)
    # A complete source-fill contour, rather than isolated rim fragments,
    # establishes the shape. Cropped discs and irregular blobs cannot pass.
    if np.quantile(fill_error,.95)>1.1*scale:return None
    theta=np.linspace(0,2*np.pi,180,endpoint=False)
    unit=np.c_[np.cos(theta),np.sin(theta)]
    reference=(unit*axes)@rot.T+center
    normal=(unit/axes)@rot.T;normal/=np.linalg.norm(normal,axis=1)[:,None]
    rgb=np.asarray(clean.convert('RGB'),float)
    band=np.arange(-3*scale,7*scale+.05*scale,.2*scale)
    coords=reference[:,None]+normal[:,None]*band[None,:,None]
    if (coords[...,0].min()<0 or coords[...,1].min()<0 or
        coords[...,0].max()>clean.width-1 or coords[...,1].max()>clean.height-1):return None
    samples=np.stack([map_coordinates(rgb[:,:,c],[coords[...,1],coords[...,0]],order=1)for c in range(3)],axis=-1)
    # A stroke is a distinct narrow band against both the fill and outside.
    gray=samples.mean(2)
    contrast=np.minimum(np.median(gray[:,:4],axis=1),np.median(gray[:,-4:],axis=1))[:,None]-gray
    strength=gaussian_filter1d(np.maximum(contrast,0),.5,axis=1)
    best=strength.argmax(1);peak=strength[np.arange(len(theta)),best]
    strong=peak>10
    stroke=None
    if strong.mean()>=.75:
        peak_colors=samples[np.arange(len(theta)),best]
        stroke=np.median(peak_colors[strong],axis=0)
        strong&=np.linalg.norm(peak_colors-stroke,axis=1)<25
        offsets=[];supported=[];widths=[]
        for i in range(len(theta)):
            hit=(np.linalg.norm(samples[i]-stroke,axis=1)<20)&(strength[i]>.5*peak[i])
            spans=np.flatnonzero(np.diff(np.r_[False,hit,False].astype(int))).reshape(-1,2)
            spans=[(a,b)for a,b in spans if a<=best[i]<b and a>0 and b<len(band)]
            if not spans or not strong[i]:offsets.append(0.);supported.append(False);continue
            a,b=spans[0];offsets.append((band[a]+band[b-1])/2);supported.append(True);widths.append((b-a)*.2*scale)
        supported=np.asarray(supported);offsets=np.asarray(offsets)
    else:
        # Unoutlined opaque ellipses: locate the fill-to-background transition.
        distance=np.linalg.norm(samples-np.asarray(pigment),axis=2)
        hits=distance<16;offsets=[];supported=[]
        for row in hits:
            edges=np.flatnonzero(row[:-1]&~row[1:]);edges=edges[(band[edges]>-2*scale)&(band[edges]<4*scale)]
            supported.append(len(edges)==1);offsets.append(float((band[edges[0]]+band[edges[0]+1])/2)if len(edges)==1 else 0.)
        supported=np.asarray(supported);offsets=np.asarray(offsets);widths=[];stroke=None
    sectors=[supported[index].mean()for index in np.array_split(np.arange(len(theta)),4)]
    if supported.mean()<.8 or min(sectors)<.45:return None
    points=reference[supported]+normal[supported]*offsets[supported,None]
    # Fit the observed perimeter; do not impose equal axes on actual ellipses.
    def residual(z):
        a=z[4];r=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        q=(points-z[:2])@r
        return (np.sqrt((q*q/(z[2:4]**2)).sum(1))-1)*min(z[2:4])
    fit=least_squares(residual,[*center,*axes,angle],bounds=([cx-4*scale,cy-4*scale,.6*axes[0],.6*axes[1],angle-.3],
        [cx+4*scale,cy+4*scale,1.5*axes[0],1.5*axes[1],angle+.3]),loss='soft_l1',f_scale=.5*scale)
    if not fit.success:return None
    z=fit.x;errors=np.abs(residual(z));circular=False
    if abs(z[2]-z[3])<.6*scale:
        circle=least_squares(lambda p:np.linalg.norm(points-p[:2],axis=1)-p[2],[*z[:2],np.mean(z[2:4])],loss='soft_l1',f_scale=.5*scale)
        e=np.abs(np.linalg.norm(points-circle.x[:2],axis=1)-circle.x[2])
        if np.quantile(e,.9)<.85*scale and np.quantile(e,.95)<1.5*scale:
            z=np.r_[circle.x[:2],circle.x[2],circle.x[2],0.];errors=e;circular=True
    if np.quantile(errors,.9)>.85*scale or np.quantile(errors,.95)>1.5*scale:return None
    area=mask.sum();expected=np.pi*axes.prod()
    if not .88<=area/expected<=1.18:return None
    return {'parameters':{'center':z[:2].tolist(),'width':float(2*z[2]),'height':float(2*z[3]),'angle_radians':float(z[4])},
            'source_fill':np.asarray(pigment).tolist(),'source_stroke':stroke.tolist()if stroke is not None else None,
            'source_width_px':float(np.median(widths)/scale)if widths else 0.,
            'perimeter_support':float(supported.mean()),'sector_support':sectors,
            'perimeter_p95_px':float(np.quantile(errors,.95)/scale),'source_circle_supported':circular,
            'source_component_ellipse_area_ratio':float(area/expected)}
