"""Optional radius-only fit on source stroke profiles, gated by an existing SAM.

Runs inside the existing numerical fitter; no new model, crop or critic calls.
The mask locates the owning contour. CLEAN determines the stroke centerline.
"""
import math
import numpy as np
import cv2
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize_scalar
from .paths import box_commands
from .primitives import rr_signed_distance


def radius_from_pixels(parameters, clean, mask, *, source_scale=1., sam_tolerance=2.):
    p=parameters; w,h=p['width'],p['height']; old=p['radius']
    audit={'applied':False,'radius_before':old,'source_scale':source_scale}
    binary=(np.asarray(mask)>0).astype(np.uint8)
    if not binary.any():return None,{**audit,'reason':'empty_sam'}
    # Filled masks support their exterior; outlined masks support their actual
    # foreground. Never fill disconnected dashes into an invented silhouette.
    yy,xx=np.nonzero(binary)
    if binary.sum()/max(1,(xx.max()-xx.min()+1)*(yy.max()-yy.min()+1))>.4:
        contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        binary=np.zeros_like(binary);cv2.drawContours(binary,contours,-1,1,1)
    distance=cv2.distanceTransform(1-binary,cv2.DIST_L2,5)
    rgb=np.asarray(clean,dtype=float); gray=rgb.mean(axis=2)
    a=p['angle_radians']; rotation=np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
    search=max(2.,4*source_scale); offsets=np.linspace(-search,search,25)
    theta=np.linspace(.15,math.pi/2-.15,10)
    normals=np.concatenate([np.column_stack([np.cos(theta+i*math.pi/2),np.sin(theta+i*math.pi/2)]) for i in range(4)])
    normals=normals@rotation.T
    def score(radius):
        centers=np.array([[w/2-radius,h/2-radius],[-w/2+radius,h/2-radius],
                          [-w/2+radius,-h/2+radius],[w/2-radius,-h/2+radius]])
        points=np.repeat(centers,10,axis=0)@rotation.T+np.array(p['center'])+radius*normals
        xy=points[:,None,:]+normals[:,None,:]*offsets[None,:,None]
        pixels=map_coordinates(gray,[xy[:,:,1],xy[:,:,0]],order=1,mode='constant',cval=255.)
        near=map_coordinates(distance,[points[:,1],points[:,0]],order=1,mode='constant',cval=1e4)
        background=np.minimum(np.median(pixels[:,:3],axis=1),np.median(pixels[:,-3:],axis=1))
        ink=np.maximum(0,background[:,None]-pixels)
        peak=ink.max(axis=1)
        # Only a local band distinct from BOTH sides is a stroked centerline.
        weights=np.where(ink>=.5*peak[:,None],ink,0)
        center=(weights*offsets).sum(axis=1)/np.maximum(weights.sum(axis=1),1e-8)
        good=(peak>12)&(near<=sam_tolerance)&(weights.sum(axis=1)>0)
        errors=np.where(good,np.minimum(np.abs(center),search),search)
        per_corner=np.mean(errors.reshape(4,10),axis=1)
        supported=np.mean(good.reshape(4,10),axis=1)>=.6
        # No single well-segmented corner can determine all four corner radii.
        value=float(np.mean(per_corner))
        return value,int(supported.sum())
    grid=np.linspace(0,min(w,h)/2,33)
    measured=[score(r)[0] for r in grid];i=int(np.argmin(measured))
    lo,hi=grid[max(0,i-1)],grid[min(len(grid)-1,i+1)]
    result=minimize_scalar(lambda r:score(r)[0],bounds=(lo,hi),method='bounded',options={'xatol':.05*source_scale})
    new=float(result.x);before,corners_before=score(old);after,corners=score(new)
    audit.update(radius_candidate=new,loss_before=before,loss_after=after,supported_corners=corners)
    if not result.success or corners<2 or after>source_scale or before-after<.25*source_scale:
        return None,{**audit,'reason':'no_reliable_corner_improvement'}
    return new,{**audit,'applied':True,'reason':'source_centerline_corners_supported_by_sam'}


def refine(local, job, clean, evidence):
    from ..sam_evidence import project_mask
    from .adapter import bounds, union_bounds
    audits=[]
    if evidence is None or not hasattr(evidence,'support'):return audits
    group=evidence.support(job['candidate'])
    if group is None or not evidence.mask_usable(group,job['candidate']):return audits
    frame=job['frame'];crop=frame['source_crop_half_open'];size=frame['actual_crop_input_size_px']
    scale=max(size[i]/(crop[i+2]-crop[i]) for i in (0,1))
    prediction=group['representative']; network=prediction['frame']['box']
    uncertainty=max(2.,1.5*max(network[2]-network[0],network[3]-network[1])/1008)*scale
    mask=project_mask(prediction,crop,size)
    protected=set((job.get('revision_context') or {}).get('preserved_ids',[]))
    for part in local['parts']:
        if part['kind']!='rounded_rectangle' or part['status']!='ok' or part['id'] in protected:continue
        radius,audit=radius_from_pixels(part['parameters'],clean,mask,source_scale=scale,sam_tolerance=uncertainty)
        audit.update(part_id=part['id'],sam_id=group['id']);audits.append(audit)
        if radius is None:continue
        p=part['parameters'];p.update(radius=radius,pptx_adjust=radius/min(p['width'],p['height']))
        part['commands']=box_commands([*p['center'],p['width'],p['height'],p['angle_radians']],radius)
        part['bbox_2d']=bounds(part['commands']);part['diagnostics']['radius_pixel_refinement']=audit
        table={q['id']:q['xy'] for q in local['observation']['points']}
        points=np.array([table[k] for k in part['observed_point_ids']])
        residual=np.abs(rr_signed_distance(points,[*p['center'],p['width'],p['height'],p['angle_radians'],p['pptx_adjust']]))
        part['diagnostics'].update(rms_px=float(np.sqrt(np.mean(residual**2))),p90_px=float(np.percentile(residual,90)),max_px=float(residual.max()))
    local['bbox_2d']=union_bounds(local['parts'])
    return audits
