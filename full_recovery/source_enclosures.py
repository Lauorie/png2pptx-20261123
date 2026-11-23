"""Recover a source-supported simple enclosure independently of image content.

Only a near-complete elliptical rim filling the registered crop is accepted.
The fit is measured from pixels; descriptions, task IDs and filenames play no
part in the decision. Other silhouettes remain in their existing owner.
"""
import cv2
import numpy as np


def ellipse(source, bounds):
    b=np.array(bounds,float);lo=np.floor(b[:2]).astype(int);hi=np.ceil(b[2:]).astype(int)
    rgb=np.asarray(source.convert('RGB'))[lo[1]:hi[1],lo[0]:hi[0]]
    if min(rgb.shape[:2])<20:return None
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(float)
    gx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3);gy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
    yy,xx=np.indices(gray.shape);center=(np.array(gray.shape[::-1])-1)/2
    radial=np.stack([(xx-center[0])/max(1,center[0]),(yy-center[1])/max(1,center[1])],axis=2)
    radius=np.linalg.norm(radial,axis=2);mag=np.hypot(gx,gy)
    alignment=np.abs(gx*radial[:,:,0]+gy*radial[:,:,1])/np.maximum(1e-8,mag*radius)
    mask=(radius>.83)&(radius<1.08)&(mag>80)&(alignment>.8)
    points=np.column_stack([xx[mask],yy[mask]]).astype('float32')
    if len(points)<40:return None
    fit=cv2.fitEllipse(points);(cx,cy),(w,h),angle=fit
    if min(w,h)<=0:return None
    if (np.linalg.norm(np.array([cx,cy])-center)>.08*min(gray.shape)
            or min(w,h)<.82*min(gray.shape) or max(w,h)>1.12*max(gray.shape)):return None
    theta=np.linspace(0,2*np.pi,180,endpoint=False);a=np.deg2rad(angle)
    rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    ring=np.column_stack([w/2*np.cos(theta),h/2*np.sin(theta)])@rot.T+[cx,cy]
    normals=np.column_stack([np.cos(theta)/(w/2),np.sin(theta)/(h/2)])@rot.T
    normals/=np.linalg.norm(normals,axis=1)[:,None]
    offsets=np.arange(-2.,2.01,.5)
    xy=np.rint(ring[:,None]+normals[:,None]*offsets[None,:,None]).astype(int)
    valid=((xy[:,:,0]>=0)&(xy[:,:,0]<gray.shape[1])&(xy[:,:,1]>=0)&(xy[:,:,1]<gray.shape[0]))
    xy[:,:,0]=np.clip(xy[:,:,0],0,gray.shape[1]-1);xy[:,:,1]=np.clip(xy[:,:,1],0,gray.shape[0]-1)
    edges=mag[xy[:,:,1],xy[:,:,0]]
    gradients=np.stack([gx[xy[:,:,1],xy[:,:,0]],gy[xy[:,:,1],xy[:,:,0]]],axis=2)
    directional=np.abs((gradients*normals[:,None]).sum(2))/np.maximum(edges,1e-8)>.9
    supported=((edges>80)&directional&valid).any(1);coverage=float(supported.mean())
    if coverage<.88:return None
    values=rgb[xy[:,:,1],xy[:,:,0]].mean(2);idx=values.argmin(1)
    core=rgb[xy[np.arange(len(theta)),idx,1],xy[np.arange(len(theta)),idx,0]][supported]
    pigment=np.median(core,axis=0)
    if np.ptp(pigment)>30 or pigment.mean()>210:return None
    threshold=(255+pigment.mean())/2
    widths=((values<threshold)&valid).sum(1)*.5
    width=float(np.median(widths[supported]))
    if not .5<=width<=.045*min(gray.shape):return None
    # Texture gradients can support an arbitrary inscribed ellipse. A rim
    # must also be a complete source contour with the same silhouette.
    ink=(gray<threshold).astype('uint8')
    contours,_=cv2.findContours(ink,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    tolerance=max(1.5,width)
    from scipy.spatial import cKDTree
    if not any(len(c)>=20
               and np.quantile(cKDTree(ring).query(c[:,0,:])[0],.95)<=tolerance
               and np.quantile(cKDTree(c[:,0,:]).query(ring)[0],.95)<=tolerance
               for c in contours):return None
    return {'kind':'ellipse','center':(np.array([cx,cy])+lo).tolist(),'size':[float(w),float(h)],
            'rotation_deg':float(angle),'stroke':'#'+''.join('%02X'%round(c) for c in pigment),
            'width_px':width,'source_rim_coverage':coverage,'method':'source_gradient_ellipse.1'}


def clip_interior(image, placement, enclosure):
    from PIL import Image
    rgba=np.asarray(image.convert('RGBA')).copy();yy,xx=np.indices(rgba.shape[:2])
    b=np.array(placement,float);x=b[0]+(xx+.5)*(b[2]-b[0])/rgba.shape[1]-.5
    y=b[1]+(yy+.5)*(b[3]-b[1])/rgba.shape[0]-.5
    a=np.deg2rad(enclosure['rotation_deg']);dx=x-enclosure['center'][0];dy=y-enclosure['center'][1]
    u=dx*np.cos(a)+dy*np.sin(a);v=-dx*np.sin(a)+dy*np.cos(a)
    radii=np.array(enclosure['size'])/2-enclosure['width_px']/2
    inside=(u/radii[0])**2+(v/radii[1])**2<1
    rgba[~inside,3]=0
    return Image.fromarray(rgba,'RGBA')


def retained_rim(source, frame):
    """Measure a complete thin source rim near a retained native enclosure.

    The fitted geometry nominates the neighborhood, not a clipping boundary.
    A partial, connected or ambiguous rim cannot authorize removing pixels.
    """
    from scipy.spatial import cKDTree
    bounds=np.asarray(frame['bbox_2d'],float);span=bounds[2:]-bounds[:2]
    if np.any(span<=0):return None
    lo=np.maximum(0,np.floor(bounds[:2]-.5*span)).astype(int)
    hi=np.minimum(source.size,np.ceil(bounds[2:]+.5*span)).astype(int)
    rgb=np.asarray(source.convert('RGB'))[lo[1]:hi[1],lo[0]:hi[0]]
    if not rgb.size:return None
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    _,ink=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    contours,hierarchy=cv2.findContours(ink,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:return None
    candidates=[]
    for index,inner in enumerate(contours):
        parent=int(hierarchy[0,index,3])
        if parent<0:continue
        outer=contours[parent];points=outer[:,0,:]
        if (points.min(0)<=0).any() or (points.max(0)>=np.asarray(ink.shape[::-1])-1).any():continue
        lower=points.min(0)+lo;upper=points.max(0)+lo;ratio=(upper-lower)/span
        if (ratio<.65).any() or (ratio>1.35).any():continue
        if (np.abs((lower+upper)/2-(bounds[:2]+bounds[2:])/2)>.3*span).any():continue
        area=cv2.contourArea(outer);hole=cv2.contourArea(inner)
        if area<=0 or not 0<(area-hole)/area<.2:continue
        distance=cKDTree(inner[:,0,:]).query(points)[0]
        if distance.max()>.05*min(upper-lower):continue
        mask=np.zeros_like(ink);cv2.drawContours(mask,[outer],-1,255,-1)
        cv2.drawContours(mask,[inner],-1,0,-1)
        mask=cv2.dilate(mask,np.ones((3,3),'uint8'))
        candidates.append((mask,{'source_box_half_open':[*lo.tolist(),*hi.tolist()],
            'source_outer_bounds':[*lower.tolist(),*upper.tolist()],
            'maximum_rim_width_px':float(distance.max()),'source_contour':parent}))
    return candidates[0] if len(candidates)==1 else None


def remove_retained_rim(image, placement, measured):
    """Remove the copied source ring; keep its interior and exterior content."""
    from PIL import Image
    mask,receipt=measured;rgba=np.asarray(image.convert('RGBA')).copy()
    b=np.asarray(placement,float);lo=np.asarray(receipt['source_box_half_open'][:2])
    x=np.rint(b[0]+(np.arange(image.width)+.5)*(b[2]-b[0])/image.width-.5).astype(int)-lo[0]
    y=np.rint(b[1]+(np.arange(image.height)+.5)*(b[3]-b[1])/image.height-.5).astype(int)-lo[1]
    valid_x=(x>=0)&(x<mask.shape[1]);valid_y=(y>=0)&(y<mask.shape[0])
    hit=mask[np.clip(y,0,mask.shape[0]-1)[:,None],np.clip(x,0,mask.shape[1]-1)[None,:]]>0
    hit&=valid_y[:,None]&valid_x[None,:]
    removed=int(np.count_nonzero(hit&(rgba[:,:,3]>0)))
    rgba[hit,3]=0
    return Image.fromarray(rgba,'RGBA'),{**receipt,'removed_nontransparent_pixels':removed}
