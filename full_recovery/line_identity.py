"""Conservative identity evidence for independently fitted straight strokes."""
import numpy as np
from shape_integration.fitting.connectivity import node_pair


def pattern(style):
    value = style.get('dash', 'solid')
    if isinstance(value, dict):value = value.get('preset', value.get('type', 'solid'))
    return {'continuous':'solid', 'dashed':'dash', 'dotted':'dot'}.get(value, value)


def compatible(shape_style, arrow_style):
    if pattern(shape_style)!=pattern(arrow_style):return False
    a, b = shape_style.get('width_px'), arrow_style.get('width_px')
    if a and b and abs(a-b)>max(1.5, .6*max(a,b)):return False
    a, b = shape_style.get('stroke'), arrow_style.get('color')
    if isinstance(a,str) and isinstance(b,str):
        colors = [np.array([int(c[i:i+2],16) for i in (1,3,5)]) for c in (a,b)]
        if np.linalg.norm(colors[0]-colors[1])>80:return False
    return True


def distinct_source_routes(rgb,a,b):
    """Protect two independently visible paths even when they share both nodes."""
    from .handoff import distances
    def supported(row,other):
        points=np.asarray(row['points'],float)
        width=float(row.get('style',{}).get('width_px') or 1)
        radius=max(1.5,min(3.,width/2+1))
        away=distances(points,[other['points']])>2*radius+2
        away[:len(points)//5]=False;away[-len(points)//5:]=False
        if away.sum()<.3*len(points):return False
        pigment=row.get('style',{}).get('stroke') or row.get('style',{}).get('color')
        if not isinstance(pigment,str):return False
        color=np.array([int(pigment[j:j+2],16) for j in (1,3,5)])
        tangent=np.gradient(points,axis=0)
        normal=np.c_[-tangent[:,1],tangent[:,0]]/np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-9)
        offsets=np.arange(-radius,radius+.5,.5)
        xy=np.rint(points[away,None]+normal[away,None]*offsets[None,:,None]).astype(int)
        valid=((xy[:,:,0]>=0)&(xy[:,:,0]<rgb.shape[1])&(xy[:,:,1]>=0)&(xy[:,:,1]<rgb.shape[0])).all(1)
        if valid.sum()<15:return False
        xy=xy[valid]
        near=(np.linalg.norm(rgb[xy[:,:,1],xy[:,:,0]].astype(float)-color,axis=2)<65).any(1)
        return float(near.mean())>=.8
    return supported(a,b) and supported(b,a)



def separate_parallel_sources(rgb,a,b,color,tolerance,other_color=None):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=(2,2) or b.shape!=(2,2):return False
    if np.linalg.norm(a[0]-b[0])+np.linalg.norm(a[1]-b[1])>np.linalg.norm(a[0]-b[1])+np.linalg.norm(a[1]-b[0]):b=b[::-1]
    v=a[1]-a[0];length=np.linalg.norm(v)
    if length<5:return False
    normal=np.array([-v[1],v[0]])/length
    t=np.linspace(.2,.8,80);points=a[0]+t[:,None]*v
    offsets=np.arange(-int(np.ceil(tolerance))-3,int(np.ceil(tolerance))+4)
    xy=np.rint(points[:,None]+normal[None,None]*offsets[None,:,None]).astype(int)
    valid=((xy[:,:,0]>=0)&(xy[:,:,0]<rgb.shape[1])&(xy[:,:,1]>=0)&(xy[:,:,1]<rgb.shape[0])).all(1)
    if valid.sum()<30:return False
    xy=xy[valid];pigment=np.array([int(color[i:i+2],16) for i in (1,3,5)])
    pixels=rgb[xy[:,:,1],xy[:,:,0]].astype(float)
    distance=np.linalg.norm(pixels-pigment,axis=2)
    if other_color:
        other=np.array([int(other_color[i:i+2],16) for i in (1,3,5)])
        distance=np.minimum(distance,np.linalg.norm(pixels-other,axis=2))
    persistent=(distance<70).mean(0)>.65
    bands=np.flatnonzero(np.diff(np.r_[False,persistent,False].astype(int))).reshape(-1,2)
    if len(bands)<2:return False
    centers=np.array([np.mean(offsets[start:end]) for start,end in bands])
    shift=float(np.mean((b-a)@normal))
    return bool(np.argmin(abs(centers))!=np.argmin(abs(centers-shift)))


def single_source_stroke(rgb, a, b, color, tolerance, width=1.):
    """Two predictions must bracket ONE observed stroke, not parallel routes.

    Discard junction/end regions. A normal strip with zero/multiple pigment
    bands abstains; mere spatial proximity is never identity evidence.
    """
    a, b = np.asarray(a,float), np.asarray(b,float)
    if a.shape!=(2,2) or b.shape!=(2,2):return 0.
    if np.linalg.norm(a[0]-b[0])+np.linalg.norm(a[1]-b[1]) > np.linalg.norm(a[0]-b[1])+np.linalg.norm(a[1]-b[0]):b=b[::-1]
    t = np.linspace(.15,.85,60)
    q = a[0]+t[:,None]*(a[1]-a[0]); other = b[0]+t[:,None]*(b[1]-b[0])
    tangent = a[1]-a[0]; length = np.linalg.norm(tangent)
    if length<4:return 0.
    normal = np.array([-tangent[1],tangent[0]])/length
    offsets = np.arange(-int(np.ceil(tolerance))-3,int(np.ceil(tolerance))+4)
    xy = np.rint(q[:,None]+normal[None,None]*offsets[None,:,None]).astype(int)
    valid = ((xy[:,:,0]>=0)&(xy[:,:,0]<rgb.shape[1])&(xy[:,:,1]>=0)&(xy[:,:,1]<rgb.shape[0])).all(1)
    xy[:,:,0] = np.clip(xy[:,:,0],0,rgb.shape[1]-1);xy[:,:,1] = np.clip(xy[:,:,1],0,rgb.shape[0]-1)
    pigment = np.array([int(color[i:i+2],16) for i in (1,3,5)])
    ink = np.linalg.norm(rgb[xy[:,:,1],xy[:,:,0]].astype(float)-pigment,axis=2)<70
    shift = (other-q)@normal; supported = []
    for i in np.flatnonzero(valid):
        spans = np.flatnonzero(np.diff(np.r_[False,ink[i],False].astype(int))).reshape(-1,2)
        if len(spans)!=1:
            supported.append(False);continue
        start,end = spans[0]
        center = float(np.mean(offsets[start:end]))
        supported.append(start>0 and end<len(offsets) and
                         min(0,shift[i])-width-1<=center<=max(0,shift[i])+width+1)
    return float(np.mean(supported)) if len(supported)>=30 else 0.
