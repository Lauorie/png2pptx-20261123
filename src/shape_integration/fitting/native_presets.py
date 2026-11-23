"""Exact preset export from existing geometry; no recognition or regularising.

DrawingML corner is an L with independently adjustable leg widths. Its default
orientation lacks the top-right box corner; flips cover the other orientations.
Unrepresentable geometry continues through the existing custom-path exporter.
"""
import numpy as np


def from_commands(commands):
    if not commands or commands[-1][0]!='Z' or any(c[0] not in ('M','L','Z') for c in commands):return None
    if sum(c[0]=='M' for c in commands)!=1:return None
    q=np.asarray([c[1:3] for c in commands if c[0] in ('M','L')],float)
    if len(q)>1 and np.allclose(q[0],q[-1],rtol=0,atol=1e-8):q=q[:-1]
    if len(q)!=6:return None
    lo=q.min(0);hi=q.max(0);w,h=hi-lo;eps=1e-7*max(w,h,1.)
    if min(w,h)<=eps:return None
    delta=np.roll(q,-1,axis=0)-q
    if not np.all((np.abs(delta[:,0])<eps)^(np.abs(delta[:,1])<eps)):return None
    xs=np.unique(q[:,0]);ys=np.unique(q[:,1])
    if len(xs)!=3 or len(ys)!=3:return None
    corners=[(x,y) for x in (lo[0],hi[0]) for y in (lo[1],hi[1])]
    missing=[c for c in corners if np.linalg.norm(q-np.asarray(c),axis=1).min()>eps]
    if len(missing)!=1:return None
    flip_h=missing[0][0]==lo[0];flip_v=missing[0][1]==hi[1]
    x=hi[0]-xs[1] if flip_h else xs[1]-lo[0]
    y=hi[1]-ys[1] if flip_v else ys[1]-lo[1]
    canonical=np.array([[0,0],[x,0],[x,y],[w,y],[w,h],[0,h]],float)
    if flip_h:canonical[:,0]=w-canonical[:,0]
    if flip_v:canonical[:,1]=h-canonical[:,1]
    canonical+=lo
    # Equal vertex sets alone could conceal a self-crossing path.
    if not any(np.max(abs(q-np.roll(c,shift,axis=0)))<=eps for c in
               (canonical,canonical[::-1]) for shift in range(6)):return None
    return {'preset':'corner','center':((lo+hi)/2).tolist(),'width':float(w),'height':float(h),
            'angle_radians':0.,'adjustments':[(h-y)/min(w,h),x/min(w,h)],
            'flip_h':bool(flip_h),'flip_v':bool(flip_v)}
