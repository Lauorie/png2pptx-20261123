"""Shorten only the rendered shaft; semantic endpoints and head anchors stay fixed."""
import numpy as np


def split(c,t):
    a=(1-t)*c[:-1]+t*c[1:];b=(1-t)*a[:-1]+t*a[1:];p=(1-t)*b[0]+t*b[1]
    return np.array([c[0],a[0],b[0],p]),np.array([p,b[1],a[2],c[3]])


def path_d(points,curves=None):
    if curves:
        return 'M'+','.join(map(str,curves[0][0]))+' '+' '.join('C'+' '.join(','.join(map(str,p)) for p in c[1:]) for c in curves)
    return 'M'+' L'.join(','.join(map(str,p)) for p in points)


def head_trim(total,width,head):
    return min(total*.45,width*({'arrow':.7}.get(head['type'],1.5))*{'sm':2/3,'med':1.,'lg':5/3}[head.get('length','med')]) if head else 0


def shaft_d(points,curves,width,heads):
    """Arc-length trim under native-like head glyphs, including cubic end spans."""
    if not any(heads):return path_d(points,curves)
    if curves:
        segments=[np.asarray(c,float) for c in curves];tables=[]
        for c in segments:
            t=np.linspace(0,1,129);u=t[:,None]
            samples=(1-u)**3*c[0]+3*(1-u)**2*u*c[1]+3*(1-u)*u**2*c[2]+u**3*c[3]
            tables.append((t,np.r_[0,np.cumsum(np.linalg.norm(np.diff(samples,axis=0),axis=1))]))
        lengths=np.array([table[1][-1] for table in tables])
    else:
        points=np.asarray(points,float);segments=list(zip(points,points[1:]));lengths=np.linalg.norm(np.diff(points,axis=0),axis=1)
    cumulative=np.r_[0,np.cumsum(lengths)];total=cumulative[-1]
    amounts=[head_trim(total,width,h) for h in heads]
    lo,hi=amounts[0],total-amounts[1];output=[]
    for i,segment in enumerate(segments):
        if lengths[i]<=1e-9 or cumulative[i+1]<=lo or cumulative[i]>=hi:continue
        a=max(lo-cumulative[i],0);b=min(hi-cumulative[i],lengths[i])
        if curves:
            t,s=tables[i];ta,tb=np.interp([a,b],s,t)
            left,_=split(segment,tb)
            _,part=split(left,ta/tb if tb else 0);output.append(part.tolist())
        else:
            p,q=segment;a,b=p+(q-p)*a/lengths[i],p+(q-p)*b/lengths[i]
            if not output:output.append(a.tolist())
            output.append(b.tolist())
    return path_d(output[0] if curves else output,output if curves else None) if output else ''
