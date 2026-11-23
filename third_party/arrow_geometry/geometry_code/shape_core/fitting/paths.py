"""Typed least-squares geometry and interpolating cubic paths; no image-specific heuristics."""
import math
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
from .primitives import FitError, solver_record, require_solver


def rectangle_corners(parameters):
    cx,cy,w,h,angle=parameters;c=np.array([cx,cy]);u=np.array([math.cos(angle),math.sin(angle)])*w/2
    v=np.array([-math.sin(angle),math.cos(angle)])*h/2
    return np.array([c-u-v,c+u-v,c+u+v,c-u+v])


def fit_rectangle(points):
    p=np.array(points,float)
    if (p.shape!=(4,2) or not np.isfinite(p).all()
            or len(np.unique(p,axis=0))!=4):
        raise FitError('Rectangle needs four distinct finite corners')
    edges=np.roll(p,-1,axis=0)-p
    cross=edges[:,0]*np.roll(edges[:,1],-1)-edges[:,1]*np.roll(edges[:,0],-1)
    if not (cross>1e-6).all():
        raise FitError('Rectangle corners must be clockwise and convex')
    c=p.mean(axis=0);a=p[1]-p[0]
    initial=[*c,np.linalg.norm(a),np.linalg.norm(p[3]-p[0]),math.atan2(a[1],a[0])]
    result=least_squares(lambda x:(rectangle_corners(x)-p).ravel(),initial,
                         bounds=([-np.inf,-np.inf,1e-6,1e-6,-np.inf],[np.inf]*5))
    require_solver(result)
    fit=rectangle_corners(result.x);res=np.linalg.norm(fit-p,axis=1)
    return result.x.tolist(),fit,{'rms_px':float(np.sqrt(np.mean(res**2))),
        'max_px':float(res.max()),'solver':solver_record(result)}


def curve_commands(points,periodic=False):
    p=np.array(points,float);length=np.linalg.norm(np.diff(p,axis=0),axis=1)
    if np.any(length<1e-9):raise ValueError('Duplicate curve knots')
    t=np.r_[0,np.cumsum(length)];s=CubicSpline(t,p,axis=0,bc_type='periodic' if periodic else 'natural')
    commands=[['M',*p[0]]]
    for i,dt in enumerate(length):
        control1=p[i]+s(t[i],1)*dt/3;control2=p[i+1]-s(t[i+1],1)*dt/3
        commands.append(['C',*control1,*control2,*p[i+1]])
    return [[str(c[0]),*map(float,c[1:])] for c in commands]


def sample_commands(commands,steps=16):
    out=[];current=None;start=None
    for command in commands:
        if command[0]=='M':current=np.array(command[1:]);start=current.copy();out.append(current.tolist())
        elif command[0]=='L':current=np.array(command[1:]);out.append(current.tolist())
        elif command[0]=='C':
            p0=current;p1,p2,p3=[np.array(command[i:i+2]) for i in (1,3,5)]
            for t in np.linspace(0,1,steps+1)[1:]:out.append(((1-t)**3*p0+3*(1-t)**2*t*p1+3*(1-t)*t*t*p2+t**3*p3).tolist())
            current=p3
        elif command[0]=='Z':
            if current is not None and not np.array_equal(current,start):out.append(start.tolist())
            current=start
    return out


def box_commands(parameters,radius=0):
    cx,cy,w,h,angle=parameters;co,si=math.cos(angle),math.sin(angle)
    def world(x,y):return [cx+co*x-si*y,cy+si*x+co*y]
    if radius<=0:
        q=rectangle_corners(parameters).tolist();return [['M',*q[0]],*[['L',*p] for p in q[1:]],['Z']]
    r=radius;k=4/3*math.tan(math.pi/8);x=w/2;y=h/2
    local=[['M',-x+r,-y],['L',x-r,-y],['C',x-r+k*r,-y,x,-y+r-k*r,x,-y+r],
           ['L',x,y-r],['C',x,y-r+k*r,x-r+k*r,y,x-r,y],['L',-x+r,y],
           ['C',-x+r-k*r,y,-x,y-r+k*r,-x,y-r],['L',-x,-y+r],
           ['C',-x,-y+r-k*r,-x+r-k*r,-y,-x+r,-y],['Z']]
    return [[c[0],*[v for i in range(1,len(c),2) for v in world(c[i],c[i+1])]] for c in local]


def ellipse_commands(center,u,v):
    k=4/3*math.tan(math.pi/8);c=np.array(center);u=np.array(u);v=np.array(v)
    knots=[c+u,c+v,c-u,c-v,c+u];tangents=[v,-u,-v,u,v]
    return [['M',*map(float,knots[0])],*[['C',*map(float,knots[i]+k*tangents[i]),*map(float,knots[i+1]-k*tangents[i+1]),*map(float,knots[i+1])] for i in range(4)],['Z']]
