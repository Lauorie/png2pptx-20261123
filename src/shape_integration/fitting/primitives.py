"""Contour point fits. No image-specific cases or raster-edge snapping."""
import math
import numpy as np
from scipy.optimize import least_squares


class FitError(ValueError):
    """A part failed; retain its raw observations and numerical diagnostics."""

    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def solver_record(result):
    return {'success': bool(result.success), 'status': int(result.status),
            'message': str(result.message), 'nfev': int(result.nfev),
            'cost': float(result.cost) if np.isfinite(result.cost) else None}


def require_solver(result):
    if not result.success or not np.isfinite(result.x).all() or not np.isfinite(result.cost):
        raise FitError('Optimizer did not return a finite converged solution',
                       {'solver': solver_record(result)})


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def contour(p, count=720):
    c=np.array(p['center']);w=p['width'];h=p['height'];a=p['angle_radians']
    if p['kind']=='ellipse':
        t=np.linspace(0,2*np.pi,count,endpoint=False)
        q=np.column_stack([w/2*np.cos(t),h/2*np.sin(t)])
    else:
        r=p['radius'];points=[];n=max(10,count//8)
        for cx,cy,start in [(w/2-r,-h/2+r,-np.pi/2),(w/2-r,h/2-r,0),
                             (-w/2+r,h/2-r,np.pi/2),(-w/2+r,-h/2+r,np.pi)]:
            ts=np.linspace(start,start+np.pi/2,n)
            arc=np.column_stack([cx+r*np.cos(ts),cy+r*np.sin(ts)])
            if points:points.extend(np.linspace(points[-1],arc[0],n)[1:])
            points.extend(arc)
        points.extend(np.linspace(points[-1],points[0],n)[1:]);q=np.array(points)
    return q@rotation(a).T+c


def rr_signed_distance(points, params):
    cx,cy,w,h,angle,adjust=params
    local=(points-[cx,cy])@rotation(angle)
    radius=adjust*min(w,h)
    q=np.abs(local)-([w/2-radius,h/2-radius])
    return np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(np.max(q,axis=1),0)-radius


def ellipse_approx_distance(points,params):
    cx,cy,w,h,angle=params
    local=(points-[cx,cy])@rotation(angle);a=w/2;b=h/2
    f=(local[:,0]/a)**2+(local[:,1]/b)**2-1
    grad=2*np.sqrt((local[:,0]/a**2)**2+(local[:,1]/b**2)**2)
    return f/np.maximum(grad,1e-9)


def fit_boundary(kind, points, robust_scale_px=2.0):
    counts = {'ellipse': 8, 'rounded_rectangle': 16}
    if kind not in counts:
        raise FitError('Unsupported primitive kind: ' + str(kind))
    points=np.asarray(points,float)
    if points.shape != (counts[kind], 2) or not np.isfinite(points).all():
        raise FitError(f'{kind} requires exactly {counts[kind]} finite contour points')
    duplicate_groups = [set(np.flatnonzero(np.all(points == p, axis=1))) for p in points]
    # At a capsule limit, a collapsed straight side puts its tangent/midpoint
    # roles at the same position. Distinct IDs remain intact for the solver.
    collapsed_sides = [{15, 0, 1}, {3, 4, 5}, {7, 8, 9}, {11, 12, 13}]
    allowed_duplicates = kind == 'rounded_rectangle' and all(
        len(group) == 1 or any(group <= side for side in collapsed_sides)
        for group in duplicate_groups)
    if ((len(np.unique(points, axis=0)) != len(points) and not allowed_duplicates)
            or np.linalg.matrix_rank(points-points.mean(axis=0), tol=1e-6) < 2):
        raise FitError('Degenerate primitive observations')
    if not np.isfinite(robust_scale_px) or robust_scale_px <= 0:
        raise FitError('robust_scale_px must be finite and positive')
    origin=(points.min(axis=0)+points.max(axis=0))/2
    scale=float(max(np.ptp(points,axis=0)))
    if scale<=0:raise ValueError('Degenerate point cloud')
    q=(points-origin)/scale
    _,axes=np.linalg.eigh(np.cov(q.T));pca=math.atan2(axes[1,-1],axes[0,-1])
    angles=[0,pca,pca+np.pi/2]
    candidates=[]
    for angle in angles:
        angle=(angle+np.pi/2)%np.pi-np.pi/2
        projected=q@rotation(angle);span=np.ptp(projected,axis=0)
        for radius in ([.06,.18,.34] if kind=='rounded_rectangle' else [None]):
            x0=[0,0,*np.maximum(span,.02),angle]
            lower=[-2,-2,.005,.005,-np.pi];upper=[2,2,4,4,np.pi]
            if kind=='rounded_rectangle':x0.append(radius);lower.append(0);upper.append(.5)
            fn=rr_signed_distance if kind=='rounded_rectangle' else ellipse_approx_distance
            result=least_squares(lambda x:fn(q,x),x0,bounds=(lower,upper),loss='soft_l1',
                                 f_scale=robust_scale_px/scale,max_nfev=600,ftol=1e-11,xtol=1e-11,gtol=1e-11)
            candidates.append(result)
    # Keep the original minimum-cost choice. Failed convergence is explicit;
    # do not silently replace it with a different initialization or old method.
    best=min(candidates,key=lambda x:x.cost)
    require_solver(best)
    x=best.x
    solvers={'initializations':[solver_record(c) for c in candidates],
             'selected':solver_record(best)}
    if kind=='ellipse':
        # Orthogonal-distance refinement: optimize point-to-ellipse foot angles as well.
        local=(q-x[:2])@rotation(x[4]);t=np.arctan2(local[:,1]/(x[3]/2),local[:,0]/(x[2]/2))
        def residual(z):
            pred=np.column_stack([z[2]/2*np.cos(z[5:]),z[3]/2*np.sin(z[5:])])@rotation(z[4]).T+z[:2]
            return (pred-q).ravel()
        z0=np.r_[x,t]
        joint=least_squares(residual,z0,bounds=(np.r_[[-2,-2,.005,.005,-np.pi],np.full(len(t),-np.inf)],
            np.r_[[2,2,4,4,np.pi],np.full(len(t),np.inf)]),loss='soft_l1',f_scale=robust_scale_px/scale,
            max_nfev=200,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        require_solver(joint)
        solvers['refinement']=solver_record(joint)
        x=joint.x[:5];distance=np.linalg.norm(residual(joint.x).reshape(-1,2),axis=1)*scale
    else:distance=np.abs(rr_signed_distance(q,x))*scale
    center=x[:2]*scale+origin;w=x[2]*scale;h=x[3]*scale;angle=x[4]
    # Canonical width axis closest to image x; equivalent geometry, simpler native export.
    angle=(angle+np.pi/2)%np.pi-np.pi/2
    if abs(angle)>np.pi/4:w,h=h,w;angle=angle-np.sign(angle)*np.pi/2
    params={'kind':kind,'center':center.tolist(),'width':float(w),'height':float(h),'angle_radians':float(angle)}
    if kind=='rounded_rectangle':params.update(radius=float(x[5]*min(w,h)),pptx_adjust=float(x[5]))
    if (not np.isfinite([*center,w,h,angle,*distance]).all() or min(w,h)<=0
            or (kind=='rounded_rectangle' and not 0<=params['radius']<=min(w,h)/2)):
        raise FitError('Invalid fitted parameters', {'solvers':solvers})
    params['point_fit_diagnostics']={'count':len(points),'rms_px':float(np.sqrt(np.mean(distance**2))),
        'median_px':float(np.median(distance)),'p90_px':float(np.percentile(distance,90)),
        'max_px':float(distance.max()),'robust_loss':'soft_l1','robust_scale_px':robust_scale_px,
        'objective':'exact signed distance to rounded rectangle' if kind=='rounded_rectangle' else 'joint orthogonal point-to-ellipse distances',
        'warning':'These residuals measure agreement with model points, not accuracy against source pixels.',
        'solvers':solvers}
    return params
