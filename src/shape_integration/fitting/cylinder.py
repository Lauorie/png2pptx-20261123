"""Observed elliptical rim and extrusion, exported as an adjustable Office can."""
import math
import numpy as np
from .primitives import FitError, fit_boundary, rotation


def fit(points):
    q = np.asarray(points, float)
    if q.shape != (11, 2) or not np.isfinite(q).all():
        raise FitError('Cylinder needs eleven finite observed contour points')
    top = fit_boundary('ellipse', q[:8])
    c = np.array(top['center']); w = top['width']; h = top['height']; a = top['angle_radians']
    if w <= h:
        raise FitError('Use paths for a cylinder without a flattened transverse rim')
    r = rotation(a)
    anchors = np.array([[w/2, 0], [0, h/2], [-w/2, 0]])
    lower = (q[8:] - c) @ r
    depth = float(np.mean((lower - anchors)[:, 1]))
    if depth <= 0:
        raise FitError('Cylinder lower rim must be below the top rim in its local frame')
    residual = np.linalg.norm(lower - anchors - [0, depth], axis=1)
    params = {k: top[k] for k in ('center', 'width', 'height', 'angle_radians')}
    params['depth'] = depth
    diagnostics = {'top_rim': top['point_fit_diagnostics'],
                   'lower_rim_rms_px': float(np.sqrt(np.mean(residual**2))),
                   'lower_rim_max_px': float(residual.max()),
                   'warning': 'Fits model observations; CLEAN critic still decides source agreement.'}
    return params, diagnostics


def commands(p):
    c = np.array(p['center']); w = p['width']; h = p['height']; d = p['depth']
    r = rotation(p['angle_radians']); x = w/2; y = h/2; k = 4/3*math.tan(math.pi/8)
    # Open front seam first, closed outer silhouette last. One fillable body;
    # the independently styled top-face ellipse paints above this body.
    local = [['M', -x, 0], ['C', -x, k*y, -k*x, y, 0, y],
             ['C', k*x, y, x, k*y, x, 0],
             ['M', -x, 0], ['C', -x, -k*y, -k*x, -y, 0, -y],
             ['C', k*x, -y, x, -k*y, x, 0], ['L', x, d],
             ['C', x, d+k*y, k*x, d+y, 0, d+y],
             ['C', -k*x, d+y, -x, d+k*y, -x, d], ['Z']]
    return [[cmd[0], *[float(v) for i in range(1, len(cmd), 2)
                      for v in (np.array(cmd[i:i+2]) @ r.T + c)]] for cmd in local]


def native(p, frame):
    from .frames import FitFrame
    f = FitFrame.from_metadata(frame)
    sx, sy = [(f.crop_bounds[i+2]-f.crop_bounds[i])/f.input_size[i]*f.original_size[i]/f.global_size[i] for i in (0, 1)]
    a = p['angle_radians']; r = rotation(a); axes = np.diag([sx, sy]) @ r
    u, v = axes[:, 0], axes[:, 1]
    if abs(float(u @ v)) > 1e-10*np.linalg.norm(u)*np.linalg.norm(v):
        return None  # Anisotropic transport of a tilted can needs the custom path.
    w = p['width']*np.linalg.norm(u); h = (p['height']+p['depth'])*np.linalg.norm(v)
    rim = p['height']*np.linalg.norm(v)
    # DrawingML can pins adj to 50000*h/ss and uses y1=ss*adj/200000.
    # Thus the top ellipse DIAMETER cannot exceed half the total height.
    # Flat discs beyond that handle range must keep the exact custom body.
    if rim>h/2+1e-8:return None
    center = f.to_original((np.array(p['center']) + r @ [0, p['depth']/2]).tolist())
    return {'preset': 'can', 'center': center, 'width': float(w), 'height': float(h),
            'angle_radians': math.atan2(u[1], u[0]),
            'adjustments': [float(rim/min(w, h))]}
