"""Geometry-only identities, exact cubic bounds, masks and fill reference frames."""
import math
import json
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, LineString, GeometryCollection
from shapely.ops import unary_union
from ..fitting.paths import sample_commands


def part_key(part):
    return json.dumps(part['key'], ensure_ascii=False)


def parts(scene):
    return {key: scene['parts'][key] for obj in scene['objects'].values()
            for key in obj['part_keys'] if scene['parts'][key]['status'] == 'ok'}


def closed(part):
    return bool(part['commands'] and part['commands'][-1][0] == 'Z')


def bounds(part):
    points = []; current = None; start = None
    for cmd in part['commands']:
        if cmd[0] == 'M':
            current = np.array(cmd[1:], float); start = current.copy(); points.append(current)
        elif cmd[0] == 'L':
            current = np.array(cmd[1:], float); points.append(current)
        elif cmd[0] == 'C':
            p = np.array([current, cmd[1:3], cmd[3:5], cmd[5:7]], float)
            ts = [0., 1.]
            for axis in range(2):
                v = p[:, axis]
                coeff = [-v[0]+3*v[1]-3*v[2]+v[3], 2*(v[0]-2*v[1]+v[2]), v[1]-v[0]]
                for t in np.roots(np.trim_zeros(coeff, 'f')) if any(coeff) else []:
                    if abs(t.imag) < 1e-9 and 0 < t.real < 1: ts.append(float(t.real))
            points.extend((1-t)**3*p[0]+3*(1-t)**2*t*p[1]+3*(1-t)*t*t*p[2]+t**3*p[3] for t in ts)
            current = p[3]
        elif cmd[0] == 'Z': current = start
        else: raise ValueError('Unsupported path command: '+str(cmd[0]))
    p = np.array(points)
    if not len(p) or not np.isfinite(p).all(): raise ValueError('Invalid style geometry')
    return [*p.min(axis=0).tolist(), *p.max(axis=0).tolist()]


def subpaths(part):
    current = []
    for c in part['commands']:
        if c[0] == 'M' and current:
            yield current; current = []
        current.append(c)
    if current: yield current


def footprint(part):
    """Approximation for relations only; invalid geometry is exposed, never repaired."""
    commands = list(subpaths(part)); polys = []; lines = []
    for path in commands:
        p = sample_commands(path, 96)
        if len(p) < 2: continue
        line = LineString(p); lines.append(line)
        if path[-1][0] == 'Z' and len(p) >= 4:
            poly = Polygon(p)
            if not poly.is_valid:
                return unary_union(lines), 'invalid_polygon_visibility'
            polys.append(poly)
    if len(polys) > 1:
        return unary_union(lines), 'multiple_subpaths_visibility_unsupported'
    if polys: return polys[0], None
    return unary_union(lines) if lines else GeometryCollection(), None


def raster(geometry, box):
    x0,y0,x1,y1 = box; im = Image.new('L', (x1-x0, y1-y0)); draw = ImageDraw.Draw(im)
    def polygon(g):
        draw.polygon([(x-x0,y-y0) for x,y in g.exterior.coords], fill=255)
        for ring in g.interiors:
            draw.polygon([(x-x0,y-y0) for x,y in ring.coords], fill=0)
    def visit(g):
        if g.is_empty: return
        if g.geom_type == 'Polygon': polygon(g)
        elif hasattr(g, 'geoms'):
            for child in g.geoms: visit(child)
    visit(geometry)
    return np.asarray(im) > 0


def crop_box(ps, size, padding=.08, minimum=8):
    bb = np.array([bounds(p) for p in ps]); lo = bb[:,:2].min(axis=0); hi = bb[:,2:].max(axis=0)
    pad = np.maximum((hi-lo)*padding, minimum)
    a = np.maximum(0, np.floor(lo-pad)).astype(int)
    b = np.minimum(size, np.ceil(hi+pad+1)).astype(int)
    if np.any(b <= a): raise ValueError('Fitted geometry has no source-image intersection')
    return [int(a[0]), int(a[1]), int(b[0]), int(b[1])]


def fill_frame(part):
    """Match the existing PPT exporter reference rectangle, including rotation."""
    p = part.get('native_parameters_original')
    if p:
        return np.array(p['center']), float(p['width']), float(p['height']), float(p['angle_radians'])
    xy = np.array([(c[i],c[i+1]) for c in part['commands'] for i in range(1,len(c),2)])
    lo = xy.min(axis=0); hi = np.maximum(xy.max(axis=0),lo+1e-3)
    return (lo+hi)/2, *(hi-lo), 0.


def gradient_line(part, angle):
    center, w, h, rotation = fill_frame(part)
    a = math.radians(angle); local = a-rotation
    direction = np.array([math.cos(a), math.sin(a)])
    length = abs(math.cos(local))*w+abs(math.sin(local))*h
    return center-direction*length/2, center+direction*length/2


def gradient_position(part, angle, xy):
    a,b = gradient_line(part, angle); d = b-a
    return np.clip((np.asarray(xy)-a) @ d / (d @ d), 0, 1)
