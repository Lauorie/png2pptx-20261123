"""Ink graph -> arrow rows (geometry + style). Pure pixels, no model coordinates."""
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage
from skimage.measure import approximate_polygon, find_contours
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent / 'lib'))
import fit as libfit  # noqa: E402
import style as libstyle  # noqa: E402
import inkgraph as ig  # noqa: E402


def build_graph(ink):
    dt = ndimage.distance_transform_edt(ink)
    g = ig.build(skeletonize(ink))
    ig.prune(g, dt)
    ig.analyze_ends(g, ink, dt)
    return g, dt


def link_and_route(g, dt, instance_masks):
    def instance_of(n):
        x, y = np.rint(g.nodes[n]['xy']).astype(int)
        return {k for k, m in enumerate(instance_masks) if m[max(0, y - 1):y + 2, max(0, x - 1):x + 2].any()}
    links = ig.link_gaps(g, dt, instance_of)
    pairs = ig.pair_junctions(g, dt)
    return ig.routes(g, pairs), links


def end_info(g, n):
    node = g.nodes[n]
    if g.degree(n) == 1:
        return node['tip'], bool(node.get('head')), 'free'
    return node['xy'], False, 'junction'


def tls(pts):
    c = pts.mean(axis=0)
    _, v = np.linalg.eigh((pts - c).T @ (pts - c) + 1e-9 * np.eye(2))
    return c, v[:, -1]


def polyline(path, halfwidth):
    """Straight segments + corners (optionally rounded). None if the route is not polygonal."""
    tol = max(3., 2 * halfwidth)
    poly = approximate_polygon(path, tol)
    if len(poly) <= 2 or len(poly) > 10:
        return None
    idx = [int(np.argmin(np.linalg.norm(path - v, axis=1))) for v in poly]
    idx[0], idx[-1] = 0, len(path) - 1
    if any(b - a < 3 for a, b in zip(idx, idx[1:])):
        return None
    lines = []
    for a, b in zip(idx, idx[1:]):
        n = b - a; trim = max(1, n // 5)
        lines.append(tls(path[a + trim: b - trim + 1]))
    corners = []
    for (c1, d1), (c2, d2), i in zip(lines, lines[1:], idx[1:-1]):
        m = np.column_stack([d1, -d2])
        if abs(np.linalg.det(m)) < .05:
            corners.append(path[i]); continue
        s = np.linalg.solve(m, c2 - c1)
        corners.append(c1 + s[0] * d1)
    proj = lambda p, c, d: c + ((p - c) @ d) * d
    pts = np.vstack([proj(path[0], *lines[0]), *corners, proj(path[-1], *lines[-1])]) if corners else None
    if pts is None:
        return None
    # Residual of the pixel route against the sharp polyline, away from corners.
    seg_d = []
    for p in path[::2]:
        seg_d.append(min(_seg_dist(p, a, b) for a, b in zip(pts, pts[1:])))
    if np.percentile(seg_d, 90) > tol:
        return None
    # Corner rounding: distance from the sharp corner to the pixel route.
    beziers, rounded = [], False
    cur = pts[0]
    for k, c in enumerate(pts[1:-1], 1):
        u_in = c - pts[k - 1]; L_in = np.linalg.norm(u_in); u_in /= L_in
        u_out = pts[k + 1] - c; L_out = np.linalg.norm(u_out); u_out /= L_out
        phi = float(np.arccos(np.clip(u_in @ u_out, -1, 1)))
        dist = float(np.min(np.linalg.norm(path - c, axis=1)))
        if phi < .15 or dist < max(1.5, .6 * halfwidth):
            beziers.append(_line(cur, c)); cur = c; continue
        rho = dist / (1 / np.cos(phi / 2) - 1)
        t = min(rho * np.tan(phi / 2), .45 * L_in, .45 * L_out)
        rho = t / np.tan(phi / 2); h = 4 / 3 * np.tan(phi / 4) * rho
        a, b = c - t * u_in, c + t * u_out
        beziers.append(_line(cur, a)); beziers.append([a.tolist(), (a + h * u_in).tolist(), (b - h * u_out).tolist(), b.tolist()])
        cur = b; rounded = True
    beziers.append(_line(cur, pts[-1]))
    return pts, (beziers if rounded else [])


def _line(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return [a.tolist(), (a + (b - a) / 3).tolist(), (a + 2 * (b - a) / 3).tolist(), b.tolist()]


def _seg_dist(p, a, b):
    d = b - a; t = np.clip((p - a) @ d / max(d @ d, 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a + t * d)))


def shape(path, halfwidth):
    """Straight / polyline / curve from the pixel route; returns fitted points + beziers."""
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    tol = max(2., 1.2 * halfwidth, .02 * length)
    poly = approximate_polygon(path, tol)
    if len(poly) <= 2 or length < 4 * tol:
        return 'straight', np.asarray([path[0], path[-1]]), []
    fitted = polyline(path, halfwidth)
    if fitted is not None:
        return 'polyline', fitted[0], fitted[1]
    sampled, beziers = libfit.smooth(path, max(.5, halfwidth ** 2 * .08) * len(path))
    # Keep the exact pixel terminals (tips / junction centers).
    if beziers:
        da, db = path[0] - np.asarray(beziers[0][0]), path[-1] - np.asarray(beziers[-1][3])
        beziers[0][0] = path[0].tolist(); beziers[0][1] = (np.asarray(beziers[0][1]) + da).tolist()
        beziers[-1][3] = path[-1].tolist(); beziers[-1][2] = (np.asarray(beziers[-1][2]) + db).tolist()
        t = np.linspace(0, 1, 9)[:, None]; pts = []
        for c in beziers:
            c = np.asarray(c)
            p = (1 - t) ** 3 * c[0] + 3 * (1 - t) ** 2 * t * c[1] + 3 * (1 - t) * t ** 2 * c[2] + t ** 3 * c[3]
            pts.extend(p if not pts else p[1:])
        sampled = np.asarray(pts)
    return 'curve', sampled, beziers


def route_row(g, dt, r, ident):
    path = np.asarray(r['path'], float)
    (p0, h0, k0), (p1, h1, k1) = end_info(g, r['start']), end_info(g, r['end'])
    if np.linalg.norm(p0 - path[0]) > .5:
        path = np.vstack([p0, path])
    if np.linalg.norm(p1 - path[-1]) > .5:
        path = np.vstack([path, p1])
    row = path_row(path, ['triangle' if h0 else 'none', 'triangle' if h1 else 'none'], ident, dt)
    row.update(ends=dict(start=k0, end=k1), gaps=len(r['gaps']), source='ink_graph')
    return row


def halfwidth_of(dt, path):
    p = np.rint(path).astype(int)
    p[:, 0] = np.clip(p[:, 0], 0, dt.shape[1] - 1); p[:, 1] = np.clip(p[:, 1], 0, dt.shape[0] - 1)
    v = dt[p[:, 1], p[:, 0]]; v = v[v > 0]
    if not len(v):
        return 1.
    return float(np.median(v[len(v) // 5: len(v) - len(v) // 5])) if len(v) >= 10 else float(np.median(v))


def path_row(path, heads, ident, dt):
    """Any ordered pixel route (tips included) -> fitted row. heads = [start, end] native types."""
    path = np.asarray(path, float)
    path = path[np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-6]]
    hw = halfwidth_of(dt, path)
    kind, pts, beziers = shape(path, hw)
    h0, h1 = heads[0] != 'none', heads[1] != 'none'
    return dict(id=ident, kind=kind, start=pts[0].tolist(), via=pts[1:-1].tolist(), end=pts[-1].tolist(),
                beziers=beziers, heads=list(heads), start_head=h0, end_head=h1,
                start_head_type=heads[0] if h0 else None, end_head_type=heads[1] if h1 else None,
                width_estimate=max(1., 2 * hw - 1), route_path=path.tolist())


def block_row(mask, ident):
    contours = []
    for c in find_contours(np.pad(mask.astype(float), 1), .5):
        c = approximate_polygon(c[:, ::-1] - 1, tolerance=.75)
        if len(c) > 1 and np.allclose(c[0], c[-1]):
            c = c[:-1]
        if len(c) >= 3:
            contours.append(c.tolist())
    if not contours:
        return None
    main = np.asarray(max(contours, key=len))
    return dict(id=ident, kind='block_arrow', start=main[0].tolist(), via=main[1:-1].tolist(), end=main[-1].tolist(),
                contours=contours, beziers=[], heads=['none', 'none'], start_head=False, end_head=False,
                start_head_type=None, end_head_type=None, closed=True)


def block_fill(image, row):
    """Fill color from the shape core: erode past any outline stroke first."""
    from PIL import Image as _I, ImageDraw as _D
    poly = _I.new('L', (image.shape[1], image.shape[0]))
    for c in row['contours'][:1]:
        _D.Draw(poly).polygon([tuple(p) for p in c], fill=255)
    inside = np.asarray(poly) > 0
    if not inside.any():
        return None
    ys, xs = np.nonzero(inside)
    k = max(2, int(.18 * min(np.ptp(xs) + 1, np.ptp(ys) + 1)))
    core = ndimage.binary_erosion(inside, iterations=k)
    samples = image[core if core.sum() >= 20 else inside]
    return '#' + ''.join(f'{int(round(v)):02x}' for v in libstyle.color_mode(samples))


def styled(image, ink, row):
    if row['kind'] == 'block_arrow':
        row['style'] = libstyle.filled(image, None, row)
        fill = block_fill(image, row)
        if fill:
            row['style'].update(color=fill, fill_source='eroded shape core')
        return row
    s, _ = libstyle.shaft(image, ink, row)
    s.update(id=row['id'])
    if row.get('dash_hint'):
        s['dash'] = dict(type='dotted' if row['dash_hint'] == 'sysDot' else 'dashed', preset=row['dash_hint'], source='route gap links')
    for end, head in zip(['start', 'end'], row['heads']):
        size = (row.get('head_size') or {}).get(end, 'med')
        s[end] = None if head == 'none' else dict(type=head, width=size, length=size)
    if s.get('width_px') is None:
        s['width_px'] = row.get('width_estimate') or 2
    row['style'] = s
    return row
