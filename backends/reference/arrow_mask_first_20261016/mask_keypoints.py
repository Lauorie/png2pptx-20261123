"""Native mask -> evidence graph and bend candidates, without arrow identity decisions.

No component deletion, spur pruning, longest-path selection, branch pairing or
gap linking. An edge is a skeleton chain, NOT an arrow. RDP vertices on a curve
are samples, NOT necessarily sharp corners. Pixel adjacency follows the idea
in arrow_graph_20261013/inkgraph.py; graph tracing here keeps every special pixel.
"""
import cv2
import numpy as np
from skimage.morphology import skeletonize


def to_global(points, view):
    points = np.asarray(points, float).reshape(-1, 2)
    return (points + .5) / np.asarray(view['scale']) - .5 + np.asarray(view['bbox_global'][:2])


def pixel_graph(skeleton):
    ys, xs = np.nonzero(skeleton)
    index = np.full(skeleton.shape, -1, np.int32)
    index[ys, xs] = np.arange(len(xs))
    padded = np.pad(skeleton, 1)
    adj = [[] for _ in xs]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == dy == 0:
                continue
            ok = padded[ys + 1 + dy, xs + 1 + dx].copy()
            # Omit only a redundant diagonal when an orthogonal connection exists.
            if dx and dy:
                ok &= ~(padded[ys + 1 + dy, xs + 1] | padded[ys + 1, xs + 1 + dx])
            for i in np.flatnonzero(ok):
                adj[i].append(int(index[ys[i] + dy, xs[i] + dx]))
    return np.column_stack([xs, ys]), adj


def chains(skeleton):
    """Lossless pixel-chain decomposition, including isolated pixels and loops."""
    xy, adj = pixel_graph(skeleton)
    degree = np.array([len(a) for a in adj])
    special = set(map(int, np.flatnonzero(degree != 2)))
    visited, paths = set(), []

    def key(a, b):
        return (min(a, b), max(a, b))

    def trace(a, b):
        path = [a, b]
        visited.add(key(a, b))
        previous, current = a, b
        while current not in special and current != a:
            following = next(n for n in adj[current] if n != previous)
            visited.add(key(current, following))
            path.append(following)
            previous, current = current, following
        paths.append(path)

    for a in sorted(special):
        for b in adj[a]:
            if key(a, b) not in visited:
                trace(a, b)
    # Remaining links belong to pure degree-two cycles.
    for a, neighbors in enumerate(adj):
        for b in neighbors:
            if key(a, b) not in visited:
                trace(a, b)
    return xy, degree, [np.asarray(p, np.int32) for p in paths]


def line_fit(points):
    points = np.asarray(points, float)
    center = points.mean(axis=0)
    _, vectors = np.linalg.eigh((points - center).T @ (points - center))
    direction = vectors[:, -1]
    normal = np.array([-direction[1], direction[0]])
    rms = np.sqrt(np.mean(((points - center) @ normal) ** 2))
    return center, direction, float(rms)


def simplify(path, epsilon):
    if len(path) < 3:
        return path.astype(float), []
    points = cv2.approxPolyDP(path.astype(np.float32), epsilon, False).reshape(-1, 2).astype(float)
    if len(points) < 3:
        return points, []
    # RDP returns input vertices; resolve indices monotonically, including cycles.
    indices, cursor = [], 0
    for p in points:
        i = cursor + int(np.argmin(np.linalg.norm(path[cursor:] - p, axis=1)))
        indices.append(i)
        cursor = min(i + 1, len(path) - 1)
    axes = []
    for a, b in zip(indices, indices[1:]):
        section = path[a:b + 1]
        trim = min(len(section) // 5, max(0, (len(section) - 2) // 2))
        # Avoid the small rounding of skeleton pixels near a rasterized corner.
        axes.append(line_fit(section[trim:len(section)-trim] if trim else section))
    bends = []
    for i in range(1, len(points) - 1):
        c0, d0, r0 = axes[i - 1]
        c1, d1, r1 = axes[i]
        system = np.column_stack([d0, -d1])
        intersection = None
        if abs(np.linalg.det(system)) > 1e-6:
            intersection = c0 + np.linalg.solve(system, c1 - c0)[0] * d0
        u, v = points[i] - points[i-1], points[i+1] - points[i]
        cosine = np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12)
        bends.append(dict(xy_native=points[i].tolist(), turn_degrees=round(float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))), 2),
                          axis_intersection_native=intersection.tolist() if intersection is not None else None,
                          adjacent_axis_rms_native=[r0, r1]))
    return points, bends


def extract(mask, epsilon=1.5, category='line'):
    """Return native pixel geometry plus exact raw skeleton pixels and chain indices."""
    mask = np.asarray(mask, bool)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return dict(nodes=[], edges=[], contours=[], skeleton_pixels=0, components=0), np.empty((0,2),np.int32), []
    left, top = int(xs.min()), int(ys.min())
    roi = np.pad(mask[top:ys.max()+1, left:xs.max()+1], 2)
    origin = np.array([left - 2, top - 2])
    components = int(cv2.connectedComponents(roi.astype(np.uint8), connectivity=8)[0] - 1)
    if category == 'block_arrow':
        contours, hierarchy = cv2.findContours(roi.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        rows = []
        for i, path in enumerate(contours):
            simple = cv2.approxPolyDP(path, epsilon, True).reshape(-1, 2) + origin
            rows.append(dict(id=f'C{i+1}', points_native=simple.tolist(), hierarchy=hierarchy[0,i].tolist()))
        return dict(nodes=[], edges=[], contours=rows, skeleton_pixels=0, components=components,
                    method='contour candidates for filled block arrow'), np.empty((0,2),np.int32), []
    skeleton = skeletonize(roi)
    xy, degree, paths = chains(skeleton)
    radius = cv2.distanceTransform(roi.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    radii = radius[xy[:,1],xy[:,0]] if len(xy) else np.array([])
    xy += origin
    nodes = []
    for i in np.flatnonzero(degree != 2):
        nodes.append(dict(id=f'N{i}', pixel_index=int(i), xy_native=xy[i].tolist(), degree=int(degree[i]),
                          role='isolated' if degree[i]==0 else 'endpoint' if degree[i]==1 else 'junction',
                          inscribed_radius_native=float(radii[i])))
    edges = []
    for i, path in enumerate(paths):
        points, bends = simplify(xy[path], epsilon)
        edges.append(dict(id=f'E{i+1}', start_pixel=int(path[0]), end_pixel=int(path[-1]), closed=bool(path[0]==path[-1]),
                          points_native=points.tolist(), bends=bends, raw_pixels=len(path),
                          length_native=float(np.linalg.norm(np.diff(xy[path],axis=0),axis=1).sum()),
                          median_inscribed_diameter_native=float(np.median(radii[path])*2)))
    return dict(nodes=nodes, edges=edges, contours=[], skeleton_pixels=len(xy), components=components,
                method='skeleton chains + RDP + independent adjacent TLS axes; no semantic joining'), xy, paths


def attach_frame(data, view):
    for node in data['nodes']:
        node['xy_global'] = to_global([node['xy_native']], view)[0].tolist()
    for edge in data['edges']:
        edge['points_global'] = to_global(edge['points_native'], view).tolist()
        for bend in edge['bends']:
            bend['xy_global'] = to_global([bend['xy_native']], view)[0].tolist()
            p = bend['axis_intersection_native']
            bend['axis_intersection_global'] = to_global([p], view)[0].tolist() if p is not None else None
    for contour in data['contours']:
        contour['points_global'] = to_global(contour['points_native'], view).tolist()
    return data
