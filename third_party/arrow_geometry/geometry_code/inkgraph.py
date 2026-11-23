"""Ink network from SAM masks.

Arrow ink = union of SAM line masks. Its skeleton is a graph; arrows are
edge-disjoint routes through that graph. Geometry always comes from pixels;
only the semantic choices (which ink is an arrow, where heads are, which ends
continue across gaps, how junctions decompose) are open for the model.
"""
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

STEPS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def pixel_graph(skel):
    """8-adjacency without redundant diagonals (avoids fake junctions at corners)."""
    h, w = skel.shape
    ys, xs = np.nonzero(skel)
    index = -np.ones(skel.shape, int)
    index[ys, xs] = np.arange(len(ys))
    pad = np.pad(skel, 1)
    adj = [[] for _ in range(len(ys))]
    for dy, dx in STEPS:
        yy, xx = ys + dy, xs + dx
        ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        ok[ok] = skel[yy[ok], xx[ok]]
        if dy and dx:
            ok &= ~(pad[ys + 1 + dy, xs + 1] | pad[ys + 1, xs + 1 + dx])
        for i in np.flatnonzero(ok):
            adj[i].append(int(index[yy[i], xx[i]]))
    return np.column_stack([xs, ys]).astype(float), adj


class Graph:
    """Nodes carry positions; edges carry ordered pixel paths (x, y)."""

    def __init__(self):
        self.nodes, self.edges, self._next = {}, {}, 0

    def add_node(self, xy, **kw):
        k = self._next; self._next += 1
        self.nodes[k] = dict(xy=np.asarray(xy, float), edges=set(), **kw)
        return k

    def add_edge(self, a, b, path, **kw):
        k = self._next; self._next += 1
        self.edges[k] = dict(a=a, b=b, path=np.asarray(path, float), **kw)
        self.nodes[a]['edges'].add(k); self.nodes[b]['edges'].add(k)
        return k

    def remove_edge(self, e):
        d = self.edges.pop(e)
        self.nodes[d['a']]['edges'].discard(e); self.nodes[d['b']]['edges'].discard(e)

    def degree(self, n):
        return sum(2 if self.edges[e]['a'] == self.edges[e]['b'] else 1 for e in self.nodes[n]['edges'])

    def other(self, e, n):
        d = self.edges[e]
        return d['b'] if d['a'] == n else d['a']

    def oriented(self, e, n):
        """Edge path starting at node n."""
        d = self.edges[e]
        return d['path'] if d['a'] == n else d['path'][::-1]

    def length(self, e):
        p = self.edges[e]['path']
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.

    def merge_chains(self):
        """Dissolve degree-2 nodes that only join two plain edges."""
        changed = True
        while changed:
            changed = False
            for n in list(self.nodes):
                node = self.nodes.get(n)
                if node is None or node.get('keep') or len(node['edges']) != 2:
                    continue
                e1, e2 = node['edges']
                if self.edges[e1].get('gap') or self.edges[e2].get('gap'):
                    continue
                a, b = self.other(e1, n), self.other(e2, n)
                if a == n or b == n:
                    continue
                path = np.vstack([self.oriented(e1, a), self.oriented(e2, n)[1:]])
                self.remove_edge(e1); self.remove_edge(e2); del self.nodes[n]
                self.add_edge(a, b, path)
                changed = True


def build(skel):
    xy, adj = pixel_graph(skel)
    deg = np.array([len(a) for a in adj])
    g = Graph()
    owner = -np.ones(len(xy), int)
    # Junction clusters / terminals become nodes.
    special = deg != 2
    for i in np.flatnonzero(special):
        if owner[i] >= 0:
            continue
        stack, members = [i], []
        owner[i] = -2
        while stack:
            j = stack.pop(); members.append(j)
            for k in adj[j]:
                if special[k] and owner[k] == -1:
                    owner[k] = -2; stack.append(k)
        n = g.add_node(xy[members].mean(axis=0), pixels=members)
        owner[members] = n
    visited = set()

    def trace(p, q):
        path, prev, cur = [p, q], p, q
        while owner[cur] < 0:
            nxt = [r for r in adj[cur] if r != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            path.append(cur)
            if cur == p:
                break
        visited.add((p, q)); visited.add((cur, prev))
        return path, cur

    for n in list(g.nodes):
        for p in g.nodes[n]['pixels']:
            for q in adj[p]:
                if owner[q] == n or (p, q) in visited:
                    continue
                path, end = trace(p, q)
                g.add_edge(n, owner[end] if owner[end] >= 0 else n, xy[path])
    # Pure loops (no terminal / junction) get one artificial node.
    covered = set()
    for d in g.edges.values():
        covered.update(map(tuple, np.rint(d['path']).astype(int).tolist()))
    for i in range(len(xy)):
        if owner[i] >= 0 or tuple(xy[i].astype(int)) in covered:
            continue
        n = g.add_node(xy[i], pixels=[i]); owner[i] = n
        for q in adj[i]:
            if (i, q) in visited:
                continue
            path, end = trace(i, q)
            g.add_edge(n, owner[end] if owner[end] >= 0 else n, xy[path])
            covered.update(map(tuple, np.rint(xy[path]).astype(int).tolist()))
    return g


def sample_dt(dt, path):
    p = np.rint(path).astype(int)
    return dt[p[:, 1], p[:, 0]]


def tangent(path, width, reverse=False):
    """Outward-facing direction at path[0] (pointing away from the path body)."""
    p = path[::-1] if reverse else path
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if s[-1] < 1e-6:
        return np.zeros(2)
    a = min(max(1.5, 1.0 * width), s[-1] * .3)
    b = min(max(6., 5 * width), s[-1])
    use = (s >= a) & (s <= b)
    pts = p[use] if use.sum() >= 2 else p[: max(2, len(p) // 2)]
    c = pts.mean(axis=0)
    _, v = np.linalg.eigh((pts - c).T @ (pts - c) + 1e-9 * np.eye(2))
    d = v[:, -1]
    if d @ (p[0] - c) < 0:
        d = -d
    return d / max(np.linalg.norm(d), 1e-9)


def edge_width(dt, path):
    v = sample_dt(dt, path)
    n = len(v)
    mid = v[n // 5: n - n // 5] if n >= 10 else v
    return float(np.median(mid)) if len(mid) else 1.


def prune(g, dt):
    """Remove short terminal spurs at junctions (arrowhead barbs, skeleton noise).

    Spurs are recorded on the junction so a head tip can be recovered.
    Threshold scales with the local inscribed radius; never removes all edges.
    """
    for n in list(g.nodes):
        if n not in g.nodes or g.degree(n) < 3:
            continue
        r = float(max(sample_dt(dt, g.nodes[n]['xy'][None])[0], 1.))
        spurs = []
        for e in g.nodes[n]['edges']:
            m = g.other(e, n)
            if m != n and g.degree(m) == 1 and g.length(e) < 3.0 * r + 2:
                spurs.append(e)
        keep_one = len(spurs) == len(g.nodes[n]['edges'])
        if keep_one:
            spurs.remove(max(spurs, key=g.length))
        removed = []
        for e in spurs:
            m = g.other(e, n)
            removed.append(g.oriented(e, n)[-1])
            g.remove_edge(e)
            if not g.nodes[m]['edges']:
                del g.nodes[m]
        if removed:
            g.nodes[n].setdefault('spurs', []).extend(removed)
    g.merge_chains()


def march(mask, p, d, limit):
    """Last in-mask point along a ray from p (sub-pixel steps)."""
    h, w = mask.shape
    last = p.copy()
    for s in np.arange(.5, limit, .5):
        q = p + s * d
        x, y = int(round(q[0])), int(round(q[1]))
        if not (0 <= x < w and 0 <= y < h) or not mask[y, x]:
            break
        last = q
    return last


def analyze_ends(g, mask, dt):
    """Head evidence and extended tip at every terminal node."""
    for n, node in g.nodes.items():
        if g.degree(n) == 0:
            node.update(tip=node['xy'].copy(), head=False, bulge=1., shaft_halfwidth=1., head_halfwidth=1.)
        if g.degree(n) != 1:
            continue
        (e,) = node['edges']
        path = g.oriented(e, n)
        w = edge_width(dt, g.edges[e]['path'])
        node['shaft_halfwidth'] = w
        d = tangent(path, w)
        node['dir'] = d
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        near = path[s <= max(10., 6 * w)]
        h = float(sample_dt(dt, near).max()) if len(near) else w
        h = max(h, float(sample_dt(dt, node['xy'][None])[0]))
        node['head_halfwidth'] = h
        spurs = node.get('spurs', [])
        node['bulge'] = h / max(w, .5)
        node['head'] = bool(node['bulge'] >= 1.7 and h - w >= 1.5) or len(spurs) >= 2
        tip = march(mask, node['xy'], d, 6 * h + 4)
        if spurs:
            # Candidate tip from pruned spurs: the one continuing the shaft.
            cand = np.asarray(spurs)
            score = (cand - node['xy']) @ d
            best = cand[np.argmax(score)]
            if score.max() > 0 and np.linalg.norm(best - node['xy']) > np.linalg.norm(tip - node['xy']):
                tip = march(mask, node['xy'], (best - node['xy']) / np.linalg.norm(best - node['xy']), 6 * h + 4)
        node['tip'] = tip
    blob_heads(g, mask, dt)


def blob_heads(g, mask, dt):
    """A detached short blob that narrows to one end is a head without its shaft.

    Its pointed end becomes the head tip; the broad end is a free end that can
    link to the shaft. Rectangular dashes/dots are not tapered and stay as is.
    """
    labels, _ = ndimage.label(mask, np.ones((3, 3), int))
    for e, ed in list(g.edges.items()):
        a, b = ed['a'], ed['b']
        if a == b or g.degree(a) != 1 or g.degree(b) != 1 or ed.get('gap'):
            continue
        r = float(sample_dt(dt, ed['path']).max())
        if g.length(e) > 4 * r + 6 or r < 2:
            continue
        x, y = np.rint(ed['path'][len(ed['path']) // 2]).astype(int)
        ys, xs = np.nonzero(labels == labels[y, x])
        pts = np.column_stack([xs, ys]).astype(float)
        c = pts.mean(axis=0)
        u = g.nodes[b]['xy'] - g.nodes[a]['xy']
        if np.linalg.norm(u) < 1:
            _, v = np.linalg.eigh((pts - c).T @ (pts - c)); u = v[:, -1]
        u = u / np.linalg.norm(u); nrm = np.array([-u[1], u[0]])
        t = (pts - c) @ u; s = (pts - c) @ nrm
        lo, hi = t.min(), t.max(); q = .25 * (hi - lo)
        w_lo = np.ptp(s[t <= lo + q]) + 1; w_hi = np.ptp(s[t >= hi - q]) + 1
        bins = np.floor((t - lo) / 2).astype(int)
        w_max = max(np.ptp(s[bins == k]) + 1 for k in np.unique(bins))
        # A detached head is widest at its broad back; a short arrow with a
        # shaft is narrow at BOTH ends (shaft, tip) and widest in between.
        if max(w_lo, w_hi) < .7 * w_max or min(w_lo, w_hi) > .6 * w_max:
            continue
        tip_at_a = w_lo < w_hi  # t increases from a to b
        tip_node, back_node = (a, b) if tip_at_a else (b, a)
        end_t = lo if tip_at_a else hi
        sel = abs(t - end_t) <= 1.5
        tip = c + end_t * u + np.median(s[sel]) * nrm
        out = -u if tip_at_a else u  # outward direction at the tip
        tn, bn = g.nodes[tip_node], g.nodes[back_node]
        tn.update(head=True, tip=tip, dir=out, blob=True)
        bn.update(head=False, dir=-out, blob_back=True,
                  tip=c + (hi if tip_at_a else lo) * u)


def pair_junctions(g, dt, max_turn=60):
    """At each junction pair incident edges by tangent continuity (straight-through)."""
    pairs = {}
    for n, node in g.nodes.items():
        es = [e for e in node['edges'] if g.edges[e]['a'] != g.edges[e]['b']]
        if g.degree(n) < 3:
            continue
        r = float(max(sample_dt(dt, node['xy'][None])[0], 1.))
        dirs = {e: tangent(g.oriented(e, n), max(r, edge_width(dt, g.edges[e]['path']))) for e in es}
        cand = []
        for i, a in enumerate(es):
            for b in es[i + 1:]:
                turn = np.degrees(np.arccos(np.clip(-dirs[a] @ dirs[b], -1, 1)))
                cand.append((turn, a, b))
        used = set()
        node['pairs'] = []
        for turn, a, b in sorted(cand):
            if turn > max_turn or a in used or b in used:
                continue
            used.update([a, b]); node['pairs'].append((a, b, float(turn)))
            pairs[(n, a)] = b; pairs[(n, b)] = a
    return pairs


def routes(g, pairs):
    """Maximal chains: continue through degree-2 nodes and paired junction edges."""
    used, out = set(), []

    def nxt(n, e):
        if g.degree(n) == 2 and len(g.nodes[n]['edges']) == 2:
            (f,) = [x for x in g.nodes[n]['edges'] if x != e] or [None]
            return f
        return pairs.get((n, e))

    def walk(n, e):
        chain, node = [], n
        while e is not None and e not in used:
            used.add(e)
            chain.append((e, node))
            node = g.other(e, node)
            e = nxt(node, e)
        return chain, node

    starts = [n for n in g.nodes if g.degree(n) != 2]
    for n in starts:
        for e in list(g.nodes[n]['edges']):
            if e in used:
                continue
            if g.degree(n) >= 3 and (n, e) in pairs and pairs[(n, e)] not in used:
                continue  # interior of a through-route; start from an end instead
            chain, end = walk(n, e)
            if chain:
                out.append(dict(start=n, end=end, chain=chain))
    for e in list(g.edges):  # cycles
        if e not in used:
            n = g.edges[e]['a']
            chain, end = walk(n, e)
            out.append(dict(start=n, end=end, chain=chain, closed=True))
    for r in out:
        pts = [g.oriented(e, n) for e, n in r['chain']]
        r['path'] = np.vstack([pts[0]] + [p[1:] for p in pts[1:]])
        r['gaps'] = [i for i, (e, _) in enumerate(r['chain']) if g.edges[e].get('gap')]
    return out


def link_gaps(g, dt, instance_of, max_collinear=60, max_same=500):
    """Bridge free ends across dash gaps / occlusions. Edges marked gap=True.

    Same-SAM-instance ends may link at longer range (SAM said one object);
    otherwise only short collinear continuations. Greedy mutual matching.
    """
    ends = [n for n in g.nodes if g.degree(n) == 1 and not g.nodes[n].get('head')]
    dots = [n for n in g.nodes if g.degree(n) == 0]
    cand = []
    pts = {n: g.nodes[n]['tip'] if 'tip' in g.nodes[n] else g.nodes[n]['xy'] for n in ends + dots}
    for i, a in enumerate(ends + dots):
        for b in (ends + dots)[i + 1:]:
            ga = pts[b] - pts[a]
            d = float(np.linalg.norm(ga))
            if d < 1e-6 or d > max_same:
                continue
            u = ga / d
            ca = g.nodes[a]['dir'] @ u if 'dir' in g.nodes[a] else 1.
            cb = -(g.nodes[b]['dir'] @ u) if 'dir' in g.nodes[b] else 1.
            same = bool(instance_of(a) & instance_of(b))
            w = max(g.nodes[a].get('shaft_halfwidth', 1), g.nodes[b].get('shaft_halfwidth', 1))
            if same and min(ca, cb) > .5:
                ok = True
            elif same and d < 6 * w + 12 and min(ca, cb) > -.2:
                ok = True
            elif d < 3 * w + 4:
                ok = True  # touching pieces, e.g. a corner split between two masks
            elif min(ca, cb) > .9 and d < max_collinear:
                ok = True
            else:
                ok = False
            if ok:
                cand.append((d * (1 + 2 * (1 - ca) + 2 * (1 - cb)) * (1 if same else 1.5), a, b))
    used = set()
    links = []
    for cost, a, b in sorted(cand):
        if a in used or b in used:
            continue
        if same_component(g, a, b):
            continue
        used.update([a, b])
        e = g.add_edge(a, b, np.vstack([pts[a], pts[b]]), gap=True)
        links.append(e)
    return links


def same_component(g, a, b):
    seen, stack = {a}, [a]
    while stack:
        n = stack.pop()
        if n == b:
            return True
        for e in g.nodes[n]['edges']:
            m = g.other(e, n)
            if m not in seen:
                seen.add(m); stack.append(m)
    return False
