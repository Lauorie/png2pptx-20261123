"""Stroke measurement along a fixed contour from original pixels.

Every quantity is read from CLEAN source RGB along the unchanged fitted path:
the stroke colour is the profile colour that is not a mix of the two sides,
width is the integrated alpha coverage, and the line pattern is the 50 % coverage
run sequence along arc length (unbiased for antialiased edges). SAM support can
only veto ink that lies outside the target's line mask (text, crossing
connectors); it never creates ink or dash gaps. Unknown regions are not gaps.
"""
import numpy as np
from scipy.ndimage import map_coordinates, median_filter

STEP = .5        # arc-length spacing, source px
DOFF = .25       # normal-offset spacing, source px
DISTINCT = 24.   # RGB distance from the two-side mix segment that marks stroke ink
FOREIGN = 40.    # RGB residual after the alpha model that marks another colour
SIDE_WINDOW = 41 # running-median length of side colours, in arc samples (~20 px)


def hexcolor(c):
    return '#' + ''.join(f'{int(v):02X}' for v in np.clip(np.rint(c), 0, 255))


def rgb(c):
    return np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)], float)


def subpaths(commands):
    current = []
    for c in commands:
        if c[0] == 'M' and current:
            yield current
            current = []
        current.append(c)
    if current:
        yield current


def polyline(commands, steps=96):
    out = []; current = start = None
    for c in commands:
        if c[0] == 'M':
            current = np.array(c[1:], float); start = current.copy(); out.append(current)
        elif c[0] == 'L':
            current = np.array(c[1:], float); out.append(current)
        elif c[0] == 'C':
            p0 = current; p1, p2, p3 = (np.array(c[i:i + 2], float) for i in (1, 3, 5))
            for t in np.linspace(0, 1, steps + 1)[1:]:
                out.append((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t * t * p2 + t ** 3 * p3)
            current = p3
        elif c[0] == 'Z':
            if current is not None and not np.allclose(current, start):
                out.append(start)
            current = start
    return np.asarray(out, float)


def arc_samples(part, corner_radius):
    """Equal arc-length samples per subpath with sharp-corner flags."""
    rows = []
    for k, sub in enumerate(subpaths(part['commands'])):
        q = polyline(sub)
        if len(q) < 2:
            continue
        q = q[np.r_[True, np.linalg.norm(np.diff(q, axis=0), axis=1) > 1e-6]]
        if len(q) < 2:
            continue
        seg = np.diff(q, axis=0); length = np.linalg.norm(seg, axis=1); cum = np.r_[0, np.cumsum(length)]
        n = max(2, int(cum[-1] / STEP))
        s = (np.arange(n) + .5) * cum[-1] / n
        i = np.clip(np.searchsorted(cum, s, side='right') - 1, 0, len(length) - 1)
        tangent = seg[i] / length[i, None]
        points = q[i] + (s - cum[i])[:, None] * tangent
        closed = sub[-1][0] == 'Z'
        unit = seg / length[:, None]
        turn = np.degrees(np.arccos(np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1, 1)))
        vertex = cum[1:-1][turn > 30]
        if closed and len(unit) > 1:
            if np.degrees(np.arccos(np.clip(unit[-1] @ unit[0], -1, 1))) > 30:
                vertex = np.r_[vertex, 0., cum[-1]]
        corner = np.zeros(n, bool)
        for v in vertex:
            corner |= np.abs(s - v) < corner_radius
        if not closed:
            # Open-path ends meet nodes, heads or other lines; they say nothing about the shaft.
            guard = min(max(4., 2 * corner_radius), .25 * cum[-1])
            corner |= (s < guard) | (s > cum[-1] - guard)
        rows.append({'points': points, 'tangent': tangent, 's': s, 'corner': corner, 'closed': closed,
                     'length': float(cum[-1])})
    return rows


def sample_rgb(image, coords):
    return np.stack([map_coordinates(image[..., k], [coords[..., 1], coords[..., 0]], order=1, mode='nearest')
                     for k in range(3)], axis=-1)


def segment_distance(x, a, b):
    """Distance of colours x from the RGB segment a-b (a mix of the two sides)."""
    d = b - a; den = np.maximum(np.sum(d * d, axis=-1, keepdims=True), 1e-6)
    t = np.clip(np.sum((x - a) * d, axis=-1, keepdims=True) / den, 0, 1)
    return np.linalg.norm(x - (a + t * d), axis=-1)


def color_mode(colors):
    colors = np.asarray(colors, float)
    bins = np.floor(colors / 12).astype(int)
    _, inverse, counts = np.unique(bins, axis=0, return_inverse=True, return_counts=True)
    return np.median(colors[np.asarray(inverse).ravel() == np.argmax(counts)], axis=0)


def runs(binary):
    return np.where(np.diff(np.r_[False, binary, False].astype(int)))[0].reshape(-1, 2)


def pattern(on_valid, step):
    """Interior on/off run lengths of one valid stretch (truncated end runs dropped)."""
    r = runs(on_valid)
    on = [(b - a) * step for a, b in r[1:-1]] if len(r) > 2 else []
    gaps = [(r[i + 1, 0] - r[i, 1]) * step for i in range(len(r) - 1)]
    return on, gaps


def robust(values):
    v = np.asarray(values, float)
    m = float(np.median(v))
    return m, float(np.median(np.abs(v - m)) / max(m, .5))


def classify(on, gaps, coverage, presence, width):
    """Periodic runs first; otherwise continuous ink presence decides solid.

    ``coverage`` is the 50 %-width on-fraction; ``presence`` is the fraction of
    visible samples with distinct stroke ink (insensitive to gradient fills).
    """
    longest = 40 * max(width, 1.5)  # a longer "dash" is an interrupted solid line
    if len(on) >= 5 and len(gaps) >= 4:
        gap, gap_mad = robust(gaps)
        on_arr = np.asarray(on, float)
        lo, hi = np.quantile(on_arr, [.2, .8])
        if hi > 2.2 * lo and lo > 0 and gap >= .75 and coverage < .9:
            cut = np.sqrt(lo * hi); short = on_arr[on_arr < cut]; long = on_arr[on_arr >= cut]
            if len(short) >= 2 and len(long) >= 2:
                (a, ma), (b, mb) = robust(long), robust(short)
                if max(ma, mb, gap_mad) < .3 and b >= 1 and a <= longest:
                    return {'dash': 'dash_dot', 'dash_reliable': True, 'dash_pattern_px': [a, gap, b, gap]}
        mark, mark_mad = robust(on_arr)
        if max(mark_mad, gap_mad) < .3 and gap >= .75 and coverage < .9 and mark <= longest:
            kind = 'dot' if mark <= 1.6 * max(width, 1.) else 'dash'
            # A round dot cannot be substantially shorter than its diameter.
            # One-pixel diagonal staircase repeats beside a broad fitted band
            # otherwise become "reliable" dots and grow when rendered.
            if kind=='dot' and mark<.5*width:
                return {'dash':'unknown','dash_reliable':False,
                        'dash_reason':'pattern_finer_than_stroke_width'}
            return {'dash': kind, 'dash_reliable': True, 'dash_pattern_px': [mark, gap]}
    if presence >= .9 or coverage >= .9:
        return {'dash': 'solid', 'dash_reliable': True}
    return {'dash': 'unknown', 'dash_reliable': False,
            'dash_reason': 'too_few_repeats' if len(on) < 5 or len(gaps) < 4 else 'irregular_runs'}


def measure(image, part, excluded=None, support=None, guess_width=2.):
    """Measure one part's stroke. ``excluded`` and ``support`` map (N,2) xy -> bool."""
    image = np.asarray(image, float)
    h, w = image.shape[:2]
    # Side colours come from a narrow band just outside the expected stroke and a
    # running median along the contour: local text/labels near a border cannot
    # fake periodic gaps, while slow gradient fills are preserved.
    radius = float(np.clip(.65 * max(guess_width, 1.5) + 5.5, 8, 16))
    rows = arc_samples(part, max(3., 1.5 * guess_width))
    offsets = np.arange(-radius, radius + DOFF / 2, DOFF)
    window = np.abs(offsets) <= min(4.5, radius - 2)
    side_n = int(round(1.5 / DOFF))
    per = []
    for row in rows:
        normal = np.c_[-row['tangent'][:, 1], row['tangent'][:, 0]]
        coords = row['points'][:, None, :] + normal[:, None, :] * offsets[None, :, None]
        inside = ((coords[..., 0] >= 0) & (coords[..., 0] <= w - 1) & (coords[..., 1] >= 0) & (coords[..., 1] <= h - 1))
        blocked = excluded(coords.reshape(-1, 2)).reshape(coords.shape[:2]) if excluded is not None else np.zeros(inside.shape, bool)
        profile = sample_rgb(image, coords)
        mode = 'wrap' if row['closed'] else 'nearest'
        b_minus = median_filter(np.median(profile[:, :side_n], axis=1), size=(SIDE_WINDOW, 1), mode=mode)
        b_plus = median_filter(np.median(profile[:, -side_n:], axis=1), size=(SIDE_WINDOW, 1), mode=mode)
        distinct = segment_distance(profile, b_minus[:, None], b_plus[:, None])
        distinct[:, ~window] = -1
        peak = np.argmax(distinct, axis=1)
        rows_i = np.arange(len(peak))
        per.append({'row': row, 'coords': coords, 'profile': profile, 'b_minus': b_minus, 'b_plus': b_plus,
                    'peak': peak, 'distinct': distinct[rows_i, peak],
                    'visible': inside.all(axis=1) & ~blocked.any(axis=1)})
    if not per:
        return {'measured': False, 'reason': 'no_geometry'}
    visible = np.concatenate([p['visible'] for p in per])
    distinct = np.concatenate([p['distinct'] for p in per])
    corner = np.concatenate([p['row']['corner'] for p in per])
    present = visible & (distinct > DISTINCT)
    base = {'samples': int(len(visible)), 'visible_fraction': round(float(visible.mean()), 3)}
    if visible.mean() < .25:
        return {'measured': False, 'reason': 'insufficient_visible_boundary', **base}
    good = present & ~corner
    if good.sum() < 12 or good.sum() < .1 * (visible & ~corner).sum():
        absent = visible.sum() >= 40 and present[visible].mean() < .05
        return {'measured': True, 'present': False, 'presence_reliable': bool(absent),
                'present_fraction': round(float(present[visible].mean()), 3), **base}
    peaks = np.concatenate([p['profile'][np.arange(len(p['peak'])), p['peak']] for p in per])
    strong = good & (distinct >= np.quantile(distinct[good], .6))
    color = color_mode(peaks[strong])
    widths, coverage_rows, backgrounds = [], [], []
    for p in per:
        n = len(p['peak']); prof = p['profile']
        width_i = np.zeros(n); ok = np.zeros(n, bool); vetoed = np.zeros(n, bool)
        for i in range(n):
            if not p['visible'][i] or p['distinct'][i] <= DISTINCT:
                continue
            k = p['peak'][i]
            back = np.where((np.arange(len(offsets)) <= k)[:, None], p['b_minus'][i], p['b_plus'][i])
            delta = color - back; den = np.maximum(np.sum(delta * delta, axis=1), 1)
            raw = np.sum((prof[i] - back) * delta, axis=1) / den
            # Ink darker/stronger than the estimated colour lies on the same ray;
            # only colours off the background->stroke ray are foreign.
            residual = np.linalg.norm(prof[i] - (back + np.clip(raw, 0, 1.6)[:, None] * delta), axis=1)
            alpha = np.clip(raw, 0, 1)
            alpha[residual > FOREIGN] = 0
            if alpha[k] < .3:
                continue
            a = b = k
            while a > 0 and alpha[a - 1] > .08:
                a -= 1
            while b < len(alpha) - 1 and alpha[b + 1] > .08:
                b += 1
            if a == 0 or b == len(alpha) - 1:
                continue
            if support is not None:
                known, inside = support(p['coords'][i, a:b + 1])
                ink = alpha[a:b + 1] > .5
                if known.any() and ink.any() and inside[ink].mean() < .3:
                    vetoed[i] = True
                    continue
            width_i[i] = alpha[a:b + 1].sum() * DOFF; ok[i] = True
            backgrounds.append(back[0] if np.linalg.norm(color - p['b_minus'][i]) >= np.linalg.norm(color - p['b_plus'][i]) else back[-1])
        widths.append(width_i[ok & ~p['row']['corner']])
        coverage_rows.append((width_i, ok, vetoed))
    width_samples = np.concatenate(widths)
    if len(width_samples) < 12 or np.median(width_samples) <= 0:
        return {'measured': True, 'present': None, 'reason': 'too_few_width_samples', **base}
    width, width_mad = robust(width_samples)
    # Dash ends are partially covered; width comes from samples inside painted runs.
    interior = []
    for width_i, ok, _ in coverage_rows:
        on = ok & (width_i > .5 * width)
        inner = on & np.r_[False, on[:-1]] & np.r_[on[1:], False]
        interior.append(width_i[inner])
    interior = np.concatenate(interior)
    if len(interior) >= 12:
        width, width_mad = robust(interior)
    # Pattern from 50 % coverage along arc length; vetoed/unseen samples split stretches.
    # A sample is a gap only if the integrated coverage is low AND the stroke centre
    # does not show the stroke colour: side-colour errors (labels hugging a border)
    # cannot fake gaps, and centre aliasing on diagonals cannot either.
    centre_tol = .25 * float(np.median([np.linalg.norm(color - b) for b in backgrounds])) if backgrounds else 40.
    on_all, gap_all, on_count, valid_count, centre_rows = [], [], 0, 0, []
    for p, (width_i, ok, vetoed) in zip(per, coverage_rows):
        valid = p['visible'] & ~vetoed & ~p['row']['corner']
        centre = median_filter(p['peak'], size=21, mode='wrap' if p['row']['closed'] else 'nearest')
        ink = np.linalg.norm(p['profile'][np.arange(len(centre)), centre] - color, axis=1) < centre_tol
        on = (ok & (width_i > .5 * width)) | (ink & p['visible'] & ~vetoed)
        centre_rows.append(on)
        if p['row']['closed'] and valid.all():
            shift = int(np.argmin(on)) if (~on).any() else 0
            valid, on = np.roll(valid, -shift), np.roll(on, -shift)
        for a, b in runs(valid):
            o, g = pattern(on[a:b], STEP)
            on_all += o; gap_all += g
            on_count += int(on[a:b].sum()); valid_count += b - a
    coverage = on_count / max(valid_count, 1)
    usable = np.concatenate([p['visible'] & ~v & ~p['row']['corner'] for p, (_, _, v) in zip(per, coverage_rows)])
    presence = float((present | np.concatenate(centre_rows))[usable].mean()) if usable.any() else 0.
    dash = classify(on_all, gap_all, coverage, presence, width)
    background = np.median(np.asarray(backgrounds), axis=0) if backgrounds else np.full(3, 255.)
    if dash.get('dash_pattern_px'):
        dash['dash_pattern_px'] = [round(float(v), 3) for v in dash['dash_pattern_px']]
    # A stroke far from the fitted contour means the geometry does not follow the
    # source line; nothing measured along it is reliable.
    centres = np.concatenate([offsets[p['peak']][p['visible'] & (p['distinct'] > DISTINCT)] for p in per])
    offset = float(np.median(np.abs(centres))) if len(centres) else 0.
    reliable_geometry = offset <= max(1.5, .75 * width)
    if not reliable_geometry:
        dash.update(dash_reliable=False, dash_reason='geometry_offset')
    return {'measured': True, 'present': True, 'geometry_offset_px': round(offset, 2),
            'presence_reliable': bool(reliable_geometry and (coverage >= .5 or dash['dash_reliable'])),
            'color': hexcolor(color), 'width_px': round(width, 3), 'width_mad_ratio': round(width_mad, 3),
            'width_reliable': bool(reliable_geometry and len(width_samples) >= 20 and width_mad < .25 and presence >= .5),
            'background': hexcolor(background), 'contrast': round(float(np.linalg.norm(color - background)), 1),
            'coverage': round(coverage, 3), 'presence': round(presence, 3), 'vetoed_samples': int(sum(v.sum() for _, _, v in coverage_rows)),
            'runs': {'on': len(on_all), 'off': len(gap_all)}, **dash, **base}


def ink_width(measurement, stroke_color):
    """Width that keeps the measured ink when a different stroke colour is chosen."""
    if not measurement.get('width_px') or not stroke_color:
        return None
    back = rgb(measurement['background'])
    old = np.linalg.norm(rgb(measurement['color']) - back); new = np.linalg.norm(rgb(stroke_color) - back)
    if old < 1 or new < 1:
        return measurement['width_px']
    return float(measurement['width_px'] * np.clip(old / new, .5, 2.))
