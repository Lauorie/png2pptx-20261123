"""SAM line evidence for fixed contours: square border tiles, upsampled, masks as ownership.

Tiles are square crops centred on the fitted contour, so the SAM processor's
fixed square resize is isotropic. A mask counts as the target line only when its
pixels lie mostly in the contour band. The resulting support can veto ink that
SAM assigns elsewhere; where no target mask exists the support is unknown.
"""
import math
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, binary_dilation
from .lines import subpaths, polyline


def plan_tiles(part, image_size, side, overlap=.25):
    """Square tile boxes [x0, y0, side] covering every subpath of the contour."""
    boxes = []
    for sub in subpaths(part['commands']):
        q = polyline(sub)
        if len(q) < 2:
            continue
        seg = np.diff(q, axis=0); length = np.linalg.norm(seg, axis=1); cum = np.r_[0, np.cumsum(length)]
        if cum[-1] <= 0:
            continue
        stride = side * (1 - overlap)
        count = max(1, math.ceil(cum[-1] / stride))
        for s in (np.arange(count) + .5) * cum[-1] / count:
            i = min(int(np.searchsorted(cum, s, side='right') - 1), len(length) - 1)
            c = q[i] + (s - cum[i]) / max(length[i], 1e-9) * seg[i]
            x0 = int(round(c[0] + .5 - side / 2)); y0 = int(round(c[1] + .5 - side / 2))
            boxes.append([x0, y0, int(side)])
    unique = []
    for b in boxes:
        if b not in unique:
            unique.append(b)
    # Dense routes repeatedly visit the same source neighborhood. A shared
    # half-tile lattice keeps every sampled contour safely inside its tile
    # while allowing different subpaths/parts to reuse one SAM request.
    grid=set();stride=side/2
    for sub in subpaths(part['commands']):
        q=polyline(sub)
        for a,b in zip(q[:-1],q[1:]):
            count=max(2,int(math.ceil(np.linalg.norm(b-a)/(side/16)))+1)
            points=a+(b-a)*np.linspace(0,1,count)[:,None]
            cells=np.floor((points+.5)/stride).astype(int)
            grid.update((int(round(x*stride-side/4)),int(round(y*stride-side/4)),int(side))
                        for x,y in cells)
    if len(grid)<.75*len(unique):
        return [list(b) for b in sorted(grid)]
    return unique


def tile_image(image, box, size):
    """White-padded square crop resampled once to the SAM input size."""
    x0, y0, side = box
    canvas = Image.new('RGB', (side, side), 'white')
    l, t = max(0, x0), max(0, y0); r, b = min(image.width, x0 + side), min(image.height, y0 + side)
    if r > l and b > t:
        canvas.paste(image.crop((l, t, r, b)), (l - x0, t - y0))
    return canvas.resize((size, size), Image.Resampling.LANCZOS)


def to_tile(xy, box, size):
    x0, y0, side = box; k = size / side
    return (np.asarray(xy, float) - [x0, y0] + .5) * k - .5


def path_band(part, box, size, half_width):
    """Tile-pixel distance (in source px) from the fitted contour."""
    line = np.zeros((size, size), bool)
    for sub in subpaths(part['commands']):
        q = to_tile(polyline(sub), box, size)
        if len(q) < 2:
            continue
        for a, b in zip(q[:-1], q[1:]):
            n = max(2, int(np.ceil(np.linalg.norm(b - a))) + 1)
            delta=b-a;lower=0.;upper=1.
            # Only generate the ORIGINAL sample indices that can round into
            # this tile. Clipping must not alter sampling phase or pixels.
            for axis in (0,1):
                if abs(delta[axis])<1e-12:
                    if a[axis]<-.5 or a[axis]>size-.5:upper=-1.;break
                    continue
                lo,hi=sorted(((-.5-a[axis])/delta[axis],(size-.5-a[axis])/delta[axis]))
                lower=max(lower,lo);upper=min(upper,hi)
            if upper<lower:continue
            first=max(0,int(math.ceil(lower*(n-1)-1e-9)))
            last=min(n-1,int(math.floor(upper*(n-1)+1e-9)))
            if first>last:continue
            # np.linspace uses this multiplication sequence. Preserve it so
            # rounding at half pixels stays identical to the full raster.
            ts=np.arange(first,last+1,dtype=float)*(1./(n-1))
            if last==n-1:ts[-1]=1.
            p = np.rint(a + delta * ts[:, None]).astype(int)
            ok = (p[:, 0] >= 0) & (p[:, 0] < size) & (p[:, 1] >= 0) & (p[:, 1] < size)
            line[p[ok, 1], p[ok, 0]] = True
    if not line.any():
        return None
    return distance_transform_edt(~line) * box[2] / size


def select_targets(masks, distance, half_width, scale, min_inside=.6, min_band_area=2.):
    """Masks that are thin structures along this contour; returns union and audit.

    ``scale`` is tile pixels per source pixel; ``min_band_area`` is in source px^2.
    """
    if distance is None:
        return None, []
    band = distance <= half_width + 2.
    union = np.zeros(distance.shape, bool); audit = []
    for m in masks:
        area = int(m['mask'].sum())
        if not area:
            continue
        inside = int((m['mask'] & band).sum())
        frac = inside / area
        keep = frac >= min_inside and inside >= min_band_area * scale * scale
        audit.append({'id': m['id'], 'category': m['category'], 'score': m['score'], 'band_fraction': round(frac, 3),
                      'band_pixels': inside, 'target': bool(keep)})
        if keep:
            union |= m['mask']
    return (union if union.any() else None), audit


class Support:
    """support(coords) -> (known, inside) for source-pixel coordinates."""
    def __init__(self, tiles, size, dilate_source_px=1.):
        self.tiles = []
        for box, union in tiles:
            if union is None:
                continue
            radius = max(1, int(round(dilate_source_px * size / box[2])))
            self.tiles.append((box, binary_dilation(union, iterations=radius)))
        self.size = size

    def __bool__(self):
        return bool(self.tiles)

    def __call__(self, coords):
        coords = np.asarray(coords, float).reshape(-1, 2)
        known = np.zeros(len(coords), bool); inside = np.zeros(len(coords), bool)
        for box, union in self.tiles:
            p = np.rint(to_tile(coords, box, self.size)).astype(int)
            ok = (p[:, 0] >= 0) & (p[:, 0] < self.size) & (p[:, 1] >= 0) & (p[:, 1] < self.size)
            known |= ok
            inside[ok] |= union[p[ok, 1], p[ok, 0]]
        return known, inside
