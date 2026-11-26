"""Read a stroke's colour, width and dash rhythm from the pixels under its path.

These three attributes are visible facts once the path is known, so they are
measured instead of being asked of a model: the same figure always gives the
same answer. The path may be a couple of pixels off the ink, may cross other
lines and may run beside filled shapes; ``measure`` returns ``None`` when the
pixels do not support a confident reading, and the caller keeps what it had.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import map_coordinates

REACH = 17.            # half-length of each cross-section, px
SCALES = (2., 3., 4., 6., 8., 11.)  # px to either side at which a stroke is compared with its ground
STEP = 0.5             # spacing of samples across the stroke, px
SEARCH = 3.            # how far from the path the stroke centre may lie, px
CONTRAST = 14.         # least colour difference from both sides that counts as ink
MINIMUM_LENGTH = 12.   # px of path, after trimming the ends, needed to judge
END_TRIM = 0.15        # share of the path left out at each end (heads, joints)
END_TRIM_LIMIT = 14.   # px, upper bound of that trim
SOLID_SHARE = 0.88     # inked share of the path above which the stroke is solid
INKED_MINIMUM = 0.3    # inked share below which nothing is reported
SAME_INK = 24.         # largest colour difference between pixels of one stroke
SHORT_MINIMUM = 3.     # px of path needed to read a colour alone
SHORT_TRIM = 0.3       # share of a short path left out at each end
RHYTHM = 0.7           # share of marks, and of gaps, that must have the usual length
REGULARITY = 0.5       # largest spread/median of gap lengths in a dash rhythm


@dataclass(frozen=True)
class Stroke:
    """A measured stroke; ``dash`` is empty for solid, else (mark, gap) in px."""

    colour: str
    width_px: float
    dash: Tuple[float, ...]


def _sections(pixels: np.ndarray, path: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return colours along cross-sections of ``path`` and the offsets sampled."""
    tangent = np.gradient(path, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    offsets = np.arange(-REACH, REACH + STEP / 2, STEP)
    points = path[:, None, :] + offsets[None, :, None] * normal[:, None, :]
    colours = np.stack([map_coordinates(pixels[..., channel].astype(np.float32), [points[..., 1], points[..., 0]],
                                        order=1, mode='nearest') for channel in range(3)], axis=-1)
    return colours, offsets


def _resample(path: np.ndarray, minimum: float = MINIMUM_LENGTH, share: float = END_TRIM) -> np.ndarray:
    """Return ``path`` at one-pixel spacing with both ends trimmed."""
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    trim = min(END_TRIM_LIMIT, share * lengths[-1])
    if lengths[-1] - 2 * trim < minimum:
        return np.empty((0, 2))
    at = np.arange(trim, lengths[-1] - trim, 1.)
    return np.stack([np.interp(at, lengths, path[:, axis]) for axis in range(2)], axis=1)


def _runs(flags: np.ndarray) -> np.ndarray:
    """Return [start, end) index pairs of the true runs in ``flags``."""
    return np.flatnonzero(np.diff(np.r_[False, flags, False].astype(np.int8))).reshape(-1, 2)


def _typical(lengths: np.ndarray) -> float:
    """Return the share of ``lengths`` close to their median."""
    middle = float(np.median(lengths))
    return float((np.abs(lengths - middle) <= max(1.5, REGULARITY * middle)).mean())


def _dash(inked: np.ndarray) -> Optional[Tuple[float, ...]]:
    """Return the dash rhythm of an ink/no-ink sequence, or ``None`` if unclear.

    A rhythm is accepted when most marks and most gaps have the usual length;
    a few odd ones (a corner, a crossing, a stretch the path drifted off) do
    not matter.
    """
    share = float(inked.mean())
    if share < INKED_MINIMUM:
        return None
    marks = _runs(inked)
    gaps = np.diff(marks.reshape(-1))[1::2].astype(float)
    inner = np.diff(marks, axis=1).ravel()[1:-1].astype(float)
    regular = len(gaps) >= 3 and len(inner) >= 2 and _typical(gaps) >= RHYTHM and _typical(inner) >= RHYTHM
    if regular and share < SOLID_SHARE:
        return (float(np.median(inner)), float(np.median(gaps)))
    return () if share >= 0.6 else None


def _width(colours: np.ndarray, offsets: np.ndarray, centre: np.ndarray, reach: np.ndarray,
           ink: np.ndarray) -> Optional[float]:
    """Return the median stroke width over the given cross-sections.

    The ground is read just outside the stroke, separately on each side, so
    a shape further away does not count and the two sides may differ.
    """
    pad = int(round(1.5 / STEP))
    last = colours.shape[1] - 1
    widths = []
    for section, middle, half in zip(colours, centre, reach):
        low, high = middle - half - pad, middle + half + pad
        if low < 0 or high > last:
            low, high = middle - half, middle + half
        cover = 0.
        for ground, side in ((section[low], section[low:middle]), (section[high], section[middle:high + 1])):
            span = ink - ground
            cover += float(np.clip((side - ground) @ span / max(float(span @ span), 1e-6), 0., 1.).sum())
        widths.append(cover * STEP)
    if len(widths) < 4:
        return None
    return float(np.median(widths))


def _off_segment(point: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return how far each colour in ``point`` lies from the segment joining ``first`` and ``second``.

    A stroke's colour lies off the line between the colours on its two sides.
    A blended pixel on the edge of a filled shape lies on that line, so an
    edge is not mistaken for a stroke.
    """
    span = second - first
    along = np.clip(((point - first) * span).sum(axis=-1) / np.maximum((span * span).sum(axis=-1), 1e-6), 0., 1.)
    return np.linalg.norm(point - (first + along[..., None] * span), axis=-1)


def _ink(pixels: np.ndarray, route: np.ndarray) -> Optional[Tuple[np.ndarray, ...]]:
    """Find the stroke in each cross-section of ``route``.

    A stroke is a ridge: its colour is unlike the two points an equal
    distance to either side and is not a blend of them. The smallest such
    distance is used, so the ground is what lies right beside the stroke.

    Returns:
        Cross-sections, offsets, the stroke-centre index and the ridge
        half-width (in samples) of each section, ink flags and the ink colour;
        ``None`` when no section shows a stroke.
    """
    colours, offsets = _sections(np.asarray(pixels)[..., :3], route)
    count = len(route)
    rows = np.arange(count)
    near = np.abs(offsets) <= SEARCH
    centre, reach = np.zeros(count, dtype=int), np.zeros(count, dtype=int)
    strength, found = np.zeros(count), np.zeros(count, dtype=bool)
    for scale in SCALES:
        shift = int(round(scale / STEP))
        left, middle, right = colours[:, :-2 * shift], colours[:, shift:-shift], colours[:, 2 * shift:]
        ridge = _off_segment(middle, left, right)
        ridge[:, ~near[shift:-shift]] = 0.
        index = ridge.argmax(axis=1)
        value = ridge[rows, index]
        take = ~found & (value > CONTRAST)
        centre[take], reach[take], strength[take] = index[take] + shift, shift, value[take]
        found |= take
    if not found.any():
        return None
    peak = colours[rows, centre]
    ink = np.median(peak[found][strength[found] >= np.median(strength[found])], axis=0)
    inked = found & (np.abs(peak - ink).max(axis=1) <= 2 * SAME_INK)
    if not inked.any():
        return None
    return colours, offsets, centre, reach, inked, ink


def _hex(ink: np.ndarray) -> str:
    """Return ``ink`` as #RRGGBB."""
    return '#%02X%02X%02X' % tuple(int(round(float(value))) for value in np.clip(ink, 0, 255))


def colour(pixels: np.ndarray, path: np.ndarray) -> Optional[str]:
    """Return the colour of the stroke along ``path`` even when it is too short to measure fully."""
    route = _resample(np.asarray(path, dtype=float), minimum=SHORT_MINIMUM, share=SHORT_TRIM)
    found = None if len(route) == 0 else _ink(pixels, route)
    if found is None or found[4].mean() < INKED_MINIMUM:
        return None
    return _hex(found[5])


def measure(pixels: np.ndarray, path: np.ndarray) -> Optional[Stroke]:
    """Measure the stroke drawn along ``path``.

    Args:
        pixels: The source figure, height by width by 3, 0 to 255.
        path: Points along the stroke centre as (x, y), in drawing order.

    Returns:
        The stroke's colour, width and dash rhythm, or ``None`` when the path
        is too short or the pixels under it do not show one consistent stroke.
    """
    route = _resample(np.asarray(path, dtype=float))
    if len(route) == 0:
        return None
    found = _ink(pixels, route)
    if found is None:
        return None
    colours, offsets, centre, reach, inked, ink = found
    dash = _dash(inked)
    width = _width(colours[inked], offsets, centre[inked], reach[inked], ink)
    if dash is None or width is None:
        return None
    return Stroke(colour=_hex(ink), width_px=width, dash=dash)
