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

REACH = 12.            # half-length of each cross-section, px
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


def _dash(inked: np.ndarray) -> Optional[Tuple[float, ...]]:
    """Return the dash rhythm of an ink/no-ink sequence, or ``None`` if unclear."""
    share = float(inked.mean())
    if share < INKED_MINIMUM:
        return None
    marks = _runs(inked)
    gaps = np.diff(marks.reshape(-1))[1::2].astype(float)
    inner = np.diff(marks, axis=1).ravel()[1:-1].astype(float)
    regular = len(gaps) >= 3 and np.ptp(gaps) <= max(2., REGULARITY * np.median(gaps)) and len(inner) >= 2 \
        and np.ptp(inner) <= max(2., REGULARITY * np.median(inner))
    if regular and share < SOLID_SHARE:
        return (float(np.median(inner)), float(np.median(gaps)))
    return () if share >= 0.6 else None


def _width(colours: np.ndarray, offsets: np.ndarray, centre: np.ndarray, ink: np.ndarray,
           left: np.ndarray, right: np.ndarray) -> Optional[float]:
    """Return the median stroke width over cross-sections with one clear stroke."""
    widths = []
    for section, middle, first, second in zip(colours, centre, left, right):
        cover = np.empty(len(section))
        for ground, side in ((first, slice(None, middle)), (second, slice(middle, None))):
            span = ink - ground
            cover[side] = np.clip((section[side] - ground) @ span / max(float(span @ span), 1e-6), 0., 1.)
        solid = cover > 0.25
        if not solid[middle]:
            continue
        low = high = middle
        while low > 0 and solid[low - 1]:
            low -= 1
        while high < len(solid) - 1 and solid[high + 1]:
            high += 1
        if offsets[low] <= -REACH + 2 or offsets[high] >= REACH - 2:
            continue
        widths.append(float(cover[low:high + 1].sum()) * STEP)
    if len(widths) < 4:
        return None
    return float(np.median(widths))


def _ink(pixels: np.ndarray, route: np.ndarray) -> Optional[Tuple[np.ndarray, ...]]:
    """Return cross-sections, offsets, stroke centres, ink flags, ink colour and both side colours."""
    colours, offsets = _sections(np.asarray(pixels)[..., :3], route)
    edge = int(round(3 / STEP))
    left, right = np.median(colours[:, :edge], axis=1), np.median(colours[:, -edge:], axis=1)
    apart = np.minimum(np.abs(colours - left[:, None]).max(axis=2), np.abs(colours - right[:, None]).max(axis=2))
    centre = np.where(np.abs(offsets) <= SEARCH, apart, -1.).argmax(axis=1)
    strength = apart[np.arange(len(route)), centre]
    inked = strength > CONTRAST
    if not inked.any():
        return None
    peak = colours[np.arange(len(route)), centre]
    ink = np.median(peak[inked][strength[inked] >= np.median(strength[inked])], axis=0)
    inked &= np.abs(peak - ink).max(axis=1) <= 2 * SAME_INK
    if not inked.any():
        return None
    return colours, offsets, centre, inked, ink, left, right


def _hex(ink: np.ndarray) -> str:
    """Return ``ink`` as #RRGGBB."""
    return '#%02X%02X%02X' % tuple(int(round(float(value))) for value in np.clip(ink, 0, 255))


def colour(pixels: np.ndarray, path: np.ndarray) -> Optional[str]:
    """Return the colour of the stroke along ``path`` even when it is too short to measure fully."""
    route = _resample(np.asarray(path, dtype=float), minimum=SHORT_MINIMUM, share=SHORT_TRIM)
    found = None if len(route) == 0 else _ink(pixels, route)
    if found is None or found[3].mean() < INKED_MINIMUM:
        return None
    return _hex(found[4])


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
    colours, offsets, centre, inked, ink, left, right = found
    dash = _dash(inked)
    width = _width(colours[inked], offsets, centre[inked], ink, left[inked], right[inked])
    if dash is None or width is None:
        return None
    return Stroke(colour=_hex(ink), width_px=width, dash=dash)
