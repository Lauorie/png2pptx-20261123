"""Score how faithfully a rendered slide reproduces its source figure.

No model is involved, so the same pair always gets the same score. Three
parts, each between 0 and 1:

* ``recall``: share of the source's contours that the rendering also has
  nearby. Missing icons, lines and panels lower it.
* ``precision``: share of the rendering's contours that the source has
  nearby. Invented lines, borders and oversized shapes lower it.
* ``colour``: closeness of colours where either image has content.
* ``defects``: share of the figure's regions that are locally wrong. A
  missing icon barely moves a whole-figure average, yet it is the first
  thing a reader sees, so the figure is also judged region by region.

Contours are colour edges, so a thin arrow counts as much as the edge of a
large panel, a shape inside a coloured panel still counts, and a pale noisy
fill contributes nothing. The score does not read text: a misread word with
a similar outline costs little, and a substituted font costs a little.
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

INK_THRESHOLD = 12
EDGE_THRESHOLD = 60.
EDGE_SIGMA = 1.
TOLERANCE_DIVISOR = 700
REGIONS_ACROSS = 24
REGION_MINIMUM = 0.75
REGION_CONTENT = 12
SQUARE = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class Fidelity:
    """Fidelity of one rendering; every field lies between 0 and 1."""

    recall: float
    precision: float
    colour: float
    defects: float
    overall: float


def _background(pixels: np.ndarray) -> np.ndarray:
    """Return the most common colour on the border of ``pixels``."""
    border = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
    colours, counts = np.unique(border // 8, axis=0, return_counts=True)
    chosen = colours[counts.argmax()]
    return border[(border // 8 == chosen).all(axis=1)].mean(axis=0)


def _ink(pixels: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Return the mask of pixels that differ from the background."""
    return np.abs(pixels.astype(np.int16) - background).max(axis=2) > INK_THRESHOLD


def contour(pixels: np.ndarray) -> np.ndarray:
    """Return the mask of pixels where the colour changes sharply."""
    smooth = ndimage.gaussian_filter(pixels.astype(np.float32), (EDGE_SIGMA, EDGE_SIGMA, 0))
    strength = np.hypot(ndimage.sobel(smooth, axis=0), ndimage.sobel(smooth, axis=1)).max(axis=2)
    return strength > EDGE_THRESHOLD


def _covered(mask: np.ndarray, other: np.ndarray, tolerance: int) -> float:
    """Return the share of ``mask`` lying within ``tolerance`` of ``other``."""
    total = int(mask.sum())
    if total == 0:
        return 1.
    near = ndimage.binary_dilation(other, SQUARE, iterations=tolerance)
    return float((mask & near).sum()) / total


def _regions(mask: np.ndarray, side: int) -> np.ndarray:
    """Return the number of set pixels in each ``side`` by ``side`` region."""
    rows = np.add.reduceat(mask.astype(np.int32), np.arange(0, mask.shape[0], side), axis=0)
    return np.add.reduceat(rows, np.arange(0, mask.shape[1], side), axis=1)


def _defects(first: np.ndarray, second: np.ndarray, tolerance: int) -> float:
    """Return the share of regions with content whose contours disagree."""
    side = max(8, first.shape[1] // REGIONS_ACROSS)
    near_first = ndimage.binary_dilation(first, SQUARE, iterations=tolerance)
    near_second = ndimage.binary_dilation(second, SQUARE, iterations=tolerance)
    source, rendered = _regions(first, side), _regions(second, side)
    recall = np.where(source > 0, _regions(first & near_second, side) / np.maximum(source, 1), 1.)
    precision = np.where(rendered > 0, _regions(second & near_first, side) / np.maximum(rendered, 1), 1.)
    content = np.maximum(source, rendered) >= REGION_CONTENT
    if not content.any():
        return 0.
    return float((np.minimum(recall, precision)[content] < REGION_MINIMUM).mean())


def _colour(source: np.ndarray, rendered: np.ndarray, content: np.ndarray, tolerance: int) -> float:
    """Return colour closeness over ``content`` after smoothing both images."""
    if not content.any():
        return 1.
    size = (2 * tolerance + 1, 2 * tolerance + 1, 1)
    first = ndimage.uniform_filter(source.astype(np.float32), size)
    second = ndimage.uniform_filter(rendered.astype(np.float32), size)
    return 1. - float(np.abs(first - second)[content].mean()) / 255.


def score(source: Image.Image, rendered: Image.Image) -> Fidelity:
    """Compare a rendered slide with the figure it was recovered from.

    Args:
        source: The original figure.
        rendered: The slide rendered to an image; resized to the source size.

    Returns:
        Recall, precision, colour closeness, the share of defective regions
        and ``overall = F1(recall, precision) * colour * (1 - defects)``.
    """
    reference = source.convert('RGB')
    size: Tuple[int, int] = reference.size
    candidate = rendered.convert('RGB')
    if candidate.size != size:
        candidate = candidate.resize(size, Image.LANCZOS)
    first, second = np.asarray(reference), np.asarray(candidate)
    background = _background(first)
    first_ink, second_ink = _ink(first, background), _ink(second, background)
    tolerance = max(2, round(size[0] / TOLERANCE_DIVISOR))
    first_contour, second_contour = contour(first), contour(second)
    recall = _covered(first_contour, second_contour, tolerance)
    precision = _covered(second_contour, first_contour, tolerance)
    colour = _colour(first, second, first_ink | second_ink, tolerance)
    harmonic = 0. if recall + precision == 0 else 2 * recall * precision / (recall + precision)
    defects = _defects(first_contour, second_contour, tolerance)
    return Fidelity(recall=recall, precision=precision, colour=colour, defects=defects,
                    overall=harmonic * colour * (1. - defects))
