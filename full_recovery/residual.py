"""Keep source content that the composed slide shows nothing of.

Every branch can miss something: a paragraph of small print, an icon, a logo.
Rather than a rule per kind of omission, the finished slide is compared with
the source once. Wherever the source has content and the slide has nothing
nearby, that part of the source is placed on the slide as a picture. The
patch is not editable, but the reader sees what the figure showed.

Only blank areas qualify. Content that is present but displaced, recoloured
or set in another font has slide content near it and is left alone.

A missing thing is judged within its own outline, and the picture carries
only what lies inside that outline. Its bounding rectangle usually holds
other content too, a label beside a photograph for instance, which must
neither count against the missing thing nor be covered by its picture.
"""
import io
import json
import logging
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

from .fidelity import contour

logger = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]
NEAR = 6            # px: slide content this close explains source content
JOIN = 7            # px: unexplained content this close forms one patch
MINIMUM = 60        # unexplained contour pixels a patch needs
SHARE = 0.7         # share of the source content inside a patch's outline that must be unexplained
PAD = 2             # px added around a patch
SQUARE = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class Area:
    """One missing thing: its box in source pixels and, within the box, the pixels inside its outline."""

    box: Box
    mask: np.ndarray


def missing_areas(source: np.ndarray, rendered: np.ndarray) -> List[Area]:
    """Return the source content with no slide content near it.

    Args:
        source: The source figure, height by width by 3.
        rendered: The slide rendered at the same size.

    Returns:
        One area per missing thing, with a half-open (left, top, right, bottom) box.
    """
    present = contour(source)
    unexplained = present & ~ndimage.binary_dilation(contour(rendered), SQUARE, iterations=NEAR)
    labels, _ = ndimage.label(ndimage.binary_dilation(unexplained, SQUARE, iterations=JOIN))
    height, width = present.shape
    areas: List[Area] = []
    for number, area in enumerate(ndimage.find_objects(labels), 1):
        outline = ndimage.binary_fill_holes(labels[area] == number)
        lost = int((unexplained[area] & outline).sum())
        if lost < MINIMUM or lost < SHARE * int((present[area] & outline).sum()):
            continue
        rows, columns = np.nonzero(unexplained[area] & outline)
        box = (max(0, area[1].start + int(columns.min()) - PAD), max(0, area[0].start + int(rows.min()) - PAD),
               min(width, area[1].start + int(columns.max()) + 1 + PAD),
               min(height, area[0].start + int(rows.max()) + 1 + PAD))
        whole = np.zeros(present.shape, dtype=bool)
        whole[area] = outline
        areas.append(Area(box, whole[box[1]:box[3], box[0]:box[2]].copy()))
    return areas


def missing_boxes(source: np.ndarray, rendered: np.ndarray) -> List[Box]:
    """Return only the boxes of :func:`missing_areas`."""
    return [area.box for area in missing_areas(source, rendered)]


def patch(deck_path: Path, output: Path, source: Image.Image, boxes: Sequence[Box],
          masks: Optional[Sequence[np.ndarray]] = None) -> None:
    """Add each box of ``source`` to the single slide of ``deck_path`` as a top-most picture.

    With ``masks``, one per box, a picture is transparent wherever its mask is false.
    """
    from pptx import Presentation
    deck = Presentation(str(deck_path))
    slide = deck.slides[0]
    scale = deck.slide_width / source.width
    for number, box in enumerate(boxes, 1):
        data = io.BytesIO()
        piece = source.crop(box)
        if masks is not None:
            piece = piece.convert('RGBA')
            piece.putalpha(Image.fromarray(np.where(masks[number - 1], 255, 0).astype(np.uint8)))
        piece.save(data, format='PNG')
        data.seek(0)
        picture = slide.shapes.add_picture(data, round(box[0] * scale), round(box[1] * scale),
                                           round((box[2] - box[0]) * scale), round((box[3] - box[1]) * scale))
        picture.name = 'RESIDUAL %d' % number
    temporary = Path(str(output) + '.tmp')
    deck.save(str(temporary))
    shutil.move(str(temporary), str(output))


def recover_missing(deck_path: Path, source_path: Path, render_path: Path, folder: Path) -> List[Box]:
    """Patch ``deck_path`` in place with the source content its render lacks.

    Args:
        deck_path: The composed single-slide deck.
        source_path: The source figure.
        render_path: The deck rendered to an image.
        folder: Where the receipt is written.

    Returns:
        The patched boxes; empty when the deck was left unchanged.
    """
    with Image.open(source_path) as opened:
        source = opened.convert('RGB')
    with Image.open(render_path) as opened:
        rendered = opened.convert('RGB')
    if rendered.size != source.size:
        rendered = rendered.resize(source.size, Image.LANCZOS)
    areas = missing_areas(np.asarray(source), np.asarray(rendered))
    boxes = [area.box for area in areas]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'receipt.json').write_text(json.dumps({'protocol': 'source_residual_pictures.2', 'boxes': boxes,
                                                     'model_calls': 0}, indent=2) + '\n')
    if boxes:
        patch(deck_path, deck_path, source, boxes, [area.mask for area in areas])
        logger.info('Kept %d area(s) of the source as pictures', len(boxes))
    return boxes
