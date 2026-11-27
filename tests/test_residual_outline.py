"""A missing thing is judged, and patched, within its own outline, not its bounding rectangle."""
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pptx import Presentation
from pptx.util import Emu

from full_recovery.residual import missing_areas, missing_boxes, patch

SIZE = (400, 300)


def _hatch(draw: ImageDraw.ImageDraw, box: tuple, colour: str = '#203040') -> None:
    for x in range(box[0], box[2], 6):
        draw.line([(x, box[1]), (x, box[3])], fill=colour, width=2)


def _pair(corner_present: bool = True) -> tuple:
    """An L-shaped hatched picture; a hatched label sits in the corner the L leaves free."""
    source = Image.new('RGB', SIZE, 'white'); rendered = Image.new('RGB', SIZE, 'white')
    for image, with_picture in ((source, True), (rendered, False)):
        draw = ImageDraw.Draw(image)
        if with_picture:
            _hatch(draw, (40, 40, 360, 120)); _hatch(draw, (40, 120, 140, 260))
        if with_picture or corner_present:
            _hatch(draw, (190, 160, 360, 260), '#8A1F1F')
    return source, rendered


def test_a_missing_picture_is_found_although_its_rectangle_holds_present_content() -> None:
    source, rendered = _pair()
    boxes = missing_boxes(np.asarray(source), np.asarray(rendered))
    assert len(boxes) == 1
    left, top, right, bottom = boxes[0]
    assert left <= 42 and top <= 42 and right >= 356 and bottom >= 256


def test_the_patch_leaves_present_content_in_its_rectangle_uncovered() -> None:
    source, rendered = _pair()
    area = missing_areas(np.asarray(source), np.asarray(rendered))[0]
    left, top = area.box[:2]
    assert not area.mask[200 - top:250 - top, 230 - left:350 - left].any()
    assert area.mask[50 - top:110 - top, 60 - left:340 - left].all()


def test_flat_parts_inside_a_missing_picture_are_kept() -> None:
    source = Image.new('RGB', SIZE, 'white'); rendered = Image.new('RGB', SIZE, 'white')
    ImageDraw.Draw(source).rectangle((60, 60, 300, 220), fill='#2E7D5B', outline='#10402C', width=3)
    area = missing_areas(np.asarray(source), np.asarray(rendered))[0]
    assert area.mask.all()


def test_content_present_all_through_a_missing_frame_keeps_it_from_being_patched_over() -> None:
    source = Image.new('RGB', SIZE, 'white'); rendered = Image.new('RGB', SIZE, 'white')
    ImageDraw.Draw(source).rectangle((40, 40, 360, 260), outline='#203040', width=2)
    for image in (source, rendered):
        _hatch(ImageDraw.Draw(image), (60, 60, 340, 240), '#8A1F1F')
    assert missing_boxes(np.asarray(source), np.asarray(rendered)) == []


def test_a_patch_picture_is_transparent_outside_the_outline(tmp_path: Path) -> None:
    source, rendered = _pair()
    area = missing_areas(np.asarray(source), np.asarray(rendered))[0]
    deck = Presentation(); deck.slide_width = Emu(SIZE[0] * 9525); deck.slide_height = Emu(SIZE[1] * 9525)
    deck.slides.add_slide(deck.slide_layouts[6]); path = tmp_path / 'deck.pptx'; deck.save(str(path))
    patch(path, path, source, [area.box], [area.mask])
    picture = Presentation(str(path)).slides[0].shapes[-1]
    pixels = np.asarray(Image.open(io.BytesIO(picture.image.blob)).convert('RGBA'))
    left, top = area.box[:2]
    assert pixels[220 - top, 300 - left, 3] == 0 and pixels[80 - top, 200 - left, 3] == 255
