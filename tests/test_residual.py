"""Source content that the slide shows nothing of is kept as a picture of the source."""
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Emu

from full_recovery.residual import missing_boxes, patch, recover_missing

SIZE: Tuple[int, int] = (400, 200)
UNIT = 9525


def _figure(paragraph: bool = True, speck: bool = False, shift: int = 0, colour: str = '#1F5FBF') -> Image.Image:
    image = Image.new('RGB', SIZE, 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((20 + shift, 20, 120 + shift, 90), fill=colour, outline='black', width=2)
    if paragraph:
        for row in range(4):
            for word in range(6):
                draw.rectangle((200 + word * 28, 40 + row * 14, 220 + word * 28, 46 + row * 14), fill='black')
    if speck:
        draw.point((300, 170), fill='black')
    return image


def _boxes(source: Image.Image, rendered: Image.Image) -> list:
    return missing_boxes(np.asarray(source), np.asarray(rendered))


def test_a_paragraph_absent_from_the_slide_is_boxed() -> None:
    boxes = _boxes(_figure(), _figure(paragraph=False))
    assert len(boxes) == 1
    left, top, right, bottom = boxes[0]
    assert left <= 200 and top <= 40 and right >= 360 and bottom >= 88 and left > 130


def test_a_faithful_slide_needs_nothing() -> None:
    assert _boxes(_figure(), _figure()) == []


def test_content_that_is_present_but_slightly_displaced_is_not_boxed() -> None:
    assert _boxes(_figure(), _figure(shift=3)) == []


def test_content_that_is_present_in_another_colour_is_not_boxed() -> None:
    assert _boxes(_figure(), _figure(colour='#C0504D')) == []


def test_a_speck_is_not_worth_a_picture() -> None:
    assert _boxes(_figure(speck=True), _figure()) == []


def _deck(tmp_path: Path) -> Path:
    deck = Presentation()
    deck.slide_width, deck.slide_height = Emu(SIZE[0] * UNIT), Emu(SIZE[1] * UNIT)
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(20 * UNIT), Emu(20 * UNIT), Emu(100 * UNIT), Emu(70 * UNIT)).name = 'G1'
    deck.save(tmp_path / 'slide.pptx')
    return tmp_path / 'slide.pptx'


def test_patches_are_pictures_of_the_source_on_top_at_the_right_place(tmp_path: Path) -> None:
    deck = _deck(tmp_path)
    patch(deck, deck, _figure(), [(196, 36, 364, 92)])
    shapes = list(Presentation(str(deck)).slides[0].shapes)
    assert [shape.name for shape in shapes] == ['G1', 'RESIDUAL 1']
    picture = shapes[1]
    assert (picture.left, picture.top, picture.width, picture.height) == (196 * UNIT, 36 * UNIT, 168 * UNIT, 56 * UNIT)


def test_recovering_reports_the_boxes_and_changes_the_deck_only_when_needed(tmp_path: Path) -> None:
    deck = _deck(tmp_path)
    _figure().save(tmp_path / 'source.png')
    _figure().save(tmp_path / 'same.png')
    _figure(paragraph=False).save(tmp_path / 'render.png')
    before = deck.read_bytes()
    assert recover_missing(deck, tmp_path / 'source.png', tmp_path / 'same.png', tmp_path / 'r0') == []
    assert deck.read_bytes() == before
    boxes = recover_missing(deck, tmp_path / 'source.png', tmp_path / 'render.png', tmp_path / 'r1')
    assert len(boxes) == 1 and len(Presentation(str(deck)).slides[0].shapes) == 2
    assert (tmp_path / 'r1/receipt.json').is_file()


def test_a_render_of_another_size_is_compared_at_the_source_size(tmp_path: Path) -> None:
    deck = _deck(tmp_path)
    _figure().save(tmp_path / 'source.png')
    _figure(paragraph=False).resize((800, 400)).save(tmp_path / 'render.png')
    assert len(recover_missing(deck, tmp_path / 'source.png', tmp_path / 'render.png', tmp_path / 'r')) == 1
