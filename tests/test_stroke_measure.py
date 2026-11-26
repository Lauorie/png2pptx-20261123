"""A stroke's colour, width and dash rhythm are read from the pixels under its path."""
from typing import Sequence, Tuple

import numpy as np
import pytest
from PIL import Image, ImageDraw

from full_recovery.stroke_measure import measure

Point = Tuple[float, float]


def _canvas(background: str = 'white') -> Tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new('RGB', (400, 200), background)
    return image, ImageDraw.Draw(image)


def _path(start: Point, end: Point) -> np.ndarray:
    count = int(np.hypot(end[0] - start[0], end[1] - start[1])) + 1
    return np.linspace(start, end, count)


def _dashes(draw: ImageDraw.ImageDraw, y: int, marks: Sequence[int], colour: str, width: int) -> None:
    x = 40
    while x < 360:
        draw.line((x, y, min(360, x + marks[0] - 1), y), fill=colour, width=width)
        x += marks[0] + marks[1]


def _rgb(colour: str) -> Tuple[int, int, int]:
    return tuple(int(colour[i:i + 2], 16) for i in (1, 3, 5))


def test_a_solid_black_line_is_measured() -> None:
    image, draw = _canvas()
    draw.line((40, 100, 360, 100), fill='black', width=2)
    stroke = measure(np.asarray(image), _path((40, 100), (360, 100)))
    assert stroke is not None and stroke.dash == ()
    assert max(_rgb(stroke.colour)) < 40 and stroke.width_px == pytest.approx(2, abs=0.6)


def test_a_pale_line_on_a_tinted_panel_keeps_its_colour() -> None:
    image, draw = _canvas('#EAF3E6')
    draw.line((40, 100, 360, 100), fill='#B9BDB7', width=3)
    stroke = measure(np.asarray(image), _path((40, 100), (360, 100)))
    assert stroke is not None and stroke.dash == ()
    assert np.abs(np.subtract(_rgb(stroke.colour), _rgb('#B9BDB7'))).max() < 16
    assert stroke.width_px == pytest.approx(3, abs=0.7)


def test_a_dashed_line_reports_its_rhythm() -> None:
    image, draw = _canvas()
    _dashes(draw, 100, (10, 6), '#C0504D', 2)
    stroke = measure(np.asarray(image), _path((40, 100), (360, 100)))
    assert stroke is not None and len(stroke.dash) == 2
    assert stroke.dash[0] == pytest.approx(10, abs=2) and stroke.dash[1] == pytest.approx(6, abs=2)
    assert np.abs(np.subtract(_rgb(stroke.colour), _rgb('#C0504D'))).max() < 24


def test_a_dotted_line_has_marks_about_as_long_as_it_is_wide() -> None:
    image, draw = _canvas()
    _dashes(draw, 100, (3, 5), 'black', 3)
    stroke = measure(np.asarray(image), _path((40, 100), (360, 100)))
    assert stroke is not None and len(stroke.dash) == 2 and stroke.dash[0] <= 5


def test_a_path_two_pixels_off_the_line_still_finds_it() -> None:
    image, draw = _canvas()
    draw.line((40, 100, 360, 100), fill='#1F5FBF', width=2)
    stroke = measure(np.asarray(image), _path((40, 102), (360, 102)))
    assert stroke is not None and stroke.width_px == pytest.approx(2, abs=0.6)
    assert np.abs(np.subtract(_rgb(stroke.colour), _rgb('#1F5FBF'))).max() < 24


def test_a_thick_stroke_is_not_clipped() -> None:
    image, draw = _canvas()
    draw.line((40, 100, 360, 100), fill='#4F81BD', width=9)
    stroke = measure(np.asarray(image), _path((40, 100), (360, 100)))
    assert stroke is not None and stroke.width_px == pytest.approx(9, abs=1)


def test_crossing_lines_do_not_turn_a_solid_line_into_dashes_or_widen_it() -> None:
    image, draw = _canvas()
    draw.line((40, 100, 360, 100), fill='black', width=2)
    for x in (120, 200, 290):
        draw.line((x, 20, x, 180), fill='black', width=2)
    stroke = measure(np.asarray(image), _path((40, 100), (360, 100)))
    assert stroke is not None and stroke.dash == () and stroke.width_px == pytest.approx(2, abs=0.6)


def test_a_filled_box_beside_the_line_does_not_widen_it() -> None:
    image, draw = _canvas()
    draw.rectangle((40, 106, 360, 180), fill='#F2DCDB')
    draw.line((40, 100, 360, 100), fill='#666666', width=2)
    stroke = measure(np.asarray(image), _path((40, 100), (360, 100)))
    assert stroke is not None and stroke.width_px == pytest.approx(2, abs=0.7)


def test_a_slanted_line_is_measured_across_not_along() -> None:
    image, draw = _canvas()
    draw.line((40, 30, 360, 170), fill='black', width=3)
    stroke = measure(np.asarray(image), _path((40, 30), (360, 170)))
    assert stroke is not None and stroke.width_px == pytest.approx(3, abs=0.9) and stroke.dash == ()


def test_no_ink_under_the_path_gives_no_measurement() -> None:
    image, _ = _canvas()
    assert measure(np.asarray(image), _path((40, 100), (360, 100))) is None


def test_a_path_too_short_to_judge_gives_no_measurement() -> None:
    image, draw = _canvas()
    draw.line((40, 100, 46, 100), fill='black', width=2)
    assert measure(np.asarray(image), _path((40, 100), (46, 100))) is None


def test_a_short_stroke_still_gives_its_colour() -> None:
    from full_recovery.stroke_measure import colour
    image, draw = _canvas('#EAF3E6')
    draw.line((40, 100, 49, 100), fill='#C0504D', width=2)
    found = colour(np.asarray(image), _path((40, 100), (49, 100)))
    assert found is not None and np.abs(np.subtract(_rgb(found), _rgb('#C0504D'))).max() < 40


def test_no_ink_gives_no_colour() -> None:
    from full_recovery.stroke_measure import colour
    image, _ = _canvas()
    assert colour(np.asarray(image), _path((40, 100), (49, 100))) is None


def test_a_faint_thin_line_on_a_tinted_panel_is_measured() -> None:
    image, draw = _canvas('#EAF3E6')
    draw.line((40, 100, 360, 100), fill='#C9CEC6', width=1)
    stroke = measure(np.asarray(image), _path((40, 100), (360, 100)))
    assert stroke is not None and stroke.dash == () and stroke.width_px == pytest.approx(1, abs=0.5)
