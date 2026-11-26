"""A filled part cannot carry an outline as wide as the part itself."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'vendor')]

from shape_integration.styling.stroke_fit import fit  # noqa: E402

BAR = (570.0, 53.0, 578.0, 108.0)  # an 8 px wide bar


def test_an_outline_as_wide_as_the_bar_is_dropped() -> None:
    style, audit = fit({'fill': '#005CC1', 'stroke': '#004A91', 'width_px': 8.4}, BAR)
    assert style['stroke'] is None and style['fill'] == '#005CC1'
    assert audit['before_width_px'] == 8.4 and audit['smaller_side_px'] == 8.0


def test_a_thin_outline_on_the_same_bar_is_untouched() -> None:
    original = {'fill': '#005CC1', 'stroke': '#004A91', 'width_px': 1.4}
    assert fit(original, BAR) == (original, None)


def test_an_unfilled_line_keeps_its_width() -> None:
    original = {'fill': None, 'stroke': '#000000', 'width_px': 6.0}
    assert fit(original, (0.0, 0.0, 200.0, 2.0)) == (original, None)


def test_a_part_without_outline_is_untouched() -> None:
    original = {'fill': '#005CC1', 'stroke': None, 'width_px': 9.0}
    assert fit(original, BAR) == (original, None)


def test_the_input_style_is_not_modified() -> None:
    original = {'fill': '#005CC1', 'stroke': '#004A91', 'width_px': 8.4}
    fit(original, BAR)
    assert original['stroke'] == '#004A91'


def test_a_thick_contrasting_outline_on_a_thin_inset_is_real_and_kept() -> None:
    # A light 5 px inset framed in black is how a thick dark frame is drawn.
    original = {'fill': '#FFFFFF', 'stroke': '#000000', 'width_px': 4.0}
    assert fit(original, (0.0, 0.0, 5.0, 34.0)) == (original, None)


def test_a_wide_outline_in_a_clearly_different_colour_is_kept() -> None:
    original = {'fill': '#FFFFFF', 'stroke': '#FF9F7C', 'width_px': 7.3}
    assert fit(original, (0.0, 0.0, 8.0, 8.0)) == (original, None)
