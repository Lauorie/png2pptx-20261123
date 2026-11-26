"""The fidelity score separates the defects a reviewer sees in a recovered slide."""
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw

from full_recovery.fidelity import score

SIZE: Tuple[int, int] = (400, 240)


def _figure(box: bool = True, bar: Tuple[int, int] = (300, 312), colour: str = '#1F5FBF',
            line: bool = True, extra: bool = False) -> Image.Image:
    image = Image.new('RGB', SIZE, 'white')
    draw = ImageDraw.Draw(image)
    if box:
        draw.rectangle((40, 40, 160, 120), fill='#F4EFE9', outline='black', width=2)
    draw.rectangle((bar[0], 60, bar[1], 200), fill=colour)
    if line:
        draw.line((170, 80, 290, 80), fill='black', width=2)
    if extra:
        draw.line((20, 220, 380, 150), fill='black', width=3)
    return image


def test_an_identical_rendering_scores_one() -> None:
    result = score(_figure(), _figure())
    assert result.recall == 1 and result.precision == 1 and result.colour == 1 and result.overall == 1


def test_a_one_pixel_shift_is_forgiven() -> None:
    shifted = Image.new('RGB', SIZE, 'white')
    shifted.paste(_figure(), (1, 1))
    assert score(_figure(), shifted).overall > 0.97


def test_missing_content_lowers_recall_not_precision() -> None:
    result = score(_figure(), _figure(box=False, line=False))
    assert result.recall < 0.75 and result.precision > 0.97


def test_spurious_content_lowers_precision_not_recall() -> None:
    result = score(_figure(), _figure(extra=True))
    assert result.precision < 0.95 and result.recall > 0.97


def test_a_bar_drawn_three_times_too_wide_lowers_precision() -> None:
    assert score(_figure(), _figure(bar=(288, 324))).precision < 0.9


def test_a_wrong_colour_lowers_colour_only() -> None:
    result = score(_figure(), _figure(colour='#707070'))
    assert result.colour < 0.97 and result.recall == 1 and result.precision == 1


def test_a_blank_rendering_scores_near_zero() -> None:
    assert score(_figure(), Image.new('RGB', SIZE, 'white')).overall < 0.2


def test_sizes_are_reconciled() -> None:
    larger = _figure().resize((800, 480))
    assert score(_figure(), larger).overall > 0.9


def test_a_dark_background_figure_is_handled() -> None:
    dark = Image.fromarray(255 - np.array(_figure()))
    assert score(dark, dark).overall == 1


def _busy(icon: bool = True, fat: bool = False) -> Image.Image:
    image = Image.new('RGB', (1200, 600), 'white')
    draw = ImageDraw.Draw(image)
    for row in range(6):
        for column in range(12):
            draw.rectangle((30 + column * 96, 30 + row * 92, 100 + column * 96, 90 + row * 92), outline='black', width=2)
    if icon:
        draw.ellipse((1130, 540, 1170, 580), outline='#C00000', width=3)
    draw.rectangle((1150 - 12 * fat, 100, 1162 + 12 * fat, 300), fill='#1F5FBF')  # 12 px bar, 36 px when fat
    return image


def test_one_small_missing_icon_is_reported_as_a_defect_region() -> None:
    result = score(_busy(), _busy(icon=False))
    assert result.recall > 0.97 and result.defects > 0
    assert result.overall < score(_busy(), _busy()).overall - 0.005


def test_an_identical_rendering_has_no_defect_region() -> None:
    assert score(_busy(), _busy()).defects == 0


def test_one_small_bar_drawn_too_wide_is_reported_as_a_defect_region() -> None:
    assert score(_busy(), _busy(fat=True)).defects > 0
