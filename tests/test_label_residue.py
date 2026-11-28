"""A picture kept in place of a drawing must not still show the labels the slide adds as text.

The picture model is asked to remove those labels and sometimes returns them
untouched; the editable text then sits on top of its own copy.
"""
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw

from full_recovery.image_edit import source_locked_icon

GROUND = (236, 240, 246)
LABEL = [40, 30, 99, 45]


def source() -> Image.Image:
    image = Image.new('RGB', (160, 90), GROUND)
    draw = ImageDraw.Draw(image)
    draw.line((0, 70, 159, 70), fill=(200, 60, 60), width=3)
    for x in range(42, 98, 8):
        draw.rectangle((x, 32, x+4, 43), fill=(30, 30, 40))
    return image


def dark(image: Image.Image, box: Tuple[int, int, int, int]) -> int:
    pixels = np.asarray(image.convert('RGB').crop((box[0], box[1], box[2]+1, box[3]+1))).astype(int)
    return int((pixels.sum(axis=2) < 300).sum())


def request() -> dict:
    return {'native_labels': [{'text': 'ARC-C', 'box': LABEL}]}


def test_a_label_the_editor_left_in_place_is_removed_from_the_source_pixels() -> None:
    crop = source()
    assert dark(crop, LABEL) > 200
    result, audit = source_locked_icon(crop, crop.copy(), (0, 0, 160, 90), request())
    assert dark(result, LABEL) == 0
    assert audit['labels_left_by_editor'] == 1


def test_a_label_the_editor_removed_keeps_the_editor_pixels() -> None:
    crop = source()
    edited = crop.copy()
    draw = ImageDraw.Draw(edited)
    draw.rectangle((38, 28, 101, 47), fill=GROUND)
    draw.line((38, 38, 101, 38), fill=(60, 90, 200), width=1)
    result, audit = source_locked_icon(crop, edited, (0, 0, 160, 90), request())
    assert audit['labels_left_by_editor'] == 0
    assert result.convert('RGB').getpixel((70, 38)) == (60, 90, 200)
    assert dark(result, LABEL) == 0


def test_pixels_outside_the_label_stay_the_source_pixels() -> None:
    crop = source()
    result, _ = source_locked_icon(crop, Image.new('RGB', (160, 90), 'white'), (0, 0, 160, 90), request())
    assert result.convert('RGB').getpixel((80, 70)) == (200, 60, 60)
    assert result.convert('RGB').getpixel((5, 5)) == GROUND


def test_a_label_replaced_by_the_body_colour_counts_as_removed() -> None:
    crop = source()
    result, audit = source_locked_icon(crop, Image.new('RGB', (160, 90), (255, 0, 0)), (0, 0, 160, 90), request())
    assert audit['labels_left_by_editor'] == 0
    assert result.convert('RGB').getpixel((70, 38)) == (255, 0, 0)
