"""Lettering that neither the text branch nor a retained picture carries is read again.

The text branch leaves out lettering it takes to be inside a screenshot, and
the picture branch keeps only the photographs inside a drawn page. What falls
between the two is found from the pixels: text-like source ink that the
slide without text does not show and no text row covers, gathered into regions.
"""
from typing import Dict, List

import pytest
from PIL import Image

from full_recovery.text_regions import line_rows, regions, validate, view


def mark(x: int, y: int, w: int = 6, h: int = 9, colour: str = '#202020') -> Dict:
    return {'bbox_2d': [x, y, x+w-1, y+h-1], 'ink_pixels': w*h//2, 'native_coverage': 0., 'color': colour}


def paragraph(x: int, y: int, letters: int, lines: int) -> List[Dict]:
    return [mark(x+8*i, y+14*j) for j in range(lines) for i in range(letters)]


def test_dense_unread_lettering_becomes_one_region() -> None:
    found = regions(paragraph(100, 50, 12, 2)+[mark(600, 400), mark(900, 30), mark(20, 700)], (1000, 800))
    assert len(found) == 1
    assert found[0]['marks'] == 24
    x0, y0, x1, y1 = found[0]['bbox_2d']
    assert x0 <= 100 and y0 <= 50 and x1 >= 100+8*11+5 and y1 >= 50+14+8
    assert x1 < 400 and y1 < 200


def test_scattered_marks_are_not_a_region() -> None:
    assert regions([mark(50*i, 60*i) for i in range(1, 9)], (1000, 800)) == []


def test_a_dashed_vertical_line_is_not_a_region() -> None:
    assert regions([mark(300, 100+10*i, w=2, h=7) for i in range(20)], (1000, 800)) == []


def test_the_regions_with_most_lettering_come_first_and_the_number_is_bounded() -> None:
    marks = paragraph(50, 50, 6, 1)+paragraph(50, 300, 12, 3)+paragraph(600, 50, 9, 1)+paragraph(600, 500, 7, 1)
    found = regions(marks, (1000, 800), limit=2)
    assert [r['marks'] for r in found] == [36, 9]


def test_the_view_enlarges_the_region_and_blanks_text_that_is_already_recovered() -> None:
    source = Image.new('RGB', (400, 200), '#F0F0F0')
    for x in range(100, 140):
        for y in range(50, 60):
            source.putpixel((x, y), (0, 0, 0))
    for x in range(160, 200):
        for y in range(50, 60):
            source.putpixel((x, y), (0, 0, 0))
    region = {'bbox_2d': [90, 40, 210, 70], 'marks': 8, 'mark_height': 10.}
    picture, frame = view(source, region, [{'id': 'T1', 'bbox_2d': [158, 48, 201, 61]}])
    assert frame['scale'] > 1
    def at(x: float, y: float) -> tuple:
        return picture.getpixel((round((x-frame['origin'][0])*frame['scale']), round((y-frame['origin'][1])*frame['scale'])))
    assert at(120, 55) == (0, 0, 0)
    assert at(180, 55) == (240, 240, 240)


def test_a_reply_is_refused_when_a_line_lies_outside_the_view_or_carries_control_characters() -> None:
    good = {'lines': [{'text': 'Whale shark', 'bbox_2d': [10, 10, 200, 40], 'bold': True, 'italic': False}]}
    assert validate(good, (400, 200))['lines'][0]['text'] == 'Whale shark'
    assert validate({'lines': []}, (400, 200)) == {'lines': []}
    with pytest.raises(ValueError):
        validate({'lines': [{'text': 'x', 'bbox_2d': [10, 10, 900, 40], 'bold': False, 'italic': False}]}, (400, 200))
    with pytest.raises(ValueError):
        validate({'lines': [{'text': 'a\x00b', 'bbox_2d': [10, 10, 100, 40], 'bold': False, 'italic': False}]}, (400, 200))
    with pytest.raises(ValueError):
        validate({'texts': []}, (400, 200))


def test_a_line_takes_its_box_and_colour_from_the_source_ink_it_names() -> None:
    marks = [mark(100+8*i, 50, colour='#102030') for i in range(10)]
    frame = {'origin': [80, 30], 'scale': 2.}
    lines = [{'text': 'Whale shark', 'bbox_2d': [30, 30, 220, 70], 'bold': True, 'italic': False}]
    rows = line_rows(lines, frame, marks, used={'T1'}, font='Arial')
    assert len(rows) == 1
    row = rows[0]
    assert row['bbox_2d'] == [100., 50., 177., 58.]
    assert row['text'] == 'Whale shark' and row['runs'] == [{'text': 'Whale shark', 'script': 'normal'}]
    assert row['color'] == '#102030' and row['bold'] is True and row['kind'] == 'text'
    assert row['id'] not in {'T1'} and row['layout_mode'] == 'single_line'


def test_a_line_with_no_source_ink_under_it_is_dropped() -> None:
    marks = [mark(100+8*i, 50) for i in range(10)]
    frame = {'origin': [80, 30], 'scale': 2.}
    lines = [{'text': 'Whale shark', 'bbox_2d': [30, 30, 220, 70], 'bold': False, 'italic': False},
             {'text': 'invented', 'bbox_2d': [30, 200, 220, 240], 'bold': False, 'italic': False}]
    assert [r['text'] for r in line_rows(lines, frame, marks, used=set(), font='Arial')] == ['Whale shark']


def test_two_lines_never_share_the_same_ink() -> None:
    marks = [mark(100+8*i, 50) for i in range(10)]
    frame = {'origin': [0, 0], 'scale': 1.}
    lines = [{'text': 'Whale shark', 'bbox_2d': [95, 45, 185, 62], 'bold': False, 'italic': False},
             {'text': 'Whale shark', 'bbox_2d': [96, 46, 186, 63], 'bold': False, 'italic': False}]
    assert len(line_rows(lines, frame, marks, used=set(), font='Arial')) == 1


def test_neighbouring_pieces_of_one_paragraph_are_read_as_one_region() -> None:
    # A connector crossing a paragraph splits its marks into pieces; read apart, each view cuts words in half.
    left = paragraph(100, 50, 6, 1)
    right = paragraph(100+8*6+22, 50, 6, 1)
    below = paragraph(100, 50+9+16, 7, 1)
    found = regions(left+right+below, (1000, 800))
    assert len(found) == 1
    assert found[0]['marks'] == 19
    x0, y0, x1, y1 = found[0]['bbox_2d']
    assert x0 <= 100 and x1 >= 100+8*6+22+8*5+5 and y1 >= 50+9+16+8


def test_lettering_that_runs_up_the_page_is_left_alone() -> None:
    # The rows written here are horizontal; a rotated label set in an upright box would be squeezed into a column.
    marks = [mark(300, 100+11*i, w=9, h=8) for i in range(12)]
    frame = {'origin': [0, 0], 'scale': 1.}
    lines = [{'text': 'Subtract Baseline', 'bbox_2d': [295, 95, 315, 240], 'bold': False, 'italic': False}]
    assert line_rows(lines, frame, marks, used=set(), font='Arial') == []
