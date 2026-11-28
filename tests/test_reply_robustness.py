"""A reply the model gets slightly wrong must not cost the whole figure."""
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'vendor'), str(ROOT / 'experiment/shape_sam_guided')]

import pytest
from pptx import Presentation
from pptx.util import Emu

from shape_integration.reply_ids import distinct_ids, with_distinct_ids
from shape_integration.styling.controller import commit_refinement
from shape_integration.styling.export import apply_style


def _obj(ident: str, box: list) -> dict:
    return {'id': ident, 'label': ident, 'kind': 'rectangle', 'bbox_2d': box}


def test_repeated_elements_given_one_name_are_told_apart() -> None:
    raw = {'objects': [_obj('token', [0, 0, 5, 5]), _obj('token', [10, 0, 15, 5]), _obj('token', [20, 0, 25, 5])]}
    ids = [o['id'] for o in distinct_ids(raw, 'objects')['objects']]
    assert len(set(ids)) == 3 and ids[0] == 'token'


def test_new_names_do_not_collide_with_names_the_reply_already_uses() -> None:
    raw = {'objects': [_obj('token', [0, 0, 5, 5]), _obj('token_2', [30, 0, 35, 5]), _obj('token', [10, 0, 15, 5])]}
    ids = [o['id'] for o in distinct_ids(raw, 'objects')['objects']]
    assert len(set(ids)) == 3 and ids[:2] == ['token', 'token_2']


def test_the_same_element_listed_twice_is_listed_once() -> None:
    raw = {'objects': [_obj('token', [0, 0, 5, 5]), _obj('token', [0, 0, 5, 5])]}
    assert distinct_ids(raw, 'objects')['objects'] == [_obj('token', [0, 0, 5, 5])]


def test_a_reply_with_distinct_names_is_returned_as_it_came() -> None:
    raw = {'objects': [_obj('a', [0, 0, 5, 5]), _obj('b', [10, 0, 15, 5])]}
    before = copy.deepcopy(raw)
    assert distinct_ids(raw, 'objects') == before and raw == before


def test_a_malformed_reply_is_left_for_the_validator_to_reject() -> None:
    for raw in ({'objects': 'none'}, {'other': []}, [], {'objects': [{'label': 'no id'}, 3]}):
        assert distinct_ids(raw, 'objects') == raw


def test_validator_sees_the_repaired_reply() -> None:
    def strict(raw: dict) -> dict:
        ids = [o['id'] for o in raw['objects']]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate object/text ID')
        return raw
    raw = {'objects': [_obj('token', [0, 0, 5, 5]), _obj('token', [10, 0, 15, 5])]}
    with pytest.raises(ValueError):
        strict(raw)
    assert len(with_distinct_ids(strict, 'objects')(raw)['objects']) == 2


def _commit(old: dict, new: dict, fields: list) -> dict:
    checks = {'k': {'value': {'verdict': 'revise', 'affected_fields': fields}}}
    return commit_refinement({'k': old}, {'k': new}, checks)[0]['k']


NO_OUTLINE = {'fill': '#72B27A', 'stroke': None, 'width_px': 0.0, 'dash': 'solid'}
DASHED = {'fill': None, 'stroke': '#74A26D', 'width_px': 2.4, 'dash': 'dash'}


def test_a_new_outline_keeps_its_width_when_only_the_width_is_disputed() -> None:
    settled = _commit(NO_OUTLINE, DASHED, ['stroke_width'])
    assert settled['stroke'] == '#74A26D' and settled['width_px'] == 2.4


def test_a_new_outline_keeps_its_colour_and_pattern_when_they_are_disputed() -> None:
    settled = _commit(NO_OUTLINE, DASHED, ['stroke_color', 'line_pattern', 'stroke_width'])
    assert (settled['stroke'], settled['width_px'], settled['dash']) == ('#74A26D', 2.4, 'dash')


def test_a_removed_outline_stays_removed_when_only_its_colour_is_disputed() -> None:
    old = {'fill': None, 'stroke': '#111111', 'width_px': 2.0, 'dash': 'solid'}
    settled = _commit(old, NO_OUTLINE, ['stroke_color'])
    assert settled['stroke'] is None and settled['width_px'] == 0.0


def test_a_disputed_outline_presence_still_rolls_the_whole_outline_back() -> None:
    settled = _commit(NO_OUTLINE, DASHED, ['stroke_presence'])
    assert settled['stroke'] is None and settled['width_px'] == 0.0 and settled['dash'] == 'solid'


def test_a_disputed_width_of_an_existing_outline_still_rolls_back() -> None:
    old = {'fill': None, 'stroke': '#111111', 'width_px': 1.0, 'dash': 'solid'}
    new = {'fill': None, 'stroke': '#111111', 'width_px': 3.0, 'dash': 'solid'}
    assert _commit(old, new, ['stroke_width'])['width_px'] == 1.0


def test_writing_a_dashed_outline_of_no_width_does_not_fail() -> None:
    slide = Presentation().slides.add_slide(Presentation().slide_layouts[6])
    shape = slide.shapes.add_shape(1, Emu(0), Emu(0), Emu(100000), Emu(100000))
    apply_style(shape, {}, 9525, {'fill': None, 'stroke': '#B66A68', 'width_px': 0.0, 'dash': 'dash'})
    line = shape._element.spPr.find('{http://schemas.openxmlformats.org/drawingml/2006/main}ln')
    assert line is not None and line.find('{http://schemas.openxmlformats.org/drawingml/2006/main}noFill') is not None
