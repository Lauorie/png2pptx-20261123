from copy import deepcopy
from pathlib import Path

from shape_integration.styling.controller import affected_review_units


def part(box):
    x0,y0,x1,y1=box
    return {'bbox_2d':box,'commands':[['M',x0,y0],['L',x1,y0],['L',x1,y1],['L',x0,y1],['Z']]}


def test_large_compound_context_does_not_repeat_distant_target_reviews():
    table = {key: part(box) for key, box in
             [('changed', [0, 0, 10, 10]), ('overlap', [8, 8, 20, 20]), ('distant', [80, 80, 90, 90])]}
    previous = {key: {'fill': '#FFFFFF', 'width_px': 1} for key in table}
    current = deepcopy(previous)
    current['changed']['fill'] = '#FF0000'
    units = [{'id': 'batch', 'kind': 'object', 'keys': list(table), 'context_keys': list(table), 'root': Path('batch')}]
    result, changed, reordered = affected_review_units(units, table, previous, current, list(table), list(table))
    assert result[0]['review_keys'] == ['changed', 'overlap']
    assert result[0]['context_keys'] == list(table) and 'review_keys' not in units[0]


def test_reordered_overlap_and_shared_groups_preserve_review_dependencies():
    table = {key: part(box) for key, box in
             [('a', [0, 0, 10, 10]), ('b', [5, 5, 15, 15]), ('shared', [90, 90, 100, 100])]}
    styles = {key: {'fill': '#FFFFFF', 'width_px': 1} for key in table}
    units = [{'id': 'style_group', 'kind': 'shared_component_style', 'keys': ['a', 'shared'], 'root': Path('a')},
             {'id': 'b', 'kind': 'object', 'keys': ['b'], 'root': Path('b')}]
    result, changed, reordered = affected_review_units(units, table, styles, styles,
                                                       ['a', 'b', 'shared'], ['b', 'a', 'shared'])
    assert reordered == {'a', 'b'}
    assert result[0]['review_keys'] == ['a', 'shared'] and result[1]['review_keys'] == ['b']


def test_unchanged_style_and_order_need_no_post_edit_review():
    table = {'a': part([0, 0, 10, 10])}
    styles = {'a': {'fill': None, 'width_px': 1}}
    units = [{'id': 'a', 'kind': 'object', 'keys': ['a'], 'root': Path('a')}]
    assert affected_review_units(units, table, styles, styles, ['a'], ['a'])[0] == []
