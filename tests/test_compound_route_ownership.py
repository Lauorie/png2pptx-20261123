"""A complete stroked glyph remains intact across candidate selection/export."""
import numpy as np
import pytest

from full_recovery.quality_choice import candidates, nominate, apply_choices


@pytest.mark.parametrize('angle,scale', [(0, 1), (90, 1), (35, 2.5), (180, .6)])
def test_short_stroked_head_is_not_competing_with_its_own_shaft(angle, scale):
    theta = np.radians(angle)
    matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) * scale
    points = np.array([[0, 0], [0, 18], [-3, 12], [0, 18], [3, 12]]) @ matrix.T + [60, 70]
    commands = [[op, *xy] for op, xy in zip(['M', 'L', 'L', 'M', 'L'], points)]
    part = {'id': 'glyph', 'object_id': 'owner', 'status': 'ok', 'kind': 'path',
            'commands': commands, 'bbox_2d': [*points.min(0), *points.max(0)]}
    style = {'stroke': '#000000', 'fill': None, 'width_px': scale}
    scene = {'parts': {'glyph': part}}
    rows = candidates(scene, {'glyph': style}, [])
    assert len(rows) == 1
    assert rows[0]['id'] == 'shape:glyph'
    assert not nominate(rows)
    assert rows[0]['refs'] == ['owner/glyph']
    assert not apply_choices(rows, [])['shape_subpath_suppression']
    arrow = {'id': 'recovery', 'start': points[0].tolist(), 'end': points[1].tolist(),
             'heads': ['none', 'open'], 'style': {'color': '#000000', 'width_px': scale}}
    compared = candidates(scene, {'glyph': style}, [arrow])
    assert len(nominate(compared)) == 1
    plan = apply_choices(compared, [{'candidate_ids': [r['id'] for r in compared],
                                      'keep_ids': ['arrow:recovery']}])
    assert plan['drop_shape_keys'] == ['glyph']
    assert not plan['shape_subpath_suppression']


def test_connected_independent_routes_do_not_become_one_arrow():
    part = {'id': 'routes', 'object_id': 'owner', 'status': 'ok', 'kind': 'path',
            'commands': [['M', 10, 20], ['L', 110, 40], ['M', 10, 20], ['L', 110, 90]],
            'bbox_2d': [10, 20, 110, 90]}
    style = {'stroke': '#000000', 'fill': None, 'width_px': 1}
    rows = candidates({'parts': {'routes': part}}, {'routes': style}, [])
    assert {r['id'] for r in rows} == {'shape:routes::subpath0', 'shape:routes::subpath1'}


@pytest.mark.parametrize('reverse_order', [False, True])
def test_separate_v_head_can_be_longer_than_its_short_shaft(reverse_order):
    body = [['M', 10, 50], ['L', 28, 50]]
    head = [['M', 20, 44], ['L', 28, 50], ['L', 20, 56]]
    part = {'id': 'glyph', 'object_id': 'owner', 'status': 'ok', 'kind': 'path',
            'commands': head+body if reverse_order else body+head, 'bbox_2d': [10, 44, 28, 56]}
    rows = candidates({'parts': {'glyph': part}}, {'glyph': {'stroke': '#000000', 'fill': None, 'width_px': 1}}, [])
    assert len(rows) == 1 and rows[0]['id'] == 'shape:glyph'
    assert np.allclose(rows[0]['points'][[0, -1]], [[10, 50], [28, 50]])


@pytest.mark.parametrize('prefix', [
    [['M', 10, 70], ['L', 90, 70], ['L', 90, 20], ['L', 50, 20]],
    [['M', 10, 70], ['C', 100, 80, 100, 20, 50, 20]],
])
def test_bent_and_curved_stroked_heads_leave_no_orphan_when_the_branch_loses(prefix):
    commands = prefix + [['L', 57, 16], ['M', 50, 20], ['L', 57, 24]]
    part = {'id': 'route', 'object_id': 'owner', 'status': 'ok', 'kind': 'path',
            'commands': commands, 'bbox_2d': [10, 16, 100, 80]}
    style = {'stroke': '#000000', 'fill': None, 'width_px': 1}
    rows = candidates({'parts': {'route': part}}, {'route': style}, [])
    assert len(rows) == 1 and rows[0]['id'] == 'shape:route'
    assert rows[0]['length'] > np.linalg.norm(rows[0]['points'][-1]-rows[0]['points'][0])
    plan = apply_choices(rows, [{'candidate_ids': ['shape:route'], 'keep_ids': []}])
    assert plan['drop_shape_keys'] == ['route'] and not plan['shape_subpath_suppression']
    assert commands == part['commands']
