"""Filled closed outlines can be named as foreground of an image asset."""
from full_recovery.foreground_candidates import overlapping

TARGET = [0, 0, 100, 100]


def part(kind, bbox, closed=True):
    x0, y0, x1, y1 = bbox
    commands = [['M', x0, y0], ['L', x1, y0], ['L', x1, y1], ['L', x0, y1]] + ([['Z']] if closed else [])
    return {'kind': kind, 'bbox_2d': bbox, 'commands': commands}


def keys(parts, styles, own=()):
    return [row['key'] for row in overlapping(parts, styles, own, TARGET)]


def test_a_filled_closed_path_is_a_candidate():
    parts = {'body': part('path', [10, 10, 40, 60]), 'bar': part('rectangle', [15, 20, 35, 24])}
    styles = {'body': {'fill': '#DDDDDD'}, 'bar': {'fill': '#000000'}}
    assert keys(parts, styles) == ['body', 'bar']


def test_an_open_path_is_not_a_candidate():
    parts = {'fold': part('path', [10, 10, 40, 60], closed=False)}
    assert keys(parts, {'fold': {'fill': '#DDDDDD'}}) == []


def test_an_unfilled_outline_is_not_a_candidate():
    parts = {'body': part('path', [10, 10, 40, 60])}
    assert keys(parts, {'body': {'fill': None}}) == []


def test_a_backdrop_covering_the_target_is_not_a_candidate():
    parts = {'panel': part('path', [-5, -5, 120, 120]), 'card': part('rectangle', [-5, -5, 120, 120])}
    styles = {'panel': {'fill': '#FFFFFF'}, 'card': {'fill': '#FFFFFF'}}
    assert keys(parts, styles) == []


def test_the_owner_s_own_parts_and_distant_parts_are_excluded():
    parts = {'own': part('rectangle', [10, 10, 40, 60]), 'far': part('rectangle', [300, 300, 340, 360])}
    styles = {'own': {'fill': '#111111'}, 'far': {'fill': '#111111'}}
    assert keys(parts, styles, own=('own',)) == []


def test_each_candidate_reports_its_box():
    parts = {'body': part('path', [10, 10, 40, 60])}
    assert overlapping(parts, {'body': {'fill': '#DDDDDD'}}, (), TARGET) == [{'key': 'body', 'bbox': [10, 10, 40, 60]}]
