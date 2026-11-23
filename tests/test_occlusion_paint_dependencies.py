"""Missing paint and layer order are reviewed together without role labels."""
from shapely.geometry import LineString, box

from shape_integration.styling.occlusion import paint_dependencies


def finding(**relations):
    return {'verdict': 'revise', 'affected_fields': ['occlusion'],
            'source_evidence': 'The named foreground face hides this rear edge in CLEAN.', **relations}


def test_unfilled_front_with_crossing_rear_edge_needs_source_paint_review():
    geoms = {'rear': box(0, 0, 30, 30), 'front': box(10, 10, 40, 40)}
    styles = {'rear': {'fill': '#FFFFFF'}, 'front': {'fill': None}}
    result = paint_dependencies({'front': [finding(_front_parts=['rear'])]}, styles, geoms, .2)
    assert [(row['back'], row['front']) for row in result] == [('rear', 'front')]
    assert 'fill' not in result[0] and 'fill_opacity' not in result[0]
    # A finding about a rear object must reach the foreground paint owner too.
    assert paint_dependencies({'rear': [finding(_behind_parts=['front'])]}, styles, geoms, .2) == result


def test_existing_opaque_paint_and_nonintersecting_or_open_front_need_no_paint_review():
    geoms = {'rear': box(0, 0, 30, 30), 'opaque': box(10, 10, 40, 40),
             'distant': box(50, 50, 80, 80), 'open': LineString([(0, 15), (40, 15)])}
    styles = {key: {'fill': '#FFFFFF' if key == 'opaque' else None} for key in geoms}
    feedback = {'rear': [finding(_behind_parts=['opaque', 'distant', 'open'])]}
    assert paint_dependencies(feedback, styles, geoms, .2) == []
