import pytest
from PIL import Image, ImageDraw
from shapely.geometry import box, MultiLineString

from shape_integration.styling.occlusion import order_internal_by_paint


def fixture(fill='#4EA72E', ink='#000000', visible=False, circle_only=False):
    image = Image.new('RGB', (100, 100), 'white')
    draw = ImageDraw.Draw(image)
    draw.line((45, 20, 45, 85), fill=ink, width=2)
    draw.rectangle((10, 10, 80, 60), fill=fill)
    if visible:
        draw.line((25, 35, 65, 35), fill=ink, width=2)
    if circle_only:
        draw.ellipse((25, 15, 65, 55), outline=ink, width=2)
    geometry = {'face': box(10, 10, 80, 60), 'stroke': MultiLineString([
        [(45, 20), (45, 85)], [(25, 35), (65, 35)]])}
    styles = {'face': {'fill': fill, 'stroke': None},
              'stroke': {'fill': None, 'stroke': ink, 'width_px': 2.}}
    scene = {'objects': {'object': {'part_keys': ['face', 'stroke']}}}
    return image, geometry, styles, scene


@pytest.mark.parametrize('fill,ink', [('#4EA72E', '#000000'), ('#155165', '#FFFFFF')])
def test_visible_interior_stroke_is_not_hidden_by_an_endpoint_guess(fill, ink):
    image, geometry, styles, scene = fixture(fill, ink, visible=True)
    result = order_internal_by_paint(scene, {k: {} for k in geometry},
        {'object_order': ['object']}, styles, geometry, source=image)
    assert result['part_orders']['object'] == ['face', 'stroke']
    assert any(row['reason'] == 'source_visible_stroke_inside_face'
               for row in result['internal_rejected_heuristics'])


@pytest.mark.parametrize('circle_only', [False, True])
def test_hidden_interior_or_a_few_circle_crossings_do_not_invent_visible_strokes(circle_only):
    image, geometry, styles, scene = fixture(circle_only=circle_only)
    result = order_internal_by_paint(scene, {k: {} for k in geometry},
        {'object_order': ['object']}, styles, geometry, source=image)
    assert result['part_orders']['object'] == ['stroke', 'face']


def test_source_guard_does_not_replace_an_explicit_model_order():
    image, geometry, styles, scene = fixture(visible=True)
    result = order_internal_by_paint(scene, {k: {} for k in geometry},
        {'object_order': ['object'], 'internal_model_edges': [
            {'back': 'stroke', 'front': 'face', 'origin': 'model_selection'}]},
        styles, geometry, source=image)
    assert result['part_orders']['object'] == ['stroke', 'face']


@pytest.mark.parametrize('outline,width,expected', [
    ('#000000', 1., ['face', 'detail']),
    ('#000000', .2, ['detail', 'face']),
    (None, 1., ['detail', 'face']),
])
def test_detail_with_rounded_endpoints_uses_only_the_visible_face_outline(outline, width, expected):
    geometry = {'face': box(10, 10, 80, 60), 'detail': MultiLineString([
        [(9.7, 25), (80.3, 25)], [(30, 9.7), (30, 60.3)]])}
    styles = {'face': {'fill': '#90C090', 'stroke': outline, 'width_px': width},
              'detail': {'fill': None, 'stroke': '#000000', 'width_px': 1.}}
    scene = {'objects': {'object': {'part_keys': ['detail', 'face']}}}
    result = order_internal_by_paint(scene, {k: {} for k in geometry},
        {'object_order': ['object']}, styles, geometry)
    assert result['part_orders']['object'] == expected
