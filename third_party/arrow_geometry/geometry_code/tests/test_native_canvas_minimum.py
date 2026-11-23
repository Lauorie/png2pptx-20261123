"""Physical slide bounds must not reject recoverable thin geometry previews."""
import copy
import json
import sys
from pathlib import Path

import pytest
from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'lib')]
import native
import native_scale


@pytest.mark.parametrize('size', [(1080, 81), (1080, 57), (1080, 80), (20, 1080), (1080, 20)])
def test_thin_preview_exports_without_moving_pixel_geometry(size, tmp_path):
    row = dict(id='probe', start=[3, 3], via=[], end=[size[0]-4, size[1]-4])
    style = dict(color='#000000', width_px=2, dash=dict(type='solid', preset='solid'),
                 start=None, end=None, state='geometry_preview')
    original = copy.deepcopy(row)
    path = tmp_path/'thin.pptx'
    native.write(size, [row], [style], path)
    prs = Presentation(path)
    assert row == original
    assert min(prs.slide_width, prs.slide_height) >= 914400
    assert max(prs.slide_width, prs.slide_height) <= 56*914400
    scale = prs.slide_width/size[0]
    shape = prs.slides[0].shapes[0]
    assert abs(shape.left/scale - 3.5) < 0.001
    assert abs(shape.top/scale - 3.5) < 0.001
    assert abs(shape.line.width/scale - 2) < 0.001
    assert prs.slide_width/prs.slide_height == pytest.approx(size[0]/size[1], rel=1e-6)


def test_canvas_and_head_scaling_combine_without_clamping():
    row = dict(id='probe')
    style = dict(width_px=2, start=None, end=dict(type='triangle', width='med', length='med'))
    plan = native_scale.plan((1080, 81), [row], [style])
    assert plan['safe']
    assert min(plan['slide_emu']) >= 914400
    stroke_points = style['width_px'] * plan['slide_emu'][0] / 1080 / 12700
    assert stroke_points >= native_scale.LO_BASELINE_FLOOR_POINTS
    assert stroke_points * 3 >= 4


def test_unrepresentable_aspect_reports_unsafe_without_squashing():
    plan = native_scale.plan((1080, 10), [], [], minimum_head_points=None)
    assert not plan['safe']
    assert plan['required_inches'][0]/plan['required_inches'][1] == pytest.approx(108)
