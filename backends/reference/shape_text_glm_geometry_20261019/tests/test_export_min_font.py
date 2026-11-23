from copy import deepcopy

import pytest
from pptx import Presentation
from pptx.util import Inches, Pt

from shape_integration.texts.export import write


@pytest.mark.parametrize('requested_pt', [0., .1, .98, 1., 12.])
def test_export_clamps_paragraph_and_run_font_to_one_point(tmp_path, requested_pt):
    scale = round(Inches(13.333333)) / 400
    row = dict(id='T1', kind='text', font='Arial', resolved_font='Arial',
               frame_px=[10, 10, 20, 1], rotation_deg=0,
               font_size_px=requested_pt * 12700 / scale,
               color='#000000', bold=False, italic=False,
               runs=[dict(text='x', script='normal')])
    original = deepcopy(row)
    path = tmp_path / 'text.pptx'

    audit = write([row], [400, 200], path)

    paragraph = Presentation(path).slides[0].shapes[0].text_frame.paragraphs[0]
    expected = max(1., requested_pt)
    assert paragraph.font.size == Pt(expected)
    assert paragraph.runs[0].font.size == Pt(expected)
    assert audit['unit_mapping'][0]['font_size_pt'] == pytest.approx(expected)
    assert row == original
