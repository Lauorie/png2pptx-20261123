"""Arrow rows take their stroke style from the source pixels when it can be read."""
import numpy as np
from PIL import Image, ImageDraw

from full_recovery.arrow_stroke_style import restyle

PLACEHOLDER = {'color': '#666666', 'width_px': 2.0, 'dash': {'type': 'solid', 'preset': 'solid'},
               'start': None, 'end': {'type': 'triangle', 'width': 'med', 'length': 'med'}, 'state': 'geometry_preview'}


def _row(kind: str = 'straight') -> dict:
    return {'id': 'R1/p1', 'kind': kind, 'start': [40, 100], 'via': [], 'end': [360, 100], 'style': dict(PLACEHOLDER)}


def _figure(dashed: bool = False) -> np.ndarray:
    image = Image.new('RGB', (400, 200), '#EAF3E6')
    draw = ImageDraw.Draw(image)
    for x in range(40, 360, 16 if dashed else 400):
        draw.line((x, 100, min(360, x + (9 if dashed else 400)), 100), fill='#C0504D', width=3)
    return np.asarray(image)


def test_a_measured_row_gets_the_source_colour_and_width_and_keeps_its_heads() -> None:
    style = restyle(_figure(), _row())['style']
    assert style['color'].startswith('#C') and abs(style['width_px'] - 3) < 0.8
    assert style['dash'] == {'type': 'solid', 'preset': 'solid'} and style['end'] == PLACEHOLDER['end']
    assert style['state'] == 'pixel_measured'


def test_a_dashed_row_carries_its_rhythm_in_the_export_format() -> None:
    dash = restyle(_figure(dashed=True), _row())['style']['dash']
    assert dash['type'] == 'dashed' and dash['preset'] == 'dash' and len(dash['pattern_px']) == 2
    assert dash['mark_px'] == dash['pattern_px'][0] and dash['gap_px'] == dash['pattern_px'][1]


def test_the_input_row_is_not_changed() -> None:
    row = _row()
    restyle(_figure(), row)
    assert row['style'] == PLACEHOLDER


def test_an_unreadable_row_is_returned_as_it_was() -> None:
    blank = np.full((200, 400, 3), 255, dtype=np.uint8)
    assert restyle(blank, _row()) == _row()


def test_a_block_arrow_is_left_alone() -> None:
    assert restyle(_figure(), _row('block_arrow')) == _row('block_arrow')


def _scene() -> np.ndarray:
    image = Image.new('RGB', (400, 200), 'white')
    draw = ImageDraw.Draw(image)
    draw.line((40, 50, 360, 50), fill='#C0504D', width=4)      # long red
    draw.line((40, 90, 360, 90), fill='#1F5FBF', width=2)      # long blue
    draw.line((40, 130, 360, 130), fill='#1F5FBF', width=2)    # long blue
    draw.line((100, 160, 109, 160), fill='#C0504D', width=4)   # short red
    draw.line((200, 160, 209, 160), fill='#2E8B57', width=2)   # short green, no peer of that colour
    return np.asarray(image)


def _line(ident: str, x0: float, x1: float, y: float) -> dict:
    return {'id': ident, 'kind': 'straight', 'start': [x0, y], 'via': [], 'end': [x1, y], 'style': dict(PLACEHOLDER)}


def _restyled() -> dict:
    from full_recovery.arrow_stroke_style import restyle_all
    rows = [_line('red', 40, 360, 50), _line('blue1', 40, 360, 90), _line('blue2', 40, 360, 130),
            _line('short_red', 100, 109, 160), _line('short_green', 200, 209, 160), _line('nothing', 300, 309, 180)]
    return {row['id']: row['style'] for row in restyle_all(_scene(), rows)}


def test_a_short_arrow_takes_its_colour_from_pixels_and_its_width_from_a_like_coloured_peer() -> None:
    style = _restyled()['short_red']
    assert style['color'].startswith('#C') and abs(style['width_px'] - 4) < 0.8
    assert style['state'] == 'pixel_colour_peer_width'


def test_a_short_arrow_without_a_like_coloured_peer_takes_the_usual_width_of_the_figure() -> None:
    style = _restyled()['short_green']
    assert style['color'] != PLACEHOLDER['color'] and abs(style['width_px'] - 2) < 0.6


def test_an_arrow_with_no_ink_keeps_its_style() -> None:
    assert _restyled()['nothing'] == PLACEHOLDER
