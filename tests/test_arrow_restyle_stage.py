"""The restyle stage rewrites arrow products with styles read from the source figure."""
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = ROOT / 'third_party/arrow_geometry/geometry_code'
STYLE = {'color': '#666666', 'width_px': 2.0, 'dash': {'type': 'solid', 'preset': 'solid'}, 'start': None,
         'end': {'type': 'triangle', 'width': 'med', 'length': 'med'}, 'state': 'geometry_preview'}


def _row(ident: str, y: float) -> dict:
    return {'id': ident, 'parent_id': ident, 'label': 'A1', 'heads': ['none', 'triangle'], 'kind': 'straight',
            'start': [40., y], 'via': [], 'end': [360., y], 'beziers': [], 'geometry_heads_reported': True,
            'start_head': False, 'end_head': True, 'coordinate_source': 'glm_segmented_points', 'style': dict(STYLE)}


def _run(tmp_path: Path) -> Path:
    image = Image.new('RGB', (400, 200), 'white')
    draw = ImageDraw.Draw(image)
    draw.line((40, 60, 360, 60), fill='#C0504D', width=4)
    for x in range(40, 360, 16):
        draw.line((x, 140, x + 9, 140), fill='#1F5FBF', width=2)
    image.save(tmp_path / 'source.png')
    (tmp_path / 'scene.json').write_text(json.dumps({'size': [400, 200], 'arrows': [_row('R1', 60.), _row('R2', 140.)]}))
    done = subprocess.run([sys.executable, '-B', '-m', 'full_recovery.arrow_restyle_stage', '--geometry-code', str(GEOMETRY),
                           '--scene', str(tmp_path / 'scene.json'), '--source', str(tmp_path / 'source.png'),
                           '--output', str(tmp_path / 'out')], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    return tmp_path / 'out'


def test_the_scene_carries_measured_styles(tmp_path: Path) -> None:
    rows = json.loads((_run(tmp_path) / 'scene.json').read_text())['arrows']
    assert rows[0]['style']['color'].startswith('#C') and abs(rows[0]['style']['width_px'] - 4) < 0.8
    assert rows[1]['style']['dash']['type'] == 'dashed'
    assert rows[0]['start'] == [40., 60.] and rows[0]['style']['end'] == STYLE['end']


def test_the_pptx_is_drawn_with_those_styles(tmp_path: Path) -> None:
    out = _run(tmp_path)
    colour = json.loads((out / 'scene.json').read_text())['arrows'][0]['style']['color'][1:]
    with ZipFile(out / 'arrows.pptx') as archive:
        slide = archive.read('ppt/slides/slide1.xml').decode()
    assert colour in slide.upper() and 'custDash' in slide and '666666' not in slide
