"""Composition uses restyled arrow products when they can be made, the originals otherwise."""
import json
from pathlib import Path

from PIL import Image, ImageDraw

from full_recovery.arrow_restyle import restyled
from test_arrow_restyle_stage import ROOT, _row


def _inputs(tmp_path: Path, size: int = 400) -> dict:
    image = Image.new('RGB', (400, 200), 'white')
    ImageDraw.Draw(image).line((40, 60, 360, 60), fill='#C0504D', width=4)
    image.save(tmp_path / 'source.png')
    (tmp_path / 'scene.json').write_text(json.dumps({'size': [size, 200], 'arrows': [_row('R1', 60.)]}))
    return {'arrow_scene': tmp_path / 'scene.json', 'arrow_pptx': tmp_path / 'arrows.pptx'}


def _settings() -> dict:
    return {'fresh_arrow_root': str(ROOT / 'third_party/arrow_geometry')}


def test_restyled_products_replace_the_branch_products(tmp_path: Path) -> None:
    result = restyled(_inputs(tmp_path), tmp_path / 'source.png', tmp_path / 'out', _settings(), ROOT)
    assert set(result) == {'arrow_scene', 'arrow_pptx'} and result['arrow_pptx'].is_file()
    assert json.loads(result['arrow_scene'].read_text())['arrows'][0]['style']['state'] == 'pixel_measured'


def test_the_branch_files_are_not_touched(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    before = paths['arrow_scene'].read_bytes()
    restyled(paths, tmp_path / 'source.png', tmp_path / 'out', _settings(), ROOT)
    assert paths['arrow_scene'].read_bytes() == before


def test_without_the_arrow_writer_the_branch_products_stay(tmp_path: Path) -> None:
    assert restyled(_inputs(tmp_path), tmp_path / 'source.png', tmp_path / 'out', {}, ROOT) == {}


def test_a_failed_restyle_keeps_the_branch_products(tmp_path: Path) -> None:
    paths = _inputs(tmp_path, size=999)
    assert restyled(paths, tmp_path / 'source.png', tmp_path / 'out', _settings(), ROOT) == {}


def test_it_can_be_switched_off(tmp_path: Path) -> None:
    settings = {**_settings(), 'arrow_stroke_from_pixels': False}
    assert restyled(_inputs(tmp_path), tmp_path / 'source.png', tmp_path / 'out', settings, ROOT) == {}
