"""The text scoring script walks a run's figures and reports where slide and source text differ."""
import importlib.util
import json
from pathlib import Path
from typing import List

from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'score_text.py'


def _script():
    spec = importlib.util.spec_from_file_location('score_text', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(tmp_path: Path) -> Path:
    run = tmp_path / 'job'
    images = tmp_path / 'job.input' / 'images'
    images.mkdir(parents=True)
    for case, shade in (('good', 10), ('wrong', 20), ('lost', 30)):
        Image.new('RGB', (200, 100), (shade, 0, 0)).save(images / f'{case}.png')
    for case, shade in (('good', 11), ('wrong', 21)):
        (run / 'delivery' / case).mkdir(parents=True)
        Image.new('RGB', (400, 200), (shade, 0, 0)).save(run / 'delivery' / case / 'preview.png')
    return run


def _reader(image: Image.Image) -> List[str]:
    """Stand in for the OCR engine: the red channel says which picture this is."""
    text = {10: 'Encoder', 11: 'Encoder', 20: 'Val.Error', 21: 'Valve: Junior'}[image.getpixel((0, 0))[0]]
    return [text]


def test_every_figure_is_scored_and_a_figure_without_slide_is_marked(tmp_path: Path) -> None:
    result = _script().score_run(_run(tmp_path), _reader)
    assert result['good'].overall == 1.0
    assert result['wrong'].overall < 0.9
    assert [(d.source, d.slide) for d in result['wrong'].differences] == [('Val.Error', ''), ('', 'Valve:'), ('', 'Junior')]
    assert result['lost'] is None


def test_the_json_report_lists_the_differences(tmp_path: Path) -> None:
    script = _script()
    table = {'job': script.score_run(_run(tmp_path), _reader)}
    out = tmp_path / 'text.json'
    script.save(table, out)
    saved = json.loads(out.read_text())
    assert saved['job']['lost'] is None
    assert saved['job']['wrong']['differences'][1]['slide'] == 'Valve:'
    assert saved['job']['good']['recall'] == 1.0
