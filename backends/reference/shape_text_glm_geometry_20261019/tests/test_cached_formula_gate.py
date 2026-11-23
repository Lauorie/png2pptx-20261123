import importlib.util
from pathlib import Path
import pytest
from shape_integration.common import save

spec=importlib.util.spec_from_file_location('cached_gate',Path(__file__).resolve().parents[1]/'regate_cached_formula_run.py')
cache=importlib.util.module_from_spec(spec);spec.loader.exec_module(cache)


def test_only_exact_rendered_combination_can_be_reused(tmp_path):
    folder=tmp_path/'cpu_guard/round_01'
    rows=[{'id':'A','font_size_px':20,'background_selection':{'selected':'transparent'}}]
    save(folder/'scene.json',{'texts':rows});save(folder/'scores.json',{'A':{}})
    for name in ('text.pptx','native.png','native.pdf'):(folder/name).write_bytes(b'fixture')
    assert cache.find_exact_render(tmp_path,[{'id':'A','font_size_px':20}],{'A'})==folder
    with pytest.raises(ValueError,match='no exact saved render'):
        cache.find_exact_render(tmp_path,[{'id':'A','font_size_px':19}],{'A'})
    with pytest.raises(ValueError,match='no exact saved render'):
        cache.find_exact_render(tmp_path,rows,set())
