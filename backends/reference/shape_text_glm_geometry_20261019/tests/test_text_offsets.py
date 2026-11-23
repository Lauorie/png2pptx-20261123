import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('measure_text_offsets',Path(__file__).parents[1]/'measure_text_offsets.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


@pytest.mark.parametrize('dx,dy',[(0,0),(1,-2),(-3,5),(5,0)])
def test_fixed_scale_translation_sign_and_distance(dx,dy):
    rng=np.random.default_rng(23)
    glyph=(rng.random((13,19))>.4).astype(np.float32)
    ink=np.zeros((45,51),np.float32);x,y=15+dx,15+dy
    ink[y:y+13,x:x+19]=glyph
    result=mod.translation(ink,glyph,[15,15],8)
    assert (result['dx_to_source'],result['dy_to_source'])==(dx,dy)
    assert result['distance_px']==pytest.approx((dx*dx+dy*dy)**.5)
    assert result['fixed_scale_dice']>.99


def test_blank_template_is_not_a_perfect_match():
    assert mod.translation(np.zeros((30,30),np.float32),np.zeros((5,5),np.float32),[10,10],5) is None
