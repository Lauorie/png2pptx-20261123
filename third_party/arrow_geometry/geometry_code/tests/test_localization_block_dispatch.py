"""Upstream block type bypasses line identity, without changing linework."""
import copy
import json
from unittest.mock import patch
import numpy as np
import pytest
from PIL import Image
import bbox_geometry as bg
from test_bbox_pipeline import BASE, Model, obj, locks
from test_block_shape_review import Scripted, geometry


def target():
    return dict(obj(), render_type='block_arrow', geometry_route='closed_shape')


def test_localized_block_goes_directly_to_shape_and_review(tmp_path):
    model=Scripted([geometry(), {'decision':'accept'}])
    cfg={**BASE, 'block_shape_review_enabled':True, 'block_shape_max_refinements':1}
    with patch('bbox_geometry.critic.recover', side_effect=AssertionError('No line critic for a block')):
        result=bg.recover_target(Image.new('RGB',(100,80),'white'), target(), tmp_path, model, cfg, locks())
    assert len(model.calls)==2
    assert model.calls[0][0]==tmp_path/'block_geometry/request'
    assert result['rows'][0]['kind']=='block_arrow'
    assert result['rows'][0]['shape_parts']
    assert result['rows'][0]['shape_geometry_review']['accepted_by_model']
    assert result['rows'][0]['geometry_request']==str(tmp_path/'block_geometry/request')
    assert not (tmp_path/'geometry').exists()
    assert not (tmp_path/'initial.json').exists()  # no synthetic model response
    assert json.loads((tmp_path/'dispatch.json').read_text())['line_geometry_requested'] is False


def test_localized_block_keeps_sam_fallback(tmp_path):
    def mask(crop,*args):
        m=np.zeros((crop.height,crop.width),bool);m[20:40,30:80]=True
        return m, []
    with patch('block_shapes.recover', side_effect=ValueError('invalid shape')), patch('bbox_geometry.block_mask', side_effect=mask):
        result=bg.recover_target(Image.new('RGB',(100,80),'white'), target(), tmp_path, Model('forbidden'), BASE, locks())
    row=result['rows'][0]
    assert row['kind']=='block_arrow' and row['mask_shape_parts']
    assert row['coordinate_source']=='model_selected_mask_shape'


def test_failed_block_does_not_fall_back_to_line_or_bbox(tmp_path):
    with patch('block_shapes.recover', side_effect=ValueError('invalid shape')), patch('bbox_geometry.block_mask', return_value=(None,[])):
        with pytest.raises(ValueError, match='invalid shape'):
            bg.recover_target(Image.new('RGB',(100,80),'white'),target(),tmp_path,Model('forbidden'),BASE,locks())
    assert not (tmp_path/'geometry').exists()


def test_linework_keeps_existing_request_and_geometry(tmp_path):
    model=Model('expand');source=Image.new('RGB',(100,80),'white')
    with patch('block_shapes.recover', side_effect=AssertionError('Not a shape')):
        result=bg.recover_target(source,dict(obj(),render_type='linework'),tmp_path,model,BASE,locks())
    assert len(model.calls)==2 and '/geometry/F002/request' in model.calls[-1][0]
    row=result['rows'][0]
    assert row['kind']=='straight' and 'shape_parts' not in row
    assert np.allclose(row['start'],[20,40],atol=.01)
    assert np.allclose(row['end'],[80,40],atol=.01)
    assert row['style']['end']['length']=='med'
    assert not (tmp_path/'dispatch.json').exists()


def test_line_model_cannot_reenter_shape_pipeline(tmp_path):
    with patch('block_shapes.recover', side_effect=AssertionError('Must not route linework to shape')) as shape:
        with pytest.raises(ValueError, match='linework request'):
            bg.recover_target(Image.new('RGB',(100,80),'white'),dict(obj(),render_type='linework'),
                              tmp_path,Model('block'),BASE,locks())
    shape.assert_not_called()


def test_linework_contract_allows_visible_connector_loops():
    import geometry_critic as gc
    doc=dict(nodes=[['a',10,10],['b',70,10],['c',70,50],['d',10,50]],
        routes=[dict(id='R1',paths=[dict(id='loop',heads=['none','none'],segments=[
            dict(kind='line',nodes=[a,b]) for a,b in [('a','b'),('b','c'),('c','d'),('d','a')]])])])
    assert bg.validate(dict(doc,status='draw'),[100,80])['routes']==doc['routes']
    assert gc.validate(dict(decision='refine',reason='visible connector loop',geometry=doc),
                       [100,80],require_heads=True)['geometry']==doc
