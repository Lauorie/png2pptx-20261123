import copy,json,threading
from pathlib import Path
from unittest.mock import patch
import pytest
import numpy as np
from PIL import Image
import block_shapes as shapes
import block_shape_review as review
from block_shape_protocol import make_prompt,observation_schema
from shape_core.fitting.contracts import normalize_observation
from shape_core.fitting.adapter import fit_part

def geometry():
    points=[{'id':n,'xy':xy} for n,xy in [('a',[10,10]),('b',[70,10]),('c',[70,50]),('d',[10,50])]]
    return dict(status='ok',points=points,parts=[dict(id='body',kind='path',closed=True,
        segments=[dict(kind='line',point_ids=[a,b]) for a,b in [('a','b'),('b','c'),('c','d'),('d','a')]])])

def row():
    obs=geometry();table={p['id']:p['xy'] for p in obs['points']}
    parts=[dict(id=p['id'],**fit_part(p,table)) for p in obs['parts']]
    base=dict(id='R',kind='block_arrow',target_bbox=[8,8,72,52],identity_locator='block arrow',mask_shape_parts=copy.deepcopy(parts))
    return shapes.use_parts(base,parts,'glm_shape_geometry')

class Scripted:
    def __init__(self,answers):self.answers=iter(answers);self.calls=[]
    def ask(self,path,prompt,images,validator):
        self.calls.append((Path(path),prompt,images));answer=next(self.answers)
        if isinstance(answer,Exception):raise answer
        return validator(copy.deepcopy(answer))

def execute(tmp,answers,base=None):
    model=Scripted(answers);r=row() if base is None else base
    result,issues=review.recover(Image.new('RGB',(100,80),'#c0ddcc'),r,None,tmp,model,
        {'block_shape_max_refinements':1},{'cpu':threading.Semaphore(1)})
    return result,issues,model

def test_valid_points_are_unchanged_by_protocol():
    raw=geometry();before=copy.deepcopy(raw);q=make_prompt([100,80],[8,8,72,52],row())
    assert 'NOT Bezier' in q['sampling']['path']
    normalized=normalize_observation(raw,'R',[100,80])
    assert normalized['points']==raw['points'] and raw==before
    assert '3-5' in q['sampling']['path']

def test_accept_preserves_geometry_and_background(tmp_path):
    r=row();got,issues,model=execute(tmp_path,[{'decision':'accept'}],r)
    assert got['shape_parts']==r['shape_parts'] and not issues
    assert got['shape_geometry_review']['accepted_by_model']
    im=Image.open(tmp_path/'check_00/input/overlay.png')
    assert im.getpixel((99,79))==(192,221,204)

def test_semantic_review_does_not_compare_or_select_mask(tmp_path):
    r=row();r['mask_shape_parts'][0]['commands'][1][1]=64
    original=copy.deepcopy(r)
    got,issues,model=execute(tmp_path,[{'decision':'accept'}],r)
    assert got['shape_parts']==original['shape_parts'] and not issues
    assert got['coordinate_source']=='glm_shape_geometry' and r==original
    assert len(model.calls[0][2])==2
    assert not (tmp_path/'mask_candidate').exists()
    with pytest.raises(ValueError):review.validate({'decision':'use_mask','reason':'x'},True)

def test_refine_reobserves_with_same_protocol_and_rechecks(tmp_path):
    got,issues,model=execute(tmp_path,[{'decision':'refine','reason':'The arrow points the wrong way.'},geometry(),{'decision':'accept'}])
    assert len(model.calls)==3 and got['shape_geometry_review']['refinements']==1 and not issues
    q=model.calls[1][1]
    assert q['sampling']==make_prompt([100,80],[8,8,72,52],row())['sampling']
    assert q['revision_feedback']==['The arrow points the wrong way.']
    assert 'refine' not in model.calls[2][1]['output_schema']['decision'].split('|')

def test_review_failure_keeps_last_valid_shape(tmp_path):
    r=row();got,issues,model=execute(tmp_path,[RuntimeError('timeout')],r)
    assert got['shape_parts']==r['shape_parts'] and issues[0]['retained']=='last_valid_geometry'
    assert not got['shape_geometry_review']['accepted_by_model']

def test_failed_refinement_retains_initial_geometry(tmp_path):
    got,issues,model=execute(tmp_path,[{'decision':'refine','reason':'Wrong upper curve.'},RuntimeError('transport failure')])
    assert got['shape_parts']==row()['shape_parts'] and issues
    assert not got['shape_geometry_review']['accepted_by_model']

def test_no_fake_mask_or_extra_refinement():
    with pytest.raises(ValueError):review.validate({'decision':'use_mask','reason':'x'},True)
    with pytest.raises(ValueError):review.validate({'decision':'refine','reason':'x'},False)
    with pytest.raises(ValueError):review.validate({'decision':'accept'},True,False)

def test_bbox_route_reaches_shape_review(tmp_path):
    import bbox_geometry as bg
    from test_bbox_pipeline import BASE,Model,obj,locks
    cfg={**BASE,'block_shape_review_enabled':True,'block_shape_max_refinements':1}
    with patch('block_shapes.recover',return_value=row()),patch('block_shape_review.recover',return_value=(row(),[])) as check:
        result=bg.recover_target(Image.new('RGB',(100,80),'white'),dict(obj(),render_type='block_arrow'),tmp_path,Model('forbidden'),cfg,locks())
    check.assert_called_once();assert result['rows'][0]['kind']=='block_arrow'
