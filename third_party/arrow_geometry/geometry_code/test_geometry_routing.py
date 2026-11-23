"""Routing preserves fast-path geometry and model-owned replacement topology."""
import copy,json,threading
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from PIL import Image
import geometry_routing as routing,mask_vectors,semantic_geometry,segmented_geometry
import head_semantics


def route(rid='a',mode='mask',nodes=None,**kw):
    return dict(id=rid,sources=['G1'],masks=['M1'],kind='straight',proposal=None,
        nodes=['a','b'] if nodes is None else nodes,geometry=mode,heads=['none','triangle'],line_style='solid',**kw)


def fixture(tmp_path):
    image=Image.new('RGB',(100,100),'white');folder=tmp_path/'input';folder.mkdir(parents=True)
    image.save(folder/'clean.png');image.save(folder/'context.png')
    m=np.zeros((100,100),bool);m[19:22,10:81]=True
    Image.fromarray(m.astype('uint8')*255).save(folder/'M1.png')
    return image,[dict(id='M1',mask=m,path=str(folder/'M1.png'))],dict(cpu=threading.Semaphore(1))


class Model:
    def __init__(self,answer=None,error=None):self.calls=[];self.answer=answer;self.error=error
    def ask(self,path,prompt,images,validator):
        self.calls.append(dict(path=path,prompt=prompt,images=images))
        if self.error:raise RuntimeError(self.error)
        if prompt.get('task')==head_semantics.TASK:
            return validator(dict(heads={r['id']:r['heads'] for r in self.answer['routes']}))
        return validator(copy.deepcopy(self.answer))


def test_good_mask_has_no_extra_request_and_unchanged_geometry(tmp_path):
    crop,candidates,locks=fixture(tmp_path);model=Model(error='must not call')
    answer=dict(nodes=[['a',10,20],['b',80,20]],routes=[route()],excluded=[],uncertain=[])
    old,_,_=mask_vectors.physical_rows(answer,{},candidates,'G1')
    rows,supports,issues,calls=routing.recover(answer,{},candidates,'G1',crop,tmp_path,model,locks)
    assert calls==0 and not model.calls and not issues
    assert rows==old
    assert np.array_equal(supports['G1:a'],candidates[0]['mask'])


def test_glm_route_preserves_exact_points_and_mask_style_evidence(tmp_path):
    crop,candidates,locks=fixture(tmp_path)
    answer=dict(nodes=[],routes=[route(mode='glm',nodes=[])],excluded=[],uncertain=[])
    fitted=dict(nodes=[[1,12,19],[2,30,21],[3,60,45],[4,60,80]],routes=[dict(id='R1',heads=['none','stealth'],segments=[dict(kind='curve',nodes=[1,2,3]),dict(kind='line',nodes=[3,4])])])
    model=Model(fitted)
    with patch.object(mask_vectors,'fit_route',side_effect=AssertionError('Do not fit GLM geometry with mask')):
        rows,supports,issues,calls=routing.recover(answer,{},candidates,'G1',crop,tmp_path,model,locks)
    assert calls==1 and not issues and len(rows)==1
    assert rows[0]['start']==[12,19] and rows[0]['end']==[60,80]
    assert rows[0]['coordinate_source']=='glm_segmented_points'
    assert rows[0]['model_nodes']==fitted['nodes']
    assert rows[0]['heads']==['none','stealth'] and rows[0]['end_head_type']=='stealth'
    assert len(model.calls)==2 and rows[0]['head_source']=='glm_context_text'
    assert np.array_equal(supports['G1:a'],candidates[0]['mask'])
    assert 'previous_geometry' not in model.calls[0]['prompt']
    assert model.calls[0]['prompt']['targets'][0]['identity_kind']=='straight'
    # Earlier kind is reference only: the validated model-owned curve is retained.
    assert rows[0]['kind']=='curve'


def test_shared_network_one_request_one_junction_and_independent_crossing(tmp_path):
    crop,candidates,locks=fixture(tmp_path)
    routes=[route('a','glm',[],network='N1'),route('b','glm',[],network='N1')]
    answer=dict(nodes=[],routes=routes,excluded=[],uncertain=[])
    fitted=dict(nodes=[['a',10,20],['j',50,20],['b',50,80]],routes=[dict(id='R1',heads=['none','none'],segments=[dict(kind='line',nodes=['a','j'])]),dict(id='R2',heads=['none','triangle'],segments=[dict(kind='line',nodes=['j','b'])])])
    rows,_,issues,calls=routing.recover(answer,{},candidates,'G1',crop,tmp_path,Model(fitted),locks)
    assert calls==1 and not issues and rows[0]['end']==rows[1]['start']==[50,20]
    assert rows[0]['parent_id']==rows[1]['parent_id']
    assert rows[0]['shared_nodes']==rows[1]['shared_nodes']==['G1:F001:j']
    assert len(routing.components([route('a','glm',[]),route('b','glm',[])]))==2


def test_failed_glm_keeps_good_independent_target(tmp_path):
    crop,candidates,locks=fixture(tmp_path)
    answer=dict(nodes=[['a',10,20],['b',80,20]],routes=[route('a'),route('b','glm',[])],excluded=[],uncertain=[])
    rows,_,issues,calls=routing.recover(answer,{},candidates,'G1',crop,tmp_path,Model(error='service unavailable'),locks)
    assert calls==1 and [r['id'] for r in rows]==['G1:a']
    assert issues[0]['stage']=='glm_geometry' and issues[0]['routes']==['b']


def test_mask_solver_exception_requests_glm_instead_of_silent_anchor_fallback(tmp_path):
    crop,candidates,locks=fixture(tmp_path)
    answer=dict(nodes=[['a',10,20],['b',80,20]],routes=[route()],excluded=[],uncertain=[])
    fitted=dict(nodes=[['x',11,20],['y',79,21]],routes=[dict(id='R1',heads=['none','triangle'],segments=[dict(kind='line',nodes=['x','y'])])])
    with patch.object(mask_vectors,'fit_route',side_effect=ValueError('empty measurement')):
        rows,_,issues,calls=routing.recover(answer,{},candidates,'G1',crop,tmp_path,Model(fitted),locks)
    assert calls==1 and rows[0]['start']==[11,20]
    assert issues[0]['fallback']=='deferred_glm'


def test_deferred_schema_still_requires_identity_accounting_and_valid_heads():
    answer=dict(nodes=[],routes=[route(mode='glm',nodes=[],network='N')],excluded=[],uncertain=[])
    assert semantic_geometry.validate(answer,(100,100),['G1'],{},['M1'])['nodes']==[]
    with pytest.raises(ValueError,match='Account'):
        semantic_geometry.validate(answer,(100,100),['G1','G2'],{},['M1'])
    bad=copy.deepcopy(answer);bad['routes'][0]['heads']=['none','bad']
    with pytest.raises(ValueError,match='head'):
        semantic_geometry.validate(bad,(100,100),['G1'],{},['M1'])


def test_shared_reference_not_inferred_from_equal_coordinates():
    answer=dict(nodes=[['a',0,10],['b',30,10],['c',30,10],['d',30,50]],
        routes=[dict(id='R1',segments=[dict(kind='line',nodes=['a','b'])]),dict(id='R2',segments=[dict(kind='line',nodes=['c','d'])])])
    rows=segmented_geometry.geometry(segmented_geometry.validate(answer,['R1','R2'],(100,100)),{'R1':{},'R2':{}})
    assert all(not r['shared_nodes'] for r in rows)


def test_lone_mask_segment_with_external_network_name_keeps_fast_path(tmp_path):
    crop,candidates,locks=fixture(tmp_path)
    answer=dict(nodes=[['a',10,20],['b',80,20]],routes=[route(network='continues_off_crop')],excluded=[],uncertain=[])
    expected,_,_=mask_vectors.physical_rows(answer,{},candidates,'G1')
    model=Model(error='one mask segment does not require a joint geometry request')
    rows,_,issues,calls=routing.recover(answer,{},candidates,'G1',crop,tmp_path,model,locks)
    assert rows==expected and calls==0 and not issues and not model.calls


def test_final_node_order_owns_heads_instead_of_old_identity_order(tmp_path):
    crop,candidates,locks=fixture(tmp_path)
    answer=dict(nodes=[],routes=[route(mode='glm',nodes=[])],excluded=[],uncertain=[])
    # The new geometry runs right to left. Its visible right head must stay
    # attached to x=80, even though the old identity had it at the last end.
    fitted=dict(nodes=[['right',80,20],['left',10,20]],routes=[dict(id='R1',heads=['stealth','none'],segments=[dict(kind='line',nodes=['right','left'])])])
    rows,_,issues,_=routing.recover(answer,{},candidates,'G1',crop,tmp_path,Model(fitted),locks)
    assert not issues and rows[0]['start']==[80,20] and rows[0]['end']==[10,20]
    assert rows[0]['start_head_type']=='stealth' and not rows[0]['end_head']
    assert rows[0]['heads']==['stealth','none']


@pytest.mark.parametrize('locator',['Pale-blue connector from central tokens toward the left panel.',None])
def test_geometry_uses_locator_without_forwarding_old_guessed_route(tmp_path,locator):
    crop,candidates,locks=fixture(tmp_path)
    old='Guessed corner x=652; hug panel border, then connect to route R1.'
    r=route(mode='glm',nodes=[],description=old)
    if locator is not None:r['locator']=locator
    answer=dict(nodes=[],routes=[r],excluded=[],uncertain=[])
    fitted=dict(nodes=[['a',10,20],['b',80,20]],routes=[dict(id='R1',heads=['none','triangle'],segments=[dict(kind='line',nodes=['a','b'])])])
    model=Model(fitted)
    rows,supports,issues,calls=routing.recover(answer,{},candidates,'G1',crop,tmp_path,model,locks)
    assert not issues and calls==1 and rows[0]['start']==[10,20]
    q=model.calls[0]['prompt'];target=q['targets'][0]
    assert target['description']==(locator or '')
    assert old not in json.dumps(q)
    assert target['mask_measurements']==routing.measurements(['M1'],{'M1':candidates[0]}, {})
    assert np.array_equal(supports['G1:a'],candidates[0]['mask'])
    audit=json.loads((tmp_path/'geometry/F001/identity_context_not_sent.json').read_text())
    assert audit[0]['omitted_description']==old


def test_branched_legacy_target_keeps_every_path_through_main_routing(tmp_path):
    crop,candidates,locks=fixture(tmp_path)
    identity=dict(nodes=[],routes=[route('a','glm',[])],excluded=[],uncertain=[])
    raw=dict(nodes=[['a',10,40],['j',50,40],['b',90,40],['c',50,10],['u',20,70],['v',60,70]],
        routes=[dict(id='R1',heads=['none','triangle'],segments=[
            dict(kind='line',nodes=['a','j']),dict(kind='line',nodes=['j','b']),
            dict(kind='line',nodes=['c','j']),dict(kind='line',nodes=['u','v'])])])
    model=Model(raw)
    def heads(crop,rows,supports,*args,**kwargs):
        assert len(rows)==3 and set(supports)=={r['id'] for r in rows}
        assert all(r['heads']==['unknown','unknown'] for r in rows)
        for r in rows:r['heads']=['none','none']
        return rows,[],0
    with patch.object(head_semantics,'recover',heads):
        rows,supports,issues,calls=routing.recover(identity,{},candidates,'G1',crop,tmp_path,model,locks)
    assert not issues and calls==1 and len(rows)==3
    assert [(r['start'],r['end']) for r in rows]==[([10,40],[90,40]),([50,10],[50,40]),([20,70],[60,70])]
    assert len({r['parent_id'] for r in rows})==1
    assert all(np.array_equal(supports[r['id']],candidates[0]['mask']) for r in rows)
    assert sum(len(r['model_segments']) for r in rows)==4
    assert all(r['model_nodes']==raw['nodes'] for r in rows)


def test_explicit_paths_do_not_require_a_shared_node_or_single_traversal():
    raw=dict(nodes=[['a',10,10],['b',40,10],['c',60,30],['d',60,80]],routes=[dict(id='R1',paths=[
        dict(id='main',segments=[dict(kind='line',nodes=['a','b'])]),
        dict(id='branch',segments=[dict(kind='line',nodes=['c','d'])])])])
    checked=segmented_geometry.validate(raw,{'R1':{}},[100,100])
    rows=segmented_geometry.geometry(checked,{'R1':dict(id='G:R1',heads=['none','triangle'])})
    assert [r['id'] for r in rows]==['G:R1/main','G:R1/branch']
    assert all(r['heads']==['unknown','unknown'] for r in rows)
    assert all(not r['shared_nodes'] for r in rows)
    assert checked==raw


def test_malformed_point_references_still_fail_under_multipath_protocol():
    raw=dict(nodes=[['a',10,10],['b',40,10]],routes=[dict(id='R1',paths=[
        dict(id='P1',segments=[dict(kind='line',nodes=['a','missing'])])])])
    with pytest.raises(ValueError,match='Undefined'):
        segmented_geometry.validate(raw,['R1'],[100,100])
