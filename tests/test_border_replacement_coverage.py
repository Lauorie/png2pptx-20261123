import numpy as np
import pytest

from full_recovery.quality_choice import apply_choices
from full_recovery.handoff import resample
from full_recovery.image_assets import registry_groups
from shape_integration.composition import crop_box


def candidate(ident, layer, points, **extra):
    return dict(id=ident, layer=layer, key=ident, keys=[ident] if layer=='shape' else [],
                refs=[ident], points=resample([np.asarray(points,float)]),
                style={'width_px':1}, **extra)


def test_partial_arrow_cannot_remove_other_three_frame_edges():
    shape=candidate('frame','shape',[[0,0],[100,0],[100,80],[0,80],[0,0]],closed=True)
    arrow=candidate('short','arrows',[[0,0],[100,0]])
    result=apply_choices([shape,arrow],[{'candidate_ids':['frame','short'],'keep_ids':['short']}])
    assert result['drop_shape_names']==[]
    assert result['native_frame_guards'][0]['covered_fraction']<.4


def test_full_border_replacement_is_still_allowed():
    points=[[0,0],[100,0],[100,80],[0,80],[0,0]]
    rows=[candidate('frame','shape',points,closed=True),candidate('complete','arrows',points)]
    result=apply_choices(rows,[{'candidate_ids':['frame','complete'],'keep_ids':['complete']}])
    assert result['drop_shape_names']==['frame']
    assert result['native_frame_guards']==[]


def test_registry_shared_leaf_claims_use_one_review_but_disjoint_assets_do_not():
    assets=[{'id':x} for x in 'abcd']
    parts={'a':['one'],'b':['two'],'c':['one','two'],'d':['three']}
    groups=registry_groups(assets,parts)
    assert sorted(sorted(a['id'] for a in group) for group in groups)==[['a','b','c'],['d']]
    assert set(parts[next(g[0]['id'] for g in groups if len(g)==3)])=={'one','two'}


def test_original_coordinate_registry_does_not_require_resized_input_size():
    assert crop_box({'id':'I1','bbox':[10,20,40,60]}, {}, [100,80])==[10,20,41,61]


def test_pixel_center_edge_clamps_without_using_different_detector_scale():
    assert crop_box({'id':'I1','bbox_2d':[-.02,10,30.5,40.2],
                     'bbox':[0,30,90,120]}, {}, [100,80])==[0,10,32,42]


def test_explicit_thinking_off_reaches_wire_without_assistant_prefill(tmp_path):
    from shape_integration.common import Runtime
    runtime=Runtime({'api_concurrency':96,'model':'test','thinking':'off'},'http://localhost','')
    client=runtime.client(tmp_path,100)
    assert client.request_parameters()['thinking']=={'type':'disabled'}
    assert len(client.request_messages('system',[{'type':'text','text':'test'}]))==2


def test_matched_extent_cannot_be_accepted_without_source_accounting():
    from shape_integration.fitting.critic import source_observation_contract,validate_review
    prompt={'extent_comparison':{'required_source_sides':['bottom']}}
    ids=source_observation_contract(prompt,{'F1':'frame'})
    with pytest.raises(ValueError):validate_review({'decision':'accept','issues':[]},['F1'],ids)
    with pytest.raises(ValueError):validate_review({'decision':'accept','issues':[],
        'source_observations':[{'id':'matched_extent','source_relation':'missing',
                                'evidence':'The fitted bottom follows an inner box'}]},['F1'],ids)
