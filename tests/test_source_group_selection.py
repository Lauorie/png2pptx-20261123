"""Independent source routes may choose different branches in one review crop."""
import pytest
import numpy as np

from full_recovery.quality_choice import validate_selection,validate_source_selection,apply_choices,nominate

BRANCH={'C1':'arrows','C2':'shape','C3':'arrows','C4':'shape'}
SOURCE_BOX=[0,0,100,40]

def observed_paths():
    return {'keep_ids':['C1','C4'],'reason':'Two independently visible source connections.',
        'source_objects':[
            {'id':'upper','source_path':[[10,10],[90,10]],'candidate_ids':['C1','C2'],
             'keep_ids':['C1'],'reason':'The arrow product follows the upper source path.'},
            {'id':'lower','source_path':[[10,30],[90,30]],'candidate_ids':['C3','C4'],
             'keep_ids':['C4'],'reason':'The shape product follows the lower source path.'}]}

def check(raw,**kwargs):
    return validate_source_selection(raw,BRANCH,branch_by_id=BRANCH,source_box=SOURCE_BOX,**kwargs)

def test_binary_choice_is_per_source_route_with_unmatched_native_routes_preserved():
    raw=observed_paths()
    with pytest.raises(ValueError,match='one branch'):
        validate_selection(raw,BRANCH,branch_by_id=BRANCH)
    choice=check(raw)
    rows=[{'id':n,'key':n,'keys':[n],'refs':['owner/'+n],'layer':layer} for n,layer in BRANCH.items()]
    rows.append({'id':'independent','key':'independent','keys':['independent'],'refs':['owner/independent'],'layer':'shape'})
    plan=apply_choices(rows,[{**choice,'candidate_ids':list(BRANCH)}])
    assert plan['drop_shape_keys']==['C2']
    assert plan['drop_arrow_names']==['C3']
    assert 'independent' not in plan['drop_shape_keys']

def test_same_physical_source_route_cannot_keep_both_branches():
    raw=observed_paths();raw['source_objects'][0]['keep_ids']=['C1','C2'];raw['keep_ids'].append('C2')
    with pytest.raises(ValueError,match='one branch'):check(raw)


def test_short_and_whole_native_routes_cannot_evade_choice_with_different_source_ids():
    def route(ident,branch,end):
        points=np.c_[np.linspace(10,end,80),np.full(80,10.)]
        return {'id':ident,'layer':branch,'points':points,'length':end-10,
                'heads':['none','none'],'style':{'width_px':1}}
    rows=[route('short','arrows',40),route('whole','shape',90)]
    pairs=nominate(rows)
    dependencies=[{p['a'],p['b']}for p in pairs if p.get('shared_native_span')]
    assert dependencies==[{'short','whole'}]
    branches={'short':'arrows','whole':'shape'}
    raw={'keep_ids':['short','whole'],'reason':'Incorrectly split one whole native route.',
         'source_objects':[{'id':'upper','source_path':[[10,10],[40,10]],
            'candidate_ids':['short'],'keep_ids':['short'],'reason':'Upper part.'},
           {'id':'lower','source_path':[[40,10],[90,10]],
            'candidate_ids':['whole'],'keep_ids':['whole'],'reason':'Remaining part.'}]}
    with pytest.raises(ValueError,match='shared native spans'):
        validate_source_selection(raw,branches,branch_by_id=branches,source_box=SOURCE_BOX,dependencies=dependencies)
    raw={'keep_ids':['whole'],'reason':'Whole native route covers both source spans.',
         'source_objects':[{'id':'upper','source_ids':['upper','lower'],'source_path':[[10,10],[40,10]],
            'candidate_ids':['short','whole'],'keep_ids':['whole'],'reason':'One complete implementation.'}]}
    result=validate_source_selection(raw,branches,branch_by_id=branches,source_box=SOURCE_BOX,dependencies=dependencies)
    assert result['keep_ids']==['whole']


def test_explicit_output_envelope_still_requires_all_native_candidates():
    raw=observed_paths()
    envelope={'task':'echoed input','source_objects':[{'id':'input_metadata'}],'output':raw}
    result=check(envelope)
    assert result['keep_ids']==['C1','C4'] and result['response_envelope']=='output'
    raw['source_objects'][1]['candidate_ids']=['C4']
    with pytest.raises(ValueError,match='C3'):check(envelope)


def test_output_envelope_cannot_bypass_one_branch_per_source_path():
    raw=observed_paths();raw['source_objects'][0]['keep_ids']=['C1','C2'];raw['keep_ids'].append('C2')
    with pytest.raises(ValueError,match='one branch'):check({'output':raw})

def test_relabelling_the_same_source_path_does_not_bypass_binary_choice():
    raw=observed_paths();raw['source_objects'][1]['source_path']=[[90,10],[10,10]]
    with pytest.raises(ValueError,match='Repeated source_path'):check(raw)

def test_every_source_object_retains_a_native_recovery():
    raw=observed_paths();raw['source_objects'][1]['keep_ids']=[];raw['keep_ids']=['C1']
    with pytest.raises(ValueError,match='Keep only listed candidates'):check(raw)

def test_unaccounted_or_repeated_native_candidate_cannot_authorize_deletion():
    for alteration in ('omitted','repeated'):
        raw=observed_paths()
        if alteration=='omitted':raw['source_objects'][0]['candidate_ids']=['C1']
        else:raw['source_objects'][1]['candidate_ids'].append('C1')
        with pytest.raises(ValueError,match='candidate'):check(raw)

def test_a_contour_and_its_segment_edit_share_the_same_source_decision():
    with pytest.raises(ValueError,match='native contour'):check(observed_paths(),dependencies=[{'C1','C3'}])
    with pytest.raises(ValueError,match='required native contours'):check(observed_paths(),required=['C2'])

def test_source_anchors_use_original_crop_coordinates_and_survivor_union_is_complete():
    raw=observed_paths();raw['source_objects'][0]['source_path']=[[200,10],[220,10]]
    with pytest.raises(ValueError,match='original-pixel crop'):check(raw)
    raw=observed_paths();raw['keep_ids']=['C1']
    with pytest.raises(ValueError,match='survivor union'):check(raw)

def test_native_candidates_cannot_invent_or_move_a_frozen_source_path():
    raw=observed_paths();definitions=[{'id':o['id'],'source_path':o['source_path']} for o in raw['source_objects']]
    check(raw,source_definitions=definitions)
    raw['source_objects'][0]['id']='invented_from_native_candidate'
    with pytest.raises(ValueError,match='independently frozen'):check(raw,source_definitions=definitions)
    raw=observed_paths();raw['source_objects'][0]['source_path']=[[11,10],[91,10]]
    with pytest.raises(ValueError,match='source path is locked'):check(raw,source_definitions=definitions)

def test_independently_observed_context_does_not_delete_unmatched_native_content():
    raw=observed_paths();definitions=[{'id':o['id'],'source_path':o['source_path']} for o in raw['source_objects']]
    definitions.append({'id':'context','source_path':[[3,3],[3,35]]})
    result=check(raw,source_definitions=definitions)
    assert result['keep_ids']==['C1','C4'] and result['unassigned_source_ids']==['context']

def test_source_observation_receives_original_pixels_without_native_candidate_data(tmp_path):
    import json
    from PIL import Image
    from full_recovery.quality_choice import observe_source
    class SourceOnly:
        def call(self,folder,tag,system,prompt,images,validator,budget):
            assert len(images)==1 and tag=='inventory'
            assert not any(key in json.dumps(prompt) for key in ['candidate_geometry','native_id','candidate_branches'])
            assert Image.open(images[0][1]).getpixel((0,0))==(31,67,89)
            return validator({'source_objects':[{'id':'observed','source_path':[[8,8],[40,8]],'description':'Visible source path.'}]})
    result=observe_source(Image.new('RGB',(100,80),(31,67,89)),(10,20,30,40),tmp_path,SourceOnly())
    assert result[0]['source_path']==[[11.0,21.0],[15.0,21.0]]

def test_a_fully_observed_source_path_cannot_be_silently_left_unrepresented():
    raw=observed_paths();raw['source_objects']=raw['source_objects'][:1]
    raw['source_objects'][0]['candidate_ids']=list(BRANCH);raw['keep_ids']=['C1']
    with pytest.raises(ValueError,match='fully observed source object'):
        check(raw,required_source_ids=['upper','lower'])

def test_read_only_context_accounts_for_real_source_nodes_without_deleting_a_route():
    raw=observed_paths();raw['source_objects'].append({'id':'node','source_path':[[1,1],[1,5]],
        'candidate_ids':[],'keep_ids':[],'context_ids':['N1'],'reason':'Already in the shared native context.'})
    assert check(raw,required_source_ids=['upper','lower','node'],context_by_id={'N1':{}})['keep_ids']==['C1','C4']
    with pytest.raises(ValueError,match='read-only native context'):
        check(raw,required_source_ids=['upper','lower','node'],context_by_id={})


def test_one_atomic_native_route_can_couple_several_observed_source_spans():
    definitions=[{'id':'left','source_path':[[5,10],[45,10]]},
                 {'id':'right','source_path':[[55,10],[95,10]]}]
    raw={'keep_ids':['whole'],'reason':'Keep one complete native route.',
         'source_objects':[{'id':'left','source_ids':['left','right'],
             'candidate_ids':['whole','fragment_left','fragment_right'],
             'keep_ids':['whole'],'reason':'The whole native route crosses both source spans.'}]}
    branches={'whole':'shape','fragment_left':'arrows','fragment_right':'arrows'}
    result=validate_source_selection(raw,branches,branch_by_id=branches,
        source_box=SOURCE_BOX,source_definitions=definitions,required_source_ids=['left','right'])
    assert result['source_objects'][0]['source_definitions']==definitions
    assert result['unassigned_source_ids']==[]
    raw['keep_ids'].append('fragment_left');raw['source_objects'][0]['keep_ids'].append('fragment_left')
    with pytest.raises(ValueError,match='one branch'):
        validate_source_selection(raw,branches,branch_by_id=branches,
            source_box=SOURCE_BOX,source_definitions=definitions)


def test_coupled_source_units_cannot_invent_or_repeat_source_members():
    raw=observed_paths();definitions=[{'id':o['id'],'source_path':o['source_path']} for o in raw['source_objects']]
    raw['source_objects'][0]['source_ids']=['upper','lower']
    with pytest.raises(ValueError,match='distinct frozen source IDs'):
        check(raw,source_definitions=definitions)


def test_read_only_context_metadata_can_accompany_a_real_native_survivor():
    raw=observed_paths()
    for obj in raw['source_objects']:obj['context_ids']=['N1']
    assert check(raw,context_by_id={'N1':{}})['keep_ids']==['C1','C4']


def test_unmatched_frozen_crop_edge_context_is_reported_without_dropping_a_candidate():
    raw=observed_paths();definitions=[{'id':o['id'],'source_path':o['source_path']} for o in raw['source_objects']]
    definitions.append({'id':'edge','source_path':[[0,0],[7,4]]})
    raw['source_objects'].append({'id':'edge','candidate_ids':[],'keep_ids':[],
                                'reason':'Visible crop-edge continuation has no nominated native alternative.'})
    result=check(raw,source_definitions=definitions,required_source_ids=['upper','lower'])
    assert result['keep_ids']==['C1','C4'] and result['unassigned_source_ids']==['edge']
    with pytest.raises(ValueError,match='candidate'):
        check(raw,source_definitions=definitions,required_source_ids=['upper','lower','edge'])
    raw=observed_paths();raw['source_objects'][0]['source_ids']=['upper','invented']
    with pytest.raises(ValueError,match='independently frozen'):
        check(raw,source_definitions=definitions)


def test_same_call_source_context_cannot_change_candidate_conservation_or_binary_choice():
    raw=observed_paths()
    raw['source_objects'].append({'id':'context','source_path':[[3,3],[3,35]],
        'candidate_ids':[],'keep_ids':[],'reason':'Visible neighboring path has no current C.'})
    assert check(raw)['keep_ids']==['C1','C4']
    with pytest.raises(ValueError,match='candidate'):
        check(raw,required_source_ids=['context'])
    raw['source_objects'][0]['candidate_ids']=['C1']
    with pytest.raises(ValueError,match='C2'):check(raw)


def test_passive_source_context_cannot_authorize_deleting_all_candidates():
    raw={'keep_ids':[],'reason':'Only context.',
         'source_objects':[{'id':'context','source_path':[[3,3],[3,35]],
             'candidate_ids':[],'keep_ids':[],'reason':'No candidate.'}]}
    with pytest.raises(ValueError,match='required native contours'):check(raw)
