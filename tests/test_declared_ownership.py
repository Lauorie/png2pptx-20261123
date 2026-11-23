from copy import deepcopy
import pytest

from shape_integration.fitting.contracts import normalize_observation, ObservationError


def observation(unit='other'):
    return {'status':'ok', 'points':[{'id':'a','xy':[10,10]}, {'id':'b','xy':[30,10]},
            {'id':'c','xy':[30,30]}, {'id':'d','xy':[10,30]}, {'id':'e','xy':[60,20]}],
        'parts':[{'id':'node','kind':'rectangle','point_ids':['a','b','c','d']},
                 {'id':'route','kind':'path','closed':False,
                  'segments':[{'kind':'line','point_ids':['b','e']}]}],
        'ownership':{'unit':unit,'unit_evidence':'Observed source units','parts':[{'id':'node','role':'body'},
                                       {'id':'route','role':'connection'}]}}


def test_headless_graph_compound_keeps_internal_route_and_shared_points():
    raw=observation();before=deepcopy(raw);audit={}
    result=normalize_observation(raw,'target',[100,100],audit=audit,ownership_required=True)
    assert [p['id']for p in result['parts']]==['node','route']
    assert {p['id']for p in result['points']}=={'a','b','c','d','e'}
    assert result['status']=='ok' and raw==before
    assert audit['declared_ownership']['delegated_parts']==[]


def test_noncompound_object_does_not_claim_internal_routes():
    audit={}
    result=normalize_observation(observation(),'target',[100,100],ownership_required=True,
        allow_compound_connections=False,audit=audit)
    assert [p['id'] for p in result['parts']]==['node']
    assert audit['declared_ownership']['delegated_parts']==[
        {'id':'route','role':'connection','owner':'arrow'}]


@pytest.mark.parametrize('compound',[True,False])
def test_external_headless_route_is_delegated_without_a_registered_arrow(compound):
    raw=observation();raw['ownership']['parts'][1]['role']='external_connection'
    raw['parts'].append({'id':'seam','kind':'path','closed':False,
        'segments':[{'kind':'line','point_ids':['a','c']}]})
    raw['ownership']['parts'].append({'id':'seam','role':'intrinsic_stroke'})
    audit={}
    result=normalize_observation(raw,'target',[100,100],ownership_required=True,
        allow_compound_connections=compound,audit=audit)
    assert [p['id'] for p in result['parts']]==['node','seam']
    assert audit['declared_ownership']['delegated_parts']==[
        {'id':'route','role':'external_connection','owner':'arrow'}]


def test_source_directional_head_delegates_the_neural_connection_whole():
    raw=observation('neural_network')
    raw['points']+=[{'id':'f','xy':[55,17]}, {'id':'g','xy':[55,23]}]
    raw['parts']+=[{'id':'tip','kind':'path','closed':True,'segments':[
        {'kind':'line','point_ids':['e','f']},{'kind':'line','point_ids':['f','g']},
        {'kind':'line','point_ids':['g','e']}]}]
    raw['ownership']['parts']+=[{'id':'tip','role':'directional_head'}]
    result=normalize_observation(raw,'target',[100,100],ownership_required=True)
    assert [p['id']for p in result['parts']]==['node']
    assert {p['id']for p in result['points']}=={'a','b','c','d'}


def test_uncertain_source_roles_keep_explicit_unresolved_geometry():
    raw=observation();raw['ownership']['parts'][1]['role']='uncertain'
    result=normalize_observation(raw,'target',[100,100],ownership_required=True)
    assert result['status']=='partial' and len(result['parts'])==2


@pytest.mark.parametrize('change',['missing','duplicate','unknown','closed_shaft'])
def test_invalid_or_incomplete_roles_are_rejected(change):
    raw=observation('neural_network')
    if change=='missing':raw['ownership']['parts'].pop()
    elif change=='duplicate':raw['ownership']['parts']+=[raw['ownership']['parts'][0]]
    elif change=='unknown':raw['ownership']['parts'][1]['role']='arbitrary'
    else:raw['parts'][1]['closed']=True
    with pytest.raises(ObservationError):normalize_observation(raw,'target',[100,100],ownership_required=True)


def test_legacy_artifacts_remain_readable_but_new_requests_require_roles():
    raw=observation();raw.pop('ownership')
    assert len(normalize_observation(raw,'target',[100,100])['parts'])==2
    with pytest.raises(ObservationError):normalize_observation(raw,'target',[100,100],ownership_required=True)


def test_noncompound_node_cannot_claim_external_routes():
    audit={}
    result=normalize_observation(observation('neural_network'),'target',[100,100],
        ownership_required=True,allow_compound_connections=False,audit=audit)
    assert [p['id']for p in result['parts']]==['node']
    assert audit['declared_ownership']['delegated_parts'][0]['owner']=='arrow'


def test_bounded_edit_retains_roles_for_unchanged_parts_and_drops_retired_ids():
    from shape_integration.fitting.ownership import retained_receipt
    current={'unit':'other','retained_roles':{'edited':'contour'},'uncertain_parts':[]}
    base={'retained_roles':{'unchanged':'body','route':'connection','removed':'body'}}
    result=retained_receipt(current,base,['unchanged','route','edited'])
    assert result['retained_roles']=={'unchanged':'body','route':'connection','edited':'contour'}
    assert current['retained_roles']=={'edited':'contour'}


def test_count_conflict_cannot_delete_all_independently_named_members():
    from shape_integration.fitting.critic import validate_review
    issue={'kind':'extra','parts':['F1','F2'],'evidence':'One excess member in this family',
           'action':'Remove the unsupported member, keeping the source members'}
    with pytest.raises(ValueError,match='exactly one whole part'):
        validate_review({'decision':'revise','issues':[issue]},['F1','F2'])
    issue['kind']='topology'
    assert validate_review({'decision':'revise','issues':[issue]},['F1','F2'])['decision']=='revise'
