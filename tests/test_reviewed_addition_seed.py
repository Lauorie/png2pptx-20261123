import copy
import json
import pytest
from shape_integration.fitting.seed import reviewed_addition_seed


def inputs(tmp_path):
    local={'fit_status':'ok','observation_status':'ok','parts':[{'id':'existing','status':'ok'}]}
    (tmp_path/'fit_local.json').write_text(json.dumps(local))
    seed={'path':str(tmp_path),'fit_reused':False,'fallback_fit_available':True,'source_state':'unresolved'}
    review={'protocol_version':'shape_integration.fit_check.10','decision':'revise',
        'issues':[{'kind':'missing','parts':[],'part_ids':[]}],
        'part_id_map':{'F1':'existing'},'source_observations':[{'source_relation':'supported'}]}
    return seed,review


def test_current_source_review_repairs_only_additions_without_relabeling_old_acceptance(tmp_path):
    seed,review=inputs(tmp_path);original=copy.deepcopy(seed)
    selected=reviewed_addition_seed(seed,review)
    assert selected['fit_reused'] and selected['reviewed_edit_scope']=='current_source_missing_only'
    assert selected['historical_state_unchanged']=='unresolved' and seed==original


@pytest.mark.parametrize('invalid',['geometry','existing_part','missing_map','coverage_only','unsupported'])
def test_unreviewed_or_disputed_old_geometry_still_requires_fresh_observation(tmp_path,invalid):
    seed,review=inputs(tmp_path)
    if invalid=='geometry':review['issues'].append({'kind':'geometry','parts':['F1']})
    elif invalid=='existing_part':review['issues'][0]['part_ids']=['existing']
    elif invalid=='missing_map':review['part_id_map']={}
    elif invalid=='coverage_only':review.pop('protocol_version')
    else:review['source_observations']=[]
    assert reviewed_addition_seed(seed,review)==seed
