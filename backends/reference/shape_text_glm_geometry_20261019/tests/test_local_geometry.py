from copy import deepcopy
import math
import pytest
from shape_integration.texts.local_geometry import compact_groups, validate
from shape_integration.common import Runtime
from shape_integration.texts.controller import mapped_box


def test_compact_groups_have_unique_owners_and_no_long_chains():
    rows=[{'id':str(i),'bbox_2d':[i*90,20,i*90+30,32]} for i in range(14)]
    groups=compact_groups(rows,[2048,1024])
    assert sorted(r['id'] for g in groups for r in g)==sorted(r['id'] for r in rows)
    assert all(len(g)<=6 for g in groups)
    assert all(max(r['bbox_2d'][0] for r in g)-min(r['bbox_2d'][0] for r in g)<=160 for g in groups)


def test_geometry_rejects_content_mutation_and_ambiguous_guesses():
    item={'id':'T1','status':'located','bbox_2d':[10,20,40,150],'rotation_deg':90,'direction_status':'confirmed'}
    assert validate({'items':[item]},['T1'],[200,200])
    assert validate([item],['T1'],[200,200])=={'items':[item]}
    for changes in [{'text':'changed'},{'rotation_deg':True},{'direction_status':'ambiguous'},{'bbox_2d':[1,2,300,4]}]:
        with pytest.raises(ValueError):validate({'items':[{**item,**changes}]},['T1'],[200,200])


def test_pixel_center_mapping_roundtrip():
    f={'scale':3.17,'offset':[.3,.1],'box':[133,78,500,200]}
    box=[150.2,83.4,199.8,155.2]
    assert mapped_box(mapped_box(box,f),f,True)==pytest.approx(box,abs=1e-10)


def test_low_effort_survives_request_wrapper(tmp_path):
    config={'api_concurrency':1,'model':'glm-5.3-flash','reasoning_effort':'low'}
    parameters=Runtime(config,'http://unused',None).client(tmp_path,100000).request_parameters()
    assert parameters['max_tokens']==100000
    assert parameters['reasoning_effort']=='low'
