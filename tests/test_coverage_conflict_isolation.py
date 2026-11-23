from copy import deepcopy

import pytest

from shape_integration.common import save
from shape_integration.sam_fixed import FixedSamEvidence
from shape_integration.fitting.control import withheld_source_context, needs_more_source


def test_disputed_sam_claim_does_not_discard_independent_missing_frame(tmp_path):
    save(tmp_path/'sam_evidence.json',{'shown':[{'id':'S1','bbox_original':[40,40,50,50]}]})
    provider=object.__new__(FixedSamEvidence)
    raw={'status':'complete','uncertain':[],
         'new_objects':[{'id':'new_frame','bbox_2d':[60,20,90,70]}],
         'sam_decisions':[{'sam_id':'S1','action':'covered','target':'wiring','reason':'inside group'}]}
    existing=[{'id':'wiring','fitted_part_boxes':[[0,0,100,100]]}]
    original=deepcopy(raw)
    with pytest.raises(ValueError,match='covered contradicts'):
        provider.review_validator(lambda x:x,tmp_path,existing)(raw)
    result=provider.review_validator(lambda x:x,tmp_path,existing,isolate_coverage_conflicts=True)(raw)
    assert result['new_objects']==raw['new_objects']
    assert result['status']=='uncertain' and result['uncertain']
    assert result['sam_decisions'][0]['action']=='uncertain'
    assert result['sam_decisions'][0]['target'] is None
    assert result['sam_coverage_conflicts']==raw['sam_decisions']
    assert raw==original
    assert provider.review_validator(lambda x:x,tmp_path,existing,isolate_coverage_conflicts=True)(result)==result


def test_bad_sam_identity_still_requires_protocol_repair(tmp_path):
    save(tmp_path/'sam_evidence.json',{'shown':[{'id':'S1','bbox_original':[40,40,50,50]}]})
    provider=object.__new__(FixedSamEvidence)
    validator=provider.review_validator(lambda x:x,tmp_path,[],isolate_coverage_conflicts=True)
    with pytest.raises(ValueError):validator({'new_objects':[],'uncertain':[],'sam_decisions':[]})


def test_missing_member_can_need_context_when_retained_parts_are_interior():
    job={'frame':{'source_crop_half_open':[20,20,80,80],
                  'original_image_size_px':[100,100],'actual_crop_input_size_px':[600,600]},
         'sam_support':{'context_expansion_withheld':True,'context_bbox_original':[30,30,90,70]}}
    assert not needs_more_source([100,100,400,400],job)
    assert withheld_source_context(job)
    job['frame']['source_crop_half_open']=[0,0,100,100]
    job['sam_support']['context_bbox_original']=[-2,20,110,70]
    assert not withheld_source_context(job)
    job['frame']['source_crop_half_open']=[20,20,80,80]
    job['sam_support']['context_expansion_withheld']=False
    assert not withheld_source_context(job)


def test_unknown_part_repair_preserves_valid_independent_discovery_and_repair():
    from shape_integration.fitting.coverage import isolate_repair_parts
    scene={'objects':{'owner':{'part_keys':['key']}},'parts':{'key':{'id':'actual'}}}
    bad={'object_id':'owner','issues':[{'part_ids':['wrong_name'],'kind':'extra'}]}
    good={'object_id':'owner','issues':[{'part_ids':[],'kind':'missing'}]}
    raw={'status':'complete','uncertain':[],'repair_existing':[bad,good],
         'new_objects':[{'label':'independent frame'}]}
    result=isolate_repair_parts(raw,scene)
    assert result['new_objects']==raw['new_objects'] and result['repair_existing']==[good]
    assert result['held_repairs']==[bad] and result['status']=='uncertain'
    assert isolate_repair_parts(result,scene)==result
    assert raw['repair_existing']==[bad,good]
