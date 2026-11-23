from copy import deepcopy

import numpy as np
from PIL import Image
import pytest

from shape_integration.fitting.coverage_evidence import bind,local_anchors,augment_prompt,repair_context
from shape_integration.fitting.local import prepare_candidate
from shape_integration.fitting.frames import FitFrame
from shape_integration.fitting.observation import make_prompt


def test_only_unresolved_source_proposals_in_repaired_target_are_carried():
    repairs=[{'object_id':'target','issues':[{'kind':'missing','part_ids':[]}]}]
    existing=[{'id':'target','bbox_2d':[10,20,90,80]}]
    evidence={'shown':[{'id':ident,'bbox_original':box} for ident,box in
        [('missing',[30,40,40,50]),('covered',[50,40,60,50]),('outside',[120,40,130,50])]]}
    decisions=[{'sam_id':ident,'action':action} for ident,action in
               [('missing','uncertain'),('covered','covered'),('outside','uncertain')]]
    before=deepcopy((repairs,existing,evidence,decisions))
    result=bind(repairs,existing,evidence,decisions)
    assert [r['id'] for r in result[0]['source_anchors']]==['missing']
    assert result[0]['issues']==repairs[0]['issues']
    assert result[0]['source_anchors'][0]['association'].endswith('ownership_unverified')
    assert (repairs,existing,evidence,decisions)==before


@pytest.mark.parametrize('mode',['pixel','norm1000'])
def test_repair_prompt_and_locator_share_exact_crop_transform(tmp_path,mode):
    anchor={'id':'source_hint','bbox_original':[106,67,112,73],
            'origin':'unresolved_fitting_coverage_proposal'}
    candidate={'id':'target','label':'compound object','kind':'compound','bbox_2d':[100,60,140,100]}
    context={'coverage_source_anchors':[anchor]}
    folder=tmp_path/'prepared'
    job=prepare_candidate(Image.new('RGB',(300,200),'white'),candidate,[600,400],folder,
                          coordinate_mode=mode,input_context=context)
    rows=local_anchors(job);assert len(rows)==1
    frame=FitFrame.from_metadata(job['frame'])
    expected=frame.to_local(anchor['bbox_original'][:2])+frame.to_local(anchor['bbox_original'][2:])
    assert rows[0]['bbox_local_px']==pytest.approx(expected)
    w,h=job['frame']['actual_crop_input_size_px']
    model_values=[v*1000/((w,h)[i%2]-1) for i,v in enumerate(expected)] if mode=='norm1000' else expected
    assert rows[0]['bbox_2d']==pytest.approx(model_values,abs=.001)
    prompt=make_prompt(job)
    assert 'coverage_source_anchors' not in prompt.get('localization_context',{})
    assert prompt['coverage_source_anchors']['coordinate_mode']==mode
    assert augment_prompt({},job)['coverage_source_anchors']==prompt['coverage_source_anchors']
    with Image.open(folder/'overlay.png') as image:
        pixels=np.asarray(image.convert('RGB'))
        assert np.any(np.all(pixels==[0,131,107],axis=2))
    assert 'not contours' in prompt['coverage_source_anchors']['rule']


def test_unrelated_or_absent_evidence_does_not_change_repair_context():
    repairs=[{'object_id':'target','issues':[]}]
    assert bind(repairs,[{'id':'target','bbox_2d':[0,0,20,20]}],{},[])==repairs


@pytest.mark.parametrize('mode',['pixel','norm1000'])
def test_coverage_goal_survives_rescope_without_original_coordinates_in_prompt(tmp_path,mode):
    context={'identity_descriptions':['Only the inner colored members']}
    repair={'source_extent_original':[20,20,180,60],
        'issues':[{'kind':'missing','part_ids':[],
            'evidence':'Grey members to the right of the colored row are absent.',
            'action':'Recover the grey members; preserve the enclosing capsule.'}]}
    before=deepcopy((context,repair))
    updated=repair_context(context,repair)
    job=prepare_candidate(Image.new('RGB',(220,100),'white'),
        {'id':'group','label':'old subgroup','kind':'compound','bbox_2d':[20,20,80,60]},
        [220,100],tmp_path/mode,input_context=updated,coordinate_mode=mode)
    prompt=make_prompt(job)
    assert prompt['coverage_requested_corrections']['requests'][0]['action']==repair['issues'][0]['action']
    assert 'coverage_source_extent_original' not in prompt.get('localization_context',{})
    assert 'coverage_source_requests' not in prompt.get('localization_context',{})
    assert (context,repair)==before
    from shape_integration.fitting.critic import source_observation_contract,validate_review
    critic=augment_prompt({},job)
    ids=source_observation_contract(critic,{'F1':'old_member'})
    assert ids==['coverage_request_0']
    assert critic['required_source_observations']['targets'][0]['request']==prompt['coverage_requested_corrections']['requests'][0]
    with pytest.raises(ValueError,match='missing or uncertain'):
        validate_review({'decision':'accept','issues':[],
            'source_observations':[{'id':ids[0],'source_relation':'missing','evidence':'The outer members remain absent.'}]},['F1'],ids)


def test_source_extent_carries_unrepresented_members_beyond_old_localization():
    repairs=[{'object_id':'group','source_extent_original':[20,20,180,60],'issues':[]}]
    existing=[{'id':'group','bbox_2d':[20,20,80,60]}]
    evidence={'shown':[{'id':'outside_old_box','bbox_original':[140,30,150,40]}]}
    result=bind(repairs,existing,evidence,[{'sam_id':'outside_old_box','action':'uncertain'}])
    assert result[0]['source_anchors'][0]['id']=='outside_old_box'


def test_false_coverage_alarm_can_be_dismissed_without_inventing_or_deleting_parts():
    from shape_integration.fitting.critic import validate_review
    response={'decision':'accept','issues':[],'source_observations':[
        {'id':'coverage_request_0','source_relation':'not_present',
         'evidence':'CLEAN shows an empty gap at the proposed missing member; no additional circle exists.'},
        {'id':'repeated_structure','source_relation':'supported',
         'evidence':'Every actually visible circle has a fitted counterpart.'}]}
    assert validate_review(response,['F1'],['coverage_request_0','repeated_structure'])==response
    response['source_observations'][1]['source_relation']='not_present'
    with pytest.raises(ValueError,match='only dismisses a disproven coverage request'):
        validate_review(response,['F1'],['coverage_request_0','repeated_structure'])


@pytest.mark.parametrize('mode',['pixel','norm1000'])
def test_missing_source_extent_cannot_be_clipped_by_old_crop_or_sam(tmp_path,mode):
    candidate={'id':'group','label':'old subgroup','kind':'compound','bbox_2d':[20,20,80,60]}
    context={'coverage_source_extent_original':[20,20,180,60]}
    class Evidence:
        def prepare(self,*args):return [10,10,90,70],None
    folder=tmp_path/'expanded'
    job=prepare_candidate(Image.new('RGB',(220,100),'white'),candidate,[220,100],folder,
        crop_bounds_original=[10,10,90,70],input_context=context,coordinate_mode=mode,spatial_evidence=Evidence())
    assert job['candidate']==candidate
    assert job['frame']['source_crop_half_open'][2]>=181
    scope=make_prompt(job)['coverage_source_scope']
    assert 'not a contour' in scope['rule']
    assert scope['bbox_2d'][2] <= (1000 if mode=='norm1000' else job['frame']['actual_crop_input_size_px'][0]-1)


def test_new_scope_prepares_fresh_frame_without_reusing_old_local_geometry(tmp_path):
    from shape_integration.common import read,save,sha
    from shape_integration.fitting.control import process_object
    original=tmp_path/'source.png';Image.new('RGB',(220,100),'white').save(original)
    candidate={'id':'group','label':'old subgroup','kind':'compound','bbox_2d':[20,20,80,60]}
    old=tmp_path/'old'
    job=prepare_candidate(Image.open(original),candidate,[220,100],old,crop_bounds_original=[10,10,90,70])
    hashes={name:sha(old/name) for name in [*job['input_sha256'],'job.json']}
    seed={'path':str(old),'sha256':hashes,'fit_reused':True,'fallback_fit_available':True}
    options={'crop_padding':.1,'crop_scale':3,'crop_min_long_edge':1080,'crop_max_long_edge':2048,
        'max_revisions':1,'max_context_rounds':1,'check_enabled':False}
    destination=tmp_path/'new'
    process_object(destination,candidate,{'original':str(original),'input':str(original),
        'source_size':[220,100],'input_size':[220,100]},tmp_path,None,options,
        namespace='test_scope',prepare_only=True,seed=seed,
        input_context={'coverage_source_extent_original':[20,20,180,60]})
    new=read(destination/'revision_00/job.json')
    assert new['frame']['source_crop_half_open'][2]>=181
    assert not (destination/'revision_00/reused_revision.json').exists()
    assert read(destination/'source_scope_seed.json')['fit_reused'] is False
    assert all(sha(old/name)==digest for name,digest in hashes.items())
