from copy import deepcopy
import pytest
from shape_integration.common import save
from shape_integration.fitting.critic import validate_review,review_schema
from shape_integration.fitting.critic_retry import retry_builder


def raw(extra):
    return {'decision':'accept','issues':[],'source_observations':[
        {'id':'connectivity','source_relation':'supported','evidence':'The visible source route meets both endpoints.',
         'evidence_note':extra}]}


@pytest.mark.parametrize('extra',['',None,[],{}])
def test_empty_nested_metadata_uses_the_existing_format_retry_without_new_visual_decision(tmp_path,extra):
    value=raw(extra);save(tmp_path/'parsed_response.json',value)
    validator=lambda r:validate_review(r,['F1'],['connectivity'])
    prompt={'output_schema':review_schema(['F1'],['connectivity'])}
    retry=retry_builder('visual',prompt,[('source','not-opened')],validator)(tmp_path,'unexpected field')
    assert not retry['images']
    expected=deepcopy(value);expected['source_observations'][0].pop('evidence_note')
    assert retry['validator'](expected)==expected
    changed=deepcopy(expected);changed['source_observations'][0]['evidence']='Different route evidence.'
    with pytest.raises(ValueError,match='format repair changed'):retry['validator'](changed)


def test_nonempty_nested_note_or_ambiguity_keeps_source_images(tmp_path):
    value=raw('An extra source route may be missing.');save(tmp_path/'parsed_response.json',value)
    validator=lambda r:validate_review(r,['F1'],['connectivity'])
    images=[('source','same source image')]
    retry=retry_builder('visual',{'output_schema':review_schema(['F1'],['connectivity'])},images,validator)(tmp_path,'unexpected field')
    assert retry['images']==images
