import pytest
from shape_integration.texts.prompt_protocol import recognize, review, RECOVERY_SCOPE
from shape_integration.texts.local_content import validate


def unit(text,y):
    return dict(text=text,kind='text',latex='',runs=[{'text':text,'script':'normal'}],
                bbox_2d=[10,y,120,y+20],rotation_deg=0,direction_status='confirmed',
                visual_line_count=1,reason='Visible baseline',font='Arial',color='#000000')


def target():
    return dict(id='T1',text='One Two',kind='text',latex='',runs=[{'text':'One Two','script':'normal'}],
                bbox_2d=[10,10,120,70],rotation_deg=0,font='Arial',color='#000000',bold=False,italic=False)


def test_both_stages_share_one_policy():
    for prompt in (recognize([200,100]),review([200,100],[target()],[])):
        assert all(rule in prompt['rules'] for rule in RECOVERY_SCOPE)
        assert prompt['coordinates']['x_range']==[0,199]
    assert 'Keep existing colors' not in str(review([200,100],[target()],[]))


def test_ellipsis_policy_reaches_global_and_crop_inventory_and_review():
    from shape_integration.texts.staged_formula import inventory_prompt
    policy=next(r for r in RECOVERY_SCOPE if r.startswith('ELLIPSIS:'))
    prompts=[recognize([200,100]), recognize([200,100],local=True),
             review([200,100],[target()],[]),
             inventory_prompt([200,100],{'max_regions':8}),
             inventory_prompt([200,100],{'max_regions':8},local=True)]
    for prompt in prompts:
        assert policy in prompt['rules']
    assert 'ONE text unit' in policy
    assert 'dashed borders' in policy
    assert 'remain within that formula' in policy


@pytest.mark.parametrize('text',['...', '…', '⋯', '⋮', '⋱', '⋰'])
def test_standalone_ellipsis_survives_recognition_and_local_review(text):
    from shape_integration.texts.contracts import recognition
    proposal=unit(text,10)
    checked=recognition({'texts':[proposal]},[200,100])['texts'][0]
    assert checked['kind']=='text'
    assert checked['text']==text
    assert checked['visual_line_count']==1
    assert checked['background_fill'] is None
    reviewed=validate({'items':[{'id':'T1','status':'resolved','units':[proposal]}]},
                      [target()],[200,100])
    assert reviewed['items'][0]['text']==text


def test_review_omits_redundant_text_and_allows_latest_content_colors():
    prompt=review([200,100],[target()],[])
    assert 'text' not in prompt['response_format']['items'][0]['units'][0]
    assert 'latest valid content and colors' in ' '.join(prompt['rules'])
    assert 'omit text when it is absent from the response schema' in ' '.join(prompt['rules'])


def test_uniform_units_roundtrip_for_single_and_multiline():
    for units in ([unit('One',10)],[unit('One',10),unit('Two',45)]):
        raw={'items':[{'id':'T1','status':'resolved','reason':'Visible text','units':units}]}
        result=validate(raw,[target()],[200,100])
        assert validate(result,[target()],[200,100])==result
        assert result['items'][0]['status']==('located' if len(units)==1 else 'split')


def test_wrapped_paragraph_cannot_be_hidden_in_single_unit():
    proposal=unit('One Two',10);proposal['visual_line_count']=2
    with pytest.raises(ValueError,match='baselines'):
        validate({'items':[{'id':'T1','status':'resolved','units':[proposal]}]},[target()],[200,100])


def test_exclusion_requires_evidence_even_in_uniform_schema():
    raw={'items':[{'id':'T1','status':'excluded_image_text','units':[],'reason':'Photographed sign'}]}
    assert validate(raw,[target()],[200,100])['items'][0]['bbox_2d'] is None
    raw['items'][0]['reason']=''
    with pytest.raises(ValueError,match='evidence'):validate(raw,[target()],[200,100])


def test_numeric_strings_normalize_but_invalid_geometry_still_fails():
    proposal=unit('One',10);proposal['bbox_2d']=['10','10','120','30'];proposal['rotation_deg']='0'
    raw={'items':[{'id':'T1','status':'resolved','units':[proposal]}]}
    value=validate(raw,[target()],[200,100])['items'][0]
    assert value['bbox_2d']==[10,10,120,30]
    proposal['bbox_2d'][0]='NaN'
    with pytest.raises(ValueError):validate(raw,[target()],[200,100])
    proposal['bbox_2d'][0]='-10'
    with pytest.raises(ValueError):validate(raw,[target()],[200,100])
