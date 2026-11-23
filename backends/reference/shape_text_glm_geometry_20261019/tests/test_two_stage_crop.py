from copy import deepcopy
import pytest
from shape_integration.texts.two_stage_crop import (
    validate_geometry, validate_attributes, geometry_prompt, attribute_prompt)
from shape_integration.texts.background_selection import select


def original():
    return {'id':'T0001','text':'a','kind':'text','latex':'',
        'runs':[{'text':'a','script':'normal','color':'#000000'}],
        'bbox_2d':[1,1,20,30],'font':'Times New Roman','color':'#000000',
        'bold':False,'italic':False,'underline':False,'rotation_deg':0,'legibility':'clear','replaces':[]}


def test_geometry_accepts_bbox_only_and_discards_legacy_fill():
    raw={'items':[{'id':'T0001','status':'resolved','units':[
        {'bbox_2d':[1,1,10,10]},
        {'bbox_2d':[1,12,10,20],'background_fill':'#ffffff','background_evidence':'Separate white knockout interrupts a line'}]}]}
    result=validate_geometry(raw,['T0001'],[100,100])
    assert len(result['items'][0]['units'])==2
    assert all(u['background_fill'] is None for u in result['items'][0]['units'])


@pytest.mark.parametrize('unit',[
    {'bbox_2d':[1,1,101,20],'background_fill':None},
    {'bbox_2d':[10,1,1,20]},
    {},
])
def test_geometry_rejects_invalid_boxes(unit):
    with pytest.raises(ValueError):
        validate_geometry({'items':[{'id':'T0001','status':'resolved','units':[unit]}]},['T0001'],[100,100])


def test_attribute_stage_cannot_override_geometry_or_paint():
    source=original();unit=deepcopy(source)
    unit.pop('id');unit.update(bbox_2d=[80,80,99,99],background_fill='#FF0000',
        direction_status='confirmed',visual_line_count=1,reason='Visible serif letter')
    locked={'T0001':{'bbox_2d':[2,3,15,25],'background_fill':None,'background_evidence':''}}
    result=validate_attributes({'items':[{'id':'T0001','status':'resolved','units':[unit]}]},[source],locked,[100,100])
    assert result['items'][0]['bbox_2d']==[2,3,15,25]
    assert result['items'][0]['background_fill'] is None
    assert result['items'][0]['text']=='a'


def test_stage_two_does_not_allow_resplitting_locked_unit():
    source=original()
    with pytest.raises(ValueError,match='exactly one unit'):
        validate_attributes({'items':[{'id':'T0001','status':'resolved','units':[source,source]}]},
                            [source],{'T0001':{'bbox_2d':[2,3,15,25],'background_fill':None}},[100,100])


@pytest.mark.parametrize('old_angle,new_angle',[(90,0),(0,45),(0,90),(0,180),(0,-45),(90,-90),(0,30)])
def test_attribute_stage_corrects_rotation_without_changing_locked_bbox(old_angle,new_angle):
    source=original();source['rotation_deg']=old_angle
    unit=deepcopy(source);unit.pop('id')
    unit.update(rotation_deg=new_angle,direction_status='confirmed',visual_line_count=1,
                reason='Observed glyph orientation',background_fill='#FFFFFF')
    locked={'T0001':{'bbox_2d':[2,3,15,25],'background_fill':None,'background_evidence':''}}
    result=validate_attributes({'items':[{'id':'T0001','status':'resolved','units':[unit]}]},
                               [source],locked,[100,100])['items'][0]
    assert result['rotation_deg']==(new_angle+180)%360-180
    assert result['bbox_2d']==locked['T0001']['bbox_2d']
    assert result['background_fill'] is None


def test_background_export_preserves_model_fill_without_source_or_cpu(tmp_path):
    rows=[{'id':'T0001','background_fill':'#aabbcc','background_evidence':'Visible opaque plate'}]
    selected=select(rows,'nonexistent.png',tmp_path,None,{'background_policy':'vlm_two_stage'})
    assert selected[0]['background_fill']=='#AABBCC'
    assert selected[0]['background_selection']['cpu_selection'] is False
    assert rows[0]['background_fill']=='#aabbcc'


def test_transparent_export_clears_legacy_and_fallback_fills_without_cpu(tmp_path):
    rows=[{'id':'T0001','background_fill':'#FFFFFF','background_evidence':'Old model fill'}]
    selected=select(rows,'nonexistent.png',tmp_path,None,{'background_policy':'transparent_only'})
    assert selected[0]['background_fill'] is None
    assert selected[0]['background_evidence']==''
    assert selected[0]['background_selection']['cpu_selection'] is False
    assert rows[0]['background_fill']=='#FFFFFF'


def test_prompts_separate_bbox_only_from_attributes():
    first=geometry_prompt([200,100],[],[])
    assert set(first['response_format']['items'][0]['units'][0])=={'bbox_2d'}
    assert 'background_fill' not in str(first)
    assert 'knockout' not in str(first)
    second=attribute_prompt([200,100],[],[])
    assert 'bbox_2d' not in second['response_format']['items'][0]['units'][0]
    assert 'background_fill' not in second['response_format']['items'][0]['units'][0]
    assert not any(r.startswith('### R21 ') for r in second['rules'])
    assert 'rotation_deg' in second['response_format']['items'][0]['units'][0]
    assert any('ROTATION IS NOT LOCKED' in r for r in second['rules'])
    assert any('0, 45, 90, 180, -45, -90' in r for r in second['rules'])


def test_all_recognition_and_review_schemas_exclude_background():
    from shape_integration.texts.prompt_protocol import recognize,review
    for prompt in (recognize([200,100]),recognize([200,100],local=True),
                   review([200,100],[],[]),geometry_prompt([200,100],[],[]),
                   attribute_prompt([200,100],[],[])):
        assert 'background_fill' not in str(prompt['response_format'])
        assert 'background_evidence' not in str(prompt['response_format'])
