from copy import deepcopy

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest

from shape_integration.texts.ocr_geometry import proposals, select, iou
from shape_integration.texts.ocr_refinement import proposal


def fixture(tmp_path):
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',28)
    template=Image.new('L',(140,60));ImageDraw.Draw(template).text((3,2),'Mask',font=font,fill=255)
    template=template.crop(template.getbbox())
    rgba=Image.new('RGBA',template.size);rgba.putalpha(template)
    rgba.save(tmp_path/'template.png')
    source=Image.new('RGB',(240,140),'white');source.paste((0,0,0),(70,50),template)
    w,h=template.size
    r={'id':'T1','text':'Mask','kind':'text','latex':'','color':'#000000',
       'font':'Arial','rotation_deg':0,'runs':[{'text':'Mask','script':'normal','color':'#000000'}],
       'bbox_2d':[67,48,73+w,53+h], 'layout':{'measurement':{'ink_template':str(tmp_path/'template.png')}}}
    d={'index':0,'text':'nonsense','confidence':.01,'rotation':0,'source_bbox':[67,48,73+w,53+h]}
    return source,r,d,[70,50,69+w,49+h]


def test_bad_ocr_content_and_confidence_are_not_a_gate(tmp_path):
    source,r,d,truth=fixture(tmp_path);before=deepcopy(r)
    result=select(r,[d],source)
    assert iou(result['bbox_2d'],truth)>.85
    assert result['template_score']>.85 and r==before
    out=proposal(r,result['bbox_2d'],result)
    assert out['text']==r['text'] and out['runs']==r['runs']


@pytest.mark.parametrize('score,accepted',[(.499,False),(.50,True),(.60,True)])
def test_template_floor_is_half_in_both_selection_layers(tmp_path,monkeypatch,score,accepted):
    from shape_integration.texts import registration
    source,row,_,truth=fixture(tmp_path)
    row['bbox_2d']=truth  # Isolate score gating from oversized-box geometry guards.
    real_match=registration.best_match
    def controlled_match(ink,template):
        result=real_match(ink,template)
        return None if result is None else (score,result[1],result[2])
    monkeypatch.setattr(registration,'best_match',controlled_match)
    if not accepted:
        with pytest.raises(ValueError,match='low_score'):select(row,[],source)
    else:
        result=select(row,[],source)
        assert result['template_score']==score and result['template_min_score']==.50


@pytest.mark.parametrize('kind',['text','formula'])
def test_inner_template_floor_override_does_not_change_other_callers(tmp_path,monkeypatch,kind):
    from shape_integration.texts import registration
    source,row,_,truth=fixture(tmp_path);row['kind']=kind;row['bbox_2d']=truth
    metric=row['layout']['measurement'];real_match=registration.best_match
    def controlled_match(ink,template):
        result=real_match(ink,template)
        return None if result is None else (.50,result[1],result[2])
    monkeypatch.setattr(registration,'best_match',controlled_match)
    assert registration._register_at(row,metric,source)[2]['status']=='low_score'
    assert registration._register_at(row,metric,source,min_score=.50)[2]['status']=='accepted'


def test_overlapping_base_and_script_boxes_form_candidate():
    r={'bbox_2d':[289,316,324,360],'kind':'formula','rotation_deg':0}
    ds=[{'index':0,'text':'wrong','confidence':.1,'source_bbox':[289,316,318,351]},
        {'index':1,'text':'?','confidence':.1,'source_bbox':[311,337,322,358]}]
    ps=proposals(r,ds,[756,525])
    assert any(p['indices']==[0,1] for p in ps)


def test_missing_detector_has_explicit_prior_fallback(tmp_path):
    source,r,_,truth=fixture(tmp_path)
    result=select(r,[],source)
    assert result['proposal_type']=='vlm_prior_fallback'
    assert result['detection_indices']==[]
    assert iou(result['bbox_2d'],truth)>.85


def test_blank_source_cannot_pass_by_bad_ocr_box(tmp_path):
    source,r,d,_=fixture(tmp_path)
    with pytest.raises(ValueError):select(r,[d],Image.new('RGB',source.size,'white'))


def test_multicolor_runs_not_flattened_by_geometry(tmp_path):
    source,r,d,_=fixture(tmp_path)
    r['runs']=[{'text':'Ma','script':'normal','color':'#EE2200'},
               {'text':'sk','script':'normal','color':'#2255CC'}]
    record=select(r,[d],source)
    assert 'proposed_color' not in record
    assert proposal(r,record['bbox_2d'],record)['runs']==r['runs']
    with pytest.raises(ValueError):proposal(r,record['bbox_2d'],{'proposed_color':'#000000'})


def test_formula_uniform_explicit_color_can_be_corrected_not_multicolor():
    r={'text':'S1','kind':'formula','latex':r'\textcolor{#A8A8A8}{S_1}',
       'color':'#A8A8A8','runs':[{'text':'S1','script':'normal','color':'#A8A8A8'}]}
    out=proposal(r,[1,1,20,20],{'proposed_color':'#000000'})
    assert out['latex']==r'\textcolor{#000000}{S_1}'
    assert r['color']=='#A8A8A8'
    r['latex']=r'\textcolor{#A8A8A8}{S}_{\textcolor{#CC0000}{1}}'
    with pytest.raises(ValueError):proposal(r,[1,1,20,20],{'proposed_color':'#000000'})
