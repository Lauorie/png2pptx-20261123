from PIL import Image, ImageDraw
import pytest
from shape_integration.texts.ocr_refinement import tighten, validate_match, proposal, content_match, localize, select_match


def test_tightens_without_discarding_detached_dot():
    im = Image.new('RGB', (60,40), 'white'); d = ImageDraw.Draw(im)
    d.rectangle((20,15,25,30), fill='black'); d.rectangle((21,8,23,10), fill='black')
    rec, _ = tighten(im, [15,4,32,35])
    assert rec['bbox_2d'] == [20,8,25,30]


@pytest.mark.parametrize('fg,bg', [('black','white'),('white','#334477'),('#CC0000','#EEEEEE')])
def test_color_and_rotation_independent_bounds(fg,bg):
    im = Image.new('RGB', (60,60), bg); d = ImageDraw.Draw(im)
    d.rectangle((24,12,29,22), fill=fg); d.rectangle((24,29,29,40), fill=fg)
    rec, _ = tighten(im, [20,8,34,45])
    assert rec['bbox_2d'] == [24,12,29,40]


def test_crossing_graphic_or_clipped_glyph_is_deferred():
    im = Image.new('RGB', (60,40), 'white'); d = ImageDraw.Draw(im)
    d.rectangle((20,15,25,30), fill='black'); d.line((0,10,59,10), fill='black', width=2)
    rec, _ = tighten(im, [15,4,32,35])
    assert rec['status'] == 'uncertain'
    assert rec['bbox_2d'] is None


def test_blank_is_not_localized():
    rec, _ = tighten(Image.new('RGB',(60,40),'white'), [10,10,40,30])
    assert rec['reason'] == 'insufficient_source_ink'


def test_match_preserves_full_text_and_occurrence():
    row = {'text':'CA','bbox_2d':[20,20,30,40]}
    detection = {'text':'CA','confidence':.99,'source_bbox':[18,18,32,42]}
    validate_match(row, detection, (200,200))
    for bad in ({**detection,'text':'C'}, {**detection,'confidence':.1},
                {**detection,'source_bbox':[150,150,170,190]}):
        with pytest.raises(ValueError):
            validate_match(row, bad, (200,200))


def test_proposal_freezes_content_and_removes_stale_fit():
    row = {'text':'CA','bbox_2d':[1,2,3,4],'rotation_deg':-90,
           'runs':[{'text':'CA','color':'#000000'}],'layout':{},'font_size_px':80}
    out = proposal(row, [2,3,4,5], {'status':'located'})
    assert out['runs'] == row['runs'] and out['rotation_deg'] == -90
    assert 'layout' not in out and 'font_size_px' not in out
    assert row['font_size_px'] == 80


def test_explicit_strict_and_two_point_tolerances_override_five_point_default():
    from shape_integration.texts.bbox_color_review import nonregression_gate
    metric={'before':{'score':.60},'after':{'score':.58}}
    assert nonregression_gate(metric)[0]
    assert not nonregression_gate(metric,0.)[0]
    assert nonregression_gate(metric,.02)[0]
    assert not nonregression_gate({'before':{'score':.60},'after':{'score':.579}},.02)[0]


@pytest.mark.parametrize('target,observed', [('SS','ss'),('Modeling','Modelin'),('Alpha','Alpba'),('Beta','beta')])
def test_ocr_errors_are_localization_evidence_only(target,observed):
    assert content_match(target,observed)
    assert content_match(target,observed,False) is None


@pytest.mark.parametrize('target,observed', [('SS','S'),('CA','CB'),('Modeling','Model'),('A1','A2'),('Alpha','Gamma'),('','')])
def test_large_errors_short_labels_and_math_are_not_fuzzy(target,observed):
    assert content_match(target,observed) is None


def test_missing_suffix_search_restores_ink_not_ocr_transcription():
    im=Image.new('RGB',(120,80),'white');draw=ImageDraw.Draw(im)
    # Pale graphic crosses black glyphs: it must not join their support mask.
    draw.line((60,0,60,79),fill='#AACCEE',width=3)
    for x in range(20,100,10):draw.rectangle((x,25,x+5,40),fill='black')
    original={'text':'Modeling','bbox_2d':[15,20,99,45],'color':'#000000',
              'runs':[{'text':'Modeling','script':'normal','color':'#000000'}]}
    det={'text':'Modelin','source_bbox':[17,22,87,43],'confidence':.99,'index':0}
    rec,mask=localize(original,det,im)
    assert rec['status']=='located' and rec['bbox_2d']==[20,25,95,40]
    assert rec['bbox_2d'][2]>det['source_bbox'][2]
    result=proposal(original,rec['bbox_2d'],rec)
    assert result['text']=='Modeling' and result['runs']==original['runs']


def test_inexact_black_crossing_line_is_deferred():
    im=Image.new('RGB',(120,80),'white');draw=ImageDraw.Draw(im)
    draw.rectangle((30,25,85,40),fill='black');draw.line((60,0,60,79),fill='black',width=2)
    r={'text':'Modeling','bbox_2d':[15,20,99,45],'color':'#000000'}
    rec,_=localize(r,{'text':'Modelin','source_bbox':[17,22,87,43]},im)
    assert rec['status']=='uncertain' and rec['reason']=='color_support_touches_search_boundary'


def test_case_only_error_does_not_require_padded_detector_width():
    im=Image.new('RGB',(120,80),'white');draw=ImageDraw.Draw(im)
    draw.rectangle((45,25,52,38),fill='black');draw.rectangle((56,25,63,38),fill='black')
    r={'text':'SS','bbox_2d':[35,18,75,45],'color':'#000000'}
    rec,_=localize(r,{'text':'ss','source_bbox':[38,20,70,43]},im)
    assert rec['status']=='located' and rec['bbox_2d']==[45,25,63,38]


def test_nearby_ambiguous_labels_not_resolved_by_confidence():
    r={'text':'Modeling','bbox_2d':[10,10,90,40]}
    d={'text':'Modelin','source_bbox':[12,12,85,35],'confidence':.99,'index':0}
    assert select_match(r,[d],(200,200))==d
    with pytest.raises(ValueError):select_match(r,[d,dict(d,index=1,confidence=.90)],(200,200))
    with pytest.raises(ValueError):select_match(r,[d],(200,200),used={0})
    with pytest.raises(ValueError):validate_match(dict(r,kind='formula'),d,(200,200))
