from copy import deepcopy
import pytest
from PIL import Image, ImageDraw, ImageFont
from shape_integration.texts.contracts import recognition
from shape_integration.texts.ocr_refinement import select_match, localize, proposal
from shape_integration.texts.bbox_color_review import nonregression_gate


def formula(text,latex,base,script,kind):
    return {'text':text,'kind':'formula','latex':latex,'bbox_2d':[1,1,80,40],
            'runs':[{'text':base,'script':'normal'},{'text':script,'script':kind}]}


@pytest.mark.parametrize('r',[
    formula('ŷ+',r'\hat{y}^{+}','y','+','super'),
    formula('x',r'\hat{x}_{disambig}','x','disambig','sub'),
    formula(r'\hat{x}_a',r'\hat{x}_a','x̂','a','sub'),
])
def test_formula_reconciliation_is_structured_and_lossless(r):
    raw={'texts':[r]};before=deepcopy(raw)
    result=recognition(raw,(200,200))
    assert result['texts'][0]['latex']==r['latex']
    assert result['texts'][0]['text']==''.join(v['text'] for v in r['runs'])
    assert result['format_repairs'] and raw==before


@pytest.mark.parametrize('r',[
    formula('x',r'\hat{x}_a','x','b','sub'),
    formula('x',r'\hat{x}_a','y','a','sub'),
    formula('x',r'\hat{x}_a','x̃','a','sub'),
    formula('z',r'\hat{x}_a','x','a','sub'),
])
def test_conflicting_math_never_reconciled(r):
    with pytest.raises(ValueError):recognition({'texts':[r]},(200,200))


def test_auxiliary_region_one_pixel_boundary_only():
    raw={'texts':[],'regions':[{'bbox_2d':[10,10,200,100],'reason':'detail'}]}
    result=recognition(raw,(200,100))
    assert result['regions'][0]['bbox_2d']==[10,10,199,99]
    assert raw['regions'][0]['bbox_2d']==[10,10,200,100]
    raw['regions'][0]['bbox_2d'][2]=202
    with pytest.raises(ValueError):recognition(raw,(200,100))


def test_displaced_exact_anchor_without_matching_far_repeat():
    row={'text':'0.3','bbox_2d':[300,165,355,199]}
    d={'index':0,'text':'0.3','confidence':.99,'source_bbox':[309,106,358,141]}
    assert select_match(row,[d],(1000,600))==d
    with pytest.raises(ValueError):select_match(row,[dict(d,source_bbox=[700,400,749,435])],(1000,600))
    with pytest.raises(ValueError):select_match(row,[d,dict(d,index=1,source_bbox=[309,220,358,255])],(1000,600))


def test_same_line_fragments_combine_without_changing_text():
    r={'text':'BCE loss','bbox_2d':[100,100,220,125]}
    ds=[{'index':1,'text':'BCE','confidence':.99,'source_bbox':[102,102,155,123]},
        {'index':2,'text':'loss','confidence':.99,'source_bbox':[157,102,210,123]}]
    result=select_match(r,ds,(500,500))
    assert result['indices']==[1,2]
    with pytest.raises(ValueError):select_match(r,ds,(500,500),used={2})
    with pytest.raises(ValueError):select_match(r,[ds[0],dict(ds[1],source_bbox=[157,180,210,201])],(500,500))


def test_formula_ocr_syntax_is_only_association():
    r=formula('S3','S_3','S','3','sub');r['bbox_2d']=[100,100,150,140]
    d={'index':0,'text':'$S_{3}$','confidence':.62,'source_bbox':[101,101,149,139]}
    assert select_match(r,[d],(500,500))==d
    with pytest.raises(ValueError):select_match(r,[dict(d,text='$S_{2}$')],(500,500))


def test_zero_and_unreliable_scores_cannot_pass():
    assert not nonregression_gate({'before':{'score':0},'after':{'score':0}},.02)[0]
    assert not nonregression_gate({'before':{'score':.7},'after':{'score':.8},'reliable':False},.02)[0]
    assert nonregression_gate({'before':{'score':.7},'after':{'score':.69}},.02)[0]


def test_wrong_white_text_can_recover_black_only_with_glyph_evidence():
    from shape_integration.texts.layout import resolve_font
    im=Image.new('RGB',(260,100),'#AABBCD')
    font=ImageFont.truetype(resolve_font('Arial')['path'],30)
    ImageDraw.Draw(im).text((25,20),'Micro',font=font,fill='black')
    r={'text':'Micro','font':'Arial','kind':'text','color':'#FFFFFF','bbox_2d':[20,20,120,60],
       'runs':[{'text':'Micro','script':'normal','color':'#FFFFFF'}]}
    d={'index':0,'text':'Micro','confidence':.99,'source_bbox':[21,22,120,58]}
    rec,_=localize(r,d,im)
    assert rec['status']=='located' and rec['proposed_color']=='#000000'
    out=proposal(r,rec['bbox_2d'],rec)
    assert out['text']=='Micro' and out['runs'][0]['text']=='Micro' and out['color']=='#000000'
    assert r['color']=='#FFFFFF'
