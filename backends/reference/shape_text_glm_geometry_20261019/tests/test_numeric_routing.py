from copy import deepcopy
from zipfile import ZipFile

import pytest
from lxml import etree

from shape_integration.texts import options
from shape_integration.texts.numeric_routing import route, inventory
from shape_integration.texts.staged_formula import inventory_validate
from shape_integration.texts.local_content import validate


def number(text='3',latex=None):
    return dict(id='T0042',kind='formula',text=text,latex=text if latex is None else latex,
        runs=[{'text':text,'script':'normal','color':'#123456'}],
        bbox_2d=[10,10,40,50],rotation_deg=90,font='Arial',bold=True,
        italic=False,underline=True,color='#123456',legibility='clear',visual_line_count=1,
        formula_status='resolved',line_layout_status='native_formula')


@pytest.mark.parametrize('text',['0','3','5','35','001','3.14','-2','−2','+2','30%','.5'])
def test_whole_baseline_number_is_text(text):
    r=number(text);before=deepcopy(r);actual=route(r,'test')
    assert actual['kind']=='text' and actual['latex']==''
    assert actual['routing_audit'][-1]['before']['latex']==text
    assert r==before
    assert route(actual,'again')==actual


def test_preserve_all_content_geometry_style_and_run_colors():
    r=number('35');r['runs']=[{'text':'3','script':'normal','color':'#123456'},
                            {'text':'5','script':'normal','color':'#AB00EF'}]
    result=route(r,'test')
    changed={'kind','latex','formula_status','line_layout_status','routing_audit'}
    assert {k:v for k,v in result.items() if k not in changed}=={k:v for k,v in r.items() if k not in changed}


@pytest.mark.parametrize('text,latex',[
    ('x3','x^3'),('3',r'\sqrt{3}'),('13',r'\frac{1}{3}'),
    ('3',r'\hat{3}'),('3',r'\tilde{3}'),('3','3_1'),('3','3^{2}'),
    ('3',r'\begin{matrix}3\end{matrix}'),('3',r'\unknown{3}'),
    ('3',r'\mathbf{3}'),('3','5'),('x','x'),('α',r'\alpha'),
    ('³','3'),('3 5','3 5'),(' 3 ','3'),('1e3','1e3'),('3+5','3+5'),
])
def test_structure_or_uncertain_syntax_is_never_flattened(text,latex):
    r=number(text,latex)
    assert route(r,'test')==r


@pytest.mark.parametrize('script',['sub','super'])
def test_runs_scripts_protect_even_if_flat_text_and_latex_are_digits(script):
    r=number('13');r['runs']=[{'text':'1','script':'normal'},{'text':'3','script':script}]
    assert route(r,'test')==r


@pytest.mark.parametrize('change',[{'formula_parent_id':'EQ1'},{'part_of_formula':True},
    {'visual_line_count':2},{'legibility':'partial'},
    {'runs':[{'text':'5','script':'normal'}]}])
def test_incomplete_or_nonmatching_occurrence_is_preserved(change):
    r={**number(),**change};assert route(r,'test')==r


def test_inventory_guard_precedes_formula_requests():
    r=number(latex='')
    result=inventory_validate({'texts':[r]},[100,100],options({}))['texts'][0]
    assert result['kind']=='text' and result['formula_status']=='not_applicable'
    for field in ('bbox_2d','font','rotation_deg','runs','bold','italic','underline','color'):
        assert result[field]==r[field]


def test_cached_inventory_is_routed_without_mutating_evidence():
    old={'texts':[number(),number('x3','x^3')],'source_sha256':'known'};before=deepcopy(old)
    result=inventory(old,'cache')
    assert [r['kind'] for r in result['texts']]==['text','formula']
    assert old==before and result['source_sha256']=='known'


def test_stage_one_does_not_erase_multiline_evidence_before_numeric_guard():
    r=number('35',latex='');r['visual_line_count']=2
    result=inventory_validate({'texts':[r]},[100,100],options({}))['texts'][0]
    assert result['kind']=='formula' and result['visual_line_count']==2


def test_crop_cannot_restore_formula_for_plain_number():
    r=number();prior=route(r,'global')
    proposal={**r,'status':'located','direction_status':'confirmed','reason':'Visible numeric node label'}
    result=validate({'items':[proposal]},[prior],[100,100])['items'][0]
    assert result['kind']=='text' and result['latex']==''
    assert result['routing_audit'][-1]['stage']=='crop_reply_before_formula'
    for field in ('bbox_2d','font','rotation_deg','runs','bold','italic','underline','color'):
        assert result[field]==r[field]


def test_crop_keeps_entire_radical_even_with_digit_display_hint():
    r=number();proposal={**r,'latex':r'\sqrt{3}','status':'located',
        'direction_status':'confirmed','reason':'Visible radical'}
    result=validate({'items':[proposal]},[r],[100,100])['items'][0]
    assert result['kind']=='formula' and result['latex']==r'\sqrt{3}'


def test_routed_number_exports_native_text_with_original_style(tmp_path):
    from shape_integration.texts.export import write
    r=route(number(),'export_test')
    r.update(frame_px=[10,10,30,40],font_size_px=20,resolved_font='Arial')
    path=tmp_path/'number.pptx'
    counts=write([r],[100,100],path)
    assert counts['textboxes']==1 and counts['native_equations']==0
    with ZipFile(path) as archive:tree=etree.fromstring(archive.read('ppt/slides/slide1.xml'))
    assert not tree.xpath('//*[local-name()="oMath"]')
    assert tree.xpath('//*[local-name()="t"]/text()')==['3']
    assert tree.xpath('//*[local-name()="rPr"]/@b')==['1']
    assert 'Arial' in tree.xpath('//*[local-name()="latin"]/@typeface')
    assert '123456' in tree.xpath('//*[local-name()="srgbClr"]/@val')
