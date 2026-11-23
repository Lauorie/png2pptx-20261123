from copy import deepcopy
from zipfile import ZipFile

from lxml import etree
from PIL import Image
import pytest
from pptx import Presentation
from pptx.oxml.xmlchemy import OxmlElement

from shape_integration.texts import contracts,export
from shape_integration.texts.equations import NS
from shape_integration.texts.local_content import validate
from shape_integration.texts.prompt_protocol import RECOVERY_SCOPE,recognize,review,format_examples


def text_row(text):
    return {'id':'T1','text':text,'kind':'text','latex':'',
            'runs':[{'text':text,'script':'normal','color':'#000000'}],
            'bbox_2d':[10,10,190,40],'frame_px':[10,10,180,30],
            'font':'Arial','font_size_px':20,'resolved_font':'Arial','color':'#000000',
            'bold':False,'italic':False,'underline':False,'rotation_deg':0,
            'legibility':'clear','visual_line_count':1,
            'direction_status':'confirmed','reason':'Visible source baseline'}


def test_bullet_policy_reaches_global_and_glm_crop_prompts():
    from shape_integration.texts.staged_formula import inventory_prompt
    rule=next(r for r in RECOVERY_SCOPE if r.startswith('BULLETS:'))
    prompts=[recognize([200,100]),recognize([200,100],local=True),
             review([200,100],[text_row('Feature')],[]),
             inventory_prompt([200,100],{'max_regions':8}),
             inventory_prompt([200,100],{'max_regions':8},local=True)]
    assert all(rule in p['rules'] for p in prompts)
    assert 'ONE text unit' in rule and 'multiplication dots' in rule
    assert any(r['text']=='· Feature extraction' for r in format_examples([200,100],'Arial')['texts'])


@pytest.mark.parametrize('text',['·','•','· Feature extraction','• Second item','No bullet'])
def test_literal_bullet_content_survives_recognition_and_glm_review(text):
    row=text_row(text);before=deepcopy(row)
    checked=contracts.recognition({'texts':[row]},[200,100])['texts'][0]
    assert checked['text']==text and checked['kind']=='text' and checked['latex']==''
    result=validate({'items':[{'id':'T1','status':'resolved','units':[row]}]},[row],[200,100])
    assert result['items'][0]['text']==text
    assert row==before


@pytest.mark.parametrize('text',['·','•','· Feature extraction','No bullet'])
def test_ppt_export_preserves_literal_markers_and_explicitly_disables_auto_bullets(tmp_path,text):
    target=tmp_path/'text.pptx';row=text_row(text);before=deepcopy(row)
    export.write([row],[240,100],target)
    with ZipFile(target) as archive:tree=etree.fromstring(archive.read('ppt/slides/slide1.xml'))
    assert tree.xpath('.//p:sp/p:txBody/a:p/a:r/a:t/text()',namespaces=NS)==[text]
    props=tree.find('.//p:sp/p:txBody/a:p/a:pPr',NS)
    assert props.find('a:buNone',NS) is not None
    assert props.find('a:buChar',NS) is None
    assert props.find('a:buAutoNum',NS) is None
    assert props.get('marL')=='0' and props.get('indent')=='0'
    assert row==before


def test_disabling_inherited_bullets_is_idempotent_and_preserves_fonts():
    prs=Presentation();slide=prs.slides.add_slide(prs.slide_layouts[6])
    paragraph=slide.shapes.add_textbox(0,0,1000,1000).text_frame.paragraphs[0]
    paragraph.font.name='Arial'
    props=paragraph._p.get_or_add_pPr()
    bullet=OxmlElement('a:buChar');bullet.set('char','•');props.insert(0,bullet)
    export.disable_auto_bullets(paragraph)
    once=etree.tostring(paragraph._p)
    export.disable_auto_bullets(paragraph)
    assert etree.tostring(paragraph._p)==once
    assert paragraph.font.name=='Arial'
    assert len(props.findall('a:buNone',NS))==1


def test_native_equations_are_not_flattened_by_no_bullet_export(tmp_path):
    asset=tmp_path/'measurement.png';Image.new('RGBA',(100,30),(0,0,0,255)).save(asset)
    row=text_row('ab');row.update(kind='formula',latex=r'a\cdot b',
                                 layout={'measurement':{'equation_preview':str(asset)}})
    target=tmp_path/'formula.pptx';export.write([row],[240,100],target)
    with ZipFile(target) as archive:tree=etree.fromstring(archive.read('ppt/slides/slide1.xml'))
    assert tree.find('.//mc:Choice//a14:m',NS) is not None
    text=''.join(tree.xpath('.//m:t/text()',namespaces=NS))
    assert any(dot in text for dot in ('·','⋅','∙'))
