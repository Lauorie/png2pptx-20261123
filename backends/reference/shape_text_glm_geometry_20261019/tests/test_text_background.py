import threading
from copy import deepcopy
from zipfile import ZipFile

import numpy as np
import pytest
from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR
from pptx.util import Inches

from shape_integration.texts import contracts, export, layout
from shape_integration.texts.equations import compatibility_copy
from shape_integration.texts.local_content import validate as validate_crop
from shape_integration.texts.prompt_protocol import recognize, review
from shape_integration.styling.export import render_native

NS={'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
    'mc':'http://schemas.openxmlformats.org/markup-compatibility/2006'}


def row(**kw):
    return dict(dict(id='T1',text='Label',kind='text',latex='',font='Arial',
        color='#000000',bold=False,italic=False,underline=False,rotation_deg=0,
        runs=[{'text':'Label','script':'normal'}],bbox_2d=[100,50,230,90],
        frame_px=[100,50,160,60],font_size_px=24,resolved_font='Arial',
        legibility='clear',visual_line_count=1),**kw)


def slide_xml(path):
    with ZipFile(path) as z:return etree.fromstring(z.read('ppt/slides/slide1.xml'))


def test_old_cache_remains_transparent_and_hex_normalized():
    assert contracts.recognition({'texts':[row()]},[400,200])['texts'][0]['background_fill'] is None
    parsed=contracts.recognition({'texts':[row(background_fill='#aabbcc',background_evidence='Visible backing')]},[400,200])
    assert parsed['texts'][0]['background_fill'] is None


@pytest.mark.parametrize('fill',[False,1,'white','transparent','#FFFF',[],{}])
def test_invalid_background_rejected(fill):
    from shape_integration.texts.background import validate
    with pytest.raises(ValueError,match='background_fill'):
        validate(row(background_fill=fill))
    assert contracts.recognition({'texts':[row(background_fill=fill)]},[400,200])['texts'][0]['background_fill'] is None


def test_solid_background_requires_evidence():
    from shape_integration.texts.background import validate
    with pytest.raises(ValueError,match='background_evidence'):
        validate(row(background_fill='#FFFFFF'))


def test_crop_cannot_control_fill():
    old=row(background_fill='#FFFFFF',background_evidence='Dashed border stops around caption')
    for fields,wanted in [({},'#FFFFFF'),({'background_fill':None,'background_evidence':''},None),
                          ({'background_fill':'#ffeecc','background_evidence':'Visible label panel'},'#FFEECC')]:
        unit={**row(),**fields,'direction_status':'confirmed','reason':'Local source evidence'}
        checked=validate_crop({'items':[{'id':'T1','status':'resolved','units':[unit]}]},[old],[400,200])
        assert 'background_fill' not in checked['items'][0]
        assert 'background_evidence' not in checked['items'][0]


def test_native_fill_and_transparency_are_textbox_properties(tmp_path):
    rows=[row(background_fill='#FFFFFF',background_evidence='Line interrupted behind label'),
          row(id='T2',frame_px=[100,120,160,60])]
    path=tmp_path/'text.pptx';export.write(rows,[400,200],path)
    xml=slide_xml(path)
    assert xml.xpath('.//p:sp[p:nvSpPr/p:cNvPr[@name="TXT T1"]]/p:spPr/a:solidFill/a:srgbClr/@val',namespaces=NS)==['FFFFFF']
    assert xml.xpath('.//p:sp[p:nvSpPr/p:cNvPr[@name="TXT T2"]]/p:spPr/a:noFill',namespaces=NS)
    assert len(xml.xpath('.//p:sp',namespaces=NS))==2


def test_formula_background_survives_compatibility_branch(tmp_path):
    preview=tmp_path/'math.png';Image.new('RGBA',(20,20),(0,0,0,0)).save(preview)
    r=row(kind='formula',text='x',latex='x',runs=[{'text':'x','script':'normal'}],
          background_fill='#FFFFFF',background_evidence='Occludes connector',
          layout={'measurement':{'equation_preview':str(preview)}})
    native=tmp_path/'math.pptx';export.write([r],[400,200],native)
    xml=slide_xml(native)
    assert xml.xpath('.//mc:Choice/p:sp/p:spPr/a:solidFill/a:srgbClr/@val',namespaces=NS)==['FFFFFF']
    assert xml.xpath('.//mc:Fallback/p:sp/p:spPr/a:solidFill/a:srgbClr/@val',namespaces=NS)==['FFFFFF']
    compat=tmp_path/'compat.pptx';assert compatibility_copy(native,compat)==1
    names=slide_xml(compat).xpath('.//p:cNvPr/@name',namespaces=NS)
    assert names.index('MATH_BACKGROUND T1')<names.index('MATH_PREVIEW T1')


def test_fill_excluded_from_font_atlas(tmp_path):
    rows=[row(background_fill='#00AA55',background_evidence='Visible backing')]
    before=deepcopy(rows)
    metrics=layout.native_metrics(rows,tmp_path,threading.BoundedSemaphore(1),{'render_timeout_seconds':180})
    assert rows==before and metrics['T1']['method']=='native_ppt_atlas'
    assert not slide_xml(tmp_path/'metrics.pptx').xpath('.//a:solidFill/a:srgbClr[@val="00AA55"]',namespaces=NS)
    rgb=np.asarray(Image.open(tmp_path/'native.png').convert('RGB'))
    assert np.max(rgb.max(axis=2).astype(int)-rgb.min(axis=2).astype(int))==0


def test_real_fill_masks_line_without_modifying_base(tmp_path):
    from shape_integration.texts.background_selection import select
    prs=Presentation();prs.slide_width=Inches(10);prs.slide_height=Inches(5)
    slide=prs.slides.add_slide(prs.slide_layouts[6]);scale=prs.slide_width/400
    line=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,int(20*scale),int(80*scale),int(380*scale),int(80*scale))
    line.line.color.rgb=RGBColor(255,0,0);line.line.width=int(3*scale)
    base=tmp_path/'base.pptx';prs.save(base)
    values=[]
    for label,fill in [('transparent',None),('white','#FFFFFF')]:
        target=tmp_path/(label+'.pptx')
        result=export.write([row(background_fill=fill,background_evidence='Line interruption' if fill else '')],[400,200],target,base)
        assert result['base_objects_unchanged'] and result['base_objects']==1
        render_native(target,tmp_path/label/'native_render',[400,200],180)
        rgb=np.asarray(Image.open(tmp_path/label/'native.png').convert('RGB'))
        values.append(rgb[80,250])
        assert rgb[80,30,0]>200 and rgb[80,30,1]<50
    assert values[0][0]>200 and values[0][1]<50
    assert min(values[1])>240
    # Score actual native white/transparent overlays on the same retained base.
    before=deepcopy(row())
    chosen=select([before],tmp_path/'white/native.png',tmp_path/'selection',
                  threading.BoundedSemaphore(1),{'render_timeout_seconds':180},base)
    assert chosen[0]['background_fill']=='#FFFFFF'
    assert chosen[0]['background_selection']['white_score']>chosen[0]['background_selection']['transparent_score']
    assert before==row()


def test_prompts_do_not_request_or_expose_fill():
    from shape_integration.texts.staged_formula import inventory_prompt
    from shape_integration.texts.render_feedback import prompt
    prior=row(background_fill='#FFFFFF',background_evidence='old model claim')
    for p in [recognize([400,200]),review([400,200],[prior],[prior]),
              inventory_prompt([400,200],{'max_regions':8}),prompt([400,200],[prior],[prior])]:
        assert 'background_fill' not in str(p)
        assert 'background_evidence' not in str(p)
