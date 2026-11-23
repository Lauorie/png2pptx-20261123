from copy import deepcopy
import threading
from zipfile import ZipFile

import fitz
import pytest
from lxml import etree
from PIL import Image

from shape_integration.texts import contracts, export, layout
from shape_integration.texts.colors import recover
from shape_integration.texts.local_content import validate
from shape_integration.texts.prompt_protocol import review
from shape_integration.texts.reflow import slice_runs


def row():
    return dict(id='T1',text='Base Bold Italic Under',kind='text',latex='',
        runs=[{'text':'Base ','script':'normal','color':'#000000'},
              {'text':'Bold ','script':'normal','color':'#000000','bold':True},
              {'text':'Italic ','script':'normal','color':'#000000','italic':True},
              {'text':'Under','script':'normal','color':'#000000','underline':True}],
        bbox_2d=[10,10,380,45],font='Arial',color='#000000',bold=False,italic=False,
        underline=False,rotation_deg=0,legibility='clear',visual_line_count=1,
        frame_px=[10,10,380,50],font_size_px=24,resolved_font='Arial')


def test_crop_updates_run_styles_and_keeps_single_text_unit():
    old=row();old['runs']=[{'text':old['text'],'script':'normal'}]
    unit=row();unit.pop('id');unit.pop('text')
    unit.update(direction_status='confirmed',reason='Visible local stroke/slant/underline differences')
    response={'items':[{'id':'T1','status':'resolved','units':[unit]}]}
    result=validate(response,[old],[400,100])['items'][0]
    assert result['status']=='located'
    assert result['text']==old['text']
    assert result['runs']==row()['runs']
    assert old['runs']==[{'text':old['text'],'script':'normal'}]


@pytest.mark.parametrize('key',['bold','italic','underline'])
@pytest.mark.parametrize('bad',[None,1,'false',[]])
def test_run_style_requires_real_boolean(key,bad):
    r=row();r['runs'][0][key]=bad
    with pytest.raises(ValueError,match='runs.'+key):
        contracts.recognition({'texts':[r]},[400,100])


def test_missing_attributes_inherit_but_explicit_false_clears(tmp_path):
    r=row();r.update(text='ABC',bold=True,italic=True,underline=True)
    r['runs']=[{'text':'A','script':'normal'},
               {'text':'B','script':'super','bold':False,'italic':False,'underline':False},
               {'text':'C','script':'normal','bold':True,'italic':False,'underline':True}]
    parsed=contracts.recognition({'texts':[r]},[400,100])['texts'][0]
    assert 'bold' not in parsed['runs'][0]
    assert parsed['runs'][1]['bold'] is False
    export.write([{**r,**parsed}],[400,100],tmp_path/'styles.pptx')
    with ZipFile(tmp_path/'styles.pptx') as z:
        xml=etree.fromstring(z.read('ppt/slides/slide1.xml'))
    ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
    props=xml.xpath('.//a:r/a:rPr',namespaces=ns)
    assert [(p.get('b'),p.get('i'),p.get('u')) for p in props]==[
        ('1','1','sng'),('0','0','none'),('1','0','sng')]
    assert props[1].get('baseline')=='30000'


def test_recoloring_and_line_slicing_preserve_local_style_boundaries(tmp_path):
    r=row();before=deepcopy(r)
    # No matching source glyphs: recoloring still rebuilds runs and used to
    # merge all same-color/script spans, discarding their local styles.
    source=tmp_path/'source.png';Image.new('RGB',(400,100),'white').save(source)
    result,_=recover([r],source)
    assert result[0]['runs']==before['runs']
    assert r==before
    assert slice_runs(result[0]['runs'],5,9)==[
        {'text':'Bold','script':'normal','color':'#000000','bold':True}]


def test_pillow_metrics_use_run_specific_font_faces(monkeypatch):
    calls=[];original=layout.resolve_font
    def resolve(family,bold=False,italic=False):
        calls.append((bold,italic));return original(family,bold,italic)
    monkeypatch.setattr(layout,'resolve_font',resolve)
    assert layout.pillow_metrics(row())['width']>0
    assert calls==[(False,False),(False,False),(True,False),(False,True),(False,False)]


def test_crop_prompt_has_structured_styles_and_mixed_example():
    prompt=review([400,100],[row()],[])
    schema=prompt['response_format']['items'][0]['units'][0]['runs'][0]
    assert all(type(schema[k]) is bool for k in ('bold','italic','underline'))
    rules=' '.join(prompt['rules'])
    assert 'absolute overrides' in rules and 'MIXED STYLE EXAMPLE' in rules
    assert 'not separate text boxes' in rules


def test_native_metrics_render_local_styles(tmp_path):
    metrics=layout.native_metrics([row()],tmp_path,threading.BoundedSemaphore(1),
                                 {'render_timeout_seconds':180})
    assert metrics['T1']['method']=='native_ppt_atlas'
    with fitz.open(tmp_path/'native.pdf') as doc:
        spans=[s for b in doc[0].get_text('dict')['blocks'] for l in b.get('lines',[])
               for s in l.get('spans',[])]
    assert any('Bold' in s['text'] and s['flags'] & 16 for s in spans)
    assert any('Italic' in s['text'] and s['flags'] & 2 for s in spans)
    with ZipFile(tmp_path/'metrics.pptx') as z:
        xml=etree.fromstring(z.read('ppt/slides/slide1.xml'))
    ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
    assert xml.xpath('.//a:r[a:t="Under"]/a:rPr/@u',namespaces=ns)==['sng']
