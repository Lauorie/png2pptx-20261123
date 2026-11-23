"""Text protocol, source transforms, stable IDs and actual editable PPT layout."""
import copy
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import fitz
from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.util import Inches

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'vendor')]
from shape_integration.common import read, sha
from shape_integration.texts import options, contracts, controller, export, layout, prompts
from shape_integration.styling.export import render_native, NS


def row(text='Hello', box=(30,30,130,50), **overrides):
    return dict(text=text,bbox_2d=list(box),font='Times New Roman',color='#000000',
                bold=False,italic=False,rotation_deg=0,legibility='clear',
                runs=[{'text':text,'script':'normal'}],replaces=[],**overrides)


def parsed(value=None, **kwargs):
    return contracts.recognition({'texts':[value or row()]},[400,300],**kwargs)['texts'][0]


@pytest.mark.parametrize('change',[
    {'bbox_2d':[0,0,float('nan'),2]}, {'bbox_2d':[0,0,400,30]},
    {'rotation_deg':True}, {'rotation_deg':float('inf')}, {'bold':'false'},
    {'font':'invented font'}, {'color':'red'}, {'text':'two\nlines'},
    {'text':'hidden\x00'}, {'text':'','legibility':'clear'},
    {'runs':[{'text':'different','script':'sub'}]}, {'replaces':['T0001']},
])
def test_invalid_recognition_is_not_exported(change):
    with pytest.raises(ValueError):parsed({**row(),**change})


def test_default_font_and_scripts():
    a=parsed({'text':'x2','bbox_2d':[10,10,30,30],
              'runs':[{'text':'x','script':'normal'},{'text':'2','script':'sub'}],
              'font_size':900})
    assert a['font']=='Times New Roman' and 'font_size' not in a
    assert a['runs'][1]['script']=='sub'
    p=prompts.recognize([2048,1000])
    assert p['fonts']['default']=='Times New Roman'
    assert 'font_size' not in str(p['response_format'])


def test_equivalent_clockwise_rotations_need_no_model_retry():
    assert parsed({**row(),'rotation_deg':270})['rotation_deg']==-90
    assert parsed({**row(),'rotation_deg':360})['rotation_deg']==0


def test_local_replacement_references_and_regions():
    a={**row(),'replaces':['T0001']}
    assert parsed(a,known=['T0001'],local=True)['replaces']==['T0001']
    with pytest.raises(ValueError):contracts.recognition({'texts':[a,a]},[400,300],known=['T0001'],local=True)
    with pytest.raises(ValueError):contracts.recognition({'texts':[],'regions':[{'bbox_2d':[0,0,100,100],'reason':'dense'}]},[400,300],local=True)


def test_coordinate_roundtrip_and_global_crop_scale(tmp_path):
    source=tmp_path/'source.png';Image.new('RGB',(1000,500),'white').save(source)
    opt=options({});global_frame=controller.frame(source,tmp_path/'global',opt)
    assert global_frame['size']==[2048,1024]
    f=controller.frame(source,tmp_path/'local',opt,[100,100,150,130])
    assert f['scale']==pytest.approx(2048/1000*3,abs=.02)
    b=[105.2,110.7,145.8,125.3]
    assert controller.mapped_box(controller.mapped_box(b,f),f,True)==pytest.approx(b,abs=1e-10)


def test_merge_preserves_distinct_lines_and_observation_history():
    f={'box':[0,0,400,300],'source_size':[400,300],'scale':1.,'offset':[0,0]}
    rows=[];history=[];opt=options({})
    controller.merge(rows,[row(),row(box=(160,30,260,50))],f,'global',history,opt)
    assert len(rows)==2 and [r['id'] for r in rows]==['T0001','T0002']
    controller.merge(rows,[{**row(text='Hallo'),'replaces':['T0001']}],f,'R01',history,opt)
    assert rows[0]['text']=='Hallo' and len(rows[0]['observations'])==2
    controller.merge(rows,[{**row(text='[?]'),'legibility':'partial','replaces':['T0001']}],f,'R02',history,opt)
    assert rows[0]['text']=='Hallo' and history[-1]['action']=='keep_clear'
    controller.merge(rows,[row(box=(160.1,30,260,50))],f,'R03',history,opt)
    assert len(rows)==2


@pytest.mark.parametrize('value',[float('nan'),0,-1,True])
def test_invalid_sizing_configuration(value):
    with pytest.raises(ValueError):options({'texts':{'crop_scale':value}})


def test_native_render_text_ink_matches_boxes_and_base_survives(tmp_path):
    rows=[row('Ag', (50,50,120,90)),row('x2',(210,50,260,85)),
          row('Rotate',(55,155,85,280)),row('Slant',(205,155,295,245))]
    rows[1]['runs']=[{'text':'x','script':'normal'},{'text':'2','script':'sub'}]
    rows[2]['rotation_deg']=90;rows[3]['rotation_deg']=45
    for i,r in enumerate(rows):r['id']=f'T{i+1:04d}'
    opt=options({});slots=threading.BoundedSemaphore(1)
    fitted,errors=layout.fit_all(rows,tmp_path/'layout',slots,opt)
    assert not errors
    assert all(r['layout']['measurement']['method']=='native_ppt_atlas' for r in fitted)
    base=tmp_path/'base.pptx';prs=Presentation();prs.slide_width=Inches(10);prs.slide_height=Inches(7.5)
    slide=prs.slides.add_slide(prs.slide_layouts[6]);slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE,Inches(8),Inches(6),Inches(1),Inches(1))
    prs.save(base);before=sha(base)
    export.write(fitted,[400,300],tmp_path/'combined.pptx',base)
    loaded=Presentation(str(tmp_path/'combined.pptx'))
    assert len(loaded.slides[0].shapes)==5 and sha(base)==before
    assert etree.tostring(Presentation(str(base)).slides[0].shapes[0]._element)==etree.tostring(loaded.slides[0].shapes[0]._element)
    math_shape=loaded.slides[0].shapes[2]
    assert math_shape.text=='x2'
    assert all(r.font.name=='Times New Roman' for r in math_shape.text_frame.paragraphs[0].runs)
    assert math_shape._element.findall('.//a:rPr',NS)[1].get('baseline')=='-25000'
    assert all(s.text_frame.word_wrap is False for s in list(loaded.slides[0].shapes)[1:])
    export.write(fitted,[400,300],tmp_path/'text.pptx')
    render_native(tmp_path/'text.pptx',tmp_path/'render',[400,300])
    # Inspect rendered PDF glyph sizes independently of the PPT XML: script
    # engines already shrink baseline runs, so explicit small sz must not halve
    # their visual size a second time.
    with fitz.open(tmp_path/'native.pdf') as pdf:
        spans=[span for b in pdf[0].get_text('dict')['blocks'] for line in b.get('lines',[]) for span in line['spans']]
    normal=next(s for s in spans if s['text']=='x')
    script=next(s for s in spans if s['text']=='2')
    assert .5<script['size']/normal['size']<.75
    assert script['origin'][1]>normal['origin'][1]
    pixels=np.asarray(Image.open(tmp_path/'native.png').convert('RGB'))
    for original,fit in zip(rows,fitted):
        x0,y0,x1,y1=original['bbox_2d'];margin=10
        tile=pixels[y0-margin:y1+margin+1,x0-margin:x1+margin+1]
        yy,xx=np.where(tile.min(axis=2)<150)
        assert len(xx),original['id']
        actual=[xx.min()+x0-margin,yy.min()+y0-margin,xx.max()+x0-margin,yy.max()+y0-margin]
        assert actual[0]>=x0-2 and actual[1]>=y0-2 and actual[2]<=x1+2 and actual[3]<=y1+2,(original['id'],actual)
        assert abs((actual[0]+actual[2]-x0-x1)/2)<=2,(original['id'],actual)
        assert abs((actual[1]+actual[3]-y0-y1)/2)<=2,(original['id'],actual)
        assert max((actual[2]-actual[0]+1)/(x1-x0+1),(actual[3]-actual[1]+1)/(y1-y0+1))>.85


def test_render_failure_is_failed_case(tmp_path):
    source=tmp_path/'source.png';Image.new('RGB',(400,300),'white').save(source)
    with ThreadPoolExecutor(2) as pool, patch.object(controller.model,'ask',return_value={'texts':[row()],'regions':[],'uncertain':[]}), \
         patch.object(controller.layout,'fit_all',return_value=([row()],[])), \
         patch.object(controller.export,'final',side_effect=RuntimeError('renderer failed')):
        result=controller.run_case(source,tmp_path/'texts',None,options({}),pool,threading.BoundedSemaphore(1))
    assert result['state']=='failed'
    assert 'renderer failed' in result['errors'][-1]


def test_many_short_labels_keep_native_metrics_with_office_page_limits(tmp_path):
    rows=[{**row('S1'),'id':f'T{i:04d}'} for i in range(20)]
    fitted,errors=layout.fit_all(rows,tmp_path/'layout',threading.BoundedSemaphore(1),options({}))
    assert not errors and len(fitted)==20
    assert all(r['layout']['measurement']['method']=='native_ppt_atlas' for r in fitted)


@pytest.mark.parametrize('angle',[0,90,-90,45,-55,180])
def test_tight_frame_keeps_typography_and_rotated_line_origin(angle):
    saved={**row('Italic'),'rotation_deg':angle,'id':'T0001','font_size_px':24.,
           'frame_px':[100.,100.,300.,150.],
           'layout':{'measurement':{'reference_size_px':24.,'width':60.,'height':22.,
                                    'dx':-2.,'dy':7.}}}
    before=copy.deepcopy(saved)
    tightened=layout.tighten_frame(saved)
    assert saved==before
    assert tightened['font_size_px']==saved['font_size_px']
    assert tightened['bbox_2d']==saved['bbox_2d'] and tightened['runs']==saved['runs']
    assert tightened['frame_px'][2]<80 and tightened['frame_px'][3]<40
    # Check the Office transform independently with a homogeneous matrix.
    def world_origin(value):
        x,y,w,h=value['frame_px'];left,top=value.get('text_insets_px',[0,0])[:2]
        theta=np.deg2rad(value['rotation_deg'])
        rotation=np.array([[np.cos(theta),-np.sin(theta),0],
                           [np.sin(theta),np.cos(theta),0],[0,0,1]])
        translate=np.array([[1,0,x+w/2],[0,1,y+h/2],[0,0,1]])
        return translate@rotation@np.array([left-w/2,top-h/2,1])
    np.testing.assert_allclose(world_origin(tightened),world_origin(saved),atol=1e-10)
    assert tightened['text_insets_px'][0]>=2  # italic left bearing stays inside
    again=layout.tighten_frame(tightened)
    assert again['frame_px']==pytest.approx(tightened['frame_px'])


def test_global_failure_does_not_claim_successful_empty_text(tmp_path):
    source=tmp_path/'source.png';Image.new('RGB',(400,300),'white').save(source)
    with ThreadPoolExecutor(2) as pool, patch.object(controller.model,'ask',side_effect=RuntimeError('timeout')), \
         patch.object(controller.export,'final',return_value={'textboxes':0}):
        result=controller.run_case(source,tmp_path/'texts',None,options({}),pool,threading.BoundedSemaphore(1))
    assert result['state']=='failed' and result['exported_lines']==0
