from copy import deepcopy

import numpy as np
import pytest
from PIL import Image, ImageDraw

from full_recovery.text_occlusion import select, safe_fill, write_backings
from shape_integration.common import save


def fixture():
    row={'id':'label','text':'AB','bbox_2d':[30,35,90,65],
         'frame_px':[30,35,60,30],'background_box_px':[30,35,90,65],
         'rotation_deg':0,'color':'#000000','font_size_px':14}
    scene={'parts':{'wire':{'id':'wire','object_id':'owner','kind':'path','status':'ok',
                           'commands':[['M',5,50],['L',115,50]]}}}
    styles={'wire':{'stroke':'#000000','width_px':1,'fill':None}}
    source=Image.new('RGB',(120,100),'white')
    glyph=Image.new('RGB',source.size,'white')
    for image in [source,glyph]:
        draw=ImageDraw.Draw(image)
        for x in [38,78]:draw.rectangle((x,38,x+4,61),fill='black')
    return row,scene,styles,source,glyph


@pytest.mark.parametrize('owner',['shape','line'])
@pytest.mark.parametrize('visible',[False,True])
def test_source_visible_line_within_glyph_box_survives_and_actual_source_knockout_is_allowed(owner,visible):
    row,scene,styles,source,glyph=fixture();draw=ImageDraw.Draw(source)
    draw.line((5,50,29,50),fill='black');draw.line((91,50,115,50),fill='black')
    if visible:draw.line((5,50,115,50),fill='black')
    arrows=[];approved=['wire'];approved_arrows=[]
    if owner=='line':
        arrows=[{'id':'route','start':[5,50],'end':[115,50],'heads':['none','none'],
                 'style':{'color':'#000000','width_px':1}}]
        scene={'parts':{}};styles={};approved=[];approved_arrows=['route']
    before=deepcopy((row,scene,styles,arrows))
    reason=safe_fill(row,approved,'#FFFFFF',scene,styles,arrows,[row],
        approved_arrows=approved_arrows,source_pixels=np.asarray(source),glyph_pixels=np.asarray(glyph))
    assert bool(reason)==visible
    if visible:assert 'under_text_backing' in reason
    assert (row,scene,styles,arrows)==before


def test_isolated_crossing_glyph_does_not_prove_a_continuous_source_line():
    row,scene,styles,source,glyph=fixture()
    assert safe_fill(row,['wire'],'#FFFFFF',scene,styles,[],[row],
        source_pixels=np.asarray(source),glyph_pixels=np.asarray(glyph)) is None


def test_old_backing_scope_does_not_survive_current_transparent_decision(tmp_path):
    row,scene,styles,source,glyph=fixture()
    row['background_box_px']=[0,0,120,100]
    row['background_fill']='#FFFFFF'
    path=tmp_path/'source.png';source.save(path)
    class Review:
        calls=0
        def call(self,root,tag,system,prompt,images,validator,budget):
            self.calls+=1
            return validator({'decision':'transparent','fill':None,'occluded_parts':[],
                              'evidence':'The source has text ink with no opaque label backing.'})
    runtime=Review()
    selected,report=select([row],scene,{'parts':{k:{'style':v} for k,v in styles.items()}},
        {'arrows':[]},{},path,tmp_path/'review',runtime,save,glyph_native=glyph)
    assert runtime.calls==1 and report['filled']==0
    assert selected[0]['background_fill'] is None
    assert selected[0]['background_box_px']==[29.,34.,91.,66.]


def test_generated_native_backings_are_idempotent_and_glyph_bytes_survive(tmp_path):
    from pptx import Presentation
    from lxml import etree
    from full_recovery.compose import read_package,P,SLIDE
    prs=Presentation();slide=prs.slides.add_slide(prs.slide_layouts[6])
    text=slide.shapes.add_textbox(100,200,300,400);text.name='TXT label';text.text='AB'
    path=tmp_path/'text.pptx';prs.save(path)
    before=etree.tostring(etree.fromstring(read_package(path)[SLIDE]).find('.//{'+P+'}sp'))
    rows=[{'background_fill':'#FFFFFF','background_box_px':[20,20,60,40]}]
    write_backings(path,rows,(120,100));write_backings(path,rows,(120,100))
    tree=etree.fromstring(read_package(path)[SLIDE]);shapes=tree.findall('.//{'+P+'}sp')
    assert len(shapes)==2 and etree.tostring(shapes[-1])==before
    write_backings(path,[],(120,100))
    tree=etree.fromstring(read_package(path)[SLIDE]);shapes=tree.findall('.//{'+P+'}sp')
    assert len(shapes)==1 and etree.tostring(shapes[0])==before
