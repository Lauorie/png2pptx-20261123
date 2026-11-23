"""Source-pixel identity, native PPT stacking and image-aware OCR integration."""
import copy
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.util import Inches, Pt

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'vendor')]
from shape_integration import composition as c
from shape_integration.common import read, save, sha
from shape_integration.texts import export as text_export, runner as text_runner
from shape_integration.styling.export import render_native


def source_fixture(tmp):
    pixels=np.zeros((120,160,4),dtype=np.uint8)
    pixels[:]=[30,150,220,255]
    pixels[20:100,20:140]=[240,40,40,255]
    Image.fromarray(pixels).save(tmp/'source.png')
    item=dict(id='case',original='source.png',source_size=[160,120],source_sha256=sha(tmp/'source.png'))
    inventory=dict(source_sha256=item['source_sha256'],assets=[dict(id='I0001',source_box_half_open=[20,20,140,100],
                   background='alpha',boundary_review={'status':'unresolved'})])
    registry=tmp/'upstream/cases/case/dedup/image_boxes_original.json';save(registry,inventory)
    return item,registry,pixels


def base_fixture(tmp,with_text=True):
    prs=Presentation();prs.slide_width=Inches(8);prs.slide_height=Inches(6)
    slide=prs.slides.add_slide(prs.slide_layouts[6]);scale=prs.slide_width/160
    rect=slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE,0,0,prs.slide_width,prs.slide_height)
    rect.fill.solid();rect.fill.fore_color.rgb=RGBColor(0,200,0);rect.line.fill.background()
    if with_text:
        t=slide.shapes.add_textbox(round(40*scale),round(35*scale),round(80*scale),round(55*scale))
        t.name='TXT T0001';t.text='X';t.text_frame.margin_left=0;t.text_frame.margin_top=0
        t.text_frame.paragraphs[0].runs[0].font.size=Pt(100)
        t.text_frame.paragraphs[0].runs[0].font.color.rgb=RGBColor(255,255,255)
    p=tmp/'base.pptx';prs.save(p);return p


def test_crop_is_exact_and_alpha_fallback_is_explicit(tmp_path):
    item,registry,pixels=source_fixture(tmp_path)
    records=c.extract(tmp_path,item,registry,tmp_path/'out')
    r=records[0]
    assert r['placement_source_half_open']==[20,20,140,100]
    np.testing.assert_array_equal(np.asarray(Image.open(r['path'])),pixels[20:100,20:140])
    assert r['quality_flags']==['alpha_unavailable_original_rectangle_preserved']
    assert r['boundary_status']=='unresolved' and not r['visual_accuracy_verified']


def test_wrong_source_hash_and_missing_registry_are_not_empty_results(tmp_path):
    item,registry,_=source_fixture(tmp_path)
    inventory=read(registry);inventory['source_sha256']='wrong';save(registry,inventory)
    with pytest.raises(ValueError,match='hash differs'):c.extract(tmp_path,item,registry,tmp_path/'out')
    save(tmp_path/'empty/manifest.json',{'images':[item]})
    with pytest.raises(ValueError,match='No image registry'):c.resolve_registry(tmp_path,tmp_path/'empty',item)


def test_source_lineage_and_cycle_protection(tmp_path):
    item,registry,_=source_fixture(tmp_path)
    save(tmp_path/'upstream/manifest.json',{'images':[item]})
    save(tmp_path/'text/manifest.json',{'images':[item],'base_source':'styled'})
    save(tmp_path/'styled/manifest.json',{'images':[item],'geometry_source':'upstream'})
    found,chain=c.resolve_registry(tmp_path,tmp_path/'text',item)
    assert found==registry and len(chain)==3
    changed=copy.deepcopy(item);changed['source_size']=[1,1]
    with pytest.raises(ValueError,match='provenance differs'):c.resolve_registry(tmp_path,tmp_path/'text',changed)
    registry.unlink();save(tmp_path/'upstream/manifest.json',{'images':[item],'base_source':'text'})
    with pytest.raises(ValueError,match='No image registry'):c.resolve_registry(tmp_path,tmp_path/'text',item)


def test_reuse_exported_rgba_requires_matching_identity_and_bytes(tmp_path):
    item,registry,_=source_fixture(tmp_path)
    folder=registry.parent.parent/'images/assets/I0001';folder.mkdir(parents=True)
    Image.new('RGBA',(120,80),(10,20,30,100)).save(folder/'rgba.png')
    result=dict(export_enabled=True,rgba='rgba.png',rgba_sha256=sha(folder/'rgba.png'),
                source_sha256=item['source_sha256'],source_box_half_open=[20,20,140,100],placement_source_half_open=[20,20,140,100])
    save(folder/'result.json',result)
    r=c.extract(tmp_path,item,registry,tmp_path/'ok')[0]
    assert r['extraction_method']=='existing_rgba' and r['sha256']==result['rgba_sha256']
    del result['source_sha256'];save(folder/'result.json',result)
    save(registry.parent.parent/'images/image_stage.json',{'image':{'source_sha256':item['source_sha256']}})
    assert c.extract(tmp_path,item,registry,tmp_path/'stage_bound')[0]['extraction_method']=='existing_rgba'
    result['source_sha256']='other';save(folder/'result.json',result)
    with pytest.raises(ValueError,match='different image crop'):c.extract(tmp_path,item,registry,tmp_path/'bad_source')
    result['source_sha256']=item['source_sha256'];result['rgba_sha256']='changed';save(folder/'result.json',result)
    with pytest.raises(ValueError,match='missing or changed'):c.extract(tmp_path,item,registry,tmp_path/'bad_bytes')


def test_native_layer_order_preserves_base_and_renders_text_above_image(tmp_path):
    item,registry,_=source_fixture(tmp_path);base=base_fixture(tmp_path);before=sha(base)
    assets=c.extract(tmp_path,item,registry,tmp_path/'out');dest=tmp_path/'combined.pptx'
    result=c.inject(assets,item['source_size'],base,dest)
    assert result['image_layer_indices']==[1] and result['text_layer_indices']==[2]
    assert result['base_objects_unchanged'] and sha(base)==before
    original=Presentation(str(base));composed=Presentation(str(dest))
    assert [etree.tostring(s._element) for s in original.slides[0].shapes]==[
        etree.tostring(s._element) for s in composed.slides[0].shapes if not s.name.startswith('IMG ')]
    render_native(dest,tmp_path/'render',item['source_size'])
    pixels=np.asarray(Image.open(tmp_path/'native.png').convert('RGB'))
    assert pixels[5,5].tolist()==[0,200,0]
    assert pixels[25,25].tolist()==[240,40,40]
    assert np.count_nonzero(np.all(pixels[35:90,40:120]>245,axis=2))>40
    reused=c.inject(assets,item['source_size'],dest,tmp_path/'again.pptx')
    assert reused['reused_images']==1 and reused['inserted_images']==0
    moved=copy.deepcopy(assets);moved[0]['placement_source_half_open']=[21,20,141,100]
    with pytest.raises(ValueError,match='identity/placement conflict'):c.inject(moved,item['source_size'],dest,tmp_path/'conflict.pptx')


def test_grouped_formula_is_above_picture(tmp_path):
    item,registry,_=source_fixture(tmp_path);base=base_fixture(tmp_path,False)
    prs=Presentation(str(base));group=prs.slides[0].shapes.add_group_shape()
    t=group.shapes.add_textbox(100,100,10000,10000);t.text='x2';prs.save(base)
    assets=c.extract(tmp_path,item,registry,tmp_path/'out')
    result=c.inject(assets,item['source_size'],base,tmp_path/'combined.pptx')
    assert result['image_layer_indices']==[1] and result['text_layer_indices']==[2]


def test_full_pipeline_inserts_images_before_text_ocr(tmp_path):
    from shape_integration.pipeline import run_case
    from test_pipeline import FakeRuntime
    item,registry,_=source_fixture(tmp_path)
    item.update(input='source.png',input_sha256=item['source_sha256'],input_size=item['source_size'])
    config=read(ROOT/'config.json');config['fitting']['coverage_rounds']=0
    config.update(image_export_enabled=True,texts_enabled=True,arrows_enabled=False,styling_enabled=False)
    runtime=FakeRuntime(config);root=tmp_path/'new'
    def detect(run,item,workspace,rt):
        case=run/'cases'/item['id']
        save(case/'geometry.json',dict(image_size=item['source_size'],coordinate_system='image_pixel_centers',objects=[]))
        save(case/'dedup/image_boxes_original.json',read(registry))
        return dict(image_coverage_status='complete',errors={},local_failures=0)
    seen=[]
    def ocr(workspace,case,item,summary,*args):
        base=text_runner.base_ppt(case)
        assert base==case/'image_composition/combined.pptx'
        actual=text_export.retained_images(base,item['source_size']);seen.extend(actual)
        return summary
    with ThreadPoolExecutor(1) as pool,patch('shape_integration.batch.case',side_effect=detect),patch.object(text_runner,'append',side_effect=ocr):
        result=run_case(tmp_path,root,item,runtime,config,pool,None,None,False)
    assert len(seen)==1 and result['image_composition']['images']==1
    assert runtime.events==[]
