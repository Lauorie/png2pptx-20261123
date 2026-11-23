import copy
import json
import sys
import zipfile
from pathlib import Path
import pytest
from PIL import Image
from lxml import etree

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'lib')]
import segmented_geometry as sg
import geometry_critic as gc
import geometry_heads as gh
import native
import native_scale
import style_contract
import shared_repair


def document(heads=('triangle','oval')):
    return dict(nodes=[['a',10,20],['b',80,20]],routes=[dict(id='R1',paths=[dict(id='p1',heads=list(heads),segments=[dict(kind='line',nodes=['a','b'])])])])


def row():
    return sg.geometry(document(),{'R1':dict(id='G1:T1')})[0]


def test_geometry_requires_endpoint_types_and_preserves_order():
    d=document();del d['routes'][0]['paths'][0]['heads']
    with pytest.raises(ValueError):sg.validate(d,{'R1'},(100,100),require_heads=True)
    rows=sg.geometry(sg.validate(document(),{'R1'},(100,100),require_heads=True),{'R1':dict(id='x')})
    assert rows[0]['heads']==['triangle','oval']
    assert rows[0]['start_head_type']=='triangle' and rows[0]['end_head_type']=='oval'
    assert rows[0]['geometry_heads_reported']
    reread=gc.geometry_rows(gc.as_points(rows,with_heads=True))
    assert reread[0]['heads']==['triangle','oval']
    assert reread[0]['start']==rows[0]['start'] and reread[0]['end']==rows[0]['end']


def test_critic_can_correct_head_direction_and_does_not_silently_drop_it():
    reply=dict(decision='refine',reason='head belongs at start',geometry=document(('triangle','none')))
    result=gc.validate(reply,(100,100),require_heads=True)
    assert result['geometry']['routes'][0]['paths'][0]['heads']==['triangle','none']
    del reply['geometry']['routes'][0]['paths'][0]['heads']
    with pytest.raises(ValueError):gc.validate(reply,(100,100),require_heads=True)
    with pytest.raises(ValueError):gc.validate(dict(decision='accept',geometry=document()),(100,100),require_heads=True)


def test_pixel_style_cannot_overwrite_semantic_heads():
    r=row();s=gh.provisional(dict(start=None,end=None,width_px=1,color='#000000',dash=dict(type='solid',preset='solid')),r)
    assert s['start']==dict(type='triangle',width='med',length='med')
    assert s['end']==dict(type='oval',width='med',length='med')
    unknown=gh.provisional(s,{**r,'heads':['unknown','none']})
    assert unknown['start'] is None and unknown['end'] is None
    assert unknown['unknown_endpoints']==['start']
    wire=dict(id=r['id'],color='#000000',width_px=2,dash_px=[],start=s['start'],end=s['end'])
    style_contract.validate_style(wire,r)
    with pytest.raises(ValueError):style_contract.validate_style({**wire,'start':None},r)
    with pytest.raises(ValueError):style_contract.validate_style({**wire,'end':s['start']},r)


def test_repair_requires_own_heads_on_each_path():
    d=document();route=d['routes'][0];route.update(kind='line',line_style='solid',masks=[])
    answer=dict(status='repair',replace=[],context_bbox=None,reason='visible missing line',**d)
    checked=shared_repair.validate(answer,(100,100),(100,100),{},[],True,require_heads=True)
    assert checked['routes'][0]['paths'][0]['heads']==['triangle','oval']
    del answer['routes'][0]['paths'][0]['heads']
    with pytest.raises(ValueError):shared_repair.validate(answer,(100,100),(100,100),{},[],True,require_heads=True)


def test_uniform_scaling_changes_physical_canvas_not_pixel_geometry(tmp_path):
    r=row();s=gh.provisional(dict(color='#000000',width_px=1,dash=dict(type='solid',preset='solid')),r)
    before=copy.deepcopy(r);path=tmp_path/'safe.pptx'
    native.write((1200,700),[r],[s],path)
    audit=json.loads(path.with_suffix('.export.json').read_text());sc=audit['native_scale']
    assert sc['factor']>1 and sc['safe'] and r==before
    assert sc['minimum_head_points']==4
    with zipfile.ZipFile(path) as z:
        slide=etree.fromstring(z.read('ppt/slides/slide1.xml'));presentation=etree.fromstring(z.read('ppt/presentation.xml'))
    ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
    width=int(presentation.xpath('//p:sldSz/@cx',namespaces=ns)[0]);per_px=width/1200
    assert abs(int(slide.xpath('//a:xfrm/a:off/@x',namespaces=ns)[0])/per_px-.5-min(r['start'][0],r['end'][0]))<.001
    assert abs(int(slide.xpath('//a:ln/@w',namespaces=ns)[0])/per_px-s['width_px'])<.001
    assert slide.xpath('//a:headEnd/@type',namespaces=ns)==['triangle']
    assert slide.xpath('//a:tailEnd/@type',namespaces=ns)==['oval']
    assert slide.xpath('//a:tailEnd/@w',namespaces=ns)==['med']


def test_headless_and_block_shapes_do_not_trigger_head_scaling():
    r=row();s=dict(width_px=.01,start=None,end=None)
    assert native_scale.plan((1200,700),[r],[s])['factor']==1
    s['end']=dict(type='triangle',width='med',length='med')
    assert native_scale.plan((1200,700),[{**r,'kind':'block_arrow'}],[s])['factor']==1


def test_impossible_native_scale_is_reported_not_clamped(tmp_path):
    r=row();s=gh.provisional(dict(color='#000000',width_px=.001,dash=dict(type='solid',preset='solid')),r)
    path=tmp_path/'too_thin.pptx'
    with pytest.raises(ValueError,match='56-inch'):native.write((1200,700),[r],[s],path)
    assert not path.exists()
    assert not json.loads(path.with_suffix('.scale.json').read_text())['safe']
