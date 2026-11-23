from copy import deepcopy
import threading
from zipfile import ZipFile

import numpy as np
from PIL import Image
from lxml import etree

from shape_integration.texts import contracts, export, layout, options
from shape_integration.texts.colors import recover
from shape_integration.texts.equations import NS, compatibility_copy, measure
from shape_integration.texts.omml import latex_to_omml


def row(**kwargs):
    result = dict(id='T0001', text='x2', runs=[{'text':'x','script':'normal','color':'#2266CC'},
        {'text':'2','script':'super','color':'#FF6600'}], bbox_2d=[20,20,180,90], font='Times New Roman',
        color='#000000', bold=False, italic=False, rotation_deg=0, legibility='clear', replaces=[])
    result.update(kwargs)
    return result


def test_protocol_retains_run_colors_and_native_formula():
    a = row(kind='formula', latex=r'\sqrt{\frac{x^2}{y}}')
    parsed = contracts.recognition({'texts':[a]}, [400,200])['texts'][0]
    assert parsed['runs'][0]['color'] == '#2266CC'
    assert parsed['kind'] == 'formula'
    assert parsed['latex'] == a['latex']


def test_structural_math_keeps_scripts_roots_fractions_and_colors():
    tree = latex_to_omml(r'\sqrt{\frac{\textcolor{#2266CC}{x_1}}{\hat y^2}}', color='#FF6600')
    for tag in ['rad','f','sSub','sSup','acc']:
        assert tree.find('.//m:'+tag, NS) is not None
    colors = tree.xpath('.//a:srgbClr/@val', namespaces=NS)
    assert '2266CC' in colors and 'FF6600' in colors
    assert tree.find('.//m:radPr/m:ctrlPr/a:rPr', NS) is not None


def test_real_equation_render_and_editable_ppt(tmp_path):
    formula = row(kind='formula', latex=r'\textcolor{#FF6600}{\mu_{\theta_m}}+\sqrt{\frac{x^2}{y}}')
    metrics = measure([formula], tmp_path/'formula', threading.BoundedSemaphore(1))
    rgba = np.array(Image.open(metrics['T0001']['equation_preview']))
    ink = rgba[rgba[:,:,3]>200,:3]
    assert ((ink[:,0]>200)&(ink[:,1]>50)&(ink[:,1]<150)&(ink[:,2]<30)).sum()>10
    assert (ink.max(axis=1)<30).sum()>10
    fitted = layout.fit(formula, metrics['T0001'])
    target = tmp_path/'native.pptx'
    export.write([fitted], [400,200], target)
    before = target.read_bytes()
    with ZipFile(target) as z:
        xml = etree.fromstring(z.read('ppt/slides/slide1.xml'))
        assert xml.find('.//mc:Choice//a14:m//m:rad', NS) is not None
        assert xml.find('.//mc:Fallback/p:pic', NS) is not None
    assert compatibility_copy(target,tmp_path/'lo.pptx') == 1
    assert target.read_bytes() == before


def test_plain_run_colors_survive_export(tmp_path):
    r = row(frame_px=[20,20,100,30],font_size_px=20,resolved_font='Times New Roman')
    export.write([r],[400,200],tmp_path/'colored.pptx')
    with ZipFile(tmp_path/'colored.pptx') as z:
        xml=etree.fromstring(z.read('ppt/slides/slide1.xml'))
    assert xml.xpath('.//a:r/a:rPr/a:solidFill/a:srgbClr/@val',namespaces=NS)==['2266CC','FF6600']


def test_color_recovery_preserves_scripts_and_text_on_colored_background(tmp_path):
    a=row(text='AB',runs=[{'text':'A','script':'normal','color':'#2255CC'},{'text':'B','script':'sub','color':'#CC5500'}],
        bbox_2d=[10,10,35,30],frame_px=[10,10,30,20],font_size_px=20,
        layout={'measurement':{'reference_size_px':20,'characters':[
            {'text':'A','ink_points':[[x,y] for x in range(3,7) for y in range(3,9)]},
            {'text':'B','ink_points':[[x,y] for x in range(17,21) for y in range(8,14)]}]}})
    image=np.full((50,60,3),[220,235,250],dtype=np.uint8)
    image[12:20,12:18]=[30,90,200]
    image[17:25,26:32]=[200,80,0]
    path=tmp_path/'source.png';Image.fromarray(image).save(path)
    result,_=recover([a],path)
    assert result[0]['text']=='AB'
    assert [r['script'] for r in result[0]['runs']]==['normal','sub']
    assert [r['color'] for r in result[0]['runs']]==['#1E5AC8','#C85000']
    wrong=deepcopy(a)
    for run in wrong['runs']:run['color']='#000000'
    guarded,_=recover([wrong],path)
    assert all(run['color']=='#000000' for run in guarded[0]['runs'])
