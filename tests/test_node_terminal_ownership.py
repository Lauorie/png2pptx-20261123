from copy import deepcopy
from PIL import Image,ImageDraw
from full_recovery.node_terminal_ownership import observe
from full_recovery.compose import suppress_round_caps,P,A
from shape_integration.fitting.paths import ellipse_commands
from lxml import etree


def fixture(tmp_path):
    image=Image.new('RGB',(160,110),'white');draw=ImageDraw.Draw(image);draw.ellipse((70,43,94,67),fill='#DDAA66')
    source=tmp_path/'source.png';image.save(source)
    node={'id':'node','object_id':'owner','kind':'ellipse','status':'ok','bbox_2d':[70,43,94,67],
          'commands':ellipse_commands([82,55],[12,0],[0,12]),
          'diagnostics_local':{'source_primitive_registration':{'perimeter_support':1.}}}
    row={'id':'route','start':[10,55],'end':[68,55],'heads':['none','oval'],
         'style':{'color':'#CFAA66','width_px':2,'end':{'type':'oval'}}}
    return {'parts':{'node':node}},{'node':{'fill':'#DDAA66','stroke':None}},[row],source


def test_unique_measured_node_owns_duplicate_cap_without_a_shaft_change(tmp_path):
    scene,styles,arrows,source=fixture(tmp_path);before=deepcopy(arrows)
    caps,audit=observe(scene,styles,arrows,source)
    assert caps=={'route':[1]} and audit[0]['shaft_preserved'] and arrows==before
    root=etree.fromstring(f'<p:spTree xmlns:p="{P}" xmlns:a="{A}"><p:cxnSp><p:nvCxnSpPr><p:cNvPr id="1" name="route"/></p:nvCxnSpPr><p:spPr><a:xfrm/><a:ln w="25400"><a:headEnd type="triangle"/><a:tailEnd type="oval"/></a:ln></p:spPr></p:cxnSp></p:spTree>')
    props=root.find('.//{'+P+'}spPr');before=etree.tostring(props.find('{'+A+'}xfrm'))
    checked=suppress_round_caps(root,caps)
    assert len(checked)==1 and root.find('.//{'+A+'}tailEnd').get('type')=='none'
    assert root.find('.//{'+A+'}headEnd').get('type')=='triangle'
    assert etree.tostring(props.find('{'+A+'}xfrm'))==before


def test_directional_head_unmeasured_node_and_ambiguous_nodes_abstain(tmp_path):
    scene,styles,arrows,source=fixture(tmp_path);arrows[0]['style']['end']['type']='triangle'
    assert observe(scene,styles,arrows,source)[0]=={}
    scene,styles,arrows,source=fixture(tmp_path);scene['parts']['node']['diagnostics_local']={}
    assert observe(scene,styles,arrows,source)[0]=={}
    scene,styles,arrows,source=fixture(tmp_path);scene['parts']['copy']=deepcopy(scene['parts']['node']);styles['copy']=styles['node']
    assert observe(scene,styles,arrows,source)[0]=={}
