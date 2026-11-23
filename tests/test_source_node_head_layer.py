import numpy as np
from PIL import Image,ImageDraw
from full_recovery.handoff import source_occlusion
from shape_integration.fitting.paths import ellipse_commands


def fixture(tmp_path,visible=False):
    image=Image.new('RGB',(160,110),'white');draw=ImageDraw.Draw(image)
    draw.line((10,55,130,55),fill='#228833',width=2)
    draw.ellipse((70,43,94,67),fill='#DDAA66')
    if visible:draw.line((70,55,94,55),fill='#228833',width=2)
    source=tmp_path/'source.png';image.save(source)
    part={'id':'node','object_id':'owner','kind':'ellipse','status':'ok','bbox_2d':[70,43,94,67],
          'commands':ellipse_commands([82,55],[12,0],[0,12]),
          'diagnostics_local':{'source_primitive_registration':{'perimeter_support':1.}}}
    arrow={'id':'route','start':[10,55],'end':[68,55],'heads':['none','oval'],
           'style':{'color':'#228833','width_px':2,'end':{'type':'oval'}}}
    return {'parts':{'node':part}},{'node':{'fill':'#DDAA66','stroke':None,'width_px':0}},[arrow],source


def test_complete_source_node_stays_above_nearby_native_head(tmp_path):
    scene,styles,arrows,source=fixture(tmp_path)
    plan,audit=source_occlusion(scene,styles,arrows,source,set(),part_placements=True)
    assert plan['route']['before']==['owner/node']
    assert any(r['reason']=='complete_source_node_perimeter_above_attached_native_head'for r in audit)


def test_visible_source_line_and_unmeasured_face_do_not_get_new_layer_authority(tmp_path):
    scene,styles,arrows,source=fixture(tmp_path,True)
    plan,audit=source_occlusion(scene,styles,arrows,source,set(),part_placements=True)
    assert not any(r['reason']=='complete_source_node_perimeter_above_attached_native_head'for r in audit)
    scene,styles,arrows,source=fixture(tmp_path)
    scene['parts']['node']['diagnostics_local']={}
    plan,audit=source_occlusion(scene,styles,arrows,source,set(),part_placements=True)
    assert 'route'not in plan


def test_distant_head_and_translucent_node_keep_the_existing_protocol(tmp_path):
    scene,styles,arrows,source=fixture(tmp_path);arrows[0]['end']=[40,55]
    plan,audit=source_occlusion(scene,styles,arrows,source,set(),part_placements=True)
    assert not plan
    scene,styles,arrows,source=fixture(tmp_path);styles['node']['fill_opacity']=.5
    plan,audit=source_occlusion(scene,styles,arrows,source,set(),part_placements=True)
    assert not any(r['reason']=='complete_source_node_perimeter_above_attached_native_head'for r in audit)
