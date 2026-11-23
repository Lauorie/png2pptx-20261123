"""Source-backed extent repair, fallible masks and already-observed callouts."""
import copy
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from shape_integration.fitting.frame_edges import measure, refine_outline
from shape_integration.fitting.paths import box_commands


def test_enclosed_peer_edge_cannot_numerically_shrink_outer_padding():
    from shape_integration.fitting.frame_edges import refine
    image=Image.new('RGB',(230,190),'white');draw=ImageDraw.Draw(image)
    draw.rectangle((30,20,180,160),outline='#7593BB',width=1)
    draw.rectangle((34,24,176,156),outline='black',width=3)
    part={'id':'outer','kind':'rectangle','status':'ok','observed_point_ids':[],
          'parameters':{'center':[105,90],'width':150,'height':140,'angle_radians':0.},
          'bbox_2d':[30,20,180,160]}
    local={'parts':[part],'observation':{'points':[]},'bbox_2d':[30,20,180,160]};before=copy.deepcopy(local)
    job={'frame':{'source_crop_half_open':[0,0,230,190],'actual_crop_input_size_px':[230,190]},
         'input_context':{'registered_peer_shapes':[{'id':'inner','relation_hint':'inside_target'}]}}
    audit=refine(local,job,image,None)
    assert local==before
    assert not audit[0]['applied'] and audit[0]['enclosed_peers']==['inner']


def test_four_source_sides_repair_extent_without_copying_mask_edges():
    image=Image.new('RGB',(230,190),'white');ImageDraw.Draw(image).rounded_rectangle((30,20,180,160),radius=9,outline='black',width=2)
    p={'center':[104,86],'width':134,'height':132,'angle_radians':0.}
    result,audit=measure(p,image,[27,17,183,163])
    assert audit['applied']
    assert np.max(np.abs(np.asarray(result)-[30.5,20.5,179.5,159.5]))<1.5
    assert p['width']==134


def test_mask_on_blank_source_cannot_create_a_rectangle():
    p={'center':[80,80],'width':120,'height':120,'angle_radians':0.}
    result,audit=measure(p,Image.new('RGB',(180,180),'white'),[20,20,140,140])
    assert result is None and not audit['applied']


def test_fill_refinement_preserves_separate_dotted_enclosure():
    from shape_integration.fitting.frame_edges import measure_fill
    image=Image.new('RGB',(260,220),'white');draw=ImageDraw.Draw(image)
    for x in range(20,241,8):
        draw.ellipse((x,20,x+2,22),fill='black')
        draw.ellipse((x,198,x+2,200),fill='black')
    for y in range(20,201,8):
        draw.ellipse((20,y,22,y+2),fill='black')
        draw.ellipse((238,y,240,y+2),fill='black')
    draw.rectangle((40,40,220,180),fill='#99CCFF')
    p={'center':[130,110],'width':220,'height':180,'angle_radians':0.}
    assert measure_fill(p,image,[20,20,240,200])[0] is None
    # The same fill remains useful for a nearby observation of its own body.
    body={**p,'width':176,'height':136}
    result,audit=measure_fill(body,image,[42,42,218,178])
    assert audit['applied'] and np.max(np.abs(np.asarray(result)-[39.5,39.5,220.5,180.5]))<1


def test_supported_existing_side_resolves_parallel_frame_without_blocking_other_repairs():
    image=Image.new('RGB',(640,200),'white');draw=ImageDraw.Draw(image)
    draw.rectangle((20,50,610,150),outline='#456724',width=2)
    draw.line((20,40,610,40),fill='#456724',width=2)
    parameters={'center':[315,104],'width':586,'height':107,'angle_radians':0.}
    result,audit=measure(parameters,image,[19,47,611,151])
    assert audit['applied'] and abs(result[1]-50.5)<1
    assert abs(result[3]-149.5)<1
    assert any('ambiguity_resolved_by_source_supported_prior' in r for r in audit['sides'])
    # An observation between both lines does not choose one. Keep that side,
    # while independent source evidence still corrects the other sides.
    parameters.update(center=[315,99.5],height=109)
    result,audit=measure(parameters,image,[19,45,611,151])
    assert audit['applied'] and result[1]==45.
    assert abs(result[3]-149.5)<1
    assert audit['sides'][1]['retained_observed_position']==45.


def test_local_hint_can_repair_translation_but_cannot_create_ink_or_resize():
    image=Image.new('RGB',(220,160),'white')
    ImageDraw.Draw(image).rounded_rectangle((30,30,180,70),radius=4,outline='#3465AA',width=2)
    p={'center':[105,65],'width':150,'height':40,'angle_radians':0.}
    result,audit=measure(p,image,[29,29,181,71],translation_hint=True)
    assert audit['applied'] and np.max(np.abs(np.asarray(result)-[30.5,30.5,179.5,69.5]))<1.5
    result,audit=measure(p,Image.new('RGB',image.size,'white'),[29,29,181,71],translation_hint=True)
    assert result is None
    resized={**p,'height':64}
    assert measure(resized,image,[29,29,181,71],translation_hint=True)[0] is None


def test_same_pigment_sides_do_not_snap_red_frame_to_black_neighbour():
    image=Image.new('RGB',(220,190),'white');d=ImageDraw.Draw(image)
    d.rectangle((30,20,160,160),outline='#B40733',width=2)
    d.line((166,20,166,160),fill='black',width=3)
    p={'center':[102,90],'width':146,'height':140,'angle_radians':0.}
    result,audit=measure(p,image,[29,19,162,161])
    assert audit['applied'] and abs(result[2]-159.5)<1.5


def test_missing_body_side_is_not_closed_from_sam():
    image=Image.new('RGB',(230,190),'white');d=ImageDraw.Draw(image)
    d.line([(30,20),(180,20),(180,160),(30,160),(30,115)],fill='black',width=2)
    d.line([(30,65),(30,20)],fill='black',width=2)
    p={'center':[105,90],'width':148,'height':138,'angle_radians':0.}
    result,audit=measure(p,image,[30,20,180,160])
    assert result is None and not audit['applied']


def fixture(vertices):
    points=np.asarray(vertices,float)
    part={'id':'frame','kind':'path','status':'ok','commands':[['M',*points[0]],*[['L',*p] for p in points[1:]],['Z']], 'bbox_2d':[points[:,0].min(),points[:,1].min(),points[:,0].max(),points[:,1].max()], 'diagnostics':{}}
    local={'parts':[part]}
    job={'candidate':{'id':'f','bbox_2d':[80,20,230,160]},'frame':{'source_crop_half_open':[0,0,260,200],'actual_crop_input_size_px':[260,200]}}
    evidence=SimpleNamespace(geometry_support=lambda c:{'representative':{'bbox':[78,18,232,162]}})
    return local,job,evidence


def test_observed_pointer_keeps_topology_and_uses_source_line_intersections():
    target=np.array([[80,20],[230,20],[230,160],[80,160],[80,120],[40,110],[80,80]],float)
    image=Image.new('RGB',(260,200),'white');d=ImageDraw.Draw(image)
    for a,b in zip(target,np.roll(target,-1,axis=0)):
        length=np.linalg.norm(b-a)
        for start in np.arange(0,length,8):
            lo=a+(b-a)*start/length;hi=a+(b-a)*min(start+5,length)/length
            d.line([tuple(lo),tuple(hi)],fill='black',width=2)
    wrong=target.copy();wrong[2:4,1]+=7;wrong[4,1]+=8;wrong[6,1]-=9
    local,job,evidence=fixture(wrong);before=copy.deepcopy(local)
    audit=refine_outline(local,job,image,evidence)
    assert audit[0]['applied']
    fitted=np.array([c[1:] for c in local['parts'][0]['commands'] if c[0] in ('M','L')])
    assert len(fitted)==len(target) and np.max(np.linalg.norm(fitted-target,axis=1))<3
    assert before['parts'][0]['commands']!=local['parts'][0]['commands']


def test_no_pointer_is_invented_for_a_rectangle():
    local,job,evidence=fixture([[80,20],[230,20],[230,160],[80,160]])
    original=copy.deepcopy(local)
    assert refine_outline(local,job,Image.new('RGB',(260,200),'white'),evidence)==[]
    assert local['parts']==original['parts']


def frame_group(boxes):
    points=[];parts=[]
    for index,b in enumerate(boxes):
        ids=[f'{index}_{i}' for i in range(4)]
        xy=[[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]]
        points.extend({'id':pid,'xy':p} for pid,p in zip(ids,xy))
        p={'center':[(b[0]+b[2])/2,(b[1]+b[3])/2],
           'width':b[2]-b[0],'height':b[3]-b[1],'angle_radians':0.}
        parts.append({'id':str(index),'kind':'rectangle','status':'ok',
            'parameters':p,'bbox_2d':b,'observed_point_ids':ids,'diagnostics':{},
            'commands':box_commands([*p['center'],p['width'],p['height'],0.])})
    return {'observation':{'points':points},'parts':parts}


def test_overlapping_faces_do_not_snap_to_the_same_foreground_rectangle():
    from shape_integration.fitting.frame_edges import refine
    boxes=[[24,24,164,84],[22,22,162,82],[20,20,160,80]]
    image=Image.new('RGB',(200,110),'white');draw=ImageDraw.Draw(image)
    for b in boxes:draw.rectangle(b,fill='#7ACA8E',outline='#397944',width=2)
    local=frame_group(boxes)
    job={'candidate':{'id':'stack','bbox_2d':[20,20,164,84]},'frame':{
        'actual_crop_input_size_px':[200,110],'global_image_size_px':[200,110],
        'original_image_size_px':[200,110],'crop_bounds_global_half_open':[0,0,200,110],
        'source_crop_half_open':[0,0,200,110]}}
    audits=refine(local,job,image,None)
    result=[p['bbox_2d'] for p in local['parts']]
    assert len({tuple(b) for b in result})==3
    assert result[0][0]>result[1][0]>result[2][0]
    assert result[0][3]>result[1][3]>result[2][3]
    assert all(a['shared_fill_not_isolated'] for a in audits)


def test_adjacent_cells_and_contained_details_keep_the_existing_fit_path():
    from shape_integration.fitting.frame_edges import competing_frame_sides
    assert competing_frame_sides(frame_group([[0,0,20,20],[20,0,40,20]]))=={}
    assert competing_frame_sides(frame_group([[0,0,40,40],[10,10,20,20]]))=={}


def test_face_mask_below_header_cannot_replace_observed_outer_contour():
    from shape_integration.fitting.frame_edges import refine
    image=Image.new('RGB',(230,240),'white');draw=ImageDraw.Draw(image)
    draw.rectangle((30,20,190,220),outline='#249A9A',width=2)
    draw.line((30,50,190,50),fill='#249A9A',width=2)
    local=frame_group([[31,21,189,219]])
    job={'candidate':{'id':'box','bbox_2d':[30,20,190,220]},'frame':{
        'actual_crop_input_size_px':[230,240],'global_image_size_px':[230,240],
        'original_image_size_px':[230,240],'crop_bounds_global_half_open':[0,0,230,240],
        'source_crop_half_open':[0,0,230,240]}}
    evidence=SimpleNamespace(geometry_support=lambda candidate:{'id':'body',
        'representative':{'bbox':[30,50,190,220],'score':.99}})
    refine(local,job,image,evidence)
    assert abs(local['parts'][0]['bbox_2d'][1]-20.5)<1.5


def load_tests(loader,tests,pattern):
    import unittest
    for name,fn in sorted(globals().copy().items()):
        if name.startswith('test_') and callable(fn):
            tests.addTest(unittest.FunctionTestCase(fn,description=name))
    return tests
