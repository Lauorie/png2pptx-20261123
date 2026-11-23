"""Overlap review has a small, separate edit boundary."""
import copy
import pytest
from shapely.geometry import box, LineString
from shape_integration.styling.overlap_review import groups, validate


def test_nomination_separates_partial_faces_from_contained_details_and_open_lines():
    table = {k:{} for k in ['face','overlap','detail','wire','separate']}
    geoms = {'face':box(0,0,20,20),'overlap':box(15,15,35,35),
             'detail':box(2,2,6,6),'wire':LineString([(0,10),(40,10)]),
             'separate':box(60,60,80,80)}
    before = copy.deepcopy(table)
    result = groups(table,geoms,{'geometry_tolerance_px':.2,'minimum_overlap_area_px':1})
    assert [g['keys'] for g in result]==[['face','overlap']]
    assert table==before


def test_uncertain_surface_keeps_existing_fill_and_never_reorders():
    mapping = {'P0':'a','P1':'b'}
    styles = {'a':{'fill':'#A0B0C0','stroke':'#000000','width_px':1.7},
              'b':{'fill':None,'stroke':'#FF0000','width_px':2.1}}
    before = copy.deepcopy(styles)
    raw = {'parts':[{'id':'P0','status':'uncertain','fill':None,'behind':['P1'],
                     'source_evidence':'The crossing is obscured by a label.'},
                    {'id':'P1','status':'supported','fill':None,'behind':[],
                     'source_evidence':'The source contour remains visible.'}]}
    result = validate(raw,mapping,styles)
    assert result['parts']['P0']['fill']=='#A0B0C0'
    assert result['parts']['P0']['behind']==[]
    assert validate(result,mapping,styles)==result  # Runtime cache revalidation.
    assert styles==before


def test_missing_fill_and_cyclic_order_cannot_commit_partial_edits():
    mapping = {'P0':'a','P1':'b'};styles = {'a':{'fill':None},'b':{'fill':None}}
    raw = {'parts':[{'id':a,'status':'supported','fill':'#FFFFFF','behind':[b],
                     'source_evidence':'The rear contour stops at the foreground edge.',
                     'rendered_difference':'The actual drawing exposes the rear contour.'}
                    for a,b in [('P0','P1'),('P1','P0')]]}
    with pytest.raises(ValueError,match='Cyclic'):validate(raw,mapping,styles)
    raw['parts'][1]['behind']=[];raw['parts'][0].pop('fill')
    with pytest.raises(ValueError,match='explicit fill'):validate(raw,mapping,styles)


def test_disjoint_geometry_preserves_existing_order_and_never_calls_model(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from shape_integration.styling.overlap_review import review
    scene = {'objects':{'N':{'part_keys':['a','b']}},'parts':{
        'a':{'status':'ok','kind':'path','object_id':'N','commands':[
             ['M',0,0],['L',20,0],['L',20,20],['L',0,20],['Z']]},
        'b':{'status':'ok','kind':'path','object_id':'N','commands':[['M',0,10],['L',40,10]]}}}
    styles = {'a':{'fill':None,'stroke':'#000000','width_px':1.7},
              'b':{'fill':None,'stroke':'#FF0000','width_px':2.1}}
    relations = {'object_order':['N'],'part_orders':{'N':['b','a']},'flat_order':['b','a']}
    before = copy.deepcopy((scene,styles,relations))
    with ThreadPoolExecutor(max_workers=1) as pool:
        proposed,ordered,audit = review(scene,styles,relations,tmp_path/'unused.png',tmp_path/'review',
                        None,{'geometry_tolerance_px':.2,'minimum_overlap_area_px':1},pool)
    assert audit['model_calls']==0 and not audit['changed_fills'] and not audit['order_changed']
    assert (scene,proposed,ordered)==before


@pytest.mark.parametrize('pigment',['#FF0000','#000000','#224488'])
def test_source_observation_distinguishes_opaque_face_from_hollow_overlap(pigment):
    import numpy as np
    from PIL import Image,ImageDraw
    from shape_integration.styling.overlap_review import source_support
    geoms = {'a':box(10,30,60,90),'b':box(30,10,80,70)}
    styles = {k:{'stroke':pigment,'width_px':2} for k in geoms}
    group = {'keys':['a','b'],'pairs':[{'a':'a','b':'b'}]}
    observations = []
    for opaque in [True,False]:
        im=Image.new('RGB',(100,100),'white');draw=ImageDraw.Draw(im)
        draw.rectangle((30,10,80,70),outline=pigment,width=2)
        draw.rectangle((10,30,60,90),fill='white' if opaque else None,outline=pigment,width=2)
        observations.append(source_support(group,geoms,styles,{'P0':'a','P1':'b'},np.asarray(im)))
    opaque,hollow = observations
    assert opaque[0]['inside']['source_pigment_fraction']>.9
    assert opaque[1]['inside']['source_pigment_fraction']<.1
    assert all(x['inside']['source_pigment_fraction']>.9 for x in hollow)
