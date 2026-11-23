import numpy as np
import pytest
from PIL import Image,ImageDraw
from full_recovery.handoff import ellipsis_shape_ownership
from full_recovery.shape_duplicates import filled_duplicates
from full_recovery.image_assets import native_frames
from shape_integration.styling.occlusion import order
from full_recovery.quality_choice import nominate,protect_parallel_pairs,review,port_node


def part(oid,b,kind='rectangle'):
    x,y,r,t=b
    return {'object_id':oid,'id':oid,'bbox_2d':b,'kind':kind,'status':'ok',
            'native_parameters_original':{'center':[(x+r)/2,(y+t)/2]},
            'commands':[['M',x,y],['L',r,y],['L',r,t],['L',x,t],['Z']]}


def test_punctuation_owns_partial_duplicate_square_dots_without_deleting_nodes(tmp_path):
    im=Image.new('RGB',(100,100),'white');d=ImageDraw.Draw(im)
    boxes=[[46,y,50,y+4] for y in (20,34,48)]
    for b in boxes:d.rectangle(b,fill='black')
    p=tmp_path/'source.png';im.save(p)
    shapes={'first':part('a',boxes[0]),'duplicate':part('b',boxes[0]),
            'last':part('a',boxes[2]),'neighbor':part('node',[60,30,70,40])}
    styles={k:{'fill':'#000000'} for k in shapes}
    rows=[{'id':'T','text':'...','rotation_deg':90,'bbox_2d':[45,19,52,54],'color':'#000000'},
          {'id':'label','text':'x...y','bbox_2d':[45,19,52,54],'color':'#000000'}]
    removed,audit=ellipsis_shape_ownership({'parts':shapes},styles,rows,p)
    assert set(removed)=={'first','duplicate','last'}
    assert [a['text_id'] for a in audit]==['T']


def test_missing_source_dot_does_not_claim_an_ellipsis(tmp_path):
    im=Image.new('RGB',(100,60),'white');d=ImageDraw.Draw(im)
    for x in (20,40):d.rectangle((x,25,x+4,29),fill='black')
    p=tmp_path/'source.png';im.save(p)
    scene={'parts':{'a':part('a',[20,25,24,29])}}
    removed,_=ellipsis_shape_ownership(scene,{'a':{'fill':'#000000'}},
        [{'id':'T','text':'…','bbox_2d':[18,23,65,31]}],p)
    assert not removed


def test_coherently_displaced_ellipsis_has_one_owner_but_real_second_row_is_preserved(tmp_path):
    im=Image.new('RGB',(160,100),'white');d=ImageDraw.Draw(im)
    boxes=[[x,50,x+6,56] for x in (40,60,80)]
    for b in boxes:d.rectangle(b,fill='black')
    shifted=[[x,y-10,r,t-10] for x,y,r,t in boxes]
    scene={'parts':{str(i):part('dots',b) for i,b in enumerate(shifted)}}
    scene['parts']['node']=part('node',[110,30,135,55])
    styles={k:{'fill':'#000000'} for k in scene['parts']}
    rows=[{'id':'T','text':'…','bbox_2d':[42,50,86,57]}]
    p=tmp_path/'source.png';im.save(p)
    removed,_=ellipsis_shape_ownership(scene,styles,rows,p)
    assert set(removed)=={'0','1','2'}
    for b in shifted:d.rectangle(b,fill='black')
    im.save(p)
    removed,_=ellipsis_shape_ownership(scene,styles,rows,p)
    assert not removed


def test_small_filled_duplicate_keeps_adjacent_occurrence_and_other_colour():
    im=Image.new('RGB',(500,500),'white');d=ImageDraw.Draw(im)
    d.rectangle((100,100,109,109),fill='#008800');d.rectangle((115,100,124,109),fill='#008800')
    scene={'parts':{'a':part('compound',[100,100,110,110]),
                    'b':part('recall',[100.2,100,110.2,110]),
                    'next':part('compound',[115,100,125,110]),
                    'other':part('layer',[100,100,110,110])}}
    styles={k:{'fill':'#008800'} for k in scene['parts']};styles['other']={'fill':'#FF0000'}
    removed,audit=filled_duplicates(scene,styles,np.asarray(im))
    assert removed=={'b'} and audit[0]['keep']=='a'


def test_duplicate_selection_retains_native_front_copy_across_an_occluder(tmp_path):
    from pptx import Presentation
    from full_recovery.shape_duplicates import native_paint_order
    image=Image.new('RGB',(500,500),'white')
    ImageDraw.Draw(image).rectangle((100,100,109,109),fill='#008800')
    scene={'parts':{'back':part('back',[100,100,110,110]),
                    'front':part('front',[100.2,100,110.2,110])}}
    styles={key:{'fill':'#008800'} for key in scene['parts']}
    prs=Presentation();slide=prs.slides.add_slide(prs.slide_layouts[6])
    for name in ('back/back','opaque/panel','front/front'):
        sp=slide.shapes.add_textbox(100,100,10,10);sp.name=name+' [accepted_by_model]'
    path=tmp_path/'native.pptx';prs.save(path)
    removed,audit=filled_duplicates(scene,styles,np.asarray(image),paint_order=native_paint_order(path))
    assert removed=={'back'} and audit[0]['keep']=='front'
    assert audit[0]['selection']=='native_paint_order_then_geometry'


@pytest.mark.parametrize('visible', [True, False])
@pytest.mark.parametrize('front_stroke,front_width', [('#666666',1.32),('#131313',2.1)])
@pytest.mark.parametrize('payload_fill', ['#00AA33','#DEEBF6'])
def test_coincident_frames_preserve_source_visible_payload_without_inventing_it(visible,front_stroke,front_width,payload_fill):
    image=Image.new('RGB',(100,100),'white')
    if visible:ImageDraw.Draw(image).rectangle((35,35,54,54),fill=payload_fill)
    scene={'parts':{'rear':part('rear',[20,20,80,70]),
                    'payload':part('payload',[35,35,55,55]),
                    'front':part('front',[20.04,20,80.04,70])}}
    frame={'fill':'#FFFFFF','stroke':'#666666','width_px':1.3}
    styles={'rear':frame,'front':{**frame,'stroke':front_stroke,'width_px':front_width},'payload':{'fill':payload_fill}}
    removed,audit=filled_duplicates(scene,styles,np.asarray(image),
        paint_order={'rear/rear':0,'payload/payload':1,'front/front':2})
    if visible:
        assert removed=={'front'}
        assert audit[0]['preserved_payload_keys']==['payload']
    else:assert not removed


def test_separate_nested_frames_with_visible_payload_are_not_duplicates():
    image=Image.new('RGB',(100,100),'white');draw=ImageDraw.Draw(image)
    draw.rectangle((20,20,80,70),outline='black')
    draw.rectangle((23,23,77,67),outline='black')
    draw.rectangle((35,35,54,54),fill='#00AA33')
    scene={'parts':{'rear':part('rear',[20,20,80,70]),
                    'payload':part('payload',[35,35,55,55]),
                    'front':part('front',[23,23,77,67])}}
    frame={'fill':'#FFFFFF','stroke':'#000000','width_px':1}
    removed,_=filled_duplicates(scene,{'rear':frame,'front':frame,'payload':{'fill':'#00AA33'}},
        np.asarray(image),paint_order={'rear/rear':0,'payload/payload':1,'front/front':2})
    assert not removed


def test_empty_image_realization_does_not_consume_early_native_ownership():
    from copy import deepcopy
    from full_recovery.image_realization import realize
    texture={'assets':[], 'native_patterns':[{'surface_part':'panel'}],
             'drop_shape_keys':['hatch'], 'drop_arrow_names':[]}
    with_plan=deepcopy(texture);without_plan=deepcopy(texture)
    early={'quality_choice':{'source_receipts':[{'source_kind':'line','status':'resolved',
             'candidate_ids':['arrow:a']}]},'image_source_occlusion_order':{'a':{'before':['IMG x']}}}
    # No source file or editor exists: an empty image pass must not use either.
    realize(with_plan,{}, {},None,early,None)
    realize(without_plan,{}, {},None,{'drop_arrow_names':[]},None)
    assert with_plan==without_plan
    assert with_plan['native_patterns']==texture['native_patterns']
    assert with_plan['drop_shape_keys']==['hatch']


@pytest.mark.parametrize('inner_stroke,inner_width,inner_dash', [
    (None,1,'solid'),('#000000',1,'solid'),('#000000',2,'dash')])
def test_fill_component_does_not_replace_unique_border_paint(inner_stroke,inner_width,inner_dash):
    image=Image.new('RGB',(500,500),'white')
    ImageDraw.Draw(image).rectangle((100,100,110,110),fill='#008800',outline='black',width=2)
    scene={'parts':{'border':part('same_owner',[100,100,110,110]),
                    'inner':part('same_owner',[100,100,110,110])}}
    styles={'border':{'fill':'#008800','stroke':'#000000','width_px':2,'dash':'solid'},
            'inner':{'fill':'#008800','stroke':inner_stroke,'width_px':inner_width,'dash':inner_dash}}
    removed,_=filled_duplicates(scene,styles,np.asarray(image),
        paint_order={'same_owner/same_owner':1})
    assert not removed


def test_white_node_duplicates_allow_native_width_measurement_noise():
    from shape_integration.fitting.paths import ellipse_commands
    image=Image.new('RGB',(100,100),'white')
    ImageDraw.Draw(image).ellipse((20,20,80,80),fill='white',outline='black',width=2)
    a=part('a',[20,20,80,80],kind='ellipse')
    b=part('b',[20.5,20,80.5,80],kind='ellipse')
    a['commands']=ellipse_commands([50,50],[30,0],[0,30])
    b['commands']=ellipse_commands([50.5,50],[30,0],[0,30])
    paint={'fill':'#FFFFFF','stroke':'#000000','width_px':1.99,'dash':'solid'}
    removed,audit=filled_duplicates({'parts':{'a':a,'b':b}},
        {'a':paint,'b':{**paint,'width_px':2.01}},np.asarray(image))
    assert len(removed)==1
    assert audit[0]['single_source_outline_fraction']>=.9


@pytest.mark.parametrize('fill',['#FFFFFF','#88CC88'])
def test_source_double_outline_is_preserved_despite_coincident_fill(fill):
    from shape_integration.fitting.paths import ellipse_commands
    image=Image.new('RGB',(100,100),'white');draw=ImageDraw.Draw(image)
    draw.ellipse((20,20,80,80),fill=fill,outline='black',width=1)
    draw.ellipse((22,22,78,78),fill=fill,outline='black',width=1)
    parts={}
    for key,radius in [('outer',30),('inner',28)]:
        item=part(key,[50-radius,50-radius,50+radius,50+radius],kind='ellipse')
        item['commands']=ellipse_commands([50,50],[radius,0],[0,radius]);parts[key]=item
    styles={k:{'fill':fill,'stroke':'#000000','width_px':1.,'dash':'solid'} for k in parts}
    removed,_=filled_duplicates({'parts':parts},styles,np.asarray(image))
    assert not removed


def test_internal_cells_are_one_visual_unit_but_outer_frame_is_protected():
    scene={'parts':{'outer':part('group',[0,0,100,50]),'cell1':part('group',[10,10,40,40]),
                    'cell2':part('group',[60,10,90,40]),'glyph1':part('group',[15,15,30,30]),
                    'glyph2':part('group',[65,15,80,30])}}
    assert native_frames(scene,list(scene['parts']))==['outer']


def test_cycle_does_not_discard_container_order_or_unrelated_relations():
    ids=['front','rear','child','elsewhere'];parts={k:part(k,[0,0,10,10]) for k in ids}
    scene={'objects':{k:{'part_keys':[k]} for k in ids}}
    edges=[{'back':'front','front':'child','origin':'contains'},
           {'back':'child','front':'rear','origin':'model'},
           {'back':'rear','front':'front','origin':'model'},
           {'back':'rear','front':'elsewhere','origin':'model'}]
    result=order(scene,parts,{'scope':'components','edges':edges})
    rank={k:i for i,k in enumerate(result['object_order'])}
    assert rank['rear']<rank['front']<rank['child']
    assert rank['rear']<rank['elsewhere']
    assert len(result['unapplied_edges'])==1


def test_same_nodes_with_wrong_curvature_reach_existing_comparison():
    t=np.linspace(0,1,80)
    def row(layer,bend):
        points=np.c_[20+120*t,60+bend*np.sin(np.pi*t)]
        return {'id':layer,'layer':layer,'points':points,'length':float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),
                'endpoint_nodes':('left','right'),'node_radius_px':5,'heads':['none','none'],
                'style':{'stroke':'#000000','color':'#000000','width_px':2}}
    a,b=row('shape',-25),row('arrows',25);rows=[a,b]
    pairs=nominate(rows)
    assert len(pairs)==1 and pairs[0]['unique_endpoint_pair']
    im=Image.new('RGB',(180,130),'white');draw=ImageDraw.Draw(im)
    draw.line([tuple(p) for p in a['points']],fill='black',width=2)
    kept,protected=protect_parallel_pairs(rows,pairs,np.asarray(im))
    assert kept and not protected  # the unsupported curvature is a competitor
    draw.line([tuple(p) for p in b['points']],fill='black',width=2)
    kept,protected=protect_parallel_pairs(rows,pairs,np.asarray(im))
    assert protected and not kept  # two actual source routes must both survive


def test_detached_ports_require_unique_forward_node_and_bounded_gap():
    nodes=[('right',np.array([100.,50.]),np.array([8.,8.])),
           ('neighbor',np.array([100.,74.]),np.array([8.,8.]))]
    assert port_node(np.array([80.,50.]),np.array([1.,0.]),nodes,[])=='right'
    assert port_node(np.array([80.,50.]),np.array([-1.,0.]),nodes,[]) is None
    assert port_node(np.array([50.,50.]),np.array([1.,0.]),nodes,[]) is None
    assert port_node(np.array([80.,62.]),np.array([1.,0.]),nodes,[]) is None
    duplicate=[nodes[0],('copy',nodes[0][1],nodes[0][2])]
    assert port_node(np.array([80.,50.]),np.array([1.,0.]),duplicate,[]) is None


def test_unmatched_native_content_is_preserved_without_a_second_existence_vote(tmp_path):
    from pptx import Presentation
    from unittest.mock import patch
    source=tmp_path/'source.png';Image.new('RGB',(100,100),'white').save(source)
    scene={'source_size':[100,100], 'parts':{'line':{'id':'line','object_id':'o','kind':'path','status':'ok',
        'commands':[['M',10,10],['L',80,10]],'bbox_2d':[10,10,80,10]}}}
    class SourceReview:
        def call(self,folder,tag,system,prompt,images,validate,budget):
            raise AssertionError('An unmatched native route does not need another model review')
    native=tmp_path/'shape.pptx';prs=Presentation();prs.slides.add_slide(prs.slide_layouts[6]);prs.save(native)
    with patch('shape_integration.styling.export.render_native', return_value=source):
        result=review(scene,{'line':{'stroke':'#000000','width_px':2}},[],source,native,None,tmp_path/'review',SourceReview())
    assert result['model_calls']==0 and not result['drop_shape_keys']
    assert result['unreviewed_route_count']==1


def test_accepted_source_paint_is_not_vetoed_by_displaced_pixel_difference(tmp_path):
    from shapely.geometry import box
    from shape_integration.styling.controller import visibility_changes,commit_refinement
    before=Image.new('RGB',(100,100),'white');after=before.copy()
    ImageDraw.Draw(after).line((20,50,80,50),fill='black',width=3)
    before.save(tmp_path/'before.png');after.save(tmp_path/'after.png');before.save(tmp_path/'source.png')
    old={'face':{'fill':'#FFFFFF'}};new={'face':{'fill':None}}
    checks={'face':{'value':{'verdict':'accept'}}}
    evidence=visibility_changes(tmp_path/'source.png',tmp_path/'before.png',tmp_path/'after.png',
                                old,new,{'face':box(10,10,90,90)})
    assert evidence['face']['regressed']
    assert commit_refinement(old,new,checks,evidence)==(new,[])
    for verdict in ('uncertain','reject'):
        assert commit_refinement(old,new,{'face':{'value':{'verdict':verdict}}},evidence)==(old,['face'])
    revised={'face':{'value':{'verdict':'revise','affected_fields':['fill']}}}
    assert commit_refinement(old,new,revised,evidence)==(old,['face'])
    after.save(tmp_path/'source.png')
    evidence=visibility_changes(tmp_path/'source.png',tmp_path/'before.png',tmp_path/'after.png',
                                old,new,{'face':box(10,10,90,90)})
    assert not evidence['face']['regressed']
    assert commit_refinement(old,new,checks,evidence)==(new,[])
