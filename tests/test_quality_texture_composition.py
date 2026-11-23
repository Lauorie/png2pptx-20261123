"""Ownership conflicts, endpoint safety and clean texture alpha controls."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image,ImageDraw
from lxml import etree
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE

from full_recovery.quality_choice import apply_choices,nominate,protect_parallel_pairs,candidates,comparison_groups,review,validate_choice
from full_recovery.texture_fallback import make_asset
from full_recovery.compose import merge,read_package,segment_subset,subpath_subset,selection_subset,SLIDE,P,A
from full_recovery.handoff import reconcile,surviving_scene
from full_recovery.text_occlusion import arrow_region
from shape_integration.fitting.scene import write_pptx
from shape_integration.composition import inject










def branch_plan(items,pairs,keep='arrows'):
    nodes=sorted({n for p in pairs for n in (p['a'],p['b'])})
    allowed=['shape'] if any(r.get('immutable_border') for r in items if r['id'] in nodes) else ['shape','arrows']
    choice=validate_choice({'keep':keep,'reason':'source evidence'},allowed)
    return apply_choices(items,[{**choice,'candidate_ids':nodes}])


def test_branch_protocol_requires_one_choice_and_preserves_echo_tolerance():
    for keep in ['both','uncertain',None,['shape','arrows']]:
        with pytest.raises(ValueError):validate_choice({'keep':keep,'reason':'source'})
    assert validate_choice({'keep':'shape','reason':'source','allowed_choices':['shape','arrows']})=={'keep':'shape','reason':'source'}
    with pytest.raises(ValueError):validate_choice({'keep':'arrows','reason':'source'},['shape'])


def test_filled_intrinsic_arrow_matches_by_same_node_without_claiming_node_or_other_colour():
    part={'object_id':'o','kind':'path','status':'ok','bbox_2d':[10,16,25,24],
          'commands':[['M',10,20],['L',20,16],['L',25,20],['L',20,24],['Z']]}
    scene={'parts':{'glyph':part,'node':{'object_id':'o','kind':'ellipse','status':'ok','bbox_2d':[0,0,40,40]}}}
    styles={'glyph':{'fill':'#0000FF','stroke':None},'node':{'fill':'#DDDDDD','stroke':'#000000','width_px':1}}
    arrows=[{'id':n,'start':[12,25],'end':[29,27],'heads':['none','triangle'],'style':{'color':c,'width_px':1}}
            for n,c in [('blue','#0000FF'),('red','#FF0000')]]
    rows=candidates(scene,styles,arrows);pairs=nominate(rows)
    assert len(pairs)==1 and {pairs[0]['a'],pairs[0]['b']}=={'shape:glyph','arrow:blue'}
    result=branch_plan(rows,pairs,'arrows')
    assert result['drop_shape_keys']==['glyph'] and not result['drop_arrow_names']


def test_gradient_filled_shaft_is_compared_but_wide_face_is_not():
    def part(height):
        return {'object_id':'o','kind':'path','status':'ok','bbox_2d':[0,0,100,height],
            'commands':[['M',0,0],['L',100,0],['L',100,height],['L',0,height],['Z']]}
    style={'fill':{'type':'linear_gradient','start_color':'#44C28A','end_color':'#4488C0'},'stroke':None}
    arrow={'id':'a','start':[0,3],'end':[82,3],'heads':['none','triangle'],'style':{'color':'#4488C0','width_px':4}}
    rows=candidates({'parts':{'band':part(6),'face':part(50)}},{'band':style,'face':style},[arrow])
    assert {r['key'] for r in rows if r.get('filled_route')}=={'band'}
    assert len(nominate(rows))==1


def test_closed_block_arrow_enters_existing_choice_without_claiming_a_filled_node():
    for vertical_offset in (3,22):
        for stroke in (None,'#FF0000'):
            _check_closed_block_arrow(vertical_offset,stroke)


def _check_closed_block_arrow(vertical_offset,stroke):
    outline=[[0,10],[60,10],[60,0],[100,20],[60,40],[60,30],[0,30]]
    def part(points):
        xy=np.asarray(points)
        return {'object_id':'o','kind':'path','status':'ok',
                'bbox_2d':[*xy.min(0),*xy.max(0)],
                'commands':[['M',*points[0]],*[['L',*p] for p in points[1:]],['Z']]}
    scene={'parts':{'block':part(outline),'node':part([[0,0],[100,0],[100,40],[0,40]])}}
    style={'fill':'#FF0000','stroke':stroke}
    shifted=[[x+2,y+vertical_offset] for x,y in outline]
    arrow={'id':'a','kind':'block_arrow','start':shifted[0],'via':shifted[1:],
           'end':shifted[0],'closed':True,'heads':['none','none'],
           'style':{'color':'#FF0000','width_px':1}}
    rows=candidates(scene,{'block':style,'node':style},[arrow]);pairs=nominate(rows)
    assert not any(r.get('parent_key')=='block' for r in rows)
    assert len(pairs)==1 and {pairs[0]['a'],pairs[0]['b']}=={'shape:block','arrow:a'}
    removed=branch_plan(rows,pairs,'arrows')
    assert removed['drop_shape_keys']==['block'] and not removed['shape_segment_suppression']
    moved={**arrow,'start':[x+300 for x in shifted[0]],'via':[[x+300,y+300] for x,y in shifted[1:]],'end':[x+300 for x in shifted[0]]}
    assert not nominate(candidates(scene,{'block':style,'node':style},[moved]))


def test_misaligned_separate_head_does_not_survive_losing_shaft():
    scene={'parts':{
        'shaft':{'object_id':'o','kind':'path','status':'ok','bbox_2d':[0,0,47,0],
                 'commands':[['M',0,0],['L',47,0]]},
        'head':{'object_id':'o','kind':'path','status':'ok','bbox_2d':[40,3,59,21],
                'commands':[['M',40,3],['L',59,12],['L',40,21],['Z']]}}}
    styles={'shaft':{'stroke':'#44AFB0','width_px':4.8},'head':{'fill':'#449ABC'}}
    arrow={'id':'a','start':[0,-2],'end':[61,-2],'heads':['none','triangle'],
           'style':{'color':'#48BBA2','width_px':5}}
    rows=candidates(scene,styles,[arrow]);pairs=nominate(rows)
    assert len(pairs)==1
    removed=branch_plan(rows,pairs,'arrows')
    assert set(removed['drop_shape_keys'])=={'shaft','head'}


def test_uncontested_arrow_product_needs_no_new_existence_agent(tmp_path):
    source=tmp_path/'source.png';Image.new('RGB',(100,100),'white').save(source)
    arrow={'id':'a','start':[10,10],'end':[80,10],'heads':['none','triangle'],
           'style':{'color':'#000000','width_px':2}}
    class Forbidden:
        def call(self,*args):raise AssertionError('No shape competitor; preserve supplied arrow')
    result=review({'parts':{}},{},[arrow],source,None,None,tmp_path/'review',Forbidden())
    assert result['model_calls']==0 and result['drop_arrow_names']==[]
    assert result['ordinary_line_owner']=='local_competitor_groups' and result['line_ownership_complete']
    assert not result['reviewed_arrow_names']
    assert json.loads((tmp_path/'review/source_only_candidates.json').read_text())==[]


def test_arrowhead_tip_connects_shortened_shaft_to_its_network_node():
    scene={'parts':{
        'line':{'id':'line','object_id':'network','kind':'path','status':'ok','bbox_2d':[5,0,80,0],
                'commands':[['M',5,0],['L',80,0]]},
        'head':{'id':'head','object_id':'network','kind':'path','status':'ok','bbox_2d':[80,-5,95,5],
                'commands':[['M',80,-5],['L',95,0],['L',80,5],['Z']]},
        'left':{'object_id':'network','kind':'rectangle','status':'ok','bbox_2d':[-5,-5,5,5]},
        'right':{'object_id':'network','kind':'rectangle','status':'ok','bbox_2d':[95,-5,105,5]}}}
    rows=candidates(scene,{'line':{'fill':None,'stroke':'#000000','width_px':2,'dash':'dash'},
                           'head':{'fill':'#000000','stroke':None}},[])
    shaft=next(r for r in rows if r['id']=='shape:line')
    assert shaft['endpoint_nodes']==('left','right')
    assert shaft['keys']==['line','head']
    assert np.allclose(shaft['points'][-1],[80,0])  # Evidence grouping never rewrites geometry.


def test_selected_branch_retains_all_its_repeated_parts():
    items=[{'id':n,'layer':branch,'key':n,'keys':[n] if branch=='shape' else [],'refs':[n]}
           for branch,names in [('shape',['s1','s2','s3']),('arrows',['a1','a2'])] for n in names]
    pairs=[{'id':str(i),'a':s,'b':a} for i,(s,a) in enumerate([('s1','a1'),('s2','a1'),('s2','a2'),('s3','a2')])]
    shape=branch_plan(items,pairs,'shape');arrows=branch_plan(items,pairs,'arrows')
    assert shape['drop_shape_keys']==[] and shape['drop_arrow_names']==['a1','a2']
    assert arrows['drop_shape_keys']==['s1','s2','s3'] and arrows['drop_arrow_names']==[]


def test_short_arrow_compares_separate_head_extent_without_mutating_shaft():
    shaft={'object_id':'o','kind':'path','status':'ok','bbox_2d':[30,10,30,25],
           'commands':[['M',30,10],['L',30,25]]}
    head={'object_id':'o','kind':'path','status':'ok','bbox_2d':[24,25,36,35],
          'commands':[['M',24,25],['L',30,35],['L',36,25],['Z']]}
    scene={'parts':{'shaft':shaft,'head':head}}
    styles={'shaft':{'stroke':'#000000','width_px':1},'head':{'fill':'#000000'}}
    arrow={'id':'a','start':[25,10],'end':[25,33],'heads':['none','triangle'],
           'style':{'color':'#000000','width_px':2.5}}
    rows=candidates(scene,styles,[arrow]);before=copy.deepcopy(rows)
    pairs=nominate(rows)
    assert len(pairs)==1 and {pairs[0]['a'],pairs[0]['b']}=={'shape:shaft','arrow:a'}
    for current,original in zip(rows,before):
        assert np.array_equal(current['points'],original['points'])
        assert current['length']==original['length'] and current['style']==original['style']
    # A nearby, distinct terminal cannot join through head extent alone.
    far={**arrow,'id':'far','start':[50,10],'end':[50,33]}
    assert not nominate(candidates(scene,styles,[far]))
    # Two genuinely distinct source strokes remain protected from binary choice.
    source=Image.new('RGB',(80,60),'white');draw=ImageDraw.Draw(source)
    draw.line((25,10,25,35),fill='black',width=1)
    draw.line((30,10,30,35),fill='black',width=1)
    eligible,protected=protect_parallel_pairs(rows,pairs,np.asarray(source))
    assert not eligible and len(protected)==1


def test_different_endpoint_and_colour_are_compared_without_losing_winning_attachment():
    route=np.array([[10,10],[50,10]],float)
    a={'id':'s','key':'s','keys':['s'],'refs':['s'],'layer':'shape','length':40,'points':route,'heads':['none','none'],'style':{'stroke':'#000000'}}
    b={**a,'id':'a','key':'a','keys':[],'refs':['a'],'layer':'arrows','heads':['none','oval'],'style':{'color':'#808080'}}
    pairs=nominate([a,b]);assert len(pairs)==1
    plan=branch_plan([a,b],pairs,'arrows')
    assert plan['drop_shape_keys']==['s'] and plan['drop_arrow_names']==[]
    assert candidates({'parts':{}},{},[{'kind':'block_arrow','contours':[[[0,0],[5,0],[5,5]]]}])==[]


def test_review_can_keep_arrow_and_remove_only_redundant_shape_part():
    part={'id':'line','object_id':'o','status':'ok','kind':'path','commands':[['M',10,10],['L',90,10]],'bbox_2d':[10,10,90,10]}
    style={'stroke':'#000000','fill':None,'width_px':1,'dash':'solid'}
    arrow={'id':'a','start':[10,10],'end':[90,10],'heads':['none','none'],'style':{'width_px':1,'color':'#000000'}}
    plan=reconcile({'parts':{'edge':part}},{'parts':{'edge':{'style':style}}},{'arrows':[arrow]},{'texts':[]},
                   quality={'drop_shape_keys':['edge'],'drop_shape_names':['o/line'],'drop_arrow_names':[],
                            'reviewed_arrow_names':['a'],'model_calls':1})
    assert plan['drop_arrow_names']==[] and plan['drop_shape_names']==['o/line'] and plan['model_calls']==1


def test_two_real_parallel_source_strokes_are_preserved_before_model_review():
    image=Image.new('RGB',(150,100),'white');d=ImageDraw.Draw(image)
    d.line((20,40,120,40),fill='black',width=2);d.line((20,44,120,44),fill='black',width=2)
    a={'id':'s','layer':'shape','length':100,'points':np.array([[20,40],[120,40]]),'heads':['none','none'],'style':{'stroke':'#000000'}}
    b={**a,'id':'a','layer':'arrows','points':np.array([[20,44],[120,44]])}
    nominated=nominate([a,b]);assert len(nominated)==1
    pairs,protected=protect_parallel_pairs([a,b],nominated,np.asarray(image))
    assert not pairs and len(protected)==1


def test_independent_source_strokes_are_preserved_even_with_shared_endpoints():
    image=Image.new('RGB',(150,100),'white');draw=ImageDraw.Draw(image)
    draw.line((20,40,120,40),fill='black',width=2)
    draw.line((20,44,120,44),fill='black',width=2)
    a={'id':'s','layer':'shape','length':100,'points':np.array([[20,40],[120,40]]),
       'heads':['none','none'],'style':{'stroke':'#000000'},
       'endpoint_nodes':('left','right'),'node_radius_px':10}
    b={**a,'id':'a','layer':'arrows','points':np.array([[20,44],[120,44]])}
    pairs=nominate([a,b]);assert len(pairs)==1
    eligible,protected=protect_parallel_pairs([a,b],pairs,np.asarray(image))
    assert not eligible and protected==pairs


def test_distinct_coloured_parallel_strokes_still_require_two_source_bands():
    source=Image.new('RGB',(150,100),'white');draw=ImageDraw.Draw(source)
    draw.line((20,40,120,40),fill='#FF0000',width=2)
    a={'id':'s','layer':'shape','length':100,'points':np.array([[20,40],[120,40]]),
       'heads':['none','none'],'style':{'stroke':'#FF0000'}}
    b={**a,'id':'a','layer':'arrows','points':np.array([[20,44],[120,44]]),'style':{'color':'#0000FF'}}
    pairs=nominate([a,b]);assert len(pairs)==1
    assert not protect_parallel_pairs([a,b],pairs,np.asarray(source))[1]
    draw.line((20,44,120,44),fill='#0000FF',width=2)
    eligible,protected=protect_parallel_pairs([a,b],pairs,np.asarray(source))
    assert not eligible and len(protected)==1


def test_dropping_one_native_group_child_preserves_sibling_xml_and_transform(tmp_path):
    p=Presentation();slide=p.slides.add_slide(p.slide_layouts[6]);g=slide.shapes.add_group_shape();g.name='network'
    a=g.shapes.add_shape(MSO_SHAPE.RECTANGLE,100,200,300,400);a.name='network/duplicate'
    b=g.shapes.add_shape(MSO_SHAPE.OVAL,200,300,400,500);b.name='network/node'
    source=tmp_path/'source.pptx';p.save(source)
    before=etree.fromstring(read_package(source)[SLIDE]);node=next(n for n in before.iter('{'+P+'}sp') if n.find('.//{'+P+'}cNvPr').get('name')=='network/node')
    original=etree.tostring(node);transform=etree.tostring(before.find('.//{'+P+'}grpSp/{'+P+'}grpSpPr'))
    out=tmp_path/'out.pptx';merge(source,[],out,exclude_base_names=['network/duplicate'])
    after=etree.fromstring(read_package(out)[SLIDE]);names=[n.get('name') for n in after.iter('{'+P+'}cNvPr')]
    assert 'network/duplicate' not in names and 'network/node' in names
    sibling=next(n for n in after.iter('{'+P+'}sp') if n.find('.//{'+P+'}cNvPr').get('name')=='network/node')
    assert etree.tostring(sibling)==original
    assert etree.tostring(after.find('.//{'+P+'}grpSp/{'+P+'}grpSpPr'))==transform


def test_callout_segment_winner_does_not_delete_the_closed_frame():
    a={'id':'A','layer':'shape','key':'virtual','keys':[],'refs':['frame/outline'],'segment':4,
       'points':np.array([[0.,0.],[20.,20.]])}
    b={'id':'B','layer':'arrows','key':'arrow','keys':[],'refs':['arrow'],
       'points':np.array([[0.,0.],[20.,20.]])}
    result=branch_plan([a,b],[{'id':'ab','a':'A','b':'B'}])
    assert not result['drop_shape_keys'] and not result['drop_shape_names']
    assert result['shape_segment_suppression']=={'frame/outline':[4]}


def test_same_owner_duplicate_lines_and_triangle_attachment_are_nominated_together():
    line={'id':'shaft','object_id':'compound','status':'ok','kind':'path',
          'commands':[['M',10,20],['L',90,20]],'bbox_2d':[10,20,90,20]}
    head={'id':'head','object_id':'compound','status':'ok','kind':'path',
          'commands':[['M',88,18],['L',94,20],['L',88,22],['Z']],'bbox_2d':[88,18,94,22]}
    style={'stroke':'#000000','fill':None,'width_px':1,'dash':'solid'}
    scene={'parts':{'s':line,'h':head}}
    styles={'s':style,'h':{**style,'fill':'#000000'}}
    arrow={'id':'a','start':[10,20],'end':[90,20],'heads':['none','triangle'],
           'style':{'color':'#000000','width_px':1}}
    items=candidates(scene,styles,[arrow]);pairs=nominate(items)
    shape=next(r for r in items if r['id']=='shape:s')
    assert shape['heads']==['none','triangle'] and shape['keys']==['s','h']
    pair=next(p for p in pairs if {p['a'],p['b']}=={'shape:s','arrow:a'})
    result=branch_plan(items,[pair])
    assert result['drop_shape_keys']==['h','s'] and not result['drop_arrow_names']
    # A shared head does not turn an unmatched shaft into a duplicate.
    scene['parts']['second']={**line,'id':'second','commands':[['M',10,30],['L',90,20]]}
    styles['second']=style
    shared=candidates(scene,styles,[])
    assert next(r for r in shared if r['id']=='shape:s')['heads']==['none','triangle']
    from full_recovery.quality_choice import candidate_components
    assert candidate_components(shared)['shape:s']=={'shape:s'}
    shared_plan=apply_choices(shared,[{'keep':'arrows','candidate_ids':['shape:s']}])
    assert shared_plan['drop_shape_keys']==['s']
    assert 'compound/head' not in shared_plan['drop_shape_names']
    scene={'parts':{'a':line,'b':{**line,'id':'other'}}}
    assert nominate(candidates(scene,{'a':style,'b':style},[]))






def test_coupled_comparisons_are_never_split_across_decisions():
    pairs=[{'id':'ab','a':'A','b':'B'},{'id':'cd','a':'C','b':'D'},
           {'id':'bc','a':'B','b':'C'},{'id':'ef','a':'E','b':'F'}]
    groups=comparison_groups(pairs)
    assert [[p['id'] for p in g] for g in groups]==[['ab','bc','cd'],['ef']]


def test_single_known_endpoint_does_not_delete_unmatched_native_route():
    from full_recovery.quality_choice import candidate_components
    part={'object_id':'net','kind':'path','status':'ok','commands':[['M',20,20],['L',100,20]]}
    scene={'parts':{'shaft':{**part,'id':'shaft'},
        'node':{'object_id':'net','kind':'ellipse','status':'ok','bbox_2d':[90,10,110,30]}}}
    style={'stroke':'#000000','fill':None,'width_px':1}
    arrows=[{'id':name,'start':start,'end':[100,20],'heads':['none','none'],
             'style':{'color':'#000000','width_px':1}} for name,start in [('matched',[20,20]),('unpaired',[20,65])]]
    rows=candidates(scene,{'shaft':style},arrows);pairs=nominate(rows)
    assert len(pairs)==1
    components=candidate_components(rows)
    members=sorted(components[pairs[0]['a']]|components[pairs[0]['b']])
    assert 'arrow:unpaired' not in members
    plan=apply_choices(rows,[{'keep':'shape','candidate_ids':members}])
    assert plan['drop_arrow_names']==['matched']


def test_compound_shaft_head_is_removed_whole_regardless_of_wrong_colour(tmp_path):
    for index,fill in enumerate([None,'#33ED61','#C04080']):
        folder=tmp_path/str(index);folder.mkdir()
        check_compound_head(folder,fill)


def check_compound_head(tmp_path,fill):
    part={'object_id':'net','kind':'path','status':'ok','native_parameters_original':None}
    scene={'source_size':[150,100],'parts':{
        'shaft':{**part,'id':'shaft','bbox_2d':[10,20,100,70],
                 'commands':[['M',10,20],['L',100,20],['M',10,70],['L',100,70]]},
        'head':{**part,'id':'head','bbox_2d':[90,14,104,26],
                'commands':[['M',90,14],['L',104,20],['L',90,26],['Z']]}},
        'objects':{'net':{'id':'net','state':'accepted_by_model','part_keys':['shaft','head']}}}
    style={'stroke':'#000000','fill':None,'width_px':1}
    arrow={'id':'a','start':[10,20],'end':[100,20],'heads':['none','triangle'],
           'style':{'color':'#000000','width_px':1}}
    rows=candidates(scene,{'shaft':style,'head':{**style,'fill':fill}},[arrow])
    assert not any(r.get('parent_key')=='head' or r['key']=='head' for r in rows)
    pairs=nominate(rows);plan=branch_plan(rows,pairs)
    assert plan['drop_shape_names']==['net/head'] and plan['drop_shape_keys']==['head']
    assert plan['shape_subpath_suppression']=={'net/shaft':[0]}
    assert not plan['shape_segment_suppression']
    ppt=tmp_path/'source.pptx';out=tmp_path/'merged.pptx';write_pptx(scene,ppt)
    selected=tmp_path/'comparison.pptx'
    selection_subset(ppt,selected,[r for r in rows if r['id']=='shape:shaft::subpath0'])
    original=etree.fromstring(read_package(ppt)[SLIDE]);shown=etree.fromstring(read_package(selected)[SLIDE])
    original_shapes=list(original.iter('{'+P+'}sp'));shown_shapes=list(shown.iter('{'+P+'}sp'))
    assert len(shown_shapes)==2
    assert [s.find('.//{'+P+'}cNvPr').get('name') for s in shown_shapes]==[
        s.find('.//{'+P+'}cNvPr').get('name') for s in original_shapes]
    assert etree.tostring(shown_shapes[1])==etree.tostring(original_shapes[1])  # Complete head/paint/order.
    assert len(shown_shapes[0].findall('.//{'+A+'}moveTo'))==1
    merge(ppt,[],out,exclude_base_names=plan['drop_shape_names'],suppress_base_subpaths=plan['shape_subpath_suppression'])
    root=etree.fromstring(read_package(out)[SLIDE]);names=[n.get('name') for n in root.iter('{'+P+'}cNvPr')]
    assert not any(n.startswith('net/head') for n in names)
    paths=root.findall('.//{'+A+'}custGeom/{'+A+'}pathLst/{'+A+'}path')
    assert len(paths)==1 and len(paths[0].findall('{'+A+'}moveTo'))==1


def test_whole_route_through_nodes_and_visible_gap_share_one_selection():
    from full_recovery.quality_choice import candidate_components
    scene={'parts':{'shaft':{'object_id':'net','kind':'path','status':'ok',
        'commands':[['M',40,30],['L',60,30]]},
        **{str(x):{'object_id':'net','kind':'ellipse','status':'ok','bbox_2d':[x-10,20,x+10,40]} for x in [30,70]}}}
    arrows=[{'id':'whole','start':[0,30],'end':[100,30],'heads':['none','none'],'style':{'color':'#000000','width_px':1}}]
    rows=candidates(scene,{'shaft':{'fill':None,'stroke':'#000000','width_px':1}},arrows)
    assert not next(r for r in rows if r['layer']=='arrows').get('endpoint_nodes')
    assert candidate_components(rows)['shape:shaft']=={'shape:shaft'}
    pairs=nominate(rows)
    assert len(pairs)==1
    from full_recovery.quality_choice import allowed_branches
    assert allowed_branches(pairs,rows,scene,{})==['shape','arrows']  # SOURCE, not length, decides which tail exists.


def test_contradictory_overlapping_branch_choices_fail_before_export():
    rows=[{'id':'s','layer':'shape','keys':['s'],'refs':['s']},{'id':'a','layer':'arrows','key':'a','keys':[],'refs':['a']}]
    with pytest.raises(ValueError,match='disagree'):
        apply_choices(rows,[{'keep':'shape','candidate_ids':['s','a']},{'keep':'arrows','candidate_ids':['s','a']}])


def test_converging_native_routes_remain_independent_without_duplicate_matches():
    from full_recovery.quality_choice import candidate_components
    def row(ident,points):
        return {'id':ident,'layer':'arrows','points':np.array(points,float),'style':{'width_px':1}}
    rows=[row('upper',[[20,10],[100,50]]),row('lower',[[20,90],[100,50]]),
          row('crossing',[[60,0],[60,100]]),row('nearby',[[20,94],[100,54]])]
    graph=candidate_components(rows)
    assert graph['upper']=={'upper'} and graph['lower']=={'lower'}
    assert graph['crossing']=={'crossing'} and graph['nearby']=={'nearby'}


def test_short_thick_routes_are_nominated_by_stroke_footprint():
    a={'id':'s','layer':'shape','length':50,'points':np.array([[10,20],[60,20]],float),'style':{'stroke':'#000000','width_px':2}}
    b={**a,'id':'a','layer':'arrows','points':np.array([[10,23],[60,23]],float)}
    assert len(nominate([a,b]))==1  # Still only a proposal, never automatic deletion.


def test_same_branch_native_review_needs_one_call_and_cannot_invent_ids(tmp_path):
    from unittest.mock import patch
    from pptx.enum.shapes import MSO_CONNECTOR
    p=Presentation();slide=p.slides.add_slide(p.slide_layouts[6])
    for name in ('o/a','o/b','o/c'):
        line=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,10,20,90,20);line.name=name
    for name,x in [('o/left',0),('o/right',60)]:
        node=slide.shapes.add_shape(MSO_SHAPE.OVAL,x,0,40,60);node.name=name
    shapes=tmp_path/'shapes.pptx';p.save(shapes)
    p=Presentation();slide=p.slides.add_slide(p.slide_layouts[6])
    line=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,10,20,90,20);line.name='arrow'
    arrows=tmp_path/'arrows.pptx';p.save(arrows)
    part={'object_id':'o','kind':'path','status':'ok','commands':[['M',10,20],['L',90,20]]}
    scene={'source_size':[100,60],'parts':{'a':{**part,'id':'a'},'b':{**part,'id':'b'},'c':{**part,'id':'c'},
        **{name:{'object_id':'o','kind':'ellipse','status':'ok','bbox_2d':bounds}
           for name,bounds in [('left',[0,0,40,60]),('right',[60,0,100,60])]}}}
    style={'stroke':'#000000','fill':None,'width_px':1}
    route={'id':'arrow','start':[10,20],'end':[90,20],'heads':['none','none'],
           'style':{'color':'#000000','width_px':1}}
    source=tmp_path/'source.png';im=Image.new('RGB',(100,60),'white')
    # This node detail is outside the line-only crop, but inside node context.
    ImageDraw.Draw(im).rectangle((18,54,22,58),fill='#008000');im.save(source)
    rows=candidates(scene,{'a':style,'b':style,'c':style},[])
    assert all(r.get('endpoint_nodes') for r in rows)
    # Simulate one branch's uncertain endpoint matching without removing its line.
    next(r for r in rows if r['id']=='shape:c').pop('endpoint_nodes')
    def render(ppt,folder,size,timeout):
        # Both renders must include identical nodes above their own line objects.
        tree=etree.fromstring(read_package(ppt)[SLIDE])
        names=[n.get('name','') for n in tree.iter('{'+P+'}cNvPr')]
        assert names[-2:]==['o/left','o/right']
        original=etree.fromstring(read_package(shapes)[SLIDE])
        assert all(etree.tostring(n) in [etree.tostring(p) for p in original.findall('.//{'+P+'}sp/{'+P+'}spPr')]
                   for n in tree.findall('.//{'+P+'}sp/{'+P+'}spPr'))
        folder.mkdir(parents=True,exist_ok=True);path=folder/'native.png'
        Image.new('RGB',size,'white').save(path);return path
    class Model:
        calls=0
        def for_stage(self,**overrides):return self
        def call(self,folder,tag,system,prompt,images,validate,budget):
            self.calls+=1
            assert tag=='review'
            assert len(images)==2 and set(prompt['candidates'])=={'C1','C2','C3'}
            assert 'source_objects' not in prompt
            output={'keep_ids':['C1'],'reason':'Complete source connection retained'}
            with pytest.raises(ValueError):validate({'keep':'both','reason':'similar quality'})
            bad=copy.deepcopy(output);bad['keep_ids']=['C99']
            with pytest.raises(ValueError):validate(bad)
            bad=copy.deepcopy(output);bad['keep_ids']=[]
            with pytest.raises(ValueError):validate(bad)
            validated=validate(output)
            assert validate(validated)==validated  # Runtime resumes validated checkpoints.
            return validated
    model=Model()
    with patch('shape_integration.styling.export.render_native',render), \
            patch('full_recovery.quality_choice.candidates',return_value=rows), \
            patch('full_recovery.line_group_choice.eligible',return_value=False):
        result=review(scene,{'a':style,'b':style,'c':style},[],source,shapes,arrows,tmp_path/'review',model)
    assert model.calls==1 and result['model_calls']==1 and result['reviewed_pairs']==3
    assert result['source_only_observation_calls']==0
    assert len(result['removed'])==2 and not result['conflicting_components_retained']
    assert len(result['branch_choices'])==1
    class Undecided:
        def for_stage(self,**overrides):return self
        def call(self,folder,tag,system,prompt,images,validate,budget):
            return validate({'keep':'both','reason':'cannot choose'})
    with patch('shape_integration.styling.export.render_native',render), \
            patch('full_recovery.quality_choice.candidates',return_value=rows), \
            patch('full_recovery.line_group_choice.eligible',return_value=False):
        retained=review(scene,{'a':style,'b':style,'c':style},[],source,shapes,arrows,tmp_path/'failure',Undecided())
    incomplete=json.loads((tmp_path/'failure/audit.json').read_text())
    assert incomplete['failed_components'] and not incomplete['line_ownership_complete']
    assert not retained['removed'] and len(retained['retained_shape_keys'])==3
    assert not (tmp_path/'failure/combined.pptx').exists()


def test_shared_boundaries_of_adjacent_filled_faces_keep_their_native_outlines():
    a={**rect('cube',[(10,10),(110,10),(110,80),(10,80)]),'id':'left','status':'ok'}
    b={**rect('cube',[(110,10),(210,10),(210,80),(110,80)]),'id':'right','status':'ok'}
    style={'stroke':'#000000','fill':'#DDDDDD','width_px':2}
    scene={'parts':{'a':a,'b':b}};styles={'a':style,'b':style}
    assert not nominate(candidates(scene,styles,[]))
    # A separate line duplicating that common edge remains reviewable.
    scene['parts']['line']={'object_id':'cube','id':'line','kind':'path','status':'ok',
                           'commands':[['M',110,10],['L',110,80]]}
    styles['line']={**style,'fill':None}
    assert not nominate(candidates(scene,styles,[]))
    arrow={'id':'a','start':[110,10],'end':[110,80],'heads':['none','none'],
           'style':{'color':'#000000','width_px':2}}
    assert len(nominate(candidates(scene,styles,[arrow])))==3


def test_native_rectangle_edge_evidence_retains_source_transform_and_paint(tmp_path):
    p=Presentation();slide=p.slides.add_slide(p.slide_layouts[6])
    shape=slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,100,200,900,400);shape.name='frame'
    shape.rotation=17;shape.line.width=12700
    source=tmp_path/'source.pptx';out=tmp_path/'edge.pptx';p.save(source)
    before=etree.fromstring(read_package(source)[SLIDE]).find('.//{'+P+'}sp/{'+P+'}spPr')
    segment_subset(source,out,['frame'],0)
    after=etree.fromstring(read_package(out)[SLIDE]).find('.//{'+P+'}sp/{'+P+'}spPr')
    for tag in ('xfrm','ln'):
        assert etree.tostring(before.find('{'+A+'}'+tag))==etree.tostring(after.find('{'+A+'}'+tag))
    paths=after.find('{'+A+'}custGeom/{'+A+'}pathLst')
    assert paths[0].get('stroke')=='false'
    assert len(paths[1].findall('{'+A+'}lnTo'))==1 and after.find('{'+A+'}noFill') is not None
    assert etree.fromstring(read_package(source)[SLIDE]).find('.//{'+A+'}prstGeom').get('prst')=='rect'


def test_disconnected_route_ownership_never_deletes_its_unrelated_sibling(tmp_path):
    part={'object_id':'network','id':'fan','kind':'path','status':'ok','native_parameters_original':None,
          'commands':[['M',10,10],['L',100,50],['M',10,10],['C',40,10,80,70,100,90]],
          'bbox_2d':[10,10,100,90]}
    scene={'source_size':[120,110],'parts':{'fan':part},
           'objects':{'network':{'id':'network','state':'accepted_by_model','part_keys':['fan']}}}
    style={'stroke':'#000000','fill':None,'width_px':1}
    arrow={'id':'route','start':[10,10],'end':[100,50],'heads':['none','none'],
           'style':{'color':'#000000','width_px':1}}
    rows=candidates(scene,{'fan':style},[arrow]);pairs=nominate(rows)
    assert len(pairs)==1 and '::subpath0' in pairs[0]['a']
    plan=branch_plan(rows,pairs)
    assert plan['shape_subpath_suppression']=={'network/fan':[0]}
    assert not plan['drop_shape_keys'] and not plan['drop_shape_names']
    effective=surviving_scene(scene,plan)
    assert effective['parts']['fan']['commands']==part['commands'][2:]
    source=tmp_path/'source.pptx';out=tmp_path/'out.pptx';isolated=tmp_path/'evidence.pptx'
    write_pptx(scene,source)
    before=etree.fromstring(read_package(source)[SLIDE]);props=before.find('.//{'+P+'}sp/{'+P+'}spPr')
    path=props.find('{'+A+'}custGeom/{'+A+'}pathLst/{'+A+'}path')
    untouched=[etree.tostring(c) for c in list(path)[2:]]
    result=merge(source,[],out,suppress_base_subpaths=plan['shape_subpath_suppression'])
    after=etree.fromstring(read_package(out)[SLIDE]).find('.//{'+P+'}sp/{'+P+'}spPr')
    retained=after.find('{'+A+'}custGeom/{'+A+'}pathLst/{'+A+'}path')
    assert [etree.tostring(c) for c in retained]==untouched
    for tag in ('xfrm','ln'):
        assert etree.tostring(after.find('{'+A+'}'+tag))==etree.tostring(props.find('{'+A+'}'+tag))
    assert result['native_subpath_ownership'][0]['suppressed_subpaths']==[0]
    subpath_subset(source,isolated,['network/fan'],1)
    retained=etree.fromstring(read_package(isolated)[SLIDE]).find('.//{'+A+'}path')
    assert [etree.tostring(c) for c in retained]==untouched
    # Handing both routes to arrows removes the now-empty parent explicitly.
    rows.append({'id':'arrow:curve','layer':'arrows','key':'curve','keys':[],'refs':['curve']})
    second={'id':'curve','a':'shape:fan::subpath1','b':'arrow:curve'}
    all_plan=branch_plan(rows,[*pairs,second])
    assert all_plan['drop_shape_names']==['network/fan'] and all_plan['drop_shape_keys']==['fan']
    assert not all_plan['shape_subpath_suppression']


def test_exhausted_disconnected_part_matches_scene_and_keeps_owner_siblings(tmp_path):
    part={'object_id':'network','id':'fan','kind':'path','status':'ok','native_parameters_original':None,
          'commands':[['M',10,10],['L',100,50],['M',10,20],['L',100,90]],
          'bbox_2d':[10,10,100,90]}
    sibling={**part,'id':'other','commands':[['M',15,70],['L',40,70]],'bbox_2d':[15,70,40,70]}
    scene={'source_size':[120,110],'parts':{'fan':part,'other':sibling},
           'objects':{'network':{'id':'network','state':'accepted_by_model','part_keys':['fan','other']}}}
    # Independent internal and cross-branch decisions can exhaust one leaf.
    plan={'shape_subpath_suppression':{'network/fan':sorted({0}|{1})}}
    assert list(surviving_scene(scene,plan)['parts'])==['other']
    source=tmp_path/'source.pptx';out=tmp_path/'out.pptx';write_pptx(scene,source)
    original=source.read_bytes()
    before=etree.fromstring(read_package(source)[SLIDE])
    get=lambda root:next(s for s in root.iter('{'+P+'}sp')
                         if s.find('.//{'+P+'}cNvPr').get('name','').startswith('network/other'))
    preserved=etree.tostring(get(before))
    result=merge(source,[],out,suppress_base_subpaths=plan['shape_subpath_suppression'])
    after=etree.fromstring(read_package(out)[SLIDE])
    assert etree.tostring(get(after))==preserved
    assert not any(n.get('name','').startswith('network/fan')for n in after.iter('{'+P+'}cNvPr'))
    assert result['native_subpath_ownership'][0]['whole_native_leaf_removed']
    assert source.read_bytes()==original


def test_same_node_route_with_offset_is_reviewed_but_not_automatically_deleted():
    part={'object_id':'network','id':'edge','kind':'path','status':'ok',
          'commands':[['M',30,45],['L',170,115]]}
    scene={'parts':{'line':part,'a':{'object_id':'network','kind':'ellipse','status':'ok','bbox_2d':[10,10,50,50]},
                    'b':{'object_id':'network','kind':'ellipse','status':'ok','bbox_2d':[150,110,190,150]}}}
    style={'stroke':'#000000','fill':None,'width_px':1}
    arrow={'id':'arrow','start':[40,40],'end':[160,120],'heads':['none','none'],
           'style':{'color':'#000000','width_px':1}}
    rows=candidates(scene,{'line':style},[arrow]);pairs=nominate(rows)
    assert len(pairs)==1 and pairs[0]['nomination']=='same_endpoint_nodes'
    assert not branch_plan(rows,pairs,'shape')['drop_shape_keys']
    arrow['end']=[160,70]  # a different recovered neuron cannot own this edge.
    assert not nominate(candidates(scene,{'line':style},[arrow]))


def test_node_fill_hidden_extensions_do_not_exclude_the_same_connection():
    scene={'parts':{'line':{'object_id':'network','kind':'path','status':'ok',
                           'commands':[['M',30,30],['L',170,130]]},
                    'left':{'kind':'ellipse','status':'ok','bbox_2d':[0,0,60,60]},
                    'right':{'kind':'ellipse','status':'ok','bbox_2d':[140,100,200,160]}}}
    style={'stroke':'#000000','fill':None,'width_px':1}
    arrow={'id':'visible_shaft','start':[54,47],'end':[146,113],
           'heads':['none','none'],'style':{'color':'#000000','width_px':1}}
    rows=candidates(scene,{'line':style},[arrow]);pairs=nominate(rows)
    assert len(pairs)==1 and pairs[0]['nomination']=='same_endpoint_nodes'
    assert branch_plan(rows,pairs,'arrows')['drop_shape_keys']==['line']
    arrow.update(start=[5,40],end=[190,105])  # Different boundary attachments change the fitted angle.
    assert len(nominate(candidates(scene,{'line':style},[arrow])))==1
    arrow['end']=[146,60]
    assert not nominate(candidates(scene,{'line':style},[arrow]))


def test_box_connectors_share_endpoint_nomination_despite_boundary_fit_offsets():
    for kind in ('rectangle','rounded_rectangle'):
        scene={'parts':{
            'line':{'object_id':'diagram','kind':'path','status':'ok',
                    'commands':[['M',40,40],['L',180,130]]},
            'outer':{'kind':kind,'status':'ok','bbox_2d':[0,0,80,40]},
            'inner':{'kind':kind,'status':'ok','bbox_2d':[10,10,70,30]},
            'right':{'kind':kind,'status':'ok','bbox_2d':[140,130,220,170]}}}
        style={'stroke':'#000000','fill':None,'width_px':1}
        arrow={'id':'connector','start':[40,28],'end':[180,142],
               'heads':['none','none'],'style':{'color':'#000000','width_px':1}}
        rows=candidates(scene,{'line':style},[arrow]);pairs=nominate(rows)
        assert len(pairs)==1 and pairs[0]['nomination']=='same_endpoint_nodes'
        assert all(r['endpoint_nodes']==('outer','right') for r in rows)
        assert branch_plan(rows,pairs,'shape')['drop_arrow_names']==['connector']
        arrow['end']=[180,60]
        assert not nominate(candidates(scene,{'line':style},[arrow]))


def test_mixed_box_circle_connector_uses_specific_node_instead_of_group_frame():
    scene={'parts':{'line':{'object_id':'diagram','kind':'path','status':'ok',
                           'commands':[['M',30,30],['L',170,130]]},
                    'left':{'kind':'rounded_rectangle','status':'ok','bbox_2d':[0,0,60,60]},
                    'right':{'kind':'ellipse','status':'ok','bbox_2d':[140,100,200,160]},
                    'group':{'kind':'rectangle','status':'ok','bbox_2d':[100,90,240,170]}}}
    style={'stroke':'#000000','fill':None,'width_px':1}
    arrow={'id':'visible','start':[54,47],'end':[146,113],
           'heads':['none','none'],'style':{'color':'#808080','width_px':1}}
    rows=candidates(scene,{'line':style},[arrow])
    assert all(v['endpoint_nodes']==('left','right') for v in rows)
    assert len(nominate(rows))==1


def test_filled_or_closed_subpaths_cannot_be_deleted_as_disconnected_lines():
    part={'object_id':'o','id':'mixed','kind':'path','status':'ok',
          'commands':[['M',10,10],['L',90,10],['M',10,40],['L',90,40]]}
    filled={'stroke':'#000000','fill':'#EEEEEE','width_px':1}
    assert not candidates({'parts':{'k':part}},{'k':filled},[])
    part['commands']+= [['L',90,80],['L',10,80],['Z']]
    assert not any('subpath' in r for r in candidates({'parts':{'k':part}},
                    {'k':{**filled,'fill':None}},[]))


def test_native_callout_stroke_handoff_retains_closed_fill_coordinates_and_other_sides(tmp_path):
    points=[[80,20],[230,20],[230,160],[80,160],[80,120],[40,110],[80,80]]
    part={**rect('frame',points),'id':'outline','status':'ok','native_parameters_original':None}
    scene={'source_size':[260,200],'parts':{'f':part},'objects':{'frame':{'id':'frame','part_keys':['f'],'state':'accepted_by_model'}}}
    source=tmp_path/'source.pptx';out=tmp_path/'after.pptx';write_pptx(scene,source)
    before=etree.fromstring(read_package(source)[SLIDE]);ns={'p':P,'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
    paths=before.find('.//a:custGeom/a:pathLst',ns);coordinates=[dict(p.attrib) for p in paths[0].iter('{'+ns['a']+'}pt')]
    audit=merge(source,[],out,suppress_base_segments={'frame/outline':[4]})
    after=etree.fromstring(read_package(out)[SLIDE]);newpaths=after.find('.//a:custGeom/a:pathLst',ns)
    assert [dict(p.attrib) for p in newpaths[0].iter('{'+ns['a']+'}pt')]==coordinates
    assert newpaths[0].get('stroke')=='false' and newpaths[0][-1].tag=='{'+ns['a']+'}close'
    assert newpaths[1].get('stroke')=='true' and len(newpaths[1].findall('a:lnTo',ns))==6
    assert audit['native_border_segment_ownership'][0]['suppressed_segments']==[4]


def rect(oid,points):
    return {'object_id':oid,'kind':'path','commands':[['M',*points[0]],*[['L',*p] for p in points[1:]],['Z']]}


def test_texture_is_exact_source_pixels_and_protects_boundary_arrow_head_text_peer_image(tmp_path):
    image=Image.new('RGB',(100,100),'#E0D8C0');d=ImageDraw.Draw(image)
    for k in range(0,100,4):d.line([(0,k),(k,0)],fill='black')
    source=tmp_path/'source.png';image.save(source)
    owner=rect('o',[(10,10),(90,10),(90,90),(10,90)])
    peer=rect('p',[(60,60),(75,60),(75,75),(60,75)])
    scene={'parts':{'surface':owner,'peer':peer}}
    styles={'surface':{'width_px':2,'stroke':'#000000'},'peer':{'fill':'#FF0000','stroke':'#FF0000','width_px':1}}
    arrow={'start':[15,40],'end':[75,40],'heads':['none','oval'],'style':{'width_px':3}}
    decision={'surface_part':'surface','replace_parts':[],'reason':'missing dense hatch'}
    block={'kind':'block_arrow','contours':[[[15,20],[25,20],[25,25],[15,25]]]}
    asset=make_asset(scene,styles,[arrow,block],[{'bbox_2d':[20,62,45,76]}],source,tmp_path/'asset',decision,images=[{'bbox_2d':[76,15,86,30]}])
    patch=np.array(Image.open(asset['path']).convert('RGBA'));x0,y0=asset['placement_source_half_open'][:2]
    alpha=lambda x,y:patch[y-y0,x-x0,3]
    for x,y in [(10,40),(30,40),(75,40),(30,68),(65,65),(80,20),(20,22)]:assert alpha(x,y)==0
    assert alpha(30,25)==255
    original=np.array(image)[y0:y0+patch.shape[0],x0:x0+patch.shape[1]]
    assert np.array_equal(patch[:,:,:3][patch[:,:,3]>0],original[patch[:,:,3]>0])


def test_texture_cannot_replace_an_external_route_or_closed_boundary(tmp_path):
    source=tmp_path/'s.png';Image.new('RGB',(100,100),'white').save(source)
    surface=rect('o',[(10,10),(90,10),(90,90),(10,90)])
    external={'object_id':'o','kind':'path','commands':[['M',0,40],['L',99,40]]}
    scene={'parts':{'surface':surface,'external':external}}
    for key in ['surface','external']:
        with pytest.raises(ValueError):
            make_asset(scene,{},[],[],source,tmp_path/key,{'surface_part':'surface','replace_parts':[key],'reason':'test'})


def test_old_self_crossing_block_arrow_uses_a_conservative_protection_mask():
    points=[[10,10],[30,30],[10,30],[30,10]]
    region=arrow_region({'kind':'block_arrow','contours':[points]})
    assert region.is_valid
    from shapely.geometry import Point
    assert all(region.covers(Point(p)) for p in points)


def test_imported_arrow_without_text_body_is_classified_without_mutating_it(tmp_path):
    p=Presentation();slide=p.slides.add_slide(p.slide_layouts[6])
    arrow=slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,100,200,300,400);arrow.name='manually_repaired_arrow'
    text=arrow._element.find('{'+P+'}txBody');arrow._element.remove(text)
    source=tmp_path/'source.pptx';output=tmp_path/'out.pptx';p.save(source)
    before=etree.fromstring(read_package(source)[SLIDE]).find('.//{'+P+'}sp')
    receipt=inject([], [p.slide_width,p.slide_height], source, output)
    after=etree.fromstring(read_package(output)[SLIDE]).find('.//{'+P+'}sp')
    assert etree.tostring(before)==etree.tostring(after)
    assert receipt['base_objects_unchanged']


def load_tests(loader,tests,pattern):
    import inspect,tempfile,unittest
    for name,fn in sorted(globals().copy().items()):
        if not name.startswith('test_') or not callable(fn):continue
        def run(fn=fn):
            if inspect.signature(fn).parameters:
                with tempfile.TemporaryDirectory() as directory:fn(Path(directory))
            else:fn()
        tests.addTest(unittest.FunctionTestCase(run,description=name))
    return tests




def test_final_source_ownership_is_not_rejudged_by_legacy_fallback():
    from unittest.mock import patch
    scene={'parts':{}};arrows=[{'id':'reviewed'},{'id':'parallel'},{'id':'legacy'}]
    quality={'reviewed_arrow_names':['reviewed','parallel'],'drop_arrow_names':['reviewed']}
    with patch('full_recovery.handoff.duplicate_arrows',return_value=([],[])) as old, \
         patch('full_recovery.handoff.split_brace_duplicates',return_value=([],[])) as braces:
        plan=reconcile(scene,{'parts':{}},{'arrows':arrows},{'texts':[]},quality=quality)
    assert [r['id'] for r in old.call_args.args[2]]==['legacy']
    assert [r['id'] for r in braces.call_args.args[2]]==['legacy']
    assert plan['drop_arrow_names']==['reviewed']




def test_large_connected_component_is_not_split_at_six_pairs():
    pairs=[{'id':str(i),'a':'owner','b':'copy'+str(i)} for i in range(10)]
    assert comparison_groups(pairs)==[pairs]


def test_local_union_preserves_unique_edges_brace_nodes_and_independent_choices():
    style={'stroke':'#000000','fill':None,'width_px':1}
    scene={'parts':{}}
    for name,commands in [('upper',[['M',20,20],['L',100,20]]),
                          ('lower',[['M',20,60],['L',100,60]]),
                          ('unique',[['M',20,20],['L',100,60]]),
                          ('brace',[['M',110,10],['L',115,40],['L',110,70]])]:
        scene['parts'][name]={'id':name,'object_id':'net','kind':'path','status':'ok','commands':commands}
    for name,b in [('left',[10,10,30,30]),('right',[90,10,110,30])]:
        scene['parts'][name]={'object_id':'net','kind':'ellipse','status':'ok','bbox_2d':b}
    arrows=[{'id':name,'start':a,'end':b,'heads':['none','none'],'style':{'color':'#000000'}}
            for name,a,b in [('upper',[20,20],[100,20]),('lower',[20,60],[100,60]),('unique',[100,20],[140,80])]]
    rows=candidates(scene,{k:style for k in scene['parts'] if k not in ('left','right')},arrows)
    groups=comparison_groups(nominate(rows),rows)
    assert len(groups)==2
    decisions=[{'keep':branch,'candidate_ids':sorted({p[k] for p in g for k in ('a','b')})}
               for g,branch in zip(groups,['arrows','shape'])]
    plan=apply_choices(rows,decisions)
    assert plan['drop_shape_keys']==['upper'] and plan['drop_arrow_names']==['lower']
    assert plan['drop_shape_names']==['net/upper']


def test_complete_shaft_and_all_visible_gaps_can_choose_either_branch():
    from full_recovery.quality_choice import allowed_branches
    scene={'parts':{str(x):{'kind':'ellipse','bbox_2d':[x-10,20,x+10,40]} for x in (20,60,100)}}
    style={'stroke':'#000000','width_px':1}
    def row(name,layer,x0,x1):
        return {'id':name,'layer':layer,'points':np.array([[x,30] for x in np.linspace(x0,x1,100)]),
                'length':x1-x0,'network_nodes':['20','60','100'],'style':style}
    rows=[row('whole','arrows',20,100),row('gap1','shape',30,50),row('gap2','shape',70,90)]
    batch=[{'nomination':'shared_network_span'}]
    assert allowed_branches(batch,rows,scene,{k:{'fill':'#FFFFFF'} for k in scene['parts']})==['shape','arrows']
    assert allowed_branches(batch,rows[:2],scene,{k:{'fill':'#FFFFFF'} for k in scene['parts']})==['shape','arrows']


def test_complete_local_review_disables_legacy_brace_deletion():
    from unittest.mock import patch
    with patch('full_recovery.handoff.split_brace_duplicates',return_value=([],[])) as legacy:
        reconcile({'parts':{}},{'parts':{}},{'arrows':[{'id':'unique'}]},{'texts':[]},
                  quality={'line_ownership_complete':True})
    assert legacy.call_args.args[2]==[]


def test_crossing_two_large_nodes_does_not_make_a_diverging_long_route_a_duplicate():
    base={'length':100,'points':np.array([[20,50],[120,50]],float),'style':{'width_px':2},
          'network_nodes':['left','right'],'node_radius_px':30}
    shape={**base,'id':'s','layer':'shape','endpoint_nodes':('left','right')}
    long={**base,'id':'a','layer':'arrows','length':400,
          'points':np.array([[20,60],[420,110]],float)}
    assert not nominate([shape,long])


def test_source_absent_connection_can_be_removed_without_touching_other_shapes():
    rows=[{'id':'s','layer':'shape','key':'s','keys':['s'],'refs':['net/s']},
          {'id':'a','layer':'arrows','key':'a','keys':[],'refs':['a']},
          {'id':'unique','layer':'shape','key':'u','keys':['u'],'refs':['net/u']}]
    choice=validate_choice({'keep':'none','reason':'SOURCE has no corresponding edge'})
    plan=apply_choices(rows,[{**choice,'candidate_ids':['s','a']}])
    assert plan['drop_shape_keys']==['s'] and plan['drop_arrow_names']==['a']
    assert plan['drop_shape_names']==['net/s']


def test_unique_connection_presence_review_excludes_braces_and_faces():
    from full_recovery.quality_choice import connection_route
    row={'length':100,'points':np.array([[0,0],[0,100]],float),
         'network_nodes':['left','right'],'heads':['none','none']}
    assert connection_route(row)
    assert not connection_route({**row,'closed':True})
    assert not connection_route({**row,'segment':0})
    assert not connection_route({**row,'points':np.array([[0,0],[30,25],[0,50],[30,75],[0,100]],float)})


def test_unheaded_short_copy_does_not_use_length_to_veto_visual_choice():
    from full_recovery.quality_choice import allowed_branches
    a={'id':'s','layer':'shape','points':np.array([[40,22],[80,22]],float),'length':40,
       'heads':['none','none'],'style':{'stroke':'#000000','width_px':1}}
    b={**a,'id':'a','layer':'arrows','points':np.array([[10,20],[110,20]],float),'length':100}
    pairs=nominate([a,b]);assert len(pairs)==1
    assert allowed_branches(pairs,[a,b],{'parts':{}},{})==['shape','arrows']
    a['heads']=['none','triangle']
    assert not nominate([a,b])  # A head in the middle can mean a separate connection.
