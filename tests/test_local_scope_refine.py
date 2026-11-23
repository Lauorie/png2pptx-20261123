"""Regressions for conflicting style commits and collapsed parallel routes."""
import copy
import json
import pytest
import numpy as np
from PIL import Image, ImageDraw
from shape_integration.common import save, sha, digest
from shape_integration.styling.controller import commit_refinement, unchanged_styles, style_policy
from shape_integration.sam_fixed import FixedSamEvidence
from full_recovery.quality_choice import nominate, protect_parallel_pairs, candidates, apply_choices


def test_width_dispute_does_not_restore_an_incorrect_dash_pattern():
    old={'p':{'fill':None,'stroke':'#777777','width_px':3.,'dash':'dash_dot'}}
    new={'p':{**old['p'],'width_px':2.,'dash':'solid'}}
    checked={'p':{'value':{'verdict':'revise','affected_fields':['stroke_width']}}}
    committed,rejected=commit_refinement(old,new,checked)
    assert committed['p']['dash']=='solid'
    assert committed['p']['width_px']==3.
    assert rejected==['p']
    assert old['p']['dash']=='dash_dot'


def test_dot_dash_disagreement_cannot_restore_source_contradicted_solid():
    old={'p':{'stroke':'#777777','width_px':1.,'dash':'solid'}}
    new={'p':{**old['p'],'dash':'dot','dash_pattern_px':[1.,3.]}}
    check={'p':{'value':{'verdict':'revise','affected_fields':['line_pattern'],
                         'expected_line_pattern':'dash'}}}
    assert commit_refinement(old,new,check)==(new,[])
    assert check['p']['value']['verdict']=='revise'
    check['p']['value']['expected_line_pattern']='solid'
    assert commit_refinement(old,new,check)==(old,['p'])
    check['p']['value'].pop('expected_line_pattern')
    assert commit_refinement(old,new,check)==(old,['p'])


def test_final_source_and_measured_pattern_consensus_keeps_review_pending():
    from shape_integration.styling.controller import resolve_pattern_consensus
    style={'p':{'dash':'dot','width_px':1.2,'stroke':'#777777'}}
    check={'p':{'value':{'verdict':'revise','affected_fields':['line_pattern'],
                         'expected_line_pattern':'dash'}}}
    measured={'p':{'line':{'dash':'dash','dash_reliable':True,'dash_pattern_px':[2.5,2.]}}}
    change=resolve_pattern_consensus(style,check,measured)
    assert style['p']=={'dash':'dash','dash_pattern_px':[2.5,2.],'width_px':1.2,'stroke':'#777777'}
    assert check['p']['value']['verdict']=='revise' and list(change)==['p']
    for invalid in ({'dash_reliable':False},{'dash':'dot'},{'dash_pattern_px':[float('nan'),2.]}):
        current={'p':{'dash':'dot'}}
        assert not resolve_pattern_consensus(current,check,{'p':{'line':{**measured['p']['line'],**invalid}}})
        assert current=={'p':{'dash':'dot'}}


def test_unreviewed_style_cannot_become_a_valid_seed(tmp_path):
    source=tmp_path/'source.png';Image.new('RGB',(30,30),'white').save(source)
    p={'object_id':'o','id':'p','kind':'rectangle','commands':[['M',1,1],['L',9,1],['L',9,9],['Z']]}
    scene={'case_id':'case','source_sha256':sha(source),'parts':{'p':p}}
    root=tmp_path/'case';save(root/'fitting/scene.json',scene)
    doc={'input':{'style_policy':style_policy(),'assets_sha256':digest([])},
         'parts':{'p':{'state':'critic_revise','style':{'fill':'#FFFFFF'}}}}
    save(root/'styling/styles.json',doc)
    assert unchanged_styles(scene,scene['parts'],tmp_path)[0]=={}
    doc['parts']['p']['state']='critic_accept';save(root/'styling/styles.json',doc)
    assert unchanged_styles(scene,scene['parts'],tmp_path)[0]
    doc['input']['style_policy']={};save(root/'styling/styles.json',doc)
    assert unchanged_styles(scene,scene['parts'],tmp_path)[0]=={}


def test_missing_node_not_covered_by_large_wiring_bbox(tmp_path):
    save(tmp_path/'sam_evidence.json',{'shown':[{'id':'S1','bbox_original':[45,45,55,55]}]})
    provider=object.__new__(FixedSamEvidence)
    validator=provider.review_validator(lambda d:d,tmp_path,
        [{'id':'owner','fitted_part_boxes':[[0,0,100,100],[45,20,55,30]]}])
    row={'new_objects':[],'uncertain':[],'status':'complete','sam_decisions':[
        {'sam_id':'S1','action':'covered','target':'owner','reason':'inside wiring'}]}
    with pytest.raises(ValueError,match='covered contradicts the represented parts'):
        validator(row)
    row['sam_decisions'][0].update(action='uncertain',target=None,reason='CLEAN cannot establish ownership')
    result=validator(row)
    assert result['sam_decisions'][0]['action']=='uncertain'
    assert result['status']=='uncertain'


def test_partial_route_near_previous_length_cutoff_is_compared():
    def route(ident,layer,end):
        points=np.c_[np.linspace(10,end,80),np.full(80,30.)]
        return {'id':ident,'key':ident,'layer':layer,'points':points,'length':end-10,
                'heads':['none','none'],'style':{'stroke':'#000000','width_px':1}}
    rows=[route('a','shape',172.4),route('b','arrows',210.)]
    assert len(nominate(rows))==1


def test_node_radius_does_not_merge_parallel_bus_lines():
    image=Image.new('RGB',(150,100),'white');draw=ImageDraw.Draw(image)
    rows=[]
    for i,y in enumerate((30,40,50)):
        draw.line((20,y,120,y),fill='black',width=1)
        rows.append({'id':str(i),'layer':'shape','points':np.c_[np.linspace(20,120,80),np.full(80,y)],
                     'length':100.,'heads':['none','none'],'style':{'stroke':'#000000','width_px':1},
                     'endpoint_nodes':('left','right'),'node_radius_px':35})
    nominated=nominate(rows)
    eligible,_=protect_parallel_pairs(rows,nominated,np.asarray(image))
    assert not eligible


def test_distant_divider_cannot_compete_through_guessed_endpoint_nodes():
    rows=[]
    for ident,layer,x,y0,y1 in [('divider','shape',80,60,100),('connector','arrows',30,0,55)]:
        rows.append({'id':ident,'layer':layer,'points':np.c_[np.full(80,x),np.linspace(y0,y1,80)],
            'length':y1-y0,'heads':['none','none'],'style':{'stroke':'#000000','width_px':1},
            'endpoint_nodes':('large_top_box','large_bottom_box')})
    assert not nominate(rows)


def test_curved_competitors_do_not_require_matching_node_assignment():
    t=np.linspace(0,1.6*np.pi,96)
    p=np.c_[60+25*np.cos(t),60+25*np.sin(t)]
    def row(ident,layer,points,nodes):
        return {'id':ident,'layer':layer,'points':points,'length':float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),
                'heads':['none','triangle'],'style':{'stroke':'#777777','width_px':2},
                'endpoint_nodes':nodes,'node_radius_px':20}
    rows=[row('s','shape',p,('a','b')),row('a','arrows',p+[3,2],('c','d'))]
    found=nominate(rows)
    assert len(found)==1 and found[0]['nomination']=='overlapping_curved_routes'


def test_curved_shaft_deletion_removes_its_attached_stroked_head_only():
    curve=[['M',10,60],['C',-10,10,80,0,80,50]]
    def rows(head):
        part={'object_id':'o','id':'loop','kind':'path','status':'ok','bbox_2d':[0,0,80,60],
              'commands':curve+head}
        return candidates({'parts':{'loop':part}},
                          {'loop':{'fill':None,'stroke':'#777777','width_px':2}},[])
    attached=rows([['M',73,43],['L',80,50]])
    assert len(attached)==1 and attached[0]['attached_stroked_head']
    result=apply_choices(attached,[{'keep':'arrows','candidate_ids':['shape:loop'],'reason':'source'}])
    assert result['drop_shape_keys']==['loop']
    separate=rows([['M',100,80],['L',110,80]])
    assert len(separate)==2 and all('subpath' in r for r in separate)
    result=apply_choices(separate,[{'keep':'arrows','candidate_ids':['shape:loop::subpath0'],'reason':'source'}])
    assert result['drop_shape_keys']==[]
    assert result['shape_subpath_suppression']['o/loop']==[0]
