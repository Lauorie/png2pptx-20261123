"""Whole native routes must not hide shorter source connections from review."""
import copy
import numpy as np

from full_recovery.handoff import resample
from full_recovery.quality_choice import nominate,apply_choices,comparison_groups,shared_native_span


def routes():
    t=np.linspace(-np.pi/2,np.pi/2,80)
    short=np.c_[30-20*np.cos(t),60+20*np.sin(t)]
    long=np.vstack([[1000,40],np.c_[30-28*np.cos(t),60+20*np.sin(t)],[70,80]])
    rows=[]
    for ident,layer,points,ends,visited in [
        ('arc','shape',short,('upper','lower'),['upper','lower']),
        ('chain','arrows',long,('far','next'),['far','upper','lower','next'])]:
        points=resample([points])
        rows.append({'id':ident,'layer':layer,'owner':ident,'key':ident,'keys':[ident],
            'refs':[ident],'points':points,'heads':['none','none'],
            'endpoint_nodes':ends,'network_nodes':visited,
            'length':float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),
            'style':{'stroke':'#333333','color':'#333333','width_px':2}})
    return rows


def test_short_offset_return_curve_is_reviewed_against_a_whole_native_chain():
    rows=routes();assert rows[0]['length']/rows[1]['length']<.2
    pairs=nominate(rows)
    assert len(pairs)==1 and pairs[0]['nomination']=='embedded_endpoint_span'
    assert pairs[0]['source_endpoint_nodes']==['upper','lower']


def test_one_shared_node_does_not_establish_an_embedded_connection():
    rows=routes();rows[1]['network_nodes']=['far','upper','next']
    assert nominate(rows)==[]


def test_crossing_dividers_inside_the_same_faces_are_not_embedded_route_versions():
    rows=routes()
    for row,points in zip(rows,([[10,0],[10,100]],[[0,50],[100,50]])):
        row.update(layer='shape',points=resample([np.asarray(points,float)]),length=100.,
                   endpoint_nodes=('back_face','front_face'),network_nodes=['back_face','front_face'])
    assert nominate(rows)==[]


def test_choosing_the_whole_route_preserves_its_external_span_and_unmatched_routes():
    rows=routes();before=copy.deepcopy(rows)
    independent={**rows[1],'id':'independent','key':'independent','keys':['independent'],
                 'refs':['independent']}
    result=apply_choices([*rows,independent],[{'candidate_ids':['arc','chain'],'keep_ids':['chain']}])
    assert result['drop_shape_keys']==['arc'] and result['drop_arrow_names']==[]
    assert result['shape_segment_suppression']=={} and result['shape_subpath_suppression']=={}
    for old,new in zip(before,rows):np.testing.assert_array_equal(old['points'],new['points'])
    assert rows[1]['points'][:,0].max()==1000


def native_route(ident, layer, vertices):
    points=resample([np.asarray(vertices,float)])
    return {'id':ident,'key':ident,'keys':[ident],'refs':[ident],
            'owner':ident,'layer':layer,'points':points,'heads':['none','none'],
            'length':float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),
            'style':{'stroke':'#000000','color':'#000000','width_px':2}}


def test_native_chain_and_separate_returns_form_one_complete_competitor_group():
    rows=[native_route('upper','shape',[[60,100],[100,100],[100,20],[85,20]]),
          native_route('lower','shape',[[60,200],[100,200],[100,130],[85,130]]),
          native_route('chain','arrows',[[85,20],[101,20],[101,201],[60,201]]),
          native_route('middle_head','arrows',[[101,130],[85,130]]),
          native_route('external','arrows',[[200,0],[200,210]])]
    before=copy.deepcopy(rows)
    groups=comparison_groups(nominate(rows),rows)
    group=next(g for g in groups if any('chain' in (p['a'],p['b']) for p in g))
    members={p[k] for p in group for k in ('a','b')}
    assert members=={'upper','lower','chain','middle_head'}
    result=apply_choices(rows,[{'candidate_ids':sorted(members),'keep_ids':['upper','lower']}])
    assert result['drop_arrow_names']==['chain','middle_head']
    assert result['drop_shape_keys']==[]
    for old,new in zip(before,rows):np.testing.assert_array_equal(old['points'],new['points'])


def test_crossing_or_touching_a_native_bend_is_not_a_shared_corridor():
    bend=native_route('bend','shape',[[0,0],[50,0],[50,50]])
    crossing=native_route('cross','arrows',[[25,-30],[25,30]])
    touching=native_route('touch','arrows',[[50,0],[50,-50]])
    for other in [crossing,touching]:
        assert shared_native_span(bend['points'],other['points'],4)<=8
        assert nominate([bend,other])==[]
