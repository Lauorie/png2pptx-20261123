from copy import deepcopy
import numpy as np
from PIL import Image, ImageDraw
import pytest
from shape_integration.fitting.contracts import normalize_observation
from full_recovery.source_route_nomination import nominate


def observation(role,unit='neural_network'):
    return {'status':'ok','points':[{'id':k,'xy':v}for k,v in
        [('a',[10,10]),('b',[30,10]),('c',[30,30]),('d',[10,30]),('e',[65,20])]],
        'parts':[{'id':'node','kind':'rectangle','point_ids':['a','b','c','d']},
            {'id':'line','kind':'path','closed':False,'segments':[{'kind':'line','point_ids':['b','e']}]}],
        'ownership':{'unit':unit,'unit_evidence':'Actual source units and source directionality',
            'parts':[{'id':'node','role':'body'},{'id':'line','role':role}]}}


@pytest.mark.parametrize('unit',['neural_network','other'])
def test_source_headed_connection_is_delegated_with_its_shaft(unit):
    raw=observation('directional_connection',unit);original=deepcopy(raw);audit={}
    out=normalize_observation(raw,'symbol',[100,100],ownership_required=True,audit=audit)
    assert [p['id']for p in out['parts']]==['node'] and out['parts'][0]==raw['parts'][0]
    assert raw==original and {p['id']for p in out['points']}=={'a','b','c','d'}
    assert audit['declared_ownership']['delegated_parts'][0]['owner']=='arrow'


def test_headless_compound_routes_are_shape_owned_for_any_source_unit():
    raw=observation('connection');out=normalize_observation(raw,'symbol',[100,100],ownership_required=True)
    assert out['parts']==raw['parts']
    assert out['points']==raw['points']
    raw['ownership']['unit']='other'
    assert normalize_observation(raw,'symbol',[100,100],ownership_required=True)['parts']==raw['parts']


def test_pointed_frame_contour_keeps_its_closed_body_identity():
    raw=observation('contour','other');raw['parts'][1].update(closed=True,
        segments=[{'kind':'line','point_ids':['b','e']},{'kind':'line','point_ids':['e','c']},
                  {'kind':'line','point_ids':['c','b']}])
    out=normalize_observation(raw,'symbol',[100,100],ownership_required=True)
    assert out['parts']==raw['parts']


def test_declared_head_endpoint_delegates_the_connection_despite_old_role():
    raw=observation('connection');raw['points']+=[{'id':'f','xy':[60,17]},{'id':'g','xy':[60,23]}]
    raw['parts']+=[{'id':'head','kind':'path','closed':True,'segments':[
        {'kind':'line','point_ids':['e','f']},{'kind':'line','point_ids':['f','g']},
        {'kind':'line','point_ids':['g','e']}]}]
    raw['ownership']['parts']+=[{'id':'head','role':'directional_head'}]
    assert [p['id']for p in normalize_observation(raw,'symbol',[100,100],ownership_required=True)['parts']]==['node']
    raw['parts'][1]['segments']+=[{'kind':'line','point_ids':['a','c']}]
    from shape_integration.fitting.contracts import ObservationError
    with pytest.raises(ObservationError,match='split headed and headless'):
        normalize_observation(raw,'symbol',[100,100],ownership_required=True)


def row(ident,layer,y,heads=False):
    return {'id':ident,'key':ident,'keys':[ident],'refs':[ident],'layer':layer,
        'points':np.linspace([30.,y],[100.,y],80),'length':70.,
        'heads':['none','triangle'if heads else'none'],
        'style':{'stroke':'#333333','color':'#333333','width_px':1}}


def test_unpaired_shape_can_join_an_already_reviewed_arrow_by_source_ridge():
    image=Image.new('RGB',(150,130),'white');ImageDraw.Draw(image).line((30,45,100,45),fill='#333333',width=2)
    rows=[row('missed','shape',54),row('old','shape',45),row('full','arrows',45,True)]
    existing=[{'a':'old','b':'full'}];out=nominate(rows,np.asarray(image),existing)
    assert len(out)==1 and {out[0]['a'],out[0]['b']}=={'missed','full'}
    assert out[0]['scope']=='unreviewed_comparable_shape_route'
    assert out[0]['source_relation']=='nomination_only_complete_native_review_required'


def test_real_parallel_headless_and_ambiguous_sources_never_gain_new_relation():
    image=Image.new('RGB',(150,130),'white');draw=ImageDraw.Draw(image)
    draw.line((30,45,100,45),fill='#333333',width=2)
    draw.line((30,60,100,60),fill='#333333',width=2)
    assert nominate([row('own','shape',60),row('other','arrows',45,True)],np.asarray(image))==[]
    assert nominate([row('between','shape',52.5),row('other','arrows',45,True)],np.asarray(image))==[]
    assert nominate([row('headless','shape',54),row('other','arrows',45)],np.asarray(image))==[]


def test_two_source_recoveries_join_same_complete_group_without_external_routes():
    from full_recovery.quality_choice import comparison_groups,apply_choices
    image=Image.new('RGB',(150,130),'white');ImageDraw.Draw(image).line((30,45,100,45),fill='#333333',width=2)
    rows=[row('missed','shape',54),row('old','shape',45),row('full','arrows',45,True),
          row('unrelated','arrows',105,True)]
    pairs=[{'id':'original','a':'old','b':'full'}]
    additional=nominate(rows,np.asarray(image),pairs)
    for i,pair in enumerate(additional):pair['id']=f'extra{i}'
    groups=comparison_groups([*pairs,*additional],rows)
    assert len(groups)==1
    members=sorted({p[k]for p in groups[0]for k in('a','b')})
    assert set(members)=={'old','missed','full'}
    plan=apply_choices(rows,[{'candidate_ids':members,'keep_ids':['full']}])
    assert set(plan['drop_shape_names'])=={'old','missed'}
    assert plan['drop_arrow_names']==[]


def test_directional_only_edit_preserves_the_protected_closed_node(tmp_path):
    from shape_integration.fitting.local import prepare_candidate,observe_and_fit
    from shape_integration.fitting.adapter import fit_object
    from shape_integration.fitting.frames import FitFrame
    from shape_integration.common import save,read
    frame=FitFrame((100,100),(100,100),(100,100),(0,0,100,100))
    raw=observation('connection');audit={}
    baseline=normalize_observation(raw,'symbol',[100,100],ownership_required=True,audit=audit)
    fitted=fit_object(baseline,frame,object_id='symbol',namespace='proof')
    fitted['declared_ownership']=audit['declared_ownership'];save(tmp_path/'base/fit_local.json',fitted)
    context={'source_revision':str(tmp_path/'base'),'base_observation':baseline,
        'base_ownership':audit['declared_ownership'],'frame':frame.metadata(),
        'editable_ids':['line'],'remove_ids':[],'preserved_ids':['node'],
        'allow_new_parts':False,'issues':[],'previous_display_part_ids':{}}
    source=Image.new('RGB',(100,100),'white');draw=ImageDraw.Draw(source)
    draw.rectangle((10,10,30,30),outline='black');draw.line((30,10,65,20),fill='black')
    candidate={'id':'symbol','kind':'compound_symbol','label':'node with directed connector','bbox_2d':[5,5,75,45]}
    folder=tmp_path/'edit';job=prepare_candidate(source,candidate,[100,100],folder,
        crop_bounds_original=[0,0,100,100],long_edge=100,revision_context=context)
    response=observation('directional_connection');response['parts']=response['parts'][1:]
    response['points']=[p for p in response['points']if p['id']in('b','e')]
    response['ownership']['parts']=response['ownership']['parts'][1:]
    class Model:
        config={'fitting':{'source_frame_refinement':False}}
        def call(self,root,tag,system,prompt,images,validator,budget):
            return validator(response)
    result=observe_and_fit(job,folder,Model(),namespace='proof',max_tokens=1024)
    assert result['status']=='finished'
    revised=read(folder/'fit_local.json')
    assert [p['id']for p in revised['parts']]==['node']
    assert revised['parts'][0]['commands']==fitted['parts'][0]['commands']
    assert result['revision_merge']['removed_ids']==['line']


def test_headless_partial_pair_needs_unique_endpoints_and_common_source_ridge():
    image=Image.new('RGB',(180,120),'white');ImageDraw.Draw(image).line((30,45,135,45),fill='#333333',width=2)
    a=row('short','arrows',51);a['points']=np.linspace([40.,51],[110.,51],80);a['length']=70.
    b=row('long','shape',45);b['points']=np.linspace([30.,45],[135.,45],80);b['length']=105.
    for r in(a,b):r['endpoint_nodes']=('u','v')
    out=nominate([a,b],np.asarray(image))
    assert len(out)==1 and out[0]['scope']=='unique_endpoint_headless_source_route'
    a.pop('endpoint_nodes')
    assert nominate([a,b],np.asarray(image))==[]
    a['endpoint_nodes']=('u','v');c=deepcopy(a);c['id']='parallel';c['points'][:,1]=65
    assert nominate([a,b,c],np.asarray(image))==[]
    curved=deepcopy(c);curved['points'][:,1]+=15*np.sin(np.linspace(0,np.pi,80))
    assert nominate([a,b,curved],np.asarray(image))==[]


@pytest.mark.parametrize('scale,angle',[(.75,0),(1.5,90),(2.,35)])
def test_headless_same_source_hint_follows_scale_and_rotation(scale,angle):
    source=Image.new('RGB',(500,500),'white');theta=np.radians(angle)
    m=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])*scale
    origin=np.array([150.,150.]);draw=ImageDraw.Draw(source)
    line=np.array([[30.,45.],[135.,45.]])@m.T+origin
    draw.line([tuple(p)for p in line],fill='#333333',width=max(1,round(2*scale)))
    a=row('short','arrows',51);a['points']=np.linspace([40.,51],[110.,51],80);a['length']=70.
    b=row('long','shape',45);b['points']=np.linspace([30.,45],[135.,45],80);b['length']=105.
    for r in(a,b):
        r['points']=r['points']@m.T+origin;r['length']*=scale
        r['style']['width_px']*=scale;r['endpoint_nodes']=('u','v')
    assert len(nominate([a,b],np.asarray(source)))==1
