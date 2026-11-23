from copy import deepcopy
import json

from PIL import Image
import pytest

from full_recovery.image_assets import connected_owner, review
from shape_integration.common import sha
from shape_integration.fitting.ownership import compile


def test_source_solid_edges_do_not_become_node_link_network():
    def part(i,kind,b,commands):
        return {'id':i,'object_id':'solid','kind':kind,'status':'ok','bbox_2d':b,'commands':commands}
    parts={'left':part('left','ellipse',[0,0,10,10],[['M',0,0],['L',10,10]]),
           'right':part('right','ellipse',[90,0,100,10],[['M',90,0],['L',100,10]]),
           'edge':part('edge','path',[5,5,95,5],[['M',5,5],['L',95,5]])}
    receipt={'unit':'other','retained_roles':{'left':'body','right':'body','edge':'intrinsic_stroke'},'delegated_parts':[]}
    scene={'parts':parts,'objects':{'solid':{'declared_ownership':receipt}}}
    assert not connected_owner(scene,'solid',list(parts))
    receipt['retained_roles']['edge']='connection'
    assert connected_owner(scene,'solid',list(parts))
    scene['objects']['solid'].pop('declared_ownership')
    assert connected_owner(scene,'solid',list(parts))


def test_delegated_only_target_does_not_schedule_image(tmp_path):
    source=tmp_path/'source.png';Image.new('RGB',(200,200),'white').save(source)
    scene={'source_size':[200,200],'source_sha256':sha(source),'parts':{},'objects':{
        'owner':{'candidate':{'kind':'compound_symbol','bbox_2d':[20,20,40,40]},'part_keys':[],
                 'state':'unresolved','declared_ownership':{'retained_roles':{},
                    'delegated_parts':[{'id':'route','owner':'arrow'}]}}}}
    class Forbidden:
        def call(self,*args):raise AssertionError('No raster recovery for delegated content')
    result=review(scene,{},[],[],source,source,tmp_path/'review',Forbidden())
    assert not result['assets'] and result['model_calls']==0
    assert result['delegated_only_owners_not_reintroduced']==['owner']


def test_unfitted_geometry_can_remain_unresolved_without_forced_raster(tmp_path):
    source=tmp_path/'source.png';Image.new('RGB',(200,200),'white').save(source)
    scene={'source_size':[200,200],'source_sha256':sha(source),'parts':{},'objects':{
        'owner':{'candidate':{'kind':'rectangle','bbox_2d':[20,20,40,40]},'part_keys':[],
                 'state':'failed'}}}
    class Runtime:
        def call(self,root,tag,system,prompt,images,validator,budget):
            return validator({'action':'keep_native','reason':'ordinary frame; raster is not appropriate'})
    result=review(scene,{},[],[],source,source,tmp_path/'review',Runtime())
    assert not result['assets'] and result['model_calls']==1
    assert result['reviews'][0]['action']=='unresolved'


def test_complex_solid_can_use_bounded_image_with_source_roles(tmp_path):
    source=tmp_path/'source.png';Image.new('RGB',(200,200),'blue').save(source)
    key=json.dumps(['revision','owner','face'])
    scene={'source_size':[200,200],'source_sha256':sha(source),
           'parts':{key:{'id':'face','object_id':'owner','kind':'path','status':'ok',
                         'bbox_2d':[20,20,40,40],'commands':[['M',20,20],['L',40,20],['L',40,40],['L',20,40],['Z']]}},
           'objects':{'owner':{'candidate':{'kind':'compound_symbol','bbox_2d':[20,20,40,40]},
               'part_keys':[key],'declared_ownership':{'unit':'other','retained_roles':{'face':'body'},'delegated_parts':[]}}}}
    class Runtime:
        def call(self,root,tag,system,prompt,images,validator,budget):
            assert not prompt['connected_native_unit'] and prompt['parts'][0]['source_role']=='body'
            return validator({'action':'image','reason':'interdependent perspective and shading',
                'role':'icon','method':'crop','replace_parts':['P0'],'replace_arrows':[]})
    result=review(scene,{},[],[],source,source,tmp_path/'review',Runtime())
    assert result['drop_shape_keys']==[key] and len(result['assets'])==1


def test_delegation_without_selected_geometry_survives_scene_assembly(tmp_path):
    from shape_integration.fitting.scene import assemble
    receipt={'retained_roles':{},'delegated_parts':[{'id':'arrow','owner':'arrow'}]}
    record={'state':'unsupported','selected_revision':None,'history':[
        {'observation':{'declared_ownership':receipt}}]}
    item={'id':'sample','source_size':[100,100],'source_sha256':'source'}
    scene=assemble(tmp_path,item,[{'id':'owner'}],{'owner':record},{},{})
    assert not scene['parts'] and scene['objects']['owner']['declared_ownership']==receipt
    scene['objects']['owner']['declared_ownership']['delegated_parts'].clear()
    assert receipt['delegated_parts']  # immutable observation receipt


def test_geometry_only_patch_preserves_prior_delegation():
    from shape_integration.fitting.ownership import retained_receipt
    prior={'retained_roles':{'grid':'contour'},'delegated_parts':[{'id':'arrow','role':'arrow','owner':'arrow'}]}
    current={'retained_roles':{'rule':'divider'},'delegated_parts':[]}
    result=retained_receipt(current,prior,['grid','rule'])
    assert result['retained_roles']=={'grid':'contour','rule':'divider'}
    assert result['delegated_parts']==prior['delegated_parts']
    assert not current['delegated_parts']


def test_wholly_delegated_observation_survives_history_and_empty_scene(tmp_path):
    from shape_integration.fitting.local import prepare_candidate,observe_and_fit
    from shape_integration.fitting.scene import assemble
    source=Image.new('RGB',(100,100),'white')
    candidate={'id':'owner','label':'connection','kind':'compound_symbol','bbox_2d':[20,20,80,80]}
    folder=tmp_path/'object'
    job=prepare_candidate(source,candidate,[100,100],folder,padding=.1,long_edge=100)
    raw={'status':'ok','reason':'source connection',
         'points':[{'id':'a','xy':[20,20]},{'id':'b','xy':[80,80]}],
         'parts':[{'id':'route','kind':'path','closed':False,
                   'segments':[{'kind':'line','point_ids':['a','b']}]}],
         'ownership':{'unit':'other','unit_evidence':'connector between independent objects',
                      'parts':[{'id':'route','role':'connection'}]}}
    class Runtime:
        config={}
        def call(self,root,tag,system,prompt,images,validator,budget):
            return validator(deepcopy(raw))
    record=observe_and_fit(job,folder,Runtime(),namespace='receipt_test',max_tokens=1024)
    assert record['status']=='finished'
    receipt=record['declared_ownership']
    assert not receipt['retained_roles'] and receipt['delegated_parts'][0]['owner']=='arrow'
    scene=assemble(tmp_path,{'id':'sample','source_size':[100,100],'source_sha256':'source'},
                   [candidate],{'owner':{'state':'unsupported','selected_revision':None,
                                        'history':[{'observation':record}]}},{},{})
    assert scene['objects']['owner']['declared_ownership']==receipt and not scene['parts']
