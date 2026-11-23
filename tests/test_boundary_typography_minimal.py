from copy import deepcopy
import pytest
from PIL import Image,ImageDraw
from full_recovery.image_assets import compact_unit,connected_owner
from full_recovery.text_diagnostic_repair import formula_structure_conflicts
from full_recovery.source_text_attributes import source_attributes
from full_recovery.text_reading_views import images
from shape_integration.fitting.context import identity_context


def test_identity_does_not_reuse_crop_dependent_reason():
    obj={'id':'owner','label':'outer frame','bbox_2d':[0,0,100,200],'reason':'old crop misses bottom'}
    history=[{'id':'owner','action':'needs_context','reason':'clipped bottom'}]
    result=identity_context(obj,[obj],[],history,.2)
    assert result['identity_descriptions']==['Localization selected outer frame']


@pytest.mark.parametrize('unit', ['neural_network', 'other'])
def test_small_area_cannot_override_delegated_node_group(unit):
    def node(i,b):
        x,y,r,t=b
        return {'id':i,'object_id':'owner','kind':'ellipse','status':'ok','bbox_2d':b,
                'commands':[['M',x,y],['L',r,y],['L',r,t],['L',x,t],['Z']]}
    scene={'source_size':[1000,1000],'objects':{'owner':{
        'candidate':{'kind':'compound_symbol','bbox_2d':[10,10,110,30]},
        'declared_ownership':{'unit':unit,'delegated_parts':[{'id':'link','role':'connection','owner':'arrow'}]}}},
        'parts':{'a':node('a',[10,10,30,30]),'b':node('b',[90,10,110,30])}}
    assert compact_unit(scene,'owner')
    assert connected_owner(scene,'owner',['a','b'])
    assert connected_owner(scene,None,['a','b'])  # registry proposal spanning native owners
    scene['objects']['owner']['declared_ownership']={}
    assert not connected_owner(scene,'owner',['a','b'])
    scene['objects']['owner']['declared_ownership']={'delegated_parts':[{'id':'label','owner':'text'}]}
    assert not connected_owner(scene,'owner',['a','b'])


def test_external_connector_does_not_turn_image_parts_into_node_network():
    scene={'objects':{'owner':{'declared_ownership':{'unit':'other',
           'retained_roles':{'a':'body','b':'body'},
           'delegated_parts':[{'id':'outgoing','role':'directional_connection','owner':'arrow'}]}}},'parts':{}}
    for name,b in [('a',[10,10,30,30]),('b',[90,10,110,30])]:
        scene['parts'][name]={'id':name,'object_id':'owner','kind':'rectangle',
                              'status':'ok','bbox_2d':b}
    external=[{'start':[20,30],'end':[20,120]}]
    internal=[{'start':[30,20],'end':[90,20]}]
    assert not connected_owner(scene,'owner',['a','b'],external)
    assert connected_owner(scene,'owner',['a','b'],internal)


def test_text_formula_switch_with_script_disagreement_is_reread_not_rewritten():
    row={'id':'unit','kind':'text','text':'q7','runs':[{'text':'q','script':'normal'},{'text':'7','script':'sub'}]}
    same={'unit':[{**row,'kind':'formula','latex':'q_{7}'}]}
    wrong={'unit':[{**row,'kind':'formula','latex':'q^{7}'}]}
    before=deepcopy((row,same,wrong))
    assert not formula_structure_conflicts([row],same)
    assert [r['id']for r in formula_structure_conflicts([row],wrong)]==['unit']
    assert (row,same,wrong)==before


def test_literal_text_to_formula_without_scripts_does_not_add_reread():
    row={'id':'unit','kind':'text','text':'AB','runs':[{'text':'AB','script':'normal'}]}
    assert not formula_structure_conflicts([row],{'unit':[{**row,'kind':'formula','latex':r'\mathrm{AB}'}]})
    assert formula_structure_conflicts([row],{'unit':[{**row,'kind':'formula','latex':r'A\overline{B}'}]})


def test_same_call_details_do_not_apply_predicted_rotation(tmp_path):
    im=Image.new('RGB',(200,140),'white');d=ImageDraw.Draw(im);d.rectangle([22,12,35,98],fill='blue')
    path=tmp_path/'source.png';im.save(path)
    units=[{'id':'a','kind':'formula','rotation_deg':90,'locked_bbox_2d':[20,10,40,100]},
           {'id':'b','kind':'text','rotation_deg':0,'rough_bbox_2d':[50,20,100,50]}]
    result=images(units,path,tmp_path/'views')
    import json
    record=json.loads((tmp_path/'views/source_reading_details.json').read_text())
    assert len(result)==1 and len(record['views'])==1
    assert record['views'][0]['id']=='a'
    assert all(v['display_rotation_deg']==0 for v in record['views'])
    assert record['extra_model_calls']==0


def test_full_source_needs_context_uses_existing_protocol_correction(tmp_path):
    from shape_integration.fitting.local import prepare_candidate,observe_and_fit
    im=Image.new('RGB',(100,100),'white')
    obj={'id':'owner','label':'frame','kind':'rectangle','bbox_2d':[10,10,90,90]}
    folder=tmp_path/'object'
    job=prepare_candidate(im,obj,[100,100],folder,padding=1,long_edge=100)
    rejected={'status':'needs_context','reason':'crop misses bottom','points':[],'parts':[],
        'ownership':{'unit':'other','unit_evidence':'frame','parts':[]}}
    good={'status':'ok','reason':'full frame visible','points':[{'id':i,'xy':xy}
        for i,xy in zip('abcd',[[10,10],[90,10],[90,90],[10,90]])],
        'parts':[{'id':'frame','kind':'rectangle','point_ids':list('abcd')}],
        'ownership':{'unit':'other','unit_evidence':'frame','parts':[{'id':'frame','role':'contour'}]}}
    class Runtime:
        config={'shape_schema_retries':1}
        calls=0
        def call(self,root,tag,system,prompt,images,validator,budget):
            self.calls+=1
            with pytest.raises(ValueError,match='complete original image'):validator(rejected)
            return validator(good)
    runtime=Runtime();result=observe_and_fit(job,folder,runtime,namespace='test',max_tokens=1000)
    assert runtime.calls==1 and result['part_counts']['succeeded']==1


def test_partial_source_still_can_request_context(tmp_path):
    from shape_integration.fitting.local import prepare_candidate,observe_and_fit
    im=Image.new('RGB',(300,300),'white');folder=tmp_path/'object'
    obj={'id':'owner','label':'frame','kind':'rectangle','bbox_2d':[110,110,130,130]}
    job=prepare_candidate(im,obj,[300,300],folder,padding=.1,long_edge=100)
    class Runtime:
        config={'shape_schema_retries':0}
        def call(self,root,tag,system,prompt,images,validator,budget):
            return validator({'status':'needs_context','reason':'continuing frame cut by crop','points':[],'parts':[],
                'ownership':{'unit':'other','unit_evidence':'continuing frame','parts':[]}})
    result=observe_and_fit(job,folder,Runtime(),namespace='test',max_tokens=1000)
    assert result['observation_status']=='needs_context'


def test_registry_replacement_does_not_review_same_inner_part_again(tmp_path):
    from full_recovery.image_assets import review
    from shape_integration.common import sha
    im=Image.new('RGB',(200,200),'white');d=ImageDraw.Draw(im)
    d.rectangle([60,60,90,90],fill='blue');source=tmp_path/'source.png';im.save(source)
    frame={'id':'frame','object_id':'owner','kind':'rectangle','status':'ok','bbox_2d':[20,20,180,180],
        'commands':[['M',20,20],['L',180,20],['L',180,180],['L',20,180],['Z']],
        'native_parameters_original':{'center':[100,100],'width':160,'height':160}}
    mark={'id':'mark','object_id':'owner','kind':'path','status':'ok','bbox_2d':[60,60,90,90],
        'commands':[['M',60,60],['L',90,90]]}
    scene={'source_size':[200,200],'source_sha256':sha(source),'parts':{'frame':frame,'mark':mark},
           'objects':{'owner':{'candidate':{'kind':'rectangle','bbox_2d':[20,20,180,180]},
               'part_keys':['frame','mark'],'completeness_status':'incomplete'}}}
    proposal={'id':'registry','category':'complex_icon','source_box_half_open':[58,58,92,92],
              'target':'original mark','path':str(source)}
    seen=[]
    class Runtime:
        def call(self,root,tag,system,prompt,images,validate,budget):
            seen.append(root.parent.name)
            assert prompt['parts'][0]['key']=='P0' and len(prompt['parts'])==1
            return validate({'action':'image','reason':'source symbol','role':'icon','method':'crop',
                'replace_parts':['P0'],'replace_arrows':[]})
    result=review(scene,{},[],[],source,source,tmp_path/'review',Runtime(),images=[proposal])
    assert len(seen)==1 and result['drop_shape_keys']==['mark']
    assert len(result['assets'])==1


def test_crop_cannot_claim_glyph_inside_external_enclosure(tmp_path):
    from full_recovery.image_assets import review
    from shape_integration.common import sha
    im=Image.new('RGB',(200,200),'white');d=ImageDraw.Draw(im);d.rectangle([80,80,100,100],fill='red')
    source=tmp_path/'source.png';im.save(source)
    frame={'id':'frame','object_id':'badge','kind':'ellipse','status':'ok','bbox_2d':[40,40,120,120],
        'commands':[['M',40,40],['L',120,40],['L',120,120],['L',40,120],['Z']],
        'native_parameters_original':{'center':[80,80],'width':80,'height':80}}
    mark={'id':'mark','object_id':'badge','kind':'path','status':'ok','bbox_2d':[70,70,90,90],
        'commands':[['M',70,70],['L',90,90]]}
    scene={'source_size':[200,200],'source_sha256':sha(source),'parts':{'frame':frame,'mark':mark},
           'objects':{'badge':{'candidate':{'kind':'ellipse','bbox_2d':[40,40,120,120]},'part_keys':['frame','mark']}}}
    proposal={'id':'outside','category':'illustration','source_box_half_open':[68,68,101,101],
              'target':'external illustration','path':str(source)}
    # The review's scope is the independent image, not the badge's child mark.
    class Runtime:
        def call(self,root,tag,system,prompt,images,validate,budget):
            assert prompt['parts']==[]
            return validate({'action':'image','reason':'image only','role':'icon','method':'crop',
                'replace_parts':[],'replace_arrows':[]})
    result=review(scene,{},[],[],source,source,tmp_path/'review',Runtime(),images=[proposal])
    assert not result['drop_shape_keys']


