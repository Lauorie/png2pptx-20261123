import copy
import json
import numpy as np
import pytest
from PIL import Image, ImageDraw

from full_recovery.source_external_repair import repair
from shape_integration.common import save


def fixture(tmp_path):
    original=Image.new('RGB',(200,120),'white')
    ImageDraw.Draw(original).line([(20,65),(80,50),(150,60)],fill='black',width=2)
    images=[]
    for name in ('clean','shape_native','arrows_native'):
        path=tmp_path/(name+'.png');original.save(path);images.append((name,path))
    def native(ident,layer,points):
        return {'id':ident,'key':ident,'owner':'compound','layer':layer,'points':np.asarray(points,float),
            'refs':[ident],'heads':['none','none'],'style':{'stroke':'#000000','width_px':2}}
    aliases={'C1':native('bad_fit','shape',[(20,85),(145,85)]),
        'C2':native('complete_curve','arrows',[(20,65),(80,50),(150,60)]),
        'C3':native('unrelated','arrows',[(1,5),(10,5)])}
    objects=[{'id':'image','kind':'image_content','bbox':[130,20,190,90],
        'visible_count':1,'asset_id':'I1','description':'image interior'},
        {'id':'known','kind':'line','bbox':[0,0,12,10],'visible_count':1,'description':'already known unrelated line'}]
    mapping={'objects':[{'id':'image','candidate_ids':['C1'],'keep_ids':['C1'],
        'status':'uncertain','reason':'invalid external image claim preserved'},
        {'id':'known','candidate_ids':['C3'],'keep_ids':['C3'],'status':'resolved','reason':'known'}],
        'unassigned':[{'candidate_id':'C2','classification':'uncertain','reason':'no matching frozen source ID'}]}
    assets=[{'id':'I1','source_box_half_open':[130,20,190,90]}]
    return original,images,aliases,{'objects':objects},mapping,assets


def test_external_identity_is_observed_before_native_comparison_and_does_not_override_known_paths(tmp_path):
    original,images,aliases,inventory,mapping,assets=fixture(tmp_path)
    before=copy.deepcopy(mapping)
    class Model:
        calls=0
        def call(self,folder,tag,system,prompt,views,validator,budget):
            self.calls+=1
            if tag=='source_completion':
                assert len(views)==1 and 'candidates' not in prompt
                w,h=prompt['coordinate_frame']['size']
                return validator({'objects':[{'id':'local','kind':'line','bbox':[0,0,w-1,h-1],
                    'visible_count':1,'count_status':'exact','description':'visible curved external connection'}]})
            assert tag=='select' and {c['id'] for c in prompt['candidates']}=={'C1','C2'}
            support={c['id']:c['original_external_stroke_support']['near_colour_fraction'] for c in prompt['candidates']}
            assert support['C2']>support['C1']
            return validator({'objects':[{'id':'local','candidate_ids':['C1','C2'],'keep_ids':['C2'],
                'status':'resolved','reason':'actual original curvature and complete extent'}],'unassigned':[]})
    model=Model()
    observed,result,calls=repair(inventory,mapping,aliases,assets,images,original,[0,0,200,120],
        np.ones(2),tmp_path/'repair',model,save)
    assert calls==model.calls==2 and mapping==before
    assert result['objects'][0]['candidate_ids']==[]
    assert result['objects'][1]==before['objects'][1]
    assert result['objects'][-1]['candidate_ids']==['C1','C2'] and result['objects'][-1]['keep_ids']==['C2']
    assert not result['unassigned']
    assert observed['objects'][-1]['source_observation']=='original_only_external_completion'
    assert np.array_equal(aliases['C1']['points'],[[20,85],[145,85]])


def test_intrinsic_image_strokes_cannot_trigger_external_source_discovery(tmp_path):
    original,images,aliases,inventory,mapping,assets=fixture(tmp_path)
    assets[0]['source_box_half_open']=[0,0,200,120]
    class NoCalls:
        def call(self,*args,**kwargs):raise AssertionError('image interior is not external linework')
    observed,result,calls=repair(inventory,mapping,aliases,assets,images,original,[0,0,200,120],
        np.ones(2),tmp_path/'repair',NoCalls(),save)
    assert calls==0 and result==mapping and observed==inventory


def test_a_disjoint_wrong_image_assignment_is_not_an_external_image_connection(tmp_path):
    original,images,aliases,inventory,mapping,assets=fixture(tmp_path)
    assets[0]['source_box_half_open']=[170,5,195,20]
    class NoCalls:
        def call(self,*args,**kwargs):raise AssertionError('disjoint source ownership needs reassignment, not image-interface discovery')
    observed,result,calls=repair(inventory,mapping,aliases,assets,images,original,[0,0,200,120],
        np.ones(2),tmp_path/'repair',NoCalls(),save)
    assert calls==0 and result==mapping and observed==inventory


def test_local_completion_preserves_a_native_route_whose_unseen_extent_leaves_the_crop(tmp_path):
    original,images,aliases,inventory,mapping,assets=fixture(tmp_path)
    aliases['C4']={**aliases['C2'],'id':'crossing_context','key':'context',
        'points':np.asarray([[60,15],[60,75]],float)}
    mapping['unassigned'].append({'candidate_id':'C4','classification':'uncertain','reason':'independent context route'})
    class Model:
        def call(self,folder,tag,system,prompt,views,validator,budget):
            if tag=='source_completion':
                w,h=prompt['coordinate_frame']['size']
                return validator({'objects':[{'id':'local','kind':'line','bbox':[0,0,w-1,h-1],
                    'visible_count':1,'description':'actual source connector'}]})
            assert {c['id'] for c in prompt['candidates']}=={'C1','C2'}
            return validator({'objects':[{'id':'local','candidate_ids':['C1','C2'],'keep_ids':['C2'],
                'status':'resolved','reason':'source supported complete curve'}],'unassigned':[]})
    observed,result,calls=repair(inventory,mapping,aliases,assets,images,original,[0,0,200,120],
        np.ones(2),tmp_path/'repair',Model(),save)
    assert calls==2 and result['objects'][-1]['keep_ids']==['C2']
    assert result['unassigned']==[mapping['unassigned'][-1]]


def test_completion_budget_prioritizes_actual_source_gaps_over_image_edge_protrusions(tmp_path):
    original,images,aliases,inventory,mapping,assets=fixture(tmp_path)
    aliases['C4']={**aliases['C1'],'id':'edge','key':'edge','owner':'a_first',
                   'points':np.asarray([[22,5],[22,15]],float)}
    aliases['C5']={**aliases['C1'],'id':'far','key':'far','owner':'far_first',
                   'points':np.asarray([[70,5],[80,5]],float)}
    assets.append({'id':'I0','source_box_half_open':[0,0,20,20]})
    inventory['objects'].append({'id':'edge_image','kind':'image_content','asset_id':'I0',
        'visible_count':1,'bbox':[0,0,20,20],'description':'another source image'})
    mapping['objects'].append({'id':'edge_image','candidate_ids':['C4','C5'],'keep_ids':['C4','C5'],
        'status':'uncertain','reason':'a small intrinsic edge protrusion'})
    class OneRegion:
        calls=0
        def call(self,folder,tag,system,prompt,views,validator,budget):
            self.calls+=1
            assert self.calls<=2
            if tag=='source_completion':
                w,h=prompt['coordinate_frame']['size']
                return validator({'objects':[{'id':'local','kind':'line','bbox':[0,0,w-1,h-1],
                    'visible_count':1,'description':'external source connection'}]})
            assert {c['id'] for c in prompt['candidates']}=={'C1','C2'}
            return validator({'objects':[{'id':'local','candidate_ids':['C1','C2'],'keep_ids':['C2'],
                'status':'resolved','reason':'complete original source path'}],'unassigned':[]})
    model=OneRegion()
    observed,result,calls=repair(inventory,mapping,aliases,assets,images,original,[0,0,200,120],
        np.ones(2),tmp_path/'repair',model,save,maximum=1)
    audit=json.loads((tmp_path/'repair/external_source_completion.json').read_text())
    assert calls==2 and audit[0]['candidate_ids']==['C1']
    assert audit[0]['source_gap_score']>audit[1]['source_gap_score']
    assert audit[1]['candidate_ids']==['C4'] and audit[1]['status']=='uncertain'


@pytest.mark.parametrize('mode',['error','no_source'])
def test_failed_or_empty_source_observation_cannot_delete_native_content(tmp_path,mode):
    original,images,aliases,inventory,mapping,assets=fixture(tmp_path)
    class Undecided:
        def call(self,folder,tag,system,prompt,views,validator,budget):
            if mode=='error':raise RuntimeError('source review unavailable')
            return validator({'objects':[],'uncertain':['source path not established']})
    observed,result,calls=repair(inventory,mapping,aliases,assets,images,original,[0,0,200,120],
        np.ones(2),tmp_path/'repair',Undecided(),save)
    assert calls==1 and result==mapping and observed==inventory
    audit=json.loads((tmp_path/'repair/external_source_completion.json').read_text())
    assert audit[0]['status']=='uncertain'
