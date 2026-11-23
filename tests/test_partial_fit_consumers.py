"""A failed native part must reach repair without breaking its consumers."""
import copy
import json

from PIL import Image

from full_recovery.image_assets import review
from shape_integration.fitting.connectivity import connection_views


def test_image_review_keeps_partial_owner_and_reports_unfitted_part(tmp_path):
    source=tmp_path/'source.png';Image.new('RGB',(100,100),'white').save(source)
    scene={'source_size':[100,100],'parts':{
        'good':{'id':'good','object_id':'owner','status':'ok','kind':'rectangle',
                'bbox_2d':[10,10,30,30],'commands':[]},
        'bad':{'id':'bad','object_id':'owner','status':'failed','kind':'rectangle',
               'bbox_2d':None,'commands':[],'error':'Repeated observed corner'}},
        'objects':{'owner':{'state':'unresolved','part_keys':['good','bad'],
            'candidate':{'kind':'compound_symbol','bbox_2d':[10,10,70,70]}}}}
    before=copy.deepcopy(scene)
    class Runtime:
        def call(self,*args):
            prompt=args[3]
            assert len(prompt['parts'])==1
            assert prompt['target_box']==[10,10,70,70]
            assert prompt['unfitted_parts']==[{'id':'bad','kind':'rectangle','error':'Repeated observed corner'}]
            return args[5]({'action':'keep_native','reason':'Partial native; missing geometry still requires repair.'})
    result=review(scene,{},[],[],source,source,tmp_path/'review',Runtime(),selected=['owner'])
    assert result['unfitted_shape_keys']==['bad']
    assert not result['drop_shape_keys'] and not result['assets']
    assert result['reviews'][0]['action']=='keep_native'
    assert scene==before


def test_connection_detail_survives_failed_neighbor_and_retains_original_labels(tmp_path):
    Image.new('RGB',(100,60),'white').save(tmp_path/'clean.png')
    parts=[
        {'id':'a','status':'ok','kind':'ellipse','bbox_2d':[5,20,25,40],'commands':[]},
        {'id':'failed','status':'failed','kind':'rectangle','bbox_2d':None,'commands':[]},
        {'id':'b','status':'ok','kind':'ellipse','bbox_2d':[75,20,95,40],'commands':[]},
        *[{'id':ident,'status':'ok','kind':'path','bbox_2d':[25,y,75,y],
           'commands':[['M',25,y],['L',75,y]]} for ident,y in [('route1',29),('route2',31)]]]
    (tmp_path/'fit_original.json').write_text(json.dumps({'parts':parts}))
    before=copy.deepcopy(parts)
    views=connection_views({'parts':parts},{},tmp_path)
    evidence=json.loads((tmp_path/'connection_evidence.json').read_text())
    assert len(views)==2 and len(evidence['details'])==1
    assert evidence['details'][0]['routes']['nodes']==['F1','F3']
    assert parts==before
