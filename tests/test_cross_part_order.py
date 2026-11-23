import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from lxml import etree

from PIL import Image, ImageDraw
from shapely.geometry import box

from shape_integration.styling.controller import observed_order
from shape_integration.styling.occlusion import order_internal_by_paint, exclusions


class CrossPartOrderTests(unittest.TestCase):
    def fixture(self):
        table={k:{'object_id':owner,'kind':'rectangle'} for k,owner in
               [('panel','A'),('stripe','A'),('frame','B'),('cell','B')]}
        geoms={'panel':box(0,0,120,100),'stripe':box(40,20,50,80),
               'frame':box(10,10,110,90),'cell':box(70,20,100,80)}
        scene={'objects':{'A':{'part_keys':['panel','stripe']},'B':{'part_keys':['frame','cell']}}}
        rel={'scope':'components','object_order':['A','B'],
             'part_owners':{k:v['object_id'] for k,v in table.items()},
             'component_parts':{'A':['panel','stripe'],'B':['frame','cell']},
             'edges':[{'back':'A','front':'B','origin':'contains'}],
             'effective_edges':[{'back':'A','front':'B','origin':'contains'}]}
        styles={k:{'fill':'#FFFFFF','stroke':'#000000','width_px':1} for k in table}
        return scene,table,geoms,rel,styles

    def test_leaf_relation_does_not_hide_entire_enclosed_compound(self):
        scene,table,geoms,rel,styles=self.fixture();old=copy.deepcopy(rel['edges'])
        observed_order(table,geoms,rel,'frame',behind=['stripe'],scene=scene)
        result=order_internal_by_paint(scene,table,rel,styles,geoms)
        rank={k:i for i,k in enumerate(result['flat_order'])}
        self.assertEqual(rel['edges'],old)
        self.assertLess(rank['panel'],rank['frame'])
        self.assertLess(rank['frame'],rank['stripe'])
        self.assertLess(rank['frame'],rank['cell'])
        # The enclosing panel no longer masks the entire internal frame.
        blocked=exclusions('frame',geoms,result,styles)
        self.assertTrue(blocked.covers(geoms['stripe']))
        self.assertFalse(blocked.covers(geoms['frame']))

    def test_wrong_cross_owner_order_is_disproved_by_visible_source_edges(self):
        scene,table,geoms,rel,styles=self.fixture()
        table={k:{'object_id':k,'kind':'path'} for k in ['rear','front']}
        geoms={'rear':box(10,10,90,70),'front':box(50,30,110,90)}
        styles={k:{'fill':'#FFFFFF','stroke':'#000000','width_px':2} for k in table}
        image=Image.new('RGB',(120,100),'white');draw=ImageDraw.Draw(image)
        for key in ['rear','front']:draw.rectangle(geoms[key].bounds,fill='white',outline='black',width=2)
        scene={'objects':{k:{'part_keys':[k]} for k in table}}
        rel={'scope':'components','object_order':['rear','front'],'edges':[],
             'effective_edges':[],'part_owners':{k:k for k in table},
             'component_parts':{k:[k] for k in table}}
        observed_order(table,geoms,rel,'front',behind=['rear'],scene=scene)
        result=order_internal_by_paint(scene,table,rel,styles,geoms,source=image)
        self.assertEqual(result['flat_order'],['rear','front'])
        self.assertEqual(result['cross_part_rejected_orders'][0]['reason'],
                         'contradicts_visible_source_contours')

    def test_disjoint_leaf_relation_does_not_change_components(self):
        scene,table,geoms,rel,styles=self.fixture()
        self.assertFalse(observed_order(table,geoms,rel,'cell',behind=['stripe'],scene=scene))
        self.assertFalse(rel.get('cross_part_model_edges'))

    def test_native_ppt_keeps_requested_interleaving_and_leaf_identity(self):
        from shape_integration.fitting.scene import write_pptx
        scene,table,geoms,_,_=self.fixture()
        scene.update(source_size=[120,100],parts={})
        keys={}
        for name,part in table.items():
            owner=part['object_id'];key=['test',owner,name];keys[name]=json.dumps(key)
            x0,y0,x1,y1=geoms[name].bounds
            scene['parts'][keys[name]]={**part,'key':key,'id':name,'status':'ok',
                'native_parameters_original':{'center':[(x0+x1)/2,(y0+y1)/2],
                    'width':x1-x0,'height':y1-y0,'angle_radians':0},
                'commands':[['M',x0,y0],['L',x1,y0],['L',x1,y1],['L',x0,y1],['Z']]}
        for owner,obj in scene['objects'].items():
            obj.update(id=owner,state='accepted_by_model',part_keys=[keys[k] for k in obj['part_keys']])
        desired=['panel','frame','stripe','cell']
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'shape.pptx'
            write_pptx(scene,path,paint_order=[keys[k] for k in desired])
            with zipfile.ZipFile(path) as archive:root=etree.fromstring(archive.read('ppt/slides/slide1.xml'))
        ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
        names=root.xpath('//p:sp/p:nvSpPr/p:cNvPr/@name',namespaces=ns)
        self.assertEqual(names,[table[k]['object_id']+'/'+k+' [accepted_by_model]' for k in desired])


if __name__=='__main__':unittest.main()
