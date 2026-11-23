"""Cross-stage regressions using synthetic source geometry, no task IDs."""
import copy
import tempfile
import unittest
from pathlib import Path
from PIL import Image, ImageDraw
from full_recovery.quality_choice import apply_choices, validate_selection
from full_recovery.handoff import ellipsis_shape_ownership
from shape_integration.fitting.frame_edges import measure_fill
from shape_integration.fitting.control import process_object
from shape_integration.fitting.seed import revision_seed
from shape_integration.common import read,save,sha
import test_pipeline as fixtures
from test_pipeline import FakeRuntime,observed,reviewed

class SelectionTests(unittest.TestCase):
    def test_detached_port_fit_uncertainty_only_nominates_nearest_forward_node(self):
        import numpy as np
        from full_recovery.quality_choice import port_node
        nodes=[('node',np.array([20.,50.]),np.array([10.,10.]))]
        self.assertEqual(port_node(np.array([38.,50.]),np.array([-50.,40.]),nodes,[]),'node')
        self.assertIsNone(port_node(np.array([38.,50.]),np.array([50.,-40.]),nodes,[]))
        nodes.append(('neighbour',np.array([20.,52.]),np.array([10.,10.])))
        self.assertIsNone(port_node(np.array([38.,51.]),np.array([-50.,0.]),nodes,[]))

    def test_same_branch_duplicate_is_not_kept_with_winner(self):
        rows=[{'id':n,'layer':branch,'key':n,'keys':[n],'refs':[n]}
              for n,branch in [('s1','shape'),('s2','shape'),('a1','arrows'),('unique','shape')]]
        choice=validate_selection({'keep_ids':['s2'],'reason':'one visible source route'},['s1','s2','a1'])
        result=apply_choices(rows,[{**choice,'candidate_ids':['s1','s2','a1']}])
        self.assertEqual(result['drop_shape_keys'],['s1'])
        self.assertEqual(result['drop_arrow_names'],['a1'])
        self.assertNotIn('unique',result['drop_shape_keys'])

    def test_complementary_routes_and_required_frame_survive(self):
        self.assertEqual(validate_selection({'keep_ids':['a','b'],'reason':'two source paths'},['a','b'])['keep_ids'],['a','b'])
        with self.assertRaises(ValueError):validate_selection({'keep_ids':[],'reason':'remove'},['frame'],allow_empty=True,required=['frame'])

    def test_compound_stroked_ellipsis_owned_once_by_text(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'source.png';im=Image.new('RGB',(80,100),'white');draw=ImageDraw.Draw(im)
            commands=[]
            for y in (30,45,60):
                draw.rectangle((38,y-2,42,y+2),fill='black')
                commands.extend([['M',40,y-2],['L',42,y],['L',40,y+2],['L',38,y],['L',40,y-2]])
            im.save(source)
            scene={'parts':{'dots':{'status':'ok','kind':'path','bbox_2d':[38,28,42,62],'commands':commands}}}
            drop,_=ellipsis_shape_ownership(scene,{'dots':{'fill':None,'stroke':'#000000','width_px':2}},
                [{'id':'t','text':'⋮','color':'#000000','bbox_2d':[36,26,44,64]}],source)
            self.assertEqual(drop,['dots'])
            scene['parts']['dots']['commands'] += [['M',5,80],['L',70,80]]
            self.assertEqual(ellipsis_shape_ownership(scene,{'dots':{'fill':None,'stroke':'#000000'}},
                [{'id':'t','text':'⋮','color':'#000000','bbox_2d':[36,26,44,64]}],source)[0],[])

    def test_fill_edges_measure_each_unit_without_regularizing_pitch(self):
        image=Image.new('RGB',(200,180),'white');draw=ImageDraw.Draw(image)
        draw.rounded_rectangle((30,25,150,65),radius=3,fill='#FFC090')
        draw.rounded_rectangle((30,105,150,150),radius=3,fill='#FFC090')
        for proposed,expected in [([31,23,149,60],[29.5,24.5,150.5,65.5]),([31,110,149,155],[29.5,104.5,150.5,150.5])]:
            fitted,audit=measure_fill({'angle_radians':0},image,proposed)
            self.assertTrue(audit['applied']);self.assertEqual(fitted,expected)
        draw.polygon([(30,25),(150,25),(90,65)],fill='white')
        self.assertIsNone(measure_fill({'angle_radians':0},image,[30,25,150,65])[0])

class SeedContextTests(unittest.TestCase):
    setUp=fixtures.PipelineTests.setUp
    run_object=fixtures.PipelineTests.run_object
    def test_cached_geometry_gets_current_peers_without_changing_source(self):
        self.run_object(FakeRuntime(self.config));old=self.folder/'objects/G1';before=sha(old/'record.json')
        context={'registered_peer_shapes':[{'id':'peer','label':'output nodes','kind':'compound_symbol'}]}
        runtime=FakeRuntime(self.config)
        result=process_object(self.work/'new',self.candidate,self.item,self.work,runtime,self.options,
                              namespace='new',seed=revision_seed(old),input_context=context)
        self.assertEqual(result['state'],'accepted_by_model')
        prompt=next(p for tag,p in runtime.events if tag=='check')
        self.assertEqual(prompt['registered_peer_shapes'],context['registered_peer_shapes'])
        self.assertEqual(sha(old/'record.json'),before)
        self.assertEqual([t for t,p in runtime.events],['check'])

class CoverageDispatchTests(unittest.TestCase):
    setUp=fixtures.PipelineTests.setUp
    def test_existing_owner_missing_part_is_repaired_without_duplicate_owner(self):
        from concurrent.futures import ThreadPoolExecutor
        from shape_integration.pipeline import run_case,source_manifest
        source=self.work/'old';case=source/'cases/case'
        save(source/'manifest.json',{'images':[self.item]})
        save(case/'geometry.json',{'image_size':self.item['source_size'],'coordinate_system':'image_pixel_centers','objects':[self.candidate]})
        save(case/'summary.json',{'image_coverage_status':'complete','errors':{},'local_failures':0})
        save(case/'dedup/image_boxes_original.json',{'assets':[]})
        self.options['coverage_rounds']=1
        def respond(tag,prompt,count):
            if tag=='coverage':
                kinds=prompt['output_schema']['properties']['repair_existing']['items']['properties']['issues']['items']['properties']['kind']['enum']
                self.assertEqual(set(kinds),{'missing','extra'})
                return {'status':'uncertain','new_objects':[], 'possible_duplicates':[], 'uncertain':[],
                    'repair_existing':[{'object_id':'G1','issues':[{'kind':'missing','part_ids':[],
                        'evidence':'One source rectangle is unrepresented','action':'Add the missing rectangle to this owner'}]}]}
            if tag=='check':
                raw=reviewed(prompt)
                targets=prompt.get('required_source_observations',{}).get('targets',[])
                if targets:
                    self.assertEqual(targets[0]['id'],'coverage_request_0')
                    self.assertEqual(len(prompt['fitted_parts']),3)
                    raw['source_observations']=[{'id':row['id'],'source_relation':'supported',
                        'evidence':'The requested rectangle is present alongside the two preserved rectangles'}
                        for row in targets]
                return raw
            if 'revision_task' not in prompt:return observed('G1')
            self.assertTrue(prompt['revision_task']['allow_new_parts_for_missing_geometry'])
            return {'status':'ok','points':[{'id':'q'+str(i),'xy':list(xy)} for i,xy in enumerate([(350,350),(450,350),(450,450),(350,450)])],
                    'parts':[{'id':'c','kind':'rectangle','point_ids':['q0','q1','q2','q3']}]}
        runtime=FakeRuntime(self.config,respond)
        with ThreadPoolExecutor(max_workers=2) as pool:
            run_case(self.work,self.work/'new',self.item,runtime,self.config,pool,source,source_manifest(source,[self.item]),False)
        result=read(self.work/'new/cases/case/fitting/scene.json')
        self.assertEqual(list(result['objects']),['G1'])
        self.assertEqual({p['id'] for p in result['parts'].values()},{'a','b','c'},str(result['objects']))
        self.assertEqual([t for t,p in runtime.events].count('check'),2)
        self.assertEqual([t for t,p in runtime.events].count('coverage'),1)
