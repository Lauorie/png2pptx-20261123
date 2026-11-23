import copy
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'lib')]
import numpy as np
from PIL import Image
import evidence as ev
import group_recovery
import group_reference_identity as identity
import geometry_routing
import point_identity
import semantic_geometry
from transport.common import read_json as read,write_json as write


class PointIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.out=Path(self.tmp.name);(self.out/'input').mkdir()
        self.crop=Image.new('RGB',(100,80),'white');self.view=ev.identity_view(self.crop.size)
        self.crop.save(self.out/'input/clean.png');self.crop.save(self.out/'input/context.png')
        write(self.out/'input/transform.json',self.view)
        self.cfg=dict(identity_input_mode='group_refs',identity_geometry_mode='glm_points',
            bbox_geometry_batch='independent',bbox_geometry_mask_evidence=True,geometry_mask_view='union',
            separate_style=True,task_workers=2)
        self.locks=dict(cpu=threading.Semaphore(2))

    def tearDown(self):self.tmp.cleanup()

    def candidate(self,name,owner,y):
        m=np.zeros((80,100),bool);m[y,10:81]=True
        path=self.out/(name+'.png');Image.fromarray(m.astype('uint8')*255).save(path)
        return dict(id=name,owner=owner,mask=m,path=str(path),best_global='G1',category='line')

    def target(self,name='T1',kind='straight'):
        return dict(id=name,kind=kind,source_boxes=['G1'],locator='black connector')

    def answer(self,targets):
        a=identity.validate(dict(status='complete',targets=targets,followups=[]),self.crop.size,self.crop.size,['G1'],self.cfg)
        return identity.resolve(a,['G1'],self.crop,{},[],[],self.out,None,self.cfg,
            [dict(id='G1',bbox=[10,10,90,70])])[0]

    def test_identity_no_longer_asks_model_to_choose_mask_or_geometry_source(self):
        q,_=identity.build(self.crop,[dict(id='G1',box=[10,10,90,70])],self.view,self.crop.size,self.out,self.cfg)
        self.assertNotIn('geometry:',q['output_schema']['targets'])
        # Even an unsolicited legacy decision cannot reintroduce mask fitting.
        a=self.answer([{**self.target(),'geometry':'mask'}])
        self.assertEqual(a['routes'][0]['geometry'],'glm')
        self.assertEqual(read(self.out/'mask_geometry_calls.json')['calls'],0)

    def test_primary_pixels_do_not_include_related_neighbor_masks(self):
        a=self.candidate('M1','this_group',20);b=self.candidate('M2','other_group',40)
        point_identity.evidence_image(self.crop,[],self.view,[a,b],'this_group',[],self.out)
        main=ev.mask(self.out/'input/identity_primary_mask.png');context=ev.mask(self.out/'input/identity_context_mask.png')
        self.assertTrue(main[20,30]);self.assertFalse(main[40,30])
        self.assertTrue(context[40,30]);self.assertFalse(context[20,30])
        original=np.asarray(Image.open(self.out/'input/clean.png'))
        np.testing.assert_array_equal(original,np.asarray(self.crop))

    def test_real_group_enters_point_geometry_without_selector_or_numeric_refit(self):
        c=self.candidate('M1','G001',20);calls=[];test=self
        meta=dict(folder=str(self.out),view=self.view,attempts=[],selected_ids=['M1'],
                  candidates=[{k:v for k,v in c.items() if k!='mask'}])
        class Model:
            def ask(self,folder,q,images,validator):
                calls.append(str(folder))
                if folder.name=='identity':
                    test.assertEqual(len(images),4)
                    test.assertEqual(images[-1][1].name,'identity_evidence.png')
                    return validator(dict(status='complete',targets=[test.target()],followups=[]))
                test.assertNotIn('target_mask_geometry',str(folder))
                return validator(dict(nodes=[['a',12,25],['b',78,25]],routes=[dict(id='R1',
                    paths=[dict(id='p',segments=[dict(kind='line',nodes=['a','b'])])])]))
        group=[dict(id='G1',box=[10,20,81,21])]
        with patch.object(semantic_geometry,'prepare',side_effect=AssertionError('Unused numerical proposals generated')):
            result=group_recovery.group_task(self.crop,group,group,{'G1':'G001'},'G001',self.out,
                self.cfg,Model(),self.locks,prepared=meta,evidence_pool=[])
        self.assertEqual(len(calls),2)
        self.assertEqual(result['mask_geometry_calls'],0)
        row=read(self.out/'rows_local.json')[0]
        self.assertEqual(row['start'],[12.,25.]);self.assertEqual(row['end'],[78.,25.])
        self.assertEqual(row['coordinate_source'],'glm_segmented_points')
        self.assertFalse((self.out/'target_mask_geometry').exists())

    def test_two_confirmed_targets_start_geometry_concurrently(self):
        a=self.answer([self.target('T1'),self.target('T2')]);barrier=threading.Barrier(2)
        class Model:
            def ask(self,folder,q,images,validator):
                barrier.wait(timeout=10)
                if len(q['targets'])!=1:raise AssertionError('Targets merged into one request')
                return validator(dict(nodes=[['a',12,25],['b',78,25]],routes=[dict(id='R1',
                    paths=[dict(id='p',segments=[dict(kind='line',nodes=['a','b'])])])]))
        rows,_,issues,count=geometry_routing.recover(a,{},[],'G1',self.crop,self.out,Model(),self.locks,self.cfg)
        self.assertEqual((len(rows),issues,count),(2,[],2))
        self.assertFalse(set(rows[0]['node_ids'])&set(rows[1]['node_ids']))

    def test_block_arrow_uses_shape_points_directly_and_never_publishes_locator_on_failure(self):
        a=self.answer([self.target(kind='block_arrow')]);calls=[]
        class Model:
            def ask(self,folder,q,images,validator):
                calls.append(str(folder))
                return validator(dict(status='ok',points=[dict(id='a',xy=[20,20]),dict(id='b',xy=[75,35]),dict(id='c',xy=[20,60])],
                    parts=[dict(id='outline',kind='path',closed=True,segments=[
                        dict(kind='line',point_ids=['a','b']),dict(kind='line',point_ids=['b','c']),dict(kind='line',point_ids=['c','a'])])]))
        rows,_,issues,count=geometry_routing.recover(a,{},[],'G1',self.crop,self.out,Model(),self.locks,self.cfg)
        self.assertEqual((len(rows),issues,count),(1,[],1));self.assertEqual(len(calls),1)
        self.assertEqual(rows[0]['coordinate_source'],'glm_shape_geometry')
        self.assertNotIn('locator_only',rows[0]);self.assertEqual(rows[0]['shape_parts'][0]['commands'][0],['M',20.0,20.0])
        class Unresolved:
            def ask(self,folder,q,images,validator):return validator(dict(status='needs_context',points=[],parts=[]))
        rows,_,issues,_=geometry_routing.recover(a,{},[],'G1',self.crop,self.out,Unresolved(),self.locks,self.cfg)
        self.assertEqual(rows,[]);self.assertTrue(issues)


if __name__=='__main__':unittest.main()
