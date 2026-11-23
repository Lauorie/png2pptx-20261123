import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'lib')]
import numpy as np
from PIL import Image,ImageDraw
import evidence as ev
import geometry_routing
import group_recovery
import group_reference_identity as identity
from transport.common import read_json as read,write_json as write


class ReferenceIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.out=Path(self.tmp.name)
        (self.out/'input').mkdir();self.crop=Image.new('RGB',(100,80),'white')
        self.boxes=[dict(id='G1',bbox=[10,10,90,60])]

    def tearDown(self):self.tmp.cleanup()

    def target(self,tid='T1',refs=None):
        return dict(id=tid,source_boxes=['G1'] if refs is None else refs,
                    kind='polyline',geometry='glm',locator='yellow f label connector')

    def test_model_cannot_replace_original_candidate_bounds(self):
        raw=dict(status='complete',targets=[{**self.target(),'bbox':[20,20,30,30]}],followups=[])
        validated=identity.validate(raw,self.crop.size,self.crop.size,['G1'])
        self.assertNotIn('bbox',validated['targets'][0])
        answer,count,issues=identity.resolve(validated,['G1'],self.crop,{},[],[],self.out,None,
                                             source_boxes=self.boxes)
        self.assertEqual((count,issues),(0,[]))
        self.assertEqual(answer['routes'][0]['target_bbox'],[10,10,90,60])
        self.assertFalse(read(self.out/'target_reference_extents.json')[0]['model_estimated'])

    def test_dense_observation_can_return_multiple_targets_without_exhaustive_refs(self):
        raw=dict(status='complete',targets=[self.target('T1',[]),self.target('T2',[])],followups=[])
        valid=identity.validate(raw,self.crop.size,self.crop.size,['G1'])
        answer,_,_=identity.resolve(valid,['G1'],self.crop,{},[],[],self.out,None,source_boxes=self.boxes)
        self.assertEqual(len(answer['routes']),2)
        self.assertTrue(all(r['target_bbox']==[0,0,100,80] for r in answer['routes']))
        self.assertTrue(all(r['target_bbox_source']=='original_observation' for r in answer['routes']))
        self.assertTrue(all(not r['masks'] for r in answer['routes']))

    def test_unreferenced_target_gets_no_unrelated_geometry_box(self):
        self.crop.save(self.out/'input/clean.png');self.crop.save(self.out/'input/context.png')
        write(self.out/'input/transform.json',ev.identity_view(self.crop.size))
        raw=dict(status='complete',targets=[self.target(refs=[])],followups=[])
        valid=identity.validate(raw,self.crop.size,self.crop.size,['G1'])
        answer,_,_=identity.resolve(valid,['G1'],self.crop,{},[],[],self.out,None,source_boxes=self.boxes)
        captured={}
        class Model:
            def ask(self,folder,prompt,images,validator):
                captured.update(prompt=prompt,images=images)
                return validator(dict(nodes=[['a',10,20],['b',80,20]],routes=[dict(id='R1',
                    paths=[dict(id='p',segments=[dict(kind='line',nodes=['a','b'])])])]))
        rows,_,issues,_=geometry_routing.recover(answer,{},[],'G1',self.crop,self.out,Model(),
            dict(cpu=threading.Semaphore(1)),dict(identity_input_mode='group_refs',separate_style=True))
        self.assertEqual((len(rows),issues),(1,[]))
        self.assertNotIn('bbox',captured['prompt']['targets'][0])
        self.assertIn('locator_scope',captured['prompt']['targets'][0])
        np.testing.assert_array_equal(np.asarray(Image.open(self.out/'geometry/F001/input/target_boxes.png')),np.asarray(self.crop))

    def test_light_shared_coverage_is_actually_sent_without_mask_census(self):
        draw=ImageDraw.Draw(self.crop);draw.line([(10,20),(50,20)],fill='black');draw.line([(60,50),(89,50)],fill='black')
        self.crop.save(self.out/'input/clean.png');self.crop.save(self.out/'input/context.png')
        own=np.zeros((80,100),bool);own[20,10:51]=True
        other=np.zeros((80,100),bool);other[50,60:90]=True
        for name,m in [('own',own),('other',other)]:Image.fromarray(m.astype('uint8')*255).save(self.out/(name+'.png'))
        view=ev.identity_view(self.crop.size)
        meta=dict(folder=str(self.out),view=view,attempts=[],selected_ids=['M1'],
                  candidates=[dict(id='M1',path=str(self.out/'own.png'))])
        pool=[dict(observation='other_group',id='M1',path=str(self.out/'other.png'),view=view,bbox_global=[60,50,90,51])]
        captured={}
        class Model:
            def ask(self,folder,prompt,images,validator):
                captured.update(prompt=prompt,images=images)
                return validator(dict(status='complete',targets=[],followups=[]))
        group=[dict(id='G1',box=[10,20,51,21])]
        result=group_recovery.group_task(self.crop,group,group,{'G1':'G1'},'G1',self.out,
            dict(identity_input_mode='group_refs',identity_shared_coverage=True,separate_style=True),
            Model(),dict(cpu=threading.Semaphore(1)),prepared=meta,evidence_pool=pool)
        self.assertEqual(result['state'],'completed')
        self.assertEqual(len(captured['images']),4)
        self.assertEqual(captured['images'][-1][1].name,'shared_coverage.png')
        self.assertEqual(captured['prompt']['shared_evidence']['additional_mask_pixels'],30)
        self.assertFalse({'masks','proposals'}&captured['prompt'].keys())
        np.testing.assert_array_equal(np.asarray(Image.open(self.out/'input/clean.png')),np.asarray(self.crop))
        shared=np.asarray(Image.open(self.out/'input/shared_coverage.png'))
        self.assertFalse(np.array_equal(shared[50,70],np.asarray(self.crop)[50,70]))

    def test_identity_protocol_has_no_target_bbox_or_local_masks(self):
        q,images=identity.build(self.crop,[dict(id='G1',box=[10,10,90,60])],
            ev.identity_view(self.crop.size),self.crop.size,self.out,
            dict(bbox_display_mode='corners',bbox_display_padding=.05))
        self.assertNotIn('bbox',q['output_schema']['targets'])
        self.assertFalse({'masks','proposals','nodes'}&q.keys())
        self.assertEqual(len(images),3)
        self.assertEqual(q['group_boxes'],self.boxes)
        raw=dict(status='complete',targets=[self.target(refs=['G999'])],followups=[])
        with self.assertRaises(ValueError):identity.validate(raw,self.crop.size,self.crop.size,['G1'])


if __name__=='__main__':unittest.main()
