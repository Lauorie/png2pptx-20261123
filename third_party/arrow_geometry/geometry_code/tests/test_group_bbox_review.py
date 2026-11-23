import copy
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'lib')]
import numpy as np
from PIL import Image
import evidence as ev
import geometry_routing
import group_bbox_identity as identity
import scope_review
import semantic_geometry
import target_locator
from transport.common import write_json as write


def target(geometry='glm'):
    return dict(id='T1', bbox=[10, 10, 90, 60], kind='network' if geometry == 'glm' else 'straight',
                geometry=geometry, locator='Upper horizontal connection')


def source_row(name):
    return dict(id=name, parent_id='shared_parent', kind='straight',
                start=[10., 20.], via=[], end=[80., 20.], heads=['none', 'none'],
                style=dict(color='#123456', width_px=2, start=None, end=None,
                           dash=dict(type='solid', preset='solid')))


class GroupBBoxReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        (self.out / 'input').mkdir()
        self.crop = Image.new('RGB', (100, 80), 'white')
        self.crop.save(self.out / 'input/clean.png')
        self.crop.save(self.out / 'input/context.png')
        self.locks = dict(cpu=threading.Semaphore(1))
        write(self.out / 'input/transform.json', ev.identity_view(self.crop.size))

    def tearDown(self):
        self.tmp.cleanup()

    def test_padding_preserves_evidence_frame_and_source_pixels(self):
        original=np.asarray(self.crop).copy()
        boxes=[dict(id='G1',bbox=[20,20,80,60])]
        before=copy.deepcopy(boxes)
        annotations=target_locator.draw(self.crop,boxes,self.out/'corners.png',.05,'corners')
        self.assertEqual(boxes,before)
        self.assertEqual(annotations[0]['display_bbox'],[17.,18.,83.,62.])
        np.testing.assert_array_equal(np.asarray(self.crop),original)

    def test_independent_targets_are_parallel_and_keep_distinct_point_ids(self):
        raw=dict(status='complete',targets=[target(),{**target(),'id':'T2'}],followups=[])
        answer,_,_=identity.resolve(raw,['G1','G2'],self.crop,{},[],[],self.out,None)
        barrier=threading.Barrier(2)
        class Model:
            def ask(self,folder,prompt,images,validator):
                barrier.wait(timeout=10)
                if len(prompt['targets'])!=1:raise AssertionError('Independent targets were batched')
                return validator(dict(nodes=[['a',10,20],['b',80,20]],routes=[dict(id='R1',
                    paths=[dict(id='p',segments=[dict(kind='line',nodes=['a','b'])])])]))
        rows,_,issues,count=geometry_routing.recover(answer,{},[],'G1',self.crop,self.out,
            Model(),self.locks,dict(identity_input_mode='group_boxes',separate_style=True,
                                   bbox_geometry_batch='independent',task_workers=2))
        self.assertEqual((count,issues,len(rows)),(2,[],2))
        self.assertFalse(set(rows[0]['node_ids'])&set(rows[1]['node_ids']))
        self.assertEqual([r['identity_index'] for r in rows],[1,2])

    def test_optional_masks_do_not_become_geometry_ownership(self):
        raw=dict(status='complete',targets=[{**target(),'kind':'polyline'}],followups=[])
        answer,_,_=identity.resolve(raw,['G1'],self.crop,{},[],[],self.out,None)
        m=np.zeros((80,100),bool);m[30,20:70]=True
        path=self.out/'M1.png';Image.fromarray(m.astype('uint8')*255).save(path)
        class Model:
            def ask(self,folder,prompt,images,validator):
                if prompt['targets'][0]['optional_mask_evidence']!=['M1']:raise AssertionError('Evidence lost')
                return validator(dict(nodes=[['a',10,20],['b',80,20]],routes=[dict(id='R1',
                    paths=[dict(id='p',segments=[dict(kind='line',nodes=['a','b'])])])]))
        rows,supports,issues,_=geometry_routing.recover(answer,{},[dict(id='M1',mask=m,path=str(path))],
            'G1',self.crop,self.out,Model(),self.locks,dict(identity_input_mode='group_boxes',
                separate_style=True,bbox_geometry_mask_evidence=True))
        self.assertEqual(issues,[])
        self.assertEqual(rows[0]['source_masks'],[])
        self.assertIsNone(supports[rows[0]['id']])

    def test_identity_keeps_original_boxes_without_mask_tasks(self):
        group = [dict(id='G9', box=[10, 20, 30, 25]), dict(id='G30', box=[15, 20, 90, 30])]
        q, images = identity.build(self.crop, group, ev.identity_view(self.crop.size), self.crop.size, self.out)
        self.assertEqual([x['id'] for x in q['group_boxes']], ['G9', 'G30'])
        self.assertEqual(q['group_boxes'][0]['bbox'], [10, 20, 30, 25])
        self.assertFalse({'masks', 'proposals', 'head_types', 'nodes'} & q.keys())
        self.assertEqual(len(images), 3)
        self.assertEqual([p.name for _, p in images], ['clean.png', 'target_boxes.png', 'context.png'])
        self.assertIn('Do not discover every arrow', ' '.join(q['instructions']))
        for targets in [[target(), target()], [{**target(), 'bbox': [0, 0, 200, 200]}]]:
            with self.assertRaises(ValueError):
                identity.validate(dict(status='complete', targets=targets), self.crop.size, self.crop.size)

    def test_one_network_identity_accepts_independent_geometry_paths(self):
        raw = dict(status='complete', targets=[target()], followups=[])
        answer, count, issues = identity.resolve(raw, ['G9'], self.crop, {}, [], [], self.out, None)
        self.assertEqual((count, issues), (0, []))
        captured = {}

        class Model:
            def ask(self, folder, prompt, images, validator):
                captured.update(prompt=prompt, images=images)
                return validator(dict(nodes=[['a', 10, 20], ['b', 80, 20], ['c', 20, 60], ['d', 70, 10]],
                                      routes=[dict(id='R1', paths=[
                                          dict(id='upper', segments=[dict(kind='line', nodes=['a', 'b'])]),
                                          dict(id='crossing', segments=[dict(kind='line', nodes=['c', 'd'])])])]))

        rows, _, issues, count = geometry_routing.recover(answer, {}, [], 'G1', self.crop, self.out,
                                                         Model(), self.locks,
                                                         dict(identity_input_mode='group_boxes', separate_style=True))
        self.assertEqual((len(rows), count, issues), (2, 1, []))
        self.assertEqual(captured['prompt']['targets'][0]['bbox'], target()['bbox'])
        self.assertFalse(any('mask_measurements' in t for t in captured['prompt']['targets']))
        self.assertEqual([p.name for _, p in captured['images']], ['clean.png', 'target_boxes.png', 'context.png'])
        self.assertEqual(len({r['physical_target_id'] for r in rows}), 1)
        self.assertTrue(all(not r['shared_nodes'] for r in rows))
        self.assertEqual(rows[0]['start'], [10., 20.])

    def test_ordinary_target_still_uses_selected_mask_measurement(self):
        mask = np.zeros((80, 100), bool)
        mask[19:22, 10:81] = True
        path = self.out / 'mask.png'
        Image.fromarray(mask.astype('uint8') * 255).save(path)
        candidates = [dict(id='M1', mask=mask, path=str(path))]
        proposals, info = semantic_geometry.prepare(self.crop, [], candidates, self.out)
        self.assertTrue(proposals)
        proposal_id = next(iter(proposals))

        class Model:
            def ask(self, folder, prompt, images, validator):
                return validator(dict(nodes=[], routes=[dict(id='T1', kind='straight',
                                      geometry='mask', masks=['M1'], proposal=proposal_id, nodes=[])]))

        raw = dict(status='complete', targets=[target('mask')], followups=[])
        answer, count, issues = identity.resolve(raw, ['G9'], self.crop, proposals, candidates, info, self.out, Model())
        self.assertEqual((count, issues), (1, []))
        rows, _, issues, count = geometry_routing.recover(answer, proposals, candidates, 'G1', self.crop,
                                                         self.out, None, self.locks, dict(separate_style=True))
        self.assertEqual((len(rows), count, issues), (1, 0, []))
        self.assertEqual(rows[0]['coordinate_source'], 'mask_no_skeleton')

    def test_multiple_confirmed_targets_share_one_geometry_request_not_identity(self):
        raw = dict(status='complete', targets=[target(), {**target(), 'id': 'T2'}], followups=[])
        answer, _, _ = identity.resolve(raw, ['G1', 'G2'], self.crop, {}, [], [], self.out, None)
        calls = []

        class Model:
            def ask(self, folder, prompt, images, validator):
                calls.append(prompt)
                return validator(dict(nodes=[['a', 10, 20], ['b', 80, 20], ['c', 10, 50], ['d', 80, 50]],
                                      routes=[dict(id='R1', paths=[dict(id='p', segments=[dict(kind='line', nodes=['a', 'b'])])]),
                                              dict(id='R2', paths=[dict(id='p', segments=[dict(kind='line', nodes=['c', 'd'])])])]))

        rows, _, issues, count = geometry_routing.recover(answer, {}, [], 'G1', self.crop, self.out,
                                                         Model(), self.locks,
                                                         dict(identity_input_mode='group_boxes', separate_style=True))
        self.assertEqual((len(calls), count, issues), (1, 1, []))
        self.assertEqual(len({r['physical_target_id'] for r in rows}), 2)
        self.assertEqual(len({r['parent_id'] for r in rows}), 2)

    def test_review_removes_path_not_entire_shared_parent(self):
        original = [dict(group='g', rows=[source_row('g:T1/a'), source_row('g:T1/b')])]
        before = copy.deepcopy(original)
        changed = scope_review.apply(original, {'g:T1/a'})
        self.assertEqual(changed[0]['rows'], [before[0]['rows'][1]])
        self.assertEqual(original, before)
        with self.assertRaises(ValueError):
            scope_review.apply(original, {'unknown'})

    def test_uncertain_kept_and_review_errors_cannot_delete(self):
        original = [dict(group='g', rows=[source_row('a'), source_row('b')])]

        class Model:
            def ask(self, folder, prompt, images, validator):
                return validator(dict(remove=[dict(id='P001', reason='Closed cell border')],
                                      uncertain=[dict(id='P002', reason='Mixed connector and border')]))

        updated, issues = scope_review.run(self.crop, original, self.out / 'review', Model(), self.locks)
        self.assertEqual(updated[0]['rows'], [original[0]['rows'][1]])
        self.assertEqual(issues[0]['stage'], 'scope_review_uncertain')
        for answer in [dict(remove=[dict(id='new', reason='Invented')], uncertain=[]), None]:
            class FailingModel:
                def ask(self, folder, prompt, images, validator):
                    if answer is None:
                        raise TimeoutError('offline failure')
                    return validator(answer)
            updated, issues = scope_review.run(self.crop, original, self.out / 'failed', FailingModel(), self.locks)
            self.assertEqual(updated, original)
            self.assertTrue(issues[0]['retained_all'])


if __name__ == '__main__':
    unittest.main()
