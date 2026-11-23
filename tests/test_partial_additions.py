"""A local unfinished addition cannot erase independent repaired siblings."""
import copy
import unittest

from shape_integration.fitting.adapter import fit_object
from shape_integration.fitting.frames import FitFrame
from shape_integration.fitting.partial_additions import select
from shape_integration.common import read
import test_pipeline as fixtures
from test_pipeline import FakeRuntime
from test_shape_robustness import blocks,edit


def fitted(ids):
    raw=edit(blocks(),ids)
    raw.update(object_id='G1',reason='',uncertain=[])
    return fit_object(raw,FitFrame((1000,1000),(1000,1000),(1000,1000),(0,0,1000,1000)),
                      object_id='G1',namespace='test')


class IndependentAdditions(unittest.TestCase):
    def setUp(self):
        self.before=fitted(['panel']);self.after=fitted(['panel','cell7','cell8'])
        self.review={'decision':'revise','issues':[{'kind':'geometry','part_ids':['cell8'],
            'evidence':'The last added cell is truncated.','action':'Complete only this cell.'}]}

    def test_preserves_old_native_geometry_and_only_uncriticized_independent_additions(self):
        before,after=copy.deepcopy(self.before),copy.deepcopy(self.after)
        out,audit=select(before,after,self.review,{'allow_new_parts':True})
        self.assertEqual([p['id']for p in out['parts']],['panel','cell7'])
        self.assertEqual(out['parts'][0]['commands'],before['parts'][0]['commands'])
        self.assertEqual(out['fit_status'],'partial');self.assertFalse(audit['whole_object_accepted'])
        self.assertEqual(audit['held_additions'],['cell8'])
        self.assertEqual(before,self.before);self.assertEqual(after,self.after)

    def test_changed_old_geometry_cannot_be_promoted_by_a_local_new_part_issue(self):
        self.after['parts'][0]['commands'][0][1]+=.1
        self.assertIsNone(select(self.before,self.after,self.review,{'allow_new_parts':True}))

    def test_no_promotion_without_missing_geometry_authorization_or_with_an_old_part_issue(self):
        self.assertIsNone(select(self.before,self.after,self.review,{'allow_new_parts':False}))
        self.review['issues'][0]['part_ids']=['panel']
        self.assertIsNone(select(self.before,self.after,self.review,{'allow_new_parts':True}))

    def test_extras_and_point_dependencies_do_not_pass_as_safe_additions(self):
        self.review['issues'][0]['kind']='extra'
        self.assertIsNone(select(self.before,self.after,self.review,{'allow_new_parts':True}))
        self.review['issues'][0]['kind']='geometry'
        self.after['parts'][1]['observed_part']['point_ids'][0]='cell80'
        self.assertIsNone(select(self.before,self.after,self.review,{'allow_new_parts':True}))


class ControllerPartialCommit(unittest.TestCase):
    setUp=fixtures.PipelineTests.setUp
    run_object=fixtures.PipelineTests.run_object

    def test_budget_exhaustion_keeps_safe_addition_and_unresolved_state(self):
        self.options.update(max_revisions=1,source_frame_refinement=False)
        def respond(tag,prompt,count):
            if tag=='observe':
                return edit(blocks(),['panel']) if 'revision_task'not in prompt else blocks()
            if count==2:
                return {'decision':'revise','issues':[{'kind':'missing','parts':['F1'],
                    'evidence':'Two separate cells are absent.','action':'Add both missing cells; preserve the panel.'}]}
            return {'decision':'revise','issues':[{'kind':'geometry','parts':['F3'],
                'evidence':'The final cell is truncated.','action':'Complete only the final cell; preserve the others.'}]}
        result=self.run_object(FakeRuntime(self.config,respond))
        self.assertEqual(result['state'],'unresolved')
        self.assertEqual(result['stop_reason'],'revision_budget_exhausted')
        self.assertEqual(result['selected_revision'],'partial_additions')
        fit=read(self.folder/'objects/G1/partial_additions/fit_original.json')
        self.assertEqual([p['id']for p in fit['parts']],['panel','cell7'])
        original=read(self.folder/'objects/G1/revision_00/fit_original.json')
        self.assertEqual(fit['parts'][0]['commands'],original['parts'][0]['commands'])
        from shape_integration.fitting.seed import revision_seed
        seed=revision_seed(self.folder/'objects/G1')
        self.assertTrue(seed['fallback_fit_available'])
        self.assertFalse(seed['fit_reused'])
