"""A whole-object fallback cannot erase independently confirmed coverage repairs."""
import unittest

import test_pipeline as fixtures
from test_pipeline import FakeRuntime
from test_shape_robustness import blocks,edit
from shape_integration.common import read
from shape_integration.fitting.control import process_object
from shape_integration.fitting.seed import revision_seed


class CoverageRecoveryRetention(unittest.TestCase):
    setUp=fixtures.PipelineTests.setUp
    run_object=fixtures.PipelineTests.run_object

    def test_failed_fit_reports_the_numerical_error_to_the_next_observation(self):
        self.options.update(max_revisions=1,source_frame_refinement=False)
        def responder(tag,prompt,count):
            if tag=='check':return fixtures.reviewed(prompt)
            if count==1:
                raw=fixtures.observed('G1')
                for point in raw['points']:point['xy']=[100,100]
                return raw
            feedback=' '.join(prompt['revision_feedback'])
            self.assertIn('Rectangle needs four distinct finite corners',feedback)
            return fixtures.observed('G1')
        result=self.run_object(FakeRuntime(self.config,responder))
        self.assertEqual(result['state'],'accepted_by_model')
        self.assertEqual(result['selected_revision'],'revision_01')

    def recover(self,relation):
        self.options.update(max_revisions=1,geometry_guard_enabled=True,source_frame_refinement=False)
        def original(tag,prompt,count):
            if tag=='observe':return edit(blocks(),['panel'])
            return {'decision':'unresolved','issues':[{'kind':'ambiguity','parts':['F1'],
                'evidence':'The source interior is uncertain.','action':'Inspect the source interior.'}]}
        self.run_object(FakeRuntime(self.config,original))
        seed=revision_seed(self.folder/'objects/G1')
        def repaired(tag,prompt,count):
            if tag=='observe':
                return edit(blocks(),['panel']) if 'revision_task' in prompt else blocks()
            return {'decision':'revise','issues':[{'kind':'geometry','parts':['F1'],
                'evidence':'A residual panel edge requires adjustment.','action':'Adjust only the panel edge.'}],
                'source_observations':[{'id':'coverage_request_0','source_relation':relation,
                    'evidence':'The requested independent cells are represented in the fit.'}]}
        root=self.work/'recovery'
        result=process_object(root,self.candidate,self.item,self.work,FakeRuntime(self.config,repaired),
            self.options,namespace='recovery',seed=seed,
            input_context={'coverage_source_requests':[{'kind':'missing','evidence':'Two cells are absent.',
                'action':'Recover the two source cells.'}]})
        return result,read(root/result['selected_revision']/'fit_local.json')

    def test_confirmed_recovered_parts_survive_repeated_residual_edit(self):
        result,fit=self.recover('supported')
        self.assertEqual(result['state'],'unresolved')
        self.assertEqual(result['stop_reason'],'geometry_stalled')
        self.assertEqual(result['selected_revision'],'revision_00')
        self.assertEqual(len(fit['parts']),3)
        self.assertEqual(result['coverage_recovery_retained'],'revision_00')

    def test_unconfirmed_recovery_keeps_immutable_previous_fit(self):
        result,fit=self.recover('missing')
        self.assertEqual(result['selected_revision'],'revision_baseline')
        self.assertEqual(len(fit['parts']),1)

