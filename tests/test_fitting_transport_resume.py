"""Network failure is not a geometry decision and must not seal an omission."""
import unittest
import test_pipeline as fixtures
from test_pipeline import FakeRuntime,observed,reviewed


class ModelTransportError(RuntimeError):pass


class TransportResumeTests(unittest.TestCase):
    setUp=fixtures.PipelineTests.setUp
    run_object=fixtures.PipelineTests.run_object

    def test_service_failure_resumes_without_spending_a_geometry_revision(self):
        failed=True
        def respond(tag,prompt,count):
            nonlocal failed
            if tag=='observe' and failed:
                failed=False;raise ModelTransportError('Service unavailable before response')
            return observed('G1') if tag=='observe' else reviewed(prompt)
        runtime=FakeRuntime(self.config,respond)
        first=self.run_object(runtime)
        self.assertEqual(first['stop_reason'],'observation_transport_failed')
        self.assertIsNone(first['selected_revision']);self.assertEqual(first['corrections'],0)
        resumed=self.run_object(runtime)
        self.assertEqual(resumed['state'],'accepted_by_model')
        self.assertEqual(resumed['corrections'],0)

    def test_critic_service_failure_keeps_geometry_and_can_resume(self):
        failed=True
        def respond(tag,prompt,count):
            nonlocal failed
            if tag=='observe':return observed('G1')
            if failed:
                failed=False;raise ModelTransportError('Connection lost before review response')
            return reviewed(prompt)
        runtime=FakeRuntime(self.config,respond);first=self.run_object(runtime)
        self.assertEqual(first['stop_reason'],'check_transport_failed')
        self.assertEqual(first['selected_revision'],'revision_00')
        self.assertEqual(self.run_object(runtime)['state'],'accepted_by_model')
