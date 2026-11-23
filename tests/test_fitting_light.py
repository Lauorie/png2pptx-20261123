"""Compact observations must preserve geometry, ownership and legacy evidence."""
import copy
import unittest

import test_pipeline as fixtures
from test_pipeline import FakeRuntime, observed, reviewed
from shape_integration.common import read
from shape_integration.fitting.contracts import normalize_observation, ObservationError
from shape_integration.fitting.observation import make_prompt
from shape_integration.fitting.critic import review_prompt


class LightFittingTests(unittest.TestCase):
    setUp = fixtures.PipelineTests.setUp
    run_object = fixtures.PipelineTests.run_object

    def test_compact_compound_runs_through_fit_critic_and_raw_audit(self):
        compact = {k: v for k, v in observed('G1').items() if k in ('status', 'points', 'parts', 'ownership')}
        runtime = FakeRuntime(self.config, lambda tag, prompt, count:
                              {**compact,'ownership':{'unit':'other','unit_evidence':'Visible geometric cells','parts':[{'id':p['id'],'role':'body'}for p in compact['parts']]}} if tag == 'observe' else reviewed(prompt))
        result = self.run_object(runtime)
        self.assertEqual(result['state'], 'accepted_by_model')
        folder = self.folder/'objects/G1/revision_00'
        self.assertEqual(read(folder/'observation_raw.json'), {**compact,'ownership':{'unit':'other','unit_evidence':'Visible geometric cells','parts':[{'id':p['id'],'role':'body'}for p in compact['parts']]}})
        normalized = read(folder/'observation.json')
        self.assertEqual(normalized['object_id'], 'G1')
        self.assertEqual(normalized['parts'], compact['parts'])
        self.assertEqual(normalized['points'], compact['points'])
        audit = read(folder/'observation_normalization.json')
        self.assertEqual(audit['backend_added'], {'object_id': 'G1', 'reason': '', 'uncertain': []})
        self.assertFalse(audit['geometry_changed'])

    def test_legacy_notes_metadata_and_extra_decimals_are_preserved(self):
        raw = observed('G1'); raw['version'] = 'model-added-v1'
        raw['points'][0]['xy'][0] = 100.0123456789
        raw['uncertain'] = ['hidden lower edge remains ambiguous']
        before = copy.deepcopy(raw); audit = {}
        normalized = normalize_observation(raw, 'G1', (1080, 1080), audit=audit)
        self.assertEqual(raw, before)
        self.assertEqual(normalized, {k: v for k, v in raw.items() if k != 'version'})
        self.assertEqual(audit['removed_metadata'], {'version': 'model-added-v1'})
        self.assertEqual(audit['backend_added'], {})

    def test_unknown_fields_wrong_identity_and_incomplete_without_reason_still_fail(self):
        for raw in [dict(observed('G1'), note='extra'), observed('wrong'),
                    dict(observed('G1', 'partial'), reason=''),
                    dict(observed('G1'), points={'points': observed('G1')['points']})]:
            with self.subTest(raw=raw), self.assertRaises(ObservationError):
                normalize_observation(raw, 'G1', (1080, 1080))
        raw = observed('G1'); raw['parts'][0]['point_ids'][0] = 'missing'
        with self.assertRaisesRegex(ObservationError, 'Missing point'):
            normalize_observation(raw, 'G1', (1080, 1080))

    def test_prompt_deduplicates_identity_without_changing_neighbors_or_feedback(self):
        inventory = [{'id': 'G2', 'role': 'separate_shape_candidate',
                      'relation_hint': 'nearby', 'label': '', 'kind': 'ellipse'}]
        job = {'object_id': 'G1', 'label': 'cells', 'proposed_type': 'compound_symbol',
               'frame': {'actual_crop_input_size_px': [1080, 900]},
               'input_context': {'identity_descriptions': ['two visible cells'],
                                 'nearby_inventory': inventory, 'ownership_note': 'redundant'},
               'revision_feedback': []}
        before = copy.deepcopy(job)
        initial = make_prompt(job, ['two visible cells'])
        self.assertEqual(job, before)
        self.assertEqual(initial['target']['identity_descriptions'], ['two visible cells'])
        self.assertEqual(initial['localization_context'], {'nearby_inventory': inventory})
        for key in ('version', 'revision_feedback', 'revision_policy'):
            self.assertNotIn(key, initial)
        self.assertEqual(initial['output_schema']['required'], ['status', 'points', 'parts', 'ownership'])
        self.assertNotIn('object_id', initial['output_schema']['properties'])
        job['revision_feedback'] = ['Move the left edge toward the visible outline.']
        refined = make_prompt(job)
        self.assertEqual(refined['revision_feedback'], job['revision_feedback'])
        self.assertIn('ALL target parts', refined['revision_policy'])
        self.assertEqual(initial['rules'], refined['rules'])
        self.assertEqual(initial['sampling'], refined['sampling'])

    def test_critic_receives_the_same_registered_peer_ownership_as_observer(self):
        self.run_object(FakeRuntime(self.config))
        folder = self.folder/'objects/G1/revision_00'
        job, local = read(folder/'job.json'), read(folder/'fit_local.json')
        peers = [{'id': 'peer_circle', 'role': 'separate_shape_candidate',
                  'relation_hint': 'inside_target', 'label': 'filled circle', 'kind': 'ellipse'}]
        job['input_context'] = {'nearby_inventory': peers}
        original = copy.deepcopy(job)
        observer = make_prompt(job)
        critic = review_prompt(job, local)
        self.assertEqual(job, original)
        self.assertEqual(observer['localization_context']['nearby_inventory'], peers)
        self.assertEqual(critic['localization_context']['nearby_inventory'], peers)
        self.assertIn('Do not ask this target to add those peer objects', critic['localization_context']['rule'])


if __name__ == '__main__':
    unittest.main()
