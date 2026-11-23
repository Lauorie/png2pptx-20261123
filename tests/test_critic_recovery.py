"""Regression tests for retry budgets, ownership context and shared face geometry."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import test_pipeline as fixtures
from test_pipeline import FakeRuntime, observed, reviewed
from shape_integration.common import Runtime, read, save
from shape_integration.fitting.adapter import fit_object
from shape_integration.fitting.context import identity_context
from shape_integration.fitting.contracts import validate_observation
from shape_integration.fitting.critic import validate_review
from shape_integration.fitting.frames import FitFrame
from shape_integration.fitting.observation import sanitize_feedback


class RecoveryTests(unittest.TestCase):
    setUp = fixtures.PipelineTests.setUp
    run_object = fixtures.PipelineTests.run_object

    def test_omitted_effort_reaches_wire_and_cache_identity_with_thinking(self):
        config = {k: v for k, v in self.config.items() if k != 'reasoning_effort'}
        runtime = Runtime(config, 'http://offline.invalid/v1', '')
        stage = runtime.for_stage(shape_schema_retries=0)
        root = self.folder/'wire_calls'
        reply = {'ok': True}
        chunk = {'choices': [{'finish_reason': 'stop', 'delta': {'content': json.dumps(reply)}}]}
        async def lines():
            for line in ['data: '+json.dumps(chunk), 'data: [DONE]']:
                yield line
        with patch('shape_integration.transport.httpx.AsyncClient') as transport:
            session = transport.return_value.__aenter__.return_value
            session.stream = MagicMock()
            response = session.stream.return_value.__aenter__.return_value
            response.status_code = 200
            response.aiter_lines = MagicMock(return_value=lines())
            self.assertEqual(stage.call(root, 'observe', 'system', {'task': 'offline'}, [],
                                        lambda raw: raw, 1024), reply)
            wire = session.stream.call_args.kwargs['json']
        saved = read(root/'observe_00/wire_request.json')
        self.assertEqual(wire, saved)
        for parameters in [wire, read(root/'observe_00/request.json')['parameters'],
                           read(root/'observe_00/identity.json')['parameters']]:
            self.assertNotIn('reasoning_effort', parameters)
            self.assertEqual(parameters['thinking'], {'type': 'enabled'})
            self.assertEqual(parameters['max_tokens'], 1024)
        self.assertIs(stage.slots, runtime.slots)
        explicit = runtime.for_stage(reasoning_effort='low').client(root, 1024)
        self.assertEqual(explicit.request_parameters()['reasoning_effort'], 'low')

    def transport(self, responder):
        events = []
        class Client:
            def __init__(self, root): self.root = Path(root)
            def request_parameters(self): return {'offline': True}
            def redact(self, error): return str(error)
            def call(self, name, system, prompt, images, validator):
                events.append((name, copy.deepcopy(prompt)))
                folder = self.root/name
                save(folder/'request.json', {'prompt': prompt, 'images': [str(p) for _, p in images]})
                raw = responder(name, prompt, len(events))
                if name.startswith('observe') and isinstance(raw,dict) and isinstance(raw.get('parts'),list) and 'ownership' not in raw:
                    raw={**raw,'ownership':{'unit':'other','unit_evidence':'Visible geometric cells',
                        'parts':[{'id':part['id'],'role':'body'}for part in raw['parts']]}}
                if name.startswith('check') and isinstance(raw,dict) and 'decision' in raw and 'source_unit' in prompt.get('output_schema',{}).get('properties',{}):
                    raw={**raw,'source_unit':'other','source_unit_evidence':'Visible geometric cells'}
                save(folder/'raw.json', raw)
                save(folder/'parsed_response.json', raw)
                (folder/'content.txt').write_text(json.dumps(raw))
                try:
                    result = validator(raw)
                    save(folder/'result.json', result)
                    return result
                except ValueError as exc:
                    save(folder/'error_0.json', {'error': str(exc)})
                    raise
        runtime = Runtime(self.config, 'offline', 'offline')
        runtime.client = lambda root, budget: Client(root)
        return runtime, events

    def test_bad_point_table_retry_keeps_geometric_revision_budget(self):
        def responder(name, prompt, count):
            if count == 1:
                raw = observed('G1'); raw['points'] = {'p': [901, 902]}; return raw
            if name.startswith('observe'): return observed('G1')
            return reviewed(prompt, 'revise' if count == 3 else 'accept')
        runtime, events = self.transport(responder)
        result = self.run_object(runtime)
        self.assertEqual([n for n, _ in events], ['observe_00', 'observe_01', 'check_00', 'observe_00', 'check_00'])
        self.assertEqual(result['state'], 'accepted_by_model')
        self.assertEqual(result['corrections'], 1)
        self.assertIn('points: expected array', events[1][1]['validation_error'])
        self.assertNotIn('901', events[1][1]['validation_error'])
        fit = read(self.folder/'objects/G1/revision_00/fit_local.json')
        self.assertEqual(fit['provenance']['observation_call'], 'calls/observe_01')

    def test_bad_critic_retries_check_without_reobserving(self):
        def responder(name, prompt, count):
            if name.startswith('observe'): return observed('G1')
            return {'bad': True} if count == 2 else reviewed(prompt)
        runtime, events = self.transport(responder); result = self.run_object(runtime)
        self.assertEqual([n for n, _ in events], ['observe_00', 'check_00', 'check_01'])
        self.assertEqual(result['corrections'], 0)
        self.assertEqual(result['state'], 'accepted_by_model')

    def test_exhausted_protocol_does_not_spend_geometry_budget(self):
        runtime, events = self.transport(lambda *args: {'bad': True})
        result = self.run_object(runtime)
        self.assertEqual(result['stop_reason'], 'observation_protocol_exhausted')
        self.assertEqual(result['corrections'], 0)
        self.assertEqual(len(events), self.options['protocol_retries']+1)
        self.assertIsNone(result['selected_revision'])

    def test_three_refinements_and_four_observation_attempts_are_independent(self):
        self.options.update(max_revisions=3, protocol_retries=3)
        attempts = {'observe': 0}
        def responder(name, prompt, count):
            if name.startswith('observe'):
                attempts['observe'] += 1
                # Three initial format failures recover on the fourth attempt.
                return {'bad': True} if attempts['observe'] <= 3 else observed('G1')
            return reviewed(prompt, 'revise')
        runtime, events = self.transport(responder)
        result = self.run_object(runtime)
        self.assertEqual([name for name, _ in events[:4]], ['observe_00','observe_01','observe_02','observe_03'])
        self.assertEqual(result['corrections'], 3)
        self.assertEqual(result['stop_reason'], 'revision_budget_exhausted')
        self.assertEqual(sum(name.startswith('check') for name, _ in events), 4)
        self.assertEqual(attempts['observe'], 7)

    def test_critic_uncertainty_retains_fit_and_does_not_force_revision(self):
        def responder(tag, prompt, count):
            if tag == 'observe': return observed('G1')
            raw = reviewed(prompt, 'unresolved'); raw['issues'][0]['kind'] = 'ownership'
            raw['issues'][0]['action'] = 'Determine which target owns the right cell from the full image'; return raw
        result = self.run_object(FakeRuntime(self.config, responder))
        self.assertEqual(result['stop_reason'], 'critic_uncertain')
        self.assertEqual(result['selected_revision'], 'revision_00')
        self.assertEqual(result['corrections'], 0)

    def test_accept_cannot_contradict_evidence_or_use_unknown_parts(self):
        raw = reviewed({'fitted_parts': [{'id': 'F1'}, {'id': 'F2'}]})
        for change in ['defect', 'ambiguity', 'unknown_part']:
            changed = copy.deepcopy(raw)
            if change in ('defect', 'ambiguity'):
                changed['issues'] = [dict(kind='geometry' if change == 'defect' else 'ambiguity',
                                         parts=['F1'], evidence='Source outline is left of fitted edge', action='move edge left')]
            else:
                changed = reviewed({'fitted_parts': [{'id': 'fake'}]}, 'revise')
            with self.assertRaises(ValueError): validate_review(changed, ['F1', 'F2'])

    def test_identity_context_and_feedback_do_not_leak_coordinate_estimates(self):
        child = dict(id='G2', label='inner peer; bbox [901, 902, 903, 904]', kind='rectangle', bbox_2d=[80, 60, 90, 70])
        asset = dict(id='I1', target='photo; x=991 y=992', category='image', bbox_2d=[75, 80, 95, 100])
        ctx = identity_context(self.candidate, [self.candidate, child], [asset],
                               [{'id': 'G1', 'reason': 'Outlined cell group; x=999 y=888'}], .2)
        text = json.dumps(ctx)
        self.assertIn('inside_target', text); self.assertIn('image_asset', text)
        self.assertIn('Outlined cell group', text)
        for number in ['901', '991', '999', '888']: self.assertNotIn(number, text)
        feedback = sanitize_feedback('Colored pixels drift left; use x=991, y=992; move by 12 pixels. Preserve the shared seam.')
        self.assertIn('Colored pixels drift left', feedback)
        self.assertIn('Preserve the shared seam', feedback)
        self.assertNotIn('991', feedback); self.assertNotIn('12', feedback)

    def test_neighbor_ablation_preserves_assets_and_other_context(self):
        target = dict(id='T', label='target', kind='compound_symbol', bbox_2d=[100, 100, 200, 200])
        peers = [dict(id=oid, label=oid, kind='rectangle', bbox_2d=box) for oid, box in [
            ('P', [50, 50, 250, 250]), ('N', [205, 110, 215, 130]),
            ('I', [120, 120, 140, 140]), ('O', [190, 130, 210, 160])]]
        assets = [{**obj, 'id': 'A'+obj['id']} for obj in peers]
        baseline = identity_context(target, [target]+peers, assets, [], .2)
        for relation in ['contains_target', 'nearby']:
            actual = identity_context(target, [target]+peers, assets, [], .2,
                                      exclude_shape_relations=[relation])
            expected = copy.deepcopy(baseline)
            expected['nearby_inventory'] = [obj for obj in expected['nearby_inventory']
                if not (obj['role'] == 'separate_shape_candidate' and obj['relation_hint'] == relation)]
            self.assertEqual(actual, expected)
            self.assertEqual(len(actual['nearby_inventory']), len(baseline['nearby_inventory'])-1)
        self.assertEqual(identity_context(target, [target]+peers, assets, [], .2,
                                         exclude_shape_relations=[]), baseline)
        with self.assertRaises(ValueError):
            identity_context(target, peers, assets, [], .2, exclude_shape_relations=['typo'])

    def test_three_projected_faces_preserve_seven_shared_vertices(self):
        points = [[150, 50], [250, 80], [250, 180], [150, 220], [50, 180], [50, 80], [150, 115]]
        faces = [('top', [0, 1, 6, 5]), ('left', [5, 6, 3, 4]), ('right', [6, 1, 2, 3])]
        raw = {'object_id': 'cube', 'status': 'ok', 'reason': '', 'uncertain': [],
               'points': [{'id': str(i), 'xy': p} for i, p in enumerate(points)],
               'parts': [{'id': name, 'kind': 'path', 'closed': True,
                          'segments': [{'kind': 'line', 'point_ids': [str(a), str(b)]}
                                       for a, b in zip(ids, ids[1:]+ids[:1])]} for name, ids in faces]}
        frame = FitFrame((300, 300), (300, 300), (300, 300), (0, 0, 300, 300))
        fit = fit_object(raw, frame, object_id='cube', namespace='test/cube')
        self.assertEqual(len(fit['observation']['points']), 7)
        self.assertEqual(fit['part_counts']['succeeded'], 3)
        for part in fit['parts']:
            self.assertEqual(part['commands'][-1], ['Z'])
            self.assertEqual(part['commands'][0][1:], points[int(part['observed_point_ids'][0])])


if __name__ == '__main__': unittest.main(verbosity=2)
