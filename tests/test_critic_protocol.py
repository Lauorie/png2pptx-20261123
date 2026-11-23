"""Critic regressions drawn from saved extra-field, ID, omission and ambiguity failures."""
import copy
import json
import unittest

import test_pipeline as fixtures
from test_pipeline import FakeRuntime, observed, reviewed
import test_critic_recovery as recovery_fixtures
from shape_integration.common import read
from shape_integration.fitting.critic import (check_object, part_inventory, review_prompt,
                                             review_schema, revision_feedback, validate_review)
from shape_integration.fitting.critic_retry import retry_builder


class CriticProtocolTests(unittest.TestCase):
    setUp = fixtures.PipelineTests.setUp
    run_object = fixtures.PipelineTests.run_object
    transport = recovery_fixtures.RecoveryTests.transport

    def test_critic_receives_the_original_occurrence_locator_separately_from_predictions(self):
        def responder(name,prompt,count):
            return observed('G1') if name.startswith('observe') else reviewed(prompt)
        runtime,events=self.transport(responder)
        self.run_object(runtime)
        folder=self.folder/'objects/G1/revision_00'
        request=read(folder/'calls/check_00/request.json')
        self.assertEqual(request['images'][-1],str(folder/'overlay.png'))
        self.assertEqual(request['prompt']['target_occurrence']['id'],'G1')
        self.assertTrue(any(path.endswith('fit_geometry.png') for path in request['images']))
        self.assertEqual(sum(name.startswith('check') for name,_ in events),1)

    def test_network_prompt_retains_the_source_connection_contract(self):
        from shape_integration.fitting.semantics import CONNECTION_EVIDENCE
        self.run_object(FakeRuntime(self.config))
        folder=self.folder/'objects/G1/revision_00'
        job=read(folder/'job.json');local=read(folder/'fit_local.json')
        template=local['parts'][0]
        local['parts']=[{**template,'id':name,'kind':'ellipse','bbox_2d':box}
            for name,box in [('left',[10,40,30,60]),('right',[90,40,110,60])]]
        local['parts'].append({**template,'id':'edge','kind':'path',
                              'commands':[['M',30,50],['L',90,50]]})
        prompt=review_prompt(job,local)
        self.assertIn('network_connections',prompt)
        self.assertIn(CONNECTION_EVIDENCE,prompt['checks'])

    def test_protocol_retry_cannot_turn_ownership_doubt_into_missing_lines(self):
        def responder(name,prompt,count):
            if name.startswith('observe'):return observed('G1')
            if count==2:
                raw=reviewed(prompt,'revise')
                raw['issues'].append({'kind':'ambiguity','parts':[],
                    'evidence':'Vertical strokes may belong to the enclosing diagram.',
                    'action':'Determine ownership; do not add unconfirmed side strokes.'})
                return raw
            raw=copy.deepcopy(prompt['previous_response']);raw['decision']='unresolved';return raw
        runtime,events=self.transport(responder);result=self.run_object(runtime)
        self.assertEqual(result['stop_reason'],'critic_uncertain')
        self.assertEqual(result['corrections'],0)
        self.assertEqual(events[-1][1]['version'],'shape_integration.critic_uncertainty_repair.1')

    def test_text_occlusion_contract_is_shared_by_observer_and_critic(self):
        from shape_integration.fitting.observation import make_prompt
        self.run_object(FakeRuntime(self.config))
        folder=self.folder/'objects/G1/revision_00';job=read(folder/'job.json');local=read(folder/'fit_local.json')
        for prompt in (make_prompt(job),review_prompt(job,local)):
            value=json.dumps(prompt,ensure_ascii=False)
            self.assertTrue('Text-background occlusion' in value or '不画白色遮罩' in value)
            self.assertTrue('COMPLETE underlying' in value or '按可见延续补齐' in value)

    def test_coverage_repair_goals_reach_actual_critic_prompt_and_schema(self):
        from shape_integration.fitting.coverage_evidence import repair_context
        from shape_integration.fitting.observation import make_prompt
        self.run_object(FakeRuntime(self.config))
        folder=self.folder/'objects/G1/revision_00'
        job=read(folder/'job.json');local=read(folder/'fit_local.json')
        job['input_context']=repair_context(job.get('input_context',{}),{'issues':[{
            'kind':'missing','evidence':'The outer grey members are absent.',
            'action':'Recover the outer members and preserve the inner frame.'}]})
        observer=make_prompt(job);critic=review_prompt(job,local)
        self.assertEqual(observer['coverage_requested_corrections'],critic['coverage_requested_corrections'])
        targets=critic['required_source_observations']['targets']
        self.assertEqual(targets[0]['id'],'coverage_request_0')
        self.assertIn('source_observations',critic['output_schema']['required'])

    def test_repeated_compound_cannot_pass_without_source_count_observation(self):
        from shape_integration.fitting.critic import source_observation_contract
        self.run_object(FakeRuntime(self.config))
        folder=self.folder/'objects/G1/revision_00'
        job=read(folder/'job.json');local=read(folder/'fit_local.json')
        job['proposed_type']='compound_symbol'
        local['parts']=[{**local['parts'][0],'id':str(i)} for i in range(4)]
        prompt=review_prompt(job,local)
        ids=source_observation_contract(prompt,{'F'+str(i+1):str(i) for i in range(4)})
        self.assertEqual(ids,['repeated_structure'])
        with self.assertRaises(ValueError):validate_review({'decision':'accept','issues':[]},['F1'],ids)
        bad={'decision':'accept','issues':[],'source_observations':[{
            'id':'repeated_structure','source_relation':'missing',
            'evidence':'CLEAN has four rows; the fitted panel contains three.'}]}
        with self.assertRaisesRegex(ValueError,'accept contradicts'):validate_review(bad,['F1'],ids)

        # A colored matrix can be localized only by its outer parallelogram.
        job['proposed_type']='parallelogram'
        self.assertIn('repeated_structure_check',review_prompt(job,local))
        # Three stacked planes plus open grid dividers are also repeated
        # structure; requiring four CLOSED planes skips this real case.
        job['proposed_type']='compound_symbol'
        local['parts'][-1]={**local['parts'][-1],'kind':'path',
                            'commands':[['M',0,0],['L',10,10]]}
        self.assertIn('repeated_structure_check',review_prompt(job,local))

    def test_minimal_accept_and_required_image_evidence(self):
        raw = {'decision': 'accept', 'issues': []}
        self.assertEqual(validate_review(raw, ['F1']), raw)
        self.assertEqual(revision_feedback(raw), [])
        raw = reviewed({'fitted_parts': [{'id': 'F1'}]}, 'revise')
        raw['issues'][0]['evidence'] = ' '
        with self.assertRaisesRegex(ValueError, 'evidence and action'):
            validate_review(raw, ['F1'])

    def test_unresolved_ownership_cannot_be_hidden_in_a_geometry_revision(self):
        raw = reviewed({'fitted_parts': [{'id': 'F1'}]}, 'revise')
        raw['issues'].append(dict(kind='ownership', parts=['F1'], evidence='two peers overlap and assignment is ambiguous', action='confirm ownership from full image'))
        with self.assertRaisesRegex(ValueError, 'resolve material ownership/context ambiguity first'):
            validate_review(raw, ['F1'])

    def test_missing_unfitted_frame_reaches_next_observation(self):
        def responder(tag, prompt, count):
            if tag == 'observe':
                return observed('G1')
            raw = reviewed(prompt)
            if count == 2:
                raw.update(decision='revise', issues=[dict(kind='missing', parts=[],
                    evidence='Gray rounded frame surrounding the orange list has no colored counterpart',
                    action='Add the complete gray rounded frame; preserve existing bars and dots')])
            return raw
        result = self.run_object(FakeRuntime(self.config, responder))
        self.assertEqual(result['state'], 'accepted_by_model')
        job = read(self.folder/'objects/G1/revision_01/job.json')
        feedback = ' '.join(job['revision_feedback'])
        self.assertIn('Add the complete gray rounded frame', feedback)
        self.assertIn('no existing fitted counterpart', feedback)
        self.assertEqual(result['corrections'], 1)

    def test_display_namespace_and_actual_shared_vertices(self):
        result = self.run_object(FakeRuntime(self.config))
        folder = self.folder/'objects/G1/revision_00'
        local = read(folder/'fit_local.json'); job = read(folder/'job.json')
        mapping, inventory, shared = part_inventory(local)
        self.assertEqual(mapping, {'F1': 'a', 'F2': 'b'})
        self.assertEqual(shared, [['F1', 'F2'], ['F1', 'F2']])
        self.assertEqual([p['id'] for p in inventory], ['F1', 'F2'])
        self.assertNotIn('observed_point_ids', json.dumps(review_prompt(job, local)))
        self.assertEqual(result['history'][0]['review']['part_id_map'], {'F1': 'a', 'F2': 'b'})
        self.assertTrue(all('fit_status' not in p for p in inventory))
        # A closed path may repeat its own endpoint; it does not share with itself.
        local['parts'] = [dict(id='path', kind='path', status='ok', observed_point_ids=['q', 'r', 'q'])]
        self.assertEqual(part_inventory(local)[2], [])

    def test_extra_fields_repair_saved_response_without_images_and_resume(self):
        def responder(name, prompt, count):
            if name.startswith('observe'):
                return observed('G1')
            if count == 2:
                raw = reviewed(prompt); raw['basis'] = ''; return raw
            self.assertEqual(prompt['version'], 'shape_integration.critic_format_repair.1')
            self.assertIn('unexpected fields: basis', prompt['validation_error'])
            raw = copy.deepcopy(prompt['previous_response'])
            raw.pop('basis')
            return raw
        runtime, events = self.transport(responder)
        result = self.run_object(runtime)
        self.assertEqual(result['state'], 'accepted_by_model')
        folder = self.folder/'objects/G1/revision_00'
        self.assertEqual(len(read(folder/'calls/check_00/request.json')['images']), 3)
        self.assertEqual(read(folder/'calls/check_01/request.json')['images'], [])
        self.assertEqual(read(folder/'calls/check_01/identity.json')['actual_images'], [])
        self.assertEqual(read(folder/'calls/check_00/parsed_response.json')['basis'], '')
        before = len(events)
        restored = check_object(read(folder/'job.json'), folder, runtime, self.options)
        self.assertEqual(restored['decision'], 'accept')
        self.assertEqual(len(events), before)

    def test_format_repair_cannot_clear_a_defect_or_change_decision(self):
        def responder(name, prompt, count):
            if name.startswith('observe'):
                return observed('G1')
            if count == 2:
                raw = reviewed(prompt, 'revise'); raw['basis'] = ''; return raw
            raw = copy.deepcopy(prompt['previous_response']); raw.pop('basis')
            raw.update(decision='accept', issues=[])
            return raw
        runtime, events = self.transport(responder)
        result = self.run_object(runtime)
        self.assertEqual(result['stop_reason'], 'check_protocol_exhausted')
        self.assertEqual(result['corrections'], 0)
        self.assertEqual(result['selected_revision'], 'revision_00')
        self.assertIn('format repair changed decision', result['error'])

    def test_extra_fields_with_decision_conflict_require_images(self):
        def responder(name, prompt, count):
            if name.startswith('observe'):
                return observed('G1')
            if count == 2:
                raw = reviewed(prompt, 'revise'); raw.update(decision='accept', note='check the left side'); return raw
            if count == 3:
                self.assertEqual(prompt['previous_response']['decision'], 'accept')
                self.assertTrue(prompt['previous_response']['issues'])
                return reviewed(prompt, 'revise')
            return reviewed(prompt)
        runtime, events = self.transport(responder)
        result = self.run_object(runtime)
        self.assertEqual(result['state'], 'accepted_by_model')
        folder = self.folder/'objects/G1/revision_00/calls/check_01'
        self.assertEqual(len(read(folder/'request.json')['images']), 3)

    def test_bad_part_reference_reports_value_and_keeps_images(self):
        def responder(name, prompt, count):
            if name.startswith('observe'):
                return observed('G1')
            raw = reviewed(prompt)
            if count == 2:
                raw = reviewed(prompt, 'revise'); raw['issues'][0]['parts'] = ['G017']
            if count == 3:
                self.assertIn("received 'G017'", prompt['validation_error'])
                self.assertIn('F1, F2', prompt['validation_error'])
            return raw
        runtime, events = self.transport(responder)
        self.assertEqual(self.run_object(runtime)['state'], 'accepted_by_model')
        self.assertEqual(len(read(self.folder/'objects/G1/revision_00/calls/check_01/request.json')['images']), 3)

    def test_multiple_json_is_preserved_for_review_not_silently_selected(self):
        folder = self.work/'ambiguous'; folder.mkdir()
        text = '{"decision":"accept"}\nCorrection:\n{"decision":"unresolved"}'
        (folder/'content.txt').write_text(text)
        prompt = {'output_schema': review_schema(['F1'])}
        images = [('source', self.source)]
        retry = retry_builder('system', prompt, images, lambda raw: validate_review(raw, ['F1']))(folder, 'Extra data')
        self.assertEqual(retry['images'], images)
        self.assertEqual(retry['prompt']['previous_response'], text)

    def test_context_action_reaches_recrop_request(self):
        def responder(tag, prompt, count):
            if tag == 'observe':
                return observed('G1')
            if tag == 'plan':
                self.assertIn('Include the hidden lower cell and adjacent label', prompt['context_request'])
                return dict(bbox_2d=[0, 0, 2047, 1023], reason='include lower context')
            raw = reviewed(prompt)
            if count == 2:
                raw.update(decision='needs_context', issues=[dict(kind='context', parts=['F1'],
                    evidence='The source lower cell and fitted end are clipped by the crop',
                    action='Include the hidden lower cell and adjacent label')])
            return raw
        result = self.run_object(FakeRuntime(self.config, responder))
        self.assertEqual(result['state'], 'accepted_by_model')
        self.assertEqual(result['context_rounds'], 1)


if __name__ == '__main__':
    unittest.main()
