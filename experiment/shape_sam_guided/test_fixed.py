"""Contracts that previously lost SAM proposals or skipped useful first-pass evidence."""
import copy
from pathlib import Path
import unittest

from core import read, save
import test_guidance
from shape_integration.sam_fixed import FixedSamEvidence
from shape_integration.fitting.local import prepare_candidate
from shape_integration.fitting.control import process_object
from shape_integration.fitting.frames import FitFrame
from shape_integration.fitting.coverage import review as coverage_review


class FixedTests(unittest.TestCase):
    setUp = test_guidance.GuidanceTests.setUp
    tearDown = test_guidance.GuidanceTests.tearDown

    def test_complete_crop_still_receives_bbox_on_first_observation(self):
        self.group['representative']['mask_quality_ok'] = False
        provider = FixedSamEvidence([self.group], observer_numeric_bbox=False)
        candidate = {**self.candidate, 'bbox_2d': [60, 20, 140, 80]}
        folder = self.root/'first'
        job = prepare_candidate(self.source, candidate, [200, 100], folder, spatial_evidence=provider)
        prompt, _ = provider.observe({}, [('source', folder/'clean.png')], job, folder)
        self.assertTrue(job['sam_support'])
        self.assertIn('sam_auxiliary_evidence', prompt)
        self.assertNotIn('bbox_local', prompt['sam_auxiliary_evidence']['measurement'])
        self.assertEqual(job['candidate'], candidate)

    def test_missing_or_unlinked_sam_proposal_cannot_silently_pass(self):
        provider = FixedSamEvidence([self.group])
        save(self.root/'sam_evidence.json', {'shown': [{'id': 'S1'}]})
        validator = provider.review_validator(lambda x: x, self.root, [self.candidate])
        original = {'new_objects': [{'id': 'N1'}], 'uncertain': [], 'more_pending': False}
        for decision in (None, [], [{'sam_id': 'S1', 'action': 'add', 'target': 2, 'reason': 'visible'}]):
            raw = {**original, 'sam_decisions': decision}
            with self.assertRaises(ValueError):
                validator(raw)
        raw = {**original, 'sam_decisions': [{'sam_id': 'S1', 'action': 'add', 'target': 0, 'reason': 'distinct enclosing frame'}]}
        self.assertEqual(validator(raw)['new_objects'], original['new_objects'])
        self.assertEqual(raw['sam_decisions'][0]['target'], 0)

    def test_uncertain_proposal_remains_visible_in_coverage_status(self):
        provider = FixedSamEvidence([self.group])
        save(self.root/'sam_evidence.json', {'shown': [{'id': 'S1'}]})
        checker = provider.review_validator(lambda x: x, self.root, [])
        result = checker({'status': 'complete', 'new_objects': [], 'uncertain': [],
                          'sam_decisions': [{'sam_id': 'S1', 'action': 'uncertain', 'target': None,
                                             'reason': 'Boundary leaves this crop'}]})
        self.assertEqual(result['status'], 'uncertain')
        self.assertTrue(result['uncertain'])

    def test_discovery_keeps_unrepresented_parent_and_omits_matched_and_margin_boxes(self):
        parent = copy.deepcopy(self.group)
        parent['id'] = 'parent'; parent['representative']['bbox'] = [20, 5, 180, 95]
        margin = copy.deepcopy(self.group)
        margin['id'] = 'margin'; margin['representative']['bbox'] = [182, 5, 199, 95]
        provider = FixedSamEvidence([self.group, parent, margin])
        existing = [{**self.candidate, 'bbox_2d': self.group['representative']['bbox']}]
        groups, count = provider._unrepresented([0, 0, 180, 100], existing)
        self.assertEqual([g['id'] for g in groups], ['parent'])
        self.assertEqual(count, 1)

    def test_covered_cannot_alias_a_disjoint_panel_or_small_child(self):
        provider = FixedSamEvidence([self.group])
        save(self.root/'sam_evidence.json', {'shown': [{'id': 'S1', 'bbox_original': [60, 20, 140, 80]}]})
        existing = [{'id': 'other_panel', 'bbox_2d': [145, 20, 190, 80]},
                    {'id': 'small_child', 'bbox_2d': [80, 30, 90, 40]},
                    {'id': 'same', 'bbox_2d': [60, 20, 140, 80]}]
        validator = provider.review_validator(lambda x: x, self.root, existing)
        for target in ('other_panel', 'small_child'):
            with self.assertRaises(ValueError):
                validator({'new_objects': [], 'uncertain': [], 'sam_decisions': [
                    {'sam_id': 'S1', 'action': 'covered', 'target': target, 'reason': 'claimed coverage'}]})
        self.assertEqual(validator({'new_objects': [], 'uncertain': [], 'sam_decisions': [
            {'sam_id': 'S1', 'action': 'covered', 'target': 'same', 'reason': 'same object'}]})['sam_decisions'][0]['target'], 'same')

    def test_high_overlap_child_does_not_hide_wider_parent_from_discovery(self):
        parent = copy.deepcopy(self.group)
        parent['representative']['bbox'] = [20, 5, 180, 95]
        provider = FixedSamEvidence([parent])
        child = {'id': 'child', 'bbox_2d': [20, 5, 108, 95]}
        groups, count = provider._unrepresented([0, 0, 200, 100], [child])
        self.assertEqual([g['id'] for g in groups], [parent['id']])
        self.assertEqual(count, 1)
        # Actual recovery of the full parent is sufficient, even when its
        # original localization remains the narrower identity box.
        child['fitted_bbox_2d'] = [20, 5, 180, 95]
        self.assertEqual(provider._unrepresented([0, 0, 200, 100], [child]), ([], 0))

    def test_coverage_uses_recovered_geometry_extent_without_rewriting_localization(self):
        provider = FixedSamEvidence([self.group])
        candidate = {'id': 'O', 'bbox_2d': [60, 60, 100, 80], 'fitted_bbox_2d': [60, 20, 140, 80]}
        original = copy.deepcopy(candidate)
        groups, _ = provider._unrepresented([0, 0, 200, 100], [candidate])
        self.assertEqual(groups, [])
        save(self.root/'sam_evidence.json', {'shown': [{'id': 'S1', 'bbox_original': [60, 20, 140, 80]}]})
        checker = provider.review_validator(lambda x: x, self.root, [candidate])
        result = checker({'new_objects': [], 'uncertain': [], 'sam_decisions': [
            {'sam_id': 'S1', 'action': 'covered', 'target': 'O', 'reason': 'Recovered complete geometry is present'}]})
        self.assertEqual(result['sam_decisions'][0]['target'], 'O')
        self.assertEqual(candidate, original)

    def test_fragments_can_reference_a_new_parent_in_pixel_and_normalized_coordinates(self):
        provider = FixedSamEvidence([self.group])
        for mode, box in [('pixel', [60, 20, 140, 80]),
                          ('norm1000', [60*1000/199, 20*1000/99, 140*1000/199, 80*1000/99])]:
            save(self.root/'sam_evidence.json', {'shown': [{'id': 'S1', 'bbox_original': [80, 30, 90, 60]}],
                'output_frame': {'source_crop_half_open': [0, 0, 200, 100],
                                 'actual_crop_input_size_px': [200, 100], 'coordinate_mode': mode}})
            checker = provider.review_validator(lambda x: x, self.root, [])
            for target in (0, 'N1'):
                result = checker({'new_objects': [{'id': 'N1', 'bbox_2d': box}], 'uncertain': [],
                    'sam_decisions': [{'sam_id': 'S1', 'action': 'covered', 'target': target,
                                       'reason': 'Fragment of the newly returned parent'}]})
                self.assertEqual(result['sam_decisions'][0]['target'], target)

    def test_global_coverage_can_register_a_sam_parent_of_existing_child(self):
        self.source.save(self.root/'source.png')
        self.source.save(self.root/'fit_overlay.png')
        child = {'id': 'child', 'label': 'inner element', 'kind': 'rectangle', 'bbox_2d': [80, 30, 95, 45]}
        provider = FixedSamEvidence([self.group])
        item = {'id': 'fixture', 'original': 'source.png', 'input': 'source.png',
                'source_size': [200, 100], 'input_size': [200, 100], 'source_sha256': 'fixture'}
        scene = {'objects': {'child': {'state': 'accepted_by_model', 'part_keys': []}}, 'parts': {}}
        class Runtime:
            spatial_evidence = provider
            config = {'model_conflict_review': False}
            def call(inner, root, tag, system, prompt, images, validator, budget):
                self.assertEqual(tag, 'coverage')
                self.assertIn('sam_candidate_accounting', prompt)
                raw = {'status': 'complete', 'new_objects': [
                    {'label': 'enclosing panel', 'kind': 'rectangle', 'bbox_2d': [60, 20, 140, 80],
                     'reason': 'Separate fill boundary surrounding an existing child'}],
                    'possible_duplicates': [], 'uncertain': [],
                    'sam_decisions': [{'sam_id': 'S1', 'action': 'add', 'target': 0, 'reason': 'Visible enclosing panel'}]}
                return validator(raw)
        result = coverage_review(scene, item, self.root, self.root/'coverage', Runtime(),
            {'max_new_candidates_per_round': 8, 'coverage_max_tokens': 1000}, [child], [], .85, 0)
        self.assertEqual(result['new_objects'][0]['id'], 'F1_N001')
        self.assertEqual(result['review']['sam_decisions'][0]['action'], 'add')

    def _object_run(self, sam_decision='revise', final_decision='revise', needs_context=False, mode='mask'):
        self.source.save(self.root/'source.png')
        provider = FixedSamEvidence([self.group], observer_numeric_bbox=False, mode=mode)
        calls = []; prompts = []
        class Runtime:
            spatial_evidence = provider
            def for_stage(inner, **kwargs):
                return inner
            def call(inner, root, tag, system, prompt, images, validator, budget, **kwargs):
                calls.append(tag); prompts.append((tag, copy.deepcopy(prompt)))
                if tag == 'plan':
                    raw = {'bbox_2d': [0, 0, 199, 99], 'reason': 'Complete source context'}
                elif tag == 'observe' and needs_context and calls.count('observe') == 1:
                    raw = {'status': 'needs_context', 'reason': 'Target continues beyond this crop',
                           'points': [], 'parts': []}
                elif tag == 'observe':
                    frame = FitFrame.from_metadata(read(Path(root).parent/'job.json')['frame'])
                    xy = [frame.to_local(p) for p in [[60, 20], [140, 20], [140, 80], [60, 80]]]
                    # Exercise a real revision here. Exact repeats deliberately
                    # stop before a redundant critic under the geometry guard.
                    if calls.count('observe') > 1:
                        xy = [frame.to_local([p[0]+1, p[1]]) for p in [[60, 20], [140, 20], [140, 80], [60, 80]]]
                    raw = {'status': 'ok', 'points': [{'id': f'p{i}', 'xy': p} for i,p in enumerate(xy)],
                           'parts': [{'id': 'body', 'kind': 'rectangle', 'point_ids': ['p0','p1','p2','p3']}]}
                else:
                    first = calls.count('check') <= 1
                    decision = sam_decision if tag == 'sam_check' else final_decision if first else 'accept'
                    raw = {'decision': decision, 'issues': [] if decision == 'accept' else [
                        {'kind': 'geometry', 'parts': ['F1'], 'evidence': 'Right side misses the visible source boundary',
                         'action': 'Place the right side on the visible centerline'}]}
                result = validator(raw); save(Path(root)/(tag+'_00')/'result.json', raw); return result
        item = {'id': 'fixture', 'original': 'source.png', 'input': 'source.png',
                'source_size': [200, 100], 'input_size': [200, 100]}
        opts = {'crop_padding': .2, 'crop_scale': 3, 'crop_min_long_edge': 1080, 'crop_max_long_edge': 2048,
                'max_tokens': 1000, 'check_max_tokens': 1000, 'plan_max_tokens': 1000,
                'max_revisions': 1, 'max_context_rounds': 1, 'check_enabled': True}
        result = process_object(self.root/'object', self.candidate, item, self.root, Runtime(), opts, namespace='fixed')
        return result, calls, prompts

    def test_coverage_can_recover_a_registered_target_with_no_successful_geometry(self):
        self.source.save(self.root/'source.png')
        self.source.save(self.root/'fit_overlay.png')
        target = {**self.candidate, 'bbox_2d': [60, 20, 140, 80]}
        provider = FixedSamEvidence([self.group])
        empty = {**target, 'fitted_bbox_2d': None}
        self.assertEqual(provider._unrepresented([0, 0, 200, 100], [empty])[1], 1)
        item = {'id': 'fixture', 'original': 'source.png', 'input': 'source.png',
                'source_size': [200, 100], 'input_size': [200, 100], 'source_sha256': 'fixture'}
        scene = {'objects': {target['id']: {'state': 'unresolved', 'part_keys': []}}, 'parts': {}}
        class Runtime:
            spatial_evidence = provider
            config = {'model_conflict_review': False}
            def call(inner, root, tag, system, prompt, images, validator, budget):
                self.assertIsNone(prompt['object_states'][0]['fitted_bbox_pixels'])
                self.assertFalse(prompt['object_states'][0]['has_fitted_geometry'])
                self.assertNotIn('state', prompt['object_states'][0])
                self.assertIn('empty-fit replacements', prompt['task'])
                return validator({'status': 'complete', 'new_objects': [
                    {'label': 'visible outer panel', 'kind': 'rectangle', 'bbox_2d': [60, 20, 140, 80],
                     'reason': 'The registered target has no rendered geometry'}],
                    'possible_duplicates': [], 'uncertain': [], 'sam_decisions': [
                        {'sam_id': 'S1', 'action': 'add', 'target': 0, 'reason': 'Source confirms the missing frame'}]})
        result = coverage_review(scene, item, self.root, self.root/'coverage_empty', Runtime(),
            {'max_new_candidates_per_round': 8, 'coverage_max_tokens': 1000}, [target], [], .85, 0)
        self.assertEqual(len(result['new_objects']), 1)

    def test_unresolved_critic_does_not_mean_geometry_is_absent_in_coverage(self):
        self.source.save(self.root/'source.png')
        target = {**self.candidate, 'bbox_2d': [60, 20, 140, 80],
                  'fitted_bbox_2d': [60, 20, 140, 80]}
        provider = FixedSamEvidence([self.group])
        prompt, _ = provider.coverage(
            {'task': '', 'object_states': [{'id': target['id'], 'state': 'unresolved'}]},
            [('source', self.root/'source.png')], {'source_size': [200, 100], 'input_size': [200, 100]},
            [target], self.root)
        self.assertTrue(prompt['object_states'][0]['has_fitted_geometry'])
        self.assertNotIn('state', prompt['object_states'][0])
        self.assertIsNotNone(prompt['object_states'][0]['fitted_bbox_pixels'])

    def test_sam_check_precedes_original_critic_once_and_shares_revision_budget(self):
        result, calls, prompts = self._object_run()
        self.assertEqual(calls, ['observe', 'sam_check', 'check', 'observe', 'check'])
        self.assertEqual(result['corrections'], 1)
        sam_prompt = next(p for t,p in prompts if t == 'sam_check')
        self.assertEqual(len(sam_prompt['extent_comparison']['fit_minus_sam_left_top_right_bottom']), 4)
        checks = [p for t,p in prompts if t == 'check']
        self.assertIn('preliminary_sam_findings', checks[0])
        self.assertNotIn('sam_auxiliary_evidence', checks[0])
        self.assertNotIn('preliminary_sam_findings', checks[1])
        self.assertFalse(read(self.root/'object/revision_00/inspection.json')['sample_points_shown'])
        self.assertEqual(checks[0]['version'], 'shape_integration.fit_check.9')

    def test_original_critic_can_dismiss_incorrect_sam_findings(self):
        result, calls, _ = self._object_run(final_decision='accept')
        self.assertEqual(calls, ['observe', 'sam_check', 'check'])
        self.assertEqual(result['state'], 'accepted_by_model')
        self.assertEqual(result['corrections'], 0)

    def test_context_recovery_does_not_skip_sam_check_before_first_critic(self):
        result, calls, _ = self._object_run(final_decision='accept', needs_context=True)
        self.assertEqual(calls, ['observe', 'plan', 'observe', 'sam_check', 'check'])
        self.assertEqual(result['context_rounds'], 1)
        self.assertEqual(result['corrections'], 0)

    def test_original_critic_retains_sam_measurement_after_precheck_and_revision(self):
        _, calls, prompts = self._object_run(mode='critic')
        self.assertEqual(calls, ['observe', 'sam_check', 'check', 'observe', 'check'])
        for tag, prompt in prompts:
            if tag in {'sam_check', 'check'}:
                self.assertIn('sam_auxiliary_evidence', prompt)
                self.assertIn('extent_comparison', prompt)
        preliminary = next(p for t,p in prompts if t == 'sam_check')
        self.assertEqual(preliminary['version'], 'shape_integration.sam_precheck.1')
        self.assertEqual(len(preliminary['checks']), 2)


if __name__ == '__main__':
    unittest.main()
