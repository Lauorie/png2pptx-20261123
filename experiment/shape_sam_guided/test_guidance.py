"""Check transport, identity and optional-stage behavior; no model inference."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from core import read, save
from shape_integration.sam_guidance import GuidedSamEvidence
from shape_integration.fitting.local import prepare_candidate
from shape_integration.fitting.control import process_object
from shape_integration.fitting.frames import FitFrame


class GuidanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = Image.new('RGB', (200, 100), 'white')
        mask = np.zeros((100, 200), np.uint8)
        mask[20:81, 60:141] = 255
        path = self.root / 'mask.png'
        Image.fromarray(mask).save(path)
        self.group = {'id': 'S1', 'representative': {'bbox': [60, 20, 140, 80], 'score': .95,
            'category': 'rectangle', 'mask_path': str(path), 'mask_quality_ok': True,
            'frame': {'box': [0, 0, 200, 100], 'size': [200, 100]}}}
        self.candidate = {'id': 'O', 'label': 'target', 'kind': 'rectangle', 'bbox_2d': [25, 20, 100, 80]}

    def tearDown(self):
        self.temp.cleanup()

    def test_repair_activation_leaves_complete_initial_inputs_unchanged(self):
        evidence = GuidedSamEvidence([self.group], activation='repair')
        candidate = {**self.candidate, 'bbox_2d': [60, 20, 140, 80]}
        folder = self.root/'repair'
        job = prepare_candidate(self.source, candidate, [200, 100], folder, spatial_evidence=evidence)
        prompt, images = {'task': 'original'}, [('source', folder/'clean.png')]
        self.assertFalse(job.get('sam_support'))
        self.assertEqual(evidence.region(prompt, images, {}, {}, folder), (prompt, images))
        self.assertEqual(evidence.observe(prompt, images, job, folder), (prompt, images))
        job['revision_feedback'] = ['The source supports a different curvature.']
        actual, _ = evidence.observe(prompt, images, job, folder)
        self.assertTrue(actual['sam_auxiliary_evidence']['measurement']['mask_available'])

    def test_repair_activation_supplies_evidence_for_actual_crop_extension(self):
        evidence = GuidedSamEvidence([self.group], activation='repair')
        folder = self.root/'clipped'
        job = prepare_candidate(self.source, self.candidate, [200, 100], folder,
                                padding=0, spatial_evidence=evidence)
        self.assertGreater(job['frame']['source_crop_half_open'][2], 140)
        actual, _ = evidence.observe({}, [('source', folder/'clean.png')], job, folder)
        self.assertIn('sam_auxiliary_evidence', actual)

    def test_no_empty_sam_prompt_when_no_new_extent_or_reliable_mask(self):
        self.group['representative']['mask_quality_ok'] = False
        evidence = GuidedSamEvidence([self.group], activation='repair', observer_numeric_bbox=False)
        candidate = {**self.candidate, 'bbox_2d': [60, 20, 140, 80]}
        folder = self.root/'no_boundary'
        job = prepare_candidate(self.source, candidate, [200, 100], folder, spatial_evidence=evidence,
                                feedback=['Inspect the corner curvature.'])
        prompt, images = {'task': 'original'}, [('source', folder/'clean.png')]
        self.assertEqual(evidence.observe(prompt, images, job, folder), (prompt, images))

    def test_shifted_extent_expands_crop_without_rewriting_identity(self):
        evidence = GuidedSamEvidence([self.group], mode='bbox')
        original = copy.deepcopy(self.candidate)
        job = prepare_candidate(self.source, self.candidate, [200, 100], self.root / 'job',
                                padding=0, spatial_evidence=evidence)
        self.assertEqual(self.candidate, original)
        crop = job['frame']['source_crop_half_open']
        self.assertLessEqual(crop[0], 25)
        self.assertGreater(crop[2], 140)
        self.assertEqual(job['sam_support']['bbox_original'], [60, 20, 140, 80])
        prompt, images = evidence.observe({}, [('clean', self.root / 'job/clean.png')], job, self.root / 'job')
        self.assertEqual(len(images), 1)
        self.assertFalse(prompt['sam_auxiliary_evidence']['measurement']['mask_available'])
        self.assertEqual(job['candidate'], original)

    def test_peer_and_low_confidence_do_not_claim_target(self):
        e = GuidedSamEvidence([self.group])
        e.set_inventory([self.candidate, {'id': 'neighbor', 'bbox_2d': [60, 20, 140, 80]}])
        self.assertIsNone(e.support(self.candidate))
        low = copy.deepcopy(self.group); low['representative']['score'] = .4
        self.assertIsNone(GuidedSamEvidence([low]).support(self.candidate))

    def test_constituent_mask_cannot_shrink_a_compound_extent(self):
        candidate = {'id': 'O', 'kind': 'compound_symbol', 'bbox_2d': [60, 0, 140, 99]}
        piece = copy.deepcopy(self.group)
        piece['representative']['bbox'] = [60, 20, 140, 45]
        evidence = GuidedSamEvidence([piece])
        self.assertIsNone(evidence.support(candidate))
        crop = [45, 0, 155, 100]
        self.assertEqual(evidence.prepare(candidate, crop, [200, 100]), (crop, None))

    def test_visual_extent_keeps_crop_but_withholds_outer_bbox_coordinates(self):
        evidence = GuidedSamEvidence([self.group], mode='bbox', observer_numeric_bbox=False)
        folder = self.root / 'job'
        job = prepare_candidate(self.source, self.candidate, [200, 100], folder, spatial_evidence=evidence)
        prompt, _ = evidence.observe({}, [('source', folder/'clean.png')], job, folder)
        info = prompt['sam_auxiliary_evidence']['measurement']
        self.assertNotIn('bbox_local', info)
        self.assertNotIn('size_local', info)
        self.assertGreater(job['frame']['source_crop_half_open'][2], 140)

    def test_strict_mask_agreement_preserves_loose_bbox_localization(self):
        evidence = GuidedSamEvidence([self.group], mask_iou=.65)
        self.assertIsNotNone(evidence.support(self.candidate))
        self.assertFalse(evidence.mask_usable(self.group, self.candidate))
        aligned = {**self.candidate, 'bbox_2d': [60, 20, 140, 80]}
        self.assertTrue(evidence.mask_usable(self.group, aligned))

    def test_fragmented_mask_supplies_visual_strokes_without_false_contour_samples(self):
        self.source.paste('#222222', (0, 0, *self.source.size))
        array = np.zeros((100, 200), np.uint8)
        for x in range(60, 141, 15):
            array[20:23, x:x+7] = 255
            array[78:81, x:x+7] = 255
        for y in range(20, 81, 15):
            array[y:y+7, 60:63] = 255
            array[y:y+7, 138:141] = 255
        path = self.root/'fragments.png'
        Image.fromarray(array).save(path)
        group = copy.deepcopy(self.group)
        group['representative']['mask_path'] = str(path)
        evidence = GuidedSamEvidence([group])
        self.assertIsNotNone(evidence.support(self.candidate))
        self.assertTrue(evidence.mask_usable(group))
        folder = self.root/'job'
        job = prepare_candidate(self.source, self.candidate, [200, 100], folder, spatial_evidence=evidence)
        prompt, views = evidence.observe({}, [('source', folder/'clean.png')], job, folder)
        measurement = prompt['sam_auxiliary_evidence']['measurement']
        self.assertTrue(measurement['mask_available'])
        self.assertEqual(measurement['mask_continuity'], 'fragmented')
        self.assertEqual(measurement['mask_view'], 'isolated_foreground_fragments')
        self.assertNotIn('mask_boundary_samples_local', measurement)
        self.assertEqual(len(views), 2)
        with Image.open(views[-1][1]) as view:
            self.assertEqual(view.convert('RGB').getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(evidence.boundary_samples(group, job['frame']['source_crop_half_open'],
                                                  job['frame']['actual_crop_input_size_px']), [])
        # Both stages use the same disconnected evidence; norm1000 must not
        # fabricate samples or try to transform a nonexistent sample list.
        evidence.mode = 'critic'
        job['coordinate_mode'] = 'norm1000'
        prompt, views = evidence.critic({}, [('source', folder/'clean.png')], job, folder)
        measurement = prompt['sam_auxiliary_evidence']['measurement']
        self.assertEqual(measurement['mask_continuity'], 'fragmented')
        self.assertNotIn('mask_boundary_samples_local', measurement)
        self.assertEqual(measurement['coordinate_system'], 'norm1000')
        self.assertEqual(len(views), 2)

    def test_empty_or_failed_quality_mask_still_cannot_supply_evidence(self):
        path = self.root/'empty.png'
        Image.new('L', (200, 100), 0).save(path)
        group = copy.deepcopy(self.group)
        group['representative']['mask_path'] = str(path)
        self.assertFalse(GuidedSamEvidence([group]).mask_usable(group))
        group['representative']['mask_path'] = self.group['representative']['mask_path']
        group['representative']['mask_quality_ok'] = False
        self.assertFalse(GuidedSamEvidence([group]).mask_usable(group))

    def test_mask_coordinates_and_critic_opt_in(self):
        evidence = GuidedSamEvidence([self.group], mode='mask')
        folder = self.root / 'job'
        job = prepare_candidate(self.source, self.candidate, [200, 100], folder, spatial_evidence=evidence)
        original = {'task': 'all parts'}
        images = [('source', folder / 'clean.png')]
        prompt, views = evidence.observe(original, images, job, folder)
        self.assertEqual(original, {'task': 'all parts'})
        self.assertEqual(len(views), 2)
        measurement = prompt['sam_auxiliary_evidence']['measurement']
        self.assertTrue(measurement['mask_available'])
        self.assertNotIn('mask_boundary_samples_local', measurement)
        self.assertGreaterEqual(len(evidence.boundary_samples(self.group,
            job['frame']['source_crop_half_open'], job['frame']['actual_crop_input_size_px'])), 4)
        self.assertEqual(evidence.critic(original, images, job, folder), (original, images))
        critic = GuidedSamEvidence([self.group], mode='critic')
        revised, views = critic.critic(original, images, job, folder)
        self.assertIn('sam_auxiliary_evidence', revised)
        self.assertEqual(len(views), 2)

    def test_low_resolution_mask_keeps_bbox_guidance(self):
        group = copy.deepcopy(self.group)
        group['representative']['frame']['box'] = [0, 0, 4000, 2000]
        evidence = GuidedSamEvidence([group], mode='mask')
        folder = self.root / 'job'
        job = prepare_candidate(self.source, self.candidate, [200, 100], folder, spatial_evidence=evidence)
        prompt, images = evidence.observe({}, [('source', folder / 'clean.png')], job, folder)
        self.assertFalse(prompt['sam_auxiliary_evidence']['measurement']['mask_available'])
        self.assertGreater(job['frame']['source_crop_half_open'][2], 140)
        self.assertEqual(len(images), 1)

    def test_partial_boundary_requests_context_and_cannot_shrink_to_bad_plan(self):
        self.source.save(self.root / 'source.png')
        calls = []
        class Runtime:
            def for_stage(self, **unused):
                return self
            def call(inner, root, tag, system, prompt, images, validator, budget, **unused):
                calls.append(tag)
                if tag == 'plan':
                    raw = {'bbox_2d': [0, 0, 70, 80], 'reason': 'A deliberately too-small context plan.'}
                elif tag == 'check':
                    raw = {'decision': 'accept', 'issues': []}
                else:
                    job = read(Path(root).parent / 'job.json')
                    frame = FitFrame.from_metadata(job['frame'])
                    crop = job['frame']['source_crop_half_open']
                    complete = crop[2] > 140
                    points = [[60, 20], [140, 20], [140, 80], [60, 80]] if complete else [
                        [crop[2] - 1, 20], [60, 20], [60, 80], [crop[2] - 1, 80]]
                    xy = [frame.to_local(p) for p in points]
                    part = {'id': 'body', 'kind': 'rectangle', 'point_ids': ['p0', 'p1', 'p2', 'p3']} if complete else {
                        'id': 'body', 'kind': 'path', 'closed': False,
                        'segments': [{'kind': 'line', 'point_ids': [f'p{i}', f'p{i+1}']} for i in range(3)]}
                    raw = {'status': 'ok' if complete else 'partial', 'reason': '' if complete else 'Right side is outside the crop.',
                           'points': [{'id': f'p{i}', 'xy': p} for i, p in enumerate(xy)], 'parts': [part]}
                value = validator(raw)
                save(Path(root) / (tag + '_00') / 'result.json', raw)
                return value
        item = {'id': 'fixture', 'original': 'source.png', 'input': 'source.png',
                'source_size': [200, 100], 'input_size': [200, 100]}
        opts = {'crop_padding': .2, 'crop_scale': 3, 'crop_min_long_edge': 1080, 'crop_max_long_edge': 2048,
                'max_tokens': 1000, 'check_max_tokens': 1000, 'plan_max_tokens': 1000,
                'max_revisions': 1, 'max_context_rounds': 1, 'check_enabled': True,
                'recover_clipped_partial': True}
        result = process_object(self.root / 'object', self.candidate, item, self.root, Runtime(), opts, namespace='test')
        self.assertEqual(result['state'], 'accepted_by_model')
        self.assertEqual(calls, ['observe', 'plan', 'observe', 'check'])
        self.assertEqual(result['context_rounds'], 1)
        self.assertEqual(result['corrections'], 0)
        first = read(self.root / 'object/revision_00/job.json')['frame']['source_crop_half_open']
        second = read(self.root / 'object/revision_01/job.json')['frame']['source_crop_half_open']
        self.assertGreater(second[2], first[2])
        self.assertGreater(second[2], 140)


if __name__ == '__main__':
    unittest.main()
