"""Saved geometry is re-reviewed, corrections reach the observer, siblings overlap."""
import copy
import threading
import unittest
from unittest.mock import patch
from PIL import Image

import test_pipeline as fixtures
from test_pipeline import FakeRuntime, observed, reviewed
from shape_integration.common import read, save, sha
from shape_integration.fitting.control import process_object
from shape_integration.fitting.seed import revision_seed
from shape_integration.recheck import run


class RecheckTests(unittest.TestCase):
    setUp = fixtures.PipelineTests.setUp
    run_object = fixtures.PipelineTests.run_object

    def test_seed_skips_observation_but_refines_new_image_supported_defect(self):
        self.run_object(FakeRuntime(self.config))
        source = self.folder/'objects/G1'
        seed = revision_seed(source); before = sha(source/'record.json')
        def responder(tag, prompt, count):
            if tag == 'observe':
                self.assertIn('move the left edge', ' '.join(prompt['revision_feedback']))
                return observed('G1')
            return reviewed(prompt, 'revise' if count == 1 else 'accept')
        runtime = FakeRuntime(self.config, responder)
        target = self.work/'rechecked/G1'
        result = process_object(target, self.candidate, self.item, self.work, runtime, self.options,
                                namespace='rechecked', seed=seed)
        self.assertEqual([tag for tag, _ in runtime.events], ['check', 'observe', 'check'])
        self.assertEqual(result['state'], 'accepted_by_model')
        self.assertEqual(sha(source/'record.json'), before)
        self.assertEqual(sha(source/'revision_00/fit_local.json'), sha(target/'revision_00/fit_local.json'))
        self.assertFalse((target/'revision_00/calls/observe_00').exists())
        self.assertEqual(read(target/'revision_00/inspection.json')['version'], 'shape_integration.critic_visual.1')
        prompt = runtime.events[0][1]
        self.assertEqual(set(prompt['output_schema']['properties']), {'decision', 'issues'})
        self.assertNotIn('observation_state', prompt)
        self.assertNotIn('separately_detected', prompt)
        self.assertTrue(all(set(p) == {'id', 'color', 'kind','closed'} for p in prompt['fitted_parts']))

    def test_neighbor_names_cannot_enter_critic_or_hide_intrinsic_frame(self):
        from shape_integration.fitting.critic import review_prompt
        self.run_object(FakeRuntime(self.config))
        folder = self.folder/'objects/G1/revision_00'
        job = read(folder/'job.json'); local = read(folder/'fit_local.json')
        job['input_context']['nearby_inventory'] = [{'id': 'G_OTHER', 'role': 'separate_shape_candidate',
            'label': 'misleading neighbor', 'relation_hint': 'contains_target'},
            {'id': 'I_OTHER', 'role': 'image_asset', 'label': 'photo asset', 'relation_hint': 'nearby'}]
        prompt = review_prompt(job, local)
        self.assertNotIn('misleading neighbor', str(prompt))
        self.assertNotIn('G_OTHER', str(prompt))
        self.assertNotIn('I_OTHER', str(prompt))
        self.assertNotIn('excluded_image_assets', prompt)
        self.assertNotIn('photo asset', str(prompt))
        self.assertIn('固有外框', ' '.join(prompt['checks']))

    def test_asset_boxes_map_to_crop_pixels_and_only_enter_first_image(self):
        from shape_integration.fitting.assets import ASSET_COLOR
        import numpy as np
        asset = {'id': 'I8', 'label': 'asset name stays out of prompt', 'bbox_2d': [80, 60, 95, 80]}
        runtime = FakeRuntime(self.config)
        root = self.work/'assets/G1'
        process_object(root, self.candidate, self.item, self.work, runtime, self.options,
                       namespace='assets', image_assets=[asset])
        folder = root/'revision_00'; meta = read(folder/'frame.json')
        annotation = read(folder/'critic_assets.json'); shown = annotation['assets'][0]
        crop = meta['source_crop_half_open']; width, height = meta['actual_crop_input_size_px']
        expected = [(value+.5-crop[i % 2]) * [width, height][i % 2]
                    / (crop[i % 2+2]-crop[i % 2])-.5 for i, value in enumerate(asset['bbox_2d'])]
        np.testing.assert_allclose(shown['bbox_local'], expected, atol=1e-9)
        self.assertEqual(shown['label'], 'IMAGE I8')
        self.assertEqual(sha(folder/'clean.png'), annotation['source_sha256'])
        self.assertNotEqual(sha(folder/'critic_source.png'), sha(folder/'clean.png'))
        with Image.open(folder/'critic_source.png') as first, Image.open(folder/'fit_geometry.png') as second:
            self.assertEqual(first.size, second.size)
            cyan = tuple(int(ASSET_COLOR[i:i+2], 16) for i in (1, 3, 5))
            self.assertTrue(np.any(np.all(np.asarray(first) == cyan, axis=2)))
            self.assertFalse(np.any(np.all(np.asarray(second) == cyan, axis=2)))
        prompt = [p for tag, p in runtime.events if tag == 'check'][0]
        self.assertNotIn('I8', str(prompt))
        self.assertNotIn(asset['label'], str(prompt))
        request = read(folder/'calls/check_00/request.json')
        self.assertEqual(request['images'][0], str(folder/'critic_source.png'))
        self.assertEqual(request['images'][1], str(folder/'fit_geometry.png'))

    def test_changed_seed_is_rejected_before_model_call(self):
        self.run_object(FakeRuntime(self.config))
        source = self.folder/'objects/G1'; seed = revision_seed(source)
        path = source/'revision_00/fit_local.json'; path.write_text(path.read_text()+' ')
        runtime = FakeRuntime(self.config)
        with self.assertRaisesRegex(ValueError, 'Seed changed'):
            process_object(self.work/'new/G1', self.candidate, self.item, self.work, runtime,
                           self.options, namespace='test', seed=seed)
        self.assertEqual(runtime.events, [])

    def test_failed_historical_seed_uses_current_prompt_only_in_new_destination(self):
        self.run_object(FakeRuntime(self.config, lambda *args: {'invalid': True}))
        source = self.folder/'objects/G1'; revision = source/'revision_00'
        old = read(revision/'prompt.json'); old['prompt']['version'] = 'old-wire-protocol'
        save(revision/'prompt.json', old)
        job = read(revision/'job.json'); job['input_sha256']['prompt.json'] = sha(revision/'prompt.json')
        job.pop('observation_protocol', None)
        job['revision_feedback']=['WRONG HISTORICAL COLOUR CLAIM']
        save(revision/'job.json', job)
        seed = revision_seed(source); self.assertFalse(seed['fit_reused'])
        hashes = {p.name: sha(p) for p in revision.iterdir() if p.is_file()}
        target = self.work/'rechecked/G1'; runtime = FakeRuntime(self.config)
        result = process_object(target, self.candidate, self.item, self.work, runtime,
                                self.options, namespace='rechecked', seed=seed)
        self.assertEqual(result['state'], 'accepted_by_model')
        self.assertEqual([tag for tag, _ in runtime.events], ['observe', 'check'])
        self.assertNotIn('version', runtime.events[0][1])
        self.assertNotIn('WRONG HISTORICAL COLOUR CLAIM', str(runtime.events[0][1]))
        self.assertEqual(read(target/'revision_00/source_prompt.json'), old)
        self.assertEqual(hashes, {p.name: sha(p) for p in revision.iterdir() if p.is_file()})

    def test_numerically_successful_but_unresolved_seed_gets_fresh_observation(self):
        self.options['max_revisions'] = 0
        runtime = FakeRuntime(self.config, lambda tag, prompt, count:
            observed('G1') if tag == 'observe' else reviewed(prompt, 'revise'))
        result = self.run_object(runtime)
        source = self.folder/'objects/G1'
        self.assertEqual(result['state'], 'unresolved')
        self.assertEqual(read(source/'revision_00/status.json')['status'], 'finished')
        before = sha(source/'record.json'); seed = revision_seed(source)
        self.assertFalse(seed['fit_reused'])
        fresh = FakeRuntime(self.config)
        result = process_object(self.work/'new/G1', self.candidate, self.item, self.work,
                                fresh, self.options, namespace='new', seed=seed)
        self.assertEqual([tag for tag, _ in fresh.events], ['observe', 'check'])
        self.assertEqual(result['state'], 'accepted_by_model')
        self.assertEqual(sha(source/'record.json'), before)

    def test_failed_fresh_observation_keeps_previous_geometry_without_locking_observer(self):
        self.options['max_revisions'] = 0
        self.run_object(FakeRuntime(self.config, lambda tag, prompt, count:
            observed('G1') if tag == 'observe' else reviewed(prompt, 'revise')))
        source = self.folder/'objects/G1'; before = sha(source/'record.json')
        seed = revision_seed(source); self.assertTrue(seed['fallback_fit_available'])
        runtime = FakeRuntime(self.config, lambda tag, prompt, count: observed('G1', 'unsupported'))
        target = self.work/'retry/G1'
        result = process_object(target, self.candidate, self.item, self.work, runtime,
                                self.options, namespace='retry', seed=seed)
        self.assertEqual([tag for tag, _ in runtime.events], ['observe'])
        self.assertEqual(result['state'], 'unresolved')
        self.assertEqual(result['selected_revision'], 'revision_baseline')
        self.assertEqual(sha(target/'revision_baseline/fit_original.json'), sha(source/'revision_00/fit_original.json'))
        self.assertEqual(sha(source/'record.json'), before)
        self.assertTrue((target/'revision_00/calls/observe_00/raw.json').is_file())

    def test_full_recheck_runs_three_objects_in_one_image_concurrently(self):
        source = self.work/'prior'; case = source/'cases/case'; candidates = []
        for i in range(3):
            candidate = {**self.candidate, 'id': f'G{i+1}'}; candidates.append(candidate)
            runtime = FakeRuntime(self.config, lambda tag, prompt, count:
                observed(prompt['target']['id']) if tag == 'observe' else reviewed(prompt))
            process_object(case/'fitting/objects'/candidate['id'], candidate, self.item, self.work,
                           runtime, self.options, namespace='prior')
        save(source/'manifest.json', {'images': [self.item]})
        save(case/'geometry.json', {'image_size': self.item['source_size'],
                                   'coordinate_system': 'image_pixel_centers', 'objects': candidates})
        save(case/'summary.json', {'image_coverage_status': 'complete'})
        save(case/'dedup/image_boxes_original.json', {'assets': []})
        save(case/'fitting/candidates.json', {'objects': candidates})
        barrier = threading.Barrier(3)
        def overlap(*args, **kwargs):
            barrier.wait(timeout=10)
            return process_object(*args, **kwargs)
        config = copy.deepcopy(self.config); config['fitting']['workers'] = 3
        with patch('shape_integration.recheck.process_object', side_effect=overlap), patch('shape_integration.recheck.batch.freeze'):
            result = run(self.work, self.work/'new', config, [self.item], FakeRuntime(config), source=source)
        self.assertEqual(result['totals']['accepted_by_model'], 3)
        self.assertEqual(read(self.work/'new/progress.json')['peak_objects_by_image']['case'], 3)


if __name__ == '__main__':
    unittest.main()
