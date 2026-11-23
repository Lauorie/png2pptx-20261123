"""Candidate gating, batched model decisions, safe failures and frozen-input replay."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'vendor')]
from shape_integration.common import read, save, sha
from shape_integration.dedup import filter_candidates
from shape_integration.image_conflicts import run_filter, validate_response
from shape_integration.batch import replay_case
from shape_integration.pipeline import source_manifest


class Model:
    def __init__(self, actions=None, fail=False):
        self.calls = []; self.actions = actions or {}; self.fail = fail

    def call(self, root, tag, system, prompt, images, validator, budget):
        self.calls.append(prompt)
        save(Path(root)/(tag+'_00')/'request.json', {'prompt': prompt})
        if self.fail:
            raise RuntimeError('offline simulated timeout')
        rows = []
        for candidate in prompt['candidates']:
            action = self.actions.get(candidate['shape_id'], 'delete')
            rows.append({'shape_id': candidate['shape_id'], 'decision': action,
                         'image_id': candidate['image_ids'][0] if action == 'delete' else None,
                         'reason': 'The two boxes identify the same photo.' if action == 'delete' else 'Independent geometric content.'})
        return validator({'decisions': rows})


class ConflictTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'tests'); self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name); self.source = self.work/'original.png'
        Image.new('RGB', (1000, 400), 'white').save(self.source)
        self.ref = {'path': str(self.source), 'sha256': sha(self.source), 'size': [1000, 400]}
        self.config = read(ROOT/'config.json')
        self.shapes = [{'id': name, 'bbox_2d': [x, 10, x+100, 110], 'label': 'NEVER_INJECT_LABEL'}
                       for name, x in [('A', 10), ('B', 210), ('C', 410), ('D', 610)]]
        self.assets = [{'id': 'I'+o['id'], 'bbox_2d': o['bbox_2d'], 'target': 'NEVER_INJECT_TARGET'} for o in self.shapes[:3]]
        # Exactly .4: this last shape must bypass model review.
        self.assets.append({'id': 'ID', 'bbox_2d': [610, 10, 650, 110]})

    def gate(self, model, **kwargs):
        return run_filter(self.shapes, self.assets, self.ref, self.work/'review', model, self.config, **kwargs)

    def test_only_selected_candidates_are_batched_and_outputs_control_removal(self):
        before = copy.deepcopy((self.shapes, self.assets))
        model = Model({'S_B': 'keep', 'S_C': 'defer'}); result = self.gate(model)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual([o['shape_id'] for o in model.calls[0]['candidates']], ['S_A', 'S_B', 'S_C'])
        self.assertEqual([o['id'] for o in result['removed']], ['A'])
        self.assertEqual([o['id'] for o in result['kept']], ['B', 'C', 'D'])
        self.assertEqual(result['deferred_ids'], ['C'])
        self.assertEqual((self.shapes, self.assets), before)
        self.assertEqual(result['model_request_calls'], 1)
        for token in ['NEVER_INJECT_LABEL', 'NEVER_INJECT_TARGET', 'bbox_2d', '0.4', 'S_D', 'I_ID']:
            self.assertNotIn(token, str(model.calls[0]))
        backend = read(self.work/'review/batch_000/backend.json')
        self.assertEqual(len(backend['rows']), 6)
        self.assertEqual(max(backend['display_size']), 2048)

    def test_large_candidate_set_splits_into_batches_not_per_object(self):
        self.config['image_conflict_review']['batch_size'] = 2
        model = Model(); result = self.gate(model)
        self.assertEqual([len(p['candidates']) for p in model.calls], [2, 1])
        self.assertEqual(result['reviewer_calls'], 2)
        self.assertEqual([o['id'] for o in result['kept']], ['D'])

    def test_no_candidates_means_no_model_calls_or_images(self):
        self.assets = self.assets[3:]; model = Model(); result = self.gate(model)
        self.assertEqual(model.calls, []); self.assertEqual(result['kept'], self.shapes)
        self.assertFalse((self.work/'review/clean.png').exists())

    def test_failure_and_prepare_only_keep_every_shape(self):
        model = Model(fail=True); result = self.gate(model)
        self.assertEqual(result['removed'], [])
        self.assertEqual(result['deferred_ids'], ['A', 'B', 'C'])
        self.assertEqual(result['review_failed_batches'], 1)
        result = run_filter(self.shapes, self.assets, self.ref, self.work/'prepared', None, self.config, prepare_only=True)
        self.assertEqual(result['removed'], []); self.assertTrue(result['review_pending'])
        self.assertEqual(result['model_request_calls'], 0)

    def test_unknown_image_duplicate_id_or_missing_decision_cannot_remove(self):
        model = Model(); self.gate(model); prompt = model.calls[0]
        valid = {'decisions': [{'shape_id': c['shape_id'], 'decision': 'delete', 'image_id': c['image_ids'][0], 'reason': 'same visible photo'} for c in prompt['candidates']]}
        for modification in ['duplicate', 'missing', 'wrong_image', 'no_image']:
            bad = copy.deepcopy(valid)
            if modification == 'duplicate': bad['decisions'][1] = bad['decisions'][0]
            if modification == 'missing': bad['decisions'].pop()
            if modification == 'wrong_image': bad['decisions'][0]['image_id'] = 'I_IB'
            if modification == 'no_image': bad['decisions'][0]['image_id'] = None
            with self.assertRaises(ValueError): validate_response(bad, prompt)
        with self.assertRaises(ValueError):
            filter_candidates(self.shapes, self.assets, .4, mode='model_review', reviewer=lambda batch: [])

    def test_old_rule_mode_remains_available(self):
        self.config['model_conflict_review'] = False; model = Model(); result = self.gate(model)
        self.assertEqual(model.calls, []); self.assertEqual(result['mode'], 'rule_only')
        self.assertEqual(result['threshold'], .85)

    def test_replay_uses_pre_dedup_shapes_original_asset_coordinates_and_preserves_source(self):
        source = self.work/'old'; folder = source/'cases/case'
        item = {'id': 'case', 'original': 'original.png', 'source_sha256': self.ref['sha256'], 'source_size': self.ref['size']}
        geometry = {'image_size': self.ref['size'], 'coordinate_system': 'image_pixel_centers', 'objects': self.shapes}
        save(source/'manifest.json', {'images': [item]})
        save(folder/'shapes/geometry.json', geometry)
        save(folder/'geometry.json', {**geometry, 'objects': self.shapes[1:]})
        save(folder/'dedup/removed.json', self.shapes[:1])
        save(folder/'dedup/image_boxes_original.json', {'source_sha256': self.ref['sha256'], 'assets': self.assets})
        save(folder/'summary.json', {'errors': {}})
        provenance = source_manifest(source, [item]); before = {str(p): sha(p) for p in source.rglob('*.json')}
        model = Model({'S_B': 'keep', 'S_C': 'defer'})
        with patch('shape_integration.batch.run_shape', side_effect=AssertionError('No localization')), patch('shape_integration.image_bridge.detect', side_effect=AssertionError('No detection')):
            result = replay_case(self.work, self.work/'new', item, model, self.config, source, provenance, False)
        self.assertEqual(result['enhanced_shapes'], 4)
        self.assertEqual(result['model_conflict_review_candidates'], 3)
        self.assertEqual(result['removed_ids'], ['A'])
        self.assertEqual(result['rule_removed'], 0); self.assertEqual(result['model_removed'], 1)
        new = self.work/'new/cases/case'
        self.assertEqual(read(new/'dedup/image_boxes_original.json')['assets'], self.assets)
        self.assertEqual(read(new/'geometry.json')['objects'], self.shapes[1:])
        self.assertEqual({str(p): sha(p) for p in source.rglob('*.json')}, before)


if __name__ == '__main__':
    unittest.main()
