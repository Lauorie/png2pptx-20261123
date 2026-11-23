import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline_frames import Crop
from post_tile_sam import edge_sides, same_ink, ink_coverage, append_objects, validate_review


class PostSamSafety(unittest.TestCase):
    def test_truncated_observation_rejected_but_overlap_can_rescue(self):
        self.assertEqual(edge_sides([0, 20, 8, 60], (100, 100)), ['left'])
        self.assertEqual(edge_sides([30, 20, 38, 60], (100, 100)), [])
        self.assertEqual(edge_sides([20, 80, 30, 99], (100, 100)), ['bottom'])

    def test_nearby_parallel_ink_does_not_merge_despite_overlapping_boxes(self):
        a = {'bbox_native': [5, 5, 10, 20], '_ink': np.array([100, 200, 300])}
        b = {'bbox_native': [7, 5, 12, 20], '_ink': np.array([102, 202, 302])}
        self.assertFalse(same_ink(a, b))
        b['_ink'] = a['_ink'].copy()
        self.assertTrue(same_ink(a, b))

    def test_mask_coverage_maps_resized_tile_to_original_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr = np.zeros((60, 60), dtype='uint8')
            arr[10:40, 30:33] = 255
            path = Path(tmp)/'mask.png'
            Image.fromarray(arr).save(path)
            view = Crop((100, 100), (40, 50, 60, 70), (60, 60))
            m = {'frame': view.metadata((100, 100)), 'mask_path': str(path), 'bbox_mask': [30, 10, 32, 39]}
            covered = {'id': 'old', 'bbox_native': [49, 52, 52, 65]}
            unrelated = {'id': 'other', 'bbox_native': [44, 52, 46, 65]}
            self.assertEqual(ink_coverage(m, [covered], 0)[0], 1)
            self.assertEqual(ink_coverage(m, [unrelated], 0)[0], 0)

    def test_source_objects_remain_byte_equivalent_as_json_values(self):
        old = [{'id': 'old', 'display_id': 'I0107', 'bbox_native': [1, 2, 3, 4], 'nested': {'v': [9]}}]
        saved = copy.deepcopy(old)
        result = append_objects(old, [{'id': 'new', 'bbox_native': [7, 8, 9, 10]}])
        self.assertEqual(old, saved)
        self.assertEqual(result[:1], saved)
        self.assertEqual(result[1]['display_id'], 'I0108')
        result[0]['nested']['v'][0] = 42
        self.assertEqual(old, saved)

    def test_covered_requires_real_readonly_reference(self):
        with self.assertRaises(ValueError):
            validate_review({'decision': 'covered', 'reason': 'same', 'existing_id': 'r9'}, (100, 100), {'r1'})

    def test_partial_or_outside_addition_cannot_enter_final_list(self):
        a = {'decision': 'add', 'reason': 'visible', 'kind': 'single', 'extent': 'partial', 'bbox': [1, 2, 3, 4], 'locator': 'arrow'}
        with self.assertRaises(ValueError):
            validate_review(a, (100, 100), set())
        a.update(extent='complete', bbox=[1, 2, 103, 104])
        with self.assertRaises(ValueError):
            validate_review(a, (100, 100), set())


if __name__ == '__main__':
    unittest.main()
