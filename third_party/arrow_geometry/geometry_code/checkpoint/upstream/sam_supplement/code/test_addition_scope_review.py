import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import experiment as E
import addition_scope_review as S
import review_additions as R


def row(name, box):
    return dict(id=name, bbox=box, render_type='linework', grouping='single',
                label='old misleading interpretation', extent='complete', operation='add_only')


class ScopeTests(unittest.TestCase):
    def test_scope_removal_preserves_all_other_fields_and_baseline(self):
        old = [row('B1', [0, 0, 10, 10])]
        rows = [row(name, [20, 20, 80, 70]) for name in ['keep', 'reject', 'uncertain', 'failed']]
        saved = copy.deepcopy((old, rows))
        class Fake:
            def ask(self, folder, prompt, images, validator, **kwargs):
                name = prompt['target']['id']
                if name == 'failed': raise RuntimeError('simulated timeout')
                return validator(dict(decision=name, reason='visible evidence'))
        with tempfile.TemporaryDirectory() as d:
            kept, audit = S.review(Path(d), 'test', rows, old, Image.new('RGB', (200, 120)), Fake(), {})
            self.assertEqual(kept, [rows[0], rows[2], rows[3]])
            self.assertEqual(audit['rejected_count'], 1)
            self.assertEqual(audit['uncertain_count'], 2)
            self.assertEqual(len(audit['issues']), 2)
            self.assertEqual((old, rows), saved)

    def test_complete_original_and_exact_coordinate_mapping_without_old_label(self):
        image = Image.new('RGB', (1185, 410), 'white')
        r = row('ADD1', [471.666, 35.666, 483.333, 353.333])
        with tempfile.TemporaryDirectory() as d:
            q, images = S.evidence(Path(d), image, r, {})
            full = Image.open(images[0][1]); self.assertEqual(full.size, image.size)
            self.assertEqual(full.tobytes(), image.tobytes())
            self.assertNotIn(r['label'], str(q))
            f = E.read(Path(d) / 'frames.json')['local_frame']
            transform = f['local_to_global']
            for i, v in enumerate(q['target']['bbox_local']):
                native = v * transform['scale_xy'][i % 2] + transform['offset_xy'][i % 2]
                self.assertAlmostEqual(native, r['bbox'][i])

    def test_invalid_protocol_cannot_silently_remove_candidate(self):
        for a in [dict(decision='delete', reason='x'), dict(decision='reject', reason=''),
                  dict(decision='reject', reason='x', bbox=[1, 2, 3, 4]), {}]:
            with self.assertRaises(ValueError): S.validate(a)

    def test_disabled_or_empty_stage_makes_no_requests(self):
        class Never:
            def ask(self, *args, **kwargs): raise AssertionError('No call expected')
        with tempfile.TemporaryDirectory() as d:
            rows = [row('A1', [10, 10, 40, 40])]
            kept, audit = S.review(Path(d), 'disabled', rows, [], Image.new('RGB', (100, 100)), Never(),
                                   {'sam_addition_scope_review_enabled': False})
            self.assertEqual(kept, rows); self.assertFalse(audit['enabled'])
            kept, audit = S.review(Path(d), 'empty', [], [], Image.new('RGB', (100, 100)), Never(), {})
            self.assertFalse(kept); self.assertEqual(audit['decisions'], [])

    def test_real_pipeline_handoff_excludes_only_rejected_addition(self):
        class Fake:
            def ask(self, folder, prompt, images, validator, **kwargs):
                if 'target' in prompt:
                    return validator(dict(decision='reject', reason='closed container border'))
                return validator(dict(frame='LOCAL', decision='edit', reason='', keep_ids=['ADD1'],
                                      replacements=[], discard=[]))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'out'; source = Path(d) / 'source'; cid = 'task_test'
            cfg = dict(source_run=str(source), review_rounds=1, review_batch_size=6,
                       review_context_scale=1.6, review_context_expansions=1, review_workers=1)
            old = [row('B1', [10, 10, 20, 20])]
            E.write(root / 'inputs' / cid / 'baseline_native.json', old)
            E.save(Image.new('RGB', (300, 200), 'white'), root / 'inputs' / cid / 'source.png')
            E.write(source / 'cases' / cid / 'additions.json', [row('ADD1', [150, 40, 160, 160])])
            E.write(source / 'cases' / cid / 'status.json', {'state': 'completed'})
            E.write(root / 'config.json', cfg); E.write(root / 'cases.json', [cid])
            R.run_case(root, cid, cfg, Fake())
            result = E.read(root / 'cases' / cid / 'result.json')
            self.assertIsNotNone(result, E.read(root / 'cases' / cid / 'failure.json'))
            self.assertEqual(result['state'], 'completed')
            self.assertEqual(result['native_regions'], old)
            self.assertEqual(result['addition_scope_review']['rejected_count'], 1)
            handoff = E.read(root / 'cases' / cid / 'reconstruction_handoff.json')
            self.assertNotIn('ADD1', str(handoff))


if __name__ == '__main__': unittest.main()
