"""Endpoint preservation and current-method freezing, without inference calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parent
if ROOT.name == 'tests':
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from full_recovery.arrow_finalize import protect_endpoints, profile
from full_recovery.arrow_backend import freeze


def row(ident, start=None, end=None):
    return {'id': ident, 'style': {'start': {'type': start} if start else None,
                                   'end': {'type': end} if end else None}}


class EndpointTests(unittest.TestCase):
    def selection(self, a, b, reverse=False):
        before = {'arrows': [a, b]}
        audit = {'removed': [{'remove': a['id'], 'keep': b['id'], 'pair': 'p'}]}
        original = deepcopy(before)
        result, report = protect_endpoints(before, {'arrows': [b]}, audit,
                                           [{'id': 'p', 'reversed': reverse}])
        self.assertEqual(before, original)
        return result, report

    def test_real_start_dot_cannot_be_lost_to_headless_duplicate(self):
        result, audit = self.selection(row('dot', 'oval', 'triangle'), row('bare', None, 'triangle'))
        self.assertEqual(len(result['arrows']), 2)
        self.assertEqual(audit['removed'], [])
        self.assertEqual(audit['endpoint_guards'][0]['endpoints'], [0])

    def test_opposite_path_order_maps_physical_endpoints(self):
        result, audit = self.selection(row('a', 'oval', 'triangle'), row('b', 'triangle', 'oval'), True)
        self.assertEqual([r['id'] for r in result['arrows']], ['b'])
        self.assertEqual(len(audit['removed']), 1)

    def test_different_head_types_and_component_keeper_cannot_drop_information(self):
        result, _ = self.selection(row('a', 'oval'), row('b', 'diamond'))
        self.assertEqual(len(result['arrows']), 2)

    def test_profile_keeps_shared_request_limit_and_original_budget(self):
        model = SimpleNamespace(config={'max_tokens': 113000}, slots=threading.BoundedSemaphore(2),
                                blocked=threading.Event())
        style = profile(model, max_tokens=8192, reasoning_effort='low')
        self.assertEqual(model.config['max_tokens'], 113000)
        self.assertIs(style.slots, model.slots)
        self.assertIs(style.blocked, model.blocked)


class FreezeTests(unittest.TestCase):
    def test_current_methods_are_frozen_without_loading_any_previous_ppt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'methods'; target = root/'backend'; delivery = root/'batch/delivery'
            for path in (source/'lib', source/'transport', source/'shape_core', target, delivery):
                path.mkdir(parents=True, exist_ok=True)
            (source/'config_point_geometry_mainline.json').write_text('{"max_tokens":113000}')
            for path in (source/'BASELINE.json', source/'shape_core/SOURCE.json', delivery/'manifest.json'):
                path.write_text('{}')
            (source/'pipeline.py').write_text("def run():\n    with locks['cpu']:allrows=publish(clean,results,out)\n    rendered=False\n")
            (source/'transport/client.py').write_text('retryable=isinstance(exc,httpx.TransportError)) from None')
            dedup = delivery.parent/'code/dedup.py'; dedup.parent.mkdir()
            dedup.write_text('# delivered identity method\n')
            digest = hashlib.sha256(dedup.read_bytes()).hexdigest()
            (delivery.parent/'code_manifest.json').write_text(json.dumps({'code/dedup.py': digest}))
            config, files = freeze(ROOT, source, delivery, target)
            self.assertTrue(config['whole_path_identity_review'])
            self.assertEqual(config['style_max_tokens'], 8192)
            self.assertIn('identity_issues', (target/'pipeline.py').read_text())
            self.assertIn('TimeoutError', (target/'transport/client.py').read_text())
            self.assertEqual((target/'ownership_dedup.py').read_bytes(), dedup.read_bytes())
            self.assertIn('METHOD_SOURCE.json', files)
            self.assertFalse(list(target.rglob('*.pptx')))


if __name__ == '__main__':
    unittest.main()
