"""Input isolation and one-request multi-part integration checks (no API)."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'vendor')]
from shape_integration.common import read
from shape_integration.fitting.frames import FitFrame
from shape_integration.fitting.local import observe_and_fit, prepare_candidate
from shape_integration.fitting.observation import make_prompt


class LocalFittingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'tests')
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)/'target'
        yy, xx = np.mgrid[:24, :32]
        self.image = Image.fromarray(np.stack([xx*7, yy*9, xx+yy], axis=-1).astype('uint8'))
        self.candidate = {'id': 'G1', 'label': 'two blocks', 'kind': 'compound_symbol',
                          'bbox_2d': [8.2, 5.4, 20.8, 17.3]}

    def prepare(self):
        return prepare_candidate(self.image, self.candidate, [2048, 1536], self.folder,
                                 padding=.12, long_edge=256)

    def test_native_pixels_dimensions_and_bbox_roundtrip(self):
        job = self.prepare(); frame = FitFrame.from_metadata(job['frame'])
        with Image.open(self.folder/'native.png') as native:
            self.assertEqual(native.tobytes(), self.image.crop(job['frame']['source_crop_half_open']).tobytes())
        with Image.open(self.folder/'clean.png') as clean, Image.open(self.folder/'overlay.png') as overlay:
            self.assertEqual(clean.size, overlay.size)
            self.assertEqual(max(clean.size), 256)
            self.assertNotEqual(clean.tobytes(), overlay.tobytes())
        b = job['identity_bbox_local']
        np.testing.assert_allclose(frame.to_original(b[:2])+frame.to_original(b[2:]), self.candidate['bbox_2d'], atol=1e-10)

    def test_prompt_keeps_backend_coordinates_out(self):
        job = self.prepare()
        job['label'] = 'Blue symbol; bbox [901, 902, 903, 904]; center at x=991, y=992.'
        prompt = make_prompt(job, ['object group; global x 995-997, y 123-456'])
        text = json.dumps(prompt,ensure_ascii=False)
        self.assertFalse(set(prompt) & {'frame', 'candidate', 'bbox_2d', 'identity_bbox_local'})
        # Sanitization removes numeric geometry; harmless prose remnants are
        # allowed so this check does not require an unrelated label rewrite.
        self.assertTrue(prompt['target']['label'].startswith('Blue symbol'))
        for number in ['901', '902', '991', '995', '997']:
            self.assertNotIn(number, text)
        self.assertEqual(prompt['output_schema']['properties']['status']['enum'],
                         ['ok', 'partial', 'needs_context', 'unsupported'])

    def test_original_4_8_16_point_clauses_are_unchanged(self):
        source = ROOT/'tests/fixtures/original_local_observation_protocol.py'
        tree = ast.parse(source.read_text())
        clauses = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                   and n.value.startswith(('Circle or ellipse:', 'Rounded rectangle:', 'Rectangle:'))]
        self.assertEqual(len(clauses), 3)
        prompt = make_prompt(self.prepare())
        for clause in clauses:
            self.assertIn(clause, prompt['sampling'].values())

    def test_one_model_request_returns_two_fitted_parts(self):
        job = self.prepare()
        raw = {'object_id': 'G1', 'status': 'ok', 'reason': 'Two visible cells', 'uncertain': [],
               'points': [{'id': pid, 'xy': xy} for pid, xy in
                          [('a', [30, 30]), ('b', [100, 30]), ('c', [100, 100]),
                           ('d', [30, 100]), ('e', [190, 30]), ('f', [190, 100])]],
               'parts': [{'id': 'left', 'kind': 'rectangle', 'point_ids': ['a', 'b', 'c', 'd']},
                         {'id': 'right', 'kind': 'rectangle', 'point_ids': ['b', 'e', 'f', 'c']}]}
        raw['ownership']={'unit':'other','unit_evidence':'Visible geometric cells','parts':[{'id':p['id'],'role':'body'} for p in raw['parts']]}
        class FakeRuntime:
            config = {'shape_schema_retries': 0}
            calls = 0
            def call(self, root, tag, system, prompt, images, validator, budget):
                self.calls += 1
                if len(images) != 2:
                    raise AssertionError('Expected aligned clean and overlay images')
                return validator(raw)
        runtime = FakeRuntime()
        status = observe_and_fit(job, self.folder, runtime, namespace='test/G1', max_tokens=1000)
        self.assertEqual(runtime.calls, 1)
        self.assertEqual(status['part_counts']['succeeded'], 2)
        self.assertEqual(status['completeness_status'], 'unreviewed')
        fit = read(self.folder/'fit_original.json')
        self.assertEqual(len(fit['parts']), 2)
        self.assertNotIn('commands', fit)


if __name__ == '__main__':
    unittest.main(verbosity=2)
