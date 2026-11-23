"""Normalized wire coordinates remain isolated from pixel geometry and caches."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'vendor')]
from shape_integration.common import Runtime, read, save, sha, mapped_box
from shape_integration.fitting.coordinates import NORM1000_CONVENTION
from shape_integration.fitting.local import prepare_candidate, observe_and_fit
from shape_integration.fitting.observation import make_prompt
from shape_integration.fitting.control import process_object, prepared
from shape_integration.fitting.coverage import review
from shape_integration.pipeline import audit


def observed(status='ok'):
    return {'status': status, 'reason': 'visible rectangle' if status == 'ok' else 'Need full target context',
            'points': [{'id': str(i), 'xy': xy} for i, xy in enumerate(
                [[0, 0], [1000, 0], [1000, 1000], [0, 1000]])] if status == 'ok' else [],
            'parts': [{'id': 'rect', 'kind': 'rectangle', 'point_ids': ['0', '1', '2', '3']}] if status == 'ok' else [],
            'ownership': {'unit': 'other', 'unit_evidence': 'Visible rectangle',
                          'parts': [{'id': 'rect', 'role': 'body'}] if status == 'ok' else []}}


class NormalizedCoordinateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'tests'); self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.source = self.work/'original.png'; self.input = self.work/'input.png'
        Image.new('RGB', (400, 200), 'white').save(self.source)
        Image.new('RGB', (2048, 1024), 'white').save(self.input)
        self.item = dict(id='case', original='original.png', input='input.png', source_size=[400, 200],
                         input_size=[2048, 1024], source_sha256=sha(self.source), input_sha256=sha(self.input))
        self.candidate = dict(id='G1', label='block', kind='rectangle', bbox_2d=[70, 40, 180, 140])
        self.config = read(ROOT/'config.json')
        self.config['fitting']['coordinate_mode'] = 'norm1000'
        self.config['fitting']['coverage_rounds'] = 0
        self.options = self.config['fitting']

    def test_prompt_only_changes_coordinates_and_keeps_input_images(self):
        with Image.open(self.source) as image:
            pixel = prepare_candidate(image, self.candidate, self.item['input_size'], self.work/'pixel')
            norm = prepare_candidate(image, self.candidate, self.item['input_size'], self.work/'norm', coordinate_mode='norm1000')
        a, b = make_prompt(pixel), make_prompt(norm)
        self.assertNotIn('coordinate_mode', pixel)
        self.assertEqual(b['local_coordinates'], {'convention': NORM1000_CONVENTION})
        for key in a.keys() - {'local_coordinates', 'output_schema'}:
            self.assertEqual(a[key], b[key])
        for name in ['clean.png', 'overlay.png', 'frame.json', 'candidate.json']:
            self.assertEqual(sha(self.work/'pixel'/name), sha(self.work/'norm'/name))
        with self.assertRaisesRegex(ValueError, 'coordinate mode changed'):
            prepared(self.source, self.candidate, self.item, self.work/'norm', {**self.options, 'coordinate_mode': 'pixel'})

    def test_real_runtime_cache_does_not_scale_twice(self):
        folder = self.work/'cases/case/fitting/objects/G1/revision_00'
        job = prepared(self.source, self.candidate, self.item, folder, self.options)
        calls = []
        class Client:
            def __init__(self, root): self.root = root
            def request_parameters(self): return {'model': 'offline'}
            def redact(self, error): return str(error)
            def call(self, tag, system, prompt, images, validator):
                calls.append(tag)
                raw = observed(); result = validator(raw)
                save(self.root/tag/'parsed_response.json', raw)
                save(self.root/tag/'result.json', result)
                return result
        runtime = Runtime(self.config, 'offline', 'offline')
        with patch.object(runtime, 'client', side_effect=lambda root, budget: Client(root)):
            for _ in range(2):
                result = observe_and_fit(job, folder, runtime, namespace='test/G1', max_tokens=1000)
                self.assertEqual(result['status'], 'finished')
        self.assertEqual(calls, ['observe_00'])
        wire = read(folder/'observation_wire.json'); pixels = read(folder/'observation.json')
        self.assertEqual(wire['points'][2]['xy'], [1000, 1000])
        w, h = job['frame']['actual_crop_input_size_px']
        np.testing.assert_allclose(pixels['points'][2]['xy'], [w-1, h-1])
        self.assertEqual(read(folder/'fit_local.json')['observation']['points'], pixels['points'])
        self.assertEqual(read(folder/'calls/observe_00/result.json')['points'], wire['points'])
        self.assertTrue(audit(self.work, self.config)['passed'])

    def test_recrop_then_refine_uses_normalized_wire_and_pixel_critic_images(self):
        events = []
        def call(runtime, root, tag, system, prompt, images, validator, budget, **kwargs):
            events.append((tag, prompt))
            if tag == 'observe':
                self.assertEqual(prompt['local_coordinates'], {'convention': NORM1000_CONVENTION})
                if len(events) == 1: return validator(observed('needs_context'))
                return validator(observed())
            if tag == 'plan':
                self.assertNotIn('image_size', prompt)
                return validator({'bbox_2d': [0, 0, 1000, 1000], 'reason': 'full object'})
            if tag != 'check': raise AssertionError(tag)
            # Critic must render the converted pixel endpoints, not normalized xy.
            revision = Path(root).parent
            points = read(revision/'observation.json')['points']
            with Image.open(revision/'fit_overlay.png') as image:
                self.assertEqual(points[2]['xy'], [image.width-1, image.height-1])
                self.assertNotEqual(image.getpixel((image.width-1, image.height-1)), (255, 255, 255))
            if len(events) == 4:
                return validator({'decision': 'revise', 'issues': [{'kind': 'geometry', 'parts': ['F1'],
                    'evidence': 'Left edge lies outside the source outline.', 'action': 'Move left edge to source outline.'}]})
            return validator({'decision': 'accept', 'issues': []})
        root = self.work/'fitting/objects/G1'
        with patch.object(Runtime, 'call', call):
            result = process_object(root, self.candidate, self.item, self.work,
                                    Runtime(self.config, 'offline', 'offline'), self.options, namespace='test')
        self.assertEqual(result['state'], 'accepted_by_model')
        self.assertEqual([tag for tag, _ in events], ['observe', 'plan', 'observe', 'check', 'observe', 'check'])
        self.assertTrue(events[4][1]['revision_feedback'])
        self.assertEqual(read(root/'revision_01/frame.json')['source_crop_half_open'], [0, 0, 400, 200])

    def test_coverage_converts_boxes_before_dedup_and_queue(self):
        folder = self.work/'fitting/coverage_00'; folder.parent.mkdir()
        box = [100, 200, 800, 900]
        wire = {'status': 'complete', 'new_objects': [{'label': 'new block', 'kind': 'rectangle',
                'bbox_2d': box, 'reason': 'missing'}], 'possible_duplicates': [], 'uncertain': [], 'repair_existing': []}
        original = copy.deepcopy(wire)
        def call(runtime, root, tag, system, prompt, images, validator, budget):
            self.assertNotIn('image_size', prompt)
            invalid = copy.deepcopy(wire); invalid['new_objects'][0]['bbox_2d'][2] = 1001
            with self.assertRaises(ValueError): validator(invalid)
            return validator(wire)
        def dedup(proposed, *args):
            self.assertEqual(proposed[0]['bbox_2d'], mapped_box([204.7, 204.6, 1637.6, 920.7], [2048, 1024], [400, 200]))
            return {'kept': proposed, 'removed': [], 'deferred_ids': [], 'model_request_calls': 0}
        with patch.object(Runtime, 'call', call), patch('shape_integration.fitting.coverage.run_filter', side_effect=dedup):
            result = review({'objects': {}, 'parts': {}}, self.item, self.work, folder,
                            Runtime(self.config, 'offline', 'offline'), self.options, [], [], .85, 0)
        self.assertEqual(len(result['new_candidate_ids']), 1)
        self.assertTrue(result['new_candidate_ids'][0].startswith('F1_N001_'))
        self.assertEqual(wire, original)
        self.assertEqual(read(folder/'review_wire.json'), wire)

    def test_coverage_part_locations_keep_view_pixels_and_existing_repair_identity(self):
        part={'id':'body','kind':'rectangle','status':'ok','bbox_2d':[70,40,300,140],
              'commands':[['M',70,40],['L',300,40],['L',300,140],['L',70,140],['Z']]}
        scene={'objects':{'G1':{'state':'accepted_by_model','part_keys':['k']}},'parts':{'k':part}}
        before=copy.deepcopy(scene)
        wire={'status':'uncertain','new_objects':[],'possible_duplicates':[],'uncertain':[],
              'repair_existing':[{'object_id':'G1','issues':[{'kind':'missing','part_ids':['body'],
                'evidence':'The current native member corresponds to the wrong source target.',
                'action':'Correct the existing member locally, preserving other members.'}]}]}
        for mode in ['pixel','norm1000']:
            folder=self.work/('coverage_locations_'+mode)
            options={**self.options,'coordinate_mode':mode}
            def call(runtime,root,tag,system,prompt,images,validator,budget):
                value=prompt['object_states'][0]['represented_parts'][0]
                expected=mapped_box(part['bbox_2d'],self.item['source_size'],self.item['input_size'])
                np.testing.assert_allclose(value['fitted_bbox_pixels'],expected,atol=.01)
                self.assertGreater(value['fitted_bbox_pixels'][2],1000)
                return validator(wire)
            with patch.object(Runtime,'call',call),patch('shape_integration.fitting.coverage.run_filter',
                return_value={'kept':[],'removed':[],'deferred_ids':[],'model_request_calls':0}):
                result=review(scene,self.item,self.work,folder,Runtime(self.config,'offline','offline'),
                              options,[self.candidate],[],.85,0)
            self.assertEqual(result['repair_existing'][0]['issues'][0]['part_ids'],['body'])
        self.assertEqual(scene,before)


if __name__ == '__main__': unittest.main(verbosity=2)
