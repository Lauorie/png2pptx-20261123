import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'vendor')]
from png2pptx.images import boundary as serial
from shape_integration import image_boundary as parallel, image_bridge
from shape_integration.common import Runtime, read, sha


def semantic(value):
    """Ignore only evidence paths, which necessarily differ between run dirs."""
    if isinstance(value, dict):
        return {k: semantic(v) for k, v in value.items() if k not in {'frame', 'evidence'}}
    if isinstance(value, list):
        return [semantic(v) for v in value]
    return value


def reply(decision, bbox=None):
    edges = dict.fromkeys(('left', 'top', 'right', 'bottom'), 'ok')
    if decision == 'revise':
        edges['left'] = 'clipped'
    elif decision in ('needs_context', 'unresolved'):
        edges['left'] = 'unclear'
    return {'decision': decision, 'bbox': bbox, 'edges': edges, 'reason': '离线测试的可见边界证据'}


class ParallelBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'tests')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'source.png'
        self.size = [120, 80]
        image = Image.new('RGB', self.size, 'white')
        draw = ImageDraw.Draw(image)
        boxes = [[10, 15, 25, 35], [50, 15, 65, 35], [90, 15, 105, 35]]
        for box in boxes:
            draw.rectangle(box, fill='#009876')
        image.save(self.source)
        self.inventory = {'assets': [{'id': f'I{i:04d}', 'bbox': box, 'category': 'natural_photo',
                            'background': 'opaque_panel', 'target': f'target {i}',
                            'keep': 'all target pixels', 'remove': 'nearby labels'}
                           for i, box in enumerate(boxes, 1)], 'uncertain': []}

    def scripted(self, concurrent):
        barrier = threading.Barrier(3) if concurrent else None
        lock = threading.Lock()
        requests, first_finished = {}, set()

        def fake(client, stage, system, prompt, images, validator):
            ident = prompt['asset_id']
            first = prompt['purpose'] == 'initial_review'
            with lock:
                requests[stage] = (system, copy.deepcopy(prompt), [(label, sha(path)) for label, path in images])
                if not first:
                    self.assertEqual(first_finished, {'I0001', 'I0002', 'I0003'})
            if first and barrier:
                barrier.wait(timeout=10)
            if first and ident == 'I0001':
                bbox = list(prompt['current_bbox_hint'])
                bbox[0] -= 8
                raw = reply('revise', bbox)
            elif first and ident == 'I0003':
                raw = reply('needs_context')
            elif ident == 'I0003':
                raw = reply('unresolved')
            else:
                raw = reply('accept')
            with lock:
                if first:
                    first_finished.add(ident)
            return validator(raw), 'offline/'+stage
        return fake, requests

    def test_overlapping_requests_preserve_rounds_prompts_images_and_results(self):
        original = copy.deepcopy(self.inventory)
        # A tight budget also verifies deterministic inventory-order allocation.
        for budget in (4, 5):
            a, serial_requests = self.scripted(False)
            b, parallel_requests = self.scripted(True)
            with patch.object(serial, 'cached_json_call', side_effect=a):
                expected = serial.review_boundaries(None, self.source, self.inventory, self.size,
                                                     self.root/f'serial{budget}', max_calls=budget)
            scheduler = parallel.BoundaryScheduler(3)
            with patch.object(parallel, 'cached_json_call', side_effect=b):
                actual = scheduler(None, self.source, self.inventory, self.size,
                                   self.root/f'parallel{budget}', max_calls=budget)
            self.assertEqual(serial_requests, parallel_requests)
            self.assertEqual(semantic(expected), semantic(actual))
            self.assertEqual(actual['boundary_review']['calls'], budget)
            self.assertIsNone(scheduler._pool)
            self.assertEqual(read(self.root/f'parallel{budget}'/'scheduling.json')['pending_asset_ids'], [])
        self.assertEqual(self.inventory, original)

    def test_failed_object_does_not_cancel_other_objects(self):
        def fake(client, stage, system, prompt, images, validator):
            if prompt['asset_id'] == 'I0002':
                raise RuntimeError('retained model/transport failure')
            return validator(reply('accept')), 'offline/'+stage
        with patch.object(parallel, 'cached_json_call', side_effect=fake):
            result = parallel.BoundaryScheduler(3)(None, self.source, self.inventory, self.size,
                                                   self.root/'failed')
        states = {a['id']: a['boundary_review']['status'] for a in result['assets']}
        self.assertEqual(states, {'I0001': 'model_review_complete', 'I0002': 'unresolved',
                                  'I0003': 'model_review_complete'})
        self.assertIn('retained model/transport failure', read(self.root/'failed/I0002/round_00/error.json')['error'])
        self.assertEqual(result['boundary_review']['calls'], 3)

    def test_no_calls_for_zero_budget_or_empty_inventory(self):
        with patch.object(parallel, 'cached_json_call', side_effect=AssertionError('Unexpected call')):
            scheduler = parallel.BoundaryScheduler(2)
            zero = scheduler(None, self.source, self.inventory, self.size, self.root/'zero', max_calls=0)
            empty = scheduler(None, self.source, {'assets': [], 'uncertain': []}, self.size, self.root/'empty')
        self.assertEqual(zero['boundary_review']['calls'], 0)
        self.assertEqual(zero['boundary_review']['unresolved_ids'], ['I0001', 'I0002', 'I0003'])
        self.assertEqual(empty['boundary_review']['status'], 'not_needed')

    def test_context_cannot_expand_beyond_original_image(self):
        inventory = copy.deepcopy(self.inventory)
        inventory['assets'] = inventory['assets'][:1]
        inventory['assets'][0]['bbox'] = [0, 0, 119, 79]
        def fake(client, stage, system, prompt, images, validator):
            return validator(reply('needs_context')), 'offline/'+stage
        with patch.object(parallel, 'cached_json_call', side_effect=fake) as calls:
            result = parallel.BoundaryScheduler(2)(None, self.source, inventory, self.size, self.root/'context')
        self.assertEqual(calls.call_count, 1)
        self.assertEqual(result['assets'][0]['boundary_review']['reason'], 'no larger distinct source view available')

    def test_shared_pool_across_figures_and_global_api_slots(self):
        config = {'api_concurrency': 2, 'model': 'glm-5.3-flash'}
        runtime = Runtime(config, 'http://offline.invalid/v1', '')
        client = runtime.client(self.root/'calls', 50000)
        scheduler = parallel.BoundaryScheduler(4)
        lock = threading.Lock()
        external_started, overlapping, release = [threading.Event() for _ in range(3)]
        active, peak, pools = 0, 0, set()

        def work(external=False):
            nonlocal active, peak
            with runtime.slots:
                with lock:
                    active += 1
                    peak = max(peak, active)
                    if active == 2:
                        overlapping.set()
                if external:
                    external_started.set()
                try:
                    self.assertTrue(release.wait(timeout=10))
                finally:
                    with lock:
                        active -= 1

        def fake(api_client, stage, system, prompt, images, validator):
            self.assertIs(api_client.slots, runtime.slots)
            with lock:
                pools.add(id(scheduler._pool))
            work()
            return validator(reply('accept')), 'offline/'+stage

        with patch.object(parallel, 'cached_json_call', side_effect=fake), ThreadPoolExecutor(max_workers=3) as tasks:
            external = tasks.submit(work, True)
            self.assertTrue(external_started.wait(timeout=10))
            figures = [tasks.submit(scheduler, client, self.source, self.inventory, self.size,
                                    self.root/f'figure{i}') for i in range(2)]
            try:
                self.assertTrue(overlapping.wait(timeout=10))
            finally:
                release.set()
            external.result()
            for task in figures:
                self.assertEqual(task.result()['boundary_review']['status'], 'model_review_complete')
        self.assertEqual(peak, 2)
        self.assertEqual(len(pools), 1)
        self.assertIsNone(scheduler._pool)

    def test_adapter_uses_shared_scheduler_and_same_client_parameters(self):
        runtime = Runtime({'api_concurrency': 64, 'model': 'glm-5.3-flash'}, 'http://offline.invalid/v1', '')
        with patch.object(image_bridge.pipeline, 'make_detector'), patch.object(image_bridge.pipeline, 'review_boundaries'):
            image_bridge.configure(runtime)
            self.assertIsInstance(image_bridge.pipeline.review_boundaries, parallel.BoundaryScheduler)
            self.assertEqual(image_bridge.pipeline.review_boundaries.workers, 64)
            client = image_bridge.pipeline.make_detector(self.root/'adapter', max_tokens=50000)
            self.assertIs(client.slots, runtime.slots)
            self.assertEqual(client.request_parameters()['max_tokens'], 50000)
            self.assertEqual(client.request_parameters()['thinking'], {'type': 'enabled'})
            self.assertNotIn('reasoning_effort', client.request_parameters())

    def test_rejects_frozen_serial_directory(self):
        folder = self.root/'frozen'
        serial.review_boundaries(None, self.source, self.inventory, self.size, folder, max_calls=0)
        before = sha(folder/'identity.json')
        with self.assertRaisesRegex(ValueError, 'choose a new output directory'):
            parallel.BoundaryScheduler(2)(None, self.source, self.inventory, self.size, folder, max_calls=0)
        self.assertEqual(sha(folder/'identity.json'), before)


if __name__ == '__main__':
    unittest.main()
