"""Numerical recovery, multi-part preservation, failure and frame regressions."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT/'src'))
from shape_integration.fitting import (FitError, FitFrame, ObservationError, fit_boundary,
                                       fit_object, transform_fit, validate_observation)
from shape_integration.fitting.paths import sample_commands
from shape_integration.fitting.primitives import contour, rotation


def rr_points(width=220, height=104, radius=18, angle=.22, center=(350, 250)):
    w, h, r = width, height, radius
    points = [[0, -h/2]]
    for cx, cy, a, mid in [(w/2-r, -h/2+r, -math.pi/2, [w/2, 0]),
                           (w/2-r, h/2-r, 0, [0, h/2]),
                           (-w/2+r, h/2-r, math.pi/2, [-w/2, 0]),
                           (-w/2+r, -h/2+r, math.pi, None)]:
        points.extend([[cx+r*math.cos(a+t), cy+r*math.sin(a+t)] for t in [0, math.pi/4, math.pi/2]])
        if mid:
            points.append(mid)
    return np.asarray(points)@rotation(angle).T+center


def empty_raw():
    return {'object_id': 'target', 'status': 'ok', 'points': [], 'parts': [],
            'reason': 'Observed visible parts', 'uncertain': []}


def add_primitive(raw, kind, xy, name=None):
    name = name or f'part{len(raw["parts"])}'
    refs = []
    for i, point in enumerate(xy):
        pid = f'{name}_{i}'
        raw['points'].append({'id': pid, 'xy': np.round(point, 2).tolist()})
        refs.append(pid)
    raw['parts'].append({'id': name, 'kind': kind, 'point_ids': refs})
    return raw


class FittingTests(unittest.TestCase):
    def setUp(self):
        self.frame = FitFrame((1000, 600), (2048, 1400), (3071, 1999), (51, 71, 451, 311))
        self.ellipse = {'kind': 'ellipse', 'center': [303, 205], 'width': 188,
                        'height': 91, 'angle_radians': .37}

    def fit(self, raw):
        return fit_object(raw, self.frame, object_id='target', namespace='test/revision1')

    def test_rotated_near_circle_and_elongated_ellipses(self):
        for w, h, a in [(188, 91, .37), (120, 119.8, .51), (120, 120, 0), (260, 30, 1.13)]:
            with self.subTest(width=w, height=h):
                truth = {**self.ellipse, 'width': w, 'height': h, 'angle_radians': a}
                raw = add_primitive(empty_raw(), 'ellipse', contour(truth, 8))
                result = self.fit(raw)
                self.assertEqual(result['fit_status'], 'ok')
                self.assertEqual(result['completeness_status'], 'unreviewed')
                expected = contour(truth, 4000)
                actual = np.asarray(sample_commands(result['parts'][0]['commands'], 1024))
                error = max(cKDTree(expected).query(actual)[0].max(), cKDTree(actual).query(expected)[0].max())
                self.assertLess(error, .25)

    def test_rounded_rectangles_rotation_radius_and_noise(self):
        for radius, angle, noise in [(3, 0, 0), (18, .22, .25), (40, .8, 0)]:
            with self.subTest(radius=radius):
                xy = rr_points(radius=radius, angle=angle)
                xy += np.random.default_rng(731).normal(0, noise, xy.shape)
                raw = add_primitive(empty_raw(), 'rounded_rectangle', xy)
                before = copy.deepcopy(raw)
                result = self.fit(raw)
                self.assertEqual(raw, before)
                self.assertEqual(result['fit_status'], 'ok')
                self.assertLess(abs(result['parts'][0]['parameters']['radius']-radius), 1)
                self.assertEqual(result['parts'][0]['diagnostics']['count'], 16)

    def test_mixed_object_preserves_all_parts_without_parent_outline(self):
        raw = add_primitive(empty_raw(), 'rectangle', [[10, 10], [60, 10], [60, 40], [10, 40]])
        add_primitive(raw, 'ellipse', contour(self.ellipse, 8))
        add_primitive(raw, 'rounded_rectangle', rr_points())
        raw['points'] += [{'id': 'line_a', 'xy': [5, 300]}, {'id': 'line_b', 'xy': [80, 350]}]
        raw['parts'].append({'id': 'line', 'kind': 'path', 'closed': False,
                             'segments': [{'kind': 'line', 'point_ids': ['line_a', 'line_b']}]})
        before = copy.deepcopy(raw)
        result = self.fit(raw)
        self.assertEqual(result['part_counts'], {'observed': 4, 'succeeded': 4, 'failed': 0})
        self.assertEqual([p['id'] for p in result['parts']], [p['id'] for p in raw['parts']])
        self.assertNotIn('commands', result)
        self.assertEqual(raw, before)
        self.assertEqual(result['observation'], before)
        self.assertTrue(all(p['object_id'] == 'target' for p in result['parts']))
        self.assertEqual(result['parts'][0]['key'], ['test/revision1', 'target', 'part0'])

    def test_repeated_rectangles_keep_actual_widths(self):
        raw = empty_raw()
        widths = [19, 23, 31]
        for i, w in enumerate(widths):
            x = 10+i*60
            add_primitive(raw, 'rectangle', [[x, 10], [x+w, 10], [x+w, 40], [x, 40]])
        result = self.fit(raw)
        np.testing.assert_allclose([p['parameters']['width'] for p in result['parts']], widths, atol=1e-8)

    def test_multiple_ellipses_do_not_tie_radii(self):
        raw = empty_raw()
        for i, w in enumerate([90, 120, 160]):
            add_primitive(raw, 'ellipse', contour({**self.ellipse, 'width': w, 'center': [180+250*i, 250]}, 8))
        result = self.fit(raw)
        np.testing.assert_allclose([p['parameters']['width'] for p in result['parts']], [90, 120, 160], atol=.05)

    def test_shared_vertices_survive_and_fitted_disagreement_is_reported(self):
        raw = empty_raw()
        raw['points'] = [{'id': pid, 'xy': xy} for pid, xy in
                         [('a', [10, 10]), ('b', [30, 10]), ('c', [30, 30]),
                          ('d', [10, 30]), ('e', [54, 12]), ('f', [50, 30])]]
        raw['parts'] = [{'id': 'left', 'kind': 'rectangle', 'point_ids': ['a', 'b', 'c', 'd']},
                        {'id': 'right', 'kind': 'rectangle', 'point_ids': ['b', 'e', 'f', 'c']}]
        result = self.fit(raw)
        self.assertEqual(len(result['points']), 6)
        shared = result['shared_point_diagnostics']['shared_points']
        self.assertEqual({p['point_id'] for p in shared}, {'b', 'c'})
        self.assertGreater(max(p['max_separation_px'] for p in shared), .1)
        self.assertFalse(result['shared_point_diagnostics']['joint_constraints_applied'])

    def test_degenerate_part_does_not_discard_valid_sibling(self):
        raw = add_primitive(empty_raw(), 'ellipse', [[i+1, 2] for i in range(8)])
        add_primitive(raw, 'rectangle', [[10, 10], [40, 10], [40, 30], [10, 30]])
        result = self.fit(raw)
        self.assertEqual(result['fit_status'], 'partial')
        self.assertEqual(result['completeness_status'], 'incomplete')
        self.assertEqual([p['status'] for p in result['parts']], ['failed', 'ok'])
        self.assertEqual(result['parts'][0]['observed_part'], raw['parts'][0])
        self.assertEqual(result['bbox_2d'], [10., 10., 40., 30.])
        self.assertEqual(transform_fit(result)['parts'][0]['commands'], [])

    def test_optimizer_failure_is_preserved_without_fallback(self):
        raw = add_primitive(empty_raw(), 'ellipse', contour(self.ellipse, 8))
        failed = SimpleNamespace(success=False, status=0, message='budget exhausted',
                                 nfev=600, cost=1.0, x=np.array([0, 0, 1, .5, 0]))
        with patch('shape_integration.fitting.primitives.least_squares', return_value=failed):
            result = self.fit(raw)
        self.assertEqual(result['fit_status'], 'failed')
        self.assertEqual(result['parts'][0]['diagnostics']['solver']['status'], 0)
        self.assertIsNone(result['bbox_2d'])

    def test_partial_and_context_are_never_complete(self):
        raw = add_primitive(empty_raw(), 'rectangle', [[10, 10], [40, 10], [40, 30], [10, 30]])
        raw['status'] = 'partial'; raw['reason'] = 'A visible lower cell is not observed'
        result = self.fit(raw)
        self.assertEqual(result['fit_status'], 'partial')
        self.assertEqual(result['completeness_status'], 'incomplete')
        for status in ['needs_context', 'unsupported', 'partial']:
            raw = {**empty_raw(), 'status': status, 'reason': 'Target is clipped'}
            result = self.fit(raw)
            self.assertEqual(result['fit_status'], 'not_run')
            self.assertEqual(result['completeness_status'], 'incomplete')

    def test_contract_rejects_bad_counts_ids_coordinates_and_unused_points(self):
        valid = add_primitive(empty_raw(), 'ellipse', contour(self.ellipse, 8))
        cases = []
        raw = copy.deepcopy(valid); raw['parts'][0]['point_ids'].pop(); cases.append(raw)
        raw = copy.deepcopy(valid); raw['parts'][0]['kind'] = 'compound_symbol'; cases.append(raw)
        raw = copy.deepcopy(valid); raw['points'][0]['xy'][0] = float('nan'); cases.append(raw)
        raw = copy.deepcopy(valid); raw['points'][0]['xy'][0] = float('inf'); cases.append(raw)
        raw = copy.deepcopy(valid); raw['points'][0]['xy'][0] = 1000; cases.append(raw)
        raw = copy.deepcopy(valid); raw['object_id'] = 'other'; cases.append(raw)
        raw = copy.deepcopy(valid); raw['points'].append({'id': 'unused', 'xy': [1, 1]}); cases.append(raw)
        raw = copy.deepcopy(valid); raw['parts'][0]['point_ids'][0] = 'undefined'; cases.append(raw)
        raw = copy.deepcopy(valid); raw['parts'].append(copy.deepcopy(raw['parts'][0])); cases.append(raw)
        raw = copy.deepcopy(valid); raw['points'].append(copy.deepcopy(raw['points'][0])); cases.append(raw)
        for raw in cases:
            with self.subTest(raw=repr(raw)[:90]), self.assertRaises(ObservationError):
                self.fit(raw)
        with self.assertRaises(FitError):
            fit_boundary('circle', contour(self.ellipse, 8))
        with self.assertRaises(FitError):
            fit_boundary('rounded_rectangle', contour(self.ellipse, 8))

    def test_every_cubic_control_is_transformed_with_half_pixel_mapping(self):
        raw = add_primitive(empty_raw(), 'ellipse', contour(self.ellipse, 8))
        result = self.fit(raw); before = copy.deepcopy(result)
        mapped = transform_fit(result)
        self.assertEqual(result, before)
        for a, b in zip(result['parts'][0]['commands'], mapped['parts'][0]['commands']):
            for i in range(1, len(a), 2):
                np.testing.assert_allclose(self.frame.to_local(b[i:i+2]), a[i:i+2], atol=1e-10)
        expected = [(51+(12.3+.5)*400/1000)*3071/2048-.5,
                    (71+(45.6+.5)*240/600)*1999/1400-.5]
        np.testing.assert_allclose(self.frame.to_original([12.3, 45.6]), expected, atol=1e-10)
        self.assertNotIn('parameters', mapped['parts'][0])
        self.assertEqual(mapped['parts'][0]['parameters_local'], result['parts'][0]['parameters'])
        with self.assertRaises(ValueError):
            transform_fit(mapped)
        json.dumps(mapped, allow_nan=False)

    def test_native_crop_metadata_and_inconsistent_frames(self):
        meta = self.frame.metadata()
        meta['source_crop_half_open'] = [v*self.frame.original_size[i % 2]/self.frame.global_size[i % 2]
                                         for i, v in enumerate(self.frame.crop_bounds)]
        self.assertEqual(FitFrame.from_metadata(meta), self.frame)
        meta['source_crop_half_open'][0] += 1
        with self.assertRaises(ValueError): FitFrame.from_metadata(meta)
        with self.assertRaises(ValueError): FitFrame((0, 3), (20, 20), (20, 20), (0, 0, 10, 10))

    def test_closed_curve_and_bad_path_connectivity(self):
        raw = empty_raw()
        raw['points'] = [{'id': f'p{i}', 'xy': xy} for i, xy in
                         enumerate([[10, 10], [50, 10], [50, 50], [10, 50]])]
        raw['parts'] = [{'id': 'loop', 'kind': 'path', 'closed': True,
                         'segments': [{'kind': 'curve', 'point_ids': ['p0', 'p1', 'p2', 'p3', 'p0']}]}]
        result = self.fit(raw)
        self.assertEqual(result['fit_status'], 'ok')
        self.assertEqual(result['parts'][0]['commands'][-1], ['Z'])
        raw['parts'][0]['segments'][0]['point_ids'].pop()
        with self.assertRaises(ObservationError): self.fit(raw)


if __name__ == '__main__':
    unittest.main(verbosity=2)
