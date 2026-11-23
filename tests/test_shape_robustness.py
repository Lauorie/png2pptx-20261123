"""Part-preserving repairs, independent failures and sparse-curve regressions."""
import copy
import json
import unittest

import numpy as np

import test_pipeline as fixtures
from test_pipeline import FakeRuntime, reviewed
from shape_integration.common import read
from shape_integration.fitting.adapter import fit_object
from shape_integration.fitting.contracts import normalize_observation, ObservationError
from shape_integration.fitting.frames import FitFrame
from shape_integration.fitting.paths import curve_commands, sample_commands, command_strokes


def blocks():
    points, parts = [], []
    for name, x, y in [('panel', 100, 100), ('cell7', 300, 300), ('cell8', 500, 300)]:
        refs = [name + str(i) for i in range(4)]
        points.extend({'id': pid, 'xy': xy} for pid, xy in zip(refs,
            [[x, y], [x+80, y], [x+80, y+60], [x, y+60]]))
        parts.append({'id': name, 'kind': 'rectangle', 'point_ids': refs})
    return {'status': 'ok', 'points': points, 'parts': parts}


def edit(raw, ids):
    raw = copy.deepcopy(raw)
    raw['parts'] = [p for p in raw['parts'] if p['id'] in ids]
    refs = {pid for p in raw['parts'] for pid in p['point_ids']}
    raw['points'] = [p for p in raw['points'] if p['id'] in refs]
    return raw


class RobustnessTests(unittest.TestCase):
    setUp = fixtures.PipelineTests.setUp
    run_object = fixtures.PipelineTests.run_object

    def test_local_edit_preserves_unrequested_geometry_and_forbids_new_glyph(self):
        self.options['max_revisions'] = 1
        def responder(tag, prompt, count):
            if tag == 'observe':
                raw = blocks()
                if 'revision_task' in prompt:
                    self.assertEqual(prompt['revision_task']['edit_part_ids'], ['cell7'])
                    self.assertNotIn('ALL target parts', prompt['revision_policy'])
                    for point in raw['points']:
                        point['xy'][0] += 20
                    # Simulate the old failure: redraw a sibling, omit the
                    # background, and introduce an unrelated text path.
                    raw = edit(raw, ['cell7', 'cell8'])
                    raw['parts'].append({'id': 'letter', 'kind': 'path', 'closed': False,
                        'segments': [{'kind': 'line', 'point_ids': ['cell70', 'cell72']}]})
                return raw
            if count == 2:
                return {'decision': 'revise', 'issues': [{'kind': 'geometry', 'parts': ['F2'],
                    'evidence': 'The second cell is shifted left.', 'action': 'Align this cell with its source outline.'}]}
            self.assertIn('revision_context', prompt)
            return reviewed(prompt)
        result = self.run_object(FakeRuntime(self.config, responder))
        self.assertEqual(result['state'], 'accepted_by_model')
        before = read(self.folder/'objects/G1/revision_00/fit_local.json')
        after = read(self.folder/'objects/G1/revision_01/fit_local.json')
        instruction = read(self.folder/'objects/G1/revision_01/prompt.json')
        self.assertIn('Edit only the explicitly authorized', instruction['system'])
        a, b = [{p['id']: p for p in fit['parts']} for fit in [before, after]]
        self.assertEqual(set(b), set(a))
        for pid in ['panel', 'cell8']:
            self.assertEqual(a[pid]['commands'], b[pid]['commands'])
            self.assertEqual(a[pid]['parameters'], b[pid]['parameters'])
        self.assertNotEqual(a['cell7']['commands'], b['cell7']['commands'])
        audit = read(self.folder/'objects/G1/revision_01/observation_normalization.json')['revision_merge']
        self.assertIn('letter', audit['ignored_unrequested_ids'])

    def test_missing_edit_id_keeps_previous_geometry_as_incomplete(self):
        def responder(tag, prompt, count):
            if tag == 'observe':
                return blocks() if count == 1 else edit(blocks(), ['cell8'])
            if count == 2:
                return {'decision': 'revise', 'issues': [{'kind': 'geometry', 'parts': ['F2'],
                    'evidence': 'The cell is shifted.', 'action': 'Move the affected cell.'}]}
            return reviewed(prompt)
        self.options['max_revisions'] = 1
        result = self.run_object(FakeRuntime(self.config, responder))
        self.assertEqual(result['state'], 'unresolved')
        after = read(self.folder/'objects/G1/revision_01/fit_local.json')
        self.assertEqual(after['part_counts']['succeeded'], 3)
        self.assertEqual(after['observation_status'], 'partial')

    def test_extra_part_removal_keeps_siblings(self):
        def responder(tag, prompt, count):
            if tag == 'observe':
                if count == 1:
                    return blocks()
                self.assertEqual(prompt['revision_task']['remove_part_ids'], ['cell7'])
                return {'status': 'ok', 'points': [], 'parts': []}
            if count == 2:
                return {'decision': 'revise', 'issues': [{'kind': 'extra', 'parts': ['F2'],
                    'evidence': 'This rectangle is not a source shape.', 'action': 'Remove the extra rectangle.'}]}
            return reviewed(prompt)
        result = self.run_object(FakeRuntime(self.config, responder))
        self.assertEqual(result['state'], 'accepted_by_model')
        fit = read(self.folder/'objects/G1/revision_01/fit_local.json')
        self.assertEqual([p['id'] for p in fit['parts']], ['panel', 'cell8'])

    def test_shared_vertices_expand_the_edit_group(self):
        from shape_integration.fitting.revision import plan_revision
        self.run_object(FakeRuntime(self.config))
        p = self.folder/'objects/G1/revision_00'
        plan = plan_revision(p, {'issues': [{'kind': 'geometry', 'parts': ['F1'], 'part_ids': ['a'],
                                            'evidence': 'shared edge misplaced', 'action': 'move edge'}]})
        self.assertEqual(plan['editable_ids'], ['a', 'b'])
        self.assertEqual(plan['preserved_ids'], [])

    def test_failed_numeric_replacement_keeps_the_previous_complete_revision(self):
        def responder(tag, prompt, count):
            if tag == 'observe':
                if count == 1:
                    return blocks()
                raw = edit(blocks(), ['cell7'])
                for point in raw['points']:
                    point['xy'][1] = 300
                return raw
            return {'decision': 'revise', 'issues': [{'kind': 'geometry', 'parts': ['F2'],
                'evidence': 'The second cell is shifted.', 'action': 'Align the second cell.'}]}
        result = self.run_object(FakeRuntime(self.config, responder))
        self.assertEqual(result['state'], 'unresolved')
        self.assertEqual(result['selected_revision'], 'revision_00')
        self.assertEqual(result['stop_reason'], 'revision_fit_failed')
        self.assertEqual(result['history'][1]['observation']['failed_replacements'], ['cell7'])

    def test_bad_part_is_omitted_without_discarding_valid_siblings(self):
        raw = blocks(); raw['parts'][1]['point_ids'][0] = 'missing'
        audit = {}
        normalized = normalize_observation(raw, 'G1', (1000, 1000), recover_parts=True, audit=audit)
        self.assertEqual(normalized['status'], 'partial')
        self.assertEqual([p['id'] for p in normalized['parts']], ['panel', 'cell8'])
        self.assertEqual(audit['rejected_parts'][0]['id'], 'cell7')
        frame = FitFrame((1000, 1000), (1000, 1000), (1000, 1000), (0, 0, 1000, 1000))
        fit = fit_object(normalized, frame, object_id='G1', namespace='recovery')
        self.assertEqual(fit['part_counts']['succeeded'], 2)
        self.assertEqual(fit['fit_status'], 'partial')
        self.assertNotIn('cell70', [p['id'] for p in normalized['points']])

    def test_ambiguous_point_table_and_wrong_identity_still_fail(self):
        for change in ['duplicate', 'wrong_identity', 'nonfinite']:
            raw = blocks()
            if change == 'duplicate':
                raw['points'].append(copy.deepcopy(raw['points'][0]))
            elif change == 'wrong_identity':
                raw['object_id'] = 'wrong'
            else:
                raw['points'][0]['xy'][0] = float('nan')
            with self.subTest(change=change), self.assertRaises(ObservationError):
                normalize_observation(raw, 'G1', (1000, 1000), recover_parts=True)

    def test_branching_open_path_keeps_strokes_without_inventing_connector(self):
        raw = {'status': 'ok', 'points': [{'id': p, 'xy': xy} for p, xy in
               [('tail', [10, 30]), ('tip', [30, 30]), ('left', [22, 22]), ('right', [22, 38])]],
               'parts': [{'id': 'branch', 'kind': 'path', 'closed': False, 'segments': [
                   {'kind': 'line', 'point_ids': ['tail', 'tip']},
                   {'kind': 'line', 'point_ids': ['tip', 'left']},
                   {'kind': 'line', 'point_ids': ['tip', 'right']}]}]}
        norm = normalize_observation(raw, 'G1', (100, 100))
        frame = FitFrame((100, 100), (100, 100), (100, 100), (0, 0, 100, 100))
        fit = fit_object(norm, frame, object_id='G1', namespace='branch')
        commands = fit['parts'][0]['commands']
        self.assertEqual([c[0] for c in commands], ['M', 'L', 'L', 'M', 'L'])
        strokes = list(command_strokes(commands))
        self.assertEqual(strokes, [[[10., 30.], [30., 30.], [22., 22.]], [[30., 30.], [22., 38.]]])
        raw['parts'][0]['closed'] = True
        with self.assertRaisesRegex(ObservationError, 'part branch'):
            normalize_observation(raw, 'G1', (100, 100))

    def test_open_curve_has_no_unobserved_coordinate_extrema(self):
        # Uneven spacing creates the same instability as a short hook attached
        # to a long span, without depending on a specific diagram or crop.
        for points in [[[0, 0], [2, 10], [700, 10]],
                       [[0, 40], [3, 30], [500, 30], [505, 40]],
                       [[0, 20], [30, 0], [100, 0], [150, 20]]]:
            p = np.asarray(points, float)
            commands = curve_commands(p)
            for i, cmd in enumerate(commands[1:]):
                curve = np.asarray(sample_commands([['M', *p[i]], cmd], 128))
                self.assertTrue(np.all(curve >= np.minimum(p[i], p[i+1])-1e-9))
                self.assertTrue(np.all(curve <= np.maximum(p[i], p[i+1])+1e-9))
                np.testing.assert_allclose(curve[-1], p[i+1])
        # Periodic closed contours retain their existing smooth periodic rule.
        commands = curve_commands([[0, 0], [40, 0], [40, 40], [0, 40], [0, 0]], periodic=True)
        np.testing.assert_allclose(commands[-1][-2:], commands[0][1:])


if __name__ == '__main__':
    unittest.main()
