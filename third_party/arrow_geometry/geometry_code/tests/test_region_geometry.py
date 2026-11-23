import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'lib')]
import region_geometry as rg


def sample():
    return dict(status='ready', points=dict(a=[10., 10.], b=[90., 90.],
                                           c=[10., 90.], d=[90., 10.]),
                paths=[dict(id='diagonal1', segments=[dict(kind='line', points=['a', 'b'])],
                            start_head=True, end_head=False),
                       dict(id='diagonal2', segments=[dict(kind='line', points=['c', 'd'])],
                            start_head=False, end_head=False)], uncertain=[])


class RegionGeometryTests(unittest.TestCase):
    def test_crossings_stay_independent_and_head_order_survives(self):
        value = sample()
        before = copy.deepcopy(value)
        rows = rg.compile_rows(rg.validate(value, [100, 100]), 'sample')
        self.assertEqual(value, before)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['start'], [10., 10.])
        self.assertEqual(rows[0]['end'], [90., 90.])
        self.assertEqual(rows[0]['head_presence'], [True, False])
        self.assertEqual(rows[1]['head_presence'], [False, False])
        self.assertFalse(any(r['shared_nodes'] for r in rows))

    def test_polyline_preserves_corner_without_connector_between_paths(self):
        value = sample()
        value['points']['k'] = [90., 10.]
        value['paths'][0]['segments'][0]['points'] = ['a', 'k', 'b']
        rows = rg.compile_rows(rg.validate(value, [100, 100]), 'sample')
        self.assertEqual(rows[0]['kind'], 'polyline')
        self.assertEqual(rows[0]['via'], [[90., 10.]])
        self.assertEqual(rows[0]['beziers'], [])
        self.assertEqual(rows[1]['start'], [10., 90.])

    def test_invalid_coordinates_or_references_fail_without_repair(self):
        for key, xy in [('a', [float('nan'), 0]), ('a', [100, 10])]:
            value = sample()
            value['points'][key] = xy
            with self.assertRaises(ValueError):
                rg.validate(value, [100, 100])
        value = sample()
        value['paths'][0]['segments'][0]['points'][1] = 'missing'
        with self.assertRaises(ValueError):
            rg.validate(value, [100, 100])

    def test_nonready_does_not_silently_drop_returned_paths(self):
        value = sample()
        value.update(status='not_arrow', reason='Shape only')
        with self.assertRaises(ValueError):
            rg.validate(value, [100, 100])
        empty = rg.validate(dict(status='not_arrow', reason='Shape only', points={}, paths=[]), [100, 100])
        self.assertEqual(rg.compile_rows(empty, 'sample'), [])

    def test_disconnected_segments_must_use_separate_paths(self):
        value = sample()
        value['paths'][0]['segments'].append(dict(kind='line', points=['c', 'd']))
        with self.assertRaises(ValueError):
            rg.validate(value, [100, 100])


if __name__ == '__main__':
    unittest.main()
