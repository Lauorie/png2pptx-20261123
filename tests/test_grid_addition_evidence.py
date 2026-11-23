"""Source evidence must distinguish a hallucinated divider from visible ink."""
import copy
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageDraw

from shape_integration.fitting.geometry_guard import straight_endpoints, unsupported_grid_additions


class GridAdditionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / 'source.png'
        outline = [['M', 10, 10], ['L', 310, 10], ['L', 310, 250], ['L', 10, 250], ['Z']]
        def part(ident, commands):
            return {'id': ident, 'status': 'ok', 'kind': 'path', 'commands': commands}
        self.before = {'coordinate_frame': 'original_pixel_centers', 'bbox_2d': [10, 10, 310, 250],
                       'parts': [part('frame', outline)] + [
                           part(str(x), [['M', x, 10], ['L', x, 250]]) for x in (80, 160)]}
        self.after = copy.deepcopy(self.before)
        self.after['parts'].append(part('new', [['M', 240, 10], ['L', 240, 250]]))

    def draw(self, added=None, weak_witness=False):
        im = Image.new('RGB', (330, 270), 'white')
        draw = ImageDraw.Draw(im)
        draw.rectangle((10, 10, 310, 250), outline='black', width=2)
        for x in (80, 160, 240):
            mode = added if x == 240 else 'sparse' if weak_witness else 'solid'
            if mode == 'solid':
                draw.line((x, 10, x, 250), fill='black', width=2)
            elif mode == 'dashed':
                for y in range(10, 250, 14):
                    draw.line((x, y, x, min(y + 7, 250)), fill='black', width=2)
            elif mode == 'occluded':
                for lo, hi in ((10, 80), (200, 250)):
                    draw.line((x, lo, x, hi), fill='black', width=2)
            elif mode == 'sparse':
                for y in range(10, 250, 40):
                    draw.line((x, y, x, y + 2), fill='black', width=1)
        im.save(self.source)

    def check(self):
        return unsupported_grid_additions(self.before, self.after, self.source)

    def test_absent_divider_is_vetoed_with_visible_parallel_witnesses(self):
        self.draw()
        self.assertEqual(self.check()['decision'], 'rollback')

    def test_segmented_parallel_witnesses_still_veto_an_absent_divider(self):
        self.draw()
        for p in self.before['parts'][1:]:
            x = p['commands'][0][1]
            p['commands'] = [['M', x, 10]] + [['L', x, y] for y in (70, 130, 190, 250)]
        self.assertEqual(self.check()['decision'], 'rollback')

    def test_straight_detection_preserves_bends_subpaths_and_retracing(self):
        for commands in (
            [['M', 0, 0], ['L', 10, 1], ['L', 20, 0]],
            [['M', 0, 0], ['L', 10, 0], ['M', 15, 0], ['L', 20, 0]],
            [['M', 0, 0], ['L', 15, 0], ['L', 10, 0], ['L', 20, 0]],
            [['M', 0, 0], ['Q', 10, 0, 20, 0]],
            [['M', 0, 0], ['L', 20, 0], ['Z']],
        ):
            with self.subTest(commands=commands):
                self.assertIsNone(straight_endpoints(commands))

    def test_straight_detection_tolerates_rounded_diagonal_coordinates(self):
        points = straight_endpoints([
            ['M', 1296.916, 13.882], ['L', 1186.481, 80.530],
            ['L', 1076.047, 147.178], ['L', 965.612, 213.825], ['L', 855.178, 280.473]])
        self.assertIsNotNone(points)
        self.assertEqual(points.shape, (2, 2))

    def test_visible_dashed_and_partly_occluded_dividers_are_not_vetoed(self):
        for mode in ('solid', 'dashed', 'occluded'):
            with self.subTest(mode=mode):
                self.draw(mode)
                self.assertEqual(self.check()['decision'], 'abstain')

    def test_sparse_witnesses_cannot_certify_an_absent_divider(self):
        self.draw(weak_witness=True)
        self.assertEqual(self.check()['decision'], 'abstain')

    def test_missing_image_or_different_coordinate_frame_abstains(self):
        self.assertIsNone(self.check())
        self.draw()
        self.after['coordinate_frame'] = 'crop'
        self.assertIsNone(self.check())

    def test_renaming_and_reversing_a_part_is_not_an_addition(self):
        self.draw()
        self.after['parts'][-1]['commands'] = [['M', 80, 250], ['L', 80, 10]]
        self.assertEqual(self.check()['decision'], 'abstain')

    def test_outer_edge_is_not_an_internal_divider(self):
        self.draw()
        self.after['parts'][-1]['commands'] = [['M', 10, 10], ['L', 10, 250]]
        self.assertEqual(self.check()['decision'], 'abstain')

    def test_unrelated_external_witnesses_cannot_veto_a_line(self):
        self.draw()
        for p in self.before['parts'][1:]:
            p['commands'][0][2] = -100
        self.assertEqual(self.check()['decision'], 'abstain')
