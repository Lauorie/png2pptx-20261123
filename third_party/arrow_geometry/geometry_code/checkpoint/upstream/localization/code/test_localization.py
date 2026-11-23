import copy
import math
import random
import tempfile
import unittest
from pathlib import Path

from PIL import Image
import contracts as C
from frames import global_frame, local_views, crop_from_global, native_box, visible_global_bounds
from handoff import export
from transport.client import build_payload


def region(i="A", box=None, kind="linework"):
    return {"id": i, "bbox": box or [20, 20, 120, 120], "render_type": kind,
            "grouping": "single", "label": "visible connector", "extent": "complete"}


class CoordinateContract(unittest.TestCase):
    def test_roundtrip_noninteger_aspect_and_crop(self):
        rng = random.Random(26)
        worst = 0
        for _ in range(500):
            size = (rng.randint(2, 4001), rng.randint(2, 3001))
            view = global_frame(size)
            for xy in ([0, 0], [size[0]-1, size[1]-1], [rng.random()*(size[0]-1), rng.random()*(size[1]-1)]):
                mapped = view.to_global(view.to_local(xy))
                worst = max(worst, *(abs(mapped[k]-xy[k]) for k in (0, 1)))
            for _, local in local_views(size):
                for xy in ([0, 0], [local.input_size[0]-1, local.input_size[1]-1]):
                    mapped = local.to_local(local.to_global(xy))
                    worst = max(worst, *(abs(mapped[k]-xy[k]) for k in (0, 1)))
        self.assertLess(worst, 1e-9)

    def test_known_half_pixel_mapping_independent_formula(self):
        v = global_frame((997, 331))
        for point in ([0, 0], [2047, v.input_size[1]-1], [509.125, 149.5]):
            out = v.to_global(point)
            self.assertAlmostEqual(out[0], (point[0]+.5)*997/2048-.5)
            self.assertAlmostEqual(out[1], (point[1]+.5)*331/v.input_size[1]-.5)

    def test_boundary_intersection_is_explicit_not_raw_mutation(self):
        v = global_frame((300, 101))
        raw = [0, 0, v.input_size[0]-1, v.input_size[1]-1]
        audit = native_box(raw, v)
        self.assertEqual(audit["raw_bbox_global_px"], raw)
        self.assertLess(audit["mapped_native_unclipped"][0], 0)
        self.assertEqual(audit["bbox_native"], [0, 0, 299, 100])
        self.assertTrue(audit["boundary_intersection_applied"])

    def test_invalid_coordinates_never_silently_fixed(self):
        for box in ([-1, 0, 50, 80], [0, 0, 200, 199], [0, 0, float("nan"), 40],
                    [False, 0, 20, 20], [10, 10, 9, 90], ["0", 0, 10, 20]):
            with self.assertRaises(ValueError):
                C.number_box(box, [200, 200])

    def test_local_images_and_full_image_boundaries(self):
        for size in ((701, 203), (101, 2001), (2048, 2048)):
            im = Image.new("RGB", size)
            g = global_frame(size)
            for _, v in local_views(size):
                self.assertEqual(v.image(im).size, v.input_size)
                b = visible_global_bounds(v, g)
                C.number_box(b, g.input_size)
            b = [0, 0, g.input_size[0]-1, g.input_size[1]-1]
            c = crop_from_global(b, g)
            self.assertEqual(c.box, (0, 0, *size))
            self.assertEqual(c.image(im).size, c.input_size)

    def test_selected_inspection_crop_not_automatically_whole_figure(self):
        g = global_frame((2048, 1000))
        c = crop_from_global([600, 200, 800, 400], g)
        self.assertEqual(c.box, (600, 200, 801, 401))


class RegionProtocol(unittest.TestCase):
    def test_atomic_split_does_not_retain_parent(self):
        old = [region("A"), region("B", [200, 200, 250, 250])]
        answer = {"frame": "GLOBAL", "keep_ids": ["B"], "replacements": [
            {"source_ids": ["A"], "regions": [region("x", [20, 20, 50, 50]), region("y", [70, 70, 120, 120])],
             "reason": "two visible separable arrows"}], "discard": [], "inspect": []}
        C.review_valid(answer, [300, 300], old)
        result = C.apply_review(old, answer, "V")
        self.assertEqual(len(result), 3)
        self.assertNotIn("A", [r["id"] for r in result])
        self.assertEqual([r["observation_ids"] for r in result if r["id"] != "B"], [["A"], ["A"]])

    def test_every_observation_accounted_and_disjoint_detail(self):
        old = [region()]
        base = {"frame": "GLOBAL", "keep_ids": [], "replacements": [], "discard": [], "inspect": []}
        with self.assertRaises(ValueError): C.review_valid(base, [200, 200], old)
        base["keep_ids"] = ["A"]
        base["inspect"] = [{"refs": ["A"], "bbox": [0, 0, 180, 180], "question": "extent"}]*2
        with self.assertRaises(ValueError): C.review_valid(base, [200, 200], old)

    def test_same_frame_required_even_for_empty_reply(self):
        for frame in ("LOCAL", "normalized", None):
            with self.assertRaises(ValueError):
                C.discovery_valid({"frame": frame, "regions": []}, [200, 200])

    def test_no_containment_or_touching_rule(self):
        rows = [region("big", [0, 0, 180, 180]), region("inside", [20, 20, 40, 40]),
                region("boundary", [0, 50, 40, 51])]
        merged = C.merge_near_identical(rows)
        self.assertEqual(len(merged), 3)
        self.assertEqual({s for r in merged for s in r["observation_ids"]}, {"big", "inside", "boundary"})

    def test_high_iou_is_not_transitive_or_cross_representation(self):
        rows = [region("A", [0, 0, 100, 100]), region("B", [2, 0, 102, 100]),
                region("C", [4, 0, 104, 100]), region("D", [0, 0, 100, 100], "block_arrow")]
        merged = C.merge_near_identical(rows, .95)
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[0]["bbox"], [0, 0, 102, 100])
        self.assertEqual(merged[0]["observation_ids"], ["A", "B"])

    def test_empty_replacement_cannot_hide_removal(self):
        answer = {"frame": "GLOBAL", "keep_ids": [], "replacements": [
            {"source_ids": ["A"], "regions": [], "reason": "unknown"}], "discard": [], "inspect": []}
        with self.assertRaises(ValueError): C.review_valid(answer, [200, 200], [region()])

    def test_handoff_preserves_two_block_routes_and_raw_coordinates(self):
        rows = [region(), region("B", kind="block_arrow")]
        view = global_frame((200, 100))
        h, audit = export("case", rows, view, Path("/source.png"))
        self.assertEqual(h["tasks"][0]["geometry_route"], "centerline")
        self.assertEqual(h["tasks"][1]["geometry_route"], "closed_shape")
        self.assertEqual(h["tasks"][1]["block_recovery"]["modes"], ["glm_closed_shape", "crop_sam_mask_contour"])
        self.assertEqual(h["tasks"][0]["readonly_identities"], [])
        self.assertEqual(audit[0]["raw_bbox_global_px"], rows[0]["bbox"])

    def test_wire_preserves_default_thinking(self):
        config = {"model": "glm-5.3-flash", "temperature": .1, "max_tokens": 113000, "thinking": {"type": "enabled"}}
        payload = build_payload(config, C.SYSTEM, C.discovery_prompt([2048, 700]), [])
        self.assertNotIn("reasoning_effort", payload)
        self.assertEqual(payload["max_tokens"], 113000)
        self.assertEqual(payload["thinking"], {"type": "enabled"})

    def test_uncertain_handoff_defers_both_types_without_losing_coordinates(self):
        rows = [region('A'), region('B', kind='block_arrow'), region('C'), region('D')]
        rows[1].update(extent='uncertain', observation_ids=['G_R2', 'L1_R3'])
        rows[3].update(extent='uncertain', observation_ids=['G_R4'])
        before = copy.deepcopy(rows)
        view = global_frame((200, 100))
        handoff, audit = export('case', rows, view, Path('/source.png'), defer_uncertain=True)
        self.assertEqual([t['target_id'] for t in handoff['tasks']], ['R0001', 'R0003'])
        deferred = handoff['deferred_candidates']
        self.assertEqual([t['target_id'] for t in deferred], ['R0002', 'R0004'])
        self.assertEqual([t['render_type'] for t in deferred], ['block_arrow', 'linework'])
        for task, index in zip(deferred, [1, 3]):
            self.assertFalse(task['render_owner'])
            self.assertEqual(task['task_role'], 'missing_candidate')
            self.assertEqual(task['bbox_native'], audit[index]['bbox_native'])
            self.assertEqual(task['bbox_global_input_px'], rows[index]['bbox'])
            self.assertEqual(task['observation_ids'], rows[index]['observation_ids'])
        self.assertEqual(len(audit), len(rows))
        self.assertEqual(rows, before)
        legacy, _ = export('case', rows, view, Path('/source.png'))
        self.assertEqual(len(legacy['tasks']), 4)
        self.assertEqual(legacy['deferred_candidates'], [])

    def test_local_discovery_has_no_upstream_grouping_input(self):
        prompt = C.local_prompt([2048, 700], [0, 0, 1100, 400])
        self.assertNotIn("prior_observations", prompt)
        self.assertNotIn("candidates", prompt)
        self.assertEqual(prompt["global_size_px"], [2048, 700])
        # Both discovery branches use exactly the same decision policy.
        self.assertEqual(prompt["rules"], C.discovery_prompt([2048, 700])["rules"])


if __name__ == "__main__":
    unittest.main()
