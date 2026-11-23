import copy
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from pipeline_frames import Crop
from region_partition import process_region

from region_partition import spatial_tiles, in_core, can_commit, apply_replacement, validate_plan, nearby_existing, seam_pairs, coverage_neighbors


class CompletenessFirst(unittest.TestCase):
    def setUp(self):
        self.parent = {'id': 'parent', 'kind': 'dense_region', 'bbox_native': [0, 0, 100, 100]}
        self.a = {'id': 'a', 'kind': 'single', 'bbox_native': [5, 5, 25, 25]}
        self.b = {'id': 'b', 'kind': 'single', 'bbox_native': [40, 40, 60, 60]}
        self.base = [self.parent, self.a, self.b]
        self.children = [{'id': 'proposal-a', 'existing_id': 'a'}, {'id': 'proposal-b', 'existing_id': 'b'}]
        self.inventory = [{'answer': {'decision': 'replace', 'complete_inventory': True}, 'partial': [], 'objects': self.children}]
        self.passed = [{'complete': True, 'duplicates': False, 'missing': [], 'problem_ids': []}]

    def test_uncovered_second_arrow_keeps_parent_and_does_not_publish_partial_split(self):
        failed = [{'complete': False, 'duplicates': False, 'missing': ['second arrow'], 'problem_ids': []}]
        commit = can_commit({'action': 'split_routes'}, self.inventory, failed, [])
        self.assertFalse(commit)
        active, owners = apply_replacement(self.base, 'parent', self.children[:1], commit)
        self.assertEqual(active, self.base)
        self.assertEqual(owners, [])

    def test_complete_existing_children_retire_only_redundant_parent(self):
        snapshot = copy.deepcopy(self.base)
        self.assertTrue(can_commit({'action': 'split_routes'}, self.inventory, self.passed, []))
        active, owners = apply_replacement(self.base, 'parent', self.children, True)
        self.assertEqual(active, [self.a, self.b])
        self.assertEqual(owners, ['a', 'b'])
        self.assertEqual(self.base, snapshot)

    def test_conflict_or_missing_review_blocks_retirement(self):
        self.assertFalse(can_commit({'action': 'split_routes'}, self.inventory, self.passed, [{'duplicate': True}]))
        self.assertFalse(can_commit({'action': 'split_routes'}, self.inventory, [], []))

    def test_hard_group_cannot_be_replaced_by_new_children(self):
        inv = copy.deepcopy(self.inventory)
        inv[0]['objects'][0]['existing_id'] = None
        self.assertFalse(can_commit({'action': 'keep_group'}, inv, self.passed, []))
        self.assertTrue(can_commit({'action': 'keep_group'}, self.inventory, self.passed, []))

    def test_parent_cannot_reference_itself_or_missing_owner(self):
        with self.assertRaises(ValueError):
            apply_replacement(self.base, 'parent', [{'id': 'p', 'existing_id': 'parent'}], True)
        with self.assertRaises(ValueError):
            apply_replacement(self.base, 'parent', [{'id': 'p', 'existing_id': 'absent'}], True)

    def test_overlapping_windows_have_exactly_one_core_owner(self):
        specs = spatial_tiles([10, 15, 209, 324], (300, 400))
        for x in range(10, 210):
            for y in range(15, 325):
                b = [x-.1, y-.1, x+.1, y+.1]
                self.assertEqual(sum(in_core(b, s['core']) for s in specs), 1)
        self.assertGreater(specs[0]['crop'][2], specs[0]['core'][2])

    def test_unresolved_parents_never_become_each_others_owners(self):
        other = {**self.parent, 'id': 'other-parent'}
        selected, _ = nearby_existing(self.parent, self.base+[other], 4)
        self.assertEqual({o['id'] for o in selected}, {'a', 'b'})

    def test_bbox_hardness_cannot_request_forced_split(self):
        with self.assertRaises(ValueError):
            validate_plan({'action': 'split_routes', 'bbox_separability': 'hard', 'target_scope': 'curves', 'reason': 'overlap'})

    def test_shifted_same_arrow_across_seam_is_reviewed_despite_different_core_centers(self):
        a={'id':'a','source_tile':'left','bbox_native':[9.8,10,20.0,20]}
        b={'id':'b','source_tile':'right','bbox_native':[10.2,10,20.2,20]}
        self.assertEqual(seam_pairs([a,b]),[(a,b)])
        self.assertFalse(seam_pairs([a,{**b,'source_tile':'left'}]))

    def test_inventory_uncertainty_requires_independent_local_verification(self):
        inv=copy.deepcopy(self.inventory);inv[0]['answer']['complete_inventory']=False
        self.assertFalse(can_commit({'action':'split_routes'},inv,[],[]))
        self.assertTrue(can_commit({'action':'split_routes'},inv,self.passed,[]))
        failed=[{'complete':False,'duplicates':False,'missing':['unowned arrow'],'problem_ids':[]}]
        self.assertFalse(can_commit({'action':'split_routes'},inv,failed,[]))

    def test_neighbor_owner_is_visible_without_becoming_another_primary(self):
        a={'id':'a','bbox_native':[10,10,20,20]}
        b={'id':'b','bbox_native':[49,10,60,20]}
        far={'id':'far','bbox_native':[80,80,90,90]}
        primary=[a];snapshot=copy.deepcopy(primary)
        selected=coverage_neighbors(primary,[a,b,far],[0,0,50,50])
        self.assertEqual(selected,[b]);self.assertEqual(primary,snapshot)

    def test_hard_bbox_plan_terminates_without_forced_inventory(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);inp=root/'inputs/demo';(inp/'G').mkdir(parents=True)
            im=Image.new('RGB',(100,100),'white');im.save(inp/'source.png');im.save(inp/'G/target.png')
            view=Crop((100,100),(0,0,100,100),(100,100))
            (inp/'views.json').write_text(json.dumps({'views':[{'transform':view.metadata((100,100))}]}))
            parent={**self.parent,'display_id':'I0001','locator':'crossing curves','extent':'complete'}
            plan={'action':'keep_group','bbox_separability':'hard','target_scope':'curves','reason':'overlap'}
            with patch('region_partition.plan_region',return_value=plan), patch('region_partition.inventory',side_effect=AssertionError('No forced enumeration')) as call:
                result=process_region(root,'demo',parent,[parent],{},None)
            call.assert_not_called()
            self.assertFalse(result['commit']);self.assertEqual(result['children'],[])
            self.assertEqual(result['parent_role'],'active_reconstruction_group')


if __name__ == '__main__': unittest.main()
