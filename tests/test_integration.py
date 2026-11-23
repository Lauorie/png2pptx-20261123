import copy
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'vendor')]
from shape_integration.common import read,sha
from shape_integration.dedup import filter_candidates,intersection_over_union
from shape_integration.shape import apply_review,assign,validate_plan
from shape_integration.local_contract import prompt_for,validator
from shape_integration.image_bridge import detect
from png2pptx.frames import Crop

class Contracts(unittest.TestCase):
    def test_strict_iou_and_containment(self):
        s=[{'id':'S','bbox_2d':[0,0,100,100]}]
        exact=[{'id':'I','bbox_2d':[0,0,85,100]}]
        self.assertEqual(filter_candidates(s,exact,.85)['removed'],[])
        self.assertEqual(len(filter_candidates(s,[{'id':'I','bbox_2d':[0,0,86,100]}],.85)['removed']),1)
        self.assertEqual(intersection_over_union([0,0,10,10],[0,0,100,100]),.01)
        self.assertEqual(intersection_over_union([0,0,10,10],[10,0,20,10]),0)
        self.assertEqual(filter_candidates(s,[])['kept'],s)

    def test_review_transactions_with_nontrivial_frame(self):
        frame=Crop((100,100),(20,30,60,70),(80,80))
        old=[{'id':x,'bbox_2d':[40,60,80,100]} for x in ['A','B','C','D']]
        raw={'reviews':[{'id':'A','action':'keep','bbox_2d':None,'reason':'ok'},
                        {'id':'B','action':'redraw','bbox_2d':[10,20,30,40],'reason':'moved'},
                        {'id':'C','action':'reject','bbox_2d':None,'reason':'false'},
                        {'id':'D','action':'keep','bbox_2d':None,'reason':'partial accepted'}]}
        snapshot=copy.deepcopy(old);kept,hist,rejected,issues=apply_review(old,raw,frame,[200,200],['D'])
        self.assertEqual(old,snapshot)
        self.assertEqual({x['id'] for x in kept},{'A','B','D'})
        self.assertEqual(next(o for o in kept if o['id']=='B')['bbox_2d'],[50.,80.,70.,100.])
        self.assertEqual(next(o for o in kept if o['id']=='A')['bbox_2d'],old[0]['bbox_2d'])
        self.assertEqual(next(o for o in kept if o['id']=='D')['review_status'],'needs_context')
        self.assertEqual(rejected[0]['id'],'C');self.assertEqual(len(hist),4)

    def test_partial_redraw_retains_the_complete_registered_occurrence(self):
        frame=Crop((100,100),(40,0,100,100),(100,100))
        original=[{'id':'wide','bbox_2d':[10,20,90,80]}]
        raw={'reviews':[{'id':'wide','action':'redraw','bbox_2d':[0,20,80,80],
                         'reason':'Only the visible right side is measurable'}]}
        kept,history,rejected,issues=apply_review(original,raw,frame,[100,100],['wide'])
        self.assertEqual(kept[0]['bbox_2d'],original[0]['bbox_2d'])
        self.assertEqual(kept[0]['review_status'],'needs_context')
        self.assertEqual(history[0]['model_action'],'redraw')
        self.assertFalse(rejected)
        self.assertEqual(len(issues),1)

    def test_local_request_has_visual_only_old_boxes(self):
        a={'candidates':[{'id':'G1','label':'block','kind':'rectangle','role':'assigned','bbox_2d':[1,2,3,4]}]}
        p=prompt_for(a,{'actual_crop_input_size_px':[2048,1024]})
        self.assertEqual(set(p['candidates'][0]),{'id','label','kind','role'})
        self.assertFalse({'frame','region','global_size','source_size','crop_offset'} & set(p))
        check=validator(['G1'],[2048,1024])
        with self.assertRaises(ValueError):check({'reviews':[],'new_objects':[],'uncertain':[],'more_pending':False})

    def test_unique_region_ownership_and_orphans(self):
        objects=[{'id':'A','bbox_2d':[5,5,15,15]},{'id':'B','bbox_2d':[80,80,90,90]}]
        regions=[{'id':'R1','bbox_2d':[0,0,20,20]},{'id':'R2','bbox_2d':[10,10,30,30]}]
        owners,missing=assign(objects,regions,[100,100],[100,100]);self.assertEqual(owners,{'A':'R1'});self.assertEqual(missing,['B'])
        with self.assertRaises(ValueError):validate_plan({'regions':[{'id':'R','bbox_2d':[0,0,101,100],'description':'all'}],'uncertain':[]},[100,100])

    def test_layout_outer_edge_convention_does_not_discard_regions(self):
        raw={'regions':[{'id':'L','bbox_2d':[0,0,60,100],'description':'left'},
                        {'id':'R','bbox_2d':[50,0,100,100],'description':'right'}],'uncertain':[]}
        original=copy.deepcopy(raw)
        plan=validate_plan(raw,[100,100])
        self.assertEqual([r['bbox_2d'] for r in plan['regions']],[[0,0,60,99],[50,0,99,99]])
        self.assertEqual(raw,original)
        for box in ([-1,0,100,100],[0,0,101,100],[99,0,100,100],[0,0,100,101]):
            invalid=copy.deepcopy(raw);invalid['regions'][0]['bbox_2d']=box
            with self.assertRaises(ValueError):validate_plan(invalid,[100,100])

    def test_image_adapter_detect_only(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as d:
            folder=Path(d)
            def fake_run(args):
                self.assertTrue(args.detect_only);self.assertIsNone(args.only);self.assertIsNone(args.candidates)
                (folder/'inventory.json').write_text('{"assets": [], "uncertain": []}')
            with patch('shape_integration.image_bridge.pipeline.run',side_effect=fake_run):
                self.assertEqual(detect(ROOT/'inputs/001/original.png',folder,read(ROOT/'config.json'))['assets'],[])

    def test_frozen_13_unique_inputs(self):
        items=read(ROOT/'inputs/manifest.json')['images'];self.assertEqual(len(items),13);self.assertEqual(len({o['source_sha256'] for o in items}),13)
        self.assertEqual(sum(o['group']=='user_added' for o in items),1)
        for o in items:self.assertEqual(sha(ROOT/o['original']),o['source_sha256']);self.assertEqual(sha(ROOT/o['input']),o['input_sha256'])

if __name__=='__main__':unittest.main()
