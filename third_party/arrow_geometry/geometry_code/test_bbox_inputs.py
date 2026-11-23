import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
import bbox_inputs as bi
from bbox_geometry import prepare_target,validate
CFG={'bbox_min_extent':25,'bbox_display_side':1080,'bbox_context_scale':2.,'context_padding':12,'context_side':300}
VIEW={'original_image_size_px':[100,80],'local_to_global':{'scale_xy':[.5,.5],'offset_xy':[-.25,-.25]},'global_to_original':{'scale_xy':[1,1],'offset_xy':[0,0]}}
def task(id,refs=(),render=True):
    return {'target_id':id,'kind':'single','bbox_native':[10,10,50,40],'identity_hint':'line near x≈160 with n=3','view':VIEW,'render_owner':render,'readonly_identities':[{'id':'r'+str(i),'owner_id':owner,'bbox':[20,20,80,60],'locator_hint':'arrow'} for i,owner in enumerate(refs)]}
class Handoff(unittest.TestCase):
    def test_deferred_candidates_remain_separate_and_keep_native_coordinates(self):
        pending=dict(task('U'),task_role='missing_candidate',render_owner=False,
                     extent='uncertain',observation_ids=['G1'],render_type='block_arrow')
        raw={'case_id':'x','tasks':[task('A')],'deferred_candidates':[pending]}
        before=copy.deepcopy(raw)
        result=bi.normalize(raw,[100,80],'x')
        self.assertEqual([r['id'] for r in result['tasks']],['A'])
        self.assertEqual(result['deferred_candidates'][0]['id'],'U')
        self.assertEqual(result['deferred_candidates'][0]['bbox_native'],pending['bbox_native'])
        self.assertEqual(result['deferred_candidates'][0]['observation_ids'],['G1'])
        self.assertEqual(result['source_count'],2)
        self.assertEqual(raw,before)
        for bad in [dict(pending,target_id='A'),dict(pending,render_owner=True),dict(pending,bbox_native=[0,0,0,0])]:
            with self.assertRaises(ValueError):bi.normalize({**raw,'deferred_candidates':[bad]},[100,80],'x')
    def test_frames_differ_for_target_and_references(self):
        r=bi.normalize({'case_id':'x','tasks':[task('A',['B']),task('B')]},[100,80],'x')
        self.assertEqual(r['tasks'][0]['bbox_native'],[10,10,50,40])
        self.assertEqual(r['tasks'][0]['readonly'][0]['bbox_native'],[9.75,9.75,39.75,29.75])
        self.assertTrue(r['tasks'][0]['readonly'][0]['exclusion_verified'])
    def test_cycle_and_missing_owner_do_not_suppress(self):
        r=bi.normalize({'case_id':'x','tasks':[task('A',['B','missing']),task('B',['A'])]},[100,80],'x')
        self.assertTrue(all(not v['exclusion_verified'] for t in r['tasks'] for v in t['readonly']))
        self.assertEqual(len(r['issues']),3)
    def test_context_is_not_recreated_and_hint_coordinates_removed(self):
        r=bi.normalize({'case_id':'x','tasks':[task('A'),task('B',render=False)]},[100,80],'x')
        self.assertEqual(len(r['tasks']),1);self.assertEqual(len(r['context_regions']),1)
        self.assertNotIn('160',r['tasks'][0]['locator']);self.assertIn('n=3',r['tasks'][0]['locator'])
    def test_bad_input_rejected(self):
        for raw in [{'case_id':'x','tasks':[task('A'),task('A')]},{'case_id':'wrong','tasks':[]}]:
            with self.assertRaises(ValueError):bi.normalize(raw,[100,80],'x')
        with self.assertRaises(ValueError):bi.box([-2,0,0,5],[100,80])
    def test_prompt_has_one_target_and_readonly_not_rectangular_erasure(self):
        obj=bi.normalize({'case_id':'x','tasks':[task('A',['B']),task('B')]},[100,80],'x')['tasks'][0]
        with tempfile.TemporaryDirectory() as tmp:
            crop,frame,q,images=prepare_target(Image.new('RGB',(100,80),'white'),obj,Path(tmp),CFG)
            self.assertEqual(len(q['targets']),1);self.assertEqual(q['targets'][0]['id'],'R1')
            self.assertEqual(q['readonly_identities'][0]['owner_id'],'B')
            self.assertIn('never its rectangle',' '.join(q['instructions']))
            self.assertEqual(len(images),3)
            corners=np.array([[10,10,1],[50,40,1]])@np.array(frame['global_to_view']).T
            self.assertTrue(np.allclose(q['targets'][0]['bbox'],corners[:,:2].reshape(-1),atol=.01))
    def test_linework_response_cannot_change_representation(self):
        with self.assertRaisesRegex(ValueError,'linework request'):
            validate({'status':'block','reason':'closed wide arrow'},[100,80])
    def test_freeze_rejects_unfinished_and_hash_drift(self):
        from transport.common import write_json as write,sha256
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'source';run=root/'run';run.mkdir();(src/'inputs/x').mkdir(parents=True);(src/'cases/x').mkdir(parents=True)
            write(src/'cases.json',['x']);Image.new('RGB',(100,80),'white').save(src/'inputs/x/source.png')
            write(src/'inputs/x/views.json',{'native_size':[100,80],'source_sha256':sha256(src/'inputs/x/source.png')})
            write(src/'cases/x/result.json',{'state':'completed'});write(src/'cases/x/reconstruction_handoff.json',{'case_id':'x','tasks':[task('A')]})
            with self.assertRaises(ValueError):bi.freeze(run,{'bbox_discovery_run':str(src)})
            write(src/'cases/x/finished.json',{'done':True});bi.freeze(run,{'bbox_discovery_run':str(src)})
            bi.verify(run);write(run/'discovery_inputs/x/handoff.json',{})
            with self.assertRaises(ValueError):bi.verify(run)
if __name__=='__main__':unittest.main()

