import copy
import tempfile
import unittest
from pathlib import Path
from PIL import Image
import experiment as E
import review_additions as R
from pipeline_frames import Crop


def region(name,box):
    return dict(id=name,bbox=box,render_type='linework',grouping='single',label='visible connection',extent='complete',operation='add_only')


def answer(keep=None,replacements=None,discard=None):
    return dict(frame='LOCAL',decision='edit',keep_ids=keep or [],replacements=replacements or [],discard=discard or [],reason='')


class ReviewContract(unittest.TestCase):
    def setUp(self):
        self.view=Crop((200,200),(0,0,200,200),(200,200))
        self.baseline=[region('B1',[10,10,50,50])]
        self.focus=[region('ADD1',[40,40,70,70])]

    def test_all_original_edit_routes_rejected(self):
        for a in [answer(['B1']),answer(discard=[dict(ids=['B1'],reason='no')]),
                  answer(replacements=[dict(source_ids=['ADD1','B1'],regions=[region('R1',[20,20,80,80])],reason='merge')])]:
            result=R.validate(a,(200,200),self.focus)
            self.assertEqual(result['keep_ids'],[])
            self.assertEqual(result['replacements'],[])
            self.assertEqual(result['discard'],[])
            self.assertEqual(result['_protocol']['pending_focus_ids'],['ADD1'])

    def test_candidate_must_have_exactly_one_action(self):
        for a in [answer(),answer(['ADD1'],discard=[dict(ids=['ADD1'],reason='no')])]:
            result=R.validate(a,(200,200),self.focus)
            self.assertEqual(result['_protocol']['pending_focus_ids'],['ADD1'])

    def test_unrelated_discovery_not_allowed(self):
        a=answer(['ADD1'],replacements=[dict(source_ids=[],regions=[region('R1',[80,80,90,90])],reason='new')])
        result=R.validate(a,(200,200),self.focus)
        self.assertEqual(result['keep_ids'],['ADD1'])
        self.assertEqual(result['replacements'],[])
        self.assertTrue(result['_protocol']['rejected'])

    def test_repair_full_target_leaves_baseline_unchanged(self):
        old=copy.deepcopy(self.baseline)
        a=answer(replacements=[dict(source_ids=['ADD1'],regions=[region('R1',[10,10,80,80])],reason='complete visible target')])
        rows,pending,removed,_=R.commit(self.focus,a,self.view,'RV',self.baseline)
        final,audit=E.append_only(self.baseline,rows,(200,200))
        self.assertEqual(self.baseline,old);self.assertEqual(final[:1],old)
        self.assertEqual(rows[0]['bbox'],[10,10,80,80]);self.assertFalse(removed);self.assertFalse(pending)
        self.assertTrue(audit['passed'])

    def test_merge_new_boxes_and_split_new_box(self):
        focus=[*self.focus,region('ADD2',[60,60,90,90])]
        a=answer(replacements=[dict(source_ids=['ADD1','ADD2'],regions=[region('R1',[40,40,90,90])],reason='same new target')])
        rows,*_=R.commit(focus,a,self.view,'RV',self.baseline)
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['review_source_ids'],['ADD1','ADD2'])
        a=answer(replacements=[dict(source_ids=['ADD1'],regions=[region('R1',[40,60,50,80]),region('R2',[60,40,80,50])],reason='two separate targets')])
        rows,*_=R.commit(self.focus,a,self.view,'RV',self.baseline)
        self.assertEqual(len(rows),2)

    def test_empty_noise_discarded_without_deleting_old(self):
        a=answer(discard=[dict(ids=['ADD1'],reason='no visible ink')])
        rows,*_=R.commit(self.focus,a,self.view,'RV',self.baseline)
        final,_=E.append_only(self.baseline,rows,(200,200));self.assertEqual(final,self.baseline)

    def test_containment_applied_before_and_after_review_without_padding(self):
        inside=region('ADD0',[10,10,50,50]);overhang=region('ADD2',[10,10,50.00001,50])
        kept,removed=R.containment_filter([inside,overhang],self.baseline)
        self.assertEqual([r['id'] for r in kept],['ADD2']);self.assertEqual(removed[0]['contained_by'],['B1'])
        a=answer(replacements=[dict(source_ids=['ADD1'],regions=[region('R1',[20,20,40,40])],reason='corrected within original')])
        rows,_,removed,_=R.commit(self.focus,a,self.view,'RV',self.baseline)
        self.assertEqual(rows,[]);self.assertEqual(len(removed),1)

    def test_context_includes_complete_intersecting_old_box(self):
        old=[region('B1',[10,10,80,180])];f=[region('ADD1',[65,165,95,185])]
        view=R.context_frame(f,old,(300,300),1.6)
        self.assertLessEqual(view.box[0],10);self.assertLessEqual(view.box[1],10)
        self.assertGreater(view.box[3],180)

    def test_native_mapping_with_resize_and_offset(self):
        view=Crop((500,500),(100,200,300,400),(1080,1080))
        a=answer(replacements=[dict(source_ids=['ADD1'],regions=[region('R1',[100,200,700,800])],reason='visible')])
        rows,_,_,audit=R.commit(self.focus,a,view,'RV',[])
        self.assertEqual(rows[0]['bbox'],E.native_box([100,200,700,800],view))
        self.assertEqual(audit[0]['raw_local_bbox'],[100,200,700,800])

    def test_uncertain_geometry_held_out_of_export(self):
        r=region('R1',[40,40,70,70]);r['extent']='uncertain'
        rows,pending,_,_=R.commit(self.focus,answer(replacements=[dict(source_ids=['ADD1'],regions=[r],reason='not all visible')]),self.view,'RV',self.baseline)
        self.assertEqual(rows,[]);self.assertEqual(len(pending),1)

    def test_request_failure_is_pending_not_promoted(self):
        class Fail:
            def ask(self,*args,**kwargs):raise RuntimeError('mock transport failure')
        with tempfile.TemporaryDirectory() as d:
            result=R.review_one(Path(d),'synthetic',self.focus,self.baseline,Image.new('RGB',(200,200)),Fail(),
                dict(review_context_expansions=1,review_context_scale=1.6),1,1)
            self.assertEqual(result['rows'],[]);self.assertEqual(result['pending'],self.focus)
            self.assertTrue(result['issues'])


if __name__=='__main__':unittest.main()
