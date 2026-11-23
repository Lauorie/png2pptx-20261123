"""Offline regression against real saved replies, plus candidate isolation."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from supplement_protocol import MissingProtocol, ReviewProtocol
from pipeline_frames import Crop
import review_additions as R
import experiment as E
from transport.model import Model

def row(i="R1", box=None):
    return dict(id=i, bbox=box or [10,10,30,30], render_type="linework",
                grouping="single", label="visible arrow", extent="complete")
def answer():
    return dict(decision="edit", keep_ids=[], replacements=[], discard=[])

class ProtocolTests(unittest.TestCase):
    def test_explicit_regions_assignment_is_not_empty_guessing(self):
        v=MissingProtocol((100,100))
        self.assertEqual(v.parse(' regions=[] \n全部已经覆盖'), {'regions':[]})
        a=row()
        self.assertEqual(v.parse('regions='+json.dumps([a]))['regions'],[a])
        for text in ('没有发现', 'regions=[] regions=[{}]', 'regions=[', 'regions={}'): 
            with self.assertRaises(ValueError):v.parse(text)

    def test_empty_result_does_not_need_metadata(self):
        v=MissingProtocol((100,100))
        for a in [{"regions":[]},{"regions":[],"task":"check missing"},dict(regions=[],frame="LOCAL")]:
            self.assertEqual(v(a)["regions"],[])
        with self.assertRaises(ValueError):v({})
        with self.assertRaises(ValueError):v(dict(regions=[],frame="GLOBAL"))

    def test_clipped_candidate_is_unreviewed_not_complete(self):
        r=row();r["extent"]="clipped"
        result=MissingProtocol((100,100))({"regions":[r]})
        self.assertEqual(result["regions"][0]["extent"],"uncertain")
        self.assertEqual(result["regions"][0]["bbox"],r["bbox"])
        self.assertFalse(result["regions"][0]["review_confirmed"])

    def test_edge_overflow_clamped_without_losing_candidate_or_raw_evidence(self):
        a=row("r1");b=row("r2",[0,10,30,100])
        original={"regions":[a,b]};before=copy.deepcopy(original)
        v=MissingProtocol((100,100));out=v(original)
        self.assertEqual(len(out["regions"]),2)
        self.assertEqual(out["regions"][1]["bbox"],[0,10,30,99])
        self.assertEqual(out["regions"][1]["raw_model_bbox"],b["bbox"])
        self.assertEqual(out["_protocol"]["raw_answer"],before)
        self.assertEqual(original,before)
        self.assertEqual(v.validate_saved(out),out)
        self.assertFalse(out["_protocol"]["rejected"])
        self.assertEqual(out["regions"][0]["bbox"],a["bbox"])

    def test_all_sides_clamped_using_actual_rectangular_input(self):
        out=MissingProtocol((1080,283))({"regions":[row(box=[-2,-5,1082,283])]})
        self.assertEqual(out["regions"][0]["bbox"],[0,0,1079,282])
        self.assertTrue(out["_protocol"]["effective_coordinates_changed"])

    def test_invalid_or_fully_outside_boxes_remain_pending_without_inventing_area(self):
        for box in ([30,10,20,40],[101,10,110,40],[10,10,10,40],
                    [float('nan'),10,30,40],[False,10,30,40],['1',10,30,40]):
            out=MissingProtocol((100,100))({"regions":[row(),row('bad',box)]})
            self.assertEqual(len(out['regions']),1)
            self.assertEqual(len(out['_protocol']['rejected']),1)

    def test_multiple_answer_objects_are_not_guessed(self):
        v=MissingProtocol((100,100))
        with self.assertRaises(ValueError):v.parse('{"regions":[]}\n{"regions":[]}')

    def test_review_wrapper_and_prose(self):
        v=ReviewProtocol((100,100),[row("ADD1")])
        a=dict(answer(),keep_ids=["ADD1"])
        self.assertEqual(v(v.parse("Before\n```json\n"+json.dumps(a)+"\n```\nReason"))["keep_ids"],["ADD1"])

    def test_metadata_only_text_repair(self):
        v=MissingProtocol((100,100))
        v.repair_audit('{"regions":[]}',{"frame":"LOCAL","regions":[]})
        with self.assertRaises(ValueError):
            v.repair_audit(json.dumps({"regions":[row()]}),{"regions":[row(box=[11,10,30,30])]})

    def test_forged_adapter_metadata_rejected(self):
        v=MissingProtocol((100,100))
        with self.assertRaises(ValueError):v(dict(regions=[],_protocol={"version":"sam-candidate-protocol-v2"}))

    def test_review_unknown_old_id_cannot_delete_baseline(self):
        focus=[row("ADD1"),row("ADD2")]
        a=dict(answer(),keep_ids=["ADD1"],discard=[dict(ids=["B1"],reason="duplicate")])
        result=ReviewProtocol((100,100),focus)(a)
        self.assertEqual(result["keep_ids"],["ADD1"])
        self.assertEqual(result["_protocol"]["pending_focus_ids"],["ADD2"])
        self.assertEqual(result["discard"],[])

    def test_conflicting_dispositions_quarantine_all_affected_ops(self):
        focus=[row("ADD1"),row("ADD2")]
        a=dict(answer(),keep_ids=["ADD1","ADD2"],discard=[dict(ids=["ADD1"],reason="duplicate")])
        result=ReviewProtocol((100,100),focus)(a)
        self.assertEqual(result["keep_ids"],["ADD2"])
        self.assertEqual(result["_protocol"]["pending_focus_ids"],["ADD1"])

    def test_split_replacement_is_atomic(self):
        focus=[row("ADD1"),row("ADD2")]
        a=dict(answer(),keep_ids=["ADD2"],replacements=[dict(source_ids=["ADD1"],
            regions=[row("R1"),row("R2",[30,10,20,40])],reason="split two arrows")])
        result=ReviewProtocol((100,100),focus)(a)
        self.assertEqual(result["replacements"],[])
        self.assertEqual(result["keep_ids"],["ADD2"])
        self.assertEqual(result["_protocol"]["pending_focus_ids"],["ADD1"])

    def test_review_clamp_keeps_split_children_and_maps_bounded_coordinates(self):
        focus=[row("ADD1")]
        raw=dict(answer(),replacements=[dict(source_ids=['ADD1'],
            regions=[row('R1',[10,10,40,40]),row('R2',[-1,50,100,100])],reason='split visible arrows')])
        before=copy.deepcopy(raw);v=ReviewProtocol((100,100),focus);a=v(raw)
        self.assertFalse(a['_protocol']['pending_focus_ids'])
        self.assertEqual(len(a['replacements'][0]['regions']),2)
        self.assertEqual(a['replacements'][0]['regions'][1]['bbox'],[0,50,99,99])
        self.assertEqual(raw,before)
        view=Crop((500,500),(100,200,300,400),(100,100))
        rows,pending,_,conversions=R.commit(focus,a,view,'RV',[])
        self.assertFalse(pending)
        self.assertEqual(rows[1]['raw_local_bbox'],[-1,50,100,100])
        self.assertEqual(rows[1]['effective_local_bbox'],[0,50,99,99])
        self.assertEqual(rows[1]['mapped_exact_bbox'],E.native_box([0,50,99,99],view))
        self.assertEqual(conversions[1]['raw_local_bbox'],[-1,50,100,100])

    def test_keep_is_review_confirmation_and_baseline_immutable(self):
        baseline=[row("B1",[60,60,90,90])]
        focus=[dict(row("ADD1"),extent="uncertain",candidate_status="unreviewed",operation="add_only")]
        before=copy.deepcopy(baseline)
        a=ReviewProtocol((100,100),focus)(dict(answer(),keep_ids=["ADD1"]))
        rows,pending,_,_=R.commit(focus,a,Crop((100,100),(0,0,100,100),(100,100)),"RV",baseline)
        self.assertEqual(rows[0]["extent"],"complete")
        self.assertTrue(rows[0]["review_confirmed"])
        self.assertEqual(baseline,before)
        self.assertFalse(pending)

    def test_unreviewed_cannot_enter_reviewed_export(self):
        cand=dict(row("ADD1"),extent="uncertain",candidate_status="unreviewed",operation="add_only")
        with self.assertRaises(ValueError):E.append_only([], [cand], (100,100))
        self.assertEqual(len(E.append_only([], [cand], (100,100),allow_unreviewed=True)[0]),1)

    def test_no_visual_retry_for_protocol_failure(self):
        cfg=dict(base_url="unused",model="glm-5.3-flash",temperature=.1,max_tokens=113000,
                 thinking={"type":"enabled"},api_concurrency=1,format_retries=1,transport_retries=0)
        seen=[]
        def request(config,system,prompt,images,folder):
            seen.append(bool(images));Path(folder).mkdir(parents=True)
            return "invalid reply" if images else '{"format_repair_failed":true}'
        with tempfile.TemporaryDirectory() as tmp:
            image=Path(tmp)/"image";image.write_bytes(b"x")
            with patch("transport.model.request",request):
                with self.assertRaises(ValueError):
                    Model(cfg).ask(Path(tmp)/"request",{"output_schema":{"regions":[]}},
                                   [("CLEAN",image)],MissingProtocol((100,100)))
        self.assertEqual(seen,[True,False])

if __name__=="__main__":unittest.main()

