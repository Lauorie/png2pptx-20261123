"""A recovery writer keeps finished deliveries and cannot overlap the old owner."""
import fcntl
import json
from pathlib import Path
import tempfile
import unittest

from full_recovery.advance_shape import activate
from full_recovery.compose import sha


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


class ShapeAdvance(unittest.TestCase):
    def test_explicit_missing_part_repair_preserves_the_seed_and_archives_its_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'base';work=base/'shape';case=work/'geometry/cases/case'
            capsule=base/'updates/repair';selection=capsule/'selection.json'
            scene={'source_sha256':'source','objects':{'o':{'candidate':{'id':'o'}}}}
            save(case/'fitting/scene.json',scene);save(case/'geometry.json',{})
            proof=case/'fitting/objects/o/record.json';save(proof,{'selected_revision':'revision_00'})
            before=sha(proof)
            request={'object_id':'o','issues':[{'kind':'missing','part_ids':[],
                'evidence':'A source region lacks its closed face.','action':'Add that face.'}]}
            save(selection,{'cases':{'case':{'objects':['o'],'repair_existing':[request],
                'source_sha256':'source','scene_sha256':sha(case/'fitting/scene.json')}}})
            settings={'shape_base_run':str(base),'shape_repair_selection':str(selection),'api_global_concurrency':96}
            for p in [capsule/'settings.json',capsule/'shape/settings.json']:save(p,settings)
            save(capsule/'shape/pipeline_config.json',{})
            save(base/'source_manifest.json',{'images':[{'id':'case','source_sha256':'source'}]})
            activate(capsule)
            archived=capsule/'previous_work/shape/geometry/cases/case/fitting/objects/o/record.json'
            self.assertEqual(sha(proof),before);self.assertEqual(sha(archived),before)
            retained=json.loads((case/'fitting/resume_candidates.json').read_text())
            self.assertEqual(retained['repair_existing'],[request])
            self.assertEqual(json.loads(Path(retained['previous_scene']).read_text()),scene)

    def test_selected_object_is_rebuilt_without_erasing_siblings_or_source_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'base';work=base/'shape';capsule=base/'updates/fix';case=work/'geometry/cases/case'
            scene={'source_sha256':'source','objects':{key:{'candidate':{'id':key}} for key in ('good','bad','discovered')}}
            save(case/'fitting/scene.json',scene);save(case/'geometry.json',{})
            for key in scene['objects']:save(case/'fitting/objects'/key/'done.json',{'proof':key})
            scene_hash=sha(case/'fitting/scene.json');good_hash=sha(case/'fitting/objects/good/done.json')
            selection=capsule/'repair_selection.json'
            save(selection,{'cases':{'case':{'objects':['bad'],'source_sha256':'source','scene_sha256':scene_hash}}})
            settings={'shape_base_run':str(base),'api_global_concurrency':96,'shape_repair_selection':str(selection)}
            for path in (capsule/'settings.json',capsule/'shape/settings.json'):save(path,settings)
            save(capsule/'shape/pipeline_config.json',{});save(base/'source_manifest.json',{'images':[{'id':'case','source_sha256':'source'}]})
            save(case/'complete.json',{'old_delivery':True});save(case/'styling/identity.json',{})
            save(work/'sam_geometry/case/index.json',{'cached':True})
            result=activate(capsule)
            self.assertEqual(result['selectively_repaired_cases'],{'case':['bad']})
            self.assertEqual(sha(case/'fitting/objects/good/done.json'),good_hash)
            self.assertTrue((case/'fitting/objects/discovered/done.json').exists())
            self.assertFalse((case/'fitting/objects/bad').exists())
            self.assertTrue((capsule/'previous_work/shape/geometry/cases/case/fitting/objects/bad/done.json').exists())
            self.assertTrue((work/'sam_geometry/case/index.json').exists())
            retained=json.loads((case/'fitting/resume_candidates.json').read_text())
            self.assertEqual({r['id'] for r in retained['objects']},{'good','bad','discovered'})

    def test_failed_style_is_archived_while_finished_shapes_and_geometry_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'base';work=base/'shape';previous=base/'updates/previous';previous.mkdir(parents=True)
            capsule=base/'updates/recovery'
            settings={'shape_base_run':str(base),'previous_shape_snapshot':str(previous),'api_global_concurrency':96}
            for p in [capsule/'settings.json',capsule/'shape/settings.json']:save(p,settings)
            save(capsule/'shape/pipeline_config.json',{})
            code=capsule/'shape/code/src/shape_integration/styling/native_stroke.py'
            code.parent.mkdir(parents=True);code.write_text('fixed numerical estimator')
            save(base/'source_manifest.json',{'images':[{'id':'done'},{'id':'failed'}]})
            for cid in ('done','failed'):
                case=work/'geometry/cases'/cid
                save(case/'fitting/scene.json',{'objects':{}})
                save(case/'fitting/summary.json',{'state':'completed'})
                save(case/'styling/identity.json',{'line_evidence':{'code':sha(code)},'timeout':1200})
            ppt=work/'delivery/done/shapes.pptx';ppt.parent.mkdir(parents=True);ppt.write_bytes(b'finished delivery')
            save(work/'geometry/cases/done/complete.json',{'pptx':str(ppt),'pptx_sha256':sha(ppt)})
            save(work/'sam_styles/failed/line_evidence.json',{'cached':True})
            geometry=work/'geometry/cases/failed/fitting/scene.json';before=geometry.read_bytes()
            with (previous/'worker.lock').open('a') as old:
                fcntl.flock(old,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):activate(capsule)
            result=activate(capsule)
            self.assertEqual(set(result['completed_cases_preserved']),{'done'})
            self.assertEqual(geometry.read_bytes(),before)
            self.assertTrue((work/'geometry/cases/done/styling/identity.json').exists())
            self.assertFalse((work/'geometry/cases/failed/styling').exists())
            self.assertTrue((capsule/'previous_work/shape/geometry/cases/failed/styling/identity.json').exists())
            self.assertTrue((work/'sam_styles/failed/line_evidence.json').exists())
            self.assertEqual(activate(capsule),result)


if __name__=='__main__':unittest.main()
