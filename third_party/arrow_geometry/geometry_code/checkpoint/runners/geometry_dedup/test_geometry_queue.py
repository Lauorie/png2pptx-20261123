import json
import tempfile
import unittest
from pathlib import Path
import run_geometry_dedup24 as Q


class HandoffQueue(unittest.TestCase):
    def fixture(self, root):
        source=root/'source';case=source/'cases/x';case.mkdir(parents=True)
        (source/'inputs/x').mkdir(parents=True)
        for p in [case/'result.json',case/'reconstruction_handoff.json']:
            Q.write(p,{})
        (source/'inputs/x/source.png').write_bytes(b'input evidence')
        (case/'current.png').write_bytes(b'overlay evidence')
        return source,case

    def test_running_controller_is_not_a_finished_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,case=self.fixture(Path(tmp))
            Q.write(source/'execution.json',{'state':'running'})
            Q.write(case/'status.json',{'state':'running'})
            self.assertFalse(Q.ready(source,'x'))
            Q.write(case/'status.json',{'state':'completed_with_issues'})
            self.assertTrue(Q.ready(source,'x'))

    def test_freeze_preserves_bytes_and_rejects_changed_upstream(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,case=self.fixture(root);dest=root/'geometry'
            Q.write(case/'status.json',{'state':'completed'})
            out=Q.freeze_case(source,dest,'x')
            audit=Q.read(out/'input_provenance.json')
            for name,item in audit.items():self.assertEqual(Q.sha(out/name),item['sha256'])
            self.assertEqual(Q.freeze_case(source,dest,'x'),out)
            Q.write(case/'result.json',{'changed':True})
            with self.assertRaises(ValueError):Q.freeze_case(source,dest,'x')

    def test_not_ready_freeze_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,case=self.fixture(root)
            Q.write(case/'status.json',{'state':'running'})
            with self.assertRaises(ValueError):Q.freeze_case(source,root/'dest','x')

if __name__=='__main__':unittest.main()
