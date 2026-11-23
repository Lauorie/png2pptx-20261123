import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from full_recovery.fresh_pipeline import validate_sources, validate_arrow_completion,arrow_case_products_ready
from full_recovery.fresh_arrow import refresh


class FreshSourceIsolationTests(unittest.TestCase):
    def test_ready_case_does_not_wait_for_other_arrow_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);folder=root/'geometry/cases/ready';folder.mkdir(parents=True)
            (root/'progress.json').write_text(json.dumps({'cases':[
                {'case':'ready','state':'completed'},{'case':'later','state':'running'}]}))
            for name in ['scene.json','arrows.pptx']:(folder/name).write_bytes(b'new-output')
            self.assertFalse(arrow_case_products_ready(root,'ready'))
            (folder/'completed.json').write_text('{}')
            self.assertTrue(arrow_case_products_ready(root,'ready'))
            self.assertFalse(arrow_case_products_ready(root,'later'))
            (folder/'arrows.pptx').write_bytes(b'')
            self.assertFalse(arrow_case_products_ready(root,'ready'))

    def test_zero_exit_controller_with_case_failures_is_not_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'progress.json').write_text(json.dumps({'failed':1,'cases':[{'case':'sample','state':'failed'}]}))
            with self.assertRaisesRegex(RuntimeError,'failed cases'):
                validate_arrow_completion(root,['sample'])

    def test_arrow_completion_requires_the_case_marker_and_real_products(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);folder=root/'geometry/cases/sample';folder.mkdir(parents=True)
            (folder/'scene.json').write_text('{}');(folder/'arrows.pptx').write_bytes(b'fresh-native-package')
            with self.assertRaisesRegex(RuntimeError,'completed.json'):
                validate_arrow_completion(root,['sample'])
            (folder/'completed.json').write_text('{}')
            validate_arrow_completion(root,['sample'])

    def test_current_run_product_is_bound_to_its_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary);path=run/'branch/scene.json';path.parent.mkdir()
            path.write_bytes(b'new result')
            row={'paths':{'scene':str(path)},'sha256':{'scene':hashlib.sha256(path.read_bytes()).hexdigest()}}
            validate_sources(run,[row])
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):validate_sources(run,[row])

    def test_historical_product_cannot_enter_through_a_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent=Path(temporary);run=parent/'fresh';run.mkdir()
            old=parent/'historical_scene.json';old.write_bytes(b'old result')
            link=run/'scene.json';link.symlink_to(old)
            row={'paths':{'scene':str(link)},'sha256':{'scene':hashlib.sha256(old.read_bytes()).hexdigest()}}
            with self.assertRaisesRegex(ValueError,'Historical'):validate_sources(run,[row])

    def test_executed_arrow_manifest_cannot_be_rewritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'execution.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'executed'):refresh(root)


if __name__=='__main__':unittest.main()
