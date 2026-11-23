"""Busy admission may retry; accepted/ambiguous writes must never retry."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import urllib.error
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'vendor'),str(ROOT/'experiment/shape_sam_guided'),str(ROOT/'experiment/style_line_sam')]
from request_io import request_json,ModelTransportError
from sam_tiles import run_case
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


class SamAdmissionTests(unittest.TestCase):
    def error(self,code,message):
        return urllib.error.HTTPError('http://sam/',code,'failed',{},io.BytesIO(json.dumps({'error':message}).encode()))

    def test_only_explicit_unadmitted_busy_can_retry(self):
        opener=Mock();opener.open.side_effect=[self.error(503,'SAM is busy; retry after this request finishes'),io.BytesIO(b'{"ok":true}')]
        with patch('request_io.time.sleep'):
            self.assertTrue(request_json(opener,'http://sam/',{},busy_only=True)['ok'])
        self.assertEqual(opener.open.call_count,2)
        for error in (self.error(500,'failed'),self.error(503,'upstream unavailable'),TimeoutError('unknown accepted state')):
            opener=Mock();opener.open.side_effect=error
            with self.assertRaises(ModelTransportError):request_json(opener,'http://sam/',{},busy_only=True)
            self.assertEqual(opener.open.call_count,1)

    def test_committed_evidence_is_byte_identical_on_retry(self):
        with tempfile.TemporaryDirectory() as tmp,ThreadPoolExecutor(max_workers=1) as pool:
            root=Path(tmp);image=root/'input.png';Image.new('RGB',(20,20),'white').save(image)
            scene=root/'scene.json';scene.write_text(json.dumps({'objects':{},'parts':{},'paths':{}}))
            case={'id':'case','original':str(image),'scene':str(scene)}
            config={'side':96,'prompts':'both','score_threshold':.5,'categories':['line']}
            sam=Mock();sam.__self__=None;run_case(case,config,root/'sam',sam,pool)
            evidence=root/'sam/case/line_evidence.json';before=evidence.read_bytes();stamp=evidence.stat().st_mtime_ns
            out=run_case(case,config,root/'sam',sam,pool)
            self.assertTrue(out['reused_completed_evidence']);self.assertEqual(before,evidence.read_bytes())
            self.assertEqual(stamp,evidence.stat().st_mtime_ns);sam.assert_not_called()
            Image.new('RGB',(20,20),'black').save(image)
            with self.assertRaisesRegex(ValueError,'input changed'):run_case(case,config,root/'sam',sam,pool)


if __name__=='__main__':unittest.main()
