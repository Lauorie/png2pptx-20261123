import json
import multiprocessing as mp
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import evidence
import geometry_critic
from request_budget import counted, FileSlots
from transport import model as transport
from transport.client import TransportError

def slot_worker(folder, active, peak):
    with FileSlots(folder, 2):
        with active.get_lock():
            active.value+=1;peak.value=max(peak.value, active.value)
        time.sleep(.08)
        with active.get_lock(): active.value-=1

class BoundsTest(unittest.TestCase):
    def test_native_crop_interval_and_roundtrip(self):
        for width, expected in [(600,1080),(1080,1080),(1300,1300),(1536,1536),(2048,1536)]:
            with self.subTest(width=width), tempfile.TemporaryDirectory() as d:
                source=Image.new('RGB',(width,200),'white')
                cfg=dict(context_padding=0,min_crop_side=1,display_side=1080,display_long_side_range=[1080,1536])
                im, frame=evidence.frame(source,[],Path(d),cfg,box=[0,0,width,200])
                self.assertEqual(max(im.size),expected)
                self.assertEqual(frame['bbox_global'],[0,0,width,200])
                pts=np.array([[0,0,1],[width-1,199,1],[width/2,88,1]])
                mapped=pts@np.array(frame['global_to_view']).T
                np.testing.assert_allclose(mapped@np.array(frame['view_to_global']).T,pts,atol=1e-9)

    def test_critic_uses_original_pixel_extent(self):
        source=dict(size=[900,200],view_to_global=[[2,0,.5],[0,2,.5],[0,0,1]])
        doc=dict(nodes=[['a',0,10],['b',639,30]])
        frame=geometry_critic.frame_for(doc,(900,200),scale=1,min_extent=1,source_view=source,long_side_range=[1080,1536])
        self.assertEqual(max(frame['size']),1280)
        np.testing.assert_allclose(geometry_critic.transform(geometry_critic.transform(doc,frame),frame,True)['nodes'][1][1:],doc['nodes'][1][1:],atol=1e-9)

    def model_trial(self, outcomes):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);image=root/'source.png';Image.new('RGB',(10,10)).save(image)
            cfg=dict(base_url='http://example.invalid/v1',model='glm-5.3-flash',temperature=.1,max_tokens=113000,
                     thinking={'type':'enabled'},api_concurrency=128,transport_retries=1,format_retries=1,
                     request_max_attempts=3,retry_backoff_seconds=0)
            called=[]
            @counted
            def fake(config, system, prompt, images, output):
                output.mkdir(parents=True);(output/'wire_request.json').write_text('{}')
                called.append(prompt);value=outcomes[len(called)-1]
                if isinstance(value,Exception):raise value
                return value
            m=transport.Model(cfg,run_dir=root)
            with patch.object(transport,'request',fake),patch('time.sleep',lambda *_:None):
                try:
                    answer=m.ask(root/'request',{'task':'test'},[('CLEAN',image)],lambda v:v if v.get('ok') else (_ for _ in ()).throw(ValueError('missing ok')))
                    error=None
                except Exception as exc:answer=None;error=exc
            ledger=json.loads((root/'request/attempt_budget.json').read_text())
            return answer,error,len(called),ledger

    def test_timeout_then_success(self):
        a,e,n,l=self.model_trial([TransportError('TimeoutError: '),'{"ok":true}'])
        self.assertEqual((a,n),({'ok':True},2));self.assertIsNone(e)

    def test_three_failures_hard_cap(self):
        a,e,n,l=self.model_trial([TransportError('TimeoutError: ')]*4)
        self.assertEqual(n,3);self.assertEqual(l['used'],3);self.assertIsNotNone(e)

    def test_format_and_transport_share_budget(self):
        a,e,n,l=self.model_trial(['{bad',TransportError('TimeoutError: '),'{"ok":true}'])
        self.assertEqual((a,n),({'ok':True},3));self.assertIsNone(e)

    def test_bad_format_never_exceeds_three(self):
        a,e,n,l=self.model_trial(['{bad']*5)
        self.assertEqual(n,3);self.assertIsNotNone(e)

    def test_auth_not_retried(self):
        a,e,n,l=self.model_trial([TransportError('HTTP 401',401)])
        self.assertEqual(n,1);self.assertIsNotNone(e)

    def test_process_api_cap(self):
        with tempfile.TemporaryDirectory() as d:
            ctx=mp.get_context('fork');active=ctx.Value('i',0);peak=ctx.Value('i',0)
            jobs=[ctx.Process(target=slot_worker,args=(d,active,peak)) for _ in range(8)]
            for p in jobs:p.start()
            for p in jobs:p.join(5);self.assertEqual(p.exitcode,0)
            self.assertEqual(peak.value,2);self.assertEqual(active.value,0)

if __name__=='__main__':unittest.main()
