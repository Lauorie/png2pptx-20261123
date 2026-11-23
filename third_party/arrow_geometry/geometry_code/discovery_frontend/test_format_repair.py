import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from transport.model import Model
from transport.format_repair import preservation_audit

class FormatRepairTests(unittest.TestCase):
    def cfg(self):return {'api_concurrency':2,'base_url':'unused','model':'glm','temperature':.1,
        'thinking':{'type':'enabled'},'max_tokens':113000,'format_retries':1,'transport_retries':0}
    def validate(self,raw):
        if not isinstance(raw,dict) or not isinstance(raw.get('objects'),list):raise ValueError('objects list required')
        for o in raw['objects']:
            if not all(type(v) in (int,float) for v in o['bbox']):raise ValueError('numeric bbox required')
        return raw
    def test_text_only_fixes_numeric_types_and_keeps_image_call_once(self):
        calls=[]
        def response(cfg,system,q,images,path):
            calls.append(images)
            if len(calls)==1:return '{"objects":[{"bbox":["1","2","3","4"]}]}'
            return '{"objects":[{"bbox":[1,2,3,4]}]}'
        with tempfile.TemporaryDirectory() as tmp,patch('transport.model.request',side_effect=response):
            p=Path(tmp);im=p/'image.png';im.write_bytes(b'fixture')
            result=Model(self.cfg()).ask(p/'q',{},[('TARGET',im)],self.validate)
            self.assertEqual(result['objects'][0]['bbox'],[1,2,3,4]);self.assertEqual(len(calls),2);self.assertEqual(calls[1],[])
    def test_unsafe_format_change_falls_back_to_one_visual_retry(self):
        calls=[]
        def response(cfg,system,q,images,path):
            calls.append(images)
            if len(calls)==1:return '{"objects":[{"bbox":["1","2","3","4"]}]}'
            if len(calls)==2:return '{"objects":[{"bbox":[10,20,30,40]}]}'
            return '{"objects":[{"bbox":[1,2,3,4]}]}'
        with tempfile.TemporaryDirectory() as tmp,patch('transport.model.request',side_effect=response):
            p=Path(tmp);im=p/'image.png';im.write_bytes(b'fixture')
            Model(self.cfg()).ask(p/'q',{},[('TARGET',im)],self.validate)
            self.assertEqual(len(calls),3);self.assertEqual(calls[1],[]);self.assertEqual(len(calls[2]),1)
    def test_punctuation_repair_preserves_values(self):
        raw='{"objects":[{"bbox":[1 2,3,4],"kind":"single"}]}'
        fixed={'objects':[{'bbox':[1,2,3,4],'kind':'single'}]}
        self.assertTrue(preservation_audit(raw,fixed)['scalar_values_preserved'])
    def test_object_coordinate_swap_is_not_formatting(self):
        a={'objects':[{'id':'a','bbox':[1,2,3,4]},{'id':'b','bbox':[5,6,7,8]}]}
        b={'objects':[{'id':'b','bbox':[1,2,3,4]},{'id':'a','bbox':[5,6,7,8]}]}
        with self.assertRaises(ValueError):preservation_audit(json.dumps(a),b)
    def test_context_note_relocation_preserves_objects_and_all_text(self):
        obj={'id':'O1','source_ids':['1'],'bbox':[1,2,3,4],'locator':'network wiring'}
        note={'source_id':'1','reason':'Node circle outlines are excluded from this wiring object'}
        before={'objects':[obj],'removed':[note],'unresolved':[]}
        after={'objects':[obj],'removed':[],'unresolved':[],'context_notes':[note]}
        self.assertTrue(preservation_audit(json.dumps(before),after)['scalar_values_preserved'])

if __name__=='__main__':unittest.main()
