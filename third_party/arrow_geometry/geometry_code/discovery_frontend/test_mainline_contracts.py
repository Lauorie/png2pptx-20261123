import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from semantic_hints import clean_prompt
from transport.model import Model
from owner_refs import validate_identity_decisions,absorbed_candidates,reconcile,export_handoff
from light_core import compress

def obj(i,b,verified=False):
    return {'id':i,'display_id':'I0001' if i=='whole' else 'I0002','kind':'single','bbox_native':b,
            'source_ids':[i],'locator':i,'extent':'complete','uncertain':False,'verified_independent':verified}

class MainlineContractTests(unittest.TestCase):
    def test_hint_coordinates_do_not_follow_bbox_into_new_frame(self):
        q={'focus':[{'bbox':[147,146,1765,312],'start_hint':'第3个peer-attn(x≈1170-1225)顶部',
                     'route_hint':'弧顶(y≈204，x≈1400-1550)，Conv(n=3)，size=3x3'}]}
        original=copy.deepcopy(q);clean,audit=clean_prompt(q)
        self.assertEqual(q,original);self.assertEqual(clean['focus'][0]['bbox'],q['focus'][0]['bbox'])
        self.assertNotIn('1170',str(clean));self.assertNotIn('1400',str(clean))
        self.assertIn('n=3',str(clean));self.assertIn('3x3',str(clean));self.assertEqual(len(audit),2)

    def test_false_null_compatibility_preserves_raw_reply(self):
        raw={'decisions':[{'id':'1','independent_complete':True,'source_identity_preserved':True,'same_as_readonly':False,'reason':'different physical arrows'}]}
        result=validate_identity_decisions(raw,{'1'},2)
        self.assertIs(raw['decisions'][0]['same_as_readonly'],False)
        self.assertIsNone(result['decisions'][0]['same_as_readonly'])
        self.assertEqual(result['decisions'][0]['format_normalizations'][0]['from'],False)
        for bad in [True,'r3']:
            raw['decisions'][0]['same_as_readonly']=bad
            with self.assertRaises(ValueError):validate_identity_decisions(raw,{'1'},2)

    def test_format_retry_appends_after_existing_attempts(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);(folder/'call_000').mkdir();(folder/'call_000/evidence.txt').write_text('old evidence')
            model=Model({'api_concurrency':1,'transport_retries':0})
            calls=[]
            def fake(config,system,prompt,images,path):
                path.mkdir();calls.append(path.name);return '{"decisions":[]}'
            with patch('transport.model.request',fake):
                for _ in range(2):self.assertEqual(model.format_only(folder,{},'{"decisions":[]}',ValueError('format'),lambda x:x),{'decisions':[]})
            self.assertEqual(calls,['call_001','call_002']);self.assertEqual((folder/'call_000/evidence.txt').read_text(),'old evidence')

    def test_unknown_candidate_does_not_discard_valid_batch_members(self):
        raw={'decisions':[{'id':'1','independent_complete':True,'source_identity_preserved':True,'same_as_readonly':None,'reason':'complete'},
                          {'id':'2','independent_complete':None,'source_identity_preserved':None,'same_as_readonly':None,'reason':'uncertain'}]}
        normalized=validate_identity_decisions(raw,{'1','2'},0)
        self.assertTrue(normalized['decisions'][0]['independent_complete'])
        self.assertFalse(normalized['decisions'][1]['independent_complete'])
        self.assertIsNone(raw['decisions'][1]['independent_complete'])

    def test_same_observation_remains_attached_to_existing_single(self):
        parent=obj('whole',[0,0,100,20]);old=obj('alias',[0,0,100,20]);old['history_relation']={'relation':'same','owner_id':'whole'}
        parent['residual_members']=[old]
        self.assertEqual(absorbed_candidates([parent]),[])
        old.pop('history_relation');self.assertEqual(len(absorbed_candidates([parent])),1)

    def test_geometric_containment_does_not_absorb_identity(self):
        a=obj('whole',[0,0,100,100]);a['kind']='dense_network';b=obj('child',[20,20,30,30]);c=obj('parallel',[20,20,30,30])
        rows,audit=compress([a,b,c],semantic_only=True)
        self.assertEqual(len(rows),3);self.assertFalse(audit['merges']);self.assertFalse(audit['network_members'])

    def test_whole_part_remains_two_tasks_with_readonly_child(self):
        parent=obj('whole',[0,0,90,90]);child=obj('child',[10,10,80,25],True)
        rows,merged,conflicts=reconcile([parent,child],[{'a':'whole','b':'child','relation':'a_contains_b'}])
        self.assertFalse(merged);self.assertFalse(conflicts);self.assertEqual(len(rows),2)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inp=root/'inputs/demo';inp.mkdir(parents=True);Image.new('RGB',(100,100),'white').save(inp/'source.png')
            export_handoff(root,'demo',rows,{'local_scale':3,'local_min_edge':1080,'local_max_edge':2048})
            h=json.loads((root/'cases/demo/reconstruction_handoff.json').read_text());whole=next(t for t in h['tasks'] if t['target_id']=='whole')
            self.assertEqual([x['owner_id'] for x in whole['readonly_identities']],['child'])
            self.assertTrue(whole['readonly_rectangles_are_not_exclusion_masks']);self.assertTrue(whole['requires_review'])
            self.assertEqual(whole['bbox_native'],parent['bbox_native'])

    def test_component_cycles_do_not_create_reciprocal_exclusions(self):
        rows,_,conflicts=reconcile([obj('whole',[0,0,90,90],True),obj('child',[10,10,80,25],True)],
            [{'a':'whole','b':'child','relation':'a_contains_b'},{'a':'whole','b':'child','relation':'b_contains_a'}])
        self.assertEqual(len(conflicts),2);self.assertTrue(all(not o['readonly_component_ids'] for o in rows))

if __name__=='__main__':unittest.main()
