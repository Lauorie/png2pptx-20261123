"""Regressions for ownership and order at stage boundaries, using synthetic IDs."""
import unittest

from shape_integration.styling.contracts import rows
from shape_integration.styling.controller import order_feedback


class OrderContractTests(unittest.TestCase):
    def test_accepted_style_keeps_source_order_and_cache_rows(self):
        raw={'checks':[{'id':'A','verdict':'accept','behind':['B'],
                        'source_evidence':'B covers the lower edge of A.'},
                       {'id':'B','verdict':'accept','in_front_of':['A']}]}
        table={'A':{},'B':{}}
        result=rows(raw,['A','B'],'critic',table)
        self.assertEqual(result['_errors'],{})
        self.assertEqual(result['_source_rows'],raw['checks'])
        self.assertEqual(rows(result,['A','B'],'critic',table),result)
        value={**result['checks'][0],'_behind_parts':['B']}
        self.assertEqual(order_feedback('A',value,['A','B'])['verdict'],'accept')
        corrected=order_feedback('A',value,['B','A'])
        self.assertEqual(corrected['verdict'],'revise')
        self.assertEqual(corrected['affected_fields'],['occlusion'])

    def test_width_revision_does_not_discard_an_order_correction(self):
        raw={'checks':[{'id':'A','verdict':'revise','behind':['B'],
            'affected_fields':['stroke_width'],'expected_stroke_width':'decrease',
            'source_evidence':'B covers A; visible borders are thin.',
            'rendered_difference':'Border is thick.','suggestion':'Use a thinner border.'}]}
        result=rows(raw,['A'],'critic',{'A':{},'B':{}})
        self.assertEqual(result['_errors'],{})
        value={**result['checks'][0],'_behind_parts':['B']}
        corrected=order_feedback('A',value,['B','A'])
        self.assertEqual(corrected['affected_fields'],['stroke_width','occlusion'])
        self.assertEqual(result['_source_rows'],raw['checks'])

    def test_uncertainty_and_current_order_are_not_overridden(self):
        value={'verdict':'uncertain','_behind_parts':['B']}
        self.assertIs(order_feedback('A',value,['B','A']),value)


class HeadOwnershipTests(unittest.TestCase):
    def test_skewed_small_head_follows_its_touching_shaft(self):
        import copy
        import numpy as np
        from full_recovery.quality_choice import attach_heads,apply_choices
        shaft={'id':'shaft','layer':'shape','owner':'symbol','key':'shaft','keys':['shaft'],
               'refs':['symbol/shaft'],'points':np.asarray([[0.,0.],[20.,0.]]),
               'length':20.,'heads':['none','none'],'style':{'width_px':1.}}
        # Asymmetric triangle at a bend: no median aligns with the shaft.
        tip={'object_id':'symbol','status':'ok','kind':'path',
             'commands':[['M',19.,-1.],['L',21.,3.],['L',22.,-2.],['Z']]}
        result=attach_heads([copy.deepcopy(shaft)],{'parts':{'tip':tip}},{'tip':{'fill':'#000000'}})
        self.assertIn('tip',result[0]['keys'])
        self.assertIn('tip',apply_choices(result,[{'candidate_ids':['shaft'],'keep':'none'}])['drop_shape_keys'])
        # Nearby decoration with a real gap is not owned by this shaft.
        for command in tip['commands']:
            if len(command)>1:command[2]+=8
        result=attach_heads([copy.deepcopy(shaft)],{'parts':{'tip':tip}},{'tip':{'fill':'#000000'}})
        self.assertNotIn('tip',result[0]['keys'])

    def test_nearby_endpoints_do_not_share_a_deleted_head(self):
        import numpy as np
        from full_recovery.quality_choice import attach_heads,apply_choices
        for scale in (1.,2.):
            rows=[]
            for name,points in [('a',[[0,0],[80,30]]),('b',[[30,0],[82,26]])]:
                pts=np.asarray(points,float)*scale
                rows.append({'id':name,'layer':'shape','owner':'symbol','key':name,'keys':[name],
                    'refs':['symbol/'+name],'points':pts,'length':float(np.linalg.norm(pts[1]-pts[0])),
                    'heads':['none','none'],'style':{'width_px':scale}})
            vertices=np.asarray([[80,30],[70,22],[68,28]],float)*scale
            part={'id':'tip','object_id':'symbol','status':'ok','kind':'path',
                'commands':[['M',*vertices[0]],['L',*vertices[1]],['L',*vertices[2]],['Z']]}
            result=attach_heads(rows,{'parts':{'tip':part}},{'tip':{'fill':'#000000'}})
            owners=[r for r in result if 'tip' in r['keys']]
            self.assertEqual(len(owners),1)
            plan=apply_choices(result,[{'candidate_ids':[owners[0]['id']],'keep':'none'}])
            self.assertIn('tip',plan['drop_shape_keys'])
            self.assertIn('symbol/tip',plan['drop_shape_names'])


class LengthRetryTests(unittest.TestCase):
    def test_existing_retry_expands_budget_and_reuses_completed_result(self):
        import json
        import tempfile
        from pathlib import Path
        from shape_integration.common import Runtime
        calls=[]
        class Client:
            def __init__(self,root,budget):self.root=Path(root);self.budget=budget
            def request_parameters(self):return {'max_tokens':self.budget}
            def redact(self,error):return str(error)
            def call(self,name,system,prompt,images,validator):
                folder=self.root/name;folder.mkdir(parents=True)
                calls.append(self.budget)
                if self.budget==8192:
                    (folder/'error_0.json').write_text(json.dumps({'error':'Unfinished stream: length'}))
                    raise RuntimeError('Unfinished stream: length')
                value={'keep':'shape'}
                (folder/'result.json').write_text(json.dumps(value))
                return validator(value)
        runtime=Runtime({'api_concurrency':1,'shape_schema_retries':1},'local','')
        runtime.client=lambda root,budget:Client(root,budget)
        with tempfile.TemporaryDirectory() as root:
            for _ in range(2):
                result=runtime.call(root,'review','system',{},[],lambda d:d,8192)
                self.assertEqual(result,{'keep':'shape'})
        self.assertEqual(calls,[8192,16384])


if __name__=='__main__':unittest.main()
