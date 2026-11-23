import unittest
import contracts as C
from test_localization import region
from fusion_geometry import plan_pair


class FusionSafety(unittest.TestCase):
    def test_group_consumes_sources_and_keeps_all_observation_evidence(self):
        a=region('A');a['observation_ids']=['G1','L1']
        b=region('B',[80,30,170,150]);b['observation_ids']=['G2','L2']
        answer={'frame':'GLOBAL','keep_ids':[], 'replacements':[{'source_ids':['A','B'],
            'regions':[{**region('group',[20,20,170,150]),'grouping':'group'}],
            'reason':'two distinct branches recovered together'}],'discard':[],'inspect':[]}
        C.review_valid(answer,[200,200],[a,b],False)
        rows=C.apply_review([a,b],answer,'U')
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['observation_ids'],['G1','G2','L1','L2'])
        self.assertNotIn(rows[0]['id'],['A','B'])

    def test_pair_shares_only_exact_inputs_not_nearby_or_relabelled_targets(self):
        a={'target_id':'R1','region_id':'A','display_id':1,'bbox_native':[1,2,30,40],
           'identity_hint':'visible branch','grouping':'single','render_type':'linework'}
        same={**a,'target_id':'R9','display_id':9}
        moved={**a,'target_id':'R2','region_id':'B','bbox_native':[1,2,31,40]}
        grouped={**a,'target_id':'R3','region_id':'C','grouping':'group'}
        tasks,arms,maps=plan_pair({'tasks':[a]}, {'tasks':[same,moved,grouped]})
        self.assertEqual(len(tasks),3)
        self.assertEqual(arms['before'][0],arms['after'][0])
        self.assertEqual(len(set(arms['after'])),3)
        self.assertEqual(maps['after'][0]['original_target_id'],'R9')


if __name__=='__main__':unittest.main()
