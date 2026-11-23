"""Visual findings must be actionable and agree with the final decision."""
import copy
import unittest
from test_arrows import network,geometry_verdict,style
from shape_integration.arrows import prompts,contracts


class GeometryReviewTests(unittest.TestCase):
    def setUp(self):self.g=network()

    def test_numeric_coordinates_and_fitting_notes_do_not_enter_review(self):
        self.g['uncertain']=['OLD_COORDINATE_GUESS 9999'];self.g['status']='ready'
        p=prompts.critic(self.g)
        self.assertNotIn('geometry',p);self.assertNotIn('OLD_COORDINATE_GUESS',str(p))
        self.assertEqual(p['paths'][0]['segments'][0]['points'],['a','j'])
        self.assertEqual(list(p['response_format'])[-1],'verdict')
        self.assertNotIn('visually close',str(p));self.assertNotIn('exact pixel',str(p))

    def test_empty_or_contradictory_findings_cannot_bypass_review(self):
        with self.assertRaisesRegex(ValueError,'off_points is required'):
            contracts.geometry_review({'verdict':'accept'},self.g)
        raw=geometry_verdict('revise');raw['verdict']='accept'
        with self.assertRaisesRegex(ValueError,'conflicts'):
            contracts.geometry_review(raw,self.g)
        raw=geometry_verdict();raw['verdict']='revise'
        with self.assertRaisesRegex(ValueError,'at least one'):
            contracts.geometry_review(raw,self.g)
        raw=geometry_verdict('uncertain');raw['reason']=''
        with self.assertRaisesRegex(ValueError,'reason'):
            contracts.geometry_review(raw,self.g)

    def test_path_labels_are_not_sample_points_and_findings_require_evidence(self):
        raw=geometry_verdict('revise');raw['off_points'][0]['id']='trunk'
        with self.assertRaisesRegex(ValueError,'not a sample point'):
            contracts.geometry_review(raw,self.g)
        raw=geometry_verdict('revise');raw['off_points'][0]['move']=''
        with self.assertRaisesRegex(ValueError,'move'):
            contracts.geometry_review(raw,self.g)
        raw=geometry_verdict('revise');raw['off_points'][0]['visible']=''
        with self.assertRaisesRegex(ValueError,'visible'):
            contracts.geometry_review(raw,self.g)

    def test_missing_branch_without_id_and_endpoint_corrections_reach_refine(self):
        raw=geometry_verdict();raw.update(verdict='revise',endpoint_issues=[{'id':'b','visible':'source tip continues above fitted endpoint','move':'extend upward'}],
            missing_sections=[{'visible':'source has a branch below the junction','fix':'add the lower branch'}])
        result=contracts.geometry_review(raw,self.g)
        self.assertIn('extend upward',result['suggestion']);self.assertIn('add the lower branch',result['suggestion'])
        self.assertIn('source has a branch',result['evidence'])
        self.assertEqual(contracts.geometry_review(result,self.g),result)

    def test_nonterminal_ids_are_rejected_for_endpoint_issues(self):
        g=copy.deepcopy(self.g);g['points']['middle']=[60,100];g['paths'][0]['segments'][0]['points']=['a','middle','j']
        raw=geometry_verdict();raw.update(verdict='revise',endpoint_issues=[{'id':'middle','visible':'wrong point','move':'up'}])
        with self.assertRaisesRegex(ValueError,'not a path endpoint'):
            contracts.geometry_review(raw,g)

    def test_style_review_keeps_legacy_contract(self):
        p=prompts.critic(self.g,style(self.g),size=[240,180])
        self.assertIn('current_style',p);self.assertNotIn('off_points',p['response_format'])
        self.assertEqual(contracts.review({'verdict':'accept'}),{'verdict':'accept'})


if __name__=='__main__':unittest.main()
