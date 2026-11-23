"""Real dash gaps and missing closing sides require different observations."""
import copy,json,sys,tempfile,unittest
from unittest.mock import MagicMock
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'vendor')]
from shape_integration.fitting.source_spans import inspect


class SourceSpans(unittest.TestCase):
    def test_flagged_source_cannot_be_silently_accepted(self):
        from shape_integration.fitting.critic import validate_review
        with self.assertRaises(ValueError):validate_review({'decision':'accept','issues':[]},['F1'],['span_S0'])
        value={'decision':'accept','issues':[],'source_observations':[
            {'id':'span_S0','source_relation':'missing','evidence':'No source edge crosses the empty region.'}]}
        with self.assertRaises(ValueError):validate_review(value,['F1'],['span_S0'])
        value['source_observations'][0].update(source_relation='occluded',evidence='The opaque source label covers the middle span; both ends are visible.')
        self.assertEqual(validate_review(value,['F1'],['span_S0'])['decision'],'accept')

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.local={'parts':[{'status':'ok','commands':[['M',30,30],['L',330,30],['L',330,230],['L',30,230],['Z']]}]}
        self.job={'proposed_type':'rect','frame':{'actual_crop_input_size_px':[360,260],
            'source_crop_half_open':[0,0,360,260]}}

    def test_long_missing_closing_side_gets_one_source_pair_without_editing(self):
        source=Image.new('RGB',(360,260),'white');draw=ImageDraw.Draw(source)
        draw.line([(30,30),(330,30),(330,230),(30,230)],fill='#C4A472',width=2)
        source.save(self.root/'clean.png');before=copy.deepcopy(self.local)
        checks,images=inspect(self.local,self.job,self.root)
        self.assertEqual(len(checks),1);self.assertEqual(len(images),2)
        self.assertGreater(checks[0]['unsupported_fraction_of_segment'],.7)
        self.assertEqual(before,self.local)
        self.assertFalse(json.loads((self.root/'source_span_checks.json').read_text())['automatic_geometry_changes'])

    def test_real_short_dash_gaps_do_not_create_geometry_findings(self):
        source=Image.new('RGB',(360,260),'white');draw=ImageDraw.Draw(source)
        for x in range(30,331,11):
            draw.line((x,30,min(x+7,330),30),fill='#C4A472',width=2)
            draw.line((x,230,min(x+7,330),230),fill='#C4A472',width=2)
        for y in range(30,231,11):
            draw.line((30,y,30,min(y+7,230)),fill='#C4A472',width=2)
            draw.line((330,y,330,min(y+7,230)),fill='#C4A472',width=2)
        source.save(self.root/'clean.png')
        self.assertEqual(inspect(self.local,self.job,self.root),([],[]))

    def test_missing_subspan_of_a_long_side_is_not_diluted_by_supported_remainder(self):
        source=Image.new('RGB',(360,600),'white');draw=ImageDraw.Draw(source)
        draw.rectangle((30,30,330,530),outline='#C4A472',width=2)
        draw.rectangle((20,300,40,380),fill='white');source.save(self.root/'clean.png')
        local=copy.deepcopy(self.local);local['parts'][0]['commands']=[['M',30,30],['L',330,30],['L',330,530],['L',30,530],['Z']]
        job={'proposed_type':'rect','frame':{'actual_crop_input_size_px':[360,600],'source_crop_half_open':[0,0,360,600]}}
        checks,images=inspect(local,job,self.root)
        self.assertEqual(len(checks),1);self.assertGreater(checks[0]['unsupported_fraction_of_segment'],.1)

    def test_prediction_primed_accept_cannot_override_absent_source_stroke(self):
        from shape_integration.common import save,read
        from shape_integration.fitting.critic import check_object,revision_feedback
        source=Image.new('RGB',(360,260),'white');draw=ImageDraw.Draw(source)
        draw.line([(30,30),(330,30),(330,230),(30,230)],fill='#C4A472',width=2)
        for name in ('clean.png','critic_source.png','fit_overlay.png'):source.save(self.root/name)
        local=copy.deepcopy(self.local)
        local.update(fit_status='ok',observation_status='ok')
        local['parts'][0].update(id='frame',kind='path',observed_point_ids=[])
        save(self.root/'fit_local.json',local);save(self.root/'inspection.json',{'sample_points_shown':False})
        runtime=MagicMock();runtime.spatial_evidence=None;runtime.for_stage.return_value=runtime
        def respond(folder,tag,system,prompt,images,validator,budget,**kwargs):
            if tag=='check':
                return validator({'decision':'accept','issues':[],'source_observations':[
                    {'id':'span_S0','source_relation':'supported','evidence':'It resembles a rectangle with a dashed side.'}]})
            self.assertEqual(len(images),1);self.assertNotIn('fit',images[0][1].name)
            self.assertLessEqual(budget,8192)
            return validator({'visible_line':'no','source_evidence':'The middle of the left side contains only white pixels.'})
        runtime.call.side_effect=respond
        result=check_object({**self.job,'label':'grouping frame'},self.root,runtime,
                            {'source_span_checks':True,'check_max_tokens':100000})
        self.assertEqual(result['decision'],'revise')
        self.assertEqual(result['issues'][0]['part_ids'],['frame'])
        self.assertIn('white pixels',' '.join(revision_feedback(result)))
        self.assertEqual(read(self.root/'critic_primary_review.json')['decision'],'accept')

    def test_source_only_budget_is_shared_across_object_revisions(self):
        from shape_integration.fitting.source_spans import observe_source
        runtime=MagicMock();runtime.for_stage.return_value=runtime
        runtime.call.return_value={'visible_line':'no','source_evidence':'No vertical source pixels.'}
        image=self.root/'source.png';Image.new('RGB',(30,80),'white').save(image)
        first=self.root/'revision_00';first.mkdir()
        second=self.root/'revision_01';second.mkdir()
        options={'check_max_tokens':100000};span={'direction':'vertical'}
        self.assertEqual(observe_source(runtime,first,span,image,options)['visible_line'],'no')
        self.assertEqual(observe_source(runtime,second,span,image,options)['visible_line'],'no')
        Image.new('RGB',(30,80),'black').save(image)
        self.assertEqual(observe_source(runtime,second,span,image,options)['visible_line'],'uncertain')
        self.assertEqual(runtime.call.call_count,1)


if __name__=='__main__':unittest.main()
