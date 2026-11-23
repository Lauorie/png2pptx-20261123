import unittest,tempfile
from pathlib import Path
from PIL import Image
from full_recovery.text_orientation import augment,bind_validator,quarter_turn_ids

class OrientationEvidenceTests(unittest.TestCase):
    def test_upright_narrow_glyph_and_stacked_label_are_valid_choices(self):
        prompt={'stage':'2_attributes_at_locked_boxes','targets':[
            {'id':'narrow','kind':'text','rotation_deg':0,'locked_bbox_2d':[1,2,6,22]},
            {'id':'formula','kind':'formula','rotation_deg':0,'locked_bbox_2d':[1,2,6,22]}]}
        self.assertEqual(quarter_turn_ids(prompt),{'narrow'})
        validate=bind_validator(lambda raw:raw,{'narrow'})
        row={'id':'narrow','status':'resolved','units':[{'text':'MODULE','orientation_candidate':'C',
            'rotation_deg':90,'layout_mode':'stacked_vertical','visual_line_count':1}]}
        result=validate({'items':[row]})['items'][0]['units'][0]
        self.assertEqual(result['rotation_deg'],0)
        self.assertEqual(result['layout_mode'],'stacked_vertical')

    def test_angles_have_one_authority_and_cache_keeps_the_choice(self):
        def base(raw):
            rows=[]
            for r in raw['items']:
                rows.append({**r['units'][0],'id':r['id'],'status':'located'} if r['status']=='resolved' else r)
            return {'items':rows}
        validate=bind_validator(base,{'label'})
        raw={'items':[{'id':'label','status':'resolved','units':[{'text':'word','orientation_candidate':'A','rotation_deg':-90,'layout_mode':'stacked_vertical','visual_line_count':1}]}]}
        result=validate(raw)
        self.assertEqual(result['items'][0]['rotation_deg'],90)
        self.assertEqual(result['items'][0]['layout_mode'],'single_line')
        self.assertEqual(validate(result),result)
        self.assertEqual(raw['items'][0]['units'][0]['rotation_deg'],-90)
        raw['items'][0]['units'][0]['layout_mode']='multiline'
        self.assertEqual(validate(raw)['items'][0]['layout_mode'],'single_line')
        raw['items'][0]['units'][0]['visual_line_count']=2
        self.assertEqual(validate(raw)['items'][0]['layout_mode'],'multiline')
        with self.assertRaises(ValueError):
            validate({'items':[{'id':'label','status':'resolved','units':[{'rotation_deg':90}]}]})

    def test_existing_attribute_call_gets_readable_evidence_without_prior_angle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.png';Image.new('RGB',(100,200),'white').save(source)
            prompt={'stage':'2_attributes_at_locked_boxes','targets':[{'id':'label','kind':'text','rotation_deg':-90,'locked_bbox_2d':[20,30,40,150]}],
                    'rules':['ROTATION IS NOT LOCKED. long explanation'],
                    'response_format':{'items':[{'units':[{'rotation_deg':90,'text':'word'}]}]}}
            output,images=augment(root/'call',prompt,[('CLEAN',source)])
            self.assertEqual(len(images),1)
            self.assertEqual(len(output['rules']),4)
            self.assertNotIn('rotation_deg',output['targets'][0])
            self.assertIn('orientation_candidate',output['response_format']['items'][0]['units'][0])
            self.assertEqual(prompt['targets'][0]['rotation_deg'],-90)
            self.assertTrue(images[0][1].is_file())
            self.assertEqual(augment(root,{'stage':'geometry'},[('CLEAN',source)]),({'stage':'geometry'},[('CLEAN',source)]))
