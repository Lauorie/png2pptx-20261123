import unittest
import numpy as np
from PIL import Image,ImageDraw
from lib import style


class StyleSemantics(unittest.TestCase):
    def test_large_head_is_excluded_using_pixels_in_both_directions(self):
        for reverse in (False,True):
            image=Image.new('RGB',(400,180),'white');d=ImageDraw.Draw(image)
            d.polygon([(20,90),(160,50),(160,130)],fill='purple')
            for x in range(160,360,50):d.rectangle((x,86,x+29,94),fill='purple')
            mask=np.any(np.array(image)<200,axis=2)
            row=dict(id='L',start=[20,90],end=[389,90],via=[],width_estimate=9,
                     start_head=True,start_head_type='triangle',end_head=False)
            if reverse:
                row['start'],row['end']=row['end'],row['start']
                row.update(start_head=False,start_head_type=None,end_head=True,end_head_type='triangle')
            result=style.recover(np.array(image),mask,row,adaptive_heads=True)
            self.assertEqual(result['dash']['type'],'dashed')
            self.assertAlmostEqual(result['dash']['mark_px'],30,delta=1)
            self.assertAlmostEqual(result['dash']['gap_px'],20,delta=1)
            region=result['evidence']['head_exclusion']['end' if reverse else 'start']
            self.assertGreater(region['extent_px'],100)

    def test_model_category_survives_missing_period_measurement(self):
        measured=dict(type='unknown',preset='solid',reason='Too little visible shaft')
        for kind,preset in [('dashed','dash'),('dotted','sysDot')]:
            result=style.apply_line_style(measured,kind)
            self.assertEqual(result['type'],kind);self.assertEqual(result['preset'],preset)
            self.assertEqual(result['parameter_source'],'preset')
            self.assertNotIn('mark_px',result)
        unresolved=style.apply_line_style(measured,'unknown')
        self.assertEqual(unresolved['type'],'unknown');self.assertIsNone(unresolved['preset'])

    def test_measurement_does_not_override_model_solid(self):
        measured=dict(type='dashed',preset='dash',mark_px=10,gap_px=5)
        result=style.apply_line_style(measured,'solid')
        self.assertEqual(result['type'],'solid');self.assertNotIn('mark_px',result)
        self.assertEqual(result['measurement'],measured)

    def test_solid_with_large_head_stays_solid(self):
        image=Image.new('RGB',(400,180),'white');d=ImageDraw.Draw(image)
        d.line([(20,90),(370,90)],fill='black',width=8)
        d.polygon([(20,90),(140,40),(140,140)],fill='black')
        row=dict(id='L',start=[20,90],end=[370,90],via=[],width_estimate=8,
                 start_head=True,start_head_type='triangle',end_head=False)
        result=style.recover(np.array(image),np.any(np.array(image)<200,axis=2),row,adaptive_heads=True)
        self.assertEqual(result['dash']['type'],'solid')

    def test_mask_wider_than_ink_does_not_turn_entire_shaft_into_head(self):
        from scipy.ndimage import binary_dilation
        image=Image.new('RGB',(400,180),'white');d=ImageDraw.Draw(image)
        d.line([(20,90),(370,90)],fill='black',width=4)
        d.polygon([(20,90),(55,76),(55,104)],fill='black')
        mask=binary_dilation(np.any(np.array(image)<200,axis=2),iterations=4)
        row=dict(id='L',start=[20,90],end=[370,90],via=[],width_estimate=12,
                 start_head=True,start_head_type='triangle',end_head=False)
        result=style.recover(np.array(image),mask,row,adaptive_heads=True)
        self.assertEqual(result['dash']['type'],'solid')
        self.assertLess(result['evidence']['head_exclusion']['start']['extent_px'],60)


if __name__=='__main__':unittest.main()
