from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image,ImageDraw
from shape_integration.styling.native_stroke import NativeFitter
from shape_integration.styling.views import sam_pattern_views


class VisualPatternTests(unittest.TestCase):
    def test_only_explicit_visual_evidence_overrides_reliable_numeric_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.png';im=Image.new('RGB',(180,110),'white')
            ImageDraw.Draw(im).rectangle((20,20,160,90),outline='black',width=3);im.save(source)
            key='p';table={key:{'id':'p','object_id':'o','kind':'path','status':'ok',
                'commands':[['M',20,20],['L',160,20],['L',160,90],['L',20,90],['Z']]}}
            fitter=NativeFitter(source,table,None,'case',root)
            style={'fill':None,'stroke':'#000000','width_px':3,'dash':'solid'}
            hint={'verdict':'revise','affected_fields':['line_pattern'],'expected_line_pattern':'solid'}
            with patch('shape_integration.styling.native_stroke.classify_pattern',
                       return_value={'dash':'dash','dash_reliable':True,'dash_pattern_px':[8.,4.]}):
                numeric,n=fitter.fit(key,style,feedback=[hint])
                visual,v=fitter.fit(key,style,feedback=[{**hint,'sam_visual_evidence':'matched-images-digest'}])
            self.assertEqual(numeric['dash'],'dash')
            self.assertEqual(visual['dash'],'solid');self.assertNotIn('dash_pattern_px',visual)
            self.assertAlmostEqual(numeric['width_px'],visual['width_px'])
            self.assertEqual(v['pattern_origin'],'glm_sam_rgb_visual_observation')
            self.assertEqual(visual['fill'],style['fill']);self.assertEqual(visual['stroke'],numeric['stroke'])

    def test_mask_views_use_existing_aligned_tiles_and_no_empty_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);tile=root/'tile';tile.mkdir()
            Image.new('RGB',(24,24),'white').save(tile/'tile.png');Image.new('L',(24,24),255).save(tile/'mask.png')
            evidence={'parts':{'p':{'tiles':[{'target':True,'target_mask':str(tile/'mask.png'),
                'dir':str(tile),'box':[0,0,24]}]}}}
            images,audit=sam_pattern_views(evidence,'p',root/'views')
            self.assertEqual(len(images),2);self.assertEqual(len(audit),1)
            self.assertEqual(sam_pattern_views(evidence,'absent',root/'empty'),([],[]))


if __name__=='__main__':unittest.main()
