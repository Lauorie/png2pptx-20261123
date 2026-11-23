import copy
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'vendor')]
from shape_integration.styling import options
from shape_integration.styling.width_limit import limit, SLIDE_WIDTH_PT


class WidthLimitTests(unittest.TestCase):
    def test_physical_ceiling_preserves_thinner_widths_and_semantics(self):
        opt=options({'styling':{'maximum_stroke_width_pt':2.5}})
        base={'fill':None,'stroke':'#010101','width_px':0.,'dash':'dash','dash_pattern_px':[5.,3.]}
        for width in (500,1415,2048,4096):
            actual=[]
            for points in (.1,.4,.8,1.3,1.8,2.4,2.5,5.48):
                style={**base,'width_px':points*width/SLIDE_WIDTH_PT};before=copy.deepcopy(style)
                result,audit=limit(style,width,opt)
                self.assertEqual(before,style)
                self.assertAlmostEqual(result['width_px']*SLIDE_WIDTH_PT/width,min(points,2.5))
                self.assertEqual(result['dash_pattern_px'],[5.,3.])
                self.assertEqual(result['stroke'],base['stroke']);self.assertIsNone(result['fill'])
                self.assertEqual(bool(audit),points>2.5)
                self.assertEqual(limit(result,width,opt)[0],result)
                actual.append(round(result['width_px']*SLIDE_WIDTH_PT/width,6))
            self.assertEqual(len(set(actual)),7)

    def test_absent_stroke_is_never_created(self):
        style={'fill':'#000000','stroke':None,'width_px':0.,'dash':'solid'}
        self.assertEqual(limit(style,100,options({})),(style,None))

    def test_disabled_cap_preserves_historical_width(self):
        style={'fill':None,'stroke':'#000000','width_px':20.,'dash':'solid'}
        self.assertEqual(limit(style,100,options({'styling':{'maximum_stroke_width_pt':None}})),(style,None))

    def test_default_keeps_source_width_across_image_resolutions(self):
        for source_width in (544,1052,2048):
            style={'fill':None,'stroke':'#000000','width_px':4.,'dash':'solid'}
            self.assertEqual(limit(style,source_width,options({})),(style,None))

    def test_invalid_caps(self):
        for invalid in (0,-1,float('nan'),float('inf'),True,'2.5'):
            with self.assertRaises(ValueError):options({'styling':{'maximum_stroke_width_pt':invalid}})


if __name__=='__main__':unittest.main()
