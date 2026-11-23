"""Missing SAM stroke, antialias coverage and ambiguous source support controls."""
from pathlib import Path
import sys
import unittest
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments/final_shape_critic_20261020'),str(ROOT/'src')]
from native_corner_probe import observation
from native_corner_mask_fit import refine


class CornerMaskControls(unittest.TestCase):
    def fixture(self):
        points=np.array([[20,20],[110,20],[110,70],[200,70],[200,140],[20,140]],float)
        source=Image.new('RGB',(240,160),'white');mask=Image.new('L',source.size,0)
        rgb=ImageDraw.Draw(source);md=ImageDraw.Draw(mask);core=np.array([160,100,40])
        for i,a in enumerate(points):
            b=points[(i+1)%6];missing=i==2
            colour=tuple(np.rint(255+.4*(core-255)).astype(int)) if missing else tuple(core)
            rgb.line([tuple(a),tuple(b)],fill=colour,width=2)
            if not missing:md.line([tuple(a),tuple(b)],fill=255,width=2)
        return observation(points+3),np.asarray(mask)>0,np.asarray(source),points

    def test_missing_edge_can_use_antialiased_source_with_sam_colour_anchor(self):
        raw,mask,rgb,target=self.fixture();after,audit=refine(raw,mask,rgb)
        self.assertEqual(sum(a['accepted'] for a in audit),6)
        self.assertEqual(audit[2]['origin'],'RGB_with_SAM_colour_anchor')
        actual=np.array([p['xy'] for p in after['points']])
        self.assertLess(np.max(abs(actual-target)),1.1)

    def test_no_sam_foreground_cannot_establish_source_colour_ownership(self):
        raw,mask,rgb,_=self.fixture();after,audit=refine(raw,np.zeros_like(mask),rgb)
        self.assertEqual(after,raw);self.assertFalse(any(a['accepted'] for a in audit))

    def test_parallel_colour_matches_remain_uncertain(self):
        raw,mask,rgb,_=self.fixture();im=Image.fromarray(rgb)
        colour=tuple(np.rint(255+.4*(np.array([160,100,40])-255)).astype(int))
        ImageDraw.Draw(im).line([(110,75),(200,75)],fill=colour,width=2)
        after,audit=refine(raw,mask,np.asarray(im))
        self.assertFalse(audit[2]['accepted'])
        self.assertEqual(audit[2]['reason'],'Ambiguous parallel source-colour strokes')

    def test_filled_mask_is_not_a_localized_stroke(self):
        raw,mask,rgb,_=self.fixture();after,audit=refine(raw,np.ones_like(mask),rgb)
        self.assertEqual(after,raw);self.assertFalse(any(a['accepted'] for a in audit))


if __name__=='__main__':unittest.main()
