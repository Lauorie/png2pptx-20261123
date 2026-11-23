"""Independent raster truth and actual PPT tests for native SAM stroke fitting."""
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import os
import unittest
import numpy as np
from PIL import Image
import cairosvg
from scipy.ndimage import label, find_objects

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / 'src'
if '--src' in sys.argv:
    index=sys.argv.index('--src');SRC=Path(sys.argv[index+1]);del sys.argv[index:index+2]
sys.path[:0] = [str(SRC), str(HERE.parent / 'vendor')]
from shape_integration.styling.native_stroke import NativeFitter, fit_profiles, sample
from shape_integration.styling import contracts, model, options
from shape_integration.styling import occlusion
from shape_integration.styling.export import export,apply_style,render_native,NS
from shape_integration.fitting.scene import write_pptx
from pptx.oxml.xmlchemy import OxmlElement
from shape_integration.common import sha


def raster(svg):
    return Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg.encode()))).convert('RGB')


def ellipse(radius=4):
    angles = np.linspace(0, 2 * np.pi, 97)[:-1]
    p = np.c_[16 + radius * np.cos(angles), 16 + radius * np.sin(angles)]
    return {'id': 'a', 'object_id': 'A', 'kind': 'ellipse',
            'commands': [['M', *p[0]], *[['L', *q] for q in p[1:]], ['Z']]}


class NativeFits(unittest.TestCase):
    def test_nested_frame_fill_does_not_join_two_strokes_into_one_width(self):
        offsets=np.arange(-16.,16.001,1/10.5)
        background=np.where((offsets>-8)&(offsets<0),204.,255.)
        own=np.clip(1.5/2+.5-abs(offsets),0,1)
        outer=np.clip(2./2+.5-abs(offsets+8),0,1)
        grey=background*(1-np.maximum(own,outer))
        rgb=np.repeat(np.broadcast_to(grey,(48,len(offsets)))[:,:,None],3,axis=2)
        result=fit_profiles(rgb,offsets,np.zeros((48,len(offsets))),np.zeros(48,bool),[0,0,0])
        self.assertTrue(result['width_reliable'],result)
        self.assertAlmostEqual(result['width_px'],1.5,delta=.2,msg=result)

    def test_stable_core_of_a_genuinely_wide_line_is_not_local_background(self):
        offsets=np.arange(-16.,16.001,1/10.5)
        for pigment in (0.,120.,200.):
            alpha=np.clip(12./2+.5-abs(offsets),0,1)
            grey=255*(1-alpha)+pigment*alpha
            rgb=np.repeat(np.broadcast_to(grey,(48,len(offsets)))[:,:,None],3,axis=2)
            result=fit_profiles(rgb,offsets,np.zeros((48,len(offsets))),np.zeros(48,bool),[pigment]*3)
            self.assertAlmostEqual(result['width_px'],12.,delta=.3,msg=result)

    def test_adjacent_parallel_ink_does_not_set_background_at_native_sample_densities(self):
        for density in (4.,10.5,21.):
            with self.subTest(density=density):
                offsets=np.arange(-16.,16.001,1/density)
                centre=np.clip(1.6/2+.5-abs(offsets),0,1)
                neighbors=np.maximum(np.clip(1.5/2+.5-abs(offsets-15.6),0,1),
                                     np.clip(1.5/2+.5-abs(offsets+15.6),0,1))
                alpha=np.broadcast_to(np.maximum(centre,neighbors),(24,len(offsets)))
                rgb=np.repeat((255*(1-alpha))[:,:,None],3,axis=2)
                result=fit_profiles(rgb,offsets,np.zeros(alpha.shape),np.zeros(24,bool),[0,0,0],scale=density)
                self.assertTrue(result['width_reliable'],result)
                self.assertAlmostEqual(result['width_px'],1.6,delta=.15,msg=result)

    def test_source_subpixel_stroke_remains_subpixel_after_background_correction(self):
        offsets=np.arange(-16.,16.001,1/10.5)
        alpha=np.broadcast_to(np.clip(.2/2+.5-abs(offsets),0,1),(24,len(offsets)))
        rgb=np.repeat((255*(1-alpha))[:,:,None],3,axis=2)
        result=fit_profiles(rgb,offsets,np.zeros(alpha.shape),np.zeros(24,bool),[0,0,0])
        self.assertAlmostEqual(result['width_px'],.2,delta=.05,msg=result)

    def test_broad_profiles_outside_template_range_do_not_force_a_line_width(self):
        offsets=np.arange(-16.,16.001,1/10.5)
        alpha=np.broadcast_to(np.clip(15.5-abs(offsets),0,1),(24,len(offsets)))
        rgb=np.repeat((255*(1-alpha))[:,:,None],3,axis=2)
        result=fit_profiles(rgb,offsets,np.zeros(alpha.shape),np.zeros(24,bool),[0,0,0])
        self.assertFalse(result['measured'],result)
        self.assertTrue(result['present'],result)
        self.assertFalse(result['width_reliable'],result)
        self.assertEqual(result['reason'],'observed_profile_outside_supported_width_range')
        self.assertGreater(result['rgb_width_px'],29)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.png'
            raster('<svg xmlns="http://www.w3.org/2000/svg" width="200" height="70"><rect width="200" height="70" fill="white"/><path d="M10 35H190" stroke="black" stroke-width="30"/></svg>').save(source)
            part={'id':'a','object_id':'A','kind':'path','commands':[['M',10,34.5],['L',190,34.5]]}
            out,audit=NativeFitter(source,{'k':part},None,'test',root).fit('k',
                {'fill':None,'stroke':'#000000','width_px':1.7,'dash':'solid'})
            self.assertEqual(out['width_px'],1.7,audit)
            self.assertFalse(audit['width_measurement_applied'],audit)

    def test_critic_narrower_mode_is_not_reversed_by_wide_sam_band(self):
        offsets=np.arange(-8.,8.001,1/10.5)
        widths=np.r_[np.ones(32),np.full(32,3.)]
        alpha=np.clip(widths[:,None]/2+.5-abs(offsets),0,1)
        rgb=np.repeat((255*(1-alpha))[:,:,None],3,axis=2)
        mask=np.broadcast_to(abs(offsets)<1.5,alpha.shape)
        result=fit_profiles(rgb,offsets,mask,np.ones(64,bool),[0,0,0],width_preference='decrease')
        self.assertLess(result['width_px'],1.5,result)
        self.assertLess(result['rgb_width_px'],1.5,result)

    def test_out_of_range_profile_does_not_become_a_reliable_capped_width(self):
        offsets=np.arange(-16.,16.001,1/10.5)
        for width in (17.,21.,26.):
            alpha=np.broadcast_to(np.clip(width/2+.5-abs(offsets),0,1),(48,len(offsets)))
            rgb=np.repeat((255*(1-alpha))[:,:,None],3,axis=2)
            result=fit_profiles(rgb,offsets,np.zeros(alpha.shape),np.zeros(48,bool),[0,0,0],
                                review=True,width_preference='increase')
            self.assertFalse(result['measured'],result)
            self.assertFalse(result['width_reliable'],result)
            self.assertGreater(result['rgb_width_px'],16.,result)

    def test_source_grounded_critic_can_correct_numeric_pattern_conflict(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.png'
            raster('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="80"><rect width="300" height="80" fill="white"/><path d="M10 40H290" stroke="black" stroke-width="2"/></svg>').save(source)
            part={'id':'a','object_id':'A','kind':'path','commands':[['M',10,39.5],['L',290,39.5]]}
            fitter=NativeFitter(source,{'k':part},None,'test',root)
            with patch('shape_integration.styling.native_stroke.classify_pattern',return_value={'dash':'dash','dash_reliable':True,'dash_pattern_px':[4.,22.]}):
                out,audit=fitter.fit('k',{'fill':None,'stroke':'#000000','width_px':2.,'dash':'dash'},feedback=[{
                    'verdict':'revise','affected_fields':['line_pattern'],'expected_line_pattern':'solid',
                    'source_evidence':'The source contour is continuous.',
                    'rendered_difference':'The rendered contour has long missing segments.'}])
            self.assertEqual(out['dash'],'solid',audit)
            self.assertNotIn('dash_pattern_px',out)
            self.assertEqual(audit['numeric_pattern'],'dash')

    def test_source_selected_pattern_resolves_conflicting_numeric_ink_and_has_its_own_cache_identity(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.png'
            raster('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="80"><rect width="300" height="80" fill="white"/><path d="M10 40H290" stroke="black" stroke-width="2"/></svg>').save(source)
            part={'id':'a','object_id':'A','kind':'path','commands':[['M',10,40],['L',290,40]]}
            fitter=NativeFitter(source,{'k':part},None,'test',root)
            style={'fill':None,'stroke':'#000000','width_px':2.,'dash':'solid'}
            with patch('shape_integration.styling.native_stroke.classify_pattern',return_value={
                    'dash':'dash_dot','dash_reliable':True,'dash_pattern_px':[8.,6.,2.,6.],
                    'pattern_foreign_samples':7}):
                baseline,_=fitter.fit('k',style)
                observed,audit=fitter.fit('k',style,source_pattern='solid')
                corrected,_=fitter.fit('k',style,source_pattern='dash_dot',feedback=[{
                    'verdict':'revise','affected_fields':['line_pattern'],'expected_line_pattern':'solid',
                    'source_evidence':'The source is a continuous stroke.',
                    'rendered_difference':'The numeric pattern introduces missing segments.'}])
            self.assertEqual(baseline['dash'],'dash_dot')
            self.assertEqual(observed['dash'],'solid',audit)
            self.assertNotIn('dash_pattern_px',observed)
            self.assertEqual(observed['width_px'],baseline['width_px'])
            self.assertEqual(audit['numeric_pattern'],'dash_dot')
            self.assertEqual(corrected['dash'],'solid')

    def test_clean_measured_dashes_survive_a_wrong_solid_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.png'
            raster('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="80"><rect width="300" height="80" fill="white"/><path d="M10 40H290" stroke="black" stroke-width="2" stroke-dasharray="8 6"/></svg>').save(source)
            part={'id':'a','object_id':'A','kind':'path','commands':[['M',10,40],['L',290,40]]}
            out,audit=NativeFitter(source,{'k':part},None,'test',root).fit('k',
                {'fill':None,'stroke':'#000000','width_px':2.,'dash':'solid'},source_pattern='solid')
            self.assertEqual(out['dash'],'dash',audit)
            self.assertTrue(audit['dash_reliable'],audit)

    def test_visual_dots_cannot_replace_a_measured_continuous_contour(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.png'
            raster('<svg xmlns="http://www.w3.org/2000/svg" width="140" height="140"><rect width="140" height="140" fill="white"/><path d="M20 110L70 20" stroke="#555555" stroke-width="2"/></svg>').save(source)
            part={'id':'a','object_id':'A','kind':'path','commands':[['M',20,110],['L',70,20]]}
            out,audit=NativeFitter(source,{'k':part},None,'test',root).fit('k',
                {'fill':None,'stroke':'#555555','width_px':2.,'dash':'solid'},feedback=[{
                    'verdict':'revise','affected_fields':['line_pattern'],'expected_line_pattern':'dot',
                    'source_evidence':'Repeated marks on the enlarged diagonal.',
                    'rendered_difference':'The rendered diagonal looks solid.'}])
            self.assertTrue(audit['dash_reliable'],audit)
            self.assertEqual(out['dash'],'solid',audit)

    def test_rare_crossing_profiles_do_not_certify_a_whole_stroke(self):
        offsets=np.arange(-16.,16.001,1/10.5)
        rgb=np.full((309,len(offsets),3),255.,dtype=float)
        rgb[:12,abs(offsets)<8,:]=0.
        result=fit_profiles(rgb,offsets,np.zeros(rgb.shape[:2]),np.zeros(309,bool),np.zeros(3))
        self.assertTrue(result['measured'])
        self.assertFalse(result['width_reliable'],result)

    def test_unreliable_fit_does_not_replace_width_with_neighbouring_ink(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.png';Image.new('RGB',(120,80),'white').save(source)
            part={'id':'a','object_id':'A','kind':'path','commands':[['M',10,40],['L',110,40]]}
            fitter=NativeFitter(source,{'k':part},None,'test',root)
            measured={'measured':True,'present':True,'width_px':16.,'width_reliable':False,'source_colour':'#000000'}
            with patch('shape_integration.styling.native_stroke.fit_profiles',return_value=measured):
                out,evidence=fitter.fit('k',{'fill':None,'stroke':'#000000','width_px':1.7,'dash':'solid'})
            self.assertEqual(out['width_px'],1.7)
            self.assertFalse(evidence['width_measurement_applied'])

    def test_thin_antialias_core_does_not_recolour_the_stroke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.png'
            raster('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="60"><rect width="300" height="60" fill="white"/><path d="M10 30H290" stroke="#0F2942" stroke-width=".6"/></svg>').save(source)
            part={'id':'a','object_id':'A','kind':'path','commands':[['M',10,29.5],['L',290,29.5]]}
            fitter=NativeFitter(source,{'k':part},None,'test',root)
            style={'fill':None,'stroke':'#0F2942','width_px':.6,'dash':'solid'}
            out,audit=fitter.supported_stroke_colour(['k'],style)
            self.assertEqual(out['stroke'],style['stroke'],audit)

    def test_pigment_validation_does_not_measure_an_occluding_foreground(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.png'
            raster('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="60"><rect width="300" height="60" fill="white"/><path d="M10 30H290" stroke="#808080" stroke-width="2"/><rect x="30" y="25" width="240" height="10" fill="#502020"/></svg>').save(source)
            part={'id':'a','object_id':'A','kind':'path','commands':[['M',10,29.5],['L',290,29.5]]}
            fitter=NativeFitter(source,{'k':part},None,'test',root)
            style={'fill':None,'stroke':'#808080','width_px':2.,'dash':'solid'}
            out,audit=fitter.supported_stroke_colour(['k'],style,excluded=lambda p:(p[:,0]>30)&(p[:,0]<270))
            self.assertEqual(out['stroke'],'#808080',audit)

    def test_open_unfilled_border_validates_pigment_before_width_fitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.png'
            raster('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="60"><rect width="300" height="60" fill="white"/><path d="M10 30H290" stroke="#C4A472" stroke-width="2" stroke-dasharray="7,4"/></svg>').save(source)
            part={'id':'a','object_id':'A','kind':'path','commands':[['M',10,29.5],['L',290,29.5]]}
            fitter=NativeFitter(source,{'k':part},None,'test',root)
            style,audit=fitter.supported_stroke_colour(['k'],{'fill':None,'stroke':'#DFA943','width_px':.08,'dash':'solid'})
            self.assertTrue(audit['changed'],audit)
            self.assertEqual(style['stroke'],'#C4A472',audit)
            out,m=fitter.fit('k',style)
            self.assertAlmostEqual(out['width_px'],2.,delta=.3,msg=m)
            self.assertEqual(out['dash'],'dash',m)

    def test_containing_face_fill_cannot_replace_a_source_reviewed_open_stroke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.png'
            raster('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="60"><rect width="300" height="60" fill="white"/><rect x="10" y="25" width="280" height="10" fill="#B381B3"/></svg>').save(source)
            part={'id':'a','object_id':'A','kind':'path','commands':[['M',10,29.5],['L',290,29.5]]}
            face={'id':'b','object_id':'A','kind':'path','commands':[['M',10,25],['L',290,25],['L',290,35],['L',10,35],['Z']]}
            fitter=NativeFitter(source,{'k':part,'face':face},None,'test',root)
            style={'fill':None,'stroke':'#000000','width_px':1.5,'dash':'solid'}
            before,audit=fitter.supported_stroke_colour(['k'],style)
            self.assertTrue(audit['changed'],audit)
            self.assertEqual(before['stroke'],'#B381B3')
            after,audit=fitter.supported_stroke_colour(['k'],style,paint={'face':{'fill':'#B381B3','fill_opacity':1.}})
            self.assertEqual(after,style,audit)
            self.assertFalse(audit['changed'],audit)
            self.assertEqual(audit['reason'],'observed_core_matches_existing_face_fill')

    def test_distant_or_transparent_face_does_not_veto_a_pigment_correction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.png'
            raster('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="60"><rect width="300" height="60" fill="white"/><path d="M10 30H290" stroke="#B381B3" stroke-width="2"/></svg>').save(source)
            part={'id':'a','object_id':'A','kind':'path','commands':[['M',10,29.5],['L',290,29.5]]}
            for y,opacity in [(5,1.),(25,.5)]:
                with self.subTest(face_y=y,opacity=opacity):
                    face={'id':'b','object_id':'B','kind':'path','commands':[['M',10,y],['L',290,y],['L',290,y+10],['L',10,y+10],['Z']]}
                    fitter=NativeFitter(source,{'k':part,'face':face},None,'test',root)
                    after,audit=fitter.supported_stroke_colour(['k'],{'fill':None,'stroke':'#DFA943'},
                        paint={'face':{'fill':'#B381B3','fill_opacity':opacity}})
                    self.assertTrue(audit['changed'],audit)
                    self.assertEqual(after['stroke'],'#B381B3')

    def test_contrasting_stroke_core_over_a_face_still_corrects_wrong_pigment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.png'
            raster('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="60"><rect width="300" height="60" fill="white"/><rect x="10" y="25" width="280" height="10" fill="#B381B3"/><path d="M10 30H290" stroke="#202020" stroke-width="2"/></svg>').save(source)
            part={'id':'a','object_id':'A','kind':'path','commands':[['M',10,29.5],['L',290,29.5]]}
            face={'id':'b','object_id':'A','kind':'path','commands':[['M',10,25],['L',290,25],['L',290,35],['L',10,35],['Z']]}
            fitter=NativeFitter(source,{'k':part,'face':face},None,'test',root)
            after,audit=fitter.supported_stroke_colour(['k'],{'fill':None,'stroke':'#DFA943'},
                paint={'face':{'fill':'#B381B3'}})
            self.assertTrue(audit['changed'],audit)
            self.assertEqual(after['stroke'],'#202020')

    def test_two_backgrounds_cannot_recolour_gray_border_white(self):
        offsets=np.arange(-8.,8.001,1/10.5)
        rgb=np.full((48,len(offsets),3),40.,dtype=np.float32)
        rgb[:,offsets>=0]=255.
        # The white outer background projects beyond the requested gray from
        # the mean of two backgrounds; a real gray band must stay gray.
        rgb[:,abs(offsets-.5)<=.75]=132.
        result=fit_profiles(rgb,offsets,np.zeros(rgb.shape[:2]),np.zeros(len(rgb),bool),[132,132,132])
        self.assertTrue(result['measured'],result)
        self.assertEqual(result['source_colour'],'#848484',result)
        self.assertFalse(result['measurement_colour_calibrated'],result)

    def test_width_feedback_cannot_move_in_the_opposite_direction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.png'
            raster('<svg xmlns="http://www.w3.org/2000/svg" width="200" height="50"><rect width="200" height="50" fill="white"/><path d="M10 25H190" stroke="black" stroke-width="2"/></svg>').save(source)
            part={'id':'a','object_id':'A','kind':'path','commands':[['M',10,24.5],['L',190,24.5]]}
            out,m=NativeFitter(source,{'k':part},None,'test',root).fit('k',
                {'fill':None,'stroke':'#000000','width_px':3.5,'dash':'solid'},
                feedback=[{'verdict':'revise','affected_fields':['stroke_width'],'expected_stroke_width':'increase'}])
            self.assertEqual(out['width_px'],3.5)
            self.assertFalse(m['width_revision_applied'])

    def test_source_width_direction_selects_an_observed_mode_without_scaling(self):
        offsets=np.arange(-8.,8.001,1/10.5)
        source=raster('<svg xmlns="http://www.w3.org/2000/svg" width="200" height="70"><rect width="200" height="70" fill="white"/><path d="M0 32.5H140" stroke="black" stroke-width="3.5"/><path d="M140 32.5H200" stroke="black" stroke-width="1"/></svg>')
        rgb=sample(np.asarray(source,np.float32),np.stack(np.broadcast_arrays(np.linspace(5,195,80)[:,None],32.+offsets[None]),-1))
        args=(rgb,offsets,np.zeros(rgb.shape[:2]),np.zeros(80,bool),[0,0,0])
        thin=fit_profiles(*args,review=True,width_preference='decrease')
        thick=fit_profiles(*args,review=True,width_preference='increase')
        self.assertAlmostEqual(thin['width_px'],1.,delta=.3,msg=thin)
        self.assertAlmostEqual(thick['width_px'],3.5,delta=.3,msg=thick)

    def test_pixel_repeats_cannot_be_round_dots_wider_than_the_marks(self):
        from shape_integration.styling.lines import classify
        result=classify([1.]*10,[1.]*11,.5,.5,4.)
        self.assertFalse(result['dash_reliable'],result)
        self.assertEqual(classify([1.]*10,[2.]*11,1/3,1/3,1.)['dash'],'dot')

    def test_nearby_occluder_does_not_erase_a_visible_border(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.png'
            raster('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="80"><rect width="300" height="80" fill="white"/><path d="M10 40H290" stroke="#444444" stroke-width="2" stroke-dasharray="8,4"/><rect x="0" y="46" width="300" height="30" fill="#88AAEE"/></svg>').save(source)
            part={'id':'a','object_id':'A','kind':'path','commands':[['M',10,39.5],['L',290,39.5]]}
            out,m=NativeFitter(source,{'k':part},None,'test',root).fit('k',
                {'fill':None,'stroke':'#444444','width_px':2,'dash':'solid'},excluded=lambda xy:xy[:,1]>=46)
            self.assertTrue(m['measured'],m)
            self.assertEqual(out['dash'],'dash',m)
            self.assertAlmostEqual(out['width_px'],2,delta=.3,msg=m)

    def test_mask_holes_cannot_turn_source_solid_into_dashes(self):
        from shape_integration.styling.native_stroke import classify_pattern
        from shape_integration.styling import lines
        offsets=np.arange(-8.,8.001,.5);n=400
        alpha=np.broadcast_to(np.clip(1.5-abs(offsets),0,1),(n,len(offsets)))
        rgb=255*(1-alpha[...,None])*np.ones(3)
        mask=(abs(offsets)[None,:]<1)&((np.arange(n)//8)%2==0)[:,None]
        rows=[{'indices':np.arange(n),'corner':np.zeros(n,bool),'step':.5}]
        m=classify_pattern(rgb,offsets,[0,0,0],rows,0,2,mask,np.ones(mask.shape,bool))
        result=lines.classify(m['on_lengths'],m['gap_lengths'],m['coverage'],m['coverage'],2)
        self.assertEqual(result['dash'],'solid',m)

    def test_unsupported_stroke_colour_uses_shared_visible_border_core(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.png'
            raster('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="80"><rect width="300" height="80" fill="white"/><path d="M10 20H290V70H10Z" fill="#C5E0B4" stroke="#33456E" stroke-width="2"/></svg>').save(source)
            part={'id':'a','object_id':'A','kind':'path','commands':[['M',10,19.5],['L',290,19.5],['L',290,69.5],['L',10,69.5],['Z']]}
            fitter=NativeFitter(source,{'k':part},None,'test',root)
            result,audit=fitter.supported_stroke_colour(['k'],{'fill':'#C5E0B4','stroke':'#548235'})
            self.assertTrue(audit['changed'],audit)
            self.assertLess(np.linalg.norm(np.array([int(result['stroke'][i:i+2],16) for i in (1,3,5)])-[51,69,110]),25)
            good,audit=fitter.supported_stroke_colour(['k'],{'fill':'#C5E0B4','stroke':'#33456E'})
            self.assertEqual(good['stroke'],'#33456E');self.assertFalse(audit['changed'])

    def test_supported_light_border_is_not_replaced_by_dark_shadow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.png'
            raster('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="100"><rect width="300" height="100" fill="white"/><path d="M12 22H292V82H12Z" fill="none" stroke="#999999" stroke-width="4"/><path d="M10 20H290V80H10Z" fill="#DDEAF7" stroke="#A6CAEC" stroke-width="2"/></svg>').save(source)
            part={'id':'a','object_id':'A','kind':'path','commands':[['M',10,19.5],['L',290,19.5],['L',290,79.5],['L',10,79.5],['Z']]}
            fitter=NativeFitter(source,{'k':part},None,'test',root)
            result,audit=fitter.supported_stroke_colour(['k'],{'fill':'#DDEAF7','stroke':'#A6CAEC'})
            self.assertEqual(result['stroke'],'#A6CAEC',audit)

    def test_contrasting_outline_is_left_to_semantic_review_when_measurement_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for outline in (False,True):
                source=root/f'{outline}.png'
                extra=' stroke="black" stroke-width="1.2"' if outline else ''
                raster('<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><rect width="32" height="32" fill="#DAE3F3"/><circle cx="16.5" cy="16.5" r="4" fill="#4472C4"'+extra+'/></svg>').save(source)
                fitter=NativeFitter(source,{'k':ellipse()},None,'test',root/str(outline))
                result,m=fitter.fit('k',{'fill':'#4472C4','stroke':'#000000' if outline else '#548235','width_px':6,'dash':'solid'})
                self.assertIsNotNone(result['stroke'],m)
                self.assertNotEqual(m.get('role'),'filled_compact')

    def test_pale_requested_stroke_does_not_collapse_true_border(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.png'
            # An opaque core identifies source colour independently of width.
            raster('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="80"><rect width="300" height="80" fill="white"/><path d="M10 20H290V70H10Z" fill="#F4B183" stroke="#7E4824" stroke-width="2"/></svg>').save(source)
            part={'id':'a','object_id':'A','kind':'path','commands':[['M',25,19.1],['L',275,19.1]]}
            out,m=NativeFitter(source,{'k':part},None,'test',root).fit('k',{'fill':None,'stroke':'#B4784F','width_px':2,'dash':'solid'})
            self.assertAlmostEqual(out['width_px'],2.,delta=.3,msg=m)
            expected=np.array([126,72,36])
            self.assertLess(np.linalg.norm(np.asarray([int(out['stroke'][i:i+2],16) for i in (1,3,5)])-expected),
                            np.linalg.norm(np.array([180,120,79])-expected))

    def test_weak_antialias_fragments_do_not_define_width_mode(self):
        offsets=np.arange(-8.,8.001,1/10.5)
        source=raster('<svg xmlns="http://www.w3.org/2000/svg" width="120" height="40"><rect width="120" height="40" fill="white"/><path d="M10 20H110" stroke="black" stroke-width="2"/></svg>')
        rgb=sample(np.asarray(source,np.float32),np.stack(np.broadcast_arrays(np.linspace(25,95,40)[:,None],19.5+offsets[None]),-1))
        rgb[:16]=255-(255-rgb[:16])*.3
        m=fit_profiles(rgb,offsets,np.zeros(rgb.shape[:2]),np.zeros(40,bool),[0,0,0])
        self.assertAlmostEqual(m['width_px'],2.,delta=.15,msg=m)
        self.assertGreater(m['weak_antialias_profiles'],0)

    def test_subpixel_border_mixes_fill_and_outer_background(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.png'
            raster('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="80"><rect width="300" height="80" fill="white"/><path d="M10 20H290V70H10Z" fill="#F4B183" stroke="#7E4824" stroke-width="1.5"/></svg>').save(source)
            part={'id':'a','object_id':'A','kind':'path','commands':[['M',25,19.1],['L',275,19.1]]}
            fitter=NativeFitter(source,{'k':part},None,'test',root)
            out,m=fitter.fit('k',{'fill':None,'stroke':'#7E4824','width_px':2,'dash':'solid'})
            self.assertAlmostEqual(out['width_px'],1.5,delta=.3,msg=m)

    def test_filled_background_cannot_cover_its_visible_points(self):
        dot=ellipse();background={'id':'bg','object_id':'A','kind':'rectangle',
            'commands':[['M',2,2],['L',30,2],['L',30,30],['L',2,30],['Z']]}
        table={'dot':dot,'bg':background};scene={'objects':{'A':{'part_keys':['dot','bg']}}}
        relations={'object_order':['A'],'part_orders':{'A':['dot','bg']}}
        geoms,_=occlusion.discover(table,options({'styling':{}}))
        styles={'dot':{'fill':'#4472C4','stroke':None},'bg':{'fill':'#DAE3F3','stroke':None}}
        ordered=occlusion.order_internal_by_paint(scene,table,relations,styles,geoms)
        self.assertEqual(ordered['part_orders']['A'],['bg','dot'])
        self.assertEqual(scene['objects']['A']['part_keys'],['dot','bg'])

    def test_coloured_stroke_does_not_absorb_gray_shadow(self):
        for width in (1.2, 2., 3.):
            source = raster(f'<svg xmlns="http://www.w3.org/2000/svg" width="160" height="64"><rect width="160" height="64" fill="white"/><rect width="160" height="32" fill="#DDEAF7"/><path d="M10 34H150" stroke="#999999" stroke-width="4"/><path d="M10 32H150" stroke="#A6CAEC" stroke-width="{width}"/></svg>')
            native = np.asarray(source.resize((1680, 672), Image.Resampling.LANCZOS), np.float32)
            offsets = np.arange(-8, 8.001, 1 / 10.5)
            x = np.linspace(25, 135, 36)
            xy = np.stack(np.broadcast_arrays(x[:, None], 31.5 + offsets[None]), -1)
            rgb = sample(native, (xy + .5) * 10.5 - .5)
            mask = np.broadcast_to(abs(offsets) < width / 2, rgb.shape[:2]).copy()
            m = fit_profiles(rgb, offsets, mask, np.ones(len(x), bool), [166, 202, 236])
            self.assertTrue(m['measured'], m)
            self.assertAlmostEqual(m['width_px'], width, delta=.3,msg=m)
            self.assertGreater(m['mask_width_samples'], 0)

    def test_filled_dot_has_no_added_same_colour_outline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source.png'
            raster('<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><rect width="32" height="32" fill="#DAE3F3"/><circle cx="16.5" cy="16.5" r="4" fill="#4472C4"/></svg>').save(source)
            fitter = NativeFitter(source, {'k': ellipse()}, None, 'synthetic', root)
            out, m = fitter.fit('k', {'fill': '#4472C4', 'stroke': '#4472C4', 'width_px': 7, 'dash': 'solid'})
            self.assertEqual((out['stroke'], out['width_px'], m['role']), (None, 0., 'filled_compact'))

    def test_hollow_loop_stays_hollow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source.png'
            raster('<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><rect width="32" height="32" fill="white"/><circle cx="16.5" cy="16.5" r="4" fill="none" stroke="black" stroke-width="1.2"/></svg>').save(source)
            fitter = NativeFitter(source, {'k': ellipse()}, None, 'synthetic', root)
            out, m = fitter.fit('k', {'fill': None, 'stroke': '#000000', 'width_px': 8, 'dash': 'solid'})
            self.assertIsNone(out['fill']); self.assertEqual(out['stroke'], '#000000')
            self.assertLess(out['width_px'], 2.)

    def test_numerical_feedback_can_reenter_fitter(self):
        offsets = np.arange(-8., 8.001, 1 / 10.5)
        # A known 2 px line from a renderer; SAM is slightly broader.
        source = raster('<svg xmlns="http://www.w3.org/2000/svg" width="120" height="40"><rect width="120" height="40" fill="white"/><path d="M10 20H110" stroke="black" stroke-width="2"/></svg>')
        rgb = sample(np.asarray(source, np.float32), np.stack(np.broadcast_arrays(np.linspace(25, 95, 30)[:, None], 19.5 + offsets[None]), -1))
        masks = np.broadcast_to(abs(offsets) < 1.3, rgb.shape[:2])
        first = fit_profiles(rgb, offsets, masks, np.ones(30, bool), [0, 0, 0])
        reviewed = fit_profiles(rgb, offsets, masks, np.ones(30, bool), [0, 0, 0], review=True)
        self.assertTrue(reviewed['critic_refit']); self.assertGreater(reviewed['mask_loss_weight'], 0.)
        self.assertAlmostEqual(reviewed['width_px'], 2., delta=.7)
        self.assertLessEqual(reviewed['rgb_loss'], first['rgb_loss'] + .00001)

    def test_neighbouring_mask_fragment_does_not_widen_target(self):
        offsets=np.arange(-8.,8.001,1/10.5)
        source=raster('<svg xmlns="http://www.w3.org/2000/svg" width="120" height="40"><rect width="120" height="40" fill="white"/><path d="M10 20H110" stroke="black" stroke-width="2"/></svg>')
        rgb=sample(np.asarray(source,np.float32),np.stack(np.broadcast_arrays(np.linspace(25,95,30)[:,None],19.5+offsets[None]),-1))
        target=(abs(offsets)<1.) | (abs(offsets-3.)<.7)
        m=fit_profiles(rgb,offsets,np.broadcast_to(target,rgb.shape[:2]),np.ones(30,bool),[0,0,0])
        self.assertAlmostEqual(m['width_px'],2.,delta=.15)

    def test_crossings_are_not_regular_dash_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.png'
            crosses=''.join(f'<path d="M{x} 10V70" stroke="black" stroke-width="2"/>' for x in range(30,280,30))
            raster('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="80"><rect width="300" height="80" fill="white"/><path d="M10 40H290" stroke="#888888" stroke-width="2"/>'+crosses+'</svg>').save(source)
            part={'id':'a','object_id':'A','kind':'path','commands':[['M',10,39.5],['L',290,39.5]]}
            fitter=NativeFitter(source,{'k':part},None,'test',root)
            out,m=fitter.fit('k',{'fill':None,'stroke':'#888888','width_px':2,'dash':'solid'})
            self.assertEqual(out['dash'],'solid',m)

    def test_true_dash_lengths_come_from_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.png'
            raster('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="80"><rect width="300" height="80" fill="white"/><path d="M10 40H290" stroke="#444444" stroke-width="2" stroke-dasharray="8,4"/></svg>').save(source)
            part={'id':'a','object_id':'A','kind':'path','commands':[['M',10,39.5],['L',290,39.5]]}
            fitter=NativeFitter(source,{'k':part},None,'test',root)
            out,m=fitter.fit('k',{'fill':None,'stroke':'#444444','width_px':2,'dash':'solid'})
            self.assertEqual(out['dash'],'dash',m)
            np.testing.assert_allclose(out['dash_pattern_px'],[8,4],atol=.7)
            # An unsupported critic category cannot erase a measured source pattern.
            out,m=fitter.fit('k',{'fill':None,'stroke':'#444444','width_px':2,'dash':'solid'},
                feedback=[{'verdict':'revise','affected_fields':['line_pattern'],'expected_line_pattern':'solid'}])
            self.assertEqual(out['dash'],'dash',m)

    def test_semantic_contract_has_no_competing_width_estimator(self):
        p = ellipse(); p['_fixed'] = ['width_px', 'dash', 'dash_pattern_px']
        value = contracts.style({'fill': '#4472C4', 'stroke': None}, p)
        self.assertEqual(value['width_px'], 0.)
        for role in ('select', 'refine'):
            prompt = model.prompt(role, [{'id': 'P0'}], {})
            self.assertNotIn('width_px', prompt['output']['styles'][0])
        prompt = model.prompt('critic', [{'id': 'P0'}], {})
        self.assertIn('affected_fields', prompt['output']['checks'][0])


class NativePPT(unittest.TestCase):
    def test_native_fill_alpha_composites_without_fading_border(self):
        from pptx import Presentation
        from pptx.enum.shapes import MSO_SHAPE
        from pptx.util import Inches
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);ppt=Presentation();ppt.slide_width=Inches(2);ppt.slide_height=Inches(1)
            slide=ppt.slides.add_slide(ppt.slide_layouts[6])
            blue=slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,0,0,Inches(2),Inches(1))
            apply_style(blue,{},9144,{'fill':'#0000FF','stroke':None,'width_px':0,'dash':'solid'})
            red=slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(.5),Inches(.2),Inches(1),Inches(.6))
            apply_style(red,{},9144,{'fill':'#FF0000','fill_opacity':.5,'stroke':'#000000','width_px':2,'dash':'solid'})
            # Match the geometry writer: no theme-supplied cast shadows.
            for shape in (blue,red):
                shape._element.spPr.append(OxmlElement('a:effectLst'))
                inherited=shape._element.find('p:style',NS)
                if inherited is not None:shape._element.remove(inherited)
            ppt.save(folder/'alpha.pptx');render_native(folder/'alpha.pptx',folder/'render',[200,100])
            with Image.open(folder/'native.png') as im:
                np.testing.assert_allclose(im.convert('RGB').getpixel((100,50)),[128,0,128],atol=3)
                self.assertEqual(im.convert('RGB').getpixel((20,50)),(0,0,255))
                self.assertLess(max(im.convert('RGB').getpixel((50,50))),20)

    def test_probe_native_cap_semantics(self):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup);temporary=directory.name
        target=Path(os.environ['STYLE_SERVICE_DIR']).parent/'worker/dot_cap_probe' if os.environ.get('STYLE_SERVICE_DIR') else Path(temporary)
        target.mkdir(parents=True,exist_ok=True)
        modes=[('zero_round',0,'rnd',None),('tiny_round',1,'rnd',None),('half_round',50000,'rnd',None),
               ('one_round',100000,'rnd',None),('zero_flat',0,'flat',None),('one_flat',100000,'flat',None),
               ('sysDot_round',None,'rnd','sysDot'),('sysDot_flat',None,'flat','sysDot'),
               ('dot_round',None,'rnd','dot'),('dot_flat',None,'flat','dot')]
        parts={};objects={}
        for i,(name,*_) in enumerate(modes):
            key=json.dumps(['probe',name,name]);parts[key]={'id':name,'object_id':name,'kind':'path',
                'commands':[['M',20,20+i*22],['L',180,20+i*22]],'key':['probe',name,name],
                'status':'ok','native_parameters_original':None}
            objects[name]={'id':name,'state':'accepted_by_model','part_keys':[key]}
        scene={'source_size':[200,250],'parts':parts,'objects':objects}
        def writer(shape,part,scale):
            mode=next(v for v in modes if v[0]==part['id']);_,length,cap,preset=mode
            apply_style(shape,part,scale,{'fill':None,'stroke':'#000000','width_px':3.,'dash':'solid'})
            line=shape._element.spPr.find('a:ln',NS)
            for node in list(line):
                if node.tag.endswith(('prstDash','custDash')):line.remove(node)
            line.set('cap',cap)
            if preset:
                node=OxmlElement('a:prstDash');node.set('val',preset)
            else:
                node=OxmlElement('a:custDash');seg=OxmlElement('a:ds');seg.set('d',str(length));seg.set('sp','333333');node.append(seg)
            line.append(node)
        write_pptx(scene,target/'probe.pptx',style_writer=writer)
        render_native(target/'probe.pptx',target/'native_render',[200,250])
        image=np.asarray(Image.open(target/'native.png').convert('RGB'));result={}
        for i,(name,*_) in enumerate(modes):
            local=image[10+i*22:31+i*22]
            lab,_=label(local.mean(-1)<128);bs=[b for b in find_objects(lab) if b and b[1].start>25 and b[1].stop<175]
            result[name]={'width':float(np.median([b[1].stop-b[1].start for b in bs])) if bs else None,
                          'height':float(np.median([b[0].stop-b[0].start for b in bs])) if bs else None,
                          'period':float(np.median(np.diff(sorted((b[1].start+b[1].stop)/2 for b in bs)))) if len(bs)>1 else None}
        (target/'measurements.json').write_text(json.dumps(result,indent=2)+'\n');print('CAP_PROBE',result,flush=True)

    def test_round_dots_match_visible_diameter_and_period(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.png';Image.new('RGB',(200,100),'white').save(source)
            key=json.dumps(['test','A','a'])
            part={'id':'a','object_id':'A','kind':'path','commands':[['M',20,50],['L',180,50]],
                  'key':['test','A','a'],'status':'ok','native_parameters_original':None}
            scene={'case_id':'test','source_size':[200,100],'source_sha256':sha(source),
                   'objects':{'A':{'id':'A','state':'accepted_by_model','part_keys':[key]}},'parts':{key:part}}
            style={key:{'fill':None,'stroke':'#000000','width_px':3.,'dash':'dot','dash_pattern_px':[3.,7.]}}
            relations={'flat_order':[key],'object_order':['A'],'part_orders':{'A':[key]}}
            result=export(scene,style,relations,root/'out',source,options({'styling':{}}),threading.BoundedSemaphore(1))
            self.assertTrue(result['geometry_unchanged'])
            image=np.asarray(Image.open(root/'out/native.png').convert('RGB'))
            labels,n=label(image.mean(-1)<128)
            boxes=[b for b in find_objects(labels) if b and b[1].start>25 and b[1].stop<175]
            self.assertGreater(len(boxes),8)
            self.assertLessEqual(np.median([b[1].stop-b[1].start for b in boxes]),4.)
            centres=sorted([(b[1].start+b[1].stop)/2 for b in boxes])
            self.assertAlmostEqual(float(np.median(np.diff(centres))),10.,delta=.6)


if __name__=='__main__':unittest.main()
