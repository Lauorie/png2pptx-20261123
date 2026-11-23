"""Composition fill must be source-approved and leave other branches intact."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from full_recovery.text_occlusion import select,safe_fill,frame_candidates


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))


class FakeRuntime:
    def __init__(self,decision='fill'):self.decision=decision;self.calls=0
    def call(self,root,tag,system,prompt,images,validator,budget):
        self.calls+=1
        return validator({'decision':self.decision,'fill':'#FFFFFF' if self.decision=='fill' else None,
            'occluded_parts':['F1'] if self.decision=='fill' else [],
            'evidence':'CLEAN shows a uniform label backing interrupting the lower frame edge.'})


class TextOcclusionTests(unittest.TestCase):
    def test_approved_shaft_backing_preserves_adjacent_unapproved_border(self):
        from full_recovery.text_occlusion import stroke_backing
        row={**self.row,'bbox_2d':[145,40,175,60],'background_box_px':[144,39,176,61]}
        arrow={'id':'shaft','start':[160,0],'end':[160,100],'heads':['none','none'],
               'style':{'color':'#CC0000','width_px':2}}
        styles={k:v['style'] for k,v in self.style['parts'].items()}
        self.assertTrue(safe_fill(row,[],'#FFFFFF',self.scene,styles,[arrow],[row],
                                 approved_arrows=['shaft']).startswith('protected_shape_stroke'))
        narrowed=stroke_backing(row,[],['shaft'],self.scene,styles,[arrow])
        self.assertIsNone(safe_fill(narrowed,[],'#FFFFFF',self.scene,styles,[arrow],[row],approved_arrows=['shaft']))
        self.assertEqual(narrowed['background_box_px'],[158.,39.,162.,61.])

    def test_typography_padding_does_not_nominate_source_visible_border(self):
        row={**self.row,'bbox_2d':[55,50,105,58]}
        runtime=FakeRuntime();rows,report=self.select([row],runtime=runtime)
        self.assertEqual(runtime.calls,0)
        self.assertIsNone(rows[0]['background_fill'])

    def test_label_backings_are_below_all_text_without_moving_frames(self):
        from pptx import Presentation
        from full_recovery.text_occlusion import write_backings
        from full_recovery.compose import read_package,SLIDE,P
        from lxml import etree
        deck=Presentation();slide=deck.slides.add_slide(deck.slide_layouts[6])
        for label in ('Output','State'):slide.shapes.add_textbox(100,200,300,400).text=label
        path=self.root/'text.pptx';deck.save(path)
        before=etree.fromstring(read_package(path)[SLIDE])
        glyphs=[etree.tostring(x) for x in before.findall('.//{'+P+'}sp')]
        write_backings(path,[{'background_fill':'#FFFFFF','background_box_px':[20,20,60,40]},
                             {'background_fill':'#FFFFFF','background_box_px':[20,35,60,55]}],(160,100))
        after=etree.fromstring(read_package(path)[SLIDE]);shapes=after.findall('.//{'+P+'}sp')
        self.assertEqual(len(shapes),4)
        self.assertEqual([etree.tostring(x) for x in shapes[2:]],glyphs)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.source=self.root/'source.png';Image.new('RGB',(160,100),'white').save(self.source)
        self.scene={'parts':{'frame':{'id':'frame','kind':'rectangle','object_id':'panel','status':'ok',
            'commands':[['M',10,10],['L',150,10],['L',150,70],['L',10,70],['Z']]}}}
        self.style={'parts':{'frame':{'style':{'fill':None,'stroke':'#000000','width_px':1,'dash':'dash'}}}}
        self.row={'id':'T1','text':'caption','frame_px':[50,60,60,20],'rotation_deg':0,
                  'background_fill':'#123456','font_size_px':12,'runs':[{'text':'caption'}]}

    def select(self,rows=None,arrows=None,runtime=None):
        return select(rows or [self.row],self.scene,self.style,{'arrows':arrows or []},{},self.source,
                      self.root/'paint',runtime or FakeRuntime(),save)

    def test_source_approved_fill_changes_only_background_metadata(self):
        original=copy.deepcopy((self.scene,self.style,self.row))
        rows,report=self.select();self.assertEqual(rows[0]['background_fill'],'#FFFFFF')
        self.assertEqual(report['filled'],1)
        for key in self.row:
            if key not in ('background_fill','background_evidence'):self.assertEqual(rows[0][key],self.row[key])
        self.assertEqual((self.scene,self.style,self.row),original)

    def test_no_frame_crossing_is_transparent_without_model_call(self):
        row={**self.row,'frame_px':[50,30,60,20]};runtime=FakeRuntime()
        rows,report=self.select([row],runtime=runtime)
        self.assertIsNone(rows[0]['background_fill']);self.assertEqual(runtime.calls,0)

    def test_model_uncertainty_remains_transparent(self):
        rows,report=self.select(runtime=FakeRuntime('unresolved'))
        self.assertIsNone(rows[0]['background_fill']);self.assertEqual(report['filled'],0)

    def test_correct_arrow_shaft_or_head_is_never_masked(self):
        arrow={'id':'A1','start':[30,65],'end':[130,65],'heads':['none','triangle'],
               'style':{'width_px':2,'end':{'type':'triangle','length':'lg'}}}
        rows,report=self.select(arrows=[arrow]);self.assertIsNone(rows[0]['background_fill'])
        self.assertIn('protected_arrow',report['reviews'][0]['unresolved'])
        # A head can overlap while the centerline itself stays outside.
        arrow.update(start=[120,40],end=[120,55])
        self.assertTrue(safe_fill(self.row,['frame'],'#FFFFFF',self.scene,
            {'frame':self.style['parts']['frame']['style']},[arrow],[self.row]).startswith('protected_arrow'))

    def test_other_shape_stroke_colored_fill_picture_and_other_text_are_protected(self):
        styles={'frame':self.style['parts']['frame']['style']}
        scene=copy.deepcopy(self.scene);scene['parts']['wire']={'status':'ok','commands':[['M',45,68],['L',115,68]]}
        self.assertIn('protected_shape_stroke',safe_fill(self.row,['frame'],'#FFFFFF',scene,
            {**styles,'wire':{'stroke':'#000000','width_px':1}},[],[self.row]))
        self.assertIn('protected_shape_fill',safe_fill(self.row,['frame'],'#FFFFFF',self.scene,
            {'frame':{**styles['frame'],'fill':'#DDDDDD'}},[],[self.row]))
        self.assertIn('protected_image',safe_fill(self.row,['frame'],'#FFFFFF',self.scene,styles,[],[self.row],
            [{'id':'I1','bbox_2d':[80,65,100,75]}]))
        self.assertIn('protected_other_text',safe_fill(self.row,['frame'],'#FFFFFF',self.scene,styles,[],
            [self.row,{**self.row,'id':'T2'}]))

    def test_arrow_only_label_can_mask_explicitly_approved_hidden_shaft(self):
        self.scene={'parts':{}};self.style={'parts':{}}
        self.row.update(frame_px=[50,40,60,20])
        arrow={'id':'route','start':[10,50],'end':[150,50],'heads':['none','none'],'style':{'width_px':2}}
        with Image.open(self.source) as opened:image=opened.convert('RGB')
        draw=ImageDraw.Draw(image);draw.line([(10,50),(49,50)],fill='black',width=2)
        draw.line([(111,50),(150,50)],fill='black',width=2);image.save(self.source)
        class Review(FakeRuntime):
            def call(owner,root,tag,system,prompt,images,validator,budget):
                owner.calls+=1
                assert prompt['arrow_shafts'][0]['id']=='A1'
                return validator({'decision':'fill','fill':'#FFFFFF','occluded_parts':['A1'],
                    'evidence':'Source has white label backing interrupting only the shaft, with visible continuation on both sides.'})
        runtime=Review();rows,report=self.select(arrows=[arrow],runtime=runtime)
        self.assertEqual(runtime.calls,1);self.assertEqual(rows[0]['background_fill'],'#FFFFFF')
        self.assertEqual(report['reviews'][0]['occluded_arrows'],['route'])

    def test_approval_to_mask_shaft_never_authorizes_masking_endpoint_dot(self):
        arrow={'id':'route','start':[10,65],'end':[100,65],'heads':['none','oval'],'style':{'width_px':2}}
        reason=safe_fill(self.row,['frame'],'#FFFFFF',self.scene,
            {'frame':self.style['parts']['frame']['style']},[arrow],[self.row],approved_arrows=['route'])
        self.assertEqual(reason,'protected_arrow_head:route')

    def test_visible_source_continuation_in_text_padding_remains_protected(self):
        self.row.update(frame_px=[50,40,60,30],bbox_2d=[50,50,110,65])
        arrow={'id':'route','start':[10,43],'end':[150,43],'heads':['none','none'],
               'style':{'width_px':1,'color':'#000000'}}
        image=Image.new('RGB',(160,100),'white')
        ImageDraw.Draw(image).line([(10,43),(150,43)],fill='black')
        reason=safe_fill(self.row,[],'#FFFFFF',{'parts':{}},{},[arrow],[self.row],
                         approved_arrows=['route'],source_pixels=np.asarray(image))
        self.assertEqual(reason,'source_visible_arrow_in_text_padding:route')

    def test_text_selection_uses_surviving_segment_ownership(self):
        # The lower frame edge was handed to an arrow before text review.
        # It must not be nominated or veto the later approved arrow backing.
        arrow={'id':'route','start':[10,70],'end':[150,70],'heads':['none','none'],
               'style':{'width_px':1,'color':'#000000'}}
        class Review(FakeRuntime):
            def call(owner,root,tag,system,prompt,images,validator,budget):
                owner.calls+=1
                assert prompt['frame_parts']==[]
                return validator({'decision':'fill','fill':'#FFFFFF','occluded_parts':['A1'],
                    'evidence':'The label interrupts the retained shaft in the original.'})
        rows,report=select([self.row],self.scene,self.style,{'arrows':[arrow]},
            {'shape_segment_suppression':{'panel/frame':[2]}},self.source,self.root/'owned',Review(),save)
        self.assertEqual(rows[0]['background_fill'],'#FFFFFF')
        self.assertFalse(report['reviews'][0]['candidate_frame_parts'])

    def test_text_review_does_not_protect_an_already_removed_disconnected_line(self):
        self.scene={'parts':{'fan':{'id':'fan','object_id':'network','kind':'path','status':'ok',
            'commands':[['M',10,70],['L',150,70],['M',10,20],['L',150,20]]}}}
        self.style={'parts':{'fan':{'style':{'fill':None,'stroke':'#000000','width_px':1}}}}
        arrow={'id':'route','start':[10,70],'end':[150,70],'heads':['none','none'],
               'style':{'width_px':1,'color':'#000000'}}
        class Review(FakeRuntime):
            def call(owner,root,tag,system,prompt,images,validator,budget):
                assert not prompt['frame_parts']
                return validator({'decision':'fill','fill':'#FFFFFF','occluded_parts':['A1'],
                                  'evidence':'The original label interrupts this retained shaft.'})
        rows,report=select([self.row],self.scene,self.style,{'arrows':[arrow]},
            {'shape_subpath_suppression':{'network/fan':[0]}},self.source,self.root/'subpaths',Review(),save)
        self.assertEqual(rows[0]['background_fill'],'#FFFFFF')
        self.assertEqual(report['reviews'][0]['candidate_shape_lines'],[])


if __name__=='__main__':unittest.main()
