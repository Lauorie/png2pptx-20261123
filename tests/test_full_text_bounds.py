import tempfile
import unittest
from pathlib import Path
from pptx import Presentation
from full_recovery.text_bounds import clamp_small_text_overflow
from full_recovery.text_bounds import normalize_native_text_frames


class TextBoundsTests(unittest.TestCase):
    def deck(self, path, x, y, w=200000, h=100000, rotation=0):
        deck=Presentation();deck.slide_width=deck.slide_height=1000000
        shape=deck.slides.add_slide(deck.slide_layouts[6]).shapes.add_textbox(x,y,w,h)
        shape.name='TXT label';shape.text='unchanged';shape.rotation=rotation
        deck.save(path)

    def test_small_overflow_translates_only_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'text.pptx'
            for xy,expected in [((50000,-3300),(50000,0)),((805000,905000),(800000,900000))]:
                with self.subTest(xy=xy):
                    self.deck(path,*xy)
                    changes=clamp_small_text_overflow(path,[100,100]);self.assertEqual(len(changes),1)
                    shape=Presentation(path).slides[0].shapes[0]
                    self.assertEqual((shape.left,shape.top),expected)
                    self.assertEqual((shape.width,shape.height,shape.text),(200000,100000,'unchanged'))
                    self.assertEqual(clamp_small_text_overflow(path,[100,100]),[])

    def test_large_rotated_and_oversized_frames_are_not_silently_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'text.pptx'
            for args in [(0,-20000,200000,100000,0),(0,-3300,200000,100000,30),
                         (0,-3300,1100000,100000,0),(0,0,200000,100000,0)]:
                with self.subTest(args=args):
                    self.deck(path,*args);before=path.read_bytes()
                    self.assertEqual(clamp_small_text_overflow(path,[100,100]),[])
                    self.assertEqual(path.read_bytes(),before)

    def test_frame_normalization_preserves_inner_text_and_paragraphs(self):
        from pptx.enum.text import MSO_AUTO_SIZE
        from zipfile import ZipFile
        from lxml import etree
        ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
        cases=[([-30,30,80,20],270,[5,5,15,80],{'dx':0,'dy':5,'width':75,'height':10}),
               ([5,-.5,20,10],0,[5,1.5,15,8.5],{'dx':0,'dy':2,'width':10,'height':7}),
               ([-.3,10,20,10],0,[0,12,10,19],{'dx':.2,'dy':2,'width':10,'height':7})]
        for frame,angle,bbox,metric in cases:
            with self.subTest(angle=angle),tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'text.pptx';self.deck(path,*[round(v*10000) for v in frame],rotation=angle)
                deck=Presentation(path);shape=deck.slides[0].shapes[0];tf=shape.text_frame
                tf.word_wrap=False;tf.auto_size=MSO_AUTO_SIZE.NONE
                tf.margin_left=tf.margin_top=tf.margin_right=tf.margin_bottom=0
                paragraphs=[p._p.xml for p in tf.paragraphs];deck.save(path)
                row={'id':'label','kind':'text','bbox_2d':bbox,'font_size_px':24,
                     'layout':{'measurement':{**metric,'method':'native_ppt_atlas','reference_size_px':24}}}
                changes=normalize_native_text_frames(path,[100,100],texts=[row])
                self.assertEqual(len(changes),1)
                final=Presentation(path).slides[0].shapes[0]
                self.assertEqual([p._p.xml for p in final.text_frame.paragraphs],paragraphs)
                self.assertGreaterEqual(min(final.left,final.top),0)
                self.assertLessEqual(final.left+final.width,1000000)
                self.assertEqual(final.rotation,0)
                if angle:self.assertEqual(final.text_frame._txBody.bodyPr.get('vert'),'vert270')
                elif frame[1]<0:
                    self.assertEqual(final.top+final.text_frame.margin_top,-5000)
                    self.assertEqual(final.height-final.text_frame.margin_top,100000)
                else:
                    self.assertEqual(final.left+final.text_frame.margin_left,-3000)
                    self.assertEqual(final.width-final.text_frame.margin_left,200000)
                self.assertEqual(normalize_native_text_frames(path,[100,100],texts=[row]),[])

    def test_frame_normalization_does_not_hide_off_page_ink(self):
        from pptx.enum.text import MSO_AUTO_SIZE
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'text.pptx';self.deck(path,10000,-50000)
            deck=Presentation(path);tf=deck.slides[0].shapes[0].text_frame
            tf.word_wrap=False;tf.auto_size=MSO_AUTO_SIZE.NONE
            tf.margin_left=tf.margin_top=tf.margin_right=tf.margin_bottom=0;deck.save(path)
            row={'id':'label','kind':'text','bbox_2d':[1,0,10,10],'font_size_px':24,
                 'layout':{'measurement':{'method':'native_ppt_atlas','reference_size_px':24,
                     'dx':0,'dy':2,'width':9,'height':8}}}
            before=path.read_bytes()
            self.assertEqual(normalize_native_text_frames(path,[100,100],texts=[row]),[])
            self.assertEqual(path.read_bytes(),before)
