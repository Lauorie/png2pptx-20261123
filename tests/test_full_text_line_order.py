import unittest
from pptx import Presentation
from lxml import etree
from full_recovery.text_breaks import install


class NativeLineOrderTests(unittest.TestCase):
    def test_stacked_runs_keep_breaks_between_glyphs_and_end_properties_last(self):
        install();install()
        p=Presentation();s=p.slides.add_slide(p.slide_layouts[6])
        paragraph=s.shapes.add_textbox(0,0,100000,500000).text_frame.paragraphs[0]
        paragraph._p.get_or_add_endParaRPr()
        paragraph.add_run().text='A'
        paragraph.add_line_break()
        paragraph.add_run().text='B'
        paragraph.add_run().text='2'
        paragraph.add_line_break()
        paragraph.add_run().text='C'
        content=[etree.QName(n).localname for n in paragraph._p]
        self.assertEqual(content,['r','br','r','r','br','r','endParaRPr'])
        self.assertEqual(paragraph.text,'A\vB2\vC')
