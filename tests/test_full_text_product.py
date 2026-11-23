import copy
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree
from pptx import Presentation
from full_recovery.text_product import select_native_text, NS, P, A


class NativeTextProductTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name); self.source = self.root / 'source.pptx'
        self.output = self.root / 'selected.pptx'
        deck = Presentation(); slide = deck.slides.add_slide(deck.slide_layouts[6])
        for ident, text in [('vertical', 'A\vB\vC'), ('remove', 'duplicate')]:
            shape = slide.shapes.add_textbox(100, 200, 300, 400)
            shape.name = 'TXT ' + ident; shape.text = text
        deck.save(self.source)

    def slide(self, path):
        with ZipFile(path) as z:
            return etree.fromstring(z.read('ppt/slides/slide1.xml'))

    def test_stacked_glyphs_and_properties_survive_selection_and_fill(self):
        original = self.source.read_bytes()
        select_native_text(self.source, self.output, [{'id': 'vertical', 'background_fill': '#123456'}])
        before = self.slide(self.source).find('.//p:txBody', NS)
        after = self.slide(self.output).find('.//p:txBody', NS)
        self.assertEqual(etree.tostring(before), etree.tostring(after))
        self.assertEqual(len(after.findall('.//a:br', NS)), 2)
        self.assertEqual(len(self.slide(self.output).findall('.//p:sp', NS)), 1)
        self.assertEqual(self.slide(self.output).find('.//p:spPr/a:solidFill/a:srgbClr', NS).get('val'), '123456')
        self.assertEqual(self.source.read_bytes(), original)

    def test_alternate_formula_and_fallback_are_retained(self):
        with ZipFile(self.source) as z: package = {n: z.read(n) for n in z.namelist()}
        slide = etree.fromstring(package['ppt/slides/slide1.xml'])
        tree = slide.find('p:cSld/p:spTree', NS); shape = tree[2]; tree.remove(shape)
        mc = 'http://schemas.openxmlformats.org/markup-compatibility/2006'
        ac = etree.SubElement(tree, f'{{{mc}}}AlternateContent')
        for kind in ('Choice', 'Fallback'):
            branch = etree.SubElement(ac, f'{{{mc}}}{kind}')
            native = copy.deepcopy(shape)
            native.find('p:nvSpPr/p:cNvPr', NS).set('name', 'EQ equation' if kind == 'Choice' else 'MATH_PREVIEW equation')
            branch.append(native)
        package['ppt/slides/slide1.xml'] = etree.tostring(slide)
        with ZipFile(self.source, 'w', ZIP_DEFLATED) as z:
            for name, data in package.items(): z.writestr(name, data)
        select_native_text(self.source, self.output, [{'id': 'equation', 'background_fill': '#FFFFFF', 'background_box_px': [0, 0, 1, 1]}])
        after = self.slide(self.output)
        self.assertEqual([etree.tostring(n) for n in ac.findall('.//p:txBody', NS)],
                         [etree.tostring(n) for n in after.findall('.//p:txBody', NS)])
        self.assertEqual(len(after.findall('.//p:spPr/a:noFill', NS)), 2)

    def test_missing_label_fails_without_overwriting_source(self):
        original = self.source.read_bytes()
        with self.assertRaisesRegex(ValueError, 'absent'):
            select_native_text(self.source, self.output, [{'id': 'missing'}])
        with self.assertRaisesRegex(ValueError, 'read-only'):
            select_native_text(self.source, self.source, [])
        self.assertEqual(self.source.read_bytes(), original)
        self.assertFalse(self.output.exists())
