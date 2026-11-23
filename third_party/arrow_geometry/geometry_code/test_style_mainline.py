import copy,tempfile,threading,unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile
import numpy as np
from PIL import Image,ImageDraw
from lxml import etree
import pipeline,pixels,protocol,contracts,style_semantics,straight_mask


class MainlineStyles(unittest.TestCase):
    def test_complete_style_json_with_trailing_prose_preserves_model_answer(self):
        import json
        from transport.model import parse
        answer={'line_styles':{'L1':'solid','L2':'dashed'}}
        self.assertEqual(parse(json.dumps(answer)+'\n\nReasoning: visible stroke types.'),answer)
        with self.assertRaises(json.JSONDecodeError):parse(json.dumps(answer)+'\n'+json.dumps({'line_styles':{'L1':'dotted','L2':'solid'}}))

    def test_missing_line_style_is_not_silently_defaulted_by_contract(self):
        with self.assertRaises(ValueError):protocol.line_styles({'line_styles':{}},['E1'])
        self.assertEqual(protocol.line_styles({'line_styles':{'E1':'unknown'}},['E1']),{'E1':'unknown'})

    # Recovery now has one semantic-structure path. Its point/mask and missing
    # region integration checks live in test_no_skeleton; the retired E/N repair
    # interface is preserved only in frozen run code.

    def test_unknown_style_stays_pending_without_erasing_geometry(self):
        image=Image.new('RGB',(120,60),'white')
        row=dict(id='a',parent_id='a',kind='straight',start=[10,30],end=[100,30],via=[],heads=['none','none'],line_style='unknown',width_estimate=2)
        pipeline.style_rows(image,[row],np.zeros((60,120),bool))
        self.assertTrue(pipeline.style.pending(row));self.assertIsNone(row['style']['dash']['preset'])
        self.assertIn('id="a"',pipeline.viz.svg(image.size,[row],False))
        self.assertIn('#808080',pipeline.viz.svg(image.size,[row],False))
        self.assertIn('id="a"',pipeline.viz.svg(image.size,[row],True))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x.pptx';pipeline.native.write(image.size,[row],[row['style']],p)
            with ZipFile(p) as z:xml=etree.fromstring(z.read('ppt/slides/slide1.xml'))
            self.assertEqual(len(xml.findall('.//{http://schemas.openxmlformats.org/presentationml/2006/main}cxnSp')),1)
            import json
            self.assertEqual(json.loads(p.with_suffix('.export.json').read_text())['style_pending'],['a'])
            self.assertTrue(pipeline.style.pending(row))

    def test_numeric_refit_preserves_model_style_boundaries(self):
        m=np.zeros((60,180),bool);m[28:33,10:170]=True
        base=dict(kind='straight',via=[],identity_index=1,coordinate_source='local_sam_mask',heads=['none','none'])
        rows=[dict(base,id='E1',start=[10,30],end=[90,30],line_style='solid'),
              dict(base,id='E2',start=[90,30],end=[169,30],line_style='dashed')]
        self.assertEqual(straight_mask.refit_confirmed(rows,[dict(id='M1',mask=m)]),rows)


if __name__=='__main__':unittest.main()
