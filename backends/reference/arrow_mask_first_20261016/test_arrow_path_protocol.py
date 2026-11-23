"""Protocol invariants that protect visual geometry during branch/patch handling."""
import copy,sys,unittest,zipfile,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'lib'))
import arrow_path_protocol as p

STYLE=dict(color='#808080',width_px=2,line_style='solid',heads=['none','none'])

def sample():
    return dict(version=1,frame='CLEAN',size=[200,100],targets=[dict(id='T1',paths=[
        dict(id='P1',kind='stroke',parts=[dict(kind='line',points=[[10,50],[190,50]])],style=copy.deepcopy(STYLE)),
        dict(id='P2',kind='stroke',parts=[dict(kind='line',points=[[80,10],[80,50]])],style=copy.deepcopy(STYLE))])])

class ProtocolTests(unittest.TestCase):
    def test_one_identity_branches_render_as_two_paths_and_two_native_shapes(self):
        from lib import native
        from lxml import etree
        d=sample();rows=p.rows(d);self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['end'],[190,50]);self.assertEqual(rows[1]['start'],[80,10])
        self.assertEqual({r['parent_id'] for r in rows},{'T1'})
        with tempfile.TemporaryDirectory() as tmp:
            fn=Path(tmp)/'x.pptx';native.write(d['size'],rows,[r['style'] for r in rows],fn)
            with zipfile.ZipFile(fn) as z:root=etree.fromstring(z.read('ppt/slides/slide1.xml'))
            ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
            self.assertEqual(len(root.findall('.//p:cxnSp',ns)),2)
            self.assertTrue(all(e.get('type')=='none' for e in root.findall('.//a:headEnd',ns)+root.findall('.//a:tailEnd',ns)))

    def test_style_update_preserves_points_and_untouched_paths_exactly(self):
        d=sample();old=copy.deepcopy(d)
        a=dict(status='patch',context_bbox=None,edits=[dict(op='update_style',target='T1',path='P2',style={'width_px':4})])
        new=p.apply_patch(d,a,p.revision(d))
        self.assertEqual(d,old);self.assertEqual(new['targets'][0]['paths'][0],old['targets'][0]['paths'][0])
        self.assertEqual(new['targets'][0]['paths'][1]['parts'],old['targets'][0]['paths'][1]['parts'])

    def test_invalid_patch_is_atomic(self):
        d=sample();old=copy.deepcopy(d)
        a=dict(status='patch',edits=[dict(op='delete',target='T1',path='P1'),dict(op='update_style',target='T1',path='P2',style={'width_px':float('nan')})])
        with self.assertRaises(ValueError):p.apply_patch(d,a,p.revision(d))
        self.assertEqual(d,old)

    def test_stale_revision_and_readonly_scope_rejected(self):
        d=sample();a=dict(status='patch',edits=[dict(op='delete',target='T1',path='P1')])
        with self.assertRaises(ValueError):p.apply_patch(d,a,'stale')
        with self.assertRaises(ValueError):p.apply_patch(d,a,p.revision(d),writable_targets=[])

    def test_unresolved_and_context_requests_retain_geometry(self):
        d=sample()
        for a in [dict(status='unresolved',edits=[],context_bbox=None),dict(status='need_context',edits=[],context_bbox=[0,0,20,20])]:
            self.assertEqual(p.apply_patch(d,a,p.revision(d),context_size=[40,40]),d)

    def test_legacy_reference_break_does_not_insert_a_bridge(self):
        a=dict(nodes=[['a',10,50],['b',100,50],['c',80,10],['d',80,50]],routes=[dict(id='T1',segments=[dict(kind='line',nodes=['a','b']),dict(kind='line',nodes=['c','d'])])])
        d,audit=p.from_legacy(a,[200,100],{'T1':STYLE})
        self.assertEqual(len(d['targets'][0]['paths']),2)
        self.assertEqual([x['parts'][0]['points'] for x in d['targets'][0]['paths']],[[[10,50],[100,50]],[[80,10],[80,50]]])
        self.assertFalse(audit[0]['coordinates_changed'])
        with self.assertRaises(ValueError):p.from_legacy(a,[200,100],{'T1':{**STYLE,'heads':['none','triangle']}})

    def test_contour_is_filled_not_a_shaft_and_head(self):
        d=sample();d['targets'][0]['paths']=[dict(id='B',kind='block_arrow',contour=[[20,20],[90,50],[20,80]],style=copy.deepcopy(STYLE))]
        r=p.rows(d)[0];self.assertEqual(r['kind'],'block_arrow');self.assertEqual(r['heads'],['none','none'])

    def test_disconnected_parts_require_separate_paths_and_no_coordinate_snapping(self):
        d=sample();d['targets'][0]['paths'][0]['parts'].append(dict(kind='line',points=[[189.9,50],[189.9,80]]))
        with self.assertRaises(ValueError):p.validate(d)

    def test_tall_local_crop_native_export_preserves_aspect(self):
        from lib import native
        from pptx import Presentation
        d=sample();d['size']=[318,1536];rows=p.rows(d)
        with tempfile.TemporaryDirectory() as tmp:
            fn=Path(tmp)/'tall.pptx';native.write(d['size'],rows,[r['style'] for r in rows],fn)
            prs=Presentation(fn)
            self.assertAlmostEqual(prs.slide_height/prs.slide_width,1536/318,places=6)

if __name__=='__main__':unittest.main()
