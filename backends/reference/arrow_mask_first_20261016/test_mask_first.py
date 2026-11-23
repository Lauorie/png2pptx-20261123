import tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import pixels,protocol


def candidate(mask):return dict(id='M1',mask=mask,category='line')


class MaskFirst(unittest.TestCase):
    def test_shared_junction_keeps_one_coordinate(self):
        im=Image.new('L',(260,200));d=ImageDraw.Draw(im);d.line([(25,40),(230,40)],fill=255,width=5)
        d.line([(115,40),(115,155),(210,155)],fill=255,width=5);m=np.asarray(im)>0
        with tempfile.TemporaryDirectory() as tmp:
            draft,mask=pixels.draft([candidate(m)],im.size,Path(tmp))
            junctions=[n for n,v in draft['nodes'].items() if v['degree']==3];self.assertEqual(len(junctions),1)
            answer=dict(targets=[dict(sources=['G1'],paths=[r['id'] for r in draft['paths']],filled_masks=[],repair=None)],
                heads={n:'none' for n in draft['nodes']},line_styles={r['id']:'solid' for r in draft['paths']},junctions=junctions,excluded=[],uncertain=[])
            a=protocol.decision(answer,['G1'],draft,['M1']);rows=pixels.materialize(draft,a)
            points=[]
            for r in rows:
                for n,p in zip(r['nodes'],[r['start'],r['end']]):
                    if n==junctions[0]:points.append(p)
            self.assertEqual(len(points),3);self.assertTrue(all(np.array_equal(p,points[0]) for p in points))

    def test_heads_bound_to_named_nodes_and_same_edge_drawn_once(self):
        im=Image.new('L',(200,80));ImageDraw.Draw(im).line([(20,40),(175,40)],fill=255,width=5)
        with tempfile.TemporaryDirectory() as tmp:
            draft,_=pixels.draft([candidate(np.asarray(im)>0)],im.size,Path(tmp));edge=draft['paths'][0]
            a,b=edge['nodes'];heads={a:'arrow',b:'none'}
            decision=dict(targets=[dict(sources=['G1'],paths=[edge['id']],filled_masks=[],repair=None),dict(sources=['G2'],paths=[edge['id']],filled_masks=[],repair=None)],heads=heads,line_styles={edge['id']:'dashed'},junctions=[],excluded=[],uncertain=[])
            checked=protocol.decision(decision,['G1','G2'],draft,['M1']);rows=pixels.materialize(draft,checked)
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]['heads'],['arrow','none'])
            self.assertEqual(rows[0]['line_style'],'dashed')
            self.assertTrue(rows[0]['start_head']);self.assertFalse(rows[0]['end_head'])

    def test_pinholes_do_not_create_fake_network_but_large_loop_survives(self):
        im=Image.new('L',(220,180));d=ImageDraw.Draw(im)
        d.line([(25,90),(160,90)],fill=255,width=12)
        d.polygon([(185,90),(145,65),(145,115)],fill=255)
        d.ellipse((153,83,155,85),fill=0)
        raw=np.asarray(im)>0
        measured,holes=pixels.geometry_support(raw)
        self.assertTrue(holes);self.assertFalse(raw[84,154]);self.assertTrue(measured[84,154])
        ring=Image.new('L',(100,100));ImageDraw.Draw(ring).ellipse((10,10,90,90),outline=255,width=4)
        m,_=pixels.geometry_support(np.asarray(ring)>0);self.assertFalse(m[50,50])
        with tempfile.TemporaryDirectory() as tmp:
            draft,support=pixels.draft([candidate(raw)],im.size,Path(tmp))
            self.assertEqual(len(draft['paths']),1);self.assertTrue(np.array_equal(raw,support))

    def test_complete_json_with_prose_keeps_all_model_decisions(self):
        from transport.model import parse
        import json
        answer={'targets':[], 'heads':{}, 'junctions':[], 'excluded':['G1'], 'uncertain':[]}
        text='```json\n'+json.dumps(answer)+'\n```\nExplanation: frame edge.'
        self.assertEqual(parse(text),answer)
        with self.assertRaises(json.JSONDecodeError):parse(json.dumps(answer)+'\n'+json.dumps(dict(answer,excluded=['G2'])))

    def test_two_point_line_alias_preserves_geometry_and_heads(self):
        import contracts as ct
        a={'nodes':[['a',10.,20.],['b',80.,20.]],'routes':[{'id':'r','kind':'line','nodes':['a','b'],'heads':['none','arrow'],'sources':['G1']}], 'excluded':[], 'uncertain':[]}
        b=ct.graph(a,[100,100],['G1'])
        self.assertEqual(a['routes'][0]['kind'],'line')
        self.assertEqual(b['routes'][0]['kind'],'straight')
        self.assertEqual(b['nodes'],a['nodes']);self.assertEqual(b['routes'][0]['heads'],a['routes'][0]['heads'])

    def test_repair_retains_identity_without_fabricated_points(self):
        draft=dict(nodes={},paths=[])
        answer=dict(targets=[dict(sources=['G1'],paths=[],filled_masks=[],repair='missing local support')],heads={},line_styles={},junctions=[],excluded=[],uncertain=[])
        a=protocol.decision(answer,['G1'],draft,[]);self.assertEqual(pixels.materialize(draft,a),[])
        self.assertEqual(a['targets'][0]['sources'],['G1'])

    def test_curve_is_measured_from_mask_without_model_points(self):
        im=Image.new('L',(220,180));ImageDraw.Draw(im).arc((20,20,200,160),180,350,fill=255,width=4)
        with tempfile.TemporaryDirectory() as tmp:
            draft,_=pixels.draft([candidate(np.asarray(im)>0)],im.size,Path(tmp))
            self.assertTrue(draft['paths']);self.assertTrue(all(p['coordinate_source']=='local_sam_mask' for p in draft['paths']))
            for p in draft['paths']:
                q=np.asarray([p['start'],*p['via'],p['end']]);self.assertTrue(np.isfinite(q).all());self.assertTrue((q>=0).all());self.assertTrue((q[:,0]<220).all())


if __name__=='__main__':unittest.main()
