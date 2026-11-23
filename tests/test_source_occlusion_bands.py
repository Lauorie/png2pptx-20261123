"""A displaced centerline and label knockout cannot hide an entire route."""
from pathlib import Path
import tempfile
import unittest

from PIL import Image,ImageDraw
from full_recovery.handoff import source_occlusion


class SourceBands(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'source.png'
        self.scene={'parts':{'frame':{'id':'body','object_id':'panel','status':'ok','kind':'rectangle',
            'commands':[['M',20,10],['L',80,10],['L',80,70],['L',20,70],['Z']]}}}
        self.styles={'frame':{'fill':'#FFFFFF','stroke':'#000000','width_px':1}}
        self.arrow={'id':'route','start':[1,30],'end':[98,30],'heads':['none','none'],
                    'style':{'color':'#000000','width_px':1}}

    def review(self,source,**kwargs):
        source.save(self.path)
        return source_occlusion(self.scene,self.styles,[self.arrow],self.path,set(),part_placements=True,**kwargs)

    def test_nearby_visible_source_stroke_is_not_mistaken_for_occlusion(self):
        source=Image.new('RGB',(100,80),'white')
        ImageDraw.Draw(source).line([(1,32),(98,32)],fill='black',width=1)
        plan,_=self.review(source)
        self.assertEqual(plan,{})

    def test_real_opaque_body_hiding_line_still_places_line_behind_it(self):
        source=Image.new('RGB',(100,80),'white');d=ImageDraw.Draw(source)
        d.line([(1,30),(98,30)],fill='black');d.rectangle((20,10,80,70),fill='white',outline='black')
        plan,audit=self.review(source)
        self.assertEqual(plan['route']['before'],['panel/body'])
        self.assertEqual(audit[0]['source_visible_samples'],0)

    def test_a_contained_displaced_route_cannot_be_hidden_by_its_panel(self):
        self.arrow.update(start=[25,30],end=[75,30])
        source=Image.new('RGB',(100,80),'white')
        ImageDraw.Draw(source).line((25,35,75,35),fill='black')
        plan,audit=self.review(source)
        self.assertEqual(plan,{})
        self.assertEqual(audit[0]['reason'],'contained_route_without_occlusion_boundary_evidence')

    def test_blank_container_interior_is_an_existence_question_not_occlusion_proof(self):
        self.arrow.update(start=[25,30],end=[75,30])
        self.assertEqual(self.review(Image.new('RGB',(100,80),'white'))[0],{})

    def test_label_knockout_is_not_attributed_to_enclosing_shape(self):
        source=Image.new('RGB',(100,80),'white');d=ImageDraw.Draw(source)
        d.line([(1,30),(98,30)],fill='black');d.rectangle((25,20,75,40),fill='white')
        plan,_=self.review(source,texts=[{'frame_px':[25,20,50,20],'rotation_deg':0}])
        self.assertEqual(plan,{})

    def test_translucent_face_orders_only_source_tinted_strokes_behind_it(self):
        self.styles['frame'].update(fill='#FF0000',fill_opacity=.5)
        source=Image.new('RGB',(100,80),'white');draw=ImageDraw.Draw(source)
        draw.rectangle((20,10,80,70),fill='#FF8080')
        draw.line((1,30,98,30),fill='black')
        self.assertEqual(self.review(source)[0],{})  # Source line is in front.
        draw.line((21,30,79,30),fill='#800000')
        plan,audit=self.review(source)
        self.assertEqual(plan['route']['before'],['panel/body'])
        self.assertEqual(audit[0]['reason'],'source_translucent_face_tints_stroke')

    def test_a_flat_tinted_background_does_not_prove_a_hidden_stroke(self):
        self.styles['frame'].update(fill='#FF0000',fill_opacity=.5)
        source=Image.new('RGB',(100,80),'white')
        ImageDraw.Draw(source).rectangle((20,10,80,70),fill='#FF8080')
        self.assertEqual(self.review(source)[0],{})

    def test_a_short_but_aligned_visible_run_is_preserved_inside_a_large_face(self):
        self.scene['parts']['frame']['commands']=[['M',20,10],['L',145,10],['L',145,70],['L',20,70],['Z']]
        self.arrow.update(start=[1,30],end=[158,30])
        source=Image.new('RGB',(160,80),'white')
        ImageDraw.Draw(source).line([(70,32),(77,32)],fill='black')
        plan,_=self.review(source)
        self.assertEqual(plan,{})

    def test_node_outline_is_not_a_line_crossing_the_node_interior(self):
        import math
        points=[(50+10*math.cos(t*math.pi/32),30+10*math.sin(t*math.pi/32)) for t in range(64)]
        self.scene['parts']['frame'].update(kind='ellipse',commands=[['M',*points[0]]]+[['L',*p] for p in points[1:]]+[['Z']])
        self.styles['frame'].update(fill='#30DF50',width_px=2)
        source=Image.new('RGB',(100,80),'white');draw=ImageDraw.Draw(source)
        draw.line((1,30,98,30),fill='black',width=2)
        draw.ellipse((40,20,60,40),fill='#30DF50',outline='black',width=2)
        plan,_=self.review(source)
        self.assertEqual(plan['route']['before'],['panel/body'])
        # A real line drawn above the node must remain visible.
        draw.line((1,30,98,30),fill='black',width=2)
        self.assertEqual(self.review(source)[0],{})


if __name__=='__main__':unittest.main()
