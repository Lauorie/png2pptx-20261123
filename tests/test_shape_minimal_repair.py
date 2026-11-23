"""Trace regressions: open semantic types, nested ownership and corner evidence."""
import unittest
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import GeometryCollection
from native_qwen.element_protocol import output_schema, KINDS
from native_qwen.prompt_contract import validate_response
from shape_integration.local_contract import validator
from shape_integration.fitting.context import identity_context
from shape_integration.fitting.rounded_corners import radius_from_pixels
from shape_integration.fitting.paths import box_commands
from shape_integration.styling.rules import ranked_source_colors


class MinimalRepairTests(unittest.TestCase):
    def test_image_plan_accepts_canvas_edge_but_rejects_arbitrary_overflow(self):
        from png2pptx.images.coverage import check_regions
        regions=[{'id':'whole','bbox':[0,0,320,240],'target_ids':[]}]
        check_regions(regions,[320,240],set(),set())
        self.assertEqual(regions[0]['bbox'],[0,0,319,239])
        with self.assertRaises(ValueError):
            check_regions([{'id':'bad','bbox':[0,0,321,240],'target_ids':[]}],
                          [320,240],set(),set())

    def test_crop_fragments_defer_without_using_parent_bbox_containment(self):
        from shape_integration.shape import clipped_new_object
        meta={'actual_crop_input_size_px':[800,600],'source_crop_half_open':[0,100,400,400]}
        self.assertTrue(clipped_new_object([700,200,799,300],meta,[1200,600]))
        self.assertFalse(clipped_new_object([100,100,200,200],meta,[1200,600]))
        # Source-clipped geometry is legitimate: no larger source exists.
        meta['source_crop_half_open']=[0,0,1200,600]
        self.assertFalse(clipped_new_object([700,200,799,300],meta,[1200,600]))

    def test_new_ppt_types_cannot_reject_an_entire_localization(self):
        self.assertTrue({'hexagon','octagon','pentagon'}<=set(KINDS))
        objects=[{'id':'G'+str(i),'label':'visible polygon','kind':kind,'bbox_2d':[2,2,20,20]}
                 for i,kind in enumerate(('hexagon','octagon','custom_visible_symbol'))]
        data={'objects':objects,'uncertain':[],'more_pending':False}
        validate_response(data,output_schema('shape_bbox'))
        local={'reviews':[],'new_objects':objects,'uncertain':[],'more_pending':False}
        validator([], [100,100])(local)
        objects[0]['kind']=None
        with self.assertRaises(ValueError):validator([], [100,100])(local)

    def test_nested_identity_is_distinct_from_duplicate_filtering(self):
        outer={'id':'G032','label':'blue outer decision frame','kind':'rectangle','bbox_2d':[0,0,100,100]}
        inner={'id':'G033','label':'white inner panel','kind':'rectangle','bbox_2d':[20,20,80,80]}
        context=identity_context(inner,[outer,inner],[],[],.2,exclude_shape_relations=('contains_target',))
        self.assertEqual(context['registered_enclosing_shapes'][0]['id'],'G032')
        self.assertEqual(context['nearby_inventory'],[])
        self.assertNotIn('bbox',str(context))
        # The reverse relation does not tell the outer object's fitter to drop
        # its own intrinsic enclosing boundary.
        reverse=identity_context(outer,[outer,inner],[],[],.2)
        self.assertEqual(reverse['registered_enclosing_shapes'],[])

    def test_coverage_handoff_uses_actual_peer_parts_despite_old_container_name(self):
        frame={'id':'new_frame','label':'missing outer frame','kind':'rectangle','bbox_2d':[0,0,100,100]}
        peer={'id':'peer','label':'misleading container name','kind':'compound_symbol','bbox_2d':[10,10,90,90]}
        scene={'objects':{'peer':{'part_keys':['circle']}},
               'parts':{'circle':{'id':'node','kind':'ellipse','status':'ok'}}}
        context=identity_context(frame,[frame,peer],[],[],.2,represented_scene=scene)
        for key in ('nearby_inventory','registered_peer_shapes'):
            self.assertEqual(context[key][0]['represented_parts'],[{'id':'node','kind':'ellipse'}])
        self.assertNotIn('bbox',str(context))

    def test_source_corners_correct_straight_only_radius_but_blank_abstains(self):
        scale=4;size=(240*scale,140*scale)
        image=Image.new('RGB',size,'white');draw=ImageDraw.Draw(image)
        draw.rounded_rectangle([19*scale,19*scale,221*scale,121*scale],radius=26*scale,fill='black')
        draw.rounded_rectangle([21*scale,21*scale,219*scale,119*scale],radius=24*scale,fill='white')
        image=image.resize((240,140),Image.Resampling.LANCZOS)
        mask=Image.new('L',(240,140));ImageDraw.Draw(mask).rounded_rectangle([20,20,220,120],radius=25,fill=255)
        parameters={'center':[120,70],'width':200,'height':100,'angle_radians':0.,'radius':6.}
        radius,audit=radius_from_pixels(parameters,image,mask,sam_tolerance=3)
        self.assertTrue(audit['applied']);self.assertAlmostEqual(radius,25,delta=2)
        radius,audit=radius_from_pixels(parameters,Image.new('RGB',(240,140),'white'),mask,sam_tolerance=3)
        self.assertIsNone(radius);self.assertFalse(audit['applied'])

    def test_fill_and_stroke_color_roles_retain_light_pigment(self):
        image=Image.new('RGB',(160,120),'white');draw=ImageDraw.Draw(image)
        draw.rectangle([20,20,140,100],fill='#E4EAF5',outline='#4477AA',width=3)
        draw.text((45,50),'dark text',fill='black')
        part={'commands':box_commands([80,60,120,80,0])}
        result=ranked_source_colors(image,{'p':part},lambda _:GeometryCollection(),separate_roles=True)
        self.assertEqual(result['fill_colors'][0]['color'],'#E4EAF5')
        self.assertIn('#4477AA',[row['color'] for row in result['stroke_colors']])
        self.assertNotIn('fill_colors',ranked_source_colors(image,{'p':part},lambda _:GeometryCollection()))


if __name__=='__main__':unittest.main()
