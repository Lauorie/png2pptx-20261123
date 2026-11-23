"""Ownership fixes must preserve real nesting, partial symbols and text styling."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'vendor')]
from full_recovery.handoff import duplicate_ellipsis_text,duplicate_arrows
from full_recovery.text_stage import composition_rows
from full_recovery.api_gateway import assemble,IncompleteStream


def dot(x,y):
    return {'status':'ok','kind':'ellipse','bbox_2d':[x-2,y-2,x+2,y+2],'object_id':'network'}


class RefineTests(unittest.TestCase):
    def test_text_background_and_typography_are_preserved(self):
        rows=[{'id':'t','background_fill':'#112233','font_size':17,'text':'label','bbox_2d':[1,2,3,4]},
              {'id':'duplicate','text':'…'}]
        original=copy.deepcopy(rows)
        self.assertEqual(composition_rows({'texts':rows},{'drop_text_ids':['duplicate']}),rows[:1])
        self.assertEqual(rows,original)
        self.assertNotIn('background_fill',composition_rows({'texts':[{'id':'t'}]},{} )[0])

    def test_complete_horizontal_and_vertical_ellipses(self):
        for vertical in (False,True):
            parts={str(i):dot(10 if vertical else x,x if vertical else 10) for i,x in enumerate((5,15,25))}
            style={k:{'fill':'#000000'} for k in parts}
            row={'id':'t','text':'⋮' if vertical else '...',
                 'bbox_2d':[5,0,15,30] if vertical else [0,5,30,15]}
            self.assertEqual(duplicate_ellipsis_text({'parts':parts},style,[row])[0],['t'])
            self.assertEqual(duplicate_ellipsis_text({'parts':dict(list(parts.items())[:2])},style,[row])[0],[])

    def test_same_ellipsis_with_different_branch_colours_is_not_drawn_twice(self):
        parts={str(i):dot(x,10) for i,x in enumerate((5,15,25))}
        styles={k:{'fill':'#434343'} for k in parts}
        row={'id':'t','text':'…','color':'#000000','bbox_2d':[0,5,30,15]}
        self.assertEqual(duplicate_ellipsis_text({'parts':parts},styles,[row])[0],['t'])
        row['bbox_2d']=[0,25,30,35]
        self.assertEqual(duplicate_ellipsis_text({'parts':parts},styles,[row])[0],[])

    def test_embedded_ellipsis_decimal_hollow_nodes_and_nearby_occurrence_survive(self):
        parts={str(i):dot(x,10) for i,x in enumerate((5,15,25))}
        styles={k:{'fill':'#000000'} for k in parts}
        for text in ('x…y','0.3','.'):
            self.assertEqual(duplicate_ellipsis_text({'parts':parts},styles,[{'id':'t','text':text,'bbox_2d':[0,5,30,15]}])[0],[])
        row={'id':'t','text':'…','bbox_2d':[0,25,30,35]}
        self.assertEqual(duplicate_ellipsis_text({'parts':parts},styles,[row])[0],[])
        row['bbox_2d']=[0,5,30,15]
        self.assertEqual(duplicate_ellipsis_text({'parts':parts},{k:{'fill':None} for k in parts},[row])[0],[])

    def test_duplicate_filled_arrow_does_not_repaint_ordered_shape(self):
        q=[[0,0],[80,0],[100,25],[80,50],[0,50],[20,25]]
        part={'status':'ok','object_id':'band','commands':[['M',*q[0]],*[['L',*p] for p in q[1:]],['Z']]}
        scene={'parts':{'p':part}};styles={'p':{'fill':'#D5E7D4'}}
        arrow={'kind':'block_arrow','id':'a','contours':[q],'style':{'color':'#d5e7d4'}}
        self.assertEqual(duplicate_arrows(scene,styles,[arrow])[0],['a'])
        arrow['contours'].append([[30,15],[35,15],[35,20],[30,20]])
        self.assertEqual(duplicate_arrows(scene,styles,[arrow])[0],['a'])
        # A smaller real object nested in a broad arrow is not a duplicate.
        arrow['contours']=[[[x*2,y*2] for x,y in q]]
        self.assertEqual(duplicate_arrows(scene,styles,[arrow])[0],[])

    def test_network_error_finish_is_not_a_successful_completion(self):
        with self.assertRaises(IncompleteStream):
            assemble([json.dumps({'choices':[{'delta':{'content':'partial'},'finish_reason':'network_error'}]}),'[DONE]'])

    def test_large_units_keep_context_and_exactly_one_answer_owner(self):
        from shape_integration.styling.controller import bounded_units
        unit={'id':'grid','keys':list(map(str,range(108))),'root':Path('/unused'),'kind':'object'}
        result=bounded_units([unit],24)
        self.assertEqual([k for u in result for k in u['keys']],unit['keys'])
        self.assertTrue(all(len(u['keys'])<=24 and u['context_keys']==unit['keys'] for u in result))

    def test_source_layer_order_places_offset_shadow_behind_body(self):
        from shape_integration.styling.occlusion import order_internal_by_paint
        from shapely.geometry import box
        scene={'objects':{'o':{'part_keys':['body','shadow']}}}
        relations={'object_order':['o'],'internal_model_edges':[{'back':'shadow','front':'body','origin':'model_selection'}]}
        styles={k:{'fill':'#CCCCCC'} for k in ('body','shadow')}
        result=order_internal_by_paint(scene,styles,relations,styles,{'body':box(0,0,100,100),'shadow':box(3,3,103,103)})
        self.assertEqual(result['part_orders']['o'],['shadow','body'])

    def test_only_observed_acyclic_internal_occlusion_excludes_colour_samples(self):
        from shape_integration.styling.occlusion import exclusions
        from shapely.geometry import box
        geoms={'body':box(0,0,100,100),'shadow':box(3,3,103,103)}
        styles={k:{'fill':'#FFFFFF'} for k in geoms}
        rel={'part_owners':{k:'o' for k in geoms},'component_parts':{'o':list(geoms)},'effective_edges':[]}
        self.assertTrue(exclusions('shadow',geoms,rel,styles).is_empty)
        rel['internal_model_edges']=[{'back':'shadow','front':'body'}]
        self.assertGreater(exclusions('shadow',geoms,rel,styles).area,0)
        self.assertTrue(exclusions('body',geoms,rel,styles).is_empty)
        rel['internal_model_edges'].append({'back':'body','front':'shadow'})
        self.assertTrue(exclusions('shadow',geoms,rel,styles).is_empty)

    def test_grid_endpoint_heuristic_cannot_reverse_observed_face_order(self):
        from shape_integration.styling.occlusion import order_internal_by_paint
        from shapely.geometry import box,LineString
        scene={'objects':{'o':{'part_keys':['back','line','front']}}}
        styles={'back':{'fill':'#AAAAAA'},'front':{'fill':'#FFFFFF'},'line':{'fill':None,'stroke':'#000000'}}
        geoms={'back':box(0,0,100,100),'front':box(50,10,150,90),'line':LineString([(10,50),(90,50)])}
        rel={'object_order':['o'],'internal_model_edges':[{'back':'back','front':'front','origin':'model_selection'},
              {'back':'front','front':'line','origin':'model_selection'}]}
        result=order_internal_by_paint(scene,styles,rel,styles,geoms)
        self.assertEqual(result['part_orders']['o'],['back','front','line'])
        self.assertFalse(result['internal_style_conflicts'])
        self.assertTrue(result['internal_rejected_heuristics'])

    def test_occluded_route_with_crossing_text_does_not_request_both_orders(self):
        import tempfile
        from PIL import Image,ImageDraw
        from full_recovery.handoff import source_occlusion
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source.png';image=Image.new('RGB',(120,60),'white')
            draw=ImageDraw.Draw(image);draw.line([(45,20),(45,40)],fill='black',width=3)
            image.save(source)
            p={'id':'body','object_id':'panel','status':'ok',
               'commands':[['M',20,0],['L',100,0],['L',100,59],['L',20,59],['Z']]}
            row={'id':'route','start':[5,30],'end':[115,30],'style':{'color':'#000000'}}
            plan,_=source_occlusion({'parts':{'body':p}},{'body':{'fill':'#FFFFFF'}},[row],source,set(),part_placements=True)
            self.assertEqual(plan['route'],{'before':['panel/body'],'after':[]})

    def test_matrix_dividers_follow_faces_with_subpixel_endpoint_error(self):
        from shape_integration.styling.occlusion import order_internal_by_paint
        from shapely.geometry import box,LineString
        keys=['front','front_grid','back','back_grid']
        scene={'objects':{'o':{'part_keys':keys}}}
        styles={k:({'fill':'#CCCCCC'} if 'grid' not in k else {'fill':None,'stroke':'#000000'}) for k in keys}
        geoms={'back':box(0,0,100,100),'front':box(20,20,120,120),
               'back_grid':LineString([(50,-.1),(50,99.9)]),
               'front_grid':LineString([(70,19.9),(70,119.9)])}
        relations={'object_order':['o'],'internal_model_edges':[{'back':'back','front':'front','origin':'model_selection'}]}
        result=order_internal_by_paint(scene,styles,relations,styles,geoms)
        self.assertEqual(result['part_orders']['o'],['back','back_grid','front','front_grid'])
        self.assertFalse(result['internal_style_conflicts'])

    def test_observed_front_face_excludes_its_ink_from_rear_divider_measurement(self):
        from shape_integration.styling.occlusion import order_internal_by_paint,exclusions
        from shapely.geometry import box,LineString,Point
        keys=['back','divider','front']
        scene={'objects':{'o':{'part_keys':keys}}}
        geoms={'back':box(0,0,100,100),'front':box(20,20,120,120),
               'divider':LineString([(50,0),(50,100)])}
        styles={'back':{'fill':'#FFFFFF'},'front':{'fill':'#CCCCCC'},
                'divider':{'fill':None,'stroke':'#000000','width_px':1}}
        relations={'object_order':['o'],'effective_edges':[],
                   'part_owners':{k:'o' for k in keys},'component_parts':{'o':keys},
                   'internal_model_edges':[{'back':'back','front':'front','origin':'model_selection'}]}
        ordered=order_internal_by_paint(scene,styles,relations,styles,geoms)
        hidden=exclusions('divider',geoms,ordered,styles)
        self.assertTrue(hidden.covers(Point(50,50)))
        self.assertFalse(hidden.covers(Point(50,10)))
        # Without observed face order, generic containment cannot mask a line.
        relations['internal_model_edges']=[]
        ordered=order_internal_by_paint(scene,styles,relations,styles,geoms)
        self.assertTrue(exclusions('divider',geoms,ordered,styles).is_empty)

    def test_translucent_face_does_not_exclude_visible_rear_ink(self):
        from shape_integration.styling.occlusion import paint
        from shapely.geometry import box,Point
        hidden=paint(box(0,0,100,100),{'fill':'#CCCCCC','fill_opacity':.5,'stroke':'#000000','width_px':2})
        self.assertFalse(hidden.covers(Point(50,50)))
        self.assertTrue(hidden.covers(Point(0,50)))


if __name__=='__main__':unittest.main()
