"""Visible frame strokes correct layer order without rewriting existing paint."""
import copy
from PIL import Image,ImageDraw
from shapely.geometry import box
from shape_integration.styling.occlusion import order_internal_by_paint


def example(opaque=True,back_exists=True):
    image=Image.new('RGB',(110,110),'white');draw=ImageDraw.Draw(image)
    if back_exists:draw.rectangle((30,10,80,70),fill='#7ACA8E' if opaque else None,outline='#225533',width=2)
    draw.rectangle((10,30,60,90),fill='#7ACA8E' if opaque else None,outline='#225533',width=2)
    scene={'objects':{'stack':{'part_keys':['front','back']}}}
    table={k:{'kind':'rectangle'} for k in ['front','back']}
    geoms={'front':box(10,30,60,90),'back':box(30,10,80,70)}
    styles={k:{'fill':'#7ACA8E' if opaque else '#FFFFFF','stroke':'#225533','width_px':2.}
            for k in geoms}
    relations={'object_order':['stack'],'internal_model_edges':[
        {'back':'front','front':'back','origin':'model_selection'}]}
    return image,scene,table,geoms,styles,relations


def test_visible_occlusion_corrects_a_reversed_claim_and_preserves_geometry_and_colour():
    image,scene,table,geoms,styles,relations=example()
    before=copy.deepcopy((scene,table,styles,relations))
    result=order_internal_by_paint(scene,table,relations,styles,geoms,source=image)
    assert result['part_orders']['stack']==['back','front']
    assert len(result['source_frame_order_edges'])==1
    assert not any(e['back']=='front' and e['front']=='back' for e in result['internal_model_edges'])
    assert (scene,table,styles,relations)==before


def test_source_with_both_contours_visible_does_not_establish_opaque_paint_order():
    image,scene,table,geoms,styles,relations=example(opaque=False)
    result=order_internal_by_paint(scene,table,relations,styles,geoms,source=image)
    assert result['source_frame_order_edges']==[]
    assert result['part_orders']['stack']==['front','back']
    assert all(style['fill']=='#FFFFFF' for style in styles.values())


def test_absent_back_contour_cannot_be_mistaken_for_an_observed_occlusion():
    image,scene,table,geoms,styles,relations=example(back_exists=False)
    result=order_internal_by_paint(scene,table,relations,styles,geoms,source=image)
    assert result['source_frame_order_edges']==[]
    assert result['part_orders']['stack']==['front','back']
