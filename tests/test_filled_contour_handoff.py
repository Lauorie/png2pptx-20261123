import pytest
from PIL import Image,ImageDraw

from full_recovery.handoff import duplicate_arrows,source_occlusion


def contour():
    return {'id':'body','object_id':'unit','status':'ok','kind':'path',
        'commands':[['M',10,10],['L',90,10],['L',90,70],['L',10,70],['Z']]}


def block():
    return {'id':'filled','kind':'block_arrow','contours':[[[10,10],[90,10],[90,70],[10,70]]],
        'style':{'color':'#EEEEEE','shape_parts':{'body':{'fill':'#EEEEEE','stroke':'#000000'}}}}


def test_image_replacement_retains_native_contour_identity_for_existing_dedup():
    styles={'body':{'fill':'#EEEEEE','stroke':'#000000'}};scene={'parts':{}}
    assert duplicate_arrows(scene,styles,[block()])[0]==[]
    dropped,audit=duplicate_arrows(scene,styles,[block()],represented_image_parts={'body':contour()})
    assert dropped==['filled'] and audit[0]['reason']=='same_filled_contour_represented_in_image'
    # A different enclosing contour is not deleted just for containing the unit.
    larger=block();larger['contours']=[[[0,0],[100,0],[100,80],[0,80]]]
    assert duplicate_arrows(scene,styles,[larger],represented_image_parts={'body':contour()})[0]==[]


def test_image_owned_open_stroke_cannot_remove_an_external_native_route():
    part=contour();part['commands']=[['M',10,30],['L',90,30]]
    row={'id':'external','start':[10,30],'end':[90,30],'heads':['none','none'],
         'style':{'color':'#000000','width_px':1}}
    assert duplicate_arrows({'parts':{}},{'body':{'stroke':'#000000'}},[row],
                            represented_image_parts={'body':part})[0]==[]


@pytest.mark.parametrize('detail_visible',[True,False])
def test_filled_contour_orders_behind_only_source_visible_native_details(tmp_path,detail_visible):
    part=contour();part.update(id='divider',kind='path',commands=[['M',40,20],['L',40,60]])
    scene={'parts':{'detail':part}};styles={'detail':{'fill':None,'stroke':'#000000','width_px':2}}
    image=Image.new('RGB',(100,80),'#EEEEEE')
    if detail_visible:ImageDraw.Draw(image).line((40,20,40,60),fill='black',width=2)
    path=tmp_path/'source.png';image.save(path)
    plan,audit=source_occlusion(scene,styles,[block()],path,set(),part_placements=True)
    if detail_visible:
        assert plan['filled']['before']==['unit/divider'] and plan['filled']['after']==[]
        assert audit[0]['reason']=='source_visible_native_stroke_in_filled_contour'
    else:assert not plan and not audit


@pytest.mark.parametrize('visible',[True,False])
def test_filled_label_with_matching_border_is_ordered_by_source_face(tmp_path,visible):
    part=contour();part.update(id='label',commands=[['M',25,25],['L',75,25],['L',75,55],['L',25,55],['Z']])
    scene={'parts':{'label':part}}
    styles={'label':{'fill':'#FFFFFF','fill_opacity':.9,'stroke':'#EEEEEE','width_px':1}}
    source=Image.new('RGB',(100,80),'#EEEEEE')
    if visible:ImageDraw.Draw(source).rectangle((25,25,75,55),fill='white')
    path=tmp_path/'source.png';source.save(path)
    # Use a darker block so the white label's face is identifiable.
    arrow=block();arrow['style']['shape_parts']['body']['fill']='#555555'
    if not visible:source=Image.new('RGB',(100,80),'#555555');source.save(path)
    plan,audit=source_occlusion(scene,styles,[arrow],path,set(),part_placements=True)
    if visible:assert plan['filled']['before']==['unit/label']
    else:assert not plan
