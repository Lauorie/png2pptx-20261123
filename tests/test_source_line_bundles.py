import numpy as np
from PIL import Image, ImageDraw

from full_recovery.source_line_bundles import resolve


def fixture():
    source=Image.new('RGB',(160,110),'white');draw=ImageDraw.Draw(source)
    for y in (25,44,70):draw.line((50,y,100,y),fill='black',width=2)
    scene={'parts':{'left':{'kind':'rectangle','bbox_2d':[10,10,50,90]},
                    'right':{'kind':'rectangle','bbox_2d':[100,10,140,90]}}}
    def row(name,branch,y):
        return {'id':name,'key':name,'layer':branch,'heads':['none','none'],
            'points':np.c_[np.linspace(50,100,60),np.full(60,y)],'endpoint_nodes':('left','right'),
            'style':({'stroke':'#000000','width_px':1,'dash':'solid'} if branch=='shape' else
                     {'color':'#000000','width_px':2,'dash':{'type':'solid'}})}
    rows=[row('shape-a','shape',29),row('shape-b','shape',74),
          *[row('native-'+str(i),'arrows',y+.5) for i,y in enumerate((25,44,70))]]
    return source,scene,rows,row


def test_original_pixels_define_nonuniform_members_despite_displaced_wrong_count():
    source,scene,rows,_=fixture();result=resolve(scene,rows,source)
    assert len(result['source_objects'])==1
    assert result['source_objects'][0]['visible_count']==3
    assert result['source_objects'][0]['source_member_positions_px']==[25.5,44.5,70.5]
    assert result['choices'][0]['keep_ids']==['native-0','native-1','native-2']
    assert len(result['claimed_ids'])==5 and not result['arrow_geometry_changed']


def test_gray_overlay_and_headed_route_do_not_become_an_extra_black_bus():
    source,scene,rows,row=fixture();ImageDraw.Draw(source).line((50,55,100,55),fill='#7A7A7A',width=4)
    overlay=row('overlay','arrows',55);overlay['style']['color']='#7A7A7A';rows.append(overlay)
    headed=row('headed','arrows',80);headed['heads'][1]='triangle';rows.append(headed)
    result=resolve(scene,rows,source)
    assert result['source_objects'][0]['visible_count']==3
    assert not {'overlay','headed'}&result['claimed_ids']


def test_incomplete_recovery_cannot_erase_a_visible_source_member():
    source,scene,rows,_=fixture();rows=[r for r in rows if r['id']!='native-1']
    result=resolve(scene,rows,source)
    assert not result['choices'] and not result['claimed_ids']


def test_unique_source_identity_can_select_a_displaced_fit_without_claiming_alignment():
    source,scene,rows,_=fixture()
    for row in rows:
        if row['layer']=='arrows':row['points'][:,1]+=3
    result=resolve(scene,rows,source)
    assert result['source_objects'][0]['visible_count']==3
    assert len(result['choices'][0]['keep_ids'])==3
    assert result['source_objects'][0]['native_alignment_status']=='incomplete'
    assert result['source_objects'][0]['maximum_native_position_error_px']>=3


def test_dashed_or_weak_ambiguous_source_abstains_without_new_geometry():
    source,scene,rows,_=fixture();source=Image.new('RGB',source.size,'white');draw=ImageDraw.Draw(source)
    for y in (25,44,70):
        for x in range(50,100,12):draw.line((x,y,x+4,y),fill='black',width=2)
    result=resolve(scene,rows,source)
    assert not result['choices'] and not result['source_objects']
